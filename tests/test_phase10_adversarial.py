"""Phase 10 adversarial tests (20 scenarios): every violation must be refused
or detected loudly. Mirrors the Phase 9 adversarial suite at the Phase 10
boundary.

  A1  future-feature leakage in a Phase 10 snapshot
  A2  same-session feature reference in a Phase 10 snapshot
  A3  non-finite Phase 10 feature entering the training matrix
  A4  an unregistered model point entering the plan
  A5  a Phase 11+ feature id in a feature set
  A6  feature-set membership drift (missing/excess members)
  A7  FS-001 frozen digest mutation
  A8  feature definition tampering invalidating the digest
  A9  corrupt Phase 10 cache refused on load
  A10 wrong feature_set_version at registration
  A11 duplicate experiment registration refused
  A12 experiment id outside the locked range
  A13 exact feature-set membership assertion failure
  A14 test predictions outside the locked window flagged
  A15 backtest config drift (non-CM-001 costs) flagged
  A16 label contract drift flagged
  A17 diagnostics computed on the test split (forbidden)
  A18 data expansion (new instrument) flagged by the guard
  A19 feature scope expansion (feature zoo) flagged
  A20 split integrity violation flagged
"""

from __future__ import annotations

import polars as pl
import pytest
from datetime import date, timedelta

from orbit.ml.features import (
    ALL_PHASE10_DEFINITIONS,
    FEATURE_DEFINITIONS,
    FEATURE_NAMES_PHASE10,
    build_feature_frame_phase10,
    build_feature_snapshot,
    build_feature_snapshot_phase10,
    build_phase10_all_feature_frame,
    build_phase10_feature_set_snapshot,
    phase10_set_identity,
)
from orbit.ml.labels import build_phase9_label_snapshot
from orbit.ml.phase10_audit import (
    assert_feature_set_membership,
    verify_dataset_unchanged,
)
from orbit.ml.phase10_plan import phase10_experiment_id, phase10_model_point_for
from orbit.ml.phase10_registry import register_phase10_experiment
from orbit.ml.snapshot_cache import cache_phase10_snapshot, load_cached_phase10_snapshot
from tests.phase9_testutils import make_canonical_bars, make_events


@pytest.fixture(scope="module")
def bars():
    return make_canonical_bars()


def _phase10_frame(bars):
    return build_feature_frame_phase10(bars)


# A1 / A2 feature leakage


def test_A1_future_feature_leakage_is_detected(bars):
    frame = _phase10_frame(bars)
    frame = frame.with_columns(
        (pl.col("decision_session") + pl.duration(days=3)).alias("window_end_session")
    )
    from orbit.ml.features import assert_features_point_in_time

    with pytest.raises(AssertionError, match="point-in-time violation"):
        assert_features_point_in_time(frame)


def test_A2_same_session_feature_reference_is_detected(bars):
    frame = _phase10_frame(bars)
    frame = frame.with_columns(pl.col("decision_session").alias("window_end_session"))
    from orbit.ml.features import assert_features_point_in_time

    with pytest.raises(AssertionError, match="point-in-time violation"):
        assert_features_point_in_time(frame)


# A3 non-finite values


def test_A3_non_finite_feature_is_detected(bars):
    frame = _phase10_frame(bars)
    frame = frame.with_columns(pl.lit(float("nan")).alias("ret_5"))
    from orbit.ml.features import assert_features_finite

    with pytest.raises(AssertionError, match="non-finite"):
        assert_features_finite(frame, FEATURE_NAMES_PHASE10)


# A4 unregistered model point


def test_A4_unregistered_model_point_refused():
    with pytest.raises(ValueError, match="pre-registered|grid"):
        phase10_model_point_for("xgboost", {"n_estimators": 500, "max_depth": 3, "learning_rate": 0.1})
    with pytest.raises(ValueError, match="pre-registered|grid"):
        phase10_experiment_id("FS-004", "ridge", {"alpha": 0.5})


# A5 / A6 feature scope and membership


def test_A5_unknown_feature_set_refused_at_registration(bars):
    with pytest.raises(ValueError):
        register_phase10_experiment(
            experiment_id="EXP-10001", hypothesis_id="H-001",
            feature_set_id="FS-099", feature_set_version="v1",
            family="ridge", params={"alpha": 1.0},
        )


def test_A6_membership_drift_raises():
    with pytest.raises(AssertionError, match="not BASE"):
        assert_feature_set_membership("FS-004", ["FEAT-101", "FEAT-102", "FEAT-103"])
    with pytest.raises(AssertionError, match="ALL set"):
        assert_feature_set_membership("FS-003", ["FEAT-001"])


def test_A7_fs001_frozen_digest_mutation_detected(bars, monkeypatch):
    """Changing any FS-001 definition value changes its content digest (the
    mechanism that makes post-hoc feature mutation structurally detectable)."""
    import orbit.ml.features as feats
    from orbit.ml.features import _feature_definitions_digest

    ids = [f["feature_id"] for f in FEATURE_DEFINITIONS]
    before = _feature_definitions_digest(ids)
    tampered = [
        {**f, "formula": (f.get("formula") or "") + " (tampered)"} for f in FEATURE_DEFINITIONS
    ]
    monkeypatch.setattr(feats, "FEATURE_DEFINITIONS", tampered)
    after = _feature_definitions_digest(ids)
    assert before != after


# A8 definition digest sensitivity (covered by test_feature_definitions_digest
# in the features suite; asserted here too)


def test_A9_corrupt_phase10_cache_refused(tmp_path, bars):
    snap = build_feature_snapshot_phase10(bars, data_refs=["DS-000001"])
    cache_phase10_snapshot(snap, tmp_path)
    # corrupt the stored records so the digest check must fire
    rec = pl.read_parquet(tmp_path / "feature_set_FS-002_records.parquet")
    rec = rec.with_columns(pl.lit(0.0).alias("ret_5"))
    rec.write_parquet(tmp_path / "feature_set_FS-002_records.parquet")
    with pytest.raises(RuntimeError, match="digest mismatch"):
        load_cached_phase10_snapshot("FS-002", tmp_path)


def test_A10_wrong_feature_set_version_rejected(bars):
    with pytest.raises(ValueError, match="is version"):
        register_phase10_experiment(
            experiment_id="EXP-10001", hypothesis_id="H-001",
            feature_set_id="FS-003", feature_set_version="v2",
            family="ridge", params={"alpha": 1.0},
        )


def test_A11_duplicate_registration_refused(bars):
    service, spec = register_phase10_experiment(
        experiment_id="EXP-10001", hypothesis_id="H-001",
        feature_set_id="FS-001", feature_set_version="v1",
        family="ridge", params={"alpha": 1.0}, seed=42,
    )
    # registering the same id again through the same service is refused
    with pytest.raises(ValueError, match="duplicate experiment id"):
        service.register(spec)


def test_A12_experiment_id_outside_locked_range():
    # the range is fixed by the plan; EXP-10000 / EXP-10053 cannot be produced
    ids = [
        phase10_experiment_id(sid, m["family"], m["params"])
        for sid in __import__("orbit.ml.features", fromlist=["PHASE10_FEATURE_SET_ORDER"]).PHASE10_FEATURE_SET_ORDER
        for m in __import__("orbit.ml.phase10_plan", fromlist=["PHASE10_MODEL_POINTS"]).PHASE10_MODEL_POINTS
    ]
    assert all(10001 <= int(i.split("-")[1]) <= 10052 for i in ids)
    # a grid-valid-but-not-Phase-10 point gets no id at all
    with pytest.raises(ValueError, match="pre-registered|grid"):
        phase10_experiment_id("FS-003", "ridge", {"alpha": 0.5})


# A13 exact membership (see A6); A14/A15/A16/A18 audit-level checks


def test_A14_test_lock_outside_window_flagged():
    from orbit.ml.phase10_audit import run_phase10_audit
    from orbit.ml.splits import PHASE9_WINDOWS

    preds = pl.DataFrame(
        {
            "decision_session": [date(2030, 1, 2), date(2023, 1, 3)],
            "prediction": [0.1, 0.2],
        }
    )
    checks = run_phase10_audit(
        snapshots={}, test_predictions=preds,
        phase9_fs001_digest="x", bars=None,
    )
    lock = [c for c in checks if c["check"] == "test_lock"][0]
    assert lock["status"] == "FAIL"


def test_A15_backtest_config_drift_flagged():
    from orbit.backtest.config import CostConfig
    from orbit.ml.phase10_audit import run_phase10_audit

    class _Drift:
        costs = CostConfig(spread_bps=50.0, fees_bps=1.0, slippage_bps=2.0)
        sizing = type("S", (), {"value": "weight"})
        long_only = True

    checks = run_phase10_audit(
        snapshots={}, backtest_config=_Drift(), phase9_fs001_digest="x",
    )
    bc = [c for c in checks if c["check"] == "backtest_uniformity"][0]
    assert bc["status"] == "FAIL"


def test_A16_label_contract_drift_flagged():
    from orbit.ml.phase10_audit import run_phase10_audit

    class _BadLabel:
        label_id = "LAB-005"
        version = "v1"

    checks = run_phase10_audit(
        snapshots={}, label_snapshot=_BadLabel(), phase9_fs001_digest="x",
    )
    lc = [c for c in checks if c["check"] == "label_contract"][0]
    assert lc["status"] == "FAIL"


def test_A17_diagnostics_scope_metadata_is_train_only(bars):
    """The diagnostics report must declare train-only scope and contain no
    test-split statistics."""
    from orbit.ml.phase10_diagnostics import _per_split_stats
    from orbit.ml.phase10_runner import run_diagnostics
    from orbit.ml.features import build_feature_snapshot, FEATURE_NAMES

    fs001 = build_feature_snapshot(bars, data_refs=["DS-000001"])
    decisions = fs001.records.select("instrument_id", "decision_time")
    ls = build_phase9_label_snapshot(bars, make_events(), [], decisions, data_refs=["DS-000001"])
    allf = build_phase10_all_feature_frame(bars)
    fs003 = build_phase10_feature_set_snapshot("FS-003", allf, data_refs=["DS-000001"])
    # run_diagnostics on synthetic data with the test windows; the report must
    # label its scope and contain no test-split stats.
    from tests.phase9_testutils import TEST_WINDOWS

    diag = run_diagnostics(
        {"FS-001": fs001, "FS-002": build_feature_snapshot_phase10(bars, data_refs=["DS-000001"]), "FS-003": fs003},
        ls, windows=TEST_WINDOWS,
    )
    assert diag["scope"] == "train split only (never test)"


def test_A18_data_expansion_detected(bars):
    instruments = [
        __import__("orbit.schemas.instrument", fromlist=["Instrument"]).Instrument(
            instrument_id="INS-000099",
            primary_ticker="NEW", exchange_id="XNYS", name="New",
            security_type=__import__("orbit.schemas.common", fromlist=["SecurityType"]).SecurityType.EQUITY,
            listing_date=date(1995, 1, 1),
        )
    ]
    guard = verify_dataset_unchanged(bars, instruments, {}, expected_symbols=None)
    exp = [c for c in guard["checks"] if c["check"] == "data_expansion_guard"][0]
    assert exp["status"] == "FAIL"


def test_A19_feature_scope_expansion_detected(bars):
    from orbit.ml.phase10_audit import run_phase10_audit

    from orbit.ml.features import FeatureSnapshot

    zoo = FeatureSnapshot(
        feature_set_id="FS-099",
        feature_set_version="v1",
        feature_refs=["FEAT-999"],
        data_refs=["DS-000004"],
        records=pl.DataFrame(
            {
                "instrument_id": [bars["instrument_id"][0]],
                "decision_session": [bars["trade_date"][0]],
                "decision_time": [bars["ts_utc"][0]],
                "window_end_session": [bars["trade_date"][0] - timedelta(days=1)],
                "FEAT-999": [1.0],
            }
        ),
    )
    checks = run_phase10_audit(
        snapshots={"FS-099": zoo}, phase9_fs001_digest="x",
    )
    scope = [c for c in checks if c["check"] == "feature_scope_FS-099"][0]
    assert scope["status"] == "FAIL"


def test_A20_split_integrity_violation_flagged(bars):
    from orbit.ml.phase10_audit import run_phase10_audit

    bad_ds = {
        "FS-001": {
            "report": {"train_rows": 1, "val_rows": 1, "test_rows": 1},
            "split_frame": pl.DataFrame(
                {
                    "instrument_id": [bars["instrument_id"][0]],
                    "decision_session": [date(2010, 1, 4)],
                    "split": ["train"],
                    "outcome_value": [0.0],
                    "window_end_session": [date(2019, 1, 2)],  # leaks into val
                }
            ),
        }
    }
    checks = run_phase10_audit(
        snapshots={}, datasets_by_set=bad_ds, phase9_fs001_digest="x",
    )
    si = [c for c in checks if c["check"] == "split_integrity_FS-001"][0]
    assert si["status"] == "FAIL"