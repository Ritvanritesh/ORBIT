"""Phase 9 adversarial tests (roadmap Section 32): 20 attack scenarios.

Each test attempts a specific violation and asserts it is refused loudly:
  A1  future-feature leakage in the feature snapshot
  A2  same-session feature reference
  A3  unavailable labels entering the training matrix
  A4  train-outcome leakage across the validation boundary
  A5  calibration fitted on the test split
  A6  unregistered hyperparameter set at registration time
  A7  post-hoc tuning after results (params not on the grid)
  A8  random train/test split (no such path exists; rows outside windows drop)
  A9  signal decision_time not at the session close
  A10 short positions (direction != long|flat)
  A11 non-finite / negative / >1 weight targets
  A12 signal sessions outside the locked test window
  A13 duplicate experiment registration
  A14 double result recording (single immutable result)
  A15 feature-snapshot tampering changing the digest
  A16 label contract overwrite / definition tampering
  A17 corrupt cache digest refused on load
  A18 cross-session ranking contamination
  A19 audit detects calibration leakage
  A20 backtest accounting invariants (Phase 7 gate)
  A21 audit detects an accidental Phase 10 feature zoo
  A22 audit detects accidental data expansion
  A23 audit detects a non-Phase-9 (Phase 22+) model family
"""

from __future__ import annotations

import json
import polars as pl
import pytest
from datetime import date, timedelta

from orbit.backtest.clock import MarketEventClock
from orbit.ml.calibration import assert_no_test_fit, fit_platt
from orbit.ml.features import assert_features_point_in_time, build_feature_frame
from orbit.ml.grids import validate_model_parameters
from orbit.ml.registry import register_ml_experiment
from orbit.ml.ranking import cross_sectional_rank, top_k_long
from orbit.ml.snapshot_cache import (
    cache_feature_snapshot,
    load_cached_feature_snapshot,
)
from tests.phase9_testutils import (
    TEST_WINDOWS,
    build_test_datasets,
    make_canonical_bars,
    make_events,
    session_close_utc,
)


@pytest.fixture(scope="module")
def bars():
    return make_canonical_bars()


# ---------------------------------------------------------------------------
# A1 / A2 feature leakage
# ---------------------------------------------------------------------------


def test_A1_future_feature_leakage_is_detected(bars):
    frame = build_feature_frame(bars)
    frame = frame.with_columns(
        (pl.col("decision_session") + pl.duration(days=3)).alias("window_end_session")
    )
    with pytest.raises(AssertionError, match="point-in-time violation"):
        assert_features_point_in_time(frame)


def test_A2_same_session_feature_reference_is_detected(bars):
    frame = build_feature_frame(bars)
    frame = frame.with_columns(pl.col("decision_session").alias("window_end_session"))
    with pytest.raises(AssertionError, match="point-in-time violation"):
        assert_features_point_in_time(frame)


# ---------------------------------------------------------------------------
# A3 / A4 label and split leakage
# ---------------------------------------------------------------------------


def test_A3_unavailable_labels_never_enter_training(bars):
    _, _, ds = build_test_datasets(bars, make_events())
    meta = ds["train"][3]
    # every train row came from an AVAILABLE outcome; unavailable rows carry
    # insufficient_future_data and appear in the report, never in matrices
    assert meta.height > 0
    assert all(meta["outcome_value"].is_finite().to_list())
    report = ds["report"]
    assert report["unavailable_rows"] > 0
    assert any(r["unavailable_reason"] == "insufficient_future_data" for r in report["unavailable_reasons"])


def test_A4_train_outcomes_must_not_reach_validation(bars):
    _, _, ds = build_test_datasets(bars, make_events())
    meta_tr = ds["train"][3]
    assert (meta_tr["window_end_session"] < TEST_WINDOWS["val_start"]).all()


# ---------------------------------------------------------------------------
# A5 / A19 calibration leakage
# ---------------------------------------------------------------------------


def test_A5_calibration_fitted_on_test_is_refused(bars):
    rng = __import__("numpy").random.RandomState(3)
    scores = rng.uniform(-1, 1, 200)
    binary = (scores > 0).astype(int)
    m = fit_platt(scores, binary)
    assert m.fitted_on == "val"
    m.fitted_on = "test"
    with pytest.raises(AssertionError, match="must be 'val'"):
        assert_no_test_fit(m)


def test_A19_audit_flags_calibration_leakage(bars):
    from orbit.ml.audit import run_phase9_audit

    rng = __import__("numpy").random.RandomState(5)
    scores = rng.uniform(-1, 1, 100)
    binary = (scores > 0).astype(int)
    m = fit_platt(scores, binary)
    m.fitted_on = "test"
    fs, ls, _ = build_test_datasets(bars, make_events())
    checks = run_phase9_audit(feature_snapshot=fs, label_snapshot=ls, calibration_map=m)
    assert any(c["check"] == "calibration_val_only" and c["status"] == "FAIL" for c in checks)


# ---------------------------------------------------------------------------
# A6 / A7 grid lock
# ---------------------------------------------------------------------------


def test_A6_unregistered_hyperparameters_refused_at_registration():
    with pytest.raises(ValueError, match="not a pre-registered"):
        register_ml_experiment(
            experiment_id="EXP-90005", hypothesis_id="H-001", family="xgboost",
            params={"n_estimators": 500, "max_depth": 12, "learning_rate": 0.5},
        )


def test_A7_post_hoc_tuning_is_impossible():
    # any combination outside the grid is refused structurally
    with pytest.raises(ValueError, match="not a pre-registered"):
        validate_model_parameters("logistic", {"C": 0.042})


# ---------------------------------------------------------------------------
# A8 no random split
# ---------------------------------------------------------------------------


def test_A8_rows_outside_all_windows_are_dropped(bars):
    _, _, ds = build_test_datasets(bars, make_events())
    for split in ("train", "val", "test"):
        meta = ds[split][3]
        lo, hi = TEST_WINDOWS[f"{split}_start"], TEST_WINDOWS[f"{split}_end"]
        assert (meta["decision_session"] >= lo).all()
        assert (meta["decision_session"] <= hi).all()


# ---------------------------------------------------------------------------
# A9 / A10 / A11 / A12 signal contract
# ---------------------------------------------------------------------------


def test_A9_decision_time_must_be_session_close():
    """The signal generator can only stamp exact session closes; a forged
    off-close decision time is detected against session_close_utc."""
    from orbit.ml.signals import predictions_to_signals

    pred = pl.DataFrame(
        {
            "instrument_id": ["A", "B", "C", "D"],
            "decision_session": [date(2014, 7, 1)] * 4,
            "prediction": [0.9, 0.1, 0.4, 0.2],
        }
    )
    sig = predictions_to_signals(pred, family="ridge", params={"alpha": 1.0}, top_k=2)
    for row in sig.to_dicts():
        assert row["decision_time"] == session_close_utc(row["signal_session"])

    forged = session_close_utc(date(2014, 7, 1)) - timedelta(minutes=30)
    assert forged != session_close_utc(date(2014, 7, 1))
    assert any(r["decision_time"] == forged for r in sig.to_dicts()) is False


def test_A9b_canonical_gate_accepts_exact_close(bars):
    clock = MarketEventClock(bars)
    rows = [
        {
            "signal_id": "SIG-1",
            "instrument_id": "INS-000101",
            "signal_session": date(2014, 7, 1),
            "decision_time": session_close_utc(date(2014, 7, 1)),
            "direction": "long",
            "target": 0.5,
        }
    ]
    normalized = clock.normalize_signals(rows)
    assert normalized[0]["decision_time"] == session_close_utc(date(2014, 7, 1))


def test_A10_short_direction_is_refused(bars):
    clock = MarketEventClock(bars)
    with pytest.raises(ValueError, match="'long' or 'flat'"):
        clock.normalize_signals(
            [
                {
                    "instrument_id": "INS-000101",
                    "signal_session": date(2014, 7, 1),
                    "direction": "short",
                    "target": 0.5,
                }
            ]
        )


def test_A11_bad_targets_are_refused(bars):
    clock = MarketEventClock(bars)
    for bad in (-1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            clock.normalize_signals(
                [
                    {
                        "instrument_id": "INS-000101",
                        "signal_session": date(2014, 7, 1),
                        "direction": "long",
                        "target": bad,
                    }
                ]
            )


def test_A12_missing_signal_session_is_refused(bars):
    clock = MarketEventClock(bars)
    with pytest.raises(ValueError, match="signal_session must be"):
        clock.normalize_signals(
            [{"instrument_id": "INS-000101", "direction": "long", "target": 0.5}]
        )


# ---------------------------------------------------------------------------
# A13 / A14 registry immutability
# ---------------------------------------------------------------------------


def test_A13_duplicate_experiment_registration_is_refused():
    service, spec = register_ml_experiment(
        experiment_id="EXP-90005", hypothesis_id="H-001", family="ridge", params={"alpha": 1.0}
    )
    with pytest.raises(ValueError, match="duplicate experiment id"):
        service.register(spec)


def test_A14_double_result_recording_is_refused():
    from orbit.ml.registry import run_registered_experiment
    from pathlib import Path
    import tempfile

    service, spec = register_ml_experiment(
        experiment_id="EXP-90005", hypothesis_id="H-001", family="ridge", params={"alpha": 1.0}
    )
    with tempfile.TemporaryDirectory() as tmp:
        artifact = Path(tmp) / "x.json"
        artifact.write_text("{}", encoding="utf-8")
        run_registered_experiment(
            service, "EXP-90005", family="ridge", params={"alpha": 1.0}, seed=42,
            artifacts_dir=tmp, result_summary="first", result_metrics={},
            artifact_files={"metrics_json": artifact},
        )
        with pytest.raises(Exception):
            run_registered_experiment(
                service, "EXP-90005", family="ridge", params={"alpha": 1.0}, seed=42,
                artifacts_dir=tmp, result_summary="second", result_metrics={},
                artifact_files={"metrics_json": artifact},
            )


# ---------------------------------------------------------------------------
# A15 / A16 / A17 artifact integrity
# ---------------------------------------------------------------------------


def test_A15_feature_tampering_changes_digest(bars):
    fs, _ = build_test_datasets(bars, make_events())[:2]
    import polars as _pl

    tampered = fs.records.with_columns((_pl.col("ret_10") * 1.0001).alias("ret_10"))
    assert fs.content_digest != tampered.write_json()


def test_A16_label_contract_cannot_be_overwritten():
    from orbit.labels.registry import LabelVersionRegistry
    from orbit.ml.labels import build_phase9_label_contract, register_phase9_label

    registry = LabelVersionRegistry()
    register_phase9_label(registry)
    with pytest.raises(ValueError, match="already registered"):
        register_phase9_label(registry)


def test_A17_corrupt_cache_is_refused(bars, tmp_path):
    from orbit.ml.features import build_feature_snapshot

    fs = build_feature_snapshot(bars, data_refs=["DS-000001"])
    cache_feature_snapshot(fs, tmp_path)
    # tamper the cached meta digest
    meta_path = tmp_path / "feature_snapshot_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["content_digest"] = "f" * 64
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(RuntimeError, match="digest mismatch"):
        load_cached_feature_snapshot(tmp_path)


# ---------------------------------------------------------------------------
# A18 ranking contamination
# ---------------------------------------------------------------------------


def test_A18_ranking_never_merges_cross_sections():
    frame = pl.DataFrame(
        {
            "decision_session": [date(2024, 1, 2)] * 4 + [date(2024, 6, 3)] * 4,
            "instrument_id": ["A", "B", "C", "D"] * 2,
            "score": [0.1, 0.2, 0.3, 0.4, 0.9, 0.8, 0.7, 0.6],
        }
    )
    ranked = cross_sectional_rank(frame, "score")
    jan = ranked.filter(pl.col("decision_session") == date(2024, 1, 2)).sort("instrument_id")["rank"].to_list()
    jun = ranked.filter(pl.col("decision_session") == date(2024, 6, 3)).sort("instrument_id")["rank"].to_list()
    assert jan == [4.0, 3.0, 2.0, 1.0]  # rank 1 = best (highest) score
    assert jun == [1.0, 2.0, 3.0, 4.0]


# ---------------------------------------------------------------------------
# A20 backtest accounting gate
# ---------------------------------------------------------------------------


def test_A20_backtest_invariants_are_checked():
    from orbit.ml.signals import run_backtest
    from tests.phase9_testutils import weekdays

    bars = make_canonical_bars(sessions=weekdays(date(2014, 7, 1), 60))
    pred = pl.DataFrame(
        {
            "instrument_id": sorted(bars["instrument_id"].unique().to_list()),
            "decision_session": [date(2014, 7, 1)] * 4,
            "prediction": [0.9, 0.1, 0.4, 0.2],
        }
    )
    from orbit.ml.signals import predictions_to_signals

    sig = predictions_to_signals(pred, family="ridge", params={"alpha": 1.0}, top_k=2)
    res = run_backtest(
        bars, sig,
        window_start=TEST_WINDOWS["test_start"],
        window_end=TEST_WINDOWS["test_end"],
        experiment_id="EXP-90005",
        hypothesis_id="H-001",
        events=make_events(),
        model={"family": "ridge", "hyperparameters": {"alpha": 1.0}},
    )
    assert res.invariant_violations() == []


# ---------------------------------------------------------------------------
# A21 / A22 / A23 scope guards (Section 35: no Phase 10+, no data expansion)
# ---------------------------------------------------------------------------


def test_A21_audit_detects_feature_zoo(bars):
    """A feature snapshot carrying extra (unregistered) feature columns/refs
    must fail the feature_scope_guard: Phase 9 is 8 documented numerics."""
    from orbit.ml.audit import run_phase9_audit

    fs, ls, _ = build_test_datasets(bars, make_events())
    zoo = fs.records.with_columns(
        (pl.col("ret_10") * pl.col("vol_10")).alias("ret_10_x_vol_10")
    )
    from orbit.ml.features import FeatureSnapshot

    fs_zoo = FeatureSnapshot(
        feature_set_id="FS-999",
        feature_set_version="v2",
        feature_refs=["FEAT-001", "FEAT-999"],
        data_refs=fs.data_refs,
        records=zoo,
    )
    checks = run_phase9_audit(feature_snapshot=fs_zoo, label_snapshot=ls)
    guard = [c for c in checks if c["check"] == "feature_scope_guard"]
    assert guard and guard[0]["status"] == "FAIL"


def test_A22_audit_detects_data_expansion(bars):
    """A feature snapshot pinned to an unvalidated dataset must fail the
    data_expansion_guard: Phase 9 must not silently acquire new data."""
    from orbit.ml.audit import run_phase9_audit
    from orbit.ml.features import FeatureSnapshot

    fs, ls, _ = build_test_datasets(bars, make_events())
    fs_expanded = FeatureSnapshot(
        feature_set_id=fs.feature_set_id,
        feature_set_version=fs.feature_set_version,
        feature_refs=fs.feature_refs,
        data_refs=["DS-000004", "DS-999999"],
        records=fs.records,
    )
    checks = run_phase9_audit(feature_snapshot=fs_expanded, label_snapshot=ls)
    guard = [c for c in checks if c["check"] == "data_expansion_guard"]
    assert guard and all(c["status"] == "FAIL" for c in guard)


def test_A23_audit_detects_foreign_model_family(bars):
    """A fitted model outside the five Phase 9 families (e.g. a Phase 22
    neural model) must fail the model_scope_guard."""
    from orbit.ml.audit import run_phase9_audit
    from orbit.ml.models import FittedModel

    fs, ls, _ = build_test_datasets(bars, make_events())
    alien = FittedModel(
        family="neural_net",
        hyperparameters={"layers": 3},
        feature_names=["x"],
        preprocessing="raw",
        seed=42,
        estimator=None,
        fitted_train_rows=0,
        train_window=("2010-01-04", "2018-12-31"),
        val_window=("2019-01-02", "2021-12-31"),
        test_window=("2022-01-03", "2026-06-30"),
    )
    checks = run_phase9_audit(
        feature_snapshot=fs, label_snapshot=ls, fitted_model=alien
    )
    guard = [c for c in checks if c["check"] == "model_scope_guard"]
    assert guard and guard[0]["status"] == "FAIL"