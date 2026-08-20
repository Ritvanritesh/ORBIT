"""Phase 10 audit tests: the independent audit must pass on a clean build and
fail loudly on each tampered condition."""

from __future__ import annotations

import polars as pl
import pytest
from datetime import date

from orbit.ml.features import (
    FEATURE_DEFINITIONS,
    FEATURE_NAMES,
    FEATURE_NAMES_PHASE10,
    build_feature_snapshot,
    build_feature_snapshot_phase10,
    build_phase10_all_feature_frame,
    build_phase10_feature_set_snapshot,
)
from orbit.ml.labels import build_phase9_label_snapshot
from orbit.ml.phase10_audit import (
    audit_summary,
    run_phase10_audit,
    verify_feature_temporal_boundary,
    verify_row_identity,
)
from orbit.ml.phase10_plan import phase10_plan
from tests.phase9_testutils import TEST_WINDOWS, make_canonical_bars, make_events


@pytest.fixture(scope="module")
def clean_audit():
    bars = make_canonical_bars()
    allf = build_phase10_all_feature_frame(bars)
    fs001 = build_feature_snapshot(bars, data_refs=["DS-000001"])
    fs002 = build_feature_snapshot_phase10(bars, data_refs=["DS-000001"])
    snapshots = {"FS-001": fs001, "FS-002": fs002}
    for sid in ("FS-003", "FS-004", "FS-005", "FS-006", "FS-007", "FS-008",
                "FS-009", "FS-010", "FS-011", "FS-012", "FS-013"):
        snapshots[sid] = build_phase10_feature_set_snapshot(sid, allf, data_refs=["DS-000001"])
    decisions = fs001.records.select("instrument_id", "decision_time")
    ls = build_phase9_label_snapshot(bars, make_events(), [], decisions, data_refs=["DS-000001"])

    from orbit.ml.dataset import assemble_datasets

    datasets_by_set = {}
    for sid in ("FS-001", "FS-003"):
        names = (
            list(FEATURE_NAMES)
            if sid == "FS-001"
            else list(FEATURE_NAMES) + list(FEATURE_NAMES_PHASE10)
        )
        datasets_by_set[sid] = assemble_datasets(
            snapshots[sid], ls, windows=TEST_WINDOWS, feature_names=names
        )
    checks = run_phase10_audit(
        snapshots=snapshots,
        base_snapshot=fs001,
        label_snapshot=ls,
        datasets_by_set=datasets_by_set,
        phase9_fs001_digest=fs001.content_digest,
        bars=bars,
    )
    return {"checks": checks, "snapshots": snapshots, "ls": ls, "bars": bars}


def test_clean_audit_passes_everything(clean_audit):
    summary = audit_summary(clean_audit["checks"])
    assert summary["failed"] == 0, summary
    names = {c["check"] for c in clean_audit["checks"]}
    for required in (
        "plan_lock",
        "point_in_time_FS-001",
        "point_in_time_FS-013",
        "membership_FS-003",
        "membership_FS-009",
        "strong_temporal_boundary",
        "feature_scope_FS-003",
        "phase9_fs001_frozen",
        "label_contract",
        "split_integrity_FS-001",
        "split_integrity_FS-003",
        "row_identity_phase10_sets",
        "row_identity_fs001_warmup",
    ):
        assert required in names, required


def test_strong_temporal_boundary_recomputes_phase10_features(clean_audit):
    """The boundary check recomputes the 15 Phase 10 features from bars
    strictly before each decision session and finds no mismatches."""
    check = [c for c in clean_audit["checks"] if c["check"] == "strong_temporal_boundary"][0]
    assert check["status"] == "PASS"
    assert "mismatches" in check["evidence"]


def test_audit_catches_fs001_digest_mismatch(clean_audit):
    checks = run_phase10_audit(
        snapshots=clean_audit["snapshots"],
        base_snapshot=clean_audit["snapshots"]["FS-001"],
        label_snapshot=clean_audit["ls"],
        phase9_fs001_digest="0" * 64,  # wrong stored digest
        bars=clean_audit["bars"],
    )
    fz = [c for c in checks if c["check"] == "phase9_fs001_frozen"][0]
    assert fz["status"] == "FAIL"


def test_audit_catches_bad_membership(clean_audit):
    snap = clean_audit["snapshots"]["FS-004"]
    tampered = pl.DataFrame(snap.records).with_columns(pl.lit("FEAT-999").alias("__fake"))
    # membership assertion compares feature_refs (metadata), not columns;
    # a tampered refs list must fail
    import types

    fake = types.SimpleNamespace(**vars(snap))
    fake.feature_refs = ["FEAT-001", "FEAT-999"]
    checks = run_phase10_audit(
        snapshots={"FS-004": fake},
        phase9_fs001_digest="x",
    )
    m = [c for c in checks if c["check"] == "membership_FS-004"][0]
    assert m["status"] == "FAIL"


def test_audit_summary_blocks_on_failure():
    s = audit_summary([{"check": "a", "status": "PASS", "evidence": ""},
                       {"check": "b", "status": "FAIL", "evidence": "x"}])
    assert s["blocked"] is True
    assert s["failed"] == 1
    assert s["failed_checks"] == ["b"]


def test_verify_feature_temporal_boundary_detects_same_bar_leak(clean_audit):
    """Injecting a same-bar value into the snapshot must be caught: the
    recomputed value (from bars strictly before D) will differ."""
    bars = clean_audit["bars"]
    rec = clean_audit["snapshots"]["FS-003"].records.clone()
    # tamper per-instrument sorted row 240 (divisible by 3 -> always in the
    # audit's per-instrument 3-stride sample, and past all warm-ups)
    targets = (
        rec.sort(["instrument_id", "decision_session"])
        .group_by("instrument_id")
        .agg(pl.col("decision_session").gather(pl.int_range(240, 241)))
        .explode("decision_session")
    )
    n_targets = 0
    for t in targets.iter_rows(named=True):
        inst = t["instrument_id"]
        d = t["decision_session"]
        hist = bars.filter(
            (pl.col("instrument_id") == inst) & (pl.col("trade_date") < d)
        ).sort("trade_date")
        same_bar_close = float(
            bars.filter(
                (pl.col("instrument_id") == inst) & (pl.col("trade_date") == d)
            )["close"][0]
        )
        c_5 = float(hist["close"][-6])
        leaked = same_bar_close / c_5 - 1.0  # same-bar (D) close in numerator
        true_val = float(
            rec.filter(
                (pl.col("instrument_id") == inst) & (pl.col("decision_session") == d)
            )["ret_5"][0]
        )
        assert abs(true_val - leaked) > 1e-9
        rec = rec.with_columns(
            pl.when(
                (pl.col("instrument_id") == inst) & (pl.col("decision_session") == d)
            )
            .then(pl.lit(leaked))
            .otherwise(pl.col("ret_5"))
            .alias("ret_5")
        )
        n_targets += 1
    assert n_targets >= 2
    tb = verify_feature_temporal_boundary(rec, bars, FEATURE_NAMES_PHASE10)
    assert tb["mismatches"] >= n_targets
    assert tb["valid"] is False


def test_row_identity_clean_and_warmup_explained(clean_audit):
    """FS-002..FS-013 resolve exactly the same rows; FS-001's extra rows are
    all inside the Phase 10 warm-up zone (session index < 200) - the ablation
    is row-fair wherever the Phase 10 sets have features."""
    ri = verify_row_identity(clean_audit["snapshots"], clean_audit["bars"])
    assert ri["phase10_sets_identical"] is True
    assert ri["phase10_only_rows"] == 0
    assert ri["fs001_only_rows"] > 0
    assert ri["fs001_only_within_warm_up"] == ri["fs001_only_rows"]
    assert ri["fs001_only_beyond_warm_up"] == 0
    assert ri["valid"] is True


def test_row_identity_detects_set_drift(clean_audit):
    """A Phase 10 set that loses rows vs its siblings must fail loudly."""
    import types

    snapshots = dict(clean_audit["snapshots"])
    tampered = types.SimpleNamespace(**vars(snapshots["FS-004"]))
    tampered.records = snapshots["FS-004"].records.head(
        snapshots["FS-004"].records.height - 5
    )
    snapshots["FS-004"] = tampered
    ri = verify_row_identity(snapshots, clean_audit["bars"])
    assert ri["phase10_sets_identical"] is False
    assert ri["valid"] is False


def test_row_identity_detects_row_beyond_warmup(clean_audit):
    """An FS-001 row that the Phase 10 warm-up policy cannot explain (session
    index >= 200 but absent from the Phase 10 sets) must fail loudly."""
    import types

    snapshots = dict(clean_audit["snapshots"])
    rec = snapshots["FS-002"].records.clone()
    # sorted by (instrument_id, decision_session); per instrument the first
    # 200 sessions are the Phase 10 warm-up zone, so pick a session far past
    # it (instrument 1, session index 1000 of ~1500).
    victim = rec.sort(["instrument_id", "decision_session"]).slice(1000, 1)
    inst = victim["instrument_id"][0]
    d = victim["decision_session"][0]
    rec = rec.filter(
        ~((pl.col("instrument_id") == inst) & (pl.col("decision_session") == d))
    )
    tampered = types.SimpleNamespace(**vars(snapshots["FS-002"]))
    tampered.records = rec
    snapshots["FS-002"] = tampered
    ri = verify_row_identity(snapshots, clean_audit["bars"])
    assert ri["fs001_only_beyond_warm_up"] >= 1
    assert ri["valid"] is False


def test_audit_emits_deep_checks_with_real_artifacts(clean_audit):
    """test_lock / grid_lock / model_scope_guard / seed_lock /
    preprocessing_train_only / registry_lineage are exercised when the
    runner-level inputs are provided (they must never be silent on the real
    run)."""
    import types

    from orbit.ml.dataset import assemble_datasets
    from orbit.ml.features import FEATURE_NAMES
    from orbit.ml.phase10_plan import phase10_plan
    from orbit.ml.phase10_registry import register_phase10_experiment

    datasets = assemble_datasets(
        clean_audit["snapshots"]["FS-001"], clean_audit["ls"],
        windows=TEST_WINDOWS, feature_names=list(FEATURE_NAMES),
    )
    _, spec = register_phase10_experiment(
        experiment_id="EXP-10001", hypothesis_id="H-001",
        feature_set_id="FS-001", feature_set_version="v1",
        family="ridge", params={"alpha": 1.0}, seed=42,
        plan_digest=phase10_plan()["plan_digest"],
    )
    fitted = types.SimpleNamespace(
        family="ridge", hyperparameters={"alpha": 1.0},
        preprocessing="standardized", seed=42,
    )
    checks = run_phase10_audit(
        snapshots=clean_audit["snapshots"],
        base_snapshot=clean_audit["snapshots"]["FS-001"],
        label_snapshot=clean_audit["ls"],
        phase9_fs001_digest=clean_audit["snapshots"]["FS-001"].content_digest,
        bars=clean_audit["bars"],
        test_predictions=datasets["test"][3],
        fitted_model=fitted,
        experiment_spec=spec,
        windows=TEST_WINDOWS,
    )
    names = {c["check"]: c["status"] for c in checks}
    for required in (
        "test_lock",
        "grid_lock",
        "model_scope_guard",
        "seed_lock",
        "preprocessing_train_only",
        "registry_lineage",
    ):
        assert names.get(required) == "PASS", (required, names.get(required))


def test_audit_test_lock_uses_run_windows(clean_audit):
    """The test-lock check must compare against the windows the run actually
    used, so hermetic runs with test windows do not false-fail."""
    from orbit.ml.dataset import assemble_datasets
    from orbit.ml.features import FEATURE_NAMES

    datasets = assemble_datasets(
        clean_audit["snapshots"]["FS-001"], clean_audit["ls"],
        windows=TEST_WINDOWS, feature_names=list(FEATURE_NAMES),
    )
    checks = run_phase10_audit(
        snapshots=clean_audit["snapshots"],
        base_snapshot=clean_audit["snapshots"]["FS-001"],
        label_snapshot=clean_audit["ls"],
        phase9_fs001_digest="x",
        test_predictions=datasets["test"][3],
        windows=TEST_WINDOWS,
    )
    lock = [c for c in checks if c["check"] == "test_lock"][0]
    assert lock["status"] == "PASS"
    # without the run windows the same predictions fail against the locked
    # Phase 9 test window (hermetic data has no 2022-2026 sessions)
    checks = run_phase10_audit(
        snapshots=clean_audit["snapshots"],
        base_snapshot=clean_audit["snapshots"]["FS-001"],
        label_snapshot=clean_audit["ls"],
        phase9_fs001_digest="x",
        test_predictions=datasets["test"][3],
    )
    lock = [c for c in checks if c["check"] == "test_lock"][0]
    assert lock["status"] == "FAIL"