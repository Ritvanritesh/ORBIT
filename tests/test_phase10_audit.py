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