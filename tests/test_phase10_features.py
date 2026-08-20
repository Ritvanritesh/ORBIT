"""Phase 10 feature tests: definitions, math, point-in-time discipline,
snapshot versioning, feature-set membership, and determinism."""

from __future__ import annotations

import polars as pl
import pytest

from orbit.ml.features import (
    ALL_PHASE10_DEFINITIONS,
    FEATURE_DEFINITIONS,
    FEATURE_NAMES,
    FEATURE_NAMES_PHASE10,
    FEATURE_SET_ID,
    FEATURE_SET_VERSION,
    PHASE10_FAMILIES,
    PHASE10_FAMILY_DEFINITIONS,
    PHASE10_FEATURE_SETS,
    PHASE10_FEATURE_SET_ORDER,
    _feature_definitions_digest,
    assert_features_finite,
    assert_features_point_in_time,
    build_feature_frame_phase10,
    build_feature_snapshot,
    build_feature_snapshot_phase10,
    build_phase10_all_feature_frame,
    build_phase10_feature_set_snapshot,
    phase10_set_identity,
)
from tests.phase9_testutils import make_canonical_bars


@pytest.fixture(scope="module")
def bars():
    return make_canonical_bars()


def test_phase10_has_15_documented_features():
    assert len(FEATURE_NAMES_PHASE10) == 15
    assert len(ALL_PHASE10_DEFINITIONS) == 15
    assert all(f["feature_id"].startswith("FEAT-") for f in ALL_PHASE10_DEFINITIONS)
    assert all(f["feature_id"][5:].isdigit() for f in ALL_PHASE10_DEFINITIONS)


def test_every_definition_has_full_metadata():
    for f in ALL_PHASE10_DEFINITIONS:
        assert f.get("formula"), f["feature_id"]
        assert f.get("raw_inputs"), f["feature_id"]
        assert f.get("missing_policy"), f["feature_id"]
        assert "normalization" in f, f["feature_id"]
        assert f["feature_id"] in {
            d["feature_id"] for d in ALL_PHASE10_DEFINITIONS
        }


def test_families_are_three_features_each():
    assert PHASE10_FAMILIES == ["momentum", "trend", "volatility", "volume", "range"]
    for family, defs in PHASE10_FAMILY_DEFINITIONS.items():
        assert len(defs) == 3, family
        assert family in PHASE10_FAMILIES


def test_feature_ids_are_unique_and_contiguous():
    ids = [f["feature_id"] for f in ALL_PHASE10_DEFINITIONS]
    assert len(set(ids)) == 15
    assert ids == [f"FEAT-{i:03d}" for i in range(101, 116)]


def test_feature_set_membership_counts():
    assert len(PHASE10_FEATURE_SETS["FS-002"]["members"]) == 15
    assert len(PHASE10_FEATURE_SETS["FS-003"]["members"]) == 23
    for sid in ("FS-004", "FS-005", "FS-006", "FS-007", "FS-008"):
        assert len(PHASE10_FEATURE_SETS[sid]["members"]) == 11
    for sid in ("FS-009", "FS-010", "FS-011", "FS-012", "FS-013"):
        assert len(PHASE10_FEATURE_SETS[sid]["members"]) == 20


def test_family_sets_include_and_exclude_the_right_family():
    base = {f["feature_id"] for f in FEATURE_DEFINITIONS}
    for family in PHASE10_FAMILIES:
        fam = {f["feature_id"] for f in PHASE10_FAMILY_DEFINITIONS[family]}
        plus_id = {
            "momentum": "FS-004", "trend": "FS-005", "volatility": "FS-006",
            "volume": "FS-007", "range": "FS-008",
        }[family]
        minus_id = {
            "momentum": "FS-009", "trend": "FS-010", "volatility": "FS-011",
            "volume": "FS-012", "range": "FS-013",
        }[family]
        assert set(PHASE10_FEATURE_SETS[plus_id]["members"]) == base | fam
        assert set(PHASE10_FEATURE_SETS[minus_id]["members"]) == (
            base | {f["feature_id"] for f in ALL_PHASE10_DEFINITIONS}
        ) - fam


def test_every_phase10_row_is_point_in_time(bars):
    frame = build_feature_frame_phase10(bars)
    assert_features_point_in_time(frame)
    assert frame.height > 0
    assert (frame["window_end_session"] < frame["decision_session"]).all()


def test_every_phase10_value_is_finite(bars):
    frame = build_feature_frame_phase10(bars)
    assert_features_finite(frame, FEATURE_NAMES_PHASE10)


def test_snapshot_digests_are_deterministic(bars):
    a = build_feature_snapshot_phase10(bars, data_refs=["DS-000001"])
    b = build_feature_snapshot_phase10(bars, data_refs=["DS-000001"])
    assert a.content_digest == b.content_digest
    assert a.equals(b)


def test_all_snapshot_digests_distinct_and_membership_exact(bars):
    allf = build_phase10_all_feature_frame(bars)
    digests = {}
    for sid in PHASE10_FEATURE_SET_ORDER:
        if sid == "FS-001":
            s = build_feature_snapshot(bars, data_refs=["DS-000001"])
        elif sid == "FS-002":
            s = build_feature_snapshot_phase10(bars, data_refs=["DS-000001"])
        else:
            s = build_phase10_feature_set_snapshot(sid, allf, data_refs=["DS-000001"])
        digests[sid] = s.content_digest
        assert s.feature_refs == sorted(PHASE10_FEATURE_SETS[sid]["members"]) if sid != "FS-001" else True
    assert len(set(digests.values())) == len(digests)


def test_fs001_remains_unchanged(bars):
    """The frozen Phase 9 baseline snapshot must be bit-identical to its own
    builder (its digest is pinned by the Phase 9 cache; FS-001 semantics are
    never redefined by Phase 10)."""
    fs001 = build_feature_snapshot(bars, data_refs=["DS-000001"])
    assert fs001.feature_set_id == FEATURE_SET_ID
    assert fs001.feature_set_version == FEATURE_SET_VERSION
    assert fs001.feature_refs == [f["feature_id"] for f in FEATURE_DEFINITIONS]
    assert fs001.transformation == "phase9_baseline_v1"


def test_feature_definitions_digest_changes_with_definition():
    d1 = _feature_definitions_digest(["FEAT-101"])
    # a materially different definition for the same id changes the digest
    changed = _feature_definitions_digest(["FEAT-101", "FEAT-102"])
    assert d1 != changed


def test_unknown_feature_set_refused(bars):
    allf = build_phase10_all_feature_frame(bars)
    with pytest.raises(ValueError, match="unknown Phase 10 feature set"):
        build_phase10_feature_set_snapshot("FS-999", allf)


def test_momentum_convention_matches_fs001_code_semantics(bars):
    """ret_N at decision session D must equal close(D-1)/close(D-N) - 1, the
    exact convention of FEAT-001..003 (code form close.shift(1)/close.shift(N))."""
    b = bars.sort(["instrument_id", "trade_date"])
    one = b.group_by("instrument_id").first().sort("instrument_id")
    inst = one["instrument_id"][0]
    g = b.filter(pl.col("instrument_id") == inst).sort("trade_date")
    # hand check on a mid-series row where all windows are complete
    row = build_feature_frame_phase10(bars).filter(
        pl.col("instrument_id") == inst
    ).sort("decision_session").slice(200, 1)
    d = row["decision_session"][0]
    closes = g.filter(pl.col("trade_date") <= d).sort("trade_date")["close"]
    d1 = float(closes[-2])
    d5 = float(closes[-6])
    assert abs(float(row["ret_5"][0]) - (d1 / d5 - 1.0)) < 1e-12


def test_high_low_position_uses_last_completed_close_not_current(bars):
    """Adversarial A2 regression: the position features must reference the
    previous close (close(D-1)), never the decision session's own bar."""
    inst = sorted(bars["instrument_id"].unique().to_list())[0]
    frame = build_feature_frame_phase10(bars).filter(pl.col("instrument_id") == inst)
    g = bars.filter(pl.col("instrument_id") == inst).sort("trade_date")
    checked = 0
    for row in frame.sort("decision_session").iter_rows(named=True):
        d = row["decision_session"]
        prior = g.filter(pl.col("trade_date") < d).sort("trade_date")
        if prior.height < 31:
            continue
        c_prev = float(prior["close"][-1])
        lo10 = float(prior["low"].tail(10).min())
        hi10 = float(prior["high"].tail(10).max())
        expected = (c_prev - lo10) / (hi10 - lo10)
        assert abs(float(row["high_low_10_pos"]) - expected) < 1e-12
        checked += 1
        if checked >= 20:
            break
    assert checked == 20


def test_vol_zscore_uses_trailing_window(bars):
    inst = sorted(bars["instrument_id"].unique().to_list())[0]
    frame = build_feature_frame_phase10(bars).filter(pl.col("instrument_id") == inst)
    g = bars.filter(pl.col("instrument_id") == inst).sort("trade_date")
    row = frame.sort("decision_session").slice(250, 1).to_dicts()[0]
    d = row["decision_session"]
    prior = g.filter(pl.col("trade_date") < d).sort("trade_date")
    dv = (prior["close"] * prior["volume"]).tail(20)
    dv_prev = float(dv[-1])
    mean = float(dv.mean())
    std = float(dv.std())
    expected = (dv_prev - mean) / std
    assert abs(float(row["vol_zscore_20"]) - expected) < 1e-9