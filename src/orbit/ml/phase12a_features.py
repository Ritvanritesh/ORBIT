"""Phase 12A combined feature builder.

Assembles all three information families (market, sector, cross-sectional)
into unified feature snapshots for the benchmark suite.
"""

from __future__ import annotations

from typing import Any

import polars as pl

from orbit.ml.features import (
    build_feature_snapshot,
    attach_decision_times,
    _feature_names_for_ids,
)
from orbit.ml.phase12a_market import compute_market_features
from orbit.ml.phase12a_sector import compute_sector_features, load_sector_mapping
from orbit.ml.phase12a_cross_sectional import compute_cross_sectional_features
from orbit.ml.phase12a_plan import PHASE12A_FEATURE_SETS, PHASE12A_FEATURE_NAMES


def build_phase12a_feature_snapshots(
    bars: pl.DataFrame,
    benchmark_bars: pl.DataFrame,
    instruments: list[Any],
    data_refs: list[str] | None = None,
) -> dict[str, Any]:
    """Build all Phase 12A feature snapshots.

    Returns dict with keys:
    - "FS-001": baseline OHLCV feature snapshot
    - "FS-101": baseline + market context
    - "FS-102": baseline + sector context
    - "FS-103": baseline + cross-sectional context
    - "FS-104": baseline + all context families

    Each value is a FeatureSnapshot.
    """
    refs = data_refs or []

    # Build baseline FS-001
    print("  Building FS-001 (baseline)...")
    fs001 = build_feature_snapshot(bars, data_refs=refs)
    fs001_names = _feature_names_for_ids(fs001.feature_refs)

    # Build universe sessions from FS-001
    universe_sessions = fs001.records.select(
        "instrument_id", "decision_session"
    ).unique()

    # Family A: Market context
    print("  Building market context features...")
    market_features = compute_market_features(benchmark_bars, universe_sessions)

    # Family B: Sector context
    print("  Building sector context features...")
    sector_map = load_sector_mapping(instruments)
    sector_features = compute_sector_features(bars, sector_map, universe_sessions)

    # Family C: Cross-sectional features (from baseline features)
    print("  Building cross-sectional features...")
    xs_features = compute_cross_sectional_features(
        fs001.records, universe_sessions, fs001_names, min_population=5
    )

    snapshots = {"FS-001": fs001}

    # FS-101: Baseline + market context
    print("  Building FS-101 (baseline + market)...")
    fs101_frame = fs001.records
    if market_features.height > 0:
        market_cols = [c for c in market_features.columns
                       if c not in ("instrument_id", "decision_session", "window_end_session")]
        fs101_frame = fs101_frame.join(
            market_features.select("instrument_id", "decision_session", *market_cols),
            on=["instrument_id", "decision_session"],
            how="left",
        )
    fs101_frame = attach_decision_times(fs101_frame)
    fs101_names = [PHASE12A_FEATURE_NAMES.get(f, f) for f in PHASE12A_FEATURE_SETS["FS-101"]["feature_refs"]]
    from orbit.ml.features import FeatureSnapshot, assert_features_point_in_time
    snapshots["FS-101"] = FeatureSnapshot(
        feature_set_id="FS-101",
        feature_set_version="v1",
        feature_refs=PHASE12A_FEATURE_SETS["FS-101"]["feature_refs"],
        data_refs=refs,
        records=fs101_frame,
        transformation="phase12a_baseline_market_v1",
        limitations=["market features from SPY only", "point-in-time enforced"],
    )

    # FS-102: Baseline + sector context
    print("  Building FS-102 (baseline + sector)...")
    fs102_frame = fs001.records
    if sector_features.height > 0:
        sector_cols = [c for c in sector_features.columns
                       if c not in ("instrument_id", "decision_session", "window_end_session")]
        fs102_frame = fs102_frame.join(
            sector_features.select("instrument_id", "decision_session", *sector_cols),
            on=["instrument_id", "decision_session"],
            how="left",
        )
    fs102_frame = attach_decision_times(fs102_frame)
    snapshots["FS-102"] = FeatureSnapshot(
        feature_set_id="FS-102",
        feature_set_version="v1",
        feature_refs=PHASE12A_FEATURE_SETS["FS-102"]["feature_refs"],
        data_refs=refs,
        records=fs102_frame,
        transformation="phase12a_baseline_sector_v1",
        limitations=[
            "sector membership treated as time-invariant",
            "point-in-time enforced",
        ],
    )

    # FS-103: Baseline + cross-sectional context
    print("  Building FS-103 (baseline + cross-sectional)...")
    fs103_frame = fs001.records
    if xs_features.height > 0:
        xs_cols = [c for c in xs_features.columns
                   if c not in ("instrument_id", "decision_session", "window_end_session")]
        fs103_frame = fs103_frame.join(
            xs_features.select("instrument_id", "decision_session", *xs_cols),
            on=["instrument_id", "decision_session"],
            how="left",
        )
    fs103_frame = attach_decision_times(fs103_frame)
    snapshots["FS-103"] = FeatureSnapshot(
        feature_set_id="FS-103",
        feature_set_version="v1",
        feature_refs=PHASE12A_FEATURE_SETS["FS-103"]["feature_refs"],
        data_refs=refs,
        records=fs103_frame,
        transformation="phase12a_baseline_xs_v1",
        limitations=[
            "cross-sectional features use min 5 instruments per session",
            "point-in-time enforced",
        ],
    )

    # FS-104: Baseline + all context families
    print("  Building FS-104 (baseline + all context)...")
    fs104_frame = fs001.records
    if market_features.height > 0:
        fs104_frame = fs104_frame.join(
            market_features.select("instrument_id", "decision_session", *market_cols),
            on=["instrument_id", "decision_session"],
            how="left",
        )
    if sector_features.height > 0:
        fs104_frame = fs104_frame.join(
            sector_features.select("instrument_id", "decision_session", *sector_cols),
            on=["instrument_id", "decision_session"],
            how="left",
        )
    if xs_features.height > 0:
        fs104_frame = fs104_frame.join(
            xs_features.select("instrument_id", "decision_session", *xs_cols),
            on=["instrument_id", "decision_session"],
            how="left",
        )
    fs104_frame = attach_decision_times(fs104_frame)
    snapshots["FS-104"] = FeatureSnapshot(
        feature_set_id="FS-104",
        feature_set_version="v1",
        feature_refs=PHASE12A_FEATURE_SETS["FS-104"]["feature_refs"],
        data_refs=refs,
        records=fs104_frame,
        transformation="phase12a_baseline_all_context_v1",
        limitations=[
            "market from SPY only",
            "sector membership time-invariant",
            "cross-sectional min 5 instruments",
            "point-in-time enforced",
        ],
    )

    return snapshots
