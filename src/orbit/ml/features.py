"""FS-001 v1 feature snapshot: the Phase 9 numeric feature set.

Design (documented in docs/phase9_ml_benchmark.md):
  - Small, pre-registered set of 8 numeric features; every feature is a
    documented Phase 8 numeric primitive (momentum lookbacks, SMA ratios,
    trailing volatility, liquidity), none is a Phase 10 "feature zoo".
  - Point-in-time discipline: a feature row for decision session D is
    computed strictly from bars with session < D (the decision instant is the
    close of D, and Phase 4's strict boundary means bar D is not visible).
    Every row therefore references the instrument's own prior sessions only;
    the window_end_session column (the previous session in the series) makes
    the bound auditable.
  - Determinism: features are pure window functions of the stored OHLCV
    series; the snapshot digest is a sha256 over sorted canonical content.

Feature IDs (FEAT-001 .. FEAT-008), transformation phase9_baseline_v1:
  ret_10 / ret_20 / ret_30    momentum: close-to-close return over 10/20/30
                              completed sessions ending at D-1
  sma_ratio_5_30 / 15_40     short/long SMA ratio at the last completed bar
  vol_10 / vol_30            sample std of daily returns over 10/30 sessions
  log_dv_med_20              log1p(median dollar volume over 20 sessions)
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any, Callable

import polars as pl

FEATURE_SET_ID = "FS-001"
FEATURE_SET_VERSION = "v1"
FEATURE_TRANSFORMATION = "phase9_baseline_v1"

FEATURE_DEFINITIONS: list[dict[str, Any]] = [
    {"feature_id": "FEAT-001", "name": "ret_10", "window": 10, "kind": "momentum_return"},
    {"feature_id": "FEAT-002", "name": "ret_20", "window": 20, "kind": "momentum_return"},
    {"feature_id": "FEAT-003", "name": "ret_30", "window": 30, "kind": "momentum_return"},
    {"feature_id": "FEAT-004", "name": "sma_ratio_5_30", "short": 5, "long": 30, "kind": "moving_average_ratio"},
    {"feature_id": "FEAT-005", "name": "sma_ratio_15_40", "short": 15, "long": 40, "kind": "moving_average_ratio"},
    {"feature_id": "FEAT-006", "name": "vol_10", "window": 10, "kind": "realized_volatility"},
    {"feature_id": "FEAT-007", "name": "vol_30", "window": 30, "kind": "realized_volatility"},
    {"feature_id": "FEAT-008", "name": "log_dv_med_20", "window": 20, "kind": "liquidity"},
]

FEATURE_NAMES: list[str] = [f["name"] for f in FEATURE_DEFINITIONS]
FEATURE_ID_BY_NAME: dict[str, str] = {f["name"]: f["feature_id"] for f in FEATURE_DEFINITIONS}
MAX_FEATURE_WINDOW = 40  # longest primitive: sma_ratio_15_40

# =====================================================================
# Phase 10 feature families (defined separately so FS-001 v1 remains
# frozen; every new feature is documented in docs/phase10_feature_research.md).
#
# Conventions (identical to the FS-001 code semantics):
#   - A feature row at decision session D references only bars with
#     session <= D-1 (window_end_session = D-1; the bar of D is not
#     visible at close(D)).  No feature ever references its own or a
#     later session.
#   - Momentum returns: ret_N = close(D-1)/close(D-N) - 1, the return
#     over the N-session window ending at D-1 - exactly the convention
#     of FEAT-001..003 (close.shift(1)/close.shift(N) - 1).
#   - All features are per-instrument rolling functions of the stored
#     OHLCV series (no cross-sectional statistics, no future data).
#
# Feature IDs: FEAT-101..FEAT-115 (schema pattern ^FEAT-\d{3,}$).
# =====================================================================

PHASE10_FEATURE_VERSION = "v1"

MOMENTUM_FEATURE_DEFINITIONS: list[dict[str, Any]] = [
    {
        "feature_id": "FEAT-101",
        "name": "ret_5",
        "window": 5,
        "kind": "momentum_return",
        "formula": "close(D-1)/close(D-5) - 1",
        "raw_inputs": ["close"],
        "missing_policy": "null until 5 completed sessions before D",
        "normalization": "none (raw)",
    },
    {
        "feature_id": "FEAT-102",
        "name": "ret_60",
        "window": 60,
        "kind": "momentum_return",
        "formula": "close(D-1)/close(D-60) - 1",
        "raw_inputs": ["close"],
        "missing_policy": "null until 60 completed sessions before D",
        "normalization": "none (raw)",
    },
    {
        "feature_id": "FEAT-103",
        "name": "ret_120",
        "window": 120,
        "kind": "momentum_return",
        "formula": "close(D-1)/close(D-120) - 1",
        "raw_inputs": ["close"],
        "missing_policy": "null until 120 completed sessions before D",
        "normalization": "none (raw)",
    },
]

TREND_FEATURE_DEFINITIONS: list[dict[str, Any]] = [
    {
        "feature_id": "FEAT-104",
        "name": "sma_ratio_10_30",
        "short": 10,
        "long": 30,
        "kind": "moving_average_ratio",
        "formula": "SMA10(D-1)/SMA30(D-1) (ratio, same form as FEAT-004/005)",
        "raw_inputs": ["close"],
        "missing_policy": "null until 30 completed sessions before D",
        "normalization": "none (raw)",
    },
    {
        "feature_id": "FEAT-105",
        "name": "sma_ratio_20_50",
        "short": 20,
        "long": 50,
        "kind": "moving_average_ratio",
        "formula": "SMA20(D-1)/SMA50(D-1) (ratio, same form as FEAT-004/005)",
        "raw_inputs": ["close"],
        "missing_policy": "null until 50 completed sessions before D",
        "normalization": "none (raw)",
    },
    {
        "feature_id": "FEAT-106",
        "name": "price_distance_200ma",
        "window": 200,
        "kind": "trend_distance",
        "formula": "(close(D-1) - SMA200(D-1)) / SMA200(D-1)",
        "raw_inputs": ["close"],
        "missing_policy": "null until 200 completed sessions before D",
        "normalization": "none (raw)",
    },
]

VOLATILITY_FEATURE_DEFINITIONS: list[dict[str, Any]] = [
    {
        "feature_id": "FEAT-107",
        "name": "vol_60",
        "window": 60,
        "kind": "realized_volatility",
        "formula": "sample std of daily close-to-close returns over the 60 sessions ending at D-1 (same rolling_std semantics as FEAT-006/007)",
        "raw_inputs": ["close"],
        "missing_policy": "null until 60 completed sessions before D",
        "normalization": "none (raw)",
    },
    {
        "feature_id": "FEAT-108",
        "name": "vol_90",
        "window": 90,
        "kind": "realized_volatility",
        "formula": "sample std of daily close-to-close returns over the 90 sessions ending at D-1",
        "raw_inputs": ["close"],
        "missing_policy": "null until 90 completed sessions before D",
        "normalization": "none (raw)",
    },
    {
        "feature_id": "FEAT-109",
        "name": "vol_ratio_10_30",
        "short": 10,
        "long": 30,
        "kind": "volatility_ratio",
        "formula": "vol_10(D-1) / vol_30(D-1) (short/long realized-volatility ratio)",
        "raw_inputs": ["close"],
        "missing_policy": "null until 30 completed sessions before D; null when vol_30 is 0",
        "normalization": "none (raw)",
    },
]

VOLUME_FEATURE_DEFINITIONS: list[dict[str, Any]] = [
    {
        "feature_id": "FEAT-110",
        "name": "dv_med_10",
        "window": 10,
        "kind": "dollar_volume_median",
        "formula": "log1p(median(close*volume) over the 10 sessions ending at D-1) (same form as FEAT-008)",
        "raw_inputs": ["close", "volume"],
        "missing_policy": "null until 10 completed sessions before D",
        "normalization": "log1p (monotone, no fitted parameters)",
    },
    {
        "feature_id": "FEAT-111",
        "name": "dv_med_30",
        "window": 30,
        "kind": "dollar_volume_median",
        "formula": "log1p(median(close*volume) over the 30 sessions ending at D-1)",
        "raw_inputs": ["close", "volume"],
        "missing_policy": "null until 30 completed sessions before D",
        "normalization": "log1p (monotone, no fitted parameters)",
    },
    {
        "feature_id": "FEAT-112",
        "name": "vol_zscore_20",
        "window": 20,
        "kind": "volume_zscore",
        "formula": "(dv(D-1) - mean(dv over 20 sessions ending at D-1)) / std(dv over 20 sessions ending at D-1); dv = close*volume",
        "raw_inputs": ["close", "volume"],
        "missing_policy": "null until 20 completed sessions before D; null when the 20-session std is 0",
        "normalization": "per-instrument z-score vs the instrument's own trailing window (no cross-sectional statistics)",
    },
]

RANGE_FEATURE_DEFINITIONS: list[dict[str, Any]] = [
    {
        "feature_id": "FEAT-113",
        "name": "high_low_10_pos",
        "window": 10,
        "kind": "high_low_position",
        "formula": "(close(D-1) - min(low over 10 sessions ending at D-1)) / (max(high over 10 sessions ending at D-1) - min(low over 10 sessions ending at D-1))",
        "raw_inputs": ["close", "high", "low"],
        "missing_policy": "null until 10 completed sessions before D; null when the 10-session range is 0",
        "normalization": "none (raw)",
    },
    {
        "feature_id": "FEAT-114",
        "name": "high_low_30_pos",
        "window": 30,
        "kind": "high_low_position",
        "formula": "(close(D-1) - min(low over 30 sessions ending at D-1)) / (max(high over 30 sessions ending at D-1) - min(low over 30 sessions ending at D-1))",
        "raw_inputs": ["close", "high", "low"],
        "missing_policy": "null until 30 completed sessions before D; null when the 30-session range is 0",
        "normalization": "none (raw)",
    },
    {
        "feature_id": "FEAT-115",
        "name": "normalized_range_20",
        "window": 20,
        "kind": "normalized_range",
        "formula": "(max(high over 20 sessions ending at D-1) - min(low over 20 sessions ending at D-1)) / close(D-1)",
        "raw_inputs": ["close", "high", "low"],
        "missing_policy": "null until 20 completed sessions before D",
        "normalization": "none (raw)",
    },
]

ALL_PHASE10_DEFINITIONS: list[dict[str, Any]] = (
    MOMENTUM_FEATURE_DEFINITIONS
    + TREND_FEATURE_DEFINITIONS
    + VOLATILITY_FEATURE_DEFINITIONS
    + VOLUME_FEATURE_DEFINITIONS
    + RANGE_FEATURE_DEFINITIONS
)

FEATURE_NAMES_PHASE10: list[str] = [f["name"] for f in ALL_PHASE10_DEFINITIONS]
FEATURE_ID_BY_NAME_PHASE10: dict[str, str] = {
    f["name"]: f["feature_id"] for f in ALL_PHASE10_DEFINITIONS
}

MAX_FEATURE_WINDOW_PHASE10 = max(f.get("window", 0) for f in ALL_PHASE10_DEFINITIONS)

# Feature families: explicit grouping of the Phase 10 features.
PHASE10_FAMILIES: list[str] = ["momentum", "trend", "volatility", "volume", "range"]
PHASE10_FAMILY_DEFINITIONS: dict[str, list[dict[str, Any]]] = {
    "momentum": MOMENTUM_FEATURE_DEFINITIONS,
    "trend": TREND_FEATURE_DEFINITIONS,
    "volatility": VOLATILITY_FEATURE_DEFINITIONS,
    "volume": VOLUME_FEATURE_DEFINITIONS,
    "range": RANGE_FEATURE_DEFINITIONS,
}
FEATURE_FAMILY_BY_ID_PHASE10: dict[str, str] = {
    f["feature_id"]: family
    for family, defs in PHASE10_FAMILY_DEFINITIONS.items()
    for f in defs
}

# Pre-registered Phase 10 feature sets (immutable once used in an experiment).
# FS-001 v1 is the frozen Phase 9 baseline (never redefined here).
# FS-002 v1: the 15 new Phase 10 features (5 families x 3).
# FS-003 v1: ALL = FS-001 + FS-002 (23 features).
# FS-004..FS-008: BASE + one family (11 features each).
# FS-009..FS-013: ALL - one family (20 features each).
PHASE10_FEATURE_SETS: dict[str, dict[str, Any]] = {
    "FS-002": {
        "version": "v1",
        "members": [f["feature_id"] for f in ALL_PHASE10_DEFINITIONS],
        "role": "new",
        "description": "Phase 10 candidate families only (15 features)",
    },
    "FS-003": {
        "version": "v1",
        "members": [f["feature_id"] for f in FEATURE_DEFINITIONS]
        + [f["feature_id"] for f in ALL_PHASE10_DEFINITIONS],
        "role": "all",
        "description": "ALL: FS-001 v1 + FS-002 v1 (23 features)",
    },
}
for _family in PHASE10_FAMILIES:
    _fam_ids = [f["feature_id"] for f in PHASE10_FAMILY_DEFINITIONS[_family]]
    _base_ids = [f["feature_id"] for f in FEATURE_DEFINITIONS]
    _all_ids = [f["feature_id"] for f in FEATURE_DEFINITIONS] + [
        f["feature_id"] for f in ALL_PHASE10_DEFINITIONS
    ]
    _fs_plus = {
        "momentum": "FS-004",
        "trend": "FS-005",
        "volatility": "FS-006",
        "volume": "FS-007",
        "range": "FS-008",
    }[_family]
    _fs_minus = {
        "momentum": "FS-009",
        "trend": "FS-010",
        "volatility": "FS-011",
        "volume": "FS-012",
        "range": "FS-013",
    }[_family]
    PHASE10_FEATURE_SETS[_fs_plus] = {
        "version": "v1",
        "members": _base_ids + _fam_ids,
        "role": "base_plus_family",
        "family": _family,
        "description": f"BASE (FS-001 v1) + {_family} family ({len(_base_ids) + len(_fam_ids)} features)",
    }
    PHASE10_FEATURE_SETS[_fs_minus] = {
        "version": "v1",
        "members": [i for i in _all_ids if i not in _fam_ids],
        "role": "all_minus_family",
        "family": _family,
        "description": f"ALL (FS-003 v1) - {_family} family ({len(_all_ids) - len(_fam_ids)} features)",
    }

# Canonical ordering of Phase 10 feature sets (experiment id ordering).
PHASE10_FEATURE_SET_ORDER: list[str] = ["FS-001", "FS-002", "FS-003"] + [
    f"FS-{i:03d}" for i in range(4, 14)
]
for _fs_id in PHASE10_FEATURE_SET_ORDER[1:]:
    if _fs_id not in PHASE10_FEATURE_SETS:
        raise ValueError(f"feature set {_fs_id} missing from the Phase 10 registry")

# BASE feature set (the frozen Phase 9 baseline, used as the Phase 10 control).
PHASE10_BASE_FEATURE_SET = "FS-001"


def _feature_definitions_digest(feature_ids: list[str]) -> str:
    """sha256 over the canonical definitions of a feature set (deterministic).

    The digest covers each member's full definition payload, so a changed
    formula, window, raw input or missing-value policy changes the digest.
    """
    by_id = {
        f["feature_id"]: f
        for f in FEATURE_DEFINITIONS + ALL_PHASE10_DEFINITIONS
    }
    canonical = json.dumps(
        [by_id[fid] for fid in sorted(feature_ids)],
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def phase10_set_identity(feature_set_id: str) -> dict[str, Any]:
    """Deterministic identity of one Phase 10 feature set (for lineage)."""
    if feature_set_id == "FS-001":
        members = [f["feature_id"] for f in FEATURE_DEFINITIONS]
        version = FEATURE_SET_VERSION
        transformation = FEATURE_TRANSFORMATION
    else:
        entry = PHASE10_FEATURE_SETS.get(feature_set_id)
        if entry is None:
            raise ValueError(f"unknown Phase 10 feature set {feature_set_id!r}")
        members = list(entry["members"])
        version = entry["version"]
        transformation = (
            f"phase10_feature_set_v1|{feature_set_id}|{version}"
            f"|{_feature_definitions_digest(members)}"
        )
    return {
        "feature_set_id": feature_set_id,
        "feature_set_version": version,
        "feature_refs": members,
        "transformation": transformation,
    }


def _feature_names_for_ids(feature_ids: list[str]) -> list[str]:
    by_id = {
        f["feature_id"]: f["name"]
        for f in FEATURE_DEFINITIONS + ALL_PHASE10_DEFINITIONS
    }
    return [by_id[fid] for fid in feature_ids]


def _per_instrument_features(inst_bars: pl.DataFrame) -> pl.DataFrame:
    """Compute the 8 feature columns for one instrument's sorted session series.

    The returned frame has one row per session t (t >= 1) with features that
    reference only sessions <= t - 1 - exactly the information available at a
    decision stamped at the close of session t (strict boundary: bar t is not
    visible at close(t)). window_end_session is the previous session in this
    instrument's own series (calendar-correct, unlike a fixed one-day offset).
    Rows with an incomplete window are null and dropped by the documented NaN
    policy at matrix assembly time.
    """
    b = inst_bars.sort("trade_date").with_columns(
        (pl.col("close") * pl.col("volume")).alias("dollar_volume"),
        (pl.col("close") / pl.col("close").shift(1) - 1.0).alias("ret_1"),
    )
    out = b.with_columns(
        (pl.col("close").shift(1) / pl.col("close").shift(10) - 1.0).alias("ret_10"),
        (pl.col("close").shift(1) / pl.col("close").shift(20) - 1.0).alias("ret_20"),
        (pl.col("close").shift(1) / pl.col("close").shift(30) - 1.0).alias("ret_30"),
        (
            pl.col("close").rolling_mean(window_size=5).shift(1)
            / pl.col("close").rolling_mean(window_size=30).shift(1)
        ).alias("sma_ratio_5_30"),
        (
            pl.col("close").rolling_mean(window_size=15).shift(1)
            / pl.col("close").rolling_mean(window_size=40).shift(1)
        ).alias("sma_ratio_15_40"),
        (pl.col("ret_1").rolling_std(window_size=10).shift(1)).alias("vol_10"),
        (pl.col("ret_1").rolling_std(window_size=30).shift(1)).alias("vol_30"),
        (pl.col("dollar_volume").rolling_median(window_size=20).shift(1).log1p()).alias("log_dv_med_20"),
        pl.col("trade_date").shift(1).alias("window_end_session"),
    )
    return out.select("instrument_id", "trade_date", "window_end_session", *FEATURE_NAMES)


def _compute_phase10_features(inst_bars: pl.DataFrame) -> pl.DataFrame:
    """Compute the 15 Phase 10 feature columns for one instrument's sorted session series.

    Point-in-time discipline: all features reference only sessions strictly before
    the decision session (shifted by 1, consistent with FS-001 _per_instrument_features).
    The returned frame has one row per session t (t >= 1) with features that reference
    only sessions <= t - 1. window_end_session is the previous session in the series.
    """
    b = inst_bars.sort("trade_date").with_columns(
        (pl.col("close") * pl.col("volume")).alias("dollar_volume"),
        (pl.col("close") / pl.col("close").shift(1) - 1.0).alias("ret_1"),
    )
    out = b.with_columns(
        # Momentum (same convention as FEAT-001..003): ret_N = close(D-1)/close(D-N) - 1
        (pl.col("close").shift(1) / pl.col("close").shift(5) - 1.0).alias("ret_5"),
        (pl.col("close").shift(1) / pl.col("close").shift(60) - 1.0).alias("ret_60"),
        (pl.col("close").shift(1) / pl.col("close").shift(120) - 1.0).alias("ret_120"),
        # Trend: SMA ratios (same ratio form as FEAT-004/005) and 200-SMA distance
        (
            pl.col("close").rolling_mean(window_size=10).shift(1)
            / pl.col("close").rolling_mean(window_size=30).shift(1)
        ).alias("sma_ratio_10_30"),
        (
            pl.col("close").rolling_mean(window_size=20).shift(1)
            / pl.col("close").rolling_mean(window_size=50).shift(1)
        ).alias("sma_ratio_20_50"),
        (
            (pl.col("close").shift(1) - pl.col("close").rolling_mean(window_size=200).shift(1))
            / pl.col("close").rolling_mean(window_size=200).shift(1)
        ).alias("price_distance_200ma"),
        # Volatility: realized vol horizons + short/long ratio
        (pl.col("ret_1").rolling_std(window_size=60).shift(1)).alias("vol_60"),
        (pl.col("ret_1").rolling_std(window_size=90).shift(1)).alias("vol_90"),
        (
            pl.col("ret_1").rolling_std(window_size=10).shift(1)
            / pl.col("ret_1").rolling_std(window_size=30).shift(1)
        ).alias("vol_ratio_10_30"),
        # Volume/liquidity: dollar-volume medians + trailing z-score
        (pl.col("dollar_volume").rolling_median(window_size=10).shift(1).log1p()).alias("dv_med_10"),
        (pl.col("dollar_volume").rolling_median(window_size=30).shift(1).log1p()).alias("dv_med_30"),
        (
            (pl.col("dollar_volume").shift(1)
             - pl.col("dollar_volume").rolling_mean(window_size=20).shift(1))
            / pl.col("dollar_volume").rolling_std(window_size=20).shift(1)
        ).alias("vol_zscore_20"),
        # Range/price structure: last completed close's position in the trailing
        # high-low range (close is SHIFTED: the bar of D is never used) and the
        # trailing range normalized by the last completed close
        (
            (pl.col("close").shift(1) - pl.col("low").rolling_min(window_size=10).shift(1))
            / (pl.col("high").rolling_max(window_size=10).shift(1) - pl.col("low").rolling_min(window_size=10).shift(1))
        ).alias("high_low_10_pos"),
        (
            (pl.col("close").shift(1) - pl.col("low").rolling_min(window_size=30).shift(1))
            / (pl.col("high").rolling_max(window_size=30).shift(1) - pl.col("low").rolling_min(window_size=30).shift(1))
        ).alias("high_low_30_pos"),
        (
            (pl.col("high").rolling_max(window_size=20).shift(1) - pl.col("low").rolling_min(window_size=20).shift(1))
            / pl.col("close").shift(1)
        ).alias("normalized_range_20"),
        pl.col("trade_date").shift(1).alias("window_end_session"),
    )
    return out.select("instrument_id", "trade_date", "window_end_session", *FEATURE_NAMES_PHASE10)


def assert_features_finite(frame: pl.DataFrame, feature_names: list[str]) -> None:
    """Adversarial verification: every feature value must be finite.

    Raises AssertionError on any inf/nan value (a non-finite value would be a
    silent poisoning of the model input; the documented policy is to drop rows
    with incomplete windows at snapshot build time, never to impute)."""
    if frame.height == 0:
        return
    finite = pl.lit(True, dtype=pl.Boolean)
    for name in feature_names:
        finite = finite & pl.col(name).is_finite()
    bad = frame.filter(finite.not_())
    if bad.height:
        raise AssertionError(
            f"{bad.height} feature rows contain non-finite values in "
            f"{feature_names}"
        )


def build_feature_frame_phase10(bars: pl.DataFrame) -> pl.DataFrame:
    """Full per-session feature frame for every instrument using Phase 10 features (point-in-time).

    Columns: instrument_id, decision_session (Date), window_end_session (Date),
    <15 Phase 10 features>. Rows whose feature window is not yet complete are
    dropped here (null features); rows with non-finite values are dropped and
    counted (documented policy: no imputation anywhere).
    """
    parts = [_compute_phase10_features(g) for _, g in bars.group_by("instrument_id")]
    frame = pl.concat(parts) if parts else pl.DataFrame(
        schema={
            "instrument_id": pl.Utf8,
            "trade_date": pl.Date,
            "window_end_session": pl.Date,
            **{n: pl.Float64 for n in FEATURE_NAMES_PHASE10},
        }
    )
    frame = frame.rename({"trade_date": "decision_session"})
    frame = frame.drop_nulls(subset=FEATURE_NAMES_PHASE10)
    frame = frame.filter(
        pl.all_horizontal([pl.col(n).is_finite() for n in FEATURE_NAMES_PHASE10])
    )
    frame = frame.sort(["instrument_id", "decision_session"])
    return frame


def build_phase10_all_feature_frame(bars: pl.DataFrame) -> pl.DataFrame:
    """FS-003 superset frame: FS-001 (8) joined with FS-002 (15) = 23 features.

    Rows are the inner join on (instrument_id, decision_session) of the two
    snapshots, so every Phase 10 feature set's rows are a projection of this
    frame; window_end_session is taken from the FS-001 side (identical on the
    join key).
    """
    fs001 = build_feature_frame(bars)
    p10 = build_feature_frame_phase10(bars)
    joined = fs001.join(
        p10,
        on=["instrument_id", "decision_session"],
        how="inner",
        suffix="_p10",
    )
    joined = joined.with_columns(
        pl.coalesce(pl.col("window_end_session_p10"), pl.col("window_end_session")).alias("window_end_session")
    ).drop("window_end_session_p10")
    return joined.sort(["instrument_id", "decision_session"])


def build_phase10_feature_set_snapshot(
    feature_set_id: str,
    all_frame: pl.DataFrame,
    data_refs: list[str] | None = None,
) -> FeatureSnapshot:
    """Build the immutable FeatureSnapshot of a Phase 10 feature set.

    `all_frame` is the FS-003 superset frame (build_phase10_all_feature_frame);
    every Phase 10 set is a column projection of it, so all sets share the same
    point-in-time-verified rows. The snapshot's `transformation` field binds the
    feature-set id, version and the definitions digest into the content digest.
    """
    if feature_set_id not in PHASE10_FEATURE_SETS:
        raise ValueError(
            f"unknown Phase 10 feature set {feature_set_id!r}; "
            f"registered: {sorted(PHASE10_FEATURE_SETS)}"
        )
    identity = phase10_set_identity(feature_set_id)
    feature_names = _feature_names_for_ids(identity["feature_refs"])
    if set(feature_names).issubset(FEATURE_NAMES):
        raise ValueError(
            f"feature set {feature_set_id} contains only FS-001 features; "
            "use the frozen Phase 9 builder instead"
        )
    frame = all_frame.select(
        "instrument_id", "decision_session", "window_end_session", *feature_names
    )
    frame = attach_decision_times(frame)
    assert_features_point_in_time(frame)
    assert_features_finite(frame, feature_names)
    return FeatureSnapshot(
        feature_set_id=feature_set_id,
        feature_set_version=identity["feature_set_version"],
        feature_refs=identity["feature_refs"],
        data_refs=data_refs or [],
        records=frame,
        transformation=identity["transformation"],
        limitations=[
            "rows with an incomplete feature window are dropped (warm-up policy, "
            "identical to FS-001); non-finite values are dropped, never imputed"
        ],
    )


def build_feature_snapshot_phase10(
    bars: pl.DataFrame,
    data_refs: list[str] | None = None,
) -> FeatureSnapshot:
    """Build the FS-002 v1 FeatureSnapshot over a bars frame using Phase 10 features.

    decision_time is attached per session; every row is point-in-time
    verified before snapshotting.
    """
    frame = build_feature_frame_phase10(bars)
    frame = attach_decision_times(frame)
    assert_features_point_in_time(frame)
    assert_features_finite(frame, FEATURE_NAMES_PHASE10)
    identity = phase10_set_identity("FS-002")
    return FeatureSnapshot(
        feature_set_id="FS-002",
        feature_set_version="v1",
        feature_refs=identity["feature_refs"],
        data_refs=data_refs or [],
        records=frame,
        transformation=identity["transformation"],
        limitations=[
            "rows with an incomplete feature window are dropped (warm-up policy, "
            "identical to FS-001); non-finite values are dropped, never imputed"
        ],
    )


def build_feature_frame(bars: pl.DataFrame) -> pl.DataFrame:
    """Full per-session feature frame for every instrument (point-in-time).

    Columns: instrument_id, decision_session (Date), window_end_session (Date),
    decision_time (Datetime), <8 features>. Rows whose feature window is not
    yet complete are dropped here (null features); the NaN policy for
    later-stage assembly is handled at matrix build time.
    """
    parts = [_per_instrument_features(g) for _, g in bars.group_by("instrument_id")]
    frame = pl.concat(parts) if parts else pl.DataFrame(
        schema={
            "instrument_id": pl.Utf8,
            "trade_date": pl.Date,
            "window_end_session": pl.Date,
            **{n: pl.Float64 for n in FEATURE_NAMES},
        }
    )
    frame = frame.rename({"trade_date": "decision_session"})
    frame = frame.drop_nulls(subset=FEATURE_NAMES)
    frame = frame.sort(["instrument_id", "decision_session"])
    return frame


def close_utc(session: Any) -> Any:
    """Session close instant (16:00 America/New_York) via the temporal engine."""
    from orbit.temporal.times import session_close_utc

    return session_close_utc(session)


def attach_decision_times(feature_frame: pl.DataFrame) -> pl.DataFrame:
    """Add decision_time = close_utc(decision_session) to a feature frame."""
    sessions = feature_frame.select("decision_session").unique().sort("decision_session")
    dt = pl.DataFrame(
        [
            {"decision_session": r["decision_session"], "decision_time": close_utc(r["decision_session"])}
            for r in sessions.iter_rows(named=True)
        ],
        schema={"decision_session": pl.Date, "decision_time": pl.Datetime("us", None)},
    )
    return feature_frame.join(dt, on="decision_session", how="left")


def assert_features_point_in_time(frame: pl.DataFrame) -> None:
    """Adversarial verification: every feature row references only sessions
    strictly before its decision session (window_end_session < decision_session).
    Raises AssertionError on any violation."""
    bad = frame.filter(pl.col("window_end_session") >= pl.col("decision_session"))
    if bad.height:
        raise AssertionError(
            f"point-in-time violation: {bad.height} feature rows reference "
            "their own or a future session"
        )


class FeatureSnapshot:
    """One deterministic, reproducible batch of Phase 9 features."""

    def __init__(
        self,
        *,
        feature_set_id: str,
        feature_set_version: str,
        feature_refs: list[str],
        data_refs: list[str],
        records: pl.DataFrame,
        transformation: str = FEATURE_TRANSFORMATION,
        limitations: list[str] | None = None,
    ):
        self.feature_set_id = feature_set_id
        self.feature_set_version = feature_set_version
        self.feature_refs = sorted(feature_refs)
        self.data_refs = sorted(data_refs)
        self.records = records
        self.transformation = transformation
        self.limitations = list(limitations or [])
        self.created_at = datetime.now()
        self.content_digest = self._compute_digest()

    def _canonical_json(self) -> str:
        parts: dict[str, Any] = {
            "feature_set_id": self.feature_set_id,
            "feature_set_version": self.feature_set_version,
            "transformation": self.transformation,
            "feature_refs": self.feature_refs,
            "data_refs": self.data_refs,
            "limitations": sorted(self.limitations),
        }
        if self.records.height:
            parts["records"] = json.loads(
                self.records.sort(self.records.columns).write_json()
            )
        return json.dumps(parts, sort_keys=True, default=str, separators=(",", ":"))

    def _compute_digest(self) -> str:
        return hashlib.sha256(self._canonical_json().encode("utf-8")).hexdigest()

    def equals(self, other: "FeatureSnapshot") -> bool:
        return (
            self.feature_set_id == other.feature_set_id
            and self.feature_set_version == other.feature_set_version
            and self.feature_refs == other.feature_refs
            and self.data_refs == other.data_refs
            and self.content_digest == other.content_digest
        )

    def provenance(self) -> dict[str, Any]:
        return {
            "feature_set_id": self.feature_set_id,
            "feature_set_version": self.feature_set_version,
            "transformation": self.transformation,
            "feature_refs": list(self.feature_refs),
            "data_refs": list(self.data_refs),
            "content_digest": self.content_digest,
            "limitations": list(self.limitations),
            "row_count": self.records.height,
        }


def build_feature_snapshot(
    bars: pl.DataFrame,
    data_refs: list[str] | None = None,
) -> FeatureSnapshot:
    """Build the FS-001 v1 FeatureSnapshot over a bars frame.

    decision_time is attached per session; every row is point-in-time
    verified before snapshotting.
    """
    frame = build_feature_frame(bars)
    frame = attach_decision_times(frame)
    assert_features_point_in_time(frame)
    return FeatureSnapshot(
        feature_set_id=FEATURE_SET_ID,
        feature_set_version=FEATURE_SET_VERSION,
        feature_refs=[f["feature_id"] for f in FEATURE_DEFINITIONS],
        data_refs=data_refs or [],
        records=frame,
    )


__all__ = [
    "FEATURE_SET_ID",
    "FEATURE_SET_VERSION",
    "FEATURE_DEFINITIONS",
    "FEATURE_NAMES",
    "FEATURE_ID_BY_NAME",
    "MAX_FEATURE_WINDOW",
    "FeatureSnapshot",
    "build_feature_snapshot",
    "build_feature_frame",
    "attach_decision_times",
    "assert_features_point_in_time",
    "close_utc",
    # Phase 10 exports
    "PHASE10_FEATURE_VERSION",
    "MOMENTUM_FEATURE_DEFINITIONS",
    "TREND_FEATURE_DEFINITIONS",
    "VOLATILITY_FEATURE_DEFINITIONS",
    "VOLUME_FEATURE_DEFINITIONS",
    "RANGE_FEATURE_DEFINITIONS",
    "ALL_PHASE10_DEFINITIONS",
    "FEATURE_NAMES_PHASE10",
    "FEATURE_ID_BY_NAME_PHASE10",
    "MAX_FEATURE_WINDOW_PHASE10",
    "PHASE10_FAMILIES",
    "PHASE10_FAMILY_DEFINITIONS",
    "FEATURE_FAMILY_BY_ID_PHASE10",
    "PHASE10_FEATURE_SETS",
    "PHASE10_FEATURE_SET_ORDER",
    "PHASE10_BASE_FEATURE_SET",
    "phase10_set_identity",
    "_feature_definitions_digest",
    "assert_features_finite",
    "build_feature_frame_phase10",
    "build_phase10_all_feature_frame",
    "build_phase10_feature_set_snapshot",
    "build_feature_snapshot_phase10",
]