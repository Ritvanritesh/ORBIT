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
]