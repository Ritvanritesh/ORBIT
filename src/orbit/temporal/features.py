"""Feature-time rules: the temporal infrastructure future features MUST obey.

A feature is a value computed from records. Its `feature_time` is the
decision time it is attached to, and it may only use records available
strictly before that time. Phase 4 does NOT build the feature library; it
builds the enforceable rules:

  - completed_bars(): the N most recent COMPLETED bars strictly before a
    decision time (a bar for session D completes at the session close, so
    a decision before the close never sees that day's bar).
  - assert_no_future_refs(): scans any feature frame for timestamps after
    its feature_time and reports every violation. This is the detector
    that makes a future-row reference a hard test failure.

A feature frame is a polars DataFrame with at least:
    feature_time    datetime  the decision time the feature belongs to
    <time_col>      datetime  the timestamp of the record the feature used
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import polars as pl

from orbit.temporal.times import normalize_instant, session_close_utc


@dataclass(frozen=True)
class FutureRefViolation:
    """A feature row referenced information not yet available."""

    feature_time: datetime
    ref_time: datetime
    record_id: str | None = None
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "feature_time": self.feature_time.isoformat(),
            "ref_time": self.ref_time.isoformat(),
            "record_id": self.record_id,
            "detail": self.detail,
        }


def completed_bars(
    bars: pl.DataFrame,
    as_of: datetime | date | str,
    window: int,
    instrument_id: str | None = None,
) -> pl.DataFrame:
    """The most recent `window` COMPLETED bars strictly before as_of.

    A bar for session D is completed at the session close (16:00
    America/New_York on D). A decision at D 15:59 never sees bar D; a
    decision at D+1 00:00 sees bar D but never bar D+1. This is the ONLY
    sanctioned way for features to consume price history.

    Bars are returned in ascending session order.
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    t = normalize_instant(as_of)
    if t is None:
        raise ValueError("as_of_time is required")

    df = bars
    if instrument_id is not None:
        df = df.filter(pl.col("instrument_id") == instrument_id)
    if "trade_date" not in df.columns:
        raise ValueError("bars frame requires a trade_date column")
    close_col = df["trade_date"].map_elements(
        session_close_utc, return_dtype=pl.Datetime("us")
    )
    df = df.with_columns(close_col.alias("_session_close"))
    completed = df.filter(pl.col("_session_close") < t).sort("trade_date")
    if completed.height <= window:
        return completed.drop("_session_close")
    return completed.tail(window).drop("_session_close")


def assert_no_future_refs(
    feature_frame: pl.DataFrame,
    as_of: datetime | date | str,
    time_col: str,
    id_col: str | None = None,
) -> list[FutureRefViolation]:
    """Report every feature row whose referenced time is at/after as_of.

    The rule is strict: `ref < as_of` is required, so a feature that peeks
    at the same instant (e.g. today's close used at the close) is caught.

    Column semantics:
      - Date column: the reference is a SESSION date (a daily bar); its
        availability instant is the session close (16:00 America/New_York),
        so a same-day reference is always a violation for any as_of on or
        before that session's close.
      - Datetime column: the reference is an instant and must be strictly
        before as_of. Callers referencing bar ts_utc must pass the session
        close instants, not the provider's session-open timestamps.

    Returns an empty list when the frame is clean.
    """
    t = normalize_instant(as_of)
    if t is None:
        raise ValueError("as_of_time is required")
    if feature_frame.height == 0:
        return []
    if time_col not in feature_frame.columns:
        raise ValueError(f"feature frame has no column {time_col!r}")
    refs = feature_frame[time_col].to_list()
    ids = feature_frame[id_col].to_list() if id_col and id_col in feature_frame.columns else [None] * len(refs)
    is_date_col = feature_frame[time_col].dtype == pl.Date
    violations: list[FutureRefViolation] = []
    for ref, rid in zip(refs, ids):
        if is_date_col:
            if ref is None:
                continue
            avail = session_close_utc(ref)
        else:
            avail = normalize_instant(ref)
            if avail is None:
                continue
        if avail >= t:
            violations.append(
                FutureRefViolation(
                    feature_time=t,
                    ref_time=avail,
                    record_id=rid,
                    detail=(
                        f"referenced bar {ref.isoformat() if is_date_col else ref} "
                        f"is available at {avail.isoformat()}, not strictly "
                        f"before feature_time {t.isoformat()}"
                    ),
                )
            )
    return violations


def assert_feature_window_clean(
    bars: pl.DataFrame,
    as_of: datetime | date | str,
    window: int,
    feature_time_col: str = "feature_time",
    id_col: str = "record_id",
) -> list[FutureRefViolation]:
    """End-to-end check: the completed-bars window of `as_of` must not
    contain any bar at/after as_of. Used by the future-price leak tests."""
    w = completed_bars(bars, as_of, window)
    return assert_no_future_refs(w, as_of, time_col="trade_date", id_col=id_col)