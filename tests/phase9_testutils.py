"""Shared Phase 9 test helpers: small hermetic synthetic data.

Builds a tiny multi-instrument universe with canonical columns and the
Phase 9 feature/label snapshots fast enough for unit tests (tight test
windows instead of the full 2010-2026 protocol).
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import polars as pl

from orbit.ml.dataset import assemble_datasets
from orbit.ml.features import attach_decision_times, build_feature_frame
from orbit.ml.labels import build_phase9_label_snapshot
from orbit.ml.splits import assert_split_integrity

TEST_WINDOWS = {
    "train_start": date(2010, 1, 4),
    "train_end": date(2012, 6, 29),
    "val_start": date(2012, 7, 2),
    "val_end": date(2014, 6, 30),
    "test_start": date(2014, 7, 1),
    "test_end": date(2016, 6, 30),
    "embargo_days": 0,
    "purge_days": 0,
    "protocol": "fixed_chronological_test_v1",
}

TEST_INSTRUMENTS = [f"INS-{i:06d}" for i in (101, 102, 103, 104)]
TEST_BASE_PRICES = {f"INS-{i:06d}": float(p) for i, p in zip((101, 102, 103, 104), (100.0, 50.0, 200.0, 75.0))}


def weekdays(start: date, n: int) -> list[date]:
    out: list[date] = []
    d = start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def make_canonical_bars(
    instruments: list[str] | None = None,
    sessions: list[date] | None = None,
    seed: int = 42,
    drift: float = 0.0003,
) -> pl.DataFrame:
    """Deterministic canonical bars with realistic OHLCV + ts_utc + adjclose."""
    import random

    rng = random.Random(seed)
    instruments = instruments or TEST_INSTRUMENTS
    base = {ins: TEST_BASE_PRICES.get(ins, 100.0) for ins in instruments}
    sessions = sessions or weekdays(date(2010, 1, 4), 1700)
    rows: list[dict[str, Any]] = []
    prev_close: dict[str, float] = {}
    for i, d in enumerate(sessions):
        for ins in sorted(instruments):
            prev = prev_close.get(ins, base[ins])
            shock = 1.0 + rng.gauss(0.0, 0.01) + drift
            close = max(prev * shock, 1.0)
            open_ = prev
            high = max(open_, close) * (1.0 + abs(rng.gauss(0, 0.003)))
            low = min(open_, close) * (1.0 - abs(rng.gauss(0, 0.003)))
            volume = base[ins] * 10_000.0 * rng.uniform(0.8, 1.2)
            ts = session_close_utc(d)
            rows.append(
                {
                    "instrument_id": ins,
                    "symbol": f"SYM{ins[-2:]}",
                    "trade_date": d,
                    "ts_utc": ts,
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume,
                    "adjclose": close,
                    "adjustment": "split_adjusted",
                    "provider": "yahoo_chart_api",
                    "source_uri": "https://test",
                    "snapshot_id": "DS-000001",
                }
            )
            prev_close[ins] = close
    return pl.DataFrame(rows)


def make_events(instruments: list[str] | None = None) -> pl.DataFrame:
    """A minimal corporate-actions artifact (splits + dividends) so the
    SIMPLE_TOTAL_RETURN label has dividend data (never silently price-only)."""
    instruments = instruments or TEST_INSTRUMENTS
    rows = []
    for ins in instruments:
        rows.append(
            {
                "instrument_id": ins,
                "symbol": f"SYM{ins[-2:]}",
                "kind": "dividends",
                "ts": session_close_utc(date(2013, 3, 15)),
                "ratio": 0.25,
                "numerator": None,
                "denominator": None,
                "provider": "yahoo_chart_api",
                "snapshot_id": "DS-000001",
            }
        )
    return pl.DataFrame(rows)


def build_snapshots(bars: pl.DataFrame, events: pl.DataFrame | None = None):
    """Feature + label snapshots over synthetic bars (fast, hermetic)."""
    feature_frame = build_feature_frame(bars)
    feature_frame = attach_decision_times(feature_frame)
    from orbit.ml.features import FeatureSnapshot

    fs = FeatureSnapshot(
        feature_set_id="FS-001",
        feature_set_version="v1",
        feature_refs=["FEAT-001", "FEAT-002", "FEAT-003", "FEAT-004", "FEAT-005", "FEAT-006", "FEAT-007", "FEAT-008"],
        data_refs=["DS-000001"],
        records=feature_frame,
    )
    decisions = feature_frame.select("instrument_id", "decision_time")
    ls = build_phase9_label_snapshot(bars, events, [], decisions, data_refs=["DS-000001"])
    return fs, ls


def build_test_datasets(bars: pl.DataFrame, events: pl.DataFrame | None = None):
    fs, ls = build_snapshots(bars, events)
    datasets = assemble_datasets(fs, ls, windows=TEST_WINDOWS)
    return fs, ls, datasets


def session_close_utc(d: date):
    from orbit.temporal.times import session_close_utc as _scu

    return _scu(d)


__all__ = [
    "TEST_WINDOWS",
    "TEST_INSTRUMENTS",
    "TEST_BASE_PRICES",
    "weekdays",
    "make_canonical_bars",
    "make_events",
    "build_snapshots",
    "build_test_datasets",
    "session_close_utc",
]