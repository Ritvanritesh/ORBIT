"""Shared Phase 7 test helpers: synthetic bars, canonical signals and a
default run harness. Everything is deterministic and hermetic."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import polars as pl

from orbit.backtest import Backtester, BacktestConfig
from orbit.temporal.times import session_close_utc

DEFAULT_UNIVERSE = ["INS-000001", "INS-000002"]


def make_bars(
    dates: list[date],
    instruments: list[str] | None = None,
    base_prices: dict[str, float] | None = None,
    volume: float = 1_000_000.0,
    drift: float = 0.0,
) -> pl.DataFrame:
    """Synthetic canonical bars: close drifts linearly by `drift` per day,
    open = previous close (flat on day 0), OHLC sane."""
    instruments = instruments or DEFAULT_UNIVERSE
    base_prices = base_prices or {
        "INS-000001": 100.0,
        "INS-000002": 50.0,
    }
    prev_close: dict[str, float] = {}
    rows: list[dict[str, Any]] = []
    for i, d in enumerate(dates):
        for ins in sorted(instruments):
            base = base_prices[ins]
            close = base * (1.0 + drift * i)
            open_ = prev_close.get(ins, base)
            prev_close[ins] = close
            rows.append(
                {
                    "instrument_id": ins,
                    "trade_date": d,
                    "open": open_,
                    "high": max(open_, close) * 1.01,
                    "low": min(open_, close) * 0.99,
                    "close": close,
                    "volume": volume,
                }
            )
    return pl.DataFrame(rows)


def weekdays(start: date, n: int) -> list[date]:
    """The first `n` weekdays at or after `start` (calendar sessions; the
    clock accepts any distinct dates, weekdays keep the fixture realistic)."""
    out: list[date] = []
    d = start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def signals(
    instrument_id: str,
    sessions: list[date],
    *,
    direction: str = "long",
    target: float = 1000.0,
    metric: float | None = 0.05,
    strategy_ref: str | None = None,
    start_index: int = 0,
) -> list[dict[str, Any]]:
    """Canonical signal rows: decision_time exactly at the session close."""
    return [
        {
            "signal_id": f"SIG-{i + 1:06d}",
            "instrument_id": instrument_id,
            "signal_session": s,
            "decision_time": session_close_utc(s),
            "direction": direction,
            "target": target,
            "signal_metric": metric,
            "strategy_ref": strategy_ref,
        }
        for i, s in enumerate(sessions, start=start_index)
    ]


def run_default(
    bars: pl.DataFrame,
    signal_rows: list[dict[str, Any]],
    *,
    config: BacktestConfig | None = None,
    universe: list[str] | None = None,
    events: pl.DataFrame | None = None,
    code_hash: str = "c" * 64,
) -> Backtester:
    """A fully-wired backtester (manifest-complete) ready to run."""
    bt = Backtester(
        config=config or BacktestConfig(),
        universe=universe or sorted(bars["instrument_id"].unique().to_list()),
        dataset_snapshot_ids=["DS-000001"],
        code_hash=code_hash,
        experiment_id="EXP-00001",
        hypothesis_id="H-001",
        feature_refs=[{"feature_id": "FEAT-001", "feature_version": "v1"}],
        model={"family": "linear", "hyperparameters": {}},
        label_id="LAB-001",
        label_version="v1",
        cost_model_id="CM-001",
    )
    return bt


def run(
    bars: pl.DataFrame,
    signal_rows: list[dict[str, Any]],
    **kwargs,
):
    return run_default(bars, signal_rows, **kwargs).run(
        bars, signal_rows, events_artifact=kwargs.get("events")
    )