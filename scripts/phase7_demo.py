#!/usr/bin/env python
"""
Phase 7 demo script: deterministic synthetic run demonstrating
event stream, replay equality, and experiment lifecycle.
"""
from __future__ import annotations

from datetime import date

import polars as pl

from orbit.backtest.clock import MarketEventClock, session_close_utc
from orbit.backtest.config import BacktestConfig, CostConfig, SizingPolicy, ExecutionConfig
from orbit.backtest.backtester import Backtester


def make_bars(dates: list[date], base_price: float = 100.0) -> pl.DataFrame:
    """Create canonical bars for the given dates."""
    return pl.DataFrame({
        "instrument_id": ["INS-000001"] * len(dates),
        "trade_date": dates,
        "open": [base_price] * len(dates),
        "high": [base_price * 1.05] * len(dates),
        "low": [base_price * 0.95] * len(dates),
        "close": [base_price] * len(dates),
        "volume": [1000] * len(dates),
    })


def signals(instrument_id: str, sessions: list[date], target: float = 100.0, direction: str = "long", signal_id: str = None) -> list[dict]:
    """Create signal rows for the given sessions."""
    rows = []
    for i, session in enumerate(sessions):
        r = {
            "instrument_id": instrument_id,
            "signal_session": session,
            "direction": direction,
            "target": target,
            "signal_id": signal_id or f"SIG-{i + 1:06d}",
        }
        rows.append(r)
    return rows


def main():
    print("=== ORBIT Phase 7 Demo ===\n")

    # Create deterministic bar data
    dates = [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3),
             date(2024, 1, 4), date(2024, 1, 5)]
    bars = make_bars(dates, base_price=100.0)

    print(f"Bars: {bars.height} rows, instruments: {bars['instrument_id'].unique().to_list()}")

    # 1. Initialize clock
    clock = MarketEventClock(bars)
    print(f"\n1. Clock sessions: {clock.sessions()}")
    print(f"   Instruments: {clock.instruments()}")
    print(f"   Volume basis: {clock.volume_basis}")

    # 2. Create signals
    sigs = signals("INS-000001", dates[:3], target=100.0, direction="long")
    print(f"\n2. Signals: {len(sigs)} signal(s)")
    for s in sigs:
        print(f"   {s['signal_id']}: session={s['signal_session']}, "
              f"direction={s['direction']}, target={s['target']}")

    # 3. Run backtest
    config = BacktestConfig(
        universe=["INS-000001"],
        initial_cash=10_000.0,
        costs=CostConfig(spread_bps=10, fees_bps=5, slippage_bps=5),
        sizing=SizingPolicy.QUANTITY,
        execution=ExecutionConfig(execution_delay=1),
    )

    backtester = Backtester(
        config=config,
        universe=["INS-000001"],
        dataset_snapshot_ids=["DS-000001"],
        code_hash="h" * 64,
    )
    result = backtester.run(bars, sigs)

    print(f"\n3. Backtest results:")
    print(f"   Final equity: {result._last_equity:.2f}")
    print(f"   Final position: {result.final_position('INS-000001')}")
    print(f"   Fills: {len(result.fills)}")
    print(f"   Rejections: {len(result.rejections)}")
    print(f"   Accounting clean: {result.assert_accounting_clean()}")
    print(f"   Summary: {result.summary()}")

    # 4. Replay equality
    print(f"\n4. Replay equality test:")
    result2 = backtester.run(bars, sigs)
    print(f"   Rerun equity matches: {result._last_equity == result2._last_equity}")
    print(f"   Fills count matches: {len(result.fills) == len(result2.fills)}")

    # 5. Note on experiment lifecycle
    print(f"\n5. Experiment lifecycle note:")
    print("   (Backtester does not have _to_experiment in Phase 7;")
    print("    experiment lifecycle is a Phase 6 feature)")

    print(f"\n=== Demo Complete ===")


if __name__ == "__main__":
    main()