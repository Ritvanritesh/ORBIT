"""Phase 7 multi-symbol tests: position and cash accounting stay isolated
per instrument, cross-signal ordering is deterministic, and the universe
spread drives no cross-contamination."""

from __future__ import annotations

from datetime import date

import pytest

from orbit.backtest import SizingPolicy

from phase7_testutils import make_bars, run_default, signals, weekdays

DATES = weekdays(date(2024, 1, 2), 10)


def test_two_instruments_trade_independently():
    bars = make_bars(DATES, base_prices={"INS-000001": 100.0, "INS-000002": 50.0})
    signal_rows = signals("INS-000001", DATES[:2], target=1000) + signals(
        "INS-000002", DATES[:2], target=500, start_index=2
    )
    res = run_default(bars, []).run(bars, signal_rows)
    res.assert_accounting_clean()
    assert res.final_position("INS-000001") == 1000.0
    assert res.final_position("INS-000002") == 500.0
    fills = res.fills
    assert len(fills) == 2
    by_instrument = {f.instrument_id: f for f in fills}
    assert by_instrument["INS-000001"].filled_quantity == 1000.0
    assert by_instrument["INS-000002"].filled_quantity == 500.0
    assert by_instrument["INS-000001"].price == 100.0
    assert by_instrument["INS-000002"].price == 50.0
    summary = res.summary()
    # no cross-contamination: cash debits sum exactly, and the held
    # positions are marked to market at the flat closes
    assert summary["final_equity"] == pytest.approx(1_000_000.0, rel=1e-9)


def test_same_session_signals_fill_in_deterministic_order():
    bars = make_bars(DATES)
    signal_rows = signals("INS-000002", DATES[:1], target=500, start_index=0) + signals(
        "INS-000001", DATES[:1], target=1000, start_index=1
    )
    res = run_default(bars, []).run(bars, signal_rows)
    fills = res.fills
    # order ids are assigned in signal emission order; fills are processed
    # in order-id order - deterministic, documented, instrument-independent
    assert [f.signal_id for f in fills] == ["SIG-000001", "SIG-000002"]
    assert [f.order_id for f in fills] == ["ORD-000000", "ORD-000001"]
    assert [f.instrument_id for f in fills] == ["INS-000002", "INS-000001"]


def test_round_trip_buy_sell_cycle_per_instrument():
    bars = make_bars(DATES, drift=0.001)
    signal_rows = (
        signals("INS-000001", DATES[:2], target=1000)
        + signals("INS-000001", DATES[3:4], direction="flat", start_index=2)
        + signals("INS-000002", DATES[:2], target=500, start_index=4)
        + signals("INS-000002", DATES[4:5], direction="flat", start_index=6)
    )
    res = run_default(bars, []).run(bars, signal_rows)
    res.assert_accounting_clean()
    assert res.final_position("INS-000001") == 0.0
    assert res.final_position("INS-000002") == 0.0
    # realized P&L of each instrument follows its own path
    realized_1 = res.summary()["final_equity"] - 1_000_000.0
    expected_1 = 1000.0 * (100.0 * 1.003 - 100.0) + 500.0 * (
        50.0 * 1.004 - 50.0
    )
    assert realized_1 == pytest.approx(expected_1, rel=1e-9)


def test_weight_sizing_uses_last_valuation_equity():
    bars = make_bars(DATES)
    config = __import__(
        "orbit.backtest.config", fromlist=["BacktestConfig"]
    ).BacktestConfig(sizing=SizingPolicy.WEIGHT)
    signal_rows = signals("INS-000001", DATES[:1], target=0.5)
    res = run_default(bars, [], config=config).run(bars, signal_rows)
    fill = res.fills[0]
    # 50% of 1,000,000 at the day-1 open price of 100 -> 5,000 shares
    assert fill.filled_quantity == 5000.0
    res.assert_accounting_clean()


def test_quantity_sizing_is_whole_shares():
    bars = make_bars(DATES)
    signal_rows = signals("INS-000001", DATES[:1], target=123.7)
    res = run_default(bars, []).run(bars, signal_rows)
    assert res.fills[0].filled_quantity == 123.0  # floored, never rounded up


def test_sizing_target_above_cash_is_rejected_not_borrowed():
    bars = make_bars(DATES)
    signal_rows = signals("INS-000001", DATES[:1], target=50_000.0)
    res = run_default(bars, []).run(bars, signal_rows)
    assert res.summary()["n_fills"] == 0
    assert any(r.reason.value == "insufficient_cash" for r in res.rejections)