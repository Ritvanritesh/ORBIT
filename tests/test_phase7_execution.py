"""Phase 7 execution simulator tests: the pure order -> fill/rejection
transformer under every documented cost, cap and constraint."""

from __future__ import annotations

from datetime import date

import pytest

from orbit.backtest import BacktestConfig, CostConfig, ExecutionPrice, FailureKind
from orbit.backtest.config import ExecutionConfig
from orbit.backtest.clock import BarRecord
from orbit.backtest.events import FillEvent, OrderIntent, OrderSide, SignalDirection
from orbit.backtest.execution import ExecutionSimulator
from orbit.backtest.orders import OrderGenerator

BAR = BarRecord(
    instrument_id="INS-000001",
    trade_date=date(2024, 1, 3),
    open=100.0,
    high=102.0,
    low=99.0,
    close=101.0,
    volume=1_000_000.0,
    liquidity_volume=1_000_000.0,
    price_basis="split_continuous_stored",
    volume_basis="as_published",
)


def _order(side=OrderSide.BUY, quantity=100.0, instrument="INS-000001"):
    return OrderIntent(
        run_id="BT-test-1234",
        event_type="order",
        sequence=-1,
        session=date(2024, 1, 2),
        timestamp=None,
        source="test",
        order_id="ORD-000001",
        signal_id="SIG-000001",
        instrument_id=instrument,
        side=side,
        quantity=quantity,
    )


def _sim(**overrides):
    config = BacktestConfig(
        costs=CostConfig(
            spread_bps=overrides.pop("spread_bps", 0.0),
            fees_bps=overrides.pop("fees_bps", 0.0),
            slippage_bps=overrides.pop("slippage_bps", 0.0),
            fixed_fee_per_order=overrides.pop("fixed_fee_per_order", 0.0),
            fee_minimum=overrides.pop("fee_minimum", 0.0),
        ),
        execution=ExecutionConfig(
            execution_price=overrides.pop("execution_price", ExecutionPrice.OPEN),
            participation_fraction=overrides.pop("participation_fraction", 0.05),
            partial_fills=overrides.pop("partial_fills", True),
            max_order_quantity=overrides.pop("max_order_quantity", None),
        ),
        **overrides,
    )
    return ExecutionSimulator(config)


def test_zero_cost_fill_at_reference_price():
    fill = _sim().execute(_order(), BAR, cash=1e9, position_quantity=0.0)
    assert fill.event_type == "fill"
    assert fill.price == 100.0 == fill.reference_price
    assert fill.filled_quantity == 100.0
    assert fill.unfilled_quantity == 0.0
    assert fill.spread_cost == 0.0 and fill.slippage_cost == 0.0 and fill.fee == 0.0


def test_buy_pays_spread_plus_slippage_sell_receives():
    sim = _sim(spread_bps=2, slippage_bps=2)
    buy = sim.execute(_order(OrderSide.BUY), BAR, cash=1e9, position_quantity=0.0)
    assert buy.price == pytest.approx(100.0 * (1 + 4e-4), rel=1e-9)
    assert buy.spread_cost == pytest.approx(100.0 * 100 * 2e-4, rel=1e-9)
    assert buy.slippage_cost == pytest.approx(100.0 * 100 * 2e-4, rel=1e-9)
    sell = sim.execute(
        _order(OrderSide.SELL), BAR, cash=1e9, position_quantity=1000.0
    )
    assert sell.price == pytest.approx(100.0 * (1 - 4e-4), rel=1e-9)


def test_close_execution_uses_the_close_field():
    fill = _sim(execution_price=ExecutionPrice.CLOSE).execute(
        _order(), BAR, cash=1e9, position_quantity=0.0
    )
    assert fill.reference_price == BAR.close == 101.0


def test_fee_is_notional_bps_plus_fixed_floored_at_minimum():
    fill = _sim(fees_bps=10, fixed_fee_per_order=0.5).execute(
        _order(quantity=1000.0), BAR, cash=1e9, position_quantity=0.0
    )
    assert fill.fee == pytest.approx(1000 * 100 * 10e-4 + 0.5, rel=1e-9)
    floored = _sim(fees_bps=0, fee_minimum=2.0).execute(
        _order(quantity=1000.0), BAR, cash=1e9, position_quantity=0.0
    )
    assert floored.fee == 2.0


def test_liquidity_cap_partial_fill_records_unfilled_remainder():
    fill = _sim().execute(_order(quantity=1_000_000.0), BAR, cash=1e9, position_quantity=0.0)
    assert fill.filled_quantity == 50_000.0  # 5% of 1M
    assert fill.unfilled_quantity == 950_000.0
    assert fill.unfilled_reason == "liquidity_cap"


def test_liquidity_cap_respects_absolute_order_cap():
    fill = _sim(max_order_quantity=100.0).execute(
        _order(quantity=10_000.0), BAR, cash=1e9, position_quantity=0.0
    )
    assert fill.filled_quantity == 100.0
    assert fill.unfilled_quantity == 9900.0


def test_partial_fills_disabled_rejects_over_cap():
    rej = _sim(partial_fills=False).execute(
        _order(quantity=1_000_000.0), BAR, cash=1e9, position_quantity=0.0
    )
    assert rej.event_type == "rejection"
    assert rej.reason == FailureKind.LIQUIDITY_REJECTION


def test_cap_floored_to_zero_rejects():
    # tiny participation against a small volume: the cap floors to zero
    low_vol = BarRecord(**{**BAR.__dict__, "volume": 10.0, "liquidity_volume": 10.0})
    rej = _sim().execute(_order(quantity=100.0), low_vol, cash=1e9, position_quantity=0.0)
    assert rej.reason == FailureKind.LIQUIDITY_REJECTION
    assert "floors to zero" in rej.detail


def test_insufficient_cash_rejects_whole_order():
    rej = _sim().execute(_order(quantity=1000.0), BAR, cash=500.0, position_quantity=0.0)
    assert rej.reason == FailureKind.INSUFFICIENT_CASH


def test_sell_beyond_position_rejects():
    rej = _sim().execute(_order(OrderSide.SELL, quantity=500.0), BAR, cash=1e9, position_quantity=100.0)
    assert rej.reason == FailureKind.POSITION_CONSTRAINT
    ok = _sim().execute(_order(OrderSide.SELL, quantity=100.0), BAR, cash=1e9, position_quantity=100.0)
    assert ok.event_type == "fill"


def test_missing_bar_is_no_execution_bar():
    rej = _sim().execute(_order(), None, cash=1e9, position_quantity=0.0)
    assert rej.reason == FailureKind.NO_EXECUTION_BAR


def test_defective_reference_price_is_data_error():
    bad = BarRecord(**{**BAR.__dict__, "open": None})
    rej = _sim().execute(_order(), bad, cash=1e9, position_quantity=0.0)
    assert rej.reason == FailureKind.DATA_ERROR


def test_non_positive_quantity_is_invalid_order():
    rej = _sim().execute(_order(quantity=0.0), BAR, cash=1e9, position_quantity=0.0)
    assert rej.reason == FailureKind.INVALID_ORDER


def test_bar_for_another_instrument_is_execution_error():
    rej = _sim().execute(_order(instrument="INS-000002"), BAR, cash=1e9, position_quantity=0.0)
    assert rej.reason == FailureKind.EXECUTION_ERROR


def test_simulator_is_pure_and_deterministic():
    # the executor is a pure function of (order, bar, cash, position):
    # two identical attempts must produce identical outcomes - the
    # simulation never depends on hidden state
    sim = _sim()
    a = sim.execute(_order(), BAR, cash=1e9, position_quantity=0.0)
    b = sim.execute(_order(), BAR, cash=1e9, position_quantity=0.0)
    assert isinstance(a, FillEvent) and isinstance(b, FillEvent)
    assert a.as_dict() == b.as_dict()
    assert a.order_id == b.order_id


# -------------------------------------------------------- timing semantics

def test_order_eligibility_uses_execution_delay_from_signal_session():
    from orbit.backtest.clock import MarketEventClock

    bars = __import__("phase7_testutils", fromlist=["make_bars", "weekdays"]).make_bars(
        __import__("phase7_testutils", fromlist=["weekdays"]).weekdays(date(2024, 1, 2), 10)
    )
    clock = MarketEventClock(bars)
    generator = OrderGenerator(
        BacktestConfig(execution=ExecutionConfig(execution_delay=2)), clock
    )
    from orbit.backtest.events import SignalEvent, EventType

    signal = SignalEvent(
        run_id="r",
        event_type=EventType.SIGNAL,
        sequence=0,
        session=date(2024, 1, 2),
        timestamp=None,
        source="t",
        signal_id="SIG-000001",
        instrument_id="INS-000001",
        signal_session=date(2024, 1, 2),
        decision_time=None,
        direction=SignalDirection.LONG,
        target=100.0,
    )
    order = generator.generate(signal, __import__(
        "orbit.backtest.ledger", fromlist=["PortfolioLedger"]
    ).PortfolioLedger(1e6), last_equity=1e6, signal_close=100.0, order_index=0)
    assert order is not None
    assert order.eligible_session == date(2024, 1, 4)  # two sessions later
    # the order may wait order_expiry_sessions sessions for a bar
    assert order.expiry_session == date(2024, 1, 11)  # eligible + 5 sessions