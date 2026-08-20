"""Phase 7 adversarial audit tests: the second, independent pass that tries
to break the engine - tampered inputs, degenerate configurations, hostile
data, and accounting that must fail loudly instead of silently corrupting.
The bottom section holds regression tests for every finding of the first
independent audit pass (each finding, its fix, and its test)."""

from __future__ import annotations

from datetime import date

import math
import pytest

import polars as pl

from orbit.backtest import BacktestConfig, CostConfig, FailureKind
from orbit.backtest.config import ExecutionConfig, ExecutionPrice
from orbit.backtest.clock import BarRecord, MarketEventClock
from orbit.backtest.events import (
    EventType,
    FillEvent,
    OrderIntent,
    OrderSide,
    SignalEvent,
)
from orbit.backtest.execution import ExecutionSimulator
from orbit.backtest.ledger import PortfolioLedger

from phase7_testutils import make_bars, run_default, signals, weekdays

DATES = weekdays(date(2024, 1, 2), 10)
_BARS = make_bars(DATES)


# ---------------------------------------------------------- configuration

def test_long_only_false_is_refused_loudly():
    with pytest.raises(ValueError, match="long-only"):
        BacktestConfig(long_only=False)


def test_open_execution_requires_delay_at_least_one():
    with pytest.raises(ValueError, match="execution_delay"):
        ExecutionConfig(execution_price=ExecutionPrice.OPEN, execution_delay=0)


def test_negative_initial_cash_is_refused():
    with pytest.raises(ValueError):
        BacktestConfig(initial_cash=0.0)
    with pytest.raises(ValueError):
        PortfolioLedger(0.0)


def test_degenerate_cost_values_are_refused():
    with pytest.raises(ValueError):
        CostConfig(spread_bps=-1)
    with pytest.raises(ValueError):
        ExecutionConfig(participation_fraction=0.0)


def test_inverted_window_is_refused():
    with pytest.raises(ValueError, match="window_end"):
        BacktestConfig(window_start=DATES[5], window_end=DATES[2])


def test_empty_window_is_refused_at_run():
    bt = run_default(_BARS, [], config=BacktestConfig(
        window_start=date(2030, 1, 1), window_end=date(2030, 2, 1)
    ))
    with pytest.raises(ValueError, match="no market sessions"):
        bt.run(_BARS, signals("INS-000001", DATES[:1]))


# --------------------------------------------------------------- hostile data

def test_duplicate_bars_are_refused():
    dup = _BARS.vstack(_BARS.select(_BARS.columns))
    with pytest.raises(ValueError, match="duplicate"):
        MarketEventClock(dup)


def test_missing_bar_columns_are_refused():
    with pytest.raises(ValueError, match="columns"):
        MarketEventClock(_BARS.drop("volume"))


def test_zero_volume_sessions_reject_liquidity_orders():
    bars = make_bars(DATES, volume=0.0)
    res = run_default(bars, []).run(
        bars, signals("INS-000001", DATES[:1], target=1000)
    )
    rejects = [r for r in res.rejections if r.signal_id == "SIG-000001"]
    assert len(rejects) == 1
    assert rejects[0].reason == FailureKind.LIQUIDITY_REJECTION


def test_nan_close_in_bar_is_a_loud_data_error():
    import polars as pl

    bars = make_bars(DATES)
    bars = bars.with_columns(
        pl.when(
            (pl.col("instrument_id") == "INS-000001")
            & (pl.col("trade_date") == DATES[0])
        )
        .then(float("nan"))
        .otherwise(pl.col("close"))
        .alias("close")
    )
    # a defective anchor bar must never be sized against: the signal
    # produces an explicit DATA_ERROR rejection record, never a guess
    res = run_default(bars, []).run(bars, signals("INS-000001", DATES[:1]))
    rejects = [r for r in res.rejections if r.signal_id == "SIG-000001"]
    assert len(rejects) == 1
    assert rejects[0].reason == FailureKind.DATA_ERROR
    assert res.summary()["n_fills"] == 0


def test_missing_decision_time_defaults_to_session_close():
    rows = signals("INS-000001", DATES[:1])
    del rows[0]["decision_time"]
    res = run_default(_BARS, []).run(_BARS, rows)
    assert res.fills[0].signal_id == "SIG-000001"


# ------------------------------------------------------------ accounting

def test_run_accounting_identity_holds_with_partial_fills():
    bars = make_bars(DATES, volume=1_000.0)
    res = run_default(bars, []).run(
        bars, signals("INS-000001", DATES[:1], target=100_000.0)
    )
    summary = res.summary()
    assert summary["n_fills"] == 1
    assert summary["n_unfilled_partial"] == 1
    res.assert_accounting_clean()
    fill = res.fills[0]
    assert fill.filled_quantity == 50.0  # 5% of 1000
    assert fill.unfilled_quantity == 99_950.0


def test_run_accounting_identity_holds_with_rejections():
    res = run_default(_BARS, []).run(
        _BARS, signals("INS-000001", DATES[:1], target=10_000_000.0)
    )
    assert res.summary()["n_rejects"] == 1
    res.assert_accounting_clean()
    assert res.summary()["final_equity"] == 1_000_000.0


def test_no_phantom_fills_for_rejected_orders():
    res = run_default(_BARS, []).run(
        _BARS, signals("INS-000001", DATES[:1], target=10_000_000.0)
    )
    assert res.fills == []
    assert res.final_position("INS-000001") == 0.0
    assert res.ledger_snapshots[-1].cash == 1_000_000.0


def test_negative_target_signal_is_refused():
    rows = signals("INS-000001", DATES[:1], target=100)
    rows[0]["target"] = -5.0
    with pytest.raises(ValueError, match="negative"):
        run_default(_BARS, []).run(_BARS, rows)


def test_unknown_direction_is_refused():
    rows = signals("INS-000001", DATES[:1])
    rows[0]["direction"] = "short"
    with pytest.raises(ValueError):
        run_default(_BARS, []).run(_BARS, rows)


def test_fill_ids_rejection_ids_and_order_ids_are_disjoint():
    res = run_default(_BARS, []).run(
        _BARS, signals("INS-000001", DATES[:1], target=10_000_000.0)
    )
    fills = {f.fill_id for f in res.fills}
    rejs = {r.rejection_id for r in res.rejections}
    orders = {o.order_id for o in res.orders}
    assert not (fills & rejs)
    assert not (orders & fills)
    assert not (orders & rejs)


def test_every_event_traces_to_config_ref_and_run_id():
    res = run_default(_BARS, []).run(_BARS, signals("INS-000001", DATES[:1]))
    for e in res.events:
        assert e.run_id == res.run_id
        assert e.config_ref == res.manifest.config_hash
        assert e.event_id.startswith(f"{res.run_id}-E")


def test_benchmark_is_analytical_never_a_ledger_transaction():
    bars = make_bars(DATES, instruments=["INS-000001", "SPY"],
                     base_prices={"INS-000001": 100.0, "SPY": 400.0})
    config = BacktestConfig(benchmark="SPY")
    res = run_default(bars, [], config=config).run(
        bars, signals("INS-000001", DATES[:1])
    )
    # SPY never enters the ledger: no fills for it, no position
    assert all(f.instrument_id == "INS-000001" for f in res.fills)
    assert res.final_position("SPY") == 0.0
    # the valuation carries the benchmark comparison separately
    vals = res.valuations
    assert any(v.benchmark_return is not None for v in vals)
    res.assert_accounting_clean()


def test_cash_never_goes_negative_even_with_fees():
    # a buy at the edge of the cash budget must include the fee in the check
    bars = make_bars(DATES, instruments=["INS-000001"], base_prices={"INS-000001": 100.0})
    config = BacktestConfig(initial_cash=100_000.0, costs=CostConfig(fees_bps=5))
    res = run_default(bars, [], config=config).run(
        bars, signals("INS-000001", DATES[:1], target=995)
    )
    # 995 x 100 = 99,500 + 49.75 fee = 99,549.75 <= 100,000: deterministic fill
    assert res.summary()["n_fills"] == 1
    res.assert_accounting_clean()
    assert res.ledger_snapshots[-1].cash == pytest.approx(
        100_000.0 - 99_549.75, rel=1e-9
    )
    # one share more does NOT fit: 1000 x 100 + 50 fee = 100,050 > 100,000
    edge = run_default(bars, [], config=config).run(
        bars, signals("INS-000001", DATES[:1], target=1000)
    )
    rejects = [r for r in edge.rejections if r.signal_id == "SIG-000001"]
    assert len(rejects) == 1
    assert rejects[0].reason == FailureKind.INSUFFICIENT_CASH
    edge.assert_accounting_clean()


# ------------------------------------------------------ audit-fix regressions
# Each test below pins one finding of the first independent audit pass to
# its fix. Removing any of them silently re-opens the finding.


def test_sell_whose_net_proceeds_are_negative_is_rejected():
    # MAJOR finding: a sell whose fee exceeds its proceeds used to be able
    # to drive cash negative (an implicit cash loan). The fix: the sell is
    # refused with INSUFFICIENT_CASH before any cash can go below zero.
    bars = pl.DataFrame(
        {
            "instrument_id": ["INS-000001"] * 3,
            "trade_date": DATES[:3],
            "open": [1.0, 1.0, 0.02],
            "high": [1.01, 1.01, 0.03],
            "low": [0.99, 0.99, 0.01],
            "close": [1.0, 1.0, 0.02],
            "volume": [1_000_000.0] * 3,
        }
    )
    config = BacktestConfig(
        initial_cash=1_100.0, costs=CostConfig(fixed_fee_per_order=100.0)
    )
    rows = (
        signals("INS-000001", DATES[:1], target=1000)
        + signals("INS-000001", DATES[1:2], direction="flat", start_index=1)
    )
    res = run_default(bars, [], config=config).run(bars, rows)
    # buy 1000 @ 1.0 + 100 fee = 1,100 -> cash 0; the crash to 0.02 makes
    # the sell's net proceeds 20 - 100 = -80: refused, cash stays 0
    assert res.summary()["n_fills"] == 1
    rejects = [r for r in res.rejections if r.signal_id == "SIG-000002"]
    assert len(rejects) == 1
    assert rejects[0].reason == FailureKind.INSUFFICIENT_CASH
    assert rejects[0].side == OrderSide.SELL
    assert res.ledger_snapshots[-1].cash == pytest.approx(0.0, rel=1e-9)
    assert res.final_position("INS-000001") == 1000.0
    res.assert_accounting_clean()


def test_ledger_invariant_validator_flags_negative_cash():
    # a ledger that is fed a cash-loaning fill directly (bypassing the
    # executor) must be caught by the invariant validator - the simulator
    # and the validator are independent lines of defense
    ledger = PortfolioLedger(1_000.0)
    buy = FillEvent(
        run_id="R", event_type="fill", sequence=0, session=DATES[0],
        timestamp=None, source="test", side="buy", fill_id="F1", order_id="O1",
        signal_id="S1", instrument_id="INS-000001", requested_quantity=100,
        filled_quantity=100, unfilled_quantity=0, unfilled_reason=None,
        price=10.0, reference_price=10.0, execution_price="open",
        spread_cost=0.0, slippage_cost=0.0, fee=0.0, price_basis="split_continuous",
    )
    sell = FillEvent(
        run_id="R", event_type="fill", sequence=1, session=DATES[1],
        timestamp=None, source="test", side="sell", fill_id="F2", order_id="O2",
        signal_id="S2", instrument_id="INS-000001", requested_quantity=100,
        filled_quantity=100, unfilled_quantity=0, unfilled_reason=None,
        price=10.0, reference_price=10.0, execution_price="open",
        spread_cost=0.0, slippage_cost=0.0, fee=5_000.0, price_basis="split_continuous",
    )
    ledger.apply_fill(buy)
    ledger.apply_fill(sell)  # cash 1000 - 1000 (buy) - 5000 (fee) = -5000
    violations = ledger.validate_invariants()
    assert any("cash is negative" in v for v in violations)


def test_order_cannot_fill_after_expiry_when_the_bar_resumes():
    # MAJOR finding: an order whose expiry window passed used to fill the
    # moment a bar reappeared after a gap (expiry was only checked in the
    # no-bar branch). The fix: `session > expiry_session` is evaluated at
    # the top of the pending-order loop, so a resumed bar can never revive
    # an expired order.
    bars = make_bars(DATES)
    bars = bars.filter(
        ~(
            (bars["instrument_id"] == "INS-000001")
            & bars["trade_date"].is_in(DATES[1:4])
        )
    )
    config = BacktestConfig(
        execution=ExecutionConfig(execution_delay=1, order_expiry_sessions=2)
    )
    res = run_default(bars, [], config=config).run(
        bars, signals("INS-000001", DATES[:1])
    )
    # eligible = DATES[1], expiry_session = DATES[3]; the bar resumes at
    # DATES[4] > expiry_session: the order is rejected on DATES[4]
    rejects = [r for r in res.rejections if r.signal_id == "SIG-000001"]
    assert len(rejects) == 1
    assert rejects[0].reason == FailureKind.EXPIRED_ORDER
    assert rejects[0].session == DATES[4]
    assert res.summary()["n_fills"] == 0
    assert res.final_position("INS-000001") == 0.0
    res.assert_accounting_clean()


def test_run_end_expiry_rejection_is_stamped_at_the_last_session():
    # when the run window ends while an order is still pending (its
    # eligible session is the last session, and no bar exists there), the
    # terminal rejection must carry the run's last session - not the
    # order's submission session (the stream's session/timestamp must agree)
    bars = make_bars(DATES)
    bars = bars.filter(
        ~(
            (bars["instrument_id"] == "INS-000001")
            & (bars["trade_date"] == DATES[-1])
        )
    )
    res = run_default(bars, []).run(bars, signals("INS-000001", DATES[-2:-1]))
    rejects = [r for r in res.rejections if r.signal_id == "SIG-000001"]
    assert len(rejects) == 1
    assert rejects[0].reason == FailureKind.EXPIRED_ORDER
    assert rejects[0].session == DATES[-1]
    assert res.summary()["n_fills"] == 0
    res.assert_accounting_clean()


def test_same_instant_sell_cash_feeds_same_instant_buy():
    # MAJOR finding: two orders eligible at the same execution instant used
    # to be attempted in order-id order, so a BUY that depended on the cash
    # of a same-instant SELL was rejected. The fix: within one execution
    # instant, cash-generating SELLs are attempted before cash-consuming
    # BUYs (each group in order-id order).
    bars = make_bars(DATES, instruments=["INS-000001", "INS-000002"])
    config = BacktestConfig(initial_cash=60_000.0)
    rows = (
        signals("INS-000001", DATES[:1], target=500)
        + signals("INS-000001", DATES[1:2], direction="flat", start_index=1)
        + signals("INS-000002", DATES[1:2], target=1000, start_index=2)
    )
    res = run_default(bars, [], config=config).run(bars, rows)
    # day 1 open: buy 500 x 100 = 50,000 -> cash 10,000. At day 2 open:
    # sell 500 @ 100 = 50,000 (cash 60,000) must feed buy 1000 x 50 = 50,000
    assert res.summary()["n_fills"] == 3
    assert res.summary()["n_rejects"] == 0
    assert res.final_position("INS-000001") == 0.0
    assert res.final_position("INS-000002") == 1000.0
    res.assert_accounting_clean()


def test_sell_position_constraint_checks_post_cap_filled():
    # MINOR finding: the sell-side position constraint compared the
    # REQUESTED quantity against the held position while the buy side used
    # the post-cap fill - asymmetric. The fix: both sides compare the
    # post-cap `filled`.
    bar = BarRecord(
        instrument_id="INS-000001", trade_date=DATES[0], open=100.0,
        high=101.0, low=99.0, close=100.0, volume=400.0,
        liquidity_volume=400.0, price_basis="split_continuous",
        volume_basis="provider_stored",
    )
    order = OrderIntent(
        run_id="R", event_type=EventType.ORDER, sequence=0, session=DATES[0],
        timestamp=None, source="test", order_id="O1", signal_id="S1",
        instrument_id="INS-000001", side=OrderSide.SELL, quantity=200.0,
    )
    # requested 200 > held 100, but the cap floors the fill to 100 (25% of
    # 400): the sell is legal - a capped-down fill can never over-sell
    sim_capped = ExecutionSimulator(
        BacktestConfig(
            execution=ExecutionConfig(participation_fraction=0.25),
            costs=CostConfig(),
        )
    )
    out = sim_capped.execute(order, bar, cash=1_000_000.0, position_quantity=100.0)
    assert isinstance(out, FillEvent)
    assert out.filled_quantity == 100.0
    # a post-cap fill that still exceeds the held position is refused
    sim_over = ExecutionSimulator(
        BacktestConfig(
            execution=ExecutionConfig(participation_fraction=0.4),
            costs=CostConfig(),
        )
    )
    out = sim_over.execute(order, bar, cash=1_000_000.0, position_quantity=100.0)
    assert not isinstance(out, FillEvent)
    assert out.reason == FailureKind.POSITION_CONSTRAINT


def test_duplicate_signal_ids_are_refused():
    rows = signals("INS-000001", DATES[:2])
    rows[1]["signal_id"] = rows[0]["signal_id"]
    with pytest.raises(ValueError, match="duplicate signal_id"):
        run_default(_BARS, []).run(_BARS, rows)


def test_reordered_signal_rows_are_the_same_run():
    # the signal-set hash and the event stream must be order-insensitive:
    # a pure reordering of the same rows is the same scientific run
    rows = (
        signals("INS-000001", DATES[:3], target=1000)
        + signals("INS-000002", DATES[:2], target=500, start_index=3)
    )
    a = run_default(_BARS, []).run(_BARS, rows)
    b = run_default(_BARS, []).run(_BARS, list(reversed(rows)))
    assert a.manifest.signal_set_hash == b.manifest.signal_set_hash
    assert a.manifest.run_id == b.manifest.run_id
    assert [e.as_dict() for e in a.events] == [e.as_dict() for e in b.events]


def test_signal_outcome_final_quantity_is_per_signal_attribution():
    # MINOR finding: `final_quantity` used to report the instrument's
    # end-of-run position for every signal - misleading when multiple
    # signals trade the same instrument. The fix: the signed sum of the
    # position deltas of that signal's own fills.
    res = run_default(_BARS, []).run(
        _BARS,
        signals("INS-000001", DATES[:2], target=1000)
        + signals("INS-000001", DATES[2:3], direction="flat", start_index=2),
    )
    # SIG-000001 (day 0 LONG) buys 1000; SIG-000002 (day 1 LONG at the
    # same target) generates no order; SIG-000003 (day 2 FLAT) sells 1000
    by_id = {
        o.signal_id: o
        for o in res.events_of(EventType.OUTCOME)
        if o.kind == "signal_outcome"
    }
    assert by_id["SIG-000001"].final_quantity == 1000.0
    assert by_id["SIG-000002"].final_quantity == 0.0
    assert by_id["SIG-000003"].final_quantity == -1000.0
    assert all(
        o.session == DATES[-1]
        for o in res.events_of(EventType.OUTCOME)
        if o.kind == "signal_outcome"
    )


def test_null_ohlc_bar_never_crashes_the_run():
    # MAJOR finding: a null OHLC field used to raise a raw TypeError inside
    # `clock.bar` (float(None)). The fix: null fields are served as NaN and
    # the price guards decide the failure kind.
    bars = make_bars(DATES)
    bars = bars.with_columns(
        pl.when(
            (pl.col("instrument_id") == "INS-000001")
            & (pl.col("trade_date") == DATES[0])
        )
        .then(pl.lit(None, dtype=pl.Float64))
        .otherwise(pl.col("close"))
        .alias("close")
    )
    res = run_default(bars, []).run(bars, signals("INS-000001", DATES[:1]))
    rejects = [r for r in res.rejections if r.signal_id == "SIG-000001"]
    assert len(rejects) == 1
    assert rejects[0].reason == FailureKind.DATA_ERROR
    assert res.summary()["n_fills"] == 0


def test_string_enums_are_coerced_at_event_construction():
    # events are also reconstructed from normalized JSON; a raw string must
    # never survive into as_dict() as a broken enum access
    ev = FillEvent(
        run_id="R", event_type="fill", sequence=0, session=DATES[0],
        timestamp=None, source="test", side="buy", fill_id="F1", order_id="O1",
        signal_id="S1", instrument_id="INS-000001", requested_quantity=1,
        filled_quantity=1, unfilled_quantity=0, unfilled_reason=None,
        price=10.0, reference_price=10.0, execution_price="open",
        spread_cost=0.0, slippage_cost=0.0, fee=0.0, price_basis="split_continuous",
    )
    assert ev.side == OrderSide.BUY
    d = ev.as_dict()
    assert d["event_type"] == "fill"
    assert d["side"] == "buy"


def test_nan_payload_is_sanitized_to_null_in_as_dict():
    # NaN in an event payload would poison JSONL and polars frames
    ev = SignalEvent(
        run_id="R", event_type=EventType.SIGNAL, sequence=0,
        session=DATES[0], timestamp=None, source="test",
        signal_id="S1", instrument_id="INS-000001", signal_session=DATES[0],
        decision_time=None, direction="long", target=1.0,
        signal_metric=float("nan"),
    )
    d = ev.as_dict()
    assert d["signal_metric"] is None


def test_benchmark_without_any_reference_bar_is_refused():
    config = BacktestConfig(benchmark="SPY")
    bt = run_default(_BARS, [], config=config)
    with pytest.raises(ValueError, match="benchmark"):
        bt.run(_BARS, signals("INS-000001", DATES[:1]))


def test_benchmark_reference_skips_a_defective_close():
    # a defective close on the reference session must not poison the
    # benchmark reference: last_close scans backward for a usable close
    bars = make_bars(DATES, instruments=["INS-000001", "SPY"],
                     base_prices={"INS-000001": 100.0, "SPY": 400.0})
    bars = bars.with_columns(
        pl.when(
            (pl.col("instrument_id") == "SPY")
            & (pl.col("trade_date") == DATES[1])
        )
        .then(pl.lit(None, dtype=pl.Float64))
        .otherwise(pl.col("close"))
        .alias("close")
    )
    clock = MarketEventClock(bars)
    # DATES[1]'s SPY close is defective: the reference falls back to the
    # usable DATES[0] close (400.0) instead of returning None or NaN
    assert clock.last_close("SPY", DATES[1]) == pytest.approx(400.0, rel=1e-9)


def test_non_finite_target_and_unknown_direction_are_refused_in_normalization():
    rows = signals("INS-000001", DATES[:1])
    rows[0]["target"] = float("inf")
    with pytest.raises(ValueError, match="finite"):
        MarketEventClock.normalize_signals(rows)
    rows = signals("INS-000001", DATES[:1])
    rows[0]["direction"] = "sideways"
    with pytest.raises(ValueError, match="'long' or 'flat'"):
        MarketEventClock.normalize_signals(rows)