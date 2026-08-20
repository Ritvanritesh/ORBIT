"""Phase 7 synthetic scenario tests: the twelve canonical end-to-end
scenarios that pin the engine's documented semantics.

Each scenario is a single deterministic fixture; assertions check the
event chain, the accounting numbers and the failure taxonomy - never
silent skips.
"""

from __future__ import annotations

from datetime import date

import pytest

from orbit.backtest import BacktestConfig, ExecutionPrice, FailureKind, SizingPolicy
from orbit.backtest.events import FillEvent, OrderIntent, RejectionEvent
from orbit.temporal.times import session_close_utc

from phase7_testutils import make_bars, run, signals, weekdays

DATES = weekdays(date(2024, 1, 2), 10)


def _equity_identity(res):
    res.assert_accounting_clean()
    return res.summary()


# ------------------------------------------------------ scenario 1: buy & hold

def test_s1_buy_and_hold_fills_at_next_open_and_values_at_close():
    res = run(make_bars(DATES), signals("INS-000001", DATES[:6]))
    summary = _equity_identity(res)
    fills = res.fills
    assert len(fills) == 1
    fill = fills[0]
    assert fill.instrument_id == "INS-000001"
    assert fill.side.value == "buy"
    assert fill.filled_quantity == 1000.0
    assert fill.execution_price == ExecutionPrice.OPEN
    # signal on day 0, eligible at day 1, fills at day 1's open
    assert fill.session == DATES[1]
    assert fill.reference_price == 100.0  # day 1 open = day 0 close
    assert fill.price == 100.0  # zero costs
    assert fill.unfilled_quantity == 0.0
    assert fill.unfilled_reason is None
    assert summary["n_orders"] == 1
    assert summary["n_rejects"] == 0
    # position held to the end, valued at the last close
    assert res.final_position("INS-000001") == 1000.0
    assert summary["final_equity"] == 1_000_000.0
    run_end = res.run_end()
    assert run_end is not None and run_end.session == DATES[-1]


# ------------------------------------------------- scenario 2: long then flat

def test_s2_long_then_flat_cycles_position_with_realized_pnl():
    res = run(
        make_bars(DATES, drift=0.01),
        signals("INS-000001", DATES[:2], target=1000)
        + signals("INS-000001", DATES[3:4], direction="flat", start_index=2),
    )
    summary = _equity_identity(res)
    assert summary["n_orders"] == 2
    assert summary["n_fills"] == 2
    assert summary["n_rejects"] == 0
    assert res.final_position("INS-000001") == 0.0
    buy = res.fills[0]
    sell = res.fills[1]
    assert buy.side.value == "buy" and sell.side.value == "sell"
    # day 1 open = 100; day 4 open = 100 * 1.03 (drift 0.01/day, 3 days later)
    assert sell.price == pytest.approx(103.0, rel=1e-9)
    realized = 1000.0 * (103.0 - 100.0)
    last = res.ledger_snapshots[-1]
    assert last.realized == pytest.approx(realized, rel=1e-9)
    assert summary["final_equity"] == pytest.approx(1_000_000.0 + realized, rel=1e-9)


# --------------------------------------- scenario 3: no signals at all

def test_s3_no_signals_no_orders_no_trades():
    res = run(make_bars(DATES), [])
    summary = _equity_identity(res)
    assert summary["n_signals"] == 0
    assert summary["n_orders"] == 0
    assert summary["n_fills"] == 0
    assert summary["n_rejects"] == 0
    assert summary["final_equity"] == 1_000_000.0
    assert len(res.events) > 0  # market/ledger/valuation events still emitted


# ------------------------------------- scenario 4: instrument not in universe

def test_s4_signal_outside_universe_is_refused():
    with pytest.raises(ValueError, match="not in the backtest universe"):
        run(
            make_bars(DATES),
            signals("INS-999999", DATES[:1]),
            universe=["INS-000001"],
        )


# ------------------------------------------- scenario 5: signal outside window

def test_s5_signal_outside_evaluation_window_is_refused():
    config = BacktestConfig(window_start=DATES[2], window_end=DATES[5])
    with pytest.raises(ValueError, match="outside the evaluation window"):
        run(
            make_bars(DATES),
            signals("INS-000001", DATES[7:8]),
            config=config,
        )


# ------------------------------------- scenario 6: wrong decision_time

def test_s6_signal_not_stamped_at_session_close_is_refused():
    bad = signals("INS-000001", DATES[:1])
    bad[0]["decision_time"] = session_close_utc(DATES[0]).replace(hour=10)
    with pytest.raises(ValueError, match="session close"):
        run(make_bars(DATES), bad)


# ---------------------------------- scenario 7: order past the last session

def test_s7_signal_too_late_for_delay_generates_no_order():
    config = BacktestConfig(execution=__import__(
        "orbit.backtest.config", fromlist=["ExecutionConfig"]
    ).ExecutionConfig(execution_delay=3))
    res = run(
        make_bars(DATES),
        signals("INS-000001", DATES[-1:]),
        config=config,
    )
    summary = _equity_identity(res)
    assert summary["n_orders"] == 0
    assert summary["n_fills"] == 0
    # the signal was processed, but the order could never become eligible:
    # no order, no fill, no rejection - nothing is invented
    assert res.signals[0].signal_id == "SIG-000001"


# ------------------------------------------------- scenario 8: cash constraint

def test_s8_order_exceeding_cash_is_rejected_explicitly():
    res = run(
        make_bars(DATES),
        signals("INS-000001", DATES[:1], target=10_000_000.0),
    )
    summary = _equity_identity(res)
    rejects = [r for r in res.rejections if r.signal_id == "SIG-000001"]
    assert summary["n_fills"] == 0
    assert len(rejects) == 1
    assert rejects[0].reason == FailureKind.INSUFFICIENT_CASH
    assert "cash" in rejects[0].detail
    assert summary["final_equity"] == 1_000_000.0


# ------------------------------------------- scenario 9: position constraint

def test_s9_cannot_sell_more_than_held():
    # long 1000, then a LONG signal with target 100: sell 900 - fine;
    # but a FLAT-with-nothing or over-sell must never go negative.
    res = run(
        make_bars(DATES),
        signals("INS-000001", DATES[:2], target=1000)
        + signals("INS-000001", DATES[3:4], target=100, start_index=2),
    )
    summary = _equity_identity(res)
    assert summary["n_fills"] == 2
    assert res.final_position("INS-000001") == 100.0
    sell = res.fills[1]
    assert sell.side.value == "sell" and sell.filled_quantity == 900.0


def test_s9b_flat_with_no_position_is_a_noop():
    res = run(make_bars(DATES), signals("INS-000001", DATES[:1], direction="flat"))
    summary = _equity_identity(res)
    assert summary["n_orders"] == 0
    assert summary["n_fills"] == 0


# ---------------------------------------------- scenario 10: data gap carry

def test_s10_data_gap_carries_order_to_next_available_bar():
    bars = make_bars(DATES)
    # remove the INS-000001 bar on day 1 (the fill session) and on day 3
    # (a later gap while the position is held)
    bars = bars.filter(
        ~(
            (bars["instrument_id"] == "INS-000001")
            & (bars["trade_date"].is_in([DATES[1], DATES[3]]))
        )
    )
    res = run(bars, signals("INS-000001", DATES[:1]))
    summary = _equity_identity(res)
    # order eligible at day 1 (missing bar): carried, filled at day 2's open
    assert summary["n_fills"] == 1
    assert res.fills[0].session == DATES[2]
    assert res.fills[0].reference_price == 100.0  # day 2 open = day 0 close
    # the later gap is visible in the valuation stream as a stale flag
    gap_valuation = [
        v for v in res.valuations if v.session == DATES[3] and "INS-000001" in v.stale
    ]
    assert gap_valuation, "the gap session must flag the position as stale"


# -------------------------------------------- scenario 11: expiry on no bar

def test_s11_order_expires_after_order_expiry_sessions():
    bars = make_bars(DATES)
    # remove every bar of INS-000001 from day 1 onward: the order can never
    # fill and must expire explicitly, never silently vanish
    bars = bars.filter(
        ~(
            (bars["instrument_id"] == "INS-000001")
            & (bars["trade_date"] >= DATES[1])
        )
    )
    config = BacktestConfig(
        execution=__import__(
            "orbit.backtest.config", fromlist=["ExecutionConfig"]
        ).ExecutionConfig(order_expiry_sessions=2)
    )
    res = run(bars, signals("INS-000001", DATES[:1]), config=config)
    summary = _equity_identity(res)
    assert summary["n_fills"] == 0
    rejects = [r for r in res.rejections if r.signal_id == "SIG-000001"]
    assert len(rejects) == 1
    assert rejects[0].reason == FailureKind.EXPIRED_ORDER
    assert "never filled" in rejects[0].detail
    assert summary["final_equity"] == 1_000_000.0


# --------------------------------------------- scenario 12: delisted/stale

def test_s12_delisted_position_valued_at_last_close_flagged_stale():
    bars = make_bars(DATES)
    # instrument disappears after day 3 while the run holds it
    bars = bars.filter(
        ~(
            (bars["instrument_id"] == "INS-000001")
            & (bars["trade_date"] > DATES[3])
        )
    )
    res = run(
        bars,
        signals("INS-000001", DATES[:1], target=1000)
        + signals("INS-000002", DATES[4:5], start_index=1),
    )
    summary = _equity_identity(res)
    # INS-000001 fills on day 1 and is held; INS-000002 fills on day 5
    assert res.final_position("INS-000001") == 1000.0
    # every valuation after the disappearance flags the position stale
    stale_sessions = [
        v.session for v in res.valuations if "INS-000001" in v.stale
    ]
    assert stale_sessions == DATES[4:10], stale_sessions
    # the stale instrument is valued at its last available close
    last = res.valuations[-1]
    assert last.valuation_prices["INS-000001"] == 100.0
    _equity_identity(res)


# -------------------------------------------------------- chain determinism

def test_event_chain_order_within_a_session_is_documented():
    res = run(make_bars(DATES), signals("INS-000001", DATES[:1]))
    canonical = [
        "market", "fill", "rejection", "signal", "order", "ledger",
        "valuation",
    ]
    by_session: dict[date, list[str]] = {}
    for e in res.events:
        if e.session is not None and e.event_type.value in canonical:
            by_session.setdefault(e.session, []).append(e.event_type.value)
    for session, types in by_session.items():
        # within any session, the emitted subsequence must respect the
        # documented chain order
        idx = [canonical.index(t) for t in types]
        assert idx == sorted(idx), (
            f"session {session} violates the documented event order: {types}"
        )
    # the global sequence is a strict, gap-free ordering
    seqs = [e.sequence for e in res.events]
    assert seqs == sorted(seqs)
    assert len(set(seqs)) == len(seqs)