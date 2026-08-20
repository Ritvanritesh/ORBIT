"""Phase 7 ledger tests: the accounting invariants, definitions and
conservation laws under adversarial fill sequences."""

from __future__ import annotations

from datetime import date

import pytest

from orbit.backtest.events import FillEvent, OrderSide
from orbit.backtest.ledger import PortfolioLedger


def _fill(
    ledger,
    instrument="INS-000001",
    side=OrderSide.BUY,
    q=100.0,
    p=100.0,
    fee=1.0,
    fill_id="FILL-1",
):
    return FillEvent(
        run_id="r",
        event_type="fill",
        sequence=0,
        session=date(2024, 1, 2),
        timestamp=None,
        source="t",
        fill_id=fill_id,
        order_id="ORD-1",
        signal_id="SIG-1",
        instrument_id=instrument,
        side=side,
        requested_quantity=q,
        filled_quantity=q,
        unfilled_quantity=0.0,
        unfilled_reason=None,
        price=p,
        reference_price=p,
        execution_price="open",
        spread_cost=0.0,
        slippage_cost=0.0,
        fee=fee,
        price_basis="split_continuous_stored",
    )


def test_buy_debits_cash_and_establishes_position():
    ledger = PortfolioLedger(1_000_000.0)
    ledger.apply_fill(_fill(ledger, q=1000.0, p=100.0, fee=5.0))
    assert ledger.cash == 1_000_000.0 - 100_000.0 - 5.0
    pos = ledger.position("INS-000001")
    assert pos.quantity == 1000.0
    assert pos.avg_cost == 100.0  # execution notional only, fees never capitalized


def test_avg_cost_weights_by_notional():
    ledger = PortfolioLedger(1_000_000.0)
    ledger.apply_fill(_fill(ledger, q=100.0, p=100.0, fee=0.0, fill_id="FILL-1"))
    ledger.apply_fill(_fill(ledger, q=100.0, p=120.0, fee=0.0, fill_id="FILL-2"))
    assert ledger.position("INS-000001").avg_cost == 110.0


def test_sell_realizes_price_only_pnl_at_avg_cost_before():
    ledger = PortfolioLedger(1_000_000.0)
    ledger.apply_fill(_fill(ledger, q=1000.0, p=100.0, fee=0.0, fill_id="FILL-1"))
    before = ledger.position("INS-000001").avg_cost
    sell = _fill(
        ledger, side=OrderSide.SELL, q=400.0, p=110.0, fee=2.0, fill_id="FILL-2"
    )
    rec = ledger.apply_fill(sell)
    assert rec.realized_delta == 400.0 * (110.0 - before)
    assert ledger.realized_pnl == pytest.approx(4000.0, rel=1e-9)
    assert ledger.cash == pytest.approx(
        1_000_000.0 - 100_000.0 + 400.0 * 110.0 - 2.0, rel=1e-9
    )
    assert ledger.position("INS-000001").quantity == 600.0


def test_position_removed_when_flat():
    ledger = PortfolioLedger(1_000_000.0)
    ledger.apply_fill(_fill(ledger, q=100.0, p=100.0, fee=0.0, fill_id="FILL-1"))
    ledger.apply_fill(_fill(ledger, side=OrderSide.SELL, q=100.0, p=101.0, fee=0.0, fill_id="FILL-2"))
    assert ledger.position_quantity("INS-000001") == 0.0
    assert "INS-000001" not in ledger.positions


def test_negative_position_is_an_accounting_violation():
    ledger = PortfolioLedger(1_000_000.0)
    ledger.apply_fill(_fill(ledger, q=100.0, p=100.0, fee=0.0, fill_id="FILL-1"))
    with pytest.raises(ValueError, match="negative position"):
        ledger.apply_fill(
            _fill(ledger, side=OrderSide.SELL, q=500.0, p=101.0, fee=0.0, fill_id="FILL-2")
        )


def test_zero_quantity_fill_is_refused():
    ledger = PortfolioLedger(1_000_000.0)
    with pytest.raises(ValueError, match="zero-quantity"):
        ledger.apply_fill(_fill(ledger, q=0.0, fill_id="FILL-1"))


def test_equity_identity_holds_at_valuation():
    ledger = PortfolioLedger(1_000_000.0)
    ledger.apply_fill(_fill(ledger, q=1000.0, p=100.0, fee=5.0, fill_id="FILL-1"))
    ledger.apply_fill(_fill(ledger, side=OrderSide.SELL, q=300.0, p=105.0, fee=2.0, fill_id="FILL-2"))
    assert ledger.equity_identity_holds({"INS-000001": 110.0})
    snap = ledger.value({"INS-000001": 110.0})
    assert snap["equity"] == pytest.approx(
        1_000_000.0 + snap["realized"] + snap["unrealized"] - ledger.fees_total,
        rel=1e-9,
    )


def test_validate_invariants_clean_after_random_walk():
    ledger = PortfolioLedger(1_000_000.0)
    q = 0.0
    price = 100.0
    i = 0
    for delta in [100.0, -40.0, 60.0, -120.0, 30.0, -30.0]:
        i += 1
        q += delta
        side = OrderSide.BUY if delta > 0 else OrderSide.SELL
        ledger.apply_fill(
            _fill(
                ledger,
                side=side,
                q=abs(delta),
                p=price,
                fee=abs(delta) * 0.1,
                fill_id=f"FILL-{i}",
            )
        )
        price *= 1.01
    assert ledger.validate_invariants() == []
    assert ledger.equity_identity_holds({"INS-000001": price})


def test_invariant_replay_detects_tampered_cash():
    ledger = PortfolioLedger(1_000_000.0)
    ledger.apply_fill(_fill(ledger, q=1000.0, p=100.0, fee=5.0, fill_id="FILL-1"))
    # tamper with the private cash (a bug in the ledger would do the same)
    object.__setattr__(ledger, "_cash", ledger.cash - 1.0)
    violations = ledger.validate_invariants()
    assert any("cash conservation" in v for v in violations)


def test_missing_valuation_price_is_refused():
    ledger = PortfolioLedger(1_000_000.0)
    ledger.apply_fill(_fill(ledger, q=1000.0, p=100.0, fee=0.0, fill_id="FILL-1"))
    with pytest.raises(ValueError, match="valuation price missing"):
        ledger.value({})


def test_multi_instrument_positions_are_independent():
    ledger = PortfolioLedger(1_000_000.0)
    ledger.apply_fill(_fill(ledger, q=100.0, p=100.0, fee=0.0, fill_id="FILL-1"))
    ledger.apply_fill(_fill(ledger, instrument="INS-000002", q=200.0, p=50.0, fee=0.0, fill_id="FILL-2"))
    assert ledger.position_quantity("INS-000001") == 100.0
    assert ledger.position_quantity("INS-000002") == 200.0
    assert ledger.cash == 1_000_000.0 - 100.0 * 100.0 - 200.0 * 50.0
    assert ledger.validate_invariants() == []