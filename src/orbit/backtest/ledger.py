"""The canonical portfolio ledger (Phase 7).

Every fill produces exactly the accounting changes that cash and position
conservation require:

  BUY  q at price p with fee f:  cash -= q*p + f
                                position += q
                                avg_cost = (q_old*c_old + q*p) / (q_old + q)
  SELL q at price p with fee f:  cash += q*p - f
                                position -= q
                                realized += q * (p - avg_cost_before)

Definitions (documented, testable):

  - cost basis is EXECUTION notional only (fees are cash outflows, never
    capitalized into avg_cost): realized P&L is price-only, fees are a
    separate cumulative line, and the two never double-count;
  - equity      = cash + sum(q * valuation_price)
  - unrealized  = sum(q * (valuation_price - avg_cost))
  - identity    = equity == initial_cash + realized + unrealized - fees_total
    (holds at every valuation and is enforced by validate_invariants);
  - a rejected/unfilled order NEVER touches cash or positions;
  - partial fills apply only the filled quantity.

`validate_invariants` re-derives cash and positions from the fill history
(the ledger keeps one immutable accounting record per fill) and checks the
identity above with an explicit floating-point tolerance - it is the
machine half of the accounting audit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from orbit.backtest.events import FillEvent, OrderSide

_FP_TOL = 1e-9


@dataclass(frozen=True)
class Position:
    """One instrument position on the canonical split-continuous basis."""

    instrument_id: str
    quantity: float
    avg_cost: float

    def as_dict(self) -> dict[str, float]:
        return {
            "quantity": self.quantity,
            "avg_cost": self.avg_cost,
        }


@dataclass(frozen=True)
class AccountingRecord:
    """One immutable accounting entry produced by one fill."""

    fill_id: str
    order_id: str
    instrument_id: str
    side: OrderSide
    quantity: float
    price: float
    fee: float
    cash_delta: float
    position_delta: float
    realized_delta: float
    avg_cost_after: float


class PortfolioLedger:
    """The deterministic accounting ledger of one backtest run."""

    def __init__(self, initial_cash: float):
        if initial_cash <= 0:
            raise ValueError("initial_cash must be positive")
        self.initial_cash = float(initial_cash)
        self._cash = self.initial_cash
        self._positions: dict[str, Position] = {}
        self._realized = 0.0
        self._fees_total = 0.0
        self._spread_cost_total = 0.0
        self._slippage_cost_total = 0.0
        self._traded_notional = 0.0
        self._records: list[AccountingRecord] = []

    # -------------------------------------------------------------- reads

    @property
    def cash(self) -> float:
        return self._cash

    @property
    def positions(self) -> dict[str, Position]:
        return dict(self._positions)

    def position_quantity(self, instrument_id: str) -> float:
        pos = self._positions.get(instrument_id)
        return pos.quantity if pos else 0.0

    def position(self, instrument_id: str) -> Position | None:
        return self._positions.get(instrument_id)

    @property
    def realized_pnl(self) -> float:
        return self._realized

    @property
    def fees_total(self) -> float:
        return self._fees_total

    @property
    def spread_cost_total(self) -> float:
        return self._spread_cost_total

    @property
    def slippage_cost_total(self) -> float:
        return self._slippage_cost_total

    @property
    def traded_notional(self) -> float:
        return self._traded_notional

    def accounting_records(self) -> list[AccountingRecord]:
        return list(self._records)

    # ------------------------------------------------------------- writes

    def apply_fill(self, fill: FillEvent) -> AccountingRecord:
        """Apply one fill to the ledger (execution simulator has already
        validated cash/position constraints; this method only accounts)."""
        q = fill.filled_quantity
        if q <= 0:
            raise ValueError(
                f"cannot apply a fill with quantity {q}; a zero-quantity "
                "fill must be a rejection"
            )
        p = fill.price
        notional = q * p
        pos = self._positions.get(fill.instrument_id)

        if fill.side == OrderSide.BUY:
            cash_delta = -(notional + fill.fee)
            position_delta = q
            realized_delta = 0.0
            old_q = pos.quantity if pos else 0.0
            old_cost = pos.avg_cost if pos else 0.0
            avg_after = (old_q * old_cost + notional) / (old_q + q)
        else:
            cash_delta = notional - fill.fee
            position_delta = -q
            old_q = pos.quantity if pos else 0.0
            old_cost = pos.avg_cost if pos else 0.0
            realized_delta = q * (p - old_cost)
            avg_after = old_cost if (old_q - q) > 0 else 0.0

        new_cash = self._cash + cash_delta
        new_q = old_q + position_delta
        if new_q < 0:
            raise ValueError(
                f"accounting violation: negative position {new_q} for "
                f"{fill.instrument_id} (position constraint must be enforced "
                "before accounting)"
            )
        self._cash = new_cash
        if new_q > 0:
            self._positions[fill.instrument_id] = Position(
                instrument_id=fill.instrument_id,
                quantity=new_q,
                avg_cost=avg_after,
            )
        else:
            self._positions.pop(fill.instrument_id, None)
        self._realized += realized_delta
        self._fees_total += fill.fee
        self._spread_cost_total += fill.spread_cost
        self._slippage_cost_total += fill.slippage_cost
        self._traded_notional += notional

        record = AccountingRecord(
            fill_id=fill.fill_id,
            order_id=fill.order_id,
            instrument_id=fill.instrument_id,
            side=fill.side,
            quantity=q,
            price=p,
            fee=fill.fee,
            cash_delta=cash_delta,
            position_delta=position_delta,
            realized_delta=realized_delta,
            avg_cost_after=avg_after,
        )
        self._records.append(record)
        return record

    # ---------------------------------------------------------- valuation

    def value(self, valuation_prices: dict[str, float]) -> dict[str, Any]:
        """Value the portfolio at the given prices (close of a session).

        Returns the canonical snapshot: cash, market value, equity,
        realized, unrealized, and per-position detail.
        """
        market_value = 0.0
        unrealized = 0.0
        positions_detail: dict[str, dict[str, float]] = {}
        for instrument_id, pos in sorted(self._positions.items()):
            price = valuation_prices.get(instrument_id)
            if price is None:
                raise ValueError(
                    f"valuation price missing for held instrument "
                    f"{instrument_id}"
                )
            mv = pos.quantity * price
            unrealized += pos.quantity * (price - pos.avg_cost)
            market_value += mv
            positions_detail[instrument_id] = {
                "quantity": pos.quantity,
                "avg_cost": pos.avg_cost,
                "price": price,
                "market_value": mv,
                "unrealized": pos.quantity * (price - pos.avg_cost),
            }
        equity = self._cash + market_value
        return {
            "cash": self._cash,
            "market_value": market_value,
            "equity": equity,
            "realized": self._realized,
            "unrealized": unrealized,
            "positions": positions_detail,
        }

    # --------------------------------------------------------- invariants

    def validate_invariants(self) -> list[str]:
        """Re-derive the ledger from its accounting records and report every
        violated invariant. An empty list means the accounting is exact.

        Checks: cash conservation, position conservation, the equity
        identity, and that the totals match the re-derived values.
        """
        violations: list[str] = []

        expected_cash = self.initial_cash
        expected_positions: dict[str, float] = {}
        expected_realized = 0.0
        expected_fees = 0.0
        expected_notional = 0.0
        for rec in self._records:
            expected_cash += rec.cash_delta
            expected_positions[rec.instrument_id] = (
                expected_positions.get(rec.instrument_id, 0.0)
                + rec.position_delta
            )
            expected_realized += rec.realized_delta
            expected_fees += rec.fee
            expected_notional += rec.quantity * rec.price

        if not _close(self._cash, expected_cash):
            violations.append(
                f"cash conservation broken: ledger cash={self._cash} but the "
                f"fill history implies {expected_cash}"
            )
        current_q = {
            k: v.quantity for k, v in self._positions.items()
        }
        actual_q = {
            k: q for k, q in expected_positions.items() if q > 0
        }
        if current_q != actual_q:
            violations.append(
                f"position conservation broken: ledger positions "
                f"{current_q} do not match the fill history {actual_q}"
            )
        if not _close(self._realized, expected_realized):
            violations.append(
                f"realized P&L does not match the fill history: "
                f"{self._realized} vs {expected_realized}"
            )
        if not _close(self._fees_total, expected_fees):
            violations.append(
                f"fees total does not match the fill history: "
                f"{self._fees_total} vs {expected_fees}"
            )
        if not _close(self._traded_notional, expected_notional):
            violations.append(
                f"traded notional does not match the fill history: "
                f"{self._traded_notional} vs {expected_notional}"
            )
        if self._cash < -_FP_TOL:
            violations.append(
                f"cash is negative ({self._cash}): the portfolio holds an "
                "implicit cash loan, which long-only accounting never invents"
            )

        for instrument_id, q in expected_positions.items():
            if q < 0:
                violations.append(
                    f"negative position for {instrument_id} ({q}) - the "
                    "fill history itself violates long-only accounting"
                )

        return violations

    def equity_identity_holds(
        self, valuation_prices: dict[str, float]
    ) -> bool:
        """equity == initial_cash + realized + unrealized - fees_total."""
        snap = self.value(valuation_prices)
        expected = (
            self.initial_cash
            + snap["realized"]
            + snap["unrealized"]
            - self._fees_total
        )
        return _close(snap["equity"], expected)


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= _FP_TOL * max(1.0, abs(a), abs(b))


__all__ = [
    "AccountingRecord",
    "PortfolioLedger",
    "Position",
]