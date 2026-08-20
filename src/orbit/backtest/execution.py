"""The deterministic execution simulator (Phase 7).

Transforms an `OrderIntent` into a `FillEvent` or an explicit
`RejectionEvent` under the configured execution and cost assumptions:

  execution timing  - the order fills only at its eligible session's
                      execution instant (session open or close), never
                      before, and never at a session whose bar was not
                      available when the order was placed;
  available price   - the configured bar field (open or close) of the fill
                      session, on the canonical split-continuous basis;
  spread            - direction-aware: buys pay +spread_bps/1e4, sells
                      receive -spread_bps/1e4 (per-side half-spread);
  fees              - bps commission on filled notional + optional fixed
                      fee, floored at an optional minimum; applied to
                      fills, included in portfolio accounting;
  slippage          - direction-aware, parameterized, deterministic:
                      buys pay +slippage_bps/1e4, sells receive
                      -slippage_bps/1e4 (market impact is NOT modeled -
                      the roadmap delays it until honest calibration data
                      exists);
  liquidity         - max fill = participation_fraction x the fill
                      session's (as-published) volume, additionally capped
                      by an optional absolute per-order quantity; daily
                      volume is a documented liquidity PROXY;
  partial fills     - an order above the cap fills up to the cap and the
                      remainder is recorded explicitly as unfilled (or the
                      whole order is rejected when partial_fills=False);
  rejections        - insufficient cash, position constraints (long-only),
                      invalid orders, expired orders, missing execution
                      bars, unknown instruments: all explicit, none touch
                      the ledger.

The simulator is pure: it never mutates the ledger itself - the backtester
applies fills via the ledger so every accounting change stays auditable.
Every execution assumption is recorded on the fill (spread_cost,
slippage_cost, fee, reference price, execution price) and in the run
manifest.
"""

from __future__ import annotations

import math

from orbit.backtest.config import BacktestConfig, ExecutionPrice
from orbit.backtest.clock import BarRecord
from orbit.backtest.events import (
    EventType,
    FailureKind,
    FillEvent,
    OrderIntent,
    OrderSide,
    RejectionEvent,
)


def _missing_price(value: float | None) -> bool:
    """A price that cannot participate in execution: None, NaN, infinite,
    zero or negative."""
    return (
        value is None
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    )


def _missing_volume(value: float | None) -> bool:
    """A volume that means 'no liquidity': None, NaN, infinite or <= 0."""
    return (
        value is None
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    )


_FP_TOL = 1e-9


class ExecutionSimulator:
    """Pure, deterministic order -> fill/rejection transformer."""

    def __init__(self, config: BacktestConfig):
        self._config = config
        self._costs = config.costs
        self._exec = config.execution

    # ------------------------------------------------------------ public

    def execute(
        self,
        order: OrderIntent,
        bar: BarRecord | None,
        cash: float,
        position_quantity: float,
    ) -> FillEvent | RejectionEvent:
        """Attempt to execute `order` on the fill session described by
        `bar`. `cash` and `position_quantity` are the CURRENT ledger state
        (read-only here); the return value is either a fill or an explicit
        rejection."""
        if order.quantity <= 0:
            return self._reject(
                order, FailureKind.INVALID_ORDER,
                f"order quantity {order.quantity} is not positive",
            )
        if bar is None:
            return self._reject(
                order, FailureKind.NO_EXECUTION_BAR,
                "no execution bar exists for the fill session",
            )
        if bar.instrument_id != order.instrument_id:
            return self._reject(
                order, FailureKind.EXECUTION_ERROR,
                f"execution bar for {bar.instrument_id} cannot fill an order "
                f"for {order.instrument_id}",
            )

        reference = self._reference_price(bar)
        if _missing_price(reference):
            return self._reject(
                order, FailureKind.DATA_ERROR,
                f"the {self._exec.execution_price.value} price of the fill "
                f"session is missing or defective ({reference!r})",
            )

        # --------------------------------------------------- liquidity cap
        volume = bar.liquidity_volume
        if _missing_volume(volume):
            volume_cap = 0.0
        else:
            volume_cap = self._exec.participation_fraction * volume
        cap = volume_cap
        bound_by = "liquidity_cap"
        if self._exec.max_order_quantity is not None:
            if self._exec.max_order_quantity < cap:
                cap = self._exec.max_order_quantity
                bound_by = "max_order_quantity"

        requested = order.quantity
        if requested > cap:
            if not self._exec.partial_fills:
                return self._reject(
                    order, FailureKind.LIQUIDITY_REJECTION,
                    f"requested {requested} shares exceeds the liquidity cap "
                    f"{cap} (participation {self._exec.participation_fraction}"
                    f" x volume {volume}) and partial fills are disabled",
                )
            filled = math.floor(cap)
            if filled <= 0:
                return self._reject(
                    order, FailureKind.LIQUIDITY_REJECTION,
                    f"requested {requested} shares but the liquidity cap "
                    f"({cap}) floors to zero shares - the order is "
                    "unfillable",
                )
            unfilled = requested - filled
            unfilled_reason = bound_by
        else:
            filled = requested
            unfilled = 0.0
            unfilled_reason = None

        # -------------------------------------------------- fill economics
        spread_cost = filled * reference * (self._costs.spread_bps / 1e4)
        slippage_cost = filled * reference * (self._costs.slippage_bps / 1e4)
        if order.side == OrderSide.BUY:
            price = reference * (1.0 + (self._costs.spread_bps + self._costs.slippage_bps) / 1e4)
        else:
            price = reference * (1.0 - (self._costs.spread_bps + self._costs.slippage_bps) / 1e4)
        if price <= 0:
            return self._reject(
                order, FailureKind.DATA_ERROR,
                f"the fill price {price} is not positive after spread and "
                "slippage - the reference price or cost assumptions are "
                "degenerate",
            )

        fee = filled * reference * (self._costs.fees_bps / 1e4)
        fee += self._costs.fixed_fee_per_order
        if self._costs.fee_minimum > 0.0:
            fee = max(fee, self._costs.fee_minimum)

        # ---------------------------------------------------- constraints
        if order.side == OrderSide.BUY:
            if cash < filled * price + fee:
                return self._reject(
                    order, FailureKind.INSUFFICIENT_CASH,
                    f"cash {cash} is below the required {filled * price + fee}"
                    f" (fill notional {filled * price} + fee {fee})",
                )
        else:
            if position_quantity < filled:
                return self._reject(
                    order, FailureKind.POSITION_CONSTRAINT,
                    f"long-only: cannot sell {requested} shares of "
                    f"{order.instrument_id} when only {position_quantity} "
                    "are held",
                )
            if cash < fee - filled * price - _FP_TOL:
                # a sell whose net proceeds are negative (e.g. a fixed fee
                # larger than the proceeds) would drive cash below zero:
                # an implicit cash loan is never invented
                return self._reject(
                    order, FailureKind.INSUFFICIENT_CASH,
                    f"sell would leave cash negative: net proceeds "
                    f"{filled * price - fee} against cash {cash}",
                )

        fill = FillEvent(
            run_id=order.run_id,
            event_type=EventType.FILL,
            sequence=-1,
            session=None,
            timestamp=None,
            source="execution_simulator",
            config_ref=self._config.config_hash(),
            fill_id=order.order_id.replace("ORD-", "FILL-"),
            order_id=order.order_id,
            signal_id=order.signal_id,
            instrument_id=order.instrument_id,
            side=order.side,
            requested_quantity=requested,
            filled_quantity=filled,
            unfilled_quantity=unfilled,
            unfilled_reason=unfilled_reason,
            price=price,
            reference_price=reference,
            execution_price=self._exec.execution_price,
            spread_cost=spread_cost,
            slippage_cost=slippage_cost,
            fee=fee,
            price_basis=bar.price_basis,
        )
        return fill

    def liquidity_cap(self, bar: BarRecord | None) -> float:
        """The configured liquidity cap for a fill session (exposed for
        tests and diagnostics)."""
        if bar is None:
            return 0.0
        volume = bar.liquidity_volume
        if _missing_volume(volume):
            return 0.0
        cap = self._exec.participation_fraction * volume
        if self._exec.max_order_quantity is not None:
            cap = min(cap, self._exec.max_order_quantity)
        return cap

    # ------------------------------------------------------------ helpers

    def _reference_price(self, bar: BarRecord) -> float:
        if self._exec.execution_price == ExecutionPrice.OPEN:
            return bar.open
        return bar.close

    def _reject(
        self, order: OrderIntent, reason: FailureKind, detail: str
    ) -> RejectionEvent:
        return RejectionEvent(
            run_id=order.run_id,
            event_type=EventType.REJECTION,
            sequence=-1,
            session=None,
            timestamp=None,
            source="execution_simulator",
            config_ref=self._config.config_hash(),
            rejection_id=order.order_id.replace("ORD-", "REJ-"),
            order_id=order.order_id,
            signal_id=order.signal_id,
            instrument_id=order.instrument_id,
            side=order.side,
            requested_quantity=order.quantity,
            reason=reason,
            detail=detail,
        )


__all__ = ["ExecutionSimulator"]