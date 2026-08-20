"""Signal -> order intent translation (Phase 7).

The backtester consumes temporally valid signals and translates each into
an explicit `OrderIntent` through a deterministic generator. This is the
only place a signal becomes an order; the strategy layer of Phase 8 may
replace this generator, but the order/execution/accounting chain stays
unchanged.

Rebalancing semantics (documented, deterministic):

  LONG signal with target T:
      current position C
      T > C  -> BUY (T - C)
      T < C  -> SELL (C - T)
      T == C -> no order
  FLAT signal:
      C > 0  -> SELL C
      C == 0 -> no order

Sizing:
  - QUANTITY: `target` is whole shares;
  - WEIGHT:   `target` is a fraction of the equity at the previous
              valuation; the resulting share count is FLOORED to whole
              shares so the order can never over-spend the cash.

Timing: the order is submitted at the signal session's close, becomes
eligible `execution_delay` sessions later, and expires after
`order_expiry_sessions` sessions without an execution bar.
"""

from __future__ import annotations

import math
from datetime import date

from orbit.backtest.config import BacktestConfig, SizingPolicy
from orbit.backtest.clock import MarketEventClock
from orbit.backtest.events import (
    EventType,
    OrderIntent,
    OrderSide,
    OrderStatus,
    SignalDirection,
    SignalEvent,
)
from orbit.backtest.ledger import PortfolioLedger
from orbit.temporal.times import session_close_utc


class OrderGenerator:
    """Deterministic signal -> order-intent translation."""

    def __init__(self, config: BacktestConfig, clock: MarketEventClock):
        self._config = config
        self._clock = clock

    # ------------------------------------------------------------ public

    def generate(
        self,
        signal: SignalEvent,
        ledger: PortfolioLedger,
        last_equity: float,
        signal_close: float,
        order_index: int,
    ) -> OrderIntent | None:
        """One order intent for one signal, or None when the signal implies
        no trade (already at target / flat with no position)."""
        instrument = signal.instrument_id
        current = ledger.position_quantity(instrument)

        if signal.direction == SignalDirection.LONG:
            target = self._target_shares(signal, last_equity, signal_close)
            target = math.floor(target)
            if target > current:
                side = OrderSide.BUY
                quantity = target - current
            elif target < current:
                side = OrderSide.SELL
                quantity = current - target
            else:
                return None
        else:  # FLAT
            if current <= 0:
                return None
            side = OrderSide.SELL
            quantity = current

        submitted = signal.signal_session
        eligible = self._clock.next_session(
            submitted, self._config.execution.execution_delay
        )
        if eligible is None:
            # the delay pushes eligibility past the last session: the order
            # can never execute - no order is generated
            return None
        # the first session on which a fill can actually happen: fills are
        # processed BEFORE the signals of a session, so a delay-0 order is
        # submitted only after its own session's fill pass - its first fill
        # chance is the session AFTER submission. The expiry window is
        # measured from this true first fill chance, never from `eligible`
        # (a window measured from `eligible` would make every delay-0 order
        # born expired).
        if eligible == submitted:
            first_fill_chance = self._clock.next_session(submitted, 1)
            if first_fill_chance is None:
                return None
        else:
            first_fill_chance = eligible
        expiry = self._clock.next_session(
            first_fill_chance, self._config.execution.order_expiry_sessions
        )

        return OrderIntent(
            run_id=signal.run_id,
            event_type=EventType.ORDER,
            sequence=-1,
            session=submitted,
            timestamp=session_close_utc(submitted),
            source="order_generator",
            config_ref=self._config.config_hash(),
            order_id=f"ORD-{order_index:06d}",
            signal_id=signal.signal_id,
            instrument_id=instrument,
            side=side,
            quantity=quantity,
            submitted_session=submitted,
            eligible_session=eligible,
            expiry_session=expiry,
            status=OrderStatus.NEW,
        )

    # ------------------------------------------------------------ sizing

    def _target_shares(
        self, signal: SignalEvent, last_equity: float, signal_close: float
    ) -> float:
        if signal.target < 0:
            raise ValueError(
                f"signal {signal.signal_id}: a LONG target cannot be "
                f"negative ({signal.target})"
            )
        if self._config.sizing == SizingPolicy.QUANTITY:
            return float(signal.target)
        # WEIGHT sizing
        if signal.target > 1.0:
            raise ValueError(
                f"signal {signal.signal_id}: a WEIGHT target must be in "
                f"[0, 1], got {signal.target}"
            )
        if signal_close <= 0:
            raise ValueError(
                f"signal {signal.signal_id}: cannot size a WEIGHT order "
                f"against a non-positive signal close ({signal_close})"
            )
        return signal.target * last_equity / signal_close


__all__ = ["OrderGenerator"]