"""The canonical Phase 7 event model.

Every transition in the backtest chain is an observable, ordered event:

    MarketEvent  -> SignalEvent -> OrderIntent -> FillEvent/RejectionEvent
                 -> LedgerEvent (accounting) -> ValuationEvent -> OutcomeEvent

Design rules:

  - every event carries: run_id, event type, a deterministic `sequence`
    (the global emission order), the session it belongs to, a naive-UTC
    instant, the emitting `source`, and the configuration reference
    (`config_ref` - the run's config hash) so every event is traceable to
    the assumptions it ran under;
  - an OrderIntent is NOT a fill; a RejectionEvent is NOT a ledger change;
  - all events are frozen dataclasses; `as_dict()` is the canonical export
    shape used by the event stream (JSONL / polars frame).

The sequence number is assigned by the backtester in emission order, which
is fully deterministic (sorted sessions, sorted instruments, sorted signal
ids, sorted order ids). Two identical runs therefore produce byte-identical
event streams.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from typing import Any, Literal

import math

from orbit.backtest.config import ExecutionPrice


class EventType(str, Enum):
    RUN_START = "run_start"
    MARKET = "market"
    SIGNAL = "signal"
    ORDER = "order"
    FILL = "fill"
    REJECTION = "rejection"
    LEDGER = "ledger"
    VALUATION = "valuation"
    OUTCOME = "outcome"
    RUN_END = "run_end"


class SignalDirection(str, Enum):
    """The two signal directions Phase 7 understands. LONG targets a
    positive position; FLAT targets zero. There is no SHORT direction in
    the long-only MVP."""

    LONG = "long"
    FLAT = "flat"


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderStatus(str, Enum):
    """Lifecycle of an order intent. An OrderIntent is NOT a fill; its
    `status` field documents the submission-side snapshot and is always
    `new` at order creation. The true lifecycle state (filled / partially
    filled / rejected) is expressed by the FillEvent or RejectionEvent that
    follows submission - the status field is informational only and does
    not update automatically."""

    NEW = "new"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    REJECTED = "rejected"


class FailureKind(str, Enum):
    """The Phase 7 failure taxonomy. Every rejection/error names its kind -
    research diagnostics must never collapse every failure into a generic
    exception."""

    DATA_ERROR = "data_error"
    EXECUTION_ERROR = "execution_error"
    ACCOUNTING_ERROR = "accounting_error"
    CONFIGURATION_ERROR = "configuration_error"
    INVALID_ORDER = "invalid_order"
    INSUFFICIENT_CASH = "insufficient_cash"
    POSITION_CONSTRAINT = "position_constraint"
    LIQUIDITY_REJECTION = "liquidity_rejection"
    EXPIRED_ORDER = "expired_order"
    NO_EXECUTION_BAR = "no_execution_bar"
    UNKNOWN_INSTRUMENT = "unknown_instrument"
    DELISTED = "delisted"


def _iso(value: date | datetime | None) -> str | None:
    if value is None:
        return None
    return value.isoformat()


def _json_string(value: Any) -> str:
    import json

    return json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))


def _coerce(value: Any, enum_type: type[Enum]) -> Enum:
    """Coerce a string into `enum_type` at construction time: events are
    also constructed from normalized JSON (replay/round-trip), so a raw
    string must never survive into `as_dict()` as a broken enum access."""
    if isinstance(value, str) and not isinstance(value, enum_type):
        return enum_type(value)
    return value


@dataclass(frozen=True)
class Event:
    """Base of every event. Subclasses add a typed payload (keyword-only)."""

    run_id: str
    event_type: EventType
    sequence: int
    session: date | None
    timestamp: datetime | None
    source: str
    config_ref: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_type", _coerce(self.event_type, EventType))

    @property
    def event_id(self) -> str:
        return f"{self.run_id}-E{self.sequence:06d}"

    def _payload(self) -> dict[str, Any]:
        return {}

    def as_dict(self) -> dict[str, Any]:
        payload = {}
        for key, value in self._payload().items():
            if value is None:
                payload[key] = None
            elif isinstance(value, float) and not math.isfinite(value):
                # NaN/Inf in an event payload would poison JSONL and polars
                # frames: the canonical stream carries a null instead
                payload[key] = None
            elif isinstance(value, (str, int, float, bool)):
                payload[key] = value
            elif isinstance(value, (date, datetime)):
                payload[key] = value.isoformat()
            elif isinstance(value, Enum):
                payload[key] = value.value
            elif isinstance(value, (dict, list, tuple)):
                payload[key] = _json_string(value)
            else:
                payload[key] = _json_string(value)
        return {
            "event_id": self.event_id,
            "run_id": self.run_id,
            "event_type": self.event_type.value,
            "sequence": self.sequence,
            "session": _iso(self.session),
            "timestamp": _iso(self.timestamp),
            "source": self.source,
            "config_ref": self.config_ref,
            **payload,
        }


@dataclass(frozen=True, kw_only=True)
class RunStartEvent(Event):
    """The run's birth event: initial cash and universe."""

    initial_cash: float
    universe: tuple[str, ...]

    def _payload(self) -> dict[str, Any]:
        return {"initial_cash": self.initial_cash, "universe": self.universe}


@dataclass(frozen=True, kw_only=True)
class MarketEvent(Event):
    """One completed daily bar becomes available at the session close.

    `price_basis` documents the canonical Phase 3 split-continuous basis of
    the stored prices; `liquidity_volume` is the as-published volume when
    the corporate-actions artifact exists, else the stored (provider-basis)
    volume - a documented liquidity proxy.
    """

    instrument_id: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    liquidity_volume: float
    price_basis: str
    volume_basis: str

    def _payload(self) -> dict[str, Any]:
        return {
            "instrument_id": self.instrument_id,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "liquidity_volume": self.liquidity_volume,
            "price_basis": self.price_basis,
            "volume_basis": self.volume_basis,
        }


@dataclass(frozen=True, kw_only=True)
class SignalEvent(Event):
    """A strategy decision: LONG (target a position) or FLAT (target zero).

    The signal is generated at the session close of `signal_session` using
    that session's completed bar (the canonical EOD convention). `target`
    is the sizing input: whole shares under QUANTITY sizing, a weight in
    [0, 1] under WEIGHT sizing. `signal_metric` is the predicted value that
    motivated the signal (Phase 5 keeps 'predicted' and 'realized'
    strictly apart; the executed result is recorded separately on fills).
    """

    signal_id: str
    instrument_id: str
    signal_session: date
    decision_time: datetime
    direction: SignalDirection
    target: float
    signal_metric: float | None = None
    strategy_ref: str | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(
            self, "direction", _coerce(self.direction, SignalDirection)
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "signal_id": self.signal_id,
            "instrument_id": self.instrument_id,
            "signal_session": self.signal_session,
            "decision_time": self.decision_time,
            "direction": self.direction,
            "target": self.target,
            "signal_metric": self.signal_metric,
            "strategy_ref": self.strategy_ref,
        }


@dataclass(frozen=True, kw_only=True)
class OrderIntent(Event):
    """An explicit order intent - NOT a fill.

    Timings: the order is submitted at `submitted_session`'s close,
    becomes eligible at `eligible_session` (signal session + execution
    delay), and must fill no later than `expiry_session`. Phase 7
    implements market orders only; that is the single execution primitive
    of the MVP.
    """

    order_id: str
    signal_id: str
    instrument_id: str
    side: OrderSide
    quantity: float
    order_type: Literal["market"] = "market"
    submitted_session: date | None = None
    eligible_session: date | None = None
    expiry_session: date | None = None
    status: OrderStatus = OrderStatus.NEW

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(self, "side", _coerce(self.side, OrderSide))
        object.__setattr__(self, "status", _coerce(self.status, OrderStatus))

    def _payload(self) -> dict[str, Any]:
        return {
            "order_id": self.order_id,
            "signal_id": self.signal_id,
            "instrument_id": self.instrument_id,
            "side": self.side,
            "quantity": self.quantity,
            "order_type": self.order_type,
            "submitted_session": self.submitted_session,
            "eligible_session": self.eligible_session,
            "expiry_session": self.expiry_session,
            "status": self.status,
        }


@dataclass(frozen=True, kw_only=True)
class FillEvent(Event):
    """One executed (possibly partial) fill. The ledger applies ONLY the
    filled quantity; `unfilled_quantity`/`unfilled_reason` record the
    remainder explicitly - a partial fill is never silently converted into
    a complete one.

    Costs: `reference_price` is the bar field the execution semantics chose
    (open or close); `price` is the reference price adjusted by spread and
    slippage (direction-aware); `spread_cost` and `slippage_cost` are the
    per-fill dollar amounts of each assumption; `fee` is the commission.
    All recorded, all auditable.
    """

    fill_id: str
    order_id: str
    signal_id: str
    instrument_id: str
    side: OrderSide
    requested_quantity: float
    filled_quantity: float
    unfilled_quantity: float
    unfilled_reason: str | None
    price: float
    reference_price: float
    execution_price: ExecutionPrice
    spread_cost: float
    slippage_cost: float
    fee: float
    price_basis: str

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(self, "side", _coerce(self.side, OrderSide))
        object.__setattr__(
            self,
            "execution_price",
            _coerce(self.execution_price, ExecutionPrice),
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "fill_id": self.fill_id,
            "order_id": self.order_id,
            "signal_id": self.signal_id,
            "instrument_id": self.instrument_id,
            "side": self.side,
            "requested_quantity": self.requested_quantity,
            "filled_quantity": self.filled_quantity,
            "unfilled_quantity": self.unfilled_quantity,
            "unfilled_reason": self.unfilled_reason,
            "price": self.price,
            "reference_price": self.reference_price,
            "execution_price": self.execution_price,
            "spread_cost": self.spread_cost,
            "slippage_cost": self.slippage_cost,
            "fee": self.fee,
            "price_basis": self.price_basis,
        }


@dataclass(frozen=True, kw_only=True)
class RejectionEvent(Event):
    """An order that was NOT filled. A rejection never modifies the ledger
    (no phantom positions, no phantom cash)."""

    rejection_id: str
    order_id: str
    signal_id: str
    instrument_id: str
    side: OrderSide
    requested_quantity: float
    reason: FailureKind
    detail: str

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(self, "side", _coerce(self.side, OrderSide))
        object.__setattr__(self, "reason", _coerce(self.reason, FailureKind))

    def _payload(self) -> dict[str, Any]:
        return {
            "rejection_id": self.rejection_id,
            "order_id": self.order_id,
            "signal_id": self.signal_id,
            "instrument_id": self.instrument_id,
            "side": self.side,
            "requested_quantity": self.requested_quantity,
            "reason": self.reason,
            "detail": self.detail,
        }


@dataclass(frozen=True, kw_only=True)
class LedgerEvent(Event):
    """A full portfolio accounting snapshot after a session's fills.

    `equity` values positions at the session close. `fees_total`,
    `spread_cost_total`, `slippage_cost_total` are cumulative; `realized`
    is cumulative realized P&L (price-only, fees excluded), `unrealized`
    is the current mark-to-market. The accounting identity
    equity = initial_cash + realized + unrealized - fees_total is enforced
    by the ledger's invariant validator.
    """

    cash: float
    positions: dict[str, dict[str, float]]
    fees_total: float
    spread_cost_total: float
    slippage_cost_total: float
    realized: float
    unrealized: float
    equity: float

    def _payload(self) -> dict[str, Any]:
        return {
            "cash": self.cash,
            "positions": self.positions,
            "fees_total": self.fees_total,
            "spread_cost_total": self.spread_cost_total,
            "slippage_cost_total": self.slippage_cost_total,
            "realized": self.realized,
            "unrealized": self.unrealized,
            "equity": self.equity,
        }


@dataclass(frozen=True, kw_only=True)
class ValuationEvent(Event):
    """A deterministic portfolio valuation at a session close.

    `valuation_prices` are the close prices used (stored canonical basis);
    `stale` lists instruments whose last known price is older than the
    session (data gap - valued at the last available close, flagged, never
    silent). `benchmark_return` is the analytical benchmark comparison
    (computed separately; it never mutates portfolio accounting).
    """

    cash: float
    market_value: float
    equity: float
    realized: float
    unrealized: float
    valuation_prices: dict[str, float]
    stale: tuple[str, ...] = ()
    benchmark_value: float | None = None
    benchmark_return: float | None = None

    def _payload(self) -> dict[str, Any]:
        return {
            "cash": self.cash,
            "market_value": self.market_value,
            "equity": self.equity,
            "realized": self.realized,
            "unrealized": self.unrealized,
            "valuation_prices": self.valuation_prices,
            "stale": self.stale,
            "benchmark_value": self.benchmark_value,
            "benchmark_return": self.benchmark_return,
        }


@dataclass(frozen=True, kw_only=True)
class OutcomeEvent(Event):
    """An accounting outcome: either the per-signal executed summary (which
    keeps 'predicted signal' and 'simulated execution result' separate) or
    the final run result."""

    kind: Literal["signal_outcome", "run_result"]
    signal_id: str | None = None
    instrument_id: str | None = None
    direction: SignalDirection | None = None
    predicted_metric: float | None = None
    executed_quantity: float | None = None
    executed_notional: float | None = None
    total_fee: float | None = None
    final_quantity: float | None = None
    # run_result payload
    initial_cash: float | None = None
    final_cash: float | None = None
    final_equity: float | None = None
    total_return: float | None = None
    total_pnl: float | None = None
    total_fees: float | None = None
    total_spread_cost: float | None = None
    total_slippage_cost: float | None = None
    n_signals: int | None = None
    n_orders: int | None = None
    n_fills: int | None = None
    n_rejects: int | None = None
    turnover: float | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.direction is not None:
            object.__setattr__(
                self, "direction", _coerce(self.direction, SignalDirection)
            )

    def _payload(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "signal_id": self.signal_id,
            "instrument_id": self.instrument_id,
            "direction": self.direction,
            "predicted_metric": self.predicted_metric,
            "executed_quantity": self.executed_quantity,
            "executed_notional": self.executed_notional,
            "total_fee": self.total_fee,
            "final_quantity": self.final_quantity,
            "initial_cash": self.initial_cash,
            "final_cash": self.final_cash,
            "final_equity": self.final_equity,
            "total_return": self.total_return,
            "total_pnl": self.total_pnl,
            "total_fees": self.total_fees,
            "total_spread_cost": self.total_spread_cost,
            "total_slippage_cost": self.total_slippage_cost,
            "n_signals": self.n_signals,
            "n_orders": self.n_orders,
            "n_fills": self.n_fills,
            "n_rejects": self.n_rejects,
            "turnover": self.turnover,
        }


@dataclass(frozen=True, kw_only=True)
class RunEndEvent(Event):
    """The run's terminal event."""

    final_equity: float
    total_return: float

    def _payload(self) -> dict[str, Any]:
        return {"final_equity": self.final_equity, "total_return": self.total_return}


__all__ = [
    "Event",
    "EventType",
    "FailureKind",
    "FillEvent",
    "LedgerEvent",
    "MarketEvent",
    "OrderIntent",
    "OrderSide",
    "OrderStatus",
    "OutcomeEvent",
    "RejectionEvent",
    "RunEndEvent",
    "RunStartEvent",
    "SignalDirection",
    "SignalEvent",
    "ValuationEvent",
]