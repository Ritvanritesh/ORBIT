"""The Phase 7 backtester: deterministic, event-driven, replayable,
auditable and economically explicit.

The canonical chain per session:

    1. MarketEvent      - the session's completed bars become available at
                          the session close (Phase 4 instant);
    2. Fill/Rejection   - every eligible order is executed at the session's
                          execution instant (open for open fills, close for
                          close fills), in order-id order; carried orders
                          (missing execution bar) stay pending until expiry;
    3. SignalEvent      - the session's signals are processed at the close;
    4. OrderIntent      - signals translate into orders (deterministic
                          generator); orders become eligible delay sessions
                          later;
    5. LedgerEvent      - the accounting snapshot after the session's fills;
    6. ValuationEvent   - portfolio valuation at the close of the session
                          (equity = cash + market value of positions) plus
                          the separate benchmark comparison.

Timing conventions (documented, conservative, Phase 4-consistent):

  - a signal at the close of session D uses the completed bar of D and
    becomes an order at the same instant; the order is eligible at
    session D + execution_delay;
  - with the default execution (price=open, delay=1) the earliest fill is
    the NEXT session's open: the simulator never fills an order at the
    signal session's own close and never uses information unavailable at
    the decision instant;
  - within a session, eligible fills are processed before signal-driven
    order generation (deterministic, documented);
  - valuation prices are the stored closes of the session (canonical
    split-continuous basis); positions with no bar on the session are
    valued at their last available close and FLAGGED as stale - never
    silent;
  - an order that reaches the end of the run unfilled receives an explicit
    terminal rejection (EXPIRED_ORDER), so no order is ever left in an
    ambiguous state.

The backtester is deterministic by construction: no randomness, sorted
iteration everywhere, and every emitted event carries a global sequence
number in emission order.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import replace
from datetime import date, datetime, timezone
from typing import Any

import polars as pl

from orbit.backtest.clock import MarketEventClock
from orbit.backtest.config import BacktestConfig, ExecutionPrice
from orbit.backtest.events import (
    Event,
    EventType,
    FailureKind,
    FillEvent,
    LedgerEvent,
    MarketEvent,
    OrderIntent,
    OrderSide,
    OutcomeEvent,
    RejectionEvent,
    RunEndEvent,
    RunStartEvent,
    SignalDirection,
    SignalEvent,
    ValuationEvent,
)
from orbit.backtest.execution import ExecutionSimulator
from orbit.backtest.ledger import PortfolioLedger
from orbit.backtest.manifest import BacktestManifest, build_manifest
from orbit.backtest.orders import OrderGenerator
from orbit.temporal.times import normalize_instant, session_close_utc

ENGINE_VERSION = "v1.0.0"

_BAR_REQUIRED = {
    "instrument_id", "trade_date", "open", "high", "low", "close", "volume",
}


def _iso_utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Backtester:
    """Runs one deterministic backtest over bars + signals."""

    def __init__(
        self,
        config: BacktestConfig,
        *,
        universe: list[str],
        dataset_snapshot_ids: list[str],
        code_hash: str,
        experiment_id: str | None = None,
        hypothesis_id: str | None = None,
        feature_refs: list[dict[str, Any]] | None = None,
        model: dict[str, Any] | None = None,
        label_id: str | None = None,
        label_version: str | None = None,
        temporal_config_digest: str | None = None,
        cost_model_id: str | None = None,
        created_at: datetime | None = None,
    ):
        if not universe:
            raise ValueError("universe must not be empty")
        self._config = config
        self._universe = sorted(set(universe))
        self._dataset_snapshot_ids = sorted(set(dataset_snapshot_ids))
        if not self._dataset_snapshot_ids:
            raise ValueError(
                "a backtest must pin exact dataset snapshot ids (DS-xxxxxx); "
                "an unreferenced backtest is not a research result"
            )
        self._code_hash = code_hash
        self._manifest_meta = dict(
            experiment_id=experiment_id,
            hypothesis_id=hypothesis_id,
            feature_refs=feature_refs or [],
            model=model,
            label_id=label_id,
            label_version=label_version,
            temporal_config_digest=temporal_config_digest,
            cost_model_id=cost_model_id,
            created_at=created_at or _iso_utc_now(),
        )

    # ------------------------------------------------------------ public

    def run(
        self,
        bars: pl.DataFrame,
        signals: Any,
        *,
        events_artifact: pl.DataFrame | None = None,
        volume_basis: str | None = None,
    ) -> "BacktestResult":
        """Execute the backtest and return the full result object.

        `bars` is the canonical normalized frame (instrument_id,
        trade_date, open, high, low, close, volume). `signals` is a polars
        frame or list of dicts (see MarketEventClock.normalize_signals).
        """
        missing = _BAR_REQUIRED - set(bars.columns)
        if missing:
            raise ValueError(
                "the backtester requires the canonical normalized bar "
                f"columns; missing: {sorted(missing)}"
            )
        clock = MarketEventClock(bars, events=events_artifact, volume_basis=volume_basis)
        signal_rows = clock.normalize_signals(signals)
        signal_events = self._build_signal_events(signal_rows)

        config = self._config
        window = clock.window_sessions(config.window_start, config.window_end)
        if not window:
            raise ValueError(
                "the evaluation window contains no market sessions - nothing "
                "can be simulated"
            )

        unknown_signals = [
            s.signal_id
            for s in signal_events
            if s.signal_session not in window
        ]
        if unknown_signals:
            raise ValueError(
                "signals outside the evaluation window are refused (a "
                "silently dropped signal is a silent result): "
                + ", ".join(sorted(unknown_signals[:5]))
            )

        manifest = self._build_manifest(clock, signal_rows)
        run_id = manifest.run_id
        # stamp the run identity onto the signal events (they were built
        # before the run_id was derivable)
        signal_events = [replace(s, run_id=run_id) for s in signal_events]

        ledger = PortfolioLedger(config.initial_cash)
        executor = ExecutionSimulator(config)
        generator = OrderGenerator(config, clock)
        universe = set(self._universe)

        signals_by_session: dict[date, list[SignalEvent]] = {}
        for s in signal_events:
            signals_by_session.setdefault(s.signal_session, []).append(s)
        for s in signals_by_session.values():
            # canonical intra-session order: a reordered signal input is
            # the same signal set and must produce the same stream
            s.sort(key=lambda e: e.signal_id)
            s.sort(key=lambda e: e.signal_id)

        events: list[Event] = []
        pending_orders: dict[str, OrderIntent] = {}
        last_equity = config.initial_cash
        last_close_by_instrument: dict[str, float] = {}
        sequence = 0

        def emit(event: Event) -> None:
            nonlocal sequence
            object.__setattr__(event, "sequence", sequence)
            sequence += 1
            events.append(event)

        emit(
            RunStartEvent(
                run_id=run_id,
                event_type=EventType.RUN_START,
                sequence=-1,
                session=None,
                timestamp=None,
                source="backtester",
                config_ref=config.config_hash(),
                initial_cash=config.initial_cash,
                universe=tuple(self._universe),
            )
        )

        # the benchmark reference: its last close STRICTLY BEFORE the first
        # window session (analytical comparison only, never a ledger event).
        # The portfolio baseline is cash at run start (before window[0]
        # opens), so the benchmark baseline must be the last close before
        # window[0] - an inclusive reference would anchor the comparison at
        # window[0]'s own close and zero out the benchmark's first session.
        benchmark_ref: float | None = None
        if config.benchmark is not None:
            if config.benchmark not in clock.instruments():
                raise ValueError(
                    f"benchmark {config.benchmark} not in the backtest universe"
                )
            benchmark_ref = clock.last_close(config.benchmark, window[0])
            # benchmark_ref may be None if no bar exists strictly before window[0];
            # the benchmark is analytical only, never a ledger transaction

        order_counter = 0

        for session in window:
            session_instruments = sorted(
                i for i in universe if clock.has_bar(i, session)
            )

            # ---- 1. market events (bars available at the session close)
            for instrument in session_instruments:
                bar = clock.bar(instrument, session)
                emit(
                    MarketEvent(
                        run_id=run_id,
                        event_type=EventType.MARKET,
                        sequence=-1,
                        session=session,
                        timestamp=clock.availability_instant(session),
                        source="market_clock",
                        config_ref=config.config_hash(),
                        instrument_id=instrument,
                        open=bar.open,
                        high=bar.high,
                        low=bar.low,
                        close=bar.close,
                        volume=bar.volume,
                        liquidity_volume=bar.liquidity_volume,
                        price_basis=bar.price_basis,
                        volume_basis=bar.volume_basis,
                    )
                )
                if bar.close is not None and bar.close > 0:
                    last_close_by_instrument[instrument] = bar.close

            # ---- 2. execute eligible orders at the session's execution
            # ----    instant. Cash-generating SELL fills are processed
            # ----    before cash-consuming BUY fills (each group in
            # ----    order-id order): within one execution instant the
            # ----    portfolio can recycle the cash a same-instant sell
            # ----    generates, so the outcome never depends on the
            # ----    arbitrary signal-id namespace.
            fill_timestamp = clock.execution_instant(
                session,
                at_open=config.execution.execution_price == ExecutionPrice.OPEN,
            )
            eligible_now = sorted(
                order_id
                for order_id, order in pending_orders.items()
                if order.eligible_session is not None
                and order.eligible_session <= session
            )
            # sides are snapshotted BEFORE any pass: a sell fill deletes its
            # order from pending_orders, but the buy pass must still know
            # the sides of all orders that were eligible at this instant
            side_of = {oid: pending_orders[oid].side for oid in eligible_now}

            def _attempt(order_id: str) -> None:
                order = pending_orders[order_id]
                # an order whose expiry window has passed is rejected -
                # regardless of whether an execution bar exists: the bar
                # resuming after the expiry window can never revive it
                if self._expired(order, session):
                    emit(
                        _expiry_rejection(
                            run_id, order, fill_timestamp, config, session,
                            window[-1],
                        )
                    )
                    del pending_orders[order_id]
                    return
                bar = clock.bar(order.instrument_id, session)
                outcome = executor.execute(
                    order,
                    bar,
                    cash=ledger.cash,
                    position_quantity=ledger.position_quantity(
                        order.instrument_id
                    ),
                )
                if isinstance(outcome, FillEvent):
                    outcome = outcome_with_identity(
                        outcome, run_id, session, fill_timestamp
                    )
                    emit(outcome)
                    ledger.apply_fill(outcome)
                    del pending_orders[order_id]
                elif isinstance(outcome, RejectionEvent):
                    if outcome.reason == FailureKind.NO_EXECUTION_BAR:
                        # no bar on this session: carry the order until
                        # expiry (documented single-session-fill semantics
                        # with an explicit carry window)
                        return
                    outcome = outcome_with_identity(
                        outcome, run_id, session, fill_timestamp
                    )
                    emit(outcome)
                    del pending_orders[order_id]

            for order_id in eligible_now:
                if side_of[order_id] == OrderSide.SELL:
                    _attempt(order_id)
            for order_id in eligible_now:
                if side_of[order_id] == OrderSide.BUY:
                    _attempt(order_id)

            # ---- 3. signals at the session close
            for signal in signals_by_session.get(session, []):
                emit(
                    replace(
                        signal,
                        sequence=-1,
                        timestamp=signal.decision_time,
                        source="strategy",
                        config_ref=config.config_hash(),
                    )
                )

            # ---- 4. signals -> order intents (deterministic generator)
            for signal in signals_by_session.get(session, []):
                bar = clock.bar(signal.instrument_id, session)
                signal_close = bar.close if bar is not None else None
                if signal_close is None or not (
                    isinstance(signal_close, (int, float))
                    and signal_close > 0
                ):
                    # a signal whose anchor bar is missing or defective
                    # cannot be sized; it is a data error, never a guess
                    emit(
                        _data_error_event(
                            run_id, session, config, signal,
                            "the signal session bar is missing or has a "
                            "defective close",
                        )
                    )
                    continue
                order = generator.generate(
                    signal,
                    ledger,
                    last_equity=last_equity,
                    signal_close=float(signal_close),
                    order_index=order_counter,
                )
                if order is not None:
                    order_counter += 1
                    emit(order)
                    pending_orders[order.order_id] = order

            # ---- 5. ledger snapshot + valuation at the session close
            held = sorted(
                i for i in universe if ledger.position_quantity(i) > 0
            )
            valuation_prices = {
                instrument: self._valuation_price(
                    clock, instrument, session, last_close_by_instrument
                )
                for instrument in held
            }
            stale = tuple(
                i
                for i in held
                if not _has_usable_close(clock.bar(i, session))
            )
            snap = ledger.value(valuation_prices)

            benchmark_value = None
            benchmark_return = None
            if config.benchmark is not None and benchmark_ref:
                bench_close = clock.last_close(config.benchmark, session)
                if bench_close is not None:
                    benchmark_value = bench_close
                    benchmark_return = bench_close / benchmark_ref - 1.0

            emit(
                LedgerEvent(
                    run_id=run_id,
                    event_type=EventType.LEDGER,
                    sequence=-1,
                    session=session,
                    timestamp=clock.availability_instant(session),
                    source="ledger",
                    config_ref=config.config_hash(),
                    cash=snap["cash"],
                    positions={
                        k: {
                            "quantity": v["quantity"],
                            "avg_cost": v["avg_cost"],
                        }
                        for k, v in snap["positions"].items()
                    },
                    fees_total=ledger.fees_total,
                    spread_cost_total=ledger.spread_cost_total,
                    slippage_cost_total=ledger.slippage_cost_total,
                    realized=snap["realized"],
                    unrealized=snap["unrealized"],
                    equity=snap["equity"],
                )
            )
            emit(
                ValuationEvent(
                    run_id=run_id,
                    event_type=EventType.VALUATION,
                    sequence=-1,
                    session=session,
                    timestamp=clock.availability_instant(session),
                    source="ledger",
                    config_ref=config.config_hash(),
                    cash=snap["cash"],
                    market_value=snap["market_value"],
                    equity=snap["equity"],
                    realized=snap["realized"],
                    unrealized=snap["unrealized"],
                    valuation_prices=valuation_prices,
                    stale=stale,
                    benchmark_value=benchmark_value,
                    benchmark_return=benchmark_return,
                )
            )
            last_equity = snap["equity"]

        # ---- terminal events: expire unfilled orders explicitly, emit the
        # ---- per-signal outcomes and the run result
        run_end_instant = clock.availability_instant(window[-1])
        for order_id in sorted(pending_orders):
            emit(
                _expiry_rejection(
                    run_id, pending_orders[order_id], run_end_instant,
                    config, window[-1], window[-1],
                )
            )

        fills = [e for e in events if isinstance(e, FillEvent)]
        rejects = [e for e in events if isinstance(e, RejectionEvent)]
        orders = [e for e in events if isinstance(e, OrderIntent)]
        emitted_signals = [e for e in events if isinstance(e, SignalEvent)]
        valuations = [e for e in events if isinstance(e, ValuationEvent)]

        # per-signal outcomes, emitted in canonical (signal_session,
        # signal_id) order - the same canonical order the signal-set hash
        # uses, so a reordered signal input yields an identical stream
        for signal in sorted(signal_events, key=lambda s: (s.signal_session, s.signal_id)):
            signal_fills = [f for f in fills if f.signal_id == signal.signal_id]
            executed_quantity = sum(f.filled_quantity for f in signal_fills)
            executed_notional = sum(f.filled_quantity * f.price for f in signal_fills)
            total_fee = sum(f.fee for f in signal_fills)
            # quantity attributable to this signal alone: the signed position
            # deltas of its fills, evaluated in stream (emission) order - the
            # instrument's end-of-run position is not a per-signal quantity
            final_quantity = sum(
                f.filled_quantity if f.side == OrderSide.BUY else -f.filled_quantity
                for f in signal_fills
            )
            emit(
                OutcomeEvent(
                    run_id=run_id,
                    event_type=EventType.OUTCOME,
                    sequence=-1,
                    session=window[-1],
                    timestamp=run_end_instant,
                    source="backtester",
                    config_ref=config.config_hash(),
                    kind="signal_outcome",
                    signal_id=signal.signal_id,
                    instrument_id=signal.instrument_id,
                    direction=signal.direction,
                    predicted_metric=signal.signal_metric,
                    executed_quantity=executed_quantity,
                    executed_notional=executed_notional,
                    total_fee=total_fee,
                    final_quantity=final_quantity,
                )
            )

        final_equity = valuations[-1].equity if valuations else config.initial_cash
        total_return = final_equity / config.initial_cash - 1.0
        total_pnl = final_equity - config.initial_cash
        turnover = ledger.traded_notional / config.initial_cash

        emit(
            OutcomeEvent(
                run_id=run_id,
                event_type=EventType.OUTCOME,
                sequence=-1,
                session=window[-1],
                timestamp=run_end_instant,
                source="backtester",
                config_ref=config.config_hash(),
                kind="run_result",
                initial_cash=config.initial_cash,
                final_cash=ledger.cash,
                final_equity=final_equity,
                total_return=total_return,
                total_pnl=total_pnl,
                total_fees=ledger.fees_total,
                total_spread_cost=ledger.spread_cost_total,
                total_slippage_cost=ledger.slippage_cost_total,
                n_signals=len(emitted_signals),
                n_orders=len(orders),
                n_fills=len(fills),
                n_rejects=len(rejects),
                turnover=turnover,
            )
        )
        emit(
            RunEndEvent(
                run_id=run_id,
                event_type=EventType.RUN_END,
                sequence=-1,
                session=window[-1],
                timestamp=run_end_instant,
                source="backtester",
                config_ref=config.config_hash(),
                final_equity=final_equity,
                total_return=total_return,
            )
        )

        return BacktestResult(
            manifest=manifest,
            events=events,
            ledger_snapshot={
                "cash": ledger.cash,
                "positions": {
                    k: v.as_dict() for k, v in sorted(ledger.positions.items())
                },
            },
            last_equity=final_equity,
        )

    # ------------------------------------------------------------ helpers

    def _build_signal_events(self, rows: list[dict[str, Any]]) -> list[SignalEvent]:
        """Convert normalized signal rows into SignalEvents, validating the
        temporal convention: the decision instant must be the session close
        of the signal session (a signal stamped before its own session
        close could have used a bar that was not yet available)."""
        universe = set(self._universe)
        seen_ids: set[str] = set()
        out: list[SignalEvent] = []
        for r in rows:
            if r["signal_id"] in seen_ids:
                raise ValueError(
                    f"duplicate signal_id {r['signal_id']}: every signal must "
                    "have a unique id (a duplicated id would conflate "
                    "distinct decisions downstream)"
                )
            seen_ids.add(r["signal_id"])
            if r["instrument_id"] not in universe:
                raise ValueError(
                    f"signal {r['signal_id']}: instrument "
                    f"{r['instrument_id']} is not in the backtest universe "
                    "(a signal for an untradable name is refused)"
                )
            direction = SignalDirection(r["direction"])
            if r["target"] < 0:
                raise ValueError(
                    f"signal {r['signal_id']}: target cannot be negative"
                )
            decision_time = normalize_instant(r["decision_time"])
            expected = session_close_utc(r["signal_session"])
            if decision_time is not None and decision_time != expected:
                raise ValueError(
                    f"signal {r['signal_id']}: decision_time "
                    f"{decision_time.isoformat()} is not the session close "
                    f"{expected.isoformat()} of its signal_session "
                    f"{r['signal_session']} - a signal that is not stamped at "
                    "its session close could have used information that was "
                    "not yet available"
                )
            out.append(
                SignalEvent(
                    run_id="",
                    event_type=EventType.SIGNAL,
                    sequence=-1,
                    session=r["signal_session"],
                    timestamp=decision_time or expected,
                    source="strategy",
                    config_ref=None,
                    signal_id=r["signal_id"],
                    instrument_id=r["instrument_id"],
                    signal_session=r["signal_session"],
                    decision_time=decision_time or expected,
                    direction=direction,
                    target=float(r["target"]),
                    signal_metric=r["signal_metric"],
                    strategy_ref=r["strategy_ref"],
                )
            )
        return out

    def _build_manifest(
        self, clock: MarketEventClock, signal_rows: list[dict[str, Any]]
    ) -> BacktestManifest:
        window_start = self._config.window_start
        window_end = self._config.window_end
        if window_start is None or window_end is None:
            sessions = clock.sessions()
            window_start = window_start or sessions[0]
            window_end = window_end or sessions[-1]
        return build_manifest(
            config=self._config,
            engine_version=ENGINE_VERSION,
            universe=self._universe,
            liquidity_volume_basis=clock.volume_basis,
            dataset_snapshot_ids=self._dataset_snapshot_ids,
            code_hash=self._code_hash,
            config_hash=self._config.config_hash(),
            signal_set_hash=_signal_set_hash(signal_rows),
            **self._manifest_meta,
        )

    def _valuation_price(
        self,
        clock: MarketEventClock,
        instrument: str,
        session: date,
        last_close_by_instrument: dict[str, float],
    ) -> float:
        bar = clock.bar(instrument, session)
        if bar is not None and bar.close is not None and bar.close > 0:
            return bar.close
        # data gap: value at the last available close (flagged as stale in
        # the ValuationEvent; never silently invented)
        price = last_close_by_instrument.get(instrument)
        if price is None:
            price = clock.last_close(instrument, session)
        if price is None:
            raise ValueError(
                f"no valuation price exists for held instrument {instrument} "
                "at any session at or before the valuation session"
            )
        return price

    @staticmethod
    def _expired(order: OrderIntent, session: date) -> bool:
        if order.expiry_session is None:
            return False
        return session > order.expiry_session


# ------------------------------------------------------------ module glue


def _signal_set_hash(rows: list[dict[str, Any]]) -> str:
    """sha256 over the canonical signal set (a signal change is a different
    run, never a silent overwrite). The canonical form is sorted by
    (signal_session, signal_id), so a pure reordering of the same signal
    rows is the same run identity."""
    canonical = [
        {
            "signal_id": r["signal_id"],
            "instrument_id": r["instrument_id"],
            "signal_session": r["signal_session"].isoformat(),
            "direction": r["direction"],
            "target": r["target"],
            "signal_metric": (
                r["signal_metric"].isoformat()
                if isinstance(r["signal_metric"], (date, datetime))
                else r["signal_metric"]
            ),
            "strategy_ref": r["strategy_ref"],
        }
        for r in rows
    ]
    canonical.sort(key=lambda s: (s["signal_session"], s["signal_id"]))
    raw = json.dumps(
        canonical, sort_keys=True, default=str, separators=(",", ":")
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _has_usable_close(bar: Any) -> bool:
    """A close that can be valued against: present, finite and positive.
    NaN fails every comparison, so a defective close is never mistaken
    for a usable one."""
    if bar is None:
        return False
    close = bar.close
    return isinstance(close, (int, float)) and math.isfinite(close) and close > 0


def _data_error_event(
    run_id: str, session: date, config: BacktestConfig, signal: SignalEvent, detail: str
) -> RejectionEvent:
    """A signal with a defective anchor bar produces a loud data-error
    rejection record (never a silent skip)."""
    return RejectionEvent(
        run_id=run_id,
        event_type=EventType.REJECTION,
        sequence=-1,
        session=session,
        timestamp=session_close_utc(session),
        source="order_generator",
        config_ref=config.config_hash(),
        rejection_id=f"REJ-DATA-{signal.signal_id}",
        order_id="",
        signal_id=signal.signal_id,
        instrument_id=signal.instrument_id,
        side=OrderSide.BUY,
        requested_quantity=0.0,
        reason=FailureKind.DATA_ERROR,
        detail=detail,
    )


def _expiry_rejection(
    run_id: str,
    order: OrderIntent,
    instant: datetime,
    config: BacktestConfig,
    session: date,
    window_end: date,
) -> RejectionEvent:
    # an order whose eligible session lies beyond the run window never got
    # a fill chance: the rejection is the run's terminal record, not an
    # expiry-window verdict - the reason detail must say so
    never_eligible = (
        order.eligible_session is None or order.eligible_session > window_end
    )
    return RejectionEvent(
        run_id=run_id,
        event_type=EventType.REJECTION,
        sequence=-1,
        session=session,
        timestamp=instant,
        source="backtester",
        config_ref=config.config_hash(),
        rejection_id=order.order_id.replace("ORD-", "REJ-"),
        order_id=order.order_id,
        signal_id=order.signal_id,
        instrument_id=order.instrument_id,
        side=order.side,
        requested_quantity=order.quantity,
        reason=FailureKind.EXPIRED_ORDER,
        detail=(
            "the run ended before the order could fill (never filled, "
            "never silently dropped)"
            if never_eligible
            else "the order's expiry window passed before an execution bar "
            "existed (never filled, never silently dropped)"
        ),
    )


def outcome_with_identity(
    outcome: FillEvent | RejectionEvent,
    run_id: str,
    session: date,
    timestamp: datetime,
) -> FillEvent | RejectionEvent:
    """Stamp a pure simulator outcome with its run identity, session and
    execution instant (the simulator stays pure; identity is assigned by
    the backtester)."""
    return type(outcome)(
        run_id=run_id,
        event_type=outcome.event_type,
        sequence=outcome.sequence,
        session=session,
        timestamp=timestamp,
        source=outcome.source,
        config_ref=outcome.config_ref,
        **{
            name: getattr(outcome, name)
            for name in type(outcome).__dataclass_fields__
            if name
            not in {
                "run_id", "event_type", "sequence", "session", "timestamp",
                "source", "config_ref",
            }
        },
    )


# imported at the bottom to avoid a circular import (BacktestResult needs
# the result module which imports nothing from backtester)
from orbit.backtest.result import BacktestResult  # noqa: E402


__all__ = ["ENGINE_VERSION", "Backtester"]