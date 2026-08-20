"""The structured output of a Phase 7 backtest run.

A completed run produces: the run manifest, the full auditable event
stream, fills/orders/rejections/valuations, position and cash history,
and a summary of diagnostics. Every output is deterministic and exportable
(JSONL / parquet / JSON), and `equals()` is the exact replay comparison
(two identical runs compare equal event-for-event, fill-for-fill).

The summary deliberately stays small: Phase 8 builds the sophisticated
strategy metrics. Phase 7 reports execution and accounting facts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl

from orbit.backtest.events import (
    Event,
    EventType,
    FillEvent,
    LedgerEvent,
    OutcomeEvent,
    RejectionEvent,
    RunEndEvent,
    RunStartEvent,
    SignalEvent,
    ValuationEvent,
)
from orbit.backtest.manifest import BacktestManifest

_FP_TOL = 1e-9


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= _FP_TOL * max(1.0, abs(a), abs(b))


class BacktestResult:
    """One completed backtest run: manifest + events + derived histories."""

    def __init__(
        self,
        *,
        manifest: BacktestManifest,
        events: list[Event],
        ledger_snapshot: dict[str, Any],
        last_equity: float,
    ):
        self.manifest = manifest
        self._events = list(events)
        self._ledger_snapshot = ledger_snapshot
        self._last_equity = last_equity

    # ------------------------------------------------------------- access

    @property
    def run_id(self) -> str:
        return self.manifest.run_id

    @property
    def events(self) -> list[Event]:
        return list(self._events)

    def events_of(self, event_type: EventType | str) -> list[Event]:
        if isinstance(event_type, str):
            event_type = EventType(event_type)
        return [e for e in self._events if e.event_type == event_type]

    @property
    def fills(self) -> list[FillEvent]:
        return [e for e in self._events if isinstance(e, FillEvent)]

    @property
    def rejections(self) -> list[RejectionEvent]:
        return [e for e in self._events if isinstance(e, RejectionEvent)]

    @property
    def orders(self) -> list[Event]:
        return self.events_of(EventType.ORDER)

    @property
    def signals(self) -> list[SignalEvent]:
        return [e for e in self._events if isinstance(e, SignalEvent)]

    @property
    def valuations(self) -> list[ValuationEvent]:
        return [e for e in self._events if isinstance(e, ValuationEvent)]

    @property
    def ledger_snapshots(self) -> list[LedgerEvent]:
        return [e for e in self._events if isinstance(e, LedgerEvent)]

    def run_start(self) -> RunStartEvent | None:
        for e in self._events:
            if isinstance(e, RunStartEvent):
                return e
        return None

    def run_end(self) -> RunEndEvent | None:
        for e in self._events:
            if isinstance(e, RunEndEvent):
                return e
        return None

    def final_position(self, instrument_id: str) -> float:
        return self._ledger_snapshot["positions"].get(instrument_id, {}).get(
            "quantity", 0.0
        )

    def equity_curve(self) -> pl.DataFrame:
        """session, cash, market_value, equity, realized, unrealized,
        benchmark_return (where configured)."""
        rows = []
        for v in self.valuations:
            rows.append(
                {
                    "session": v.session,
                    "cash": v.cash,
                    "market_value": v.market_value,
                    "equity": v.equity,
                    "realized": v.realized,
                    "unrealized": v.unrealized,
                    "benchmark_return": v.benchmark_return,
                }
            )
        return pl.DataFrame(
            rows,
            schema={
                "session": pl.Date,
                "cash": pl.Float64,
                "market_value": pl.Float64,
                "equity": pl.Float64,
                "realized": pl.Float64,
                "unrealized": pl.Float64,
                "benchmark_return": pl.Float64,
            },
        )

    def position_history(self) -> pl.DataFrame:
        """session, instrument_id, quantity, avg_cost, price, market_value,
        unrealized - one row per held position per valuation."""
        snapshots = {e.session: e for e in self.ledger_snapshots}
        rows = []
        for v in self.valuations:
            snap = snapshots.get(v.session)
            positions = snap.positions if snap else {}
            for instrument, pos in sorted(positions.items()):
                price = v.valuation_prices.get(instrument)
                if price is None:
                    continue
                rows.append(
                    {
                        "session": v.session,
                        "instrument_id": instrument,
                        "quantity": pos["quantity"],
                        "avg_cost": pos["avg_cost"],
                        "price": price,
                        "market_value": pos["quantity"] * price,
                        "unrealized": pos["quantity"] * (price - pos["avg_cost"]),
                    }
                )
        return pl.DataFrame(rows)

    def cash_history(self) -> pl.DataFrame:
        rows = [
            {"session": e.session, "cash": e.cash, "equity": e.equity}
            for e in self.ledger_snapshots
        ]
        return pl.DataFrame(
            rows,
            schema={
                "session": pl.Date,
                "cash": pl.Float64,
                "equity": pl.Float64,
            },
        )

    def events_frame(self) -> pl.DataFrame:
        """The full event stream as a columnar frame (the auditable export)."""
        return pl.DataFrame([e.as_dict() for e in self._events])

    # ------------------------------------------------------------ summary

    def summary(self) -> dict[str, Any]:
        fills = self.fills
        rejects = self.rejections
        orders = self.orders
        run_outcome = None
        for e in self._events:
            if isinstance(e, OutcomeEvent) and e.kind == "run_result":
                run_outcome = e
        result: dict[str, Any] = {
            "run_id": self.run_id,
            "final_equity": run_outcome.final_equity if run_outcome else self._last_equity,
            "total_return": (
                run_outcome.total_return if run_outcome else None
            ),
            "total_pnl": run_outcome.total_pnl if run_outcome else None,
            "turnover": run_outcome.turnover if run_outcome else 0.0,
            "total_fees": run_outcome.total_fees if run_outcome else 0.0,
            "total_spread_cost": (
                run_outcome.total_spread_cost if run_outcome else 0.0
            ),
            "total_slippage_cost": (
                run_outcome.total_slippage_cost if run_outcome else 0.0
            ),
            "n_signals": run_outcome.n_signals if run_outcome else 0,
            "n_orders": run_outcome.n_orders if run_outcome else len(orders),
            "n_fills": run_outcome.n_fills if run_outcome else len(fills),
            "n_rejects": run_outcome.n_rejects if run_outcome else len(rejects),
            "n_unfilled_partial": sum(
                1 for f in fills if f.unfilled_quantity > 0
            ),
            "final_cash": (
                run_outcome.final_cash if run_outcome else self._ledger_snapshot["cash"]
            ),
            "positions": {
                k: dict(v) for k, v in self._ledger_snapshot["positions"].items()
            },
            "invariant_violations": self.invariant_violations(),
        }
        return result

    def invariant_violations(self) -> list[str]:
        """Replay of the accounting invariants over the run's own events:
        cash conservation, position conservation, equity identity and
        realized/unrealized consistency, re-derived from the event stream
        (a tampered or buggy ledger shows up here)."""
        violations: list[str] = []
        initial_cash = None
        for e in self._events:
            if isinstance(e, RunStartEvent):
                initial_cash = e.initial_cash
        if initial_cash is None:
            violations.append("no run_start event; the run is not a valid "
                              "research record")
            return violations

        cash = initial_cash
        positions: dict[str, float] = {}
        realized = 0.0
        fees = 0.0
        for e in self._events:
            if isinstance(e, FillEvent):
                notional = e.filled_quantity * e.price
                if e.side.value == "buy":
                    cash -= notional + e.fee
                    positions[e.instrument_id] = (
                        positions.get(e.instrument_id, 0.0) + e.filled_quantity
                    )
                else:
                    cash += notional - e.fee
                    positions[e.instrument_id] = (
                        positions.get(e.instrument_id, 0.0) - e.filled_quantity
                    )
                    if positions[e.instrument_id] < 0:
                        violations.append(
                            f"fill {e.fill_id} drives the position of "
                            f"{e.instrument_id} negative (a sell beyond the "
                            "held position reached the ledger)"
                        )
                fees += e.fee

        last_snapshot = self.ledger_snapshots[-1] if self.ledger_snapshots else None
        if last_snapshot is not None:
            if not _close(cash, last_snapshot.cash):
                violations.append(
                    f"cash conservation: the event stream implies {cash} but "
                    f"the final ledger snapshot holds {last_snapshot.cash}"
                )
            if cash < -_FP_TOL:
                violations.append(
                    f"the event stream leaves cash negative ({cash}): an "
                    "implicit cash loan is never part of long-only accounting"
                )
            expected_positions = {k: q for k, q in positions.items() if q > 0}
            snapshot_q = {
                k: v["quantity"] for k, v in last_snapshot.positions.items()
            }
            if expected_positions != snapshot_q:
                violations.append(
                    "position conservation: the event stream positions "
                    f"{expected_positions} do not match the final snapshot "
                    f"{snapshot_q}"
                )
            if not _close(fees, last_snapshot.fees_total):
                violations.append(
                    "fee accounting: the event stream implies "
                    f"{fees} but the snapshot holds {last_snapshot.fees_total}"
                )
            identity = (
                initial_cash
                + last_snapshot.realized
                + last_snapshot.unrealized
                - last_snapshot.fees_total
            )
            if not _close(identity, last_snapshot.equity):
                violations.append(
                    "equity identity broken: initial + realized + unrealized "
                    f"- fees = {identity} but equity = {last_snapshot.equity}"
                )
        return violations

    # -------------------------------------------------------------- export

    def to_jsonl(self, path: str | Path) -> Path:
        """Export the event stream as JSONL (the auditable replay artifact)."""
        p = Path(path)
        with open(p, "w", encoding="utf-8") as f:
            for e in self._events:
                f.write(json.dumps(e.as_dict(), sort_keys=True) + "\n")
        return p

    def to_dict(self) -> dict[str, Any]:
        """Full serializable result (manifest + summary + event stream)."""
        return {
            "manifest": json.loads(
                self.manifest.model_dump_json(exclude={"created_at"})
            ),
            "summary": self.summary(),
            "events": [e.as_dict() for e in self._events],
        }

    # --------------------------------------------------------------- replay

    def equals(self, other: "BacktestResult") -> bool:
        """Exact replay comparison: manifests (excluding wall-clock),
        summaries and the full event streams must be identical. Only
        documented floating-point tolerances are allowed (none here: the
        event stream is compared field-exactly)."""
        if not isinstance(other, BacktestResult):
            return False
        if self.manifest.compute_content_hash() != other.manifest.compute_content_hash():
            return False
        if self.manifest.run_id != other.manifest.run_id:
            return False
        if self.events_frame().write_json() != other.events_frame().write_json():
            return False
        if json.dumps(self.summary(), sort_keys=True, default=str) != json.dumps(
            other.summary(), sort_keys=True, default=str
        ):
            return False
        return True

    def assert_accounting_clean(self) -> list[str]:
        """Run the invariant replay and raise on any violation (tests)."""
        violations = self.invariant_violations()
        if violations:
            raise AssertionError(
                "accounting invariants violated: " + "; ".join(violations)
            )
        return violations


__all__ = ["BacktestResult"]