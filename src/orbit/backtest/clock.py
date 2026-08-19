# The deterministic market event clock (Phase 7).
#
# Defines exactly, per session: when a market observation becomes available,
# when a signal may be generated, when an order may be submitted, when a
# simulated order can fill, when portfolio valuation occurs.
# Everything is deterministic: the same bars + same sessions, the same
# availability instants and the same event stream.

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from enum import Enum
from typing import Any

import polars as pl

from orbit.temporal.adapters import as_published_bars
from orbit.temporal.times import EXCHANGE_TZ, normalize_instant, session_close_utc

_SESSION_OPEN_LOCAL = time(9, 30)

_BAR_REQUIRED = {"instrument_id", "trade_date", "open", "high", "low", "close", "volume"}

# The stored price basis the simulator fills and values on. The Phase 3
# normalizer stores split-adjusted OHLC ('split_adjusted'); 'adjclose' is
# never a trading price.
CANONICAL_PRICE_BASIS = "split_continuous_stored"


def session_open_utc(session: date) -> datetime:
    """09:30 America/New_York on `session`, as naive UTC (DST-aware)."""
    local = datetime.combine(session, _SESSION_OPEN_LOCAL, tzinfo=EXCHANGE_TZ)
    return local.astimezone(timezone.utc)


@dataclass(frozen=True)
class BarRecord:
    """One canonical daily bar as the clock serves it.

    `price_basis` is always the canonical split-continuous stored basis for
    prices; `liquidity_volume` is the as-published volume when the
    corporate-actions artifact exists, else the stored (provider-basis)
    volume - a documented liquidity proxy.
    """

    instrument_id: str
    trade_date: date
    open: float
    high: float
    low: float
    close: float
    volume: float
    liquidity_volume: float
    price_basis: str
    volume_basis: str

    @property
    def session(self) -> date:
        return self.trade_date


# The market event clock over a canonical bars frame.
class MarketEventClock:
    """The deterministic session clock over a canonical bars frame."""

    def __init__(
        self,
        bars: "pl.DataFrame",
        events: "pl.DataFrame | None" = None,
        volume_basis: str | None = None,
    ):
        missing = _BAR_REQUIRED - set(bars.columns)
        if missing:
            raise ValueError(
                f"the market clock requires the canonical normalized bar "
                f"columns; missing: {sorted(missing)}"
            )

        # Check for duplicate (instrument_id, trade_date) pairs
        n_total = len(bars)
        n_unique = len(bars.select(["instrument_id", "trade_date"]).unique())
        if n_unique < n_total:
            raise ValueError("duplicate (instrument_id, trade_date) bars found")

        self._bars = bars.sort(["trade_date", "instrument_id"])

        if events is not None and events.height:
            published = as_published_bars(bars, events)
            self._liquidity_volume = {
                (r[0], r[1]): r[2]
                for r in published.select(
                    ["instrument_id", "trade_date", "volume"]
                ).iter_rows()
            }
            self._volume_basis = "as_published"
        else:
            self._liquidity_volume = {}
            self._volume_basis = "provider_stored"

        self._by_key: dict[tuple[str, date], dict] = {}
        for row in self._bars.to_dicts():
            key = (row["instrument_id"], row["trade_date"])
            self._by_key[key] = row

        self._sessions: list[date] = sorted(self._bars["trade_date"].unique().to_list())
        self._session_index: dict[date, int] = {
            s: i for i, s in enumerate(self._sessions)
        }

    @property
    def volume_basis(self) -> str:
        """Return the volume basis string for manifest/identity purposes."""
        return self._volume_basis

    # -------------------------------------------------------------- dates

    def sessions(self) -> list[date]:
        return list(self._sessions)

    def session_index(self, session: date) -> int | None:
        return self._session_index.get(session)

    def session_at(self, index: int) -> date | None:
        if 0 <= index < len(self._sessions):
            return self._sessions[index]
        return None

    def next_session(self, session: date, steps: int) -> date | None:
        """The session `steps` sessions after `session` on the global
        session line (None when beyond the last session)."""
        if steps < 0:
            raise ValueError("steps must be >= 0")
        idx = self._session_index.get(session)
        if idx is None:
            return None
        return self.session_at(idx + steps)

    def session_after(self, session: date) -> date | None:
        return self.next_session(session, 1)

    def has_session(self, session: date) -> bool:
        return session in self._session_index

    # -------------------------------------------------------------- bars

    def has_bar(self, instrument_id: str, session: date) -> bool:
        return (instrument_id, session) in self._by_key

    def bar(self, instrument_id: str, session: date) -> BarRecord | None:
        """The canonical bar for (instrument, session), or None when the
        instrument has no bar on that session (data gap / delisting).

        A bar whose OHLC field is null is served with NaN in that field:
        the price/volume guards (_missing_price/_missing_volume) are
        the single authority on defective prices and decide the failure
        kind - a null field never crashes the run with a raw TypeError.
        """
        row = self._by_key.get((instrument_id, session))
        if row is None:
            return None
        liq_key = (instrument_id, session)
        liquidity_volume = self._liquidity_volume.get(liq_key)
        if liquidity_volume is None:
            liquidity_volume = float(row["volume"]) if row["volume"] is not None else math.nan

        def _f(value: Any) -> float:
            if value is None:
                return math.nan
            return float(value)

        return BarRecord(
            instrument_id=instrument_id,
            trade_date=row["trade_date"],
            open=_f(row["open"]),
            high=_f(row["high"]),
            low=_f(row["low"]),
            close=_f(row["close"]),
            volume=_f(row["volume"]),
            liquidity_volume=float(liquidity_volume),
            price_basis=CANONICAL_PRICE_BASIS,
            volume_basis=self._volume_basis,
        )

    # -------------------------------------------------------------- valuations

    def last_close(self, instrument_id: str, session: date, strict: bool = False) -> float | None:
        """The most recent close at or before `session` (used for valuation
        of instruments with a data gap). None when the instrument has no
        usable close at or before the session. Defective closes (null,
        non-finite, non-positive) are skipped by a backward scan: a bad
        bar must never poison the reference price.

        If `strict=True`, the close must be strictly before `session`
        (key[1] < session), never matching the session's own close."""
        candidates = [
            key[1]
            for key in self._by_key
            if key[0] == instrument_id and (key[1] < session if strict else key[1] <= session)
        ]
        for candidate in sorted(candidates, reverse=True):
            row = self._by_key.get((instrument_id, candidate))
            if row is None:
                continue
            close = row["close"]
            if close is None or not math.isfinite(close) or close <= 0:
                continue
            return float(close)
        return None

    def instrument_sessions(self, instrument_id: str) -> list[date]:
        """The sessions on which the instrument has a bar."""
        return sorted(
            key[1]
            for key in self._by_key
            if key[0] == instrument_id
        )

    def instruments(self) -> list[str]:
        return sorted({key[0] for key in self._by_key})

    # -------------------------------------------------------------- timing

    def availability_instant(self, session: date) -> datetime:
        """The instant a bar for `session` becomes available: the session
        close (Phase 4 convention, never ts_utc)."""
        return session_close_utc(session)

    def execution_instant(self, session: date, at_open: bool) -> datetime:
        """The execution instant of a fill on `session`: the session open
        for open fills, the session close for close fills."""
        if at_open:
            return session_open_utc(session)
        return session_close_utc(session)

    def window_sessions(self, start: date | None, end: date | None) -> list[date]:
        """The clock sessions inside [start, end] (None bounds are open)."""
        out = []
        for s in self._sessions:
            if start is not None and s < start:
                continue
            if end is not None and s > end:
                continue
            out.append(s)
        return out

    # -------------------------------------------------------------- parsing

    @staticmethod
    def normalize_signals(signals: Any) -> list[dict[str, Any]]:
        """Normalize a signals input (polars frame or list of dicts) to a
        canonical list of dicts with validated signal sessions.

        Required columns: instrument_id, signal_session (date) and one of
        direction/target. Optional: signal_id, decision_time,
        signal_metric, strategy_ref, session.
        """
        if isinstance(signals, pl.DataFrame):
            rows = signals.to_dicts()
        else:
            rows = [dict(s) for s in signals]
        normalized: list[dict[str, Any]] = []
        for i, r in enumerate(rows):
            instrument_id = r.get("instrument_id")
            if not instrument_id:
                raise ValueError(f"signal #{i}: instrument_id is required")
            session = r.get("signal_session") or r.get("session")
            if isinstance(session, datetime):
                session = session.date()
            if not isinstance(session, date):
                raise ValueError(
                    f"signal #{i} for {instrument_id}: signal_session must be "
                    f"a date, got {session!r}"
                )
            direction = str(r.get("direction") or "long").casefold()
            if direction not in {"long", "flat"}:
                raise ValueError(
                    f"signal #{i} for {instrument_id}: direction must be "
                    f"'long' or 'flat', got {direction!r}"
                )
            target = r.get("target")
            if target is None:
                target = 0.0
            else:
                target = float(target)
                if not math.isfinite(target) or target < 0:
                    raise ValueError(
                        f"signal #{i} for {instrument_id}: target must be a "
                        f"finite non-negative fraction, got {target!r}"
                    )
            normalized.append(
                {
                    "signal_id": str(r.get("signal_id") or f"SIG-{i + 1:06d}"),
                    "instrument_id": str(instrument_id),
                    "signal_session": session,
                    "direction": direction,
                    "target": target,
                    "signal_metric": r.get("signal_metric"),
                    "decision_time": normalize_instant(
                        r.get("decision_time")
                    ),
                    "strategy_ref": r.get("strategy_ref"),
                }
            )
        return normalized


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

    @property
    def volume_basis(self) -> str:
        """Return the volume basis string for manifest/identity purposes."""
        return self._volume_basis