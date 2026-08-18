"""The Phase 5 Label Engine: deterministic future-outcome computation.

Given a `LabelContract` and a set of decision rows, the engine computes one
reproducible outcome per decision:

    decision instant
        -> reference (entry) session        (Phase 4 close semantics)
        -> the next H trading sessions      (per-instrument session series)
        -> outcome session + outcome value  (one defined formula)

Data contract (Phase 3/4 canonical):
  - bars are the normalized market shape (instrument_id, trade_date, OHLCV);
    the stored close is the canonical SPLIT-CONTINUOUS basis (the Phase 3
    normalizer stores split-adjusted OHLC; consecutive-close ratios are
    continuous across splits, so a split inside the outcome window never
    creates an artificial return);
  - the sibling events artifact (splits + dividends) supplies corporate
    actions; as-published closes are reconstructed through the SAME
    `as_published_bars()` function the temporal engine uses, and are
    recorded on every label row for audit;
  - an optional instrument master (Phase 2 `Instrument` records) supplies
    delisting knowledge so a disappearing security is never silently read
    as a zero (or any fabricated) outcome.

Timing semantics (documented in docs/phase5_labels.md):
  - a bar for session D is completed at the session close, 16:00
    America/New_York (Phase 4 `session_close_utc`);
  - DECISION_INSTANT anchor: the reference session is the LAST completed
    session strictly BEFORE the decision instant (a decision at exactly the
    close does not see that day's bar - strict boundary);
  - POST_EVENT anchor: the reference session is the FIRST completed session
    strictly AFTER the event's availability instant (the PEAD anchor);
  - the outcome window is the next H sessions STRICTLY AFTER the reference
    session, counted on the instrument's own session series (calendar gaps
    such as weekends/holidays never count as sessions);
  - if fewer than H sessions exist after the reference session, the label
    is UNAVAILABLE with an explicit reason - the horizon is never silently
    shortened and missing future prices are never filled.

Everything is deterministic: same bars + contract + decisions -> identical
frame, row for row (the snapshot layer adds a content digest).
"""

from __future__ import annotations

import math
from datetime import date, datetime, timezone
from enum import Enum
from typing import Any
from zoneinfo import ZoneInfo

import polars as pl

from orbit.labels.contract import AnchorMode, LabelContract, ReturnConvention
from orbit.schemas.common import LabelType
from orbit.labels.outcomes import (
    compute_drawdown,
    compute_return,
    excess_return,
    realized_volatility,
    window_returns,
)
from orbit.temporal.adapters import _ADJUSTED_LABELS, as_published_bars
from orbit.temporal.times import normalize_instant, session_close_utc

_EXCHANGE_TZ = ZoneInfo("America/New_York")

ENGINE_VERSION = "v1.0.0"

_BAR_REQUIRED = {
    "instrument_id", "trade_date", "open", "high", "low", "close", "volume",
}
_EVENT_REQUIRED = {"instrument_id", "kind", "ts", "ratio"}


class UnavailableReason(str, Enum):
    """Explicit reasons an outcome is unavailable. A reason is never a
    value: unavailable outcomes always carry NULL outcome_value.

    MISSING_DIVIDEND_DATA is reserved: total-return labels require the
    corporate-actions events artifact, and its ABSENCE raises at
    compute() (a silent price-return label is worse than an error). The
    artifact is the authoritative dividend ledger per instrument - an
    instrument with no dividend rows in the artifact is treated as having
    no dividends (documented in docs/phase5_labels.md)."""

    NO_ENTRY_BAR = "no_entry_bar"
    MISSING_ANCHOR = "missing_anchor"
    MISSING_ENTRY_PRICE = "missing_entry_price"
    INSUFFICIENT_FUTURE_DATA = "insufficient_future_data"
    MISSING_OUTCOME_PRICE = "missing_outcome_price"
    MISSING_WINDOW_PRICE = "missing_window_price"
    DELISTED = "delisted"
    BENCHMARK_UNAVAILABLE = "benchmark_unavailable"
    INSUFFICIENT_OBSERVATIONS = "insufficient_observations"
    MISSING_DIVIDEND_DATA = "missing_dividend_data"
    CORPORATE_ACTION_DATA_INCOMPLETE = "corporate_action_data_incomplete"


# The canonical label output schema, in fixed column order (a stable
# contract for downstream phases and the snapshot digest).
LABEL_OUTPUT_COLUMNS: list[tuple[str, pl.DataType]] = [
    ("label_id", pl.Utf8),
    ("version", pl.Utf8),
    ("contract_digest", pl.Utf8),
    ("engine_version", pl.Utf8),
    ("decision_id", pl.Utf8),
    ("instrument_id", pl.Utf8),
    ("decision_time", pl.Datetime("us")),
    ("anchor_instant", pl.Datetime("us")),
    ("anchor_mode", pl.Utf8),
    ("entry_session", pl.Date),
    ("entry_timestamp", pl.Datetime("us")),
    ("entry_close", pl.Float64),
    ("entry_close_as_published", pl.Float64),
    ("outcome_session", pl.Date),
    ("outcome_timestamp", pl.Datetime("us")),
    ("outcome_close", pl.Float64),
    ("outcome_close_as_published", pl.Float64),
    ("window_start_session", pl.Date),
    ("window_end_session", pl.Date),
    ("window_start_instant", pl.Datetime("us")),
    ("window_end_instant", pl.Datetime("us")),
    ("total_dividends", pl.Float64),
    ("horizon_sessions", pl.Int64),
    ("sessions_available", pl.Int64),
    ("target_type", pl.Utf8),
    ("return_convention", pl.Utf8),
    ("benchmark", pl.Utf8),
    ("benchmark_entry_session", pl.Date),
    ("benchmark_outcome_session", pl.Date),
    ("benchmark_return", pl.Float64),
    ("outcome_status", pl.Utf8),
    ("unavailable_reason", pl.Utf8),
    ("outcome_value", pl.Float64),
    ("outcome_detail", pl.Utf8),
]


def _event_session(ts: datetime) -> date | None:
    """Exchange-local session date of a corporate-action event timestamp
    (the same rule as the Phase 3 normalizer's trade_date)."""
    if ts is None:
        return None
    return ts.replace(tzinfo=timezone.utc).astimezone(_EXCHANGE_TZ).date()


def empty_label_frame() -> pl.DataFrame:
    return pl.DataFrame(schema=LABEL_OUTPUT_COLUMNS)


class LabelEngine:
    """Deterministic label computation over canonical market bars."""

    def __init__(
        self,
        bars: pl.DataFrame,
        events: pl.DataFrame | None = None,
        instruments: list[Any] | None = None,
        volume_basis: str | None = None,
    ):
        missing = _BAR_REQUIRED - set(bars.columns)
        if missing:
            raise ValueError(
                "label engine requires the canonical normalized bar columns; "
                f"missing: {sorted(missing)}"
            )
        if events is not None and not _EVENT_REQUIRED.issubset(events.columns):
            raise ValueError(
                "the events artifact must carry instrument_id, kind, ts, ratio"
            )
        self._events = events
        if instruments is None:
            self._instrument_by_id: dict[str, dict[str, Any]] = {}
        elif isinstance(instruments, pl.DataFrame):
            self._instrument_by_id = {
                r["instrument_id"]: r for r in instruments.to_dicts()
            }
        else:
            self._instrument_by_id = {}
            for it in instruments:
                if isinstance(it, dict):
                    self._instrument_by_id[it["instrument_id"]] = it
                else:
                    self._instrument_by_id[it.instrument_id] = {
                        "instrument_id": it.instrument_id,
                        "delisting_date": getattr(it, "delisting_date", None),
                    }

        # Canonical basis guard: label returns require the split-continuous
        # stored series (Phase 3 normalizer's 'split_adjusted' basis). A
        # raw-basis provider's stored closes would create artificial split
        # returns; the engine refuses loudly instead of guessing.
        if "adjustment" in bars.columns:
            labels = {a for a in bars["adjustment"].drop_nulls().unique().to_list()}
            if labels and not labels.issubset(_ADJUSTED_LABELS):
                raise ValueError(
                    "label returns require the canonical split-continuous "
                    f"basis; bars carry adjustment label(s) {sorted(labels)} "
                    "outside the adjusted labels (a raw-basis provider would "
                    "create artificial returns across splits)"
                )

        stored = (
            bars.select(["instrument_id", "trade_date", "close"])
            .sort(["instrument_id", "trade_date"])
        )
        dup = stored.group_by(["instrument_id", "trade_date"]).agg(pl.len().alias("n"))
        dup = dup.filter(pl.col("n") > 1)
        if dup.height:
            raise ValueError(
                "duplicate (instrument_id, trade_date) bars are not allowed: "
                f"{dup.row(0, named=True)}"
            )

        published = as_published_bars(bars, events, volume_basis=volume_basis)
        self._published = published.select(
            ["instrument_id", "trade_date", "close", "price_basis"]
        )

        merged = stored.join(
            self._published,
            on=["instrument_id", "trade_date"],
            how="left",
            suffix="_as_published",
        )
        merged = merged.with_columns(
            pl.col("trade_date")
            .map_elements(session_close_utc, return_dtype=pl.Datetime("us"))
            .alias("session_close")
        )

        self._bars = merged
        self._series: dict[str, list[dict[str, Any]]] = {}
        for inst_id in merged["instrument_id"].unique().sort().to_list():
            sub = merged.filter(pl.col("instrument_id") == inst_id).sort("trade_date")
            self._series[inst_id] = sub.to_dicts()

        self._splits, self._dividends, self._action_incomplete = self._action_maps(
            events
        )

    # ------------------------------------------------------------- queries

    def bars_universe(self) -> list[str]:
        return sorted(self._series)

    def instrument_sessions(self, instrument_id: str) -> list[date]:
        rows = self._series.get(instrument_id, [])
        return [r["trade_date"] for r in rows]

    def entry_bar(
        self,
        instrument_id: str,
        decision_time: datetime | date | str,
        anchor_instant: datetime | date | str | None = None,
        anchor_mode: AnchorMode = AnchorMode.DECISION_INSTANT,
    ) -> dict[str, Any] | None:
        """The reference session for a decision, or None when no completed
        bar qualifies (the caller turns None into NO_ENTRY_BAR)."""
        rows = self._series.get(instrument_id)
        if not rows:
            return None
        if anchor_mode == AnchorMode.POST_EVENT:
            anchor = normalize_instant(anchor_instant)
            if anchor is None:
                raise ValueError("anchor_instant is required for POST_EVENT")
            for r in rows:
                if r["session_close"] > anchor:
                    return r
            return None
        t = normalize_instant(decision_time)
        if t is None:
            raise ValueError("decision_time is required")
        entry = None
        for r in rows:
            if r["session_close"] < t:
                entry = r
            else:
                break
        return entry

    def outcome_window(
        self, instrument_id: str, entry_session: date, horizon: int
    ) -> list[dict[str, Any]]:
        """The next `horizon` sessions strictly after `entry_session` on the
        instrument's own session series (calendar gaps never count)."""
        if horizon < 1:
            raise ValueError("horizon must be >= 1")
        rows = self._series.get(instrument_id, [])
        return [r for r in rows if r["trade_date"] > entry_session][:horizon]

    # ------------------------------------------------------------- compute

    def compute(
        self, contract: LabelContract, decisions: Any
    ) -> pl.DataFrame:
        """Compute the outcome for every decision row.

        `decisions` is a polars DataFrame or a list of dicts with columns
        `instrument_id`, `decision_time` (datetime/date/ISO string) and
        optional `decision_id`, `anchor_instant` (POST_EVENT anchors).
        """
        if not isinstance(contract, LabelContract):
            raise TypeError(
                "compute() requires a LabelContract (register it first via "
                "LabelVersionRegistry)"
            )
        if contract.target_type in (LabelType.FORWARD_RETURN, LabelType.EXCESS_RETURN):
            if (
                contract.return_convention == ReturnConvention.SIMPLE_TOTAL_RETURN
                and self._events is None
            ):
                raise ValueError(
                    f"contract {contract.label_id} v{contract.version} uses "
                    "SIMPLE_TOTAL_RETURN but no corporate-actions events "
                    "artifact was supplied; a total-return label without "
                    "dividend data would silently be a price-return label"
                )
        rows = [self._compute_one(contract, d) for d in self._normalize_decisions(decisions)]
        out = pl.from_dicts(rows, schema=dict(LABEL_OUTPUT_COLUMNS))
        return out.sort(
            ["instrument_id", "decision_time", "decision_id"]
        )

    def compute_one(
        self,
        contract: LabelContract,
        instrument_id: str,
        decision_time: datetime | date | str,
        *,
        decision_id: str | None = None,
        anchor_instant: datetime | date | str | None = None,
    ) -> dict[str, Any]:
        """Single-decision convenience wrapper (tests, audits)."""
        frame = self.compute(
            contract,
            [
                {
                    "instrument_id": instrument_id,
                    "decision_time": decision_time,
                    "decision_id": decision_id,
                    "anchor_instant": anchor_instant,
                }
            ],
        )
        return frame.row(0, named=True)

    # ------------------------------------------------------------- internals

    def _normalize_decisions(self, decisions: Any) -> list[dict[str, Any]]:
        if isinstance(decisions, pl.DataFrame):
            missing = {"instrument_id", "decision_time"} - set(decisions.columns)
            if missing:
                raise ValueError(
                    "decisions frame requires instrument_id and decision_time; "
                    f"missing {sorted(missing)}"
                )
            rows = decisions.to_dicts()
        else:
            rows = [dict(d) for d in decisions]
        normalized: list[dict[str, Any]] = []
        for r in rows:
            t = normalize_instant(r.get("decision_time"))
            if t is None:
                raise ValueError(
                    "every decision needs a decision_time; got "
                    f"{r.get('decision_time')!r}"
                )
            instrument_id = r.get("instrument_id")
            if not instrument_id:
                raise ValueError("every decision needs an instrument_id")
            decision_id = r.get("decision_id") or (
                f"{instrument_id}|{t.isoformat()}"
            )
            anchor = (
                normalize_instant(r.get("anchor_instant"))
                if r.get("anchor_instant") is not None
                else None
            )
            normalized.append(
                {
                    "decision_id": str(decision_id),
                    "instrument_id": str(instrument_id),
                    "decision_time": t,
                    "anchor_instant": anchor,
                }
            )
        return normalized

    def _compute_one(
        self, contract: LabelContract, d: dict[str, Any]
    ) -> dict[str, Any]:
        H = contract.horizon
        base: dict[str, Any] = {
            "label_id": contract.label_id,
            "version": contract.version,
            "contract_digest": contract.content_hash(),
            "engine_version": ENGINE_VERSION,
            "decision_id": d["decision_id"],
            "instrument_id": d["instrument_id"],
            "decision_time": d["decision_time"],
            "anchor_instant": d["anchor_instant"],
            "anchor_mode": contract.anchor_mode.value,
            "entry_session": None,
            "entry_timestamp": None,
            "entry_close": None,
            "entry_close_as_published": None,
            "outcome_session": None,
            "outcome_timestamp": None,
            "outcome_close": None,
            "outcome_close_as_published": None,
            "window_start_session": None,
            "window_end_session": None,
            "window_start_instant": None,
            "window_end_instant": None,
            "horizon_sessions": H,
            "sessions_available": 0,
            "target_type": contract.target_type.value,
            "return_convention": (
                contract.return_convention.value
                if contract.return_convention
                else None
            ),
            "benchmark": contract.benchmark,
            "benchmark_entry_session": None,
            "benchmark_outcome_session": None,
            "benchmark_return": None,
            "outcome_status": "unavailable",
            "unavailable_reason": None,
            "outcome_value": None,
            "outcome_detail": None,
        }

        inst = d["instrument_id"]
        rows = self._series.get(inst)
        if not rows:
            return self._unavailable(
                base, UnavailableReason.NO_ENTRY_BAR,
                f"no bars for instrument {inst}",
            )
        if (
            contract.anchor_mode == AnchorMode.POST_EVENT
            and d["anchor_instant"] is None
        ):
            return self._unavailable(
                base, UnavailableReason.MISSING_ANCHOR,
                "POST_EVENT anchor requires an anchor_instant",
            )
        entry_idx, entry = self._entry_index(rows, contract.anchor_mode, d)
        if entry is None:
            return self._unavailable(
                base, UnavailableReason.NO_ENTRY_BAR,
                "no completed bar qualifies as the reference session",
            )
        if entry["close"] is None:
            return self._unavailable(
                base, UnavailableReason.MISSING_ENTRY_PRICE,
                f"entry session {entry['trade_date']} has no close",
            )
        base["entry_session"] = entry["trade_date"]
        base["entry_timestamp"] = entry["session_close"]
        base["entry_close"] = entry["close"]
        base["entry_close_as_published"] = entry["close_as_published"]
        base["window_start_session"] = entry["trade_date"]
        base["window_start_instant"] = entry["session_close"]

        window = rows[entry_idx + 1: entry_idx + 1 + H]
        base["sessions_available"] = len(window)
        if len(window) < H:
            reason, detail = self._shortfall(inst, rows, len(window), H)
            return self._unavailable(base, reason, detail)
        outcome = window[-1]
        if outcome["close"] is None:
            return self._unavailable(
                base, UnavailableReason.MISSING_OUTCOME_PRICE,
                f"outcome session {outcome['trade_date']} has no close",
            )
        base["outcome_session"] = outcome["trade_date"]
        base["outcome_timestamp"] = outcome["session_close"]
        base["outcome_close"] = outcome["close"]
        base["outcome_close_as_published"] = outcome["close_as_published"]
        base["window_end_session"] = outcome["trade_date"]
        base["window_end_instant"] = outcome["session_close"]

        if inst in self._action_incomplete:
            return self._unavailable(
                base, UnavailableReason.CORPORATE_ACTION_DATA_INCOMPLETE,
                "a corporate-action event in this instrument's events artifact "
                "is missing its amount/ratio or timestamp; the return basis "
                "cannot be established",
            )

        closes = [r["close"] for r in rows]
        if any(c is None for c in closes[entry_idx: entry_idx + 1 + H]):
            return self._unavailable(
                base, UnavailableReason.MISSING_WINDOW_PRICE,
                "a close inside the outcome window is missing",
            )

        target = contract.target_type
        if target.value == "forward_return":
            dividends = self._window_dividends(inst, window)
            base["total_dividends"] = (
                sum(dividends.values())
                if contract.return_convention == ReturnConvention.SIMPLE_TOTAL_RETURN
                else 0.0
            )
            base["outcome_value"] = compute_return(
                closes, entry_idx, H, contract.return_convention, dividends,
            )
        elif target.value == "excess_return":
            asset_dividends = self._window_dividends(inst, window)
            base["total_dividends"] = (
                sum(asset_dividends.values())
                if contract.return_convention == ReturnConvention.SIMPLE_TOTAL_RETURN
                else 0.0
            )
            asset_return = compute_return(
                closes, entry_idx, H, contract.return_convention,
                asset_dividends,
            )
            bench = self._benchmark_return(contract, d)
            if bench is None:
                return self._unavailable(
                    base, UnavailableReason.BENCHMARK_UNAVAILABLE,
                    f"benchmark {contract.benchmark} outcome unavailable for "
                    "this decision",
                )
            base["benchmark_entry_session"] = bench["entry_session"]
            base["benchmark_outcome_session"] = bench["outcome_session"]
            base["benchmark_return"] = bench["value"]
            base["outcome_value"] = excess_return(asset_return, bench["value"])
        elif target.value == "volatility":
            returns = window_returns(closes, entry_idx, H)
            if len(returns) < contract.min_observations:
                return self._unavailable(
                    base, UnavailableReason.INSUFFICIENT_OBSERVATIONS,
                    f"window yields {len(returns)} returns but the contract "
                    f"requires at least {contract.min_observations}",
                )
            base["outcome_value"] = realized_volatility(
                returns, contract.annualization
            )
        elif target.value == "drawdown":
            base["outcome_value"] = compute_drawdown(
                closes, entry_idx, H, contract.drawdown_type
            )
        else:  # pragma: no cover - the contract validator already refused
            raise ValueError(f"unsupported target type: {target.value}")

        base["outcome_status"] = "available"
        return base

    def _entry_index(
        self,
        rows: list[dict[str, Any]],
        anchor_mode: AnchorMode,
        d: dict[str, Any],
    ) -> tuple[int | None, dict[str, Any] | None]:
        if anchor_mode == AnchorMode.POST_EVENT:
            anchor = d["anchor_instant"]
            for idx, r in enumerate(rows):
                if r["session_close"] > anchor:
                    return idx, r
            return None, None
        t = d["decision_time"]
        entry_idx = -1
        for idx, r in enumerate(rows):
            if r["session_close"] < t:
                entry_idx = idx
            else:
                break
        if entry_idx < 0:
            return None, None
        return entry_idx, rows[entry_idx]

    def _benchmark_return(
        self, contract: LabelContract, d: dict[str, Any]
    ) -> dict[str, Any] | None:
        bench = contract.benchmark
        rows = self._series.get(bench) if bench else None
        if not rows:
            return None
        if bench in self._action_incomplete:
            return None
        entry_idx, entry = self._entry_index(rows, contract.anchor_mode, d)
        if entry is None or entry["close"] is None:
            return None
        window = rows[entry_idx + 1: entry_idx + 1 + contract.horizon]
        if len(window) < contract.horizon:
            return None
        outcome = window[-1]
        if outcome["close"] is None:
            return None
        closes = [r["close"] for r in rows]
        if any(c is None for c in closes[entry_idx: entry_idx + 1 + contract.horizon]):
            return None
        value = compute_return(
            closes, entry_idx, contract.horizon, contract.return_convention,
            self._window_dividends(bench, window),
        )
        return {
            "value": value,
            "entry_session": entry["trade_date"],
            "outcome_session": outcome["trade_date"],
        }

    def _window_dividends(
        self, instrument_id: str, window: list[dict[str, Any]]
    ) -> dict[int, float]:
        """Ex-date dividends of the window sessions, converted to the
        split-continuous share basis (raw amount / split factor at the
        ex-date session), keyed by window-relative position."""
        divs = self._dividends.get(instrument_id)
        if not divs:
            return {}
        by_session: dict[date, float] = {}
        for ex_date, amount in divs:
            by_session[ex_date] = by_session.get(ex_date, 0.0) + amount
        out: dict[int, float] = {}
        for i, r in enumerate(window, start=1):
            amount = by_session.get(r["trade_date"])
            if amount:
                factor = self._stored_basis_factor(instrument_id, r["trade_date"])
                out[i] = amount / factor if factor else amount
        return out

    def _shortfall(
        self,
        instrument_id: str,
        rows: list[dict[str, Any]],
        achieved: int,
        horizon: int,
    ) -> tuple[UnavailableReason, str]:
        """Classify an incomplete outcome window. A security that disappears
        with its instrument master recording a delisting is never read as a
        zero return: the label is unavailable with reason DELISTED. When the
        bar series simply ends (no delisting record, or the data stops well
        before the recorded delisting), the reason is
        INSUFFICIENT_FUTURE_DATA."""
        inst = self._instrument_by_id.get(instrument_id)
        delisting_date = inst.get("delisting_date") if inst is not None else None
        if delisting_date is not None and rows:
            last_session = rows[-1]["trade_date"]
            if last_session <= delisting_date:
                return (
                    UnavailableReason.DELISTED,
                    f"instrument delisted on {delisting_date.isoformat()} "
                    f"(last bar {last_session.isoformat()}); only {achieved} "
                    f"of {horizon} sessions available - no outcome is "
                    "fabricated",
                )
        return (
            UnavailableReason.INSUFFICIENT_FUTURE_DATA,
            f"only {achieved} of {horizon} sessions available after the "
            "reference session; the horizon is never silently shortened",
        )

    def _action_maps(
        self, events: pl.DataFrame | None
    ) -> tuple[
        dict[str, list[tuple[date, float]]],
        dict[str, list[tuple[date, float]]],
        set[str],
    ]:
        splits: dict[str, list[tuple[date, float]]] = {}
        dividends: dict[str, list[tuple[date, float]]] = {}
        incomplete: set[str] = set()
        if events is None or events.height == 0:
            return splits, dividends, incomplete
        for row in events.iter_rows(named=True):
            inst = row["instrument_id"]
            ex_date = _event_session(row["ts"])
            ratio = row.get("ratio")
            if ex_date is None or ratio is None or ratio <= 0:
                incomplete.add(inst)
                continue
            if row["kind"] == "splits":
                splits.setdefault(inst, []).append((ex_date, float(ratio)))
            elif row["kind"] == "dividends":
                dividends.setdefault(inst, []).append((ex_date, float(ratio)))
        return splits, dividends, incomplete

    def _stored_basis_factor(self, instrument_id: str, session: date) -> float:
        """Product of split ratios whose ex-date is strictly after `session`
        (the conversion between the raw basis and the stored split-
        continuous basis). Mirrors the temporal adapter's factor."""
        factor = 1.0
        for ex_date, ratio in self._splits.get(instrument_id, []):
            if ex_date > session:
                factor *= ratio
        return factor

    @staticmethod
    def _unavailable(
        base: dict[str, Any], reason: UnavailableReason, detail: str
    ) -> dict[str, Any]:
        base["outcome_status"] = "unavailable"
        base["unavailable_reason"] = reason.value
        base["outcome_value"] = None
        base["outcome_detail"] = detail
        return base


def overlapping_pairs(
    frame: pl.DataFrame,
    sessions_by_instrument: dict[str, list[date]] | None = None,
) -> list[dict[str, Any]]:
    """Every pair of AVAILABLE labels of the SAME instrument whose outcome
    windows overlap on session dates (window_end_a >= window_start_b and
    window_end_b >= window_start_a; windows include the entry session).

    `overlap_sessions` counts the actual SESSIONS in the intersection
    (calendar gaps never count). Pass `sessions_by_instrument` (e.g. the
    engine's `instrument_sessions`) for the exact count; without it the
    count is the calendar-day span (an upper bound) and the overlap
    predicate itself is unaffected.

    This is the overlap information later phases need for purging, embargo
    and statistical inference - identification only, no statistical
    machinery here. Deterministic: input order does not affect the result.
    """
    rows = [
        r
        for r in frame.sort(
            ["instrument_id", "decision_time", "decision_id"]
        ).iter_rows(named=True)
        if r["outcome_status"] == "available"
    ]
    pairs: list[dict[str, Any]] = []
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            a, b = rows[i], rows[j]
            if a["instrument_id"] != b["instrument_id"]:
                continue
            if (
                a["window_end_session"] >= b["window_start_session"]
                and b["window_end_session"] >= a["window_start_session"]
            ):
                lo = max(a["window_start_session"], b["window_start_session"])
                hi = min(a["window_end_session"], b["window_end_session"])
                sessions = sessions_by_instrument.get(a["instrument_id"])
                if sessions:
                    overlap_sessions = sum(1 for s in sessions if lo <= s <= hi)
                else:
                    overlap_sessions = (hi - lo).days + 1
                pairs.append(
                    {
                        "instrument_id": a["instrument_id"],
                        "label_a": a["label_id"],
                        "label_b": b["label_id"],
                        "decision_id_a": a["decision_id"],
                        "decision_id_b": b["decision_id"],
                        "window_start_a": a["window_start_session"],
                        "window_end_a": a["window_end_session"],
                        "window_start_b": b["window_start_session"],
                        "window_end_b": b["window_end_session"],
                        "overlap_sessions": overlap_sessions,
                    }
                )
    return pairs


__all__ = [
    "ENGINE_VERSION",
    "LABEL_OUTPUT_COLUMNS",
    "LabelEngine",
    "UnavailableReason",
    "empty_label_frame",
    "overlapping_pairs",
]