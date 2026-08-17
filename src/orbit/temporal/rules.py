"""The core availability rules of the Temporal Truth Engine.

One function decides every record:

    allowed(x, t) = publication_time(x) < t
                    AND all other effective-time constraints hold

The rules are deliberately few, explicit, and conservative:

1. NOT_POINT_IN_TIME (revised macro
   served as latest vintage only)    -> REJECT (a later revision could replace
                                        the value; we cannot establish what was
                                        known, so nothing from this series enters)
2. MISSING_PUBLICATION_TIME          -> REJECT (availability is never invented)
3. PUBLICATION_AT_OR_AFTER_AS_OF     -> REJECT (strict boundary, exact-tie rejects)
4. NO_VINTAGE_AT_AS_OF               -> REJECT (no version released before t)
5. EVENT_AFTER_AS_OF                 -> REJECT (forward-dated records are not
                                        part of the historical information set)
6. otherwise                         -> ALLOW (decision code records which
                                        precision convention was applied)

Warnings are attached but never flip a decision:
    INGESTED_AFTER_AS_OF    the information was public before t but ORBIT
                            downloaded it later (provenance note only)
    DATE_PRECISION_NOTE     publication known only to the day; next-day rule
                            applied

A false rejection shrinks the information set; a false acceptance can
manufacture fake alpha. Every rule prefers the first error.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from orbit.temporal.times import (
    DecisionCode,
    TimePrecision,
    Timing,
    next_day_midnight,
    normalize_instant,
)


@dataclass(frozen=True)
class AvailabilityDecision:
    """The engine's answer for one record at one decision time."""

    record_id: str
    as_of_time: datetime
    allowed: bool
    code: DecisionCode
    detail: str = ""
    warnings: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "as_of_time": self.as_of_time.isoformat(),
            "allowed": self.allowed,
            "code": self.code.value,
            "detail": self.detail,
            "warnings": list(self.warnings),
        }


def decide(timing: Timing, as_of: datetime | Any) -> AvailabilityDecision:
    """Decide whether `timing` was available at `as_of`.

    `as_of` may be a datetime/date/string; it is normalized through the
    single UTC-naive conversion point.
    """
    t = normalize_instant(as_of)
    if t is None:
        raise ValueError("as_of_time is required")

    # 0. revised series without vintage history: NOT point-in-time, period.
    #    The policy is casefolded: a record WITH a policy is revised unless
    #    it is exactly "non_revised" (a casing accident must never admit
    #    revised data as point-in-time). Records WITHOUT a policy (bars,
    #    facts) are unaffected - "no policy" is not "revised".
    policy = (timing.series_policy or "").casefold()
    if policy not in ("", "non_revised") and timing.vintage_date is None:
        return AvailabilityDecision(
            record_id=timing.record_id, as_of_time=t, allowed=False,
            code=DecisionCode.NOT_POINT_IN_TIME,
            detail=(
                "series is revised and only the latest vintage is recorded; "
                "the value at as_of cannot be established, so the record is "
                "excluded (no silent substitution of today's revision)"
            ),
        )

    # 1. missing publication -> never available
    if timing.publication_time is None:
        return AvailabilityDecision(
            record_id=timing.record_id, as_of_time=t, allowed=False,
            code=DecisionCode.MISSING_PUBLICATION_TIME,
            detail=(
                "no publication time is recorded; availability is never "
                "invented for a record with an unknown publication instant"
            ),
        )

    # 2. date precision (or unknown/null) -> next-day availability;
    #    datetime -> strict '<'. Unknown precision is treated as DATE (the
    #    conservative direction: never available before the next day)
    pub: datetime = timing.publication_time
    if timing.publication_precision != TimePrecision.DATETIME:
        available_instant = next_day_midnight(pub.date())
    else:
        available_instant = pub

    warnings: list[str] = []
    if timing.publication_precision != TimePrecision.DATETIME:
        warnings.append(
            f"publication known to the day ({pub.date().isoformat()}); "
            "next-day availability applied"
        )
    if timing.ingestion_time is not None and timing.ingestion_time > t:
        warnings.append(
            f"ingested {timing.ingestion_time.isoformat()} AFTER as_of; "
            "publication time (not ingestion) decides availability"
        )

    if available_instant >= t:
        return AvailabilityDecision(
            record_id=timing.record_id, as_of_time=t, allowed=False,
            code=DecisionCode.PUBLICATION_AT_OR_AFTER_AS_OF,
            detail=(
                f"available_instant={available_instant.isoformat()} is not "
                f"strictly before as_of={t.isoformat()}"
            ),
            warnings=tuple(warnings),
        )

    # 4. vintage data: a version must have been released before as_of
    if timing.vintage_date is not None and timing.vintage_date >= t.date():
        return AvailabilityDecision(
            record_id=timing.record_id, as_of_time=t, allowed=False,
            code=DecisionCode.NO_VINTAGE_AT_AS_OF,
            detail=(
                f"vintage {timing.vintage_id or timing.vintage_date.isoformat()} "
                f"released {timing.vintage_date.isoformat()} is not strictly "
                f"before as_of date {t.date().isoformat()}"
            ),
            warnings=tuple(warnings),
        )

    # 5. forward-dated event records are not historical knowledge
    if timing.event_time is not None and timing.event_time > t:
        return AvailabilityDecision(
            record_id=timing.record_id, as_of_time=t, allowed=False,
            code=DecisionCode.EVENT_AFTER_AS_OF,
            detail=(
                f"event_time {timing.event_time.isoformat()} is after "
                f"as_of={t.isoformat()}; forward-dated records are excluded "
                "from the historical information set"
            ),
            warnings=tuple(warnings),
        )

    code = (
        DecisionCode.ALLOWED_VINTAGE_RESOLVED
        if timing.vintage_date is not None
        else (
            DecisionCode.ALLOWED_DATE_PRECISION
            if timing.publication_precision == TimePrecision.DATE
            else DecisionCode.ALLOWED_BEFORE_PUBLICATION
        )
    )
    return AvailabilityDecision(
        record_id=timing.record_id, as_of_time=t, allowed=True, code=code,
        detail=f"publication available at {available_instant.isoformat()} < {t.isoformat()}",
        warnings=tuple(warnings),
    )


def decide_frame(
    timings: list[Timing] | list[dict[str, Any]],
    as_of: datetime | Any,
) -> list[AvailabilityDecision]:
    """Vector wrapper over `decide` for lists of Timing objects or dicts."""
    decisions: list[AvailabilityDecision] = []
    for item in timings:
        if isinstance(item, Timing):
            decisions.append(decide(item, as_of))
        else:
            decisions.append(decide(Timing(**item), as_of))
    return decisions


def reasons_summary(decisions: list[AvailabilityDecision]) -> dict[str, int]:
    """Count decisions by code (used by tests and audit output)."""
    counts: dict[str, int] = {}
    for d in decisions:
        counts[d.code.value] = counts.get(d.code.value, 0) + 1
    return counts


@dataclass
class RuleTrace:
    """Full trace of one engine evaluation: what each rule saw."""

    record_id: str
    normalized_as_of: datetime
    publication_time: datetime | None
    publication_precision: str | None
    available_instant: datetime | None
    event_time: datetime | None
    ingestion_time: datetime | None
    vintage_date: str | None
    series_policy: str | None
    decision: AvailabilityDecision = field(default=None)  # type: ignore[assignment]


def trace_rule(timing: Timing, as_of: datetime | Any) -> RuleTrace:
    """Evaluate one record and expose every intermediate value.

    This is the audit half of the engine: "why exactly was this record
    allowed/rejected?" is answered by the trace, not by re-reading code.
    """
    t = normalize_instant(as_of)
    pub = timing.publication_time
    available_instant = None
    if pub is not None:
        if timing.publication_precision == TimePrecision.DATE:
            available_instant = next_day_midnight(pub.date())
        else:
            available_instant = pub
    decision = decide(timing, t)
    return RuleTrace(
        record_id=timing.record_id,
        normalized_as_of=t,
        publication_time=pub,
        publication_precision=timing.publication_precision.value,
        available_instant=available_instant,
        event_time=timing.event_time,
        ingestion_time=timing.ingestion_time,
        vintage_date=timing.vintage_date.isoformat() if timing.vintage_date else None,
        series_policy=timing.series_policy,
        decision=decision,
    )
