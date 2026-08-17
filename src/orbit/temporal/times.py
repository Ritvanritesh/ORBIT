"""Temporal fields and time conventions for the Phase 4 Temporal Truth Engine.

ORBIT distinguishes seven temporal concepts. They are NOT interchangeable.

    event_time         When the real-world event happened. Example: a
                       company's quarter ended 2018-03-31.
    publication_time   When the information became publicly available.
                       Example: the earnings report was released 2018-04-25.
    ingestion_time     When ORBIT downloaded/received the information.
                       Example: ORBIT downloaded it 2018-04-25 18:00 UTC.
                       ingestion_time is provenance, NEVER availability:
                       the market knew the report at release, not when ORBIT
                       pulled it.
    effective_time     When the information becomes applicable for a data
                       type. For filings, effective = publication; for a
                       daily bar, effective = the session close.
    feature_time       The decision time a derived feature is attached to.
                       A feature at feature_time T may only use records
                       available strictly before T.
    as_of_time         The point in time the engine is asked: "what was
                       known exactly now?"
    vintage_id         Identifier of one particular version/revision of a
                       value. Vital for revised macro data: the value that
                       existed at a historical date can differ from today's
                       revision.

Every availability decision is made on a UTC-naive datetime line so that
timezone mistakes are structurally impossible in comparisons (the
conversion happens in exactly one place: `normalize_instant`).

CONVENTIONS (documented, testable, conservative)

1. STRICT BOUNDARY. A record is available at as_of iff
       publication_time < as_of
   A record published at EXACTLY as_of is NOT available. "One minute
   before publication" rejects; "one minute after" allows. Nothing is
   ever available at its own publication instant.

2. DATE PRECISION -> NEXT DAY. When a source gives only a publication
   DATE (no intraday time), ORBIT does not invent a timestamp. The
   record is treated as becoming available at 00:00:00 UTC of the
   FOLLOWING calendar day. A 10-Q "filed 2018-04-25" is therefore not
   available to a decision at 2018-04-25 16:00:00 (the market may have
   known it intraday; ORBIT cannot prove when, so it rejects - false
   acceptance is the dangerous error).

3. MARKET SESSION CLOSE. A daily bar for session D becomes available at
   the session close, 16:00 America/New_York on D (converted to UTC).
   The bar's ts_utc (provider bar timestamp, typically session OPEN) is
   NOT availability - using it would leak the day's close before it
   existed. A date-only as_of (start of day) therefore never sees the
   same-day bar: bar D is available iff as_of_date > D.

4. EVENT AFTER AS_OF -> REJECT. Information describing an event that
   happens after the decision time is excluded from the historical
   information set (forward-dated records cannot be part of "what was
   known"). Publication still gates everything; this rule is a second,
   documented conservative filter.

5. MISSING PUBLICATION -> REJECT. A record with no publication time is
   NEVER available. Availability is never invented.

6. REVISED MACRO, NO VINTAGE -> REJECT. A revised series served only as
   today's latest vintage is not point-in-time trustworthy; those
   observations are excluded and reported as limitations, never
   substituted.

All conventions live here and in configs/temporal.json so future code can
rely on them without re-deriving intent.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from enum import Enum
from typing import Any
from zoneinfo import ZoneInfo

EXCHANGE_TZ = ZoneInfo("America/New_York")
SESSION_CLOSE_LOCAL = time(16, 0)

TOTAL_TIME = time(23, 59, 59, 999999)


class TimePrecision(str, Enum):
    """How precisely a timestamp is known."""

    DATE = "date"
    DATETIME = "datetime"


class DecisionCode(str, Enum):
    """Every availability decision carries one of these codes (provenance)."""

    # --- allow
    ALLOWED_BEFORE_PUBLICATION = "allowed_publication_strictly_before"
    ALLOWED_DATE_PRECISION = "allowed_date_precision_next_day"
    ALLOWED_VINTAGE_RESOLVED = "allowed_vintage_resolved"

    # --- reject
    MISSING_PUBLICATION_TIME = "reject_missing_publication_time"
    PUBLICATION_AT_OR_AFTER_AS_OF = "reject_publication_at_or_after_as_of"
    EVENT_AFTER_AS_OF = "reject_event_after_as_of"
    NOT_POINT_IN_TIME = "reject_not_point_in_time_series"
    NO_VINTAGE_AT_AS_OF = "reject_no_vintage_at_as_of"
    VINTAGE_SUPERSEDED = "excluded_vintage_superseded"

    # --- warnings (never reject)
    INGESTED_AFTER_AS_OF = "warn_ingested_after_as_of"
    DATE_PRECISION_NOTE = "note_date_precision_publication"


def normalize_instant(value: date | datetime | str | None) -> datetime | None:
    """Normalize any accepted instant to a naive UTC datetime.

    - date            -> that date at 00:00:00 (start of day)
    - naive datetime  -> used as-is (ORBIT convention: naive == UTC)
    - tz-aware        -> converted to UTC and made naive
    - ISO-8601 string -> parsed then normalized
    - None            -> None
    """
    if value is None:
        return None
    if isinstance(value, str):
        s = value
        # ISO-8601 'Z' (UTC designator): fromisoformat only learns it in
        # Python 3.11+, so translate it to an explicit offset on 3.10
        if s.endswith(("Z", "z")):
            s = s[:-1] + "+00:00"
        value = datetime.fromisoformat(s)
    if isinstance(value, date) and not isinstance(value, datetime):
        return datetime(value.year, value.month, value.day)
    if not isinstance(value, datetime):
        raise TypeError(
            f"cannot normalize {type(value).__name__!r} to an instant; "
            "expected date, datetime, ISO-8601 string or None"
        )
    dt: datetime = value
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def session_close_utc(session: date) -> datetime:
    """16:00 America/New_York on `session`, as naive UTC.

    Handles DST: 16:00 ET is 21:00 UTC in EST and 20:00 UTC in EDT. The
    ZoneInfo conversion is the ONLY place this is computed, so every
    consumer (rules, tests, features) agrees.
    """
    local = datetime.combine(session, SESSION_CLOSE_LOCAL, tzinfo=EXCHANGE_TZ)
    return local.astimezone(timezone.utc).replace(tzinfo=None)


def next_day_midnight(d: date) -> datetime:
    """00:00:00 UTC of the day after `d` (date-precision convention)."""
    from datetime import timedelta

    nxt = d + timedelta(days=1)
    return datetime(nxt.year, nxt.month, nxt.day)


def end_of_day_utc(d: date) -> datetime:
    """23:59:59.999999 UTC of `d` (an exact instant for boundary tests)."""
    return datetime(d.year, d.month, d.day, TOTAL_TIME.hour, TOTAL_TIME.minute,
                    TOTAL_TIME.second, TOTAL_TIME.microsecond)


@dataclass(frozen=True)
class Timing:
    """The canonical timing of one information record.

    `publication_precision` records whether publication_time was given to
    the day or to the instant; the engine applies the NEXT-DAY convention
    only for DATE precision.

    All instants are normalized at construction (single point
    `normalize_instant`): ISO strings and `date` values become naive-UTC
    datetimes, so a caller can never hand a comparison a string by accident.
    """

    record_id: str
    domain: str
    kind: str
    event_time: datetime | None = None
    publication_time: datetime | None = None
    publication_precision: TimePrecision = TimePrecision.DATETIME
    effective_time: datetime | None = None
    ingestion_time: datetime | None = None
    vintage_id: str | None = None
    vintage_date: date | None = None
    series_policy: str | None = None
    payload: dict[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        for field_name in (
            "event_time", "publication_time", "effective_time", "ingestion_time",
        ):
            value = getattr(self, field_name)
            if value is not None:
                # normalize_instant passes naive datetimes through unchanged,
                # converts aware instants to naive UTC, and lifts date/str
                # values to datetimes - so a comparison can never receive a
                # raw aware datetime or a string by accident
                object.__setattr__(self, field_name, normalize_instant(value))
        if isinstance(self.publication_precision, str):
            # accept the plain value ("date"/"datetime") as well as the enum;
            # a bare string falling through would silently downgrade DATE
            # precision to DATETIME and make the record available a day early
            object.__setattr__(
                self, "publication_precision", TimePrecision(self.publication_precision)
            )
        if isinstance(self.vintage_date, str):
            object.__setattr__(self, "vintage_date", date.fromisoformat(self.vintage_date))

    def as_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "domain": self.domain,
            "kind": self.kind,
            "event_time": self.event_time.isoformat() if self.event_time else None,
            "publication_time": (
                self.publication_time.isoformat() if self.publication_time else None
            ),
            "publication_precision": self.publication_precision.value,
            "effective_time": self.effective_time.isoformat() if self.effective_time else None,
            "ingestion_time": self.ingestion_time.isoformat() if self.ingestion_time else None,
            "vintage_id": self.vintage_id,
            "vintage_date": self.vintage_date.isoformat() if self.vintage_date else None,
            "series_policy": self.series_policy,
            "payload": self.payload or {},
        }
