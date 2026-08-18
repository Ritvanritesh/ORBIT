"""Release calendar: deterministic, data-driven knowledge of scheduled
publication instants (prompt section 8).

Purpose
-------
Most sources give ORBIT a publication DATE only (SEC filed date, ALFRED
vintage date), so the engine conservatively treats the record as available
only from the NEXT day. Some series have a KNOWN intraday release schedule
(e.g. a macro agency publishes at a fixed local time). The release
calendar records such schedules and lets the FRED adapter sharpen a
date-precision publication into the exact scheduled UTC instant.

Safety model (hard rule)
------------------------
An enabled calendar entry makes information available EARLIER than the
next-day default. A wrong entry is therefore a leak vector, not just an
inaccuracy. Two independent guards must both pass before an entry has any
effect:

  1. `enabled: true` - the entry is deliberately activated; the default is
     OFF and every new entry is inert until someone with authority turns
     it on.
  2. `evidence` - a non-empty, verifiable description of where the
     schedule comes from (provider documentation, contract with the
     publisher, observed release history). The documentation in
     docs/phase4_temporal_truth.md states the activation policy.

Additional structural guards:
  - the weekday of the entry MUST match the weekday of the actual release
    date (a holiday-shifted release falls back to the next-day default
    instead of being sharpened with the wrong date);
  - release times are zone-aware (America/New_York by default, DST handled
    by real ZoneInfo conversion, exactly like the session close);
  - the entry is keyed to one series_id and returns ONE scheduled instant
    per release date, deterministically.

The config ships with an EMPTY calendar: ORBIT's current sources do not
depend on it, and no schedule has been verified by ORBIT yet. When a
schedule is verified, the entry is added with enabled: true and evidence.
"""

from __future__ import annotations

from datetime import date, datetime, time, timezone
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, Field, field_validator

_RELEASE_TZ_DEFAULT = "America/New_York"


class ReleaseCalendarEntry(BaseModel):
    """One scheduled publication time for one series.

    series_id     the macro series this schedule applies to
    release_time  "HH:MM" local wall-clock time of the release
    timezone      IANA zone of release_time (default America/New_York)
    weekday       datetime.weekday() convention: 0 = Monday .. 6 = Sunday
    evidence      where the schedule comes from (required, documented)
    enabled       OFF by default; only explicitly activated entries sharpen
                  availability (an unverified entry must stay inert)
    """

    model_config = ConfigDict(frozen=True)

    series_id: str
    release_time: str = Field(pattern=r"^\d{2}:\d{2}$")
    timezone: str = _RELEASE_TZ_DEFAULT
    weekday: int = Field(ge=0, le=6)
    evidence: str = Field(min_length=1)
    enabled: bool = False

    @field_validator("timezone")
    @classmethod
    def _valid_zone(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except Exception as exc:  # zoneinfo.ZoneInfoNotFoundError
            raise ValueError(f"unknown timezone in release calendar: {value!r}") from exc
        return value

    @field_validator("release_time")
    @classmethod
    def _valid_clock(cls, value: str) -> str:
        try:
            hour, minute = (int(p) for p in value.split(":"))
            time(hour, minute)
        except ValueError as exc:
            raise ValueError(f"invalid release_time: {value!r} (expected HH:MM)") from exc
        return value

    def scheduled_instant(self, release_date: date) -> datetime | None:
        """The exact naive-UTC instant of the release on `release_date`.

        Returns None when the entry is disabled or the release weekday does
        not match the entry's weekday (holiday-shifted releases fall back
        to the conservative next-day convention).
        """
        if not self.enabled:
            return None
        if release_date.weekday() != self.weekday:
            return None
        hour, minute = (int(p) for p in self.release_time.split(":"))
        local = datetime.combine(
            release_date, time(hour, minute), tzinfo=ZoneInfo(self.timezone)
        )
        return local.astimezone(timezone.utc).replace(tzinfo=None)


class ReleaseCalendar(BaseModel):
    """The full calendar: a deterministic map of series -> release schedule."""

    model_config = ConfigDict(frozen=True)

    entries: list[ReleaseCalendarEntry] = Field(default_factory=list)

    def entry_for(self, series_id: str) -> ReleaseCalendarEntry | None:
        for entry in self.entries:
            if entry.series_id == series_id:
                return entry
        return None


__all__ = ["ReleaseCalendar", "ReleaseCalendarEntry"]
