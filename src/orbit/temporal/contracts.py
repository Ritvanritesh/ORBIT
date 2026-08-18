"""Temporal contracts and configuration.

The temporal contract is the written promise every future feature/model
code must obey: for every data type it states WHAT happened, WHEN it
happened, WHEN it was published, WHEN ORBIT received it, WHICH version it
is, and WHEN a prediction may use it. The machine-readable half lives in
configs/temporal.json; this module loads it and exposes the conventions.

Conservative defaults (never silent):
  - an unknown series revision status is treated as `revised` (reject)
  - a record with an unknown publication time is never available
  - date-precision publication becomes available the NEXT day
  - the availability boundary is strict: publication < as_of

The convention fields (boundary, date_precision, exchange_tz,
session_close, market_bar_available, forward_dated_events) are implemented
as constants in orbit.temporal.times and cannot be changed by editing the
config alone. load_temporal_contract() therefore REFUSES a config whose
convention fields diverge from the implementation: a config that promises
different behavior than the code is a silent bypass waiting to happen.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from orbit.temporal.calendar import ReleaseCalendar, ReleaseCalendarEntry

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = _REPO_ROOT / "configs" / "temporal.json"

DEFAULT_SERIES_POLICIES: dict[str, str] = {
    # Series revision policy, used ONLY when a series has no ALFRED vintage
    # history. non_revised = the latest-vintage value for an observation is
    # the value published for it (available the day after the observation).
    # revised = the value may have been changed after publication; without
    # vintage history the as-published value cannot be established.
    "DFF": "non_revised",      # effective federal funds rate: daily, not revised
    "CPIAUCSL": "revised",     # CPI: routinely revised
    "UNRATE": "revised",       # unemployment rate: routinely revised
}

# The conventions implemented as constants in times.py / rules.py. A config
# that names a different convention is rejected at load time - the config
# must mirror the code, never silently promise different behavior.
# `default_series_policy` is here too: the conservative default for an
# UNKNOWN series is "revised" (reject point-in-time). Flipping it to
# "non_revised" would silently admit revised series without vintage history.
IMPLEMENTED_CONVENTIONS: dict[str, str] = {
    "boundary": "strict_publication_less_than_asof",
    "date_precision": "next_day_utc_midnight",
    "exchange_tz": "America/New_York",
    "session_close": "16:00",
    "market_bar_available": "session_close",
    "forward_dated_events": "exclude",
    "default_series_policy": "revised",
}

VALID_POLICIES = frozenset({"revised", "non_revised"})


def _normalize_policy(value: str) -> str:
    p = value.strip().casefold()
    if p not in VALID_POLICIES:
        raise ValueError(
            f"invalid series policy {value!r}: expected 'revised' or "
            "'non_revised'"
        )
    return p


class TemporalContract(BaseModel):
    """The loaded, validated temporal configuration."""

    model_config = ConfigDict(frozen=True)

    engine_version: str = Field(pattern=r"^v\d+(\.\d+)*$", default="v1.0.0")
    boundary: str = Field(
        default="strict_publication_less_than_asof",
        description="availability requires publication_time strictly before as_of",
    )
    date_precision: str = Field(
        default="next_day_utc_midnight",
        description="date-precision publications become available at 00:00 UTC the next day",
    )
    exchange_tz: str = "America/New_York"
    session_close: str = "16:00"
    market_bar_available: str = Field(
        default="session_close",
        description="a daily bar is available at the session close, never at ts_utc",
    )
    series_policies: dict[str, str]
    default_series_policy: str = "revised"
    forward_dated_events: str = "exclude"
    release_calendar: ReleaseCalendar = Field(default_factory=ReleaseCalendar)

    def policy_for(self, series_id: str) -> str:
        return self.series_policies.get(series_id, self.default_series_policy)

    @field_validator("series_policies")
    @classmethod
    def _valid_policy_values(cls, v: dict[str, str]) -> dict[str, str]:
        # a casing accident or trailing whitespace must not silently change
        # a series' revision status; every value is normalized to the two
        # canonical labels
        return {k: _normalize_policy(p) for k, p in v.items()}

    @field_validator("default_series_policy")
    @classmethod
    def _valid_default_policy(cls, v: str) -> str:
        return _normalize_policy(v)


def load_temporal_contract(path: str | Path | None = None) -> TemporalContract:
    """Load the temporal contract from configs/temporal.json (or defaults).

    Raises ValueError when a convention field diverges from the implemented
    constants (the config must never promise behavior the code does not
    implement) or when a release-calendar entry is invalid.
    """
    p = Path(path) if path else DEFAULT_CONFIG_PATH
    data: dict[str, Any] = {}
    if p.exists():
        data = json.loads(p.read_text(encoding="utf-8"))
    for key, expected in IMPLEMENTED_CONVENTIONS.items():
        if key in data and data[key] != expected:
            raise ValueError(
                f"temporal config {p} sets {key}={data[key]!r} but the engine "
                f"implements {key}={expected!r}; conventions are constants in "
                "orbit.temporal.times and cannot be changed by editing the "
                "config alone"
            )
    merged = {
        "engine_version": data.get("engine_version", "v1.0.0"),
        "series_policies": {
            **DEFAULT_SERIES_POLICIES,
            **data.get("series_policies", {}),
        },
        "default_series_policy": data.get("default_series_policy", "revised"),
        "exchange_tz": data.get("exchange_tz", "America/New_York"),
        "session_close": data.get("session_close", "16:00"),
        "boundary": data.get("boundary", "strict_publication_less_than_asof"),
        "date_precision": data.get("date_precision", "next_day_utc_midnight"),
        "market_bar_available": data.get("market_bar_available", "session_close"),
        "forward_dated_events": data.get("forward_dated_events", "exclude"),
        "release_calendar": ReleaseCalendar(
            entries=[
                ReleaseCalendarEntry(**entry)
                for entry in data.get("release_calendar", [])
            ]
        ),
    }
    return TemporalContract(**merged)