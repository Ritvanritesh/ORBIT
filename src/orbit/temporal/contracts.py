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
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

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

    def policy_for(self, series_id: str) -> str:
        return self.series_policies.get(series_id, self.default_series_policy)


def load_temporal_contract(path: str | Path | None = None) -> TemporalContract:
    """Load the temporal contract from configs/temporal.json (or defaults)."""
    p = Path(path) if path else DEFAULT_CONFIG_PATH
    data: dict[str, Any] = {}
    if p.exists():
        data = json.loads(p.read_text(encoding="utf-8"))
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
    }
    return TemporalContract(**merged)