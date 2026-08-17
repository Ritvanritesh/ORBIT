"""Data contracts: immutable dataset snapshots and market bars.

DatasetSnapshot is the provenance root: every experiment pins the exact
snapshot it consumed (roadmap 27.2). MarketBar is the normalized daily bar
schema for the EOD-first research scope.
"""

from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel, ConfigDict, Field, model_validator


class DatasetSnapshot(BaseModel):
    """An immutable, checksummed delivery of a dataset."""

    model_config = ConfigDict(frozen=True)

    snapshot_id: str = Field(pattern=r"^DS-\d{6}$")
    provider: str
    source_uri: str
    checksum: str = Field(
        min_length=32,
        description="Content hash of the raw payload (e.g. sha256 hex).",
    )
    schema_version: str = Field(pattern=r"^v\d+(\.\d+)*$")
    available_from: date
    available_to: date
    ingest_time: datetime
    license_ref: str | None = None

    @model_validator(mode="after")
    def _check_range(self) -> "DatasetSnapshot":
        if self.available_to < self.available_from:
            raise ValueError("available_to cannot precede available_from")
        return self


class MarketBar(BaseModel):
    """Normalized daily bar. Timestamps are UTC, exchange-tz documented."""

    model_config = ConfigDict(frozen=True)

    instrument_id: str = Field(pattern=r"^INS-\d{6}$")
    ts: datetime = Field(description="UTC bar timestamp")
    open: float = Field(gt=0)
    high: float = Field(gt=0)
    low: float = Field(gt=0)
    close: float = Field(gt=0)
    volume: float = Field(ge=0)
    trade_count: int | None = None

    @model_validator(mode="after")
    def _check_ohlc(self) -> "MarketBar":
        if not (self.low <= self.open <= self.high and self.low <= self.close <= self.high):
            raise ValueError("OHLC inconsistent: low <= open/close <= high required")
        return self