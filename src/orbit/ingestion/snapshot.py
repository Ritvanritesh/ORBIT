"""DatasetSnapshot construction and the market DataAccessor.

A DatasetSnapshot connects: provider -> raw files -> checksums -> normalized
data -> date range -> instruments -> schema version. Any experiment can say
"I was trained using DatasetSnapshot X" and reproduce it exactly.

MarketDataAccessor implements the Phase 2 DataAccessor protocol over the
normalized parquet layer, so the UniverseEngine runs on real ingested data
with strictly-lagged reads (Phase 4 will stress-test this with adversarial
fixtures).
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl

from orbit.ingestion.paths import normalized_dir
from orbit.schemas.data import DatasetSnapshot
from orbit.schemas.instrument import Instrument


def build_dataset_snapshot(
    registry: Any,
    snapshot_id: str,
    domain: str,
    provider: str,
    *,
    instrument_count: int | None = None,
) -> DatasetSnapshot:
    """Build the immutable delivery record from the registry + manifest."""
    record = registry.snapshot(snapshot_id)
    if record is None:
        raise KeyError(f"unknown snapshot: {snapshot_id}")
    manifest = _load_manifest(record.get("manifest_path"))
    earliest = manifest.get("date_range", [None, None])[0]
    latest = manifest.get("date_range", [None, None])[1]
    ingested = record["downloaded_at"]
    if isinstance(ingested, str):
        ingested = datetime.fromisoformat(ingested)
    return DatasetSnapshot(
        snapshot_id=snapshot_id,
        provider=provider,
        source_uri=record["source_uri"],
        checksum=record["checksum"],
        schema_version=record["schema_version"],
        available_from=date.fromisoformat(earliest) if earliest else date.min,
        available_to=date.fromisoformat(latest) if latest else date.min,
        ingest_time=ingested,
        license_ref=record.get("license_ref"),
        row_count=record.get("row_count"),
        instrument_count=instrument_count,
        validation_status=record.get("validation_status"),
        manifest_path=record.get("manifest_path"),
    )


def _load_manifest(manifest_path: str | None) -> dict[str, Any]:
    if not manifest_path:
        return {}
    with open(manifest_path, "r", encoding="utf-8") as f:
        import json

        return json.load(f)


class MarketDataAccessor:
    """Read-only, strictly-lagged market data over a normalized snapshot.

    All reads are strict: a bar on as_of is NOT visible at as_of. Dollar
    volume uses close * volume; the median is taken over the window ending
    strictly before as_of (lagged liquidity, per the Phase 2 rule).

    Volume basis: for yahoo_chart_api, close AND volume are on the same
    split-adjusted share basis, so close*volume is true dollar volume. For
    a provider that ships RAW volume next to adjusted closes (e.g.
    stooq_csv), this product is discontinuous at splits and must be
    reconstructed from events before use.
    """

    def __init__(
        self,
        instruments: list[Instrument],
        snapshot_id: str,
        provider: str = "yahoo_chart_api",
        data_root: Path | None = None,
    ):
        self._instruments = instruments
        self._snapshot_id = snapshot_id
        self._bars: pl.DataFrame | None = None
        path = (
            (data_root / "normalized" / "market" / provider / snapshot_id / "bars.parquet")
            if data_root
            else normalized_dir("market", provider, snapshot_id) / "bars.parquet"
        )
        self._bars_path = path

    def instruments(self) -> list[Instrument]:
        return self._instruments

    def _load(self) -> pl.DataFrame:
        if self._bars is None:
            self._bars = pl.read_parquet(self._bars_path)
        return self._bars

    def _rows_for(self, instrument_id: str) -> pl.DataFrame:
        return self._load().filter(pl.col("instrument_id") == instrument_id)

    def trailing_dollar_volume(
        self, instrument_id: str, as_of: date, window_days: int
    ) -> float | None:
        rows = self._rows_for(instrument_id)
        dv = rows.filter(pl.col("trade_date") < as_of).select(
            (pl.col("close") * pl.col("volume")).alias("dv")
        )
        if dv.height == 0:
            return None
        window = dv.tail(window_days)["dv"].to_list()
        if not window:
            return None
        ordered = sorted(window)
        mid = len(ordered) // 2
        return float(ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2)

    def last_close(self, instrument_id: str, as_of: date) -> float | None:
        rows = self._rows_for(instrument_id).filter(pl.col("trade_date") < as_of).sort("trade_date")
        if rows.height == 0:
            return None
        return float(rows["close"][-1])

    def bars_between(self, instrument_id: str, start: date, end: date) -> pl.DataFrame:
        """All bars in [start, end] inclusive - for reconciliation only."""
        return self._rows_for(instrument_id).filter(
            (pl.col("trade_date") >= start) & (pl.col("trade_date") <= end)
        )