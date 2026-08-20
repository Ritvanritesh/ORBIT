"""Phase 9 data access.

Phase 9 performs no new data acquisition. It reads the existing validated
Phase 3 market snapshot (DS-000004) and the instrument master exactly as the
rest of ORBIT does: `orbit.ingestion.paths.normalized_dir` for bars/events and
the config-driven instrument master for identities.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl

from orbit.ingestion.paths import normalized_dir
from orbit.schemas.instrument import Instrument

_REPO_ROOT = Path(__file__).resolve().parents[3]

DEV_SNAPSHOT_ID = "DS-000004"
DEV_MASTER_FILE = _REPO_ROOT / "configs" / "instrument_master_dev.json"
DEV_MANIFESTS_DIR = _REPO_ROOT / "data" / "manifests"


def load_instrument_master(path: Path = DEV_MASTER_FILE) -> list[Instrument]:
    """Load and validate the dev instrument master through the Phase 2 schema."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    instruments = [Instrument(**row) for row in payload["instruments"]]
    instruments.sort(key=lambda i: i.instrument_id)
    return instruments


def load_snapshot_bars(snapshot_id: str = DEV_SNAPSHOT_ID) -> pl.DataFrame:
    """Load the normalized OHLCV bars of a market snapshot."""
    return pl.read_parquet(normalized_dir("market", "yahoo_chart_api", snapshot_id) / "bars.parquet")


def load_snapshot_events(snapshot_id: str = DEV_SNAPSHOT_ID) -> pl.DataFrame | None:
    """Load the corporate-actions artifact (splits + dividends) if present."""
    p = normalized_dir("market", "yahoo_chart_api", snapshot_id) / "events.parquet"
    return pl.read_parquet(p) if p.exists() else None


def load_snapshot_manifest(snapshot_id: str = DEV_SNAPSHOT_ID) -> dict[str, Any]:
    """Load the Phase 3 snapshot manifest (identity, lineage, constraints)."""
    p = DEV_MANIFESTS_DIR / f"{snapshot_id}.json"
    if not p.exists():
        raise FileNotFoundError(f"snapshot manifest not found: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def session_series(bars: pl.DataFrame) -> pl.DataFrame:
    """Sorted unique decision sessions of the snapshot (used as decision grid)."""
    out = (
        bars.select("trade_date")
        .unique()
        .sort("trade_date")
        .with_columns(pl.col("trade_date").alias("session"))
    )
    return out


def bars_meta(bars: pl.DataFrame) -> dict[str, Any]:
    """Compact, deterministic summary of the loaded snapshot for lineage."""
    meta = {
        "rows": bars.height,
        "sessions": bars["trade_date"].n_unique(),
        "instruments": bars["instrument_id"].n_unique(),
        "first_session": str(bars["trade_date"].min()),
        "last_session": str(bars["trade_date"].max()),
        "symbols": sorted(bars["symbol"].unique().to_list()),
    }
    return meta