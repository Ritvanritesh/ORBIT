"""Macro normalization (FRED series).

One row per observation; nulls (provider ".") preserved as-is; the vintage
note records whether this is the latest published vintage or an ALFRED
vintage request, so no one mistakes today's revisions for history.

Schema v1.1.0 adds `vintage_date`: the ALFRED release date of this version
when the snapshot was requested with vintage_date=... (null for
latest-vintage snapshots). The Phase 4 Temporal Truth Engine uses
vintage_date to resolve "what value was known at time T"; the Phase 3
v1.0.0 files (no column) keep working via the `vintage_note` marker.
"""

from __future__ import annotations

import polars as pl

MACRO_SCHEMA_VERSION = "v1.1.0"

SERIES_COLUMNS: dict[str, pl.DataType] = {
    "series_id": pl.Utf8,
    "observation_date": pl.Date,
    "value": pl.Float64,
    "vintage_date": pl.Date,
    "vintage_note": pl.Utf8,
    "provider": pl.Utf8,
    "snapshot_id": pl.Utf8,
}


def normalize_fred_series(
    parsed: pl.DataFrame,
    vintage_note: str,
    provider: str,
    snapshot_id: str,
    vintage_date: str | None = None,
) -> pl.DataFrame:
    vintage_col = (
        pl.lit(vintage_date).str.to_date("%Y-%m-%d", strict=False)
        if vintage_date
        else pl.lit(None, dtype=pl.Date)
    )
    return (
        parsed.with_columns(
            pl.col("observation_date").cast(pl.Date),
            pl.col("value").cast(pl.Float64),
            vintage_col.alias("vintage_date"),
            pl.lit(vintage_note).alias("vintage_note"),
            pl.lit(provider).alias("provider"),
            pl.lit(snapshot_id).alias("snapshot_id"),
        )
        .sort("observation_date")
        .select(list(SERIES_COLUMNS))
    )