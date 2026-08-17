"""Macro normalization (FRED series).

One row per observation; nulls (provider ".") preserved as-is; the vintage
note records whether this is the latest published vintage or an ALFRED
vintage request, so no one mistakes today's revisions for history.
"""

from __future__ import annotations

import polars as pl

MACRO_SCHEMA_VERSION = "v1.0.0"

SERIES_COLUMNS: dict[str, pl.DataType] = {
    "series_id": pl.Utf8,
    "observation_date": pl.Date,
    "value": pl.Float64,
    "vintage_note": pl.Utf8,
    "provider": pl.Utf8,
    "snapshot_id": pl.Utf8,
}


def normalize_fred_series(
    parsed: pl.DataFrame, vintage_note: str, provider: str, snapshot_id: str
) -> pl.DataFrame:
    return (
        parsed.with_columns(
            pl.col("observation_date").cast(pl.Date),
            pl.col("value").cast(pl.Float64),
            pl.lit(vintage_note).alias("vintage_note"),
            pl.lit(provider).alias("provider"),
            pl.lit(snapshot_id).alias("snapshot_id"),
        )
        .select(list(SERIES_COLUMNS))
    )