"""Fundamentals normalization (SEC company facts).

Phase 3 is acquisition and preservation, not feature engineering: the
normalized view is a mechanical flattening of the raw companyfacts JSON
(one row per reported fact, with filing metadata preserved verbatim).
Point-in-time correctness is Phase 4's job.
"""

from __future__ import annotations

import polars as pl

FUNDAMENTALS_SCHEMA_VERSION = "v1.0.0"

FACT_COLUMNS: dict[str, pl.DataType] = {
    "cik": pl.Int64,
    "entity_name": pl.Utf8,
    "taxonomy": pl.Utf8,
    "fact": pl.Utf8,
    "unit": pl.Utf8,
    "val": pl.Float64,
    "start": pl.Utf8,
    "end": pl.Utf8,
    "accn": pl.Utf8,
    "fy": pl.Int64,
    "fp": pl.Utf8,
    "form": pl.Utf8,
    "filed": pl.Utf8,
    "frame": pl.Utf8,
    "provider": pl.Utf8,
    "snapshot_id": pl.Utf8,
}


def normalize_sec_facts(
    parsed: pl.DataFrame, provider: str, snapshot_id: str
) -> pl.DataFrame:
    """Flatten + cast. Non-numeric fact values (rare text facts) become null
    in the normalized view; the raw JSON preserves them verbatim."""
    return (
        parsed.with_columns(
            pl.col("val").cast(pl.Float64, strict=False),
            pl.lit(provider).alias("provider"),
            pl.lit(snapshot_id).alias("snapshot_id"),
        )
        .select(list(FACT_COLUMNS))
    )