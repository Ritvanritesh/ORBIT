"""Adapters: turn normalized parquet artifacts into canonical timing frames.

A timing frame is the universal currency of the Temporal Truth Engine -
one row per information record with the canonical temporal columns:

    record_id             stable id (digest-safe, unique per record)
    source_key            stable identity of the thing the record is about
                          (instrument_id | cik | series_id)
    domain                market | fundamentals | macro
    kind                  bar | fact | observation
    event_time            when the described event happened (naive UTC)
    publication_time      when it became public (naive UTC; DATE-precision
                          records carry the date at 00:00:00)
    publication_precision "date" | "datetime"
    effective_time        when it becomes applicable
    ingestion_time        when ORBIT downloaded the snapshot (provenance)
    vintage_id            version id (macro revisions; null elsewhere)
    vintage_date          release date of this version (null elsewhere)
    series_policy         non_revised | revised (macro only)
    payload_json          canonical JSON of the source row

Adapters NEVER decide availability - they only attach the facts about the
record. The rules in orbit.temporal.rules decide.
"""

from __future__ import annotations

import json
import math
from datetime import date, datetime
from typing import Any

import polars as pl

from orbit.temporal.times import TimePrecision, session_close_utc

TIMING_SCHEMA: dict[str, pl.DataType] = {
    "record_id": pl.Utf8,
    "source_key": pl.Utf8,
    "domain": pl.Utf8,
    "kind": pl.Utf8,
    "event_time": pl.Datetime("us"),
    "publication_time": pl.Datetime("us"),
    "publication_precision": pl.Utf8,
    "effective_time": pl.Datetime("us"),
    "ingestion_time": pl.Datetime("us"),
    "vintage_id": pl.Utf8,
    "vintage_date": pl.Date,
    "series_policy": pl.Utf8,
    "payload_json": pl.Utf8,
}


def empty_timing_frame() -> pl.DataFrame:
    return pl.DataFrame(schema=TIMING_SCHEMA)


def _payload(row: dict[str, Any]) -> str:
    clean = {
        k: (None if isinstance(v, float) and math.isnan(v) else v)
        for k, v in row.items()
    }
    return json.dumps(clean, sort_keys=True, default=str, separators=(",", ":"))


def _day_start(d: date | None) -> datetime | None:
    if d is None:
        return None
    return datetime(d.year, d.month, d.day)


def _datetime_expr(series: pl.Series) -> pl.Series:
    """Convert a Date series to a naive-UTC Datetime series (day start)."""
    return series.map_elements(_day_start, return_dtype=pl.Datetime("us"))


def market_timing_frame(
    bars: pl.DataFrame,
    snapshot_id: str,
    ingestion_time: datetime | None = None,
) -> pl.DataFrame:
    """Timing frame for canonical daily bars.

    event_time        = trade_date (the session the bar describes)
    publication_time  = session close 16:00 America/New_York on trade_date
                        (DATETIME precision). The provider's ts_utc (session
                        OPEN for intraday-stamped feeds) is deliberately NOT
                        used as publication: that would make the day's close
                        available at the open.
    effective_time    = publication (an EOD bar is usable at the close).

    The payload EXCLUDES adjclose: the provider retroactively adjusts it for
    dividends/splits, so an adjclose value read at as_of can differ from what
    was publishable then. Point-in-time prices are built from `close` plus the
    corporate-action adjustment records (Phase 3 reconciliation), never from
    the retroactively adjusted column.
    """
    payload_cols = [c for c in bars.columns if c != "adjclose"]
    ingest = _day_start(ingestion_time)
    event = _datetime_expr(bars["trade_date"])
    close = bars["trade_date"].map_elements(
        session_close_utc, return_dtype=pl.Datetime("us")
    )
    return (
        bars.with_columns(
            (pl.col("instrument_id") + pl.lit("|") + pl.lit(snapshot_id)
             + pl.lit("|") + pl.col("trade_date").cast(pl.Utf8)).alias("record_id"),
            pl.col("instrument_id").alias("source_key"),
            pl.lit("market").alias("domain"),
            pl.lit("bar").alias("kind"),
            event.alias("event_time"),
            close.alias("publication_time"),
            pl.lit(TimePrecision.DATETIME.value).alias("publication_precision"),
            close.alias("effective_time"),
            pl.lit(ingest, dtype=pl.Datetime("us")).alias("ingestion_time"),
            pl.lit(None, dtype=pl.Utf8).alias("vintage_id"),
            pl.lit(None, dtype=pl.Date).alias("vintage_date"),
            pl.lit(None, dtype=pl.Utf8).alias("series_policy"),
            pl.struct(payload_cols).map_elements(_payload, return_dtype=pl.Utf8).alias("payload_json"),
        )
        .select(list(TIMING_SCHEMA))
    )


def sec_timing_frame(
    facts: pl.DataFrame,
    snapshot_id: str,
    ingestion_time: datetime | None = None,
) -> pl.DataFrame:
    """Timing frame for SEC companyfacts rows.

    event_time        = period end (start of day); null when the fact has no
                        end date (e.g. share-count facts)
    publication_time  = filed date at 00:00:00, DATE precision -> the rules
                        apply the next-day availability convention
    effective_time    = publication (a filing fact applies once filed)
    """
    end = _datetime_expr(
        facts["end"].cast(pl.Utf8).str.to_date("%Y-%m-%d", strict=False)
    )
    filed = _datetime_expr(
        facts["filed"].cast(pl.Utf8).str.to_date("%Y-%m-%d", strict=False)
    )
    return (
        facts.with_columns(
            (pl.lit("fact|") + pl.lit(snapshot_id) + pl.lit("|")
             + pl.col("accn").fill_null("?") + pl.lit("|")
             + pl.col("fact").fill_null("?") + pl.lit("|")
             + pl.col("unit").fill_null("?") + pl.lit("|")
             + pl.col("start").fill_null("?") + pl.lit("|")
             + pl.col("end").fill_null("?")).alias("record_id"),
            pl.col("cik").cast(pl.Utf8).alias("source_key"),
            pl.lit("fundamentals").alias("domain"),
            pl.lit("fact").alias("kind"),
            end.alias("event_time"),
            filed.alias("publication_time"),
            pl.lit(TimePrecision.DATE.value).alias("publication_precision"),
            filed.alias("effective_time"),
            pl.lit(ingest_time := _day_start(ingestion_time), dtype=pl.Datetime("us")).alias("ingestion_time"),
            pl.lit(None, dtype=pl.Utf8).alias("vintage_id"),
            pl.lit(None, dtype=pl.Date).alias("vintage_date"),
            pl.lit(None, dtype=pl.Utf8).alias("series_policy"),
            pl.struct(list(facts.columns)).map_elements(_payload, return_dtype=pl.Utf8).alias("payload_json"),
        )
        .select(list(TIMING_SCHEMA))
    )


def fred_timing_frame(
    series: pl.DataFrame,
    snapshot_id: str,
    ingestion_time: datetime | None = None,
    series_policies: dict[str, str] | None = None,
    default_policy: str = "revised",
) -> pl.DataFrame:
    """Timing frame for FRED observations.

    ALFRED vintage rows carry a vintage_date; latest-vintage rows carry the
    series revision policy so the engine can refuse revised series that
    cannot be made point-in-time.

    publication_time:
      - vintage row   -> vintage_date at 00:00:00 (DATE precision); the
                         version released that day is next-day available
      - non_revised   -> observation_date at 00:00:00 (DATE precision); the
                         value is the as-published value, available the day
                         after the observation
      - revised       -> null (the true as-published value is unknown; the
                         NOT_POINT_IN_TIME rule refuses the row)
    """
    policies = series_policies or {}
    ingest = _day_start(ingestion_time)

    has_vintage = "vintage_date" in series.columns
    vintage_date = (
        pl.col("vintage_date").cast(pl.Date) if has_vintage
        else pl.lit(None, dtype=pl.Date)
    )
    vintage_id = (
        pl.col("vintage_date").cast(pl.Utf8) if has_vintage
        else pl.lit(None, dtype=pl.Utf8)
    )
    vintage_dt = (
        pl.col("vintage_date").map_elements(_day_start, return_dtype=pl.Datetime("us"))
        if has_vintage
        else pl.lit(None, dtype=pl.Datetime("us"))
    )
    policy = (
        pl.when(vintage_date.is_not_null())
        .then(pl.lit("revised"))  # an explicit vintage is always point-in-time
        .otherwise(
            pl.col("series_id").replace_strict(
                policies, default=default_policy, return_dtype=pl.Utf8
            )
        )
    )

    obs_dt = _datetime_expr(series["observation_date"])
    pub = (
        pl.when(vintage_date.is_not_null())
        .then(vintage_dt)
        .otherwise(
            pl.when(policy == "revised")
            .then(pl.lit(None, dtype=pl.Datetime("us")))
            .otherwise(obs_dt)
        )
        .alias("pub")
    )
    return (
        series.with_columns(
            (pl.lit("obs|") + pl.lit(snapshot_id) + pl.lit("|")
             + pl.col("series_id") + pl.lit("|")
             + pl.col("observation_date").cast(pl.Utf8) + pl.lit("|")
             + vintage_id.fill_null("latest")).alias("record_id"),
            pl.col("series_id").alias("source_key"),
            pl.lit("macro").alias("domain"),
            pl.lit("observation").alias("kind"),
            obs_dt.alias("event_time"),
            pub.alias("publication_time"),
            pl.lit(TimePrecision.DATE.value).alias("publication_precision"),
            pub.alias("effective_time"),
            pl.lit(ingest, dtype=pl.Datetime("us")).alias("ingestion_time"),
            vintage_id.alias("vintage_id"),
            vintage_date.alias("vintage_date"),
            policy.alias("series_policy"),
            pl.struct(list(series.columns)).map_elements(_payload, return_dtype=pl.Utf8).alias("payload_json"),
        )
        .select(list(TIMING_SCHEMA))
    )
