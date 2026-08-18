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

from orbit.temporal.times import EXCHANGE_TZ, TimePrecision, session_close_utc

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


def _ex_date_ny(ts: datetime | None) -> date | None:
    """The exchange-local session date of a corporate-action event instant
    (same rule as the Phase 3 normalizer's trade_date)."""
    if ts is None:
        return None
    from datetime import timezone

    return ts.replace(tzinfo=timezone.utc).astimezone(EXCHANGE_TZ).date()


# OHLC adjustment labels under which stored prices are retroactively
# split-adjusted and MUST be multiplied by the split factor to recover the
# as-published price (mirrors orbit.ingestion.reconciliation._ADJUSTED_LABELS).
_ADJUSTED_LABELS = frozenset(
    {"split_adjusted", "dividend_and_split_adjusted", "fully_adjusted"}
)

# Whether a provider's VOLUME is on the same split-adjusted share basis as
# its OHLC. yahoo_chart_api volume is adjusted (verified continuous across
# splits in the dev sample, see the schema sidecar); stooq_csv volume is raw
# shares. Dividing an already-raw volume by the split factor would corrupt
# the as-published share count.
PROVIDER_VOLUME_BASIS: dict[str, str] = {
    "yahoo_chart_api": "split_adjusted",
    "stooq_csv": "raw",
}


def _volume_is_split_adjusted(
    df: pl.DataFrame, volume_basis: str | None
) -> bool:
    """Whether the stored volume is split-adjusted (so it must be divided by
    the split factor to recover as-published shares)."""
    if volume_basis is not None:
        return volume_basis == "split_adjusted"
    if "provider" in df.columns:
        providers = {
            p for p in df["provider"].unique().to_list() if p is not None
        }
        if len(providers) == 1:
            p = next(iter(providers))
            if p in PROVIDER_VOLUME_BASIS:
                return PROVIDER_VOLUME_BASIS[p] == "split_adjusted"
    # canonical contract (unknown provider): the normalizer delivers
    # split-adjusted volume
    return True


def _datetime_expr(series: pl.Series) -> pl.Series:
    """Convert a Date series to a naive-UTC Datetime series (day start)."""
    return series.map_elements(_day_start, return_dtype=pl.Datetime("us"))


def _split_factors(
    bars: pl.DataFrame, events: pl.DataFrame
) -> pl.DataFrame:
    """Per (instrument_id, trade_date) the cumulative split ratio of every
    split whose ex-date is STRICTLY AFTER that bar's session.

    The stored bars are retroactively split-adjusted by the provider
    (adjustment='split_adjusted'): adjusted(D) = raw(D) / product of the
    ratios of all splits after D. Reversing it gives the AS-PUBLISHED
    price at D: raw(D) = adjusted(D) * product. The split on the ex-date
    itself already affected that session's prices and is not undone.
    """
    required = {"instrument_id", "kind", "ts", "ratio"}
    if not required.issubset(events.columns):
        return pl.DataFrame(
            schema={"instrument_id": pl.Utf8, "trade_date": pl.Date, "factor": pl.Float64}
        )
    splits = (
        events.filter(pl.col("kind") == "splits")
        .with_columns(
            pl.col("ts").map_elements(_ex_date_ny, return_dtype=pl.Date).alias("ex_date")
        )
        .select(["instrument_id", "ex_date", "ratio"])
    )
    if splits.height == 0:
        return pl.DataFrame(
            schema={"instrument_id": pl.Utf8, "trade_date": pl.Date, "factor": pl.Float64}
        )
    return (
        bars.select(["instrument_id", "trade_date"])
        .join(splits, on="instrument_id", how="left")
        .filter(pl.col("ex_date") > pl.col("trade_date"))
        .group_by(["instrument_id", "trade_date"])
        .agg(pl.col("ratio").product().alias("factor"))
    )


def as_published_bars(
    bars: pl.DataFrame,
    events: pl.DataFrame | None = None,
    volume_basis: str | None = None,
) -> pl.DataFrame:
    """Reconstruct AS-PUBLISHED OHLCV from the stored bars + corporate
    actions (the canonical reconstruction the temporal engine and every
    data consumer must use).

    The stored bars are retroactively SPLIT-ADJUSTED by the provider: a
    pre-split close read today is NOT the price that existed at the bar's
    session. With the events artifact the stored OHLC is multiplied by the
    cumulative ratio of every split whose ex-date is strictly after the
    bar's session, and split-adjusted volume is divided by the same factor,
    recovering the values as they were publishable at the time.

    Guards:
      - OHLC is multiplied only when the bars' `adjustment` column (if
        present) is entirely within the adjusted labels; a raw-basis
        provider keeps its verbatim prices.
      - Volume is divided only when it is on the split-adjusted share basis
        (yahoo_chart_api, or an explicit `volume_basis` of
        "split_adjusted"); stooq_csv volume is raw shares and is kept
        verbatim.
      - the joined factor uses a collision-free column name: a bars frame
        that already carries its own `factor` column can never hijack the
        multiplier.

    Returns the bars frame with a `price_basis` column: "as_published"
    when reconstruction ran, "provider_split_adjusted" when the stored
    values are the retroactive provider basis (never silently presented as
    historical truth).
    """
    df = bars
    price_basis = "provider_split_adjusted"
    if events is not None and events.height:
        factors = _split_factors(bars, events)
        if factors.height:
            # a collision-free name: the bars frame could already carry its
            # own 'factor' column, and a polars join would then keep BOTH
            # ('factor' from bars, 'factor_right' from us) - silently
            # multiplying by the wrong column would corrupt every price
            factor_col = "_split_factor"
            factors = factors.rename({"factor": factor_col})
            ohlc_adjusted = True
            if "adjustment" in df.columns:
                labels = {
                    a for a in df["adjustment"].drop_nulls().unique().to_list()
                }
                if labels and not labels.issubset(_ADJUSTED_LABELS):
                    ohlc_adjusted = False
            volume_adjusted = ohlc_adjusted and _volume_is_split_adjusted(
                df, volume_basis
            )
            df = df.join(factors, on=["instrument_id", "trade_date"], how="left").with_columns(
                pl.when(pl.lit(ohlc_adjusted))
                .then(
                    (pl.col("open") * pl.col(factor_col).fill_null(1.0)).alias("open")
                )
                .otherwise(pl.col("open").alias("open"))
                .alias("open"),
                pl.when(pl.lit(ohlc_adjusted))
                .then(
                    (pl.col("high") * pl.col(factor_col).fill_null(1.0)).alias("high")
                )
                .otherwise(pl.col("high").alias("high"))
                .alias("high"),
                pl.when(pl.lit(ohlc_adjusted))
                .then(
                    (pl.col("low") * pl.col(factor_col).fill_null(1.0)).alias("low")
                )
                .otherwise(pl.col("low").alias("low"))
                .alias("low"),
                pl.when(pl.lit(ohlc_adjusted))
                .then(
                    (pl.col("close") * pl.col(factor_col).fill_null(1.0)).alias("close")
                )
                .otherwise(pl.col("close").alias("close"))
                .alias("close"),
                pl.when(pl.lit(volume_adjusted))
                .then(
                    (pl.col("volume") / pl.col(factor_col).fill_null(1.0))
                    .round()
                    .cast(pl.Int64)
                    .alias("volume")
                )
                .otherwise(pl.col("volume").alias("volume"))
                .alias("volume"),
            ).drop(factor_col)
        price_basis = "as_published"
    return df.with_columns(pl.lit(price_basis).alias("price_basis"))


def market_timing_frame(
    bars: pl.DataFrame,
    snapshot_id: str,
    ingestion_time: datetime | None = None,
    events: pl.DataFrame | None = None,
    volume_basis: str | None = None,
) -> pl.DataFrame:
    """Timing frame for canonical daily bars.

    event_time        = trade_date (the session the bar describes)
    publication_time  = session close 16:00 America/New_York on trade_date
                        (DATETIME precision). The provider's ts_utc (session
                        OPEN for intraday-stamped feeds) is deliberately NOT
                        used as publication: that would make the day's close
                        available at the open.
    effective_time    = publication (an EOD bar is usable at the close).

    PRICE BASIS. The stored bars are retroactively SPLIT-ADJUSTED by the
    provider: a pre-split close read today is NOT the price that existed at
    the bar's session. With the sibling events artifact (corporate actions)
    the adapter reconstructs the AS-PUBLISHED OHLCV via
    `as_published_bars()` and marks it `price_basis: "as_published"`.
    Without events the payload carries the provider values verbatim and is
    marked `price_basis: "provider_split_adjusted"` - never silently
    presented as historical truth. `adjclose` (retroactively dividend+split
    adjusted) is ALWAYS excluded from the payload.
    """
    df = as_published_bars(bars, events, volume_basis)
    price_basis = df["price_basis"][0]
    payload_cols = [
        c for c in df.columns
        if c not in ("adjclose", "adjustment", "price_basis")
    ]
    ingest = _day_start(ingestion_time)
    event = _datetime_expr(df["trade_date"])
    close = df["trade_date"].map_elements(
        session_close_utc, return_dtype=pl.Datetime("us")
    )
    return (
        df.with_columns(
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
            pl.struct(payload_cols + ["price_basis"]).map_elements(_payload, return_dtype=pl.Utf8).alias("payload_json"),
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
    calendar: Any = None,
) -> pl.DataFrame:
    """Timing frame for FRED observations.

    ALFRED vintage rows carry a vintage_date; latest-vintage rows carry the
    series revision policy so the engine can refuse revised series that
    cannot be made point-in-time.

    publication_time:
      - vintage row   -> the scheduled release instant when the release
                         calendar has an ENABLED entry whose weekday matches
                         the vintage release date (DATETIME precision); the
                         vintage_date at 00:00:00 otherwise (DATE precision,
                         next-day availability)
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
    cal_dt = (
        pl.struct(["series_id", "vintage_date"]).map_elements(
            lambda row: _scheduled_release_instant(calendar, row),
            return_dtype=pl.Datetime("us"),
        )
        if calendar is not None and has_vintage
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
        .then(pl.when(cal_dt.is_not_null()).then(cal_dt).otherwise(vintage_dt))
        .otherwise(
            pl.when(policy == "revised")
            .then(pl.lit(None, dtype=pl.Datetime("us")))
            .otherwise(obs_dt)
        )
        .alias("pub")
    )
    precision = (
        pl.when(cal_dt.is_not_null())
        .then(pl.lit(TimePrecision.DATETIME.value))
        .otherwise(pl.lit(TimePrecision.DATE.value))
        .alias("precision")
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
            precision.alias("publication_precision"),
            pub.alias("effective_time"),
            pl.lit(ingest, dtype=pl.Datetime("us")).alias("ingestion_time"),
            vintage_id.alias("vintage_id"),
            vintage_date.alias("vintage_date"),
            policy.alias("series_policy"),
            pl.struct(list(series.columns)).map_elements(_payload, return_dtype=pl.Utf8).alias("payload_json"),
        )
        .select(list(TIMING_SCHEMA))
    )


def _scheduled_release_instant(calendar: Any, row: dict[str, Any]) -> datetime | None:
    """The scheduled release instant for a vintage row, or None (fall back
    to date precision) when the calendar has no enabled matching entry."""
    raw: Any = row.get("vintage_date")
    if raw is None:
        return None
    vintage_date = date.fromisoformat(raw) if isinstance(raw, str) else raw
    entry = calendar.entry_for(row["series_id"])
    if entry is None:
        return None
    return entry.scheduled_instant(vintage_date)
