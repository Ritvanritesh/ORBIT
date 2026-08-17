"""Market data normalization.

Canonical daily-bar schema (schema version v1.0.0):

    instrument_id  str       stable ORBIT identity (INS-xxxxxx)
    symbol         str       provider symbol as requested
    trade_date     date      exchange-local session date (America/New_York for US)
    ts_utc         datetime  bar timestamp in UTC, naive (see TIMEZONE_RULE)
    open/high/low/close  float64  provider values, verbatim
    volume         int64
    adjclose       float64 | null  provider-adjusted close when supplied
    adjustment     str       what the provider's OHLC actually represents
    provider       str
    source_uri     str
    snapshot_id    str

TIMEZONE_RULE
    Yahoo: the provider's epoch timestamps ARE UTC; trade_date is that
    instant converted to the exchange-local session date (documented so
    Phase 4's as-of engine can rely on it).
    Stooq: the provider ships a local calendar date only; trade_date is
    that date and ts_utc is the session close (16:00 America/New_York ->
    UTC).

Never destroy provider representation: the raw layer keeps the exact
bytes; this layer is a separate, reproducible derivation.
"""

from __future__ import annotations

from datetime import date, datetime, time, timezone
from typing import Any
from zoneinfo import ZoneInfo

import polars as pl

from orbit.ingestion.paths import write_json

MARKET_SCHEMA_VERSION = "v1.0.0"
TIMEZONE_RULE = (
    "ts_utc is the provider bar timestamp in UTC (naive); trade_date is the "
    "exchange-local session date (America/New_York for US equities). For "
    "providers without intraday timestamps, ts_utc = session close "
    "(16:00 America/New_York) converted to UTC. v1.0.0 assumes US exchanges "
    "(America/New_York) for ALL instruments - non-US instruments need a new "
    "schema version with per-instrument exchange timezones."
)
EXCHANGE_TZ = ZoneInfo("America/New_York")
CLOSE_LOCAL = time(16, 0)

BAR_COLUMNS: dict[str, pl.DataType] = {
    "instrument_id": pl.Utf8,
    "symbol": pl.Utf8,
    "trade_date": pl.Date,
    "ts_utc": pl.Datetime("us"),
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Int64,
    "adjclose": pl.Float64,
    "adjustment": pl.Utf8,
    "provider": pl.Utf8,
    "source_uri": pl.Utf8,
    "snapshot_id": pl.Utf8,
}

EVENT_COLUMNS: dict[str, pl.DataType] = {
    "instrument_id": pl.Utf8,
    "symbol": pl.Utf8,
    "kind": pl.Utf8,
    "ts": pl.Datetime("us"),
    "ratio": pl.Float64,
    "numerator": pl.Float64,
    "denominator": pl.Float64,
    "provider": pl.Utf8,
    "snapshot_id": pl.Utf8,
}


def normalize_market_bars(
    parsed: dict[str, dict[str, pl.DataFrame]],
    symbol_map: dict[str, str],
    provider: str,
    source_uri: str,
    snapshot_id: str,
) -> dict[str, pl.DataFrame]:
    """Build canonical bars + corporate-action events frames.

    `parsed` maps symbol -> {"bars": DataFrame, "events": DataFrame}.
    `symbol_map` maps provider symbol -> instrument_id (INS-xxxxxx).
    """
    bars_frames: list[pl.DataFrame] = []
    event_frames: list[pl.DataFrame] = []

    for symbol, tables in parsed.items():
        inst_id = symbol_map[symbol]
        bars = tables["bars"].with_columns(
            pl.col("open").cast(pl.Float64),
            pl.col("high").cast(pl.Float64),
            pl.col("low").cast(pl.Float64),
            pl.col("close").cast(pl.Float64),
            pl.col("volume").cast(pl.Int64),
        )
        if "ts" in bars.columns and "adjclose" in bars.columns:
            canonical = _from_yahoo(bars, symbol, inst_id, provider, source_uri, snapshot_id)
        else:
            canonical = _from_stooq(bars, symbol, inst_id, provider, source_uri, snapshot_id)
        bars_frames.append(canonical)

        events = tables.get("events")
        if events is not None and events.height:
            event_frames.append(_events_frame(events, symbol, inst_id, provider, snapshot_id))

    bars_all = pl.concat(bars_frames).select(list(BAR_COLUMNS)) if bars_frames else pl.DataFrame(schema=BAR_COLUMNS)
    events_all = pl.concat(event_frames).select(list(EVENT_COLUMNS)) if event_frames else pl.DataFrame(schema=EVENT_COLUMNS)
    return {"bars": bars_all, "events": events_all}


def _from_yahoo(
    bars: pl.DataFrame, symbol: str, inst_id: str, provider: str,
    source_uri: str, snapshot_id: str,
) -> pl.DataFrame:
    return (
        bars.with_columns(
            pl.lit(inst_id).alias("instrument_id"),
            pl.col("ts").cast(pl.Datetime("us")).alias("ts_utc"),
        )
        .with_columns(
            pl.col("ts_utc").map_elements(_session_date, return_dtype=pl.Date).alias("trade_date")
        )
        .with_columns(
            pl.lit("split_adjusted").alias("adjustment")
        )
        .select(
            "instrument_id", "symbol", "trade_date", "ts_utc", "open", "high",
            "low", "close", "volume", "adjclose", "adjustment",
            pl.lit(provider).alias("provider"),
            pl.lit(source_uri).alias("source_uri"),
            pl.lit(snapshot_id).alias("snapshot_id"),
        )
    )


def _from_stooq(
    bars: pl.DataFrame, symbol: str, inst_id: str, provider: str,
    source_uri: str, snapshot_id: str,
) -> pl.DataFrame:
    return (
        bars.rename({"date": "trade_date"})
        .with_columns(
            pl.lit(inst_id).alias("instrument_id"),
            pl.lit(None, dtype=pl.Float64).alias("adjclose"),
            pl.lit("split_adjusted").alias("adjustment"),
            pl.lit(provider).alias("provider"),
            pl.lit(source_uri).alias("source_uri"),
            pl.lit(snapshot_id).alias("snapshot_id"),
        )
        .with_columns(
            pl.col("trade_date")
            .map_elements(_local_close_to_utc, return_dtype=pl.Datetime("us"))
            .alias("ts_utc")
        )
        .select(
            "instrument_id", "symbol", "trade_date", "ts_utc", "open", "high",
            "low", "close", "volume", "adjclose", "adjustment",
            "provider", "source_uri", "snapshot_id",
        )
    )


def _events_frame(
    events: pl.DataFrame, symbol: str, inst_id: str, provider: str, snapshot_id: str,
) -> pl.DataFrame:
    for col in ("numerator", "denominator", "amount"):
        if col not in events.columns:
            events = events.with_columns(pl.lit(None, dtype=pl.Float64).alias(col))
    ratio = (
        pl.when(events["kind"] == "splits")
        .then(pl.col("numerator") / pl.col("denominator"))
        .otherwise(pl.col("amount"))
        .alias("ratio")
    )
    return (
        events.with_columns(
            pl.lit(inst_id).alias("instrument_id"),
            ratio,
            pl.lit(provider).alias("provider"),
            pl.lit(snapshot_id).alias("snapshot_id"),
        )
        .select(
            "instrument_id", "symbol", "kind", "ts", "ratio",
            "numerator", "denominator", "provider", "snapshot_id",
        )
    )


def write_schema_sidecar(dir_path: Any, kind: str, columns: dict[str, pl.DataType]) -> None:
    write_json(
        dir_path / "_schema.json",
        {
            "kind": kind,
            "schema_version": MARKET_SCHEMA_VERSION,
            "timezone_rule": TIMEZONE_RULE,
            "columns": list(columns),
            "volume_basis": (
                "yahoo_chart_api volume is on the SAME split-adjusted share basis as "
                "OHLC (verified continuous across splits in the dev sample), so "
                "close*volume is true dollar volume; stooq_csv volume is raw shares "
                "and dollar volume must be reconstructed from events"
            ),
            "adjclose_note": (
                "adjclose is retroactively dividend+split adjusted by the provider; "
                "it is a look-ahead if used as a point-in-time price"
            ),
        },
    )


def build_corporate_actions(
    events: pl.DataFrame, snapshot_id: str, source: str,
) -> list[dict[str, Any]]:
    """Materialize normalized events into Phase 2 CorporateAction records.

    Consumes the canonical events frame (instrument_id, kind, ts, ratio).
    The ex-date is the exchange-local session date of the event timestamp
    (same rule as trade_date). action_id is a deterministic digest of the
    row, so re-derivation always yields identical records and the id is
    stable across snapshots.
    """
    import hashlib

    from orbit.schemas.instrument import CorporateAction

    actions: list[CorporateAction] = []
    for row in events.iter_rows(named=True):
        if row["kind"] == "splits":
            ratio = float(row["ratio"] or 0.0)
            action_type = "split" if ratio >= 1 else "reverse_split"
            value = ratio
        elif row["kind"] == "dividends":
            action_type = "dividend"
            value = float(row["ratio"] or 0.0)
        else:
            continue
        if value <= 0:
            continue
        ex_date = _session_date(row["ts"]) if row["ts"] is not None else None
        digest = hashlib.sha256(
            f"{snapshot_id}|{row['instrument_id']}|{row['kind']}|{row['ts']}|{value}".encode()
        ).hexdigest()[:8]
        actions.append(
            CorporateAction(
                action_id=f"CA-{int(digest, 16) % 1_000_000:06d}",
                instrument_id=row["instrument_id"],
                action_type=action_type,
                effective_date=ex_date,
                ex_date=ex_date,
                ratio=value,
                source=f"{source}/{snapshot_id}",
            )
        )
    return [a.model_dump(mode="json") for a in actions]


def _session_date(ts_utc: datetime) -> date:
    return ts_utc.replace(tzinfo=timezone.utc).astimezone(EXCHANGE_TZ).date()


def _local_close_to_utc(local_date: date) -> datetime:
    local_dt = datetime.combine(local_date, CLOSE_LOCAL, tzinfo=EXCHANGE_TZ)
    return local_dt.astimezone(timezone.utc).replace(tzinfo=None)