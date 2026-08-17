"""Parsing of raw provider artifacts into typed tables.

Parsing is NOT normalization: it reads whatever the provider delivered into
a neutral table so validation can inspect it. Normalization (which owns the
canonical schema, identifiers, and timezone rules) happens later on the
*validated* parse.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import polars as pl


def _naive_utc(epoch_seconds: float) -> datetime:
    return datetime.fromtimestamp(epoch_seconds, tz=timezone.utc).replace(tzinfo=None)


def parse_yahoo_chart(body: bytes, symbol: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Read a Yahoo v8 chart JSON payload into (bars, events) tables.

    Bars keep the provider's split-adjusted OHLC verbatim; `adjclose` and
    raw dividend/split event rows are preserved for reconciliation checks.
    timestamps are epoch seconds in UTC, converted to naive-UTC datetimes.
    """
    payload = json.loads(body)
    result = payload["chart"]["result"][0]
    ts = result.get("timestamp") or []
    quote = result.get("indicators", {}).get("quote", [{}])[0]
    adjclose = result.get("indicators", {}).get("adjclose", [{}])[0]
    meta = result.get("meta", {})

    bars = pl.DataFrame(
        {
            "symbol": [symbol] * len(ts),
            "ts_epoch": ts,
            "open": quote.get("open"),
            "high": quote.get("high"),
            "low": quote.get("low"),
            "close": quote.get("close"),
            "volume": quote.get("volume"),
            "adjclose": adjclose.get("adjclose"),
            "currency": [meta.get("currency")] * len(ts),
            "exchange_name": [meta.get("exchangeName")] * len(ts),
        }
    ).with_columns(pl.col("ts_epoch").map_elements(_naive_utc, return_dtype=pl.Datetime("us")).alias("ts"))

    events_rows: list[dict[str, Any]] = []
    for kind in ("dividends", "splits"):
        for key, ev in (result.get("events") or {}).get(kind, {}).items():
            events_rows.append(
                {
                    "symbol": symbol,
                    "kind": kind,
                    "ts_epoch": int(key),
                    "amount": ev.get("amount"),
                    "numerator": ev.get("numerator"),
                    "denominator": ev.get("denominator"),
                    "splitRatio": ev.get("splitRatio"),
                    "date": ev.get("date"),
                }
            )
    events = (
        pl.DataFrame(
            events_rows,
            schema={
                "symbol": pl.Utf8, "kind": pl.Utf8, "ts_epoch": pl.Int64,
                "amount": pl.Float64, "numerator": pl.Float64,
                "denominator": pl.Float64, "splitRatio": pl.Utf8,
                "date": pl.Int64,
            },
        ).with_columns(
            pl.col("ts_epoch")
            .map_elements(_naive_utc, return_dtype=pl.Datetime("us"))
            .alias("ts")
        )
        if events_rows
        else pl.DataFrame(
            schema={
                "symbol": pl.Utf8, "kind": pl.Utf8, "ts_epoch": pl.Int64,
                "amount": pl.Float64, "numerator": pl.Float64,
                "denominator": pl.Float64, "splitRatio": pl.Utf8,
                "date": pl.Int64, "ts": pl.Datetime("us"),
            }
        )
    )
    return bars, events


def parse_stooq_csv(body: bytes, symbol: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Read a Stooq daily CSV (Date,Open,High,Low,Close,Volume) into a table.

    Column names are lowercased into ORBIT's neutral convention
    (date/open/high/low/close/volume); values stay verbatim. Returns
    (bars, empty_events) to match the parser contract of the market
    pipeline (bars, events) - Stooq does not ship split/dividend events.
    """
    text = body.decode("utf-8", errors="replace")
    df = pl.read_csv(io_bytes(text), infer_schema_length=10000)
    rename = {c: c.lower() for c in df.columns}
    df = df.rename(rename)
    date_col = "date" if "date" in df.columns else df.columns[0]
    if df[date_col].dtype == pl.Utf8:
        df = df.with_columns(pl.col(date_col).str.to_date("%Y-%m-%d"))
    bars = df.with_columns(pl.lit(symbol).alias("symbol"))
    events = pl.DataFrame(
        schema={
            "symbol": pl.Utf8, "kind": pl.Utf8, "ts_epoch": pl.Int64,
            "amount": pl.Float64, "numerator": pl.Float64,
            "denominator": pl.Float64, "splitRatio": pl.Utf8,
            "date": pl.Int64, "ts": pl.Datetime("us"),
        }
    )
    return bars, events


def io_bytes(text: str) -> Any:
    import io

    return io.StringIO(text)


def parse_fred_csv(body: bytes, series_id: str) -> pl.DataFrame:
    """Read a fredgraph.csv into (observation_date, series_id, value).

    "." (missing observation) becomes null - never silently filled.
    """
    text = body.decode("utf-8", errors="replace")
    df = pl.read_csv(io_bytes(text), infer_schema_length=10000)
    if df.shape[1] < 2:
        raise ValueError(f"fred CSV for {series_id} has no value column")
    value_col = df.columns[1]
    df = df.rename({df.columns[0]: "observation_date", value_col: "value"})
    if df["value"].dtype == pl.Utf8:
        df = df.with_columns(
            pl.when(pl.col("value") == ".")
            .then(pl.lit(None, dtype=pl.Float64))
            .otherwise(pl.col("value").cast(pl.Float64, strict=False))
            .alias("value")
        )
    date_dtype = df["observation_date"].dtype
    date_expr = (
        pl.col("observation_date").str.to_date("%Y-%m-%d")
        if date_dtype == pl.Utf8
        else pl.col("observation_date").cast(pl.Date)
    )
    return df.with_columns(
        date_expr,
        pl.lit(series_id).alias("series_id"),
    )


def parse_sec_companyfacts(body: bytes) -> pl.DataFrame:
    """Flatten a companyfacts JSON into one row per reported fact.

    The flattening is mechanical - values, units, and filing metadata
    (accn, filed, fy, fp) are preserved verbatim for Phase 4. An explicit
    schema avoids polars type inference on mixed fact values.
    """
    payload = json.loads(body)
    schema = {
        "cik": pl.Int64,
        "entity_name": pl.Utf8,
        "taxonomy": pl.Utf8,
        "fact": pl.Utf8,
        "unit": pl.Utf8,
        "val": pl.Utf8,
        "start": pl.Utf8,
        "end": pl.Utf8,
        "accn": pl.Utf8,
        "fy": pl.Int64,
        "fp": pl.Utf8,
        "form": pl.Utf8,
        "filed": pl.Utf8,
        "frame": pl.Utf8,
    }
    rows: list[dict[str, Any]] = []
    for taxonomy, facts in payload.get("facts", {}).items():
        for fact_name, fact in facts.items():
            for unit, entries in fact.get("units", {}).items():
                for e in entries:
                    rows.append(
                        {
                            "cik": payload.get("cik"),
                            "entity_name": payload.get("entityName"),
                            "taxonomy": taxonomy,
                            "fact": fact_name,
                            "unit": unit,
                            "val": e.get("val"),
                            "start": e.get("start"),
                            "end": e.get("end"),
                            "accn": e.get("accn"),
                            "fy": e.get("fy"),
                            "fp": e.get("fp"),
                            "form": e.get("form"),
                            "filed": e.get("filed"),
                            "frame": e.get("frame"),
                        }
                    )
    return pl.DataFrame(rows, schema=schema)