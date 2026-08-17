"""Numerical validation of OHLCV tables.

Catches impossible values (negative prices, High < Low, Open/Close outside
the day's range, negative volume) as errors, and suspicious-but-possible
values as warnings. Nothing is deleted - findings go to the manifest.
"""

from __future__ import annotations

import polars as pl

from orbit.ingestion.validators.report import ValidationReport


def validate_numerical(df: pl.DataFrame, report: ValidationReport, symbol: str = "") -> None:
    for col in ("open", "high", "low", "close"):
        if col not in df.columns:
            continue
        neg = df.filter(pl.col(col).is_not_null() & (pl.col(col) <= 0))
        if neg.height:
            report.add("error", "non_positive_price", f"{col}: {neg.height} rows <= 0", symbol)

    if "volume" in df.columns:
        bad_vol = df.filter(pl.col("volume").is_not_null() & (pl.col("volume") < 0))
        if bad_vol.height:
            report.add("error", "negative_volume", f"volume: {bad_vol.height} rows < 0", symbol)

    if {"high", "low"}.issubset(df.columns):
        bad_range = df.filter(pl.col("high").is_not_null() & pl.col("low").is_not_null() & (pl.col("high") < pl.col("low")))
        if bad_range.height:
            report.add("error", "high_below_low", f"{bad_range.height} rows with high < low", symbol)

    for col in ("open", "close"):
        if col in df.columns and "high" in df.columns and "low" in df.columns:
            bad = df.filter(
                pl.col(col).is_not_null()
                & (pl.col(col) < pl.col("low"))
                & (pl.col(col).is_not_null() & pl.col("low").is_not_null())
            )
            if bad.height:
                report.add("error", "price_outside_range", f"{col}: {bad.height} rows below low", symbol)
            bad2 = df.filter(
                pl.col(col).is_not_null() & pl.col("high").is_not_null()
                & (pl.col(col) > pl.col("high"))
            )
            if bad2.height:
                report.add("error", "price_outside_range", f"{col}: {bad2.height} rows above high", symbol)

    if "close" in df.columns and "volume" in df.columns:
        no_vol = df.filter((pl.col("volume") == 0) & pl.col("close").is_not_null())
        if no_vol.height:
            report.add("warning", "zero_volume_day", f"{no_vol.height} zero-volume rows (possible halt)", symbol)

    if "close" in df.columns and df.height > 1:
        # "Daily" moves must be computed against the previous CALENDAR row, so
        # compute them on a date-ordered copy when a date column is present;
        # provider row order is not guaranteed.
        df_sorted = df
        date_col = next((c for c in ("trade_date", "date", "ts") if c in df.columns), None)
        if date_col is not None:
            df_sorted = df.sort(date_col)
        closes = df_sorted["close"]
        prev = closes.shift(1)
        move = (closes / prev) - 1
        huge = df_sorted.filter(move.fill_null(0).abs() > 1.0)
        if huge.height:
            report.add(
                "warning",
                "huge_daily_move",
                f"{huge.height} rows with |move| > 100% (possible data error or extreme event)",
                symbol,
            )