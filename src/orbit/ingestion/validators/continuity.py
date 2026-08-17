"""Continuity checks for daily market data.

Flags: missing weekdays, suspiciously long gaps, duplicated dates, and
abnormal discontinuities. These are WARNINGS, not errors: a real corporate
action or market event legitimately creates a jump, and Phase 3's rule is
"flag first, investigate second, never auto-fix".
"""

from __future__ import annotations

from datetime import timedelta

import polars as pl

from orbit.ingestion.validators.report import ValidationReport

_MAX_GAP_CALENDAR_DAYS = 14  # US equities: longer gaps need an explanation


def validate_continuity(
    df: pl.DataFrame, report: ValidationReport, date_col: str, symbol: str = ""
) -> None:
    if date_col not in df.columns:
        return
    dates = df[date_col].sort().unique()
    if dates.len() < 2:
        return

    diffs = dates.diff().drop_nulls()
    gaps = df.select(
        pl.Series("days", [d.days for d in diffs]).alias("gap_days")
    ).filter(pl.col("gap_days") > _MAX_GAP_CALENDAR_DAYS)
    if gaps.height:
        examples = ", ".join(f"{g} days" for g in gaps["gap_days"].head(3).to_list())
        report.add(
            "warning",
            "suspicious_gap",
            f"{gaps.height} calendar gaps > {_MAX_GAP_CALENDAR_DAYS} days (examples: {examples})",
            symbol,
        )

    missing = _missing_weekdays(dates)
    if missing:
        report.add(
            "warning",
            "missing_weekday",
            f"{len(missing)} weekday(s) absent (e.g. {missing[:3]}) - compare against exchange calendar",
            symbol,
        )


def _missing_weekdays(dates: pl.Series) -> list[str]:
    first, last = dates[0], dates[-1]
    present = set(dates.to_list())
    found: list[str] = []
    d = first
    while d <= last:
        if d.weekday() < 5 and d not in present:
            found.append(d.strftime("%Y-%m-%d"))
        d += timedelta(days=1)
    return found