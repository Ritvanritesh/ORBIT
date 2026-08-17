"""Validation: structural, duplicate, numerical, continuity checks.

Validation never silently deletes or repairs data. Every check emits an
Issue (error or warning); errors block promotion to the canonical layer,
warnings are recorded in the manifest for later investigation. Financial
data contains unusual but real events, so anomalies are flagged, not fixed.
"""

from __future__ import annotations

import polars as pl

from orbit.ingestion.validators.continuity import validate_continuity
from orbit.ingestion.validators.duplicates import validate_duplicates
from orbit.ingestion.validators.numerical import validate_numerical
from orbit.ingestion.validators.report import Issue, ValidationReport
from orbit.ingestion.validators.structural import validate_structure

__all__ = [
    "Issue",
    "ValidationReport",
    "validate_continuity",
    "validate_duplicates",
    "validate_numerical",
    "validate_structure",
]


def validate_market_bars(
    df: pl.DataFrame, symbol: str, *, date_col: str, provider: str
) -> ValidationReport:
    """Run the full validation stack on a parsed market table."""
    report = ValidationReport(source=f"{provider}/{symbol}")
    validate_structure(
        df, report, expected_columns=["open", "high", "low", "close", "volume"],
        numeric_columns=["open", "high", "low", "close", "volume"],
    )
    validate_duplicates(df, report, keys=[date_col], context=symbol)
    validate_numerical(df, report, symbol=symbol)
    validate_continuity(df, report, date_col=date_col, symbol=symbol)
    return report


def validate_fred_series(df: pl.DataFrame, series_id: str) -> ValidationReport:
    report = ValidationReport(source=f"fred/{series_id}")
    validate_structure(
        df, report, expected_columns=["observation_date", "value", "series_id"],
        numeric_columns=["value"],
    )
    validate_duplicates(df, report, keys=["observation_date"], context=series_id)
    nulls = df["value"].null_count()
    if nulls:
        report.add("warning", "missing_observations", f"{nulls} null observations kept as-is")
    if df.height == 0:
        report.add("error", "empty_series", f"fred series {series_id} has no observations")
    return report


def validate_sec_facts(df: pl.DataFrame, cik: int) -> ValidationReport:
    report = ValidationReport(source=f"sec/companyfacts/{cik}")
    validate_structure(
        df, report, expected_columns=["taxonomy", "fact", "val", "accn", "filed"],
        numeric_columns=[],
    )
    if df["val"].dtype == pl.Utf8:
        n_text = df.filter(
            pl.col("val").cast(pl.Float64, strict=False).is_null() & pl.col("val").is_not_null()
        ).height
        if n_text:
            report.add(
                "warning",
                "text_fact_values",
                f"{n_text} fact values are non-numeric text (kept verbatim in raw JSON; null in normalized view)",
            )
    validate_duplicates(
        df, report,
        keys=["taxonomy", "fact", "accn", "start", "end", "unit"],
        context=f"cik{cik}",
    )
    if df.height == 0:
        report.add("error", "empty_facts", f"no facts for CIK {cik}")
    return report