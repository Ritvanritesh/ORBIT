"""Structural validation: expected columns, types, and non-empty input.

Schema drift (a provider silently changing its column layout) is a Phase 3
failure condition, so structural checks are strict and block promotion.
Only the columns a domain expects to be numeric are type-checked; string
columns (symbols, dates) are structural presence checks only.
"""

from __future__ import annotations

import polars as pl

from orbit.ingestion.validators.report import ValidationReport


def validate_structure(
    df: pl.DataFrame,
    report: ValidationReport,
    expected_columns: list[str],
    numeric_columns: list[str] | None = None,
) -> None:
    missing = [c for c in expected_columns if c not in df.columns]
    if missing:
        report.add("error", "missing_columns", f"missing expected columns: {missing}", f"have={df.columns}")
        return
    if df.height == 0:
        report.add("error", "empty_table", "parsed table is empty")
        return
    for col in numeric_columns or []:
        if col not in df.columns:
            continue
        if df[col].dtype == pl.Utf8:
            report.add("error", "wrong_type", f"column {col} is Utf8 where numeric expected")
        if df[col].null_count() > 0:
            report.add("warning", "null_values", f"column {col} has {df[col].null_count()} nulls")