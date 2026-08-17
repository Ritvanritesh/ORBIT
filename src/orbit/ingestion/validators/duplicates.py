"""Duplicate validation.

Duplicate instrument/date records are the signature of an idempotency bug,
so duplicates are errors: the same observation appearing twice in one
snapshot means ingestion ran twice or the provider repeated rows.
"""

from __future__ import annotations

import polars as pl

from orbit.ingestion.validators.report import ValidationReport


def validate_duplicates(
    df: pl.DataFrame, report: ValidationReport, keys: list[str], context: str = ""
) -> None:
    for key in keys:
        if key not in df.columns:
            return
    counts = df.group_by(keys).len().filter(pl.col("len") > 1)
    if counts.height:
        examples = ", ".join(
            str(r) for r in counts.head(3).rows()
        )
        report.add(
            "error",
            "duplicate_records",
            f"{counts.height} duplicated key combinations (examples: {examples})",
            context=context,
        )