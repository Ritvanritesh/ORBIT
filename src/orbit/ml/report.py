"""Phase 9 benchmark report: permanent, auditable, machine + human readable.

The report is written to benchmarks/phase9_ml_benchmark.parquet (one row per
experiment/control run, including null and failed runs - failures are never
hidden) and benchmarks/phase9_ml_benchmark.md (human-readable summary).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

BENCHMARKS_DIR = Path(__file__).resolve().parents[3] / "benchmarks"
REPORT_PARQUET = BENCHMARKS_DIR / "phase9_ml_benchmark.parquet"
REPORT_MARKDOWN = BENCHMARKS_DIR / "phase9_ml_benchmark.md"

REPORT_COLUMNS: list[tuple[str, type]] = [
    ("run_kind", str),
    ("experiment_id", str),
    ("family", str),
    ("params", str),
    ("seed", int),
    ("status", str),
    ("oos_ic", float),
    ("rank_ic", float),
    ("ece", float),
    ("brier", float),
    ("mse", float),
    ("hit_rate", float),
    ("after_cost_total_return", float),
    ("after_cost_final_equity", float),
    ("turnover", float),
    ("total_costs", float),
    ("n_fills", int),
    ("n_rejects", int),
    ("n_signals", int),
    ("feature_set_id", str),
    ("label_id", str),
    ("dataset_snapshot_ids", str),
    ("cost_model_id", str),
    ("evaluation_window", str),
    ("created_at", str),
    ("notes", str),
]


def _row_to_dict(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, _ in REPORT_COLUMNS:
        out[name] = row.get(name)
    out["created_at"] = datetime.now().isoformat(timespec="seconds")
    return out


def append_report_rows(rows: list[dict[str, Any]]) -> Path:
    """Append experiment/control rows to the permanent parquet report."""
    BENCHMARKS_DIR.mkdir(parents=True, exist_ok=True)
    new = pl.DataFrame([_row_to_dict(r) for r in rows], schema=dict(REPORT_COLUMNS))
    if REPORT_PARQUET.exists():
        existing = pl.read_parquet(REPORT_PARQUET)
        combined = pl.concat([existing, new], how="vertical_relaxed")
    else:
        combined = new
    combined = combined.unique(subset=["run_kind", "experiment_id"], keep="last")
    combined.sort(["run_kind", "experiment_id"]).write_parquet(REPORT_PARQUET)
    return REPORT_PARQUET


def write_markdown_report() -> Path:
    """Regenerate the human-readable benchmark markdown from the parquet."""
    if not REPORT_PARQUET.exists():
        raise FileNotFoundError(f"no benchmark parquet at {REPORT_PARQUET}")
    frame = pl.read_parquet(REPORT_PARQUET)
    lines = [
        "# Phase 9 - Baseline ML Benchmark (permanent report)",
        "",
        f"Generated from `{REPORT_PARQUET.name}` (deterministic rows; a rerun "
        "appends or refreshes rows by experiment_id).",
        "",
        "## Protocol",
        "",
        "- Dataset: DS-000004 (20-symbol development universe)",
        "- Features: FS-001 v1 (8 point-in-time numerics, strict boundary)",
        "- Label: LAB-004 v1 (5-session forward total return)",
        "- Windows: train 2010-01-04..2018-12-31, val 2019-01-02..2021-12-31, "
        "test 2022-01-03..2026-06-30 (locked)",
        "- Evaluation: OOS IC / rank IC (per-session mean), ECE (10 bins), "
        "Brier, MSE, hit rate",
        "- Backtest: canonical Phase 7, WEIGHT sizing, CM-001 costs "
        "(2 bps spread, 1 bps fees, 2 bps slippage), open/delay=1",
        "",
        "## Results",
        "",
    ]
    for row in frame.sort(["run_kind", "family", "experiment_id"]).iter_rows(named=True):
        lines.append(f"### {row['run_kind']} - {row['family']} ({row['experiment_id']})")
        lines.append("")
        lines.append(f"- status: `{row['status']}`")
        lines.append(f"- params: `{row['params']}`")
        if row["oos_ic"] is not None:
            lines.append(f"- OOS IC: {row['oos_ic']:.4f} | rank IC: {row['rank_ic']:.4f}")
        if row["ece"] is not None:
            lines.append(f"- ECE: {row['ece']:.4f} | Brier: {row['brier']:.4f}")
        if row["after_cost_total_return"] is not None:
            lines.append(
                f"- after-cost total return: {row['after_cost_total_return']:.4%} "
                f"| turnover: {row['turnover']:.4f} | costs: {row['total_costs']:.2f}"
            )
        if row["notes"]:
            lines.append(f"- notes: {row['notes']}")
        lines.append("")
    REPORT_MARKDOWN.write_text("\n".join(lines), encoding="utf-8")
    return REPORT_MARKDOWN


__all__ = [
    "REPORT_PARQUET",
    "REPORT_MARKDOWN",
    "append_report_rows",
    "write_markdown_report",
]