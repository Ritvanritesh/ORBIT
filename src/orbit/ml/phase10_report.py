"""Phase 10 permanent reports: benchmark table + feature research report.

Machine-readable artifacts live under benchmarks/:
  phase10_plan.json               the locked, pre-registered ablation plan
  phase10_diagnostics.json        feature quality + redundancy diagnostics
  phase10_feature_research.parquet  one row per experiment (incl. null/failed)
  phase10_feature_research.md     human-readable benchmark summary
  phase10_runs/EXP-1xxxx/         per-experiment artifacts

The permanent research report docs/phase10_feature_research.md is generated
from the same data and never hides null, rejected, or failed experiments.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

from orbit.ml.features import (
    ALL_PHASE10_DEFINITIONS,
    FEATURE_DEFINITIONS,
    FEATURE_FAMILY_BY_ID_PHASE10,
    PHASE10_FEATURE_SET_ORDER,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
BENCHMARKS_DIR = _REPO_ROOT / "benchmarks"
ARTIFACTS_ROOT = BENCHMARKS_DIR / "phase10_runs"
REPORT_PARQUET = BENCHMARKS_DIR / "phase10_feature_research.parquet"
REPORT_MARKDOWN = BENCHMARKS_DIR / "phase10_feature_research.md"
PLAN_JSON = BENCHMARKS_DIR / "phase10_plan.json"
DIAGNOSTICS_JSON = BENCHMARKS_DIR / "phase10_diagnostics.json"
RESEARCH_MD = _REPO_ROOT / "docs" / "phase10_feature_research.md"

REPORT_COLUMNS: list[tuple[str, type]] = [
    ("experiment_id", str),
    ("feature_set_id", str),
    ("feature_set_version", str),
    ("set_role", str),
    ("set_family", str),
    ("n_features", int),
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
    ("label_id", str),
    ("dataset_snapshot_ids", str),
    ("cost_model_id", str),
    ("evaluation_window", str),
    ("train_rows", int),
    ("val_rows", int),
    ("test_rows", int),
    ("feature_set_digest", str),
    ("definitions_digest", str),
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
    """Append experiment rows to the permanent parquet report (upsert by id)."""
    BENCHMARKS_DIR.mkdir(parents=True, exist_ok=True)
    new = pl.DataFrame([_row_to_dict(r) for r in rows], schema=dict(REPORT_COLUMNS))
    if REPORT_PARQUET.exists():
        existing = pl.read_parquet(REPORT_PARQUET)
        combined = pl.concat([existing, new], how="vertical_relaxed")
    else:
        combined = new
    combined = combined.unique(subset=["experiment_id"], keep="last")
    combined.sort(["experiment_id"]).write_parquet(REPORT_PARQUET)
    return REPORT_PARQUET


def write_plan(plan: dict[str, Any]) -> Path:
    BENCHMARKS_DIR.mkdir(parents=True, exist_ok=True)
    PLAN_JSON.write_text(
        json.dumps(plan, sort_keys=True, indent=2, default=str), encoding="utf-8"
    )
    return PLAN_JSON


def write_diagnostics(diagnostics: dict[str, Any]) -> Path:
    BENCHMARKS_DIR.mkdir(parents=True, exist_ok=True)
    DIAGNOSTICS_JSON.write_text(
        json.dumps(diagnostics, sort_keys=True, indent=2, default=str),
        encoding="utf-8",
    )
    return DIAGNOSTICS_JSON


def write_markdown_report() -> Path:
    """Regenerate the human-readable benchmark markdown from the parquet."""
    if not REPORT_PARQUET.exists():
        raise FileNotFoundError(f"no phase10 parquet at {REPORT_PARQUET}")
    frame = pl.read_parquet(REPORT_PARQUET)
    lines = [
        "# Phase 10 - Feature Engineering + Ablation (benchmark table)",
        "",
        f"Generated from `{REPORT_PARQUET.name}` (deterministic rows).",
        "",
        "## Protocol",
        "",
        "- Dataset: DS-000004 (20-symbol development universe)",
        "- Label: LAB-004 v1 (5-session forward total return)",
        "- Windows: train 2010-01-04..2018-12-31, val 2019-01-02..2021-12-31, "
        "test 2022-01-03..2026-06-30 (locked)",
        "- Cost model: CM-001 (spread 2 bps, fees 1 bps, slippage 2 bps)",
        "- Signal construction: top-3 long, equal weight 1/3 (Phase 9 path)",
        "",
        "## Results",
        "",
    ]
    for row in frame.sort(["experiment_id"]).iter_rows(named=True):
        lines.append(
            f"### {row['experiment_id']} - {row['feature_set_id']} "
            f"({row['set_role']} {row['set_family'] or ''}) - {row['family']}"
        )
        lines.append("")
        lines.append(f"- status: `{row['status']}` | params: `{row['params']}`")
        lines.append(f"- features: {row['n_features']} (digest {str(row['feature_set_digest'])[:16]}...)")
        if row["oos_ic"] is not None:
            lines.append(
                f"- OOS IC: {row['oos_ic']:.4f} | rank IC: {row['rank_ic']:.4f}"
            )
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


def _feature_inventory_lines() -> list[str]:
    lines = [
        "## Feature inventory",
        "",
        "Every feature: ID, name, family, definition, lookback, temporal "
        "boundary, missingness policy, snapshot version. All Phase 10 features "
        "are point-in-time-valid at the strict boundary "
        "(window_end_session = D-1 < decision session D).",
        "",
        "| ID | Name | Family | Definition | Lookback | Raw inputs | Missing policy |",
        "|----|------|--------|------------|----------|------------|----------------|",
    ]
    for f in FEATURE_DEFINITIONS + ALL_PHASE10_DEFINITIONS:
        family = (
            "phase9_base"
            if f["feature_id"] in {x["feature_id"] for x in FEATURE_DEFINITIONS}
            else FEATURE_FAMILY_BY_ID_PHASE10.get(f["feature_id"], "unknown")
        )
        lines.append(
            f"| {f['feature_id']} | {f['name']} | {family} | "
            f"{f.get('formula', f.get('kind'))} | {f.get('window', f.get('lookback', ''))} | "
            f"{','.join(f.get('raw_inputs', []))} | {f.get('missing_policy', '')} |"
        )
    return lines


def write_research_report(
    *,
    plan: dict[str, Any],
    diagnostics: dict[str, Any],
    snapshots: dict[str, Any],
    phase9_fs001_digest: str,
) -> Path:
    """Generate the permanent docs/phase10_feature_research.md report."""
    from orbit.ml.features import (
        FEATURE_NAMES_PHASE10,
        PHASE10_FAMILIES,
        PHASE10_FEATURE_SETS,
        PHASE10_FEATURE_SET_ORDER,
    )

    frame = pl.read_parquet(REPORT_PARQUET) if REPORT_PARQUET.exists() else pl.DataFrame(
        schema=dict(REPORT_COLUMNS)
    )
    lines = [
        "# ORBIT Phase 10: Feature Engineering + Ablation",
        "",
        "Version 1.0 - permanent research report",
        "",
        "## 1. Purpose",
        "",
        "Phase 10 investigates whether the Phase 9 null result was caused by an "
        "insufficient feature representation. The central question: do additional, "
        "scientifically justified, point-in-time-valid feature families contain "
        "incremental information beyond FS-001 v1? The design isolates FEATURE "
        "REPRESENTATION, not model complexity: the Phase 9 model families are "
        "reused unchanged with pre-registered grid points, the dataset (DS-000004), "
        "label (LAB-004 v1), split, cost model and Phase 7 backtester are identical "
        "to Phase 9, and every experiment is registered before execution.",
        "",
        "## 2. Feature families",
        "",
        "| Family | Feature IDs | Kind |",
        "|--------|-------------|------|",
    ]
    for fam in PHASE10_FAMILIES:
        ids = [f["feature_id"] for f in __import__(
            "orbit.ml.features", fromlist=["PHASE10_FAMILY_DEFINITIONS"]
        ).PHASE10_FAMILY_DEFINITIONS[fam]]
        kinds = ", ".join(
            sorted(
                {
                    f["kind"]
                    for f in __import__(
                        "orbit.ml.features", fromlist=["PHASE10_FAMILY_DEFINITIONS"]
                    ).PHASE10_FAMILY_DEFINITIONS[fam]
                }
            )
        )
        lines.append(f"| {fam} | {', '.join(ids)} | {kinds} |")

    lines += [
        "",
        "Only families computable from the existing DS-000004 OHLCV bars are "
        "implemented. No fundamentals, macro, news, text, options, or alternative "
        "data were invented (they belong to later data-expansion phases).",
        "",
    ]
    lines += _feature_inventory_lines()
    lines += [
        "",
        "## 3. Feature snapshots",
        "",
        "FS-001 v1 is frozen (digest "
        f"{phase9_fs001_digest[:16]}...). New immutable snapshots:",
        "",
        "| Snapshot | Role | Members | Digest |",
        "|----------|------|---------|--------|",
    ]
    for sid in PHASE10_FEATURE_SET_ORDER:
        snap = snapshots.get(sid)
        if snap is None:
            continue
        lines.append(
            f"| {sid} {snap.feature_set_version} | "
            f"{'base (frozen)' if sid == 'FS-001' else PHASE10_FEATURE_SETS[sid]['role']} | "
            f"{len(snap.feature_refs)} | {snap.content_digest[:16]}... |"
        )
    lines += [
        "",
        "## 4. Diagnostics",
        "",
    ]
    diag_lines = json.dumps(diagnostics, sort_keys=True, default=str, indent=2)
    lines.append("```json")
    lines.extend(diag_lines.splitlines())
    lines.append("```")
    lines += [
        "",
        "## 5. Ablation results",
        "",
        "One row per experiment (null/failed runs are never hidden).",
        "",
    ]
    if frame.height:
        for row in frame.sort(["experiment_id"]).iter_rows(named=True):
            lines.append(
                f"- **{row['experiment_id']}** {row['feature_set_id']} "
                f"({row['set_role']} {row['set_family'] or ''}) {row['family']}: "
                f"status {row['status']}, OOS IC {row['oos_ic']}, rank IC "
                f"{row['rank_ic']}, after-cost return {row['after_cost_total_return']}, "
                f"turnover {row['turnover']}"
            )
    else:
        lines.append("(no experiment rows yet)")
    lines += [
        "",
        "## 6. Scientific conclusion",
        "",
        "See the permanent status report (PHASE_10_STATUS.md) for the final "
        "verdict and the independent reviews.",
        "",
    ]
    RESEARCH_MD.write_text("\n".join(lines), encoding="utf-8")
    return RESEARCH_MD


__all__ = [
    "REPORT_PARQUET",
    "REPORT_MARKDOWN",
    "PLAN_JSON",
    "DIAGNOSTICS_JSON",
    "RESEARCH_MD",
    "ARTIFACTS_ROOT",
    "append_report_rows",
    "write_plan",
    "write_diagnostics",
    "write_markdown_report",
    "write_research_report",
]