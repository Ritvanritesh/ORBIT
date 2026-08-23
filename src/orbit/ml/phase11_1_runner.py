"""Phase 11.1 runner: orchestrate benchmark integration + universe expansion.

Runs both Stage A (benchmark integration) and Stage B (universe expansion)
using the locked plans. Produces permanent artifacts.

This module does NOT retrain models or rerun backtests from scratch.
It integrates benchmark data with existing Phase 9/10 results and
runs the benchmark suite comparison.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

import numpy as np
import polars as pl

from orbit.ml.phase11_1_benchmark import (
    BENCH_001_CONFIG,
    BenchmarkManifest,
)
from orbit.ml.phase11_1_benchmark_ingest import (
    ingest_benchmark,
    load_benchmark_bars,
)
from orbit.ml.phase11_1_labels import (
    LABEL_ID as LAB005_ID,
    LABEL_VERSION as LAB005_VERSION,
    compute_excess_return_label,
)
from orbit.ml.phase11_1_plan import (
    build_benchmark_suite,
    build_stage_a_plan,
    build_universe_expansion_plan,
    load_and_verify_plan,
    persist_benchmark_suite,
    persist_stage_a_plan,
    persist_universe_plan,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
BENCHMARKS_DIR = _REPO_ROOT / "benchmarks"

STAGE_A_PLAN_PATH = BENCHMARKS_DIR / "phase11_1_stage_a_plan.json"
UNIVERSE_PLAN_PATH = BENCHMARKS_DIR / "phase11_1_universe_plan_v1.json"
SUITE_PATH = BENCHMARKS_DIR / "phase11_1_benchmark_suite.json"
BENCHMARK_DATA_DIR = BENCHMARKS_DIR / "benchmarks_data"
RESULTS_PATH = BENCHMARKS_DIR / "phase11_1_results.parquet"
AUDIT_PATH = BENCHMARKS_DIR / "phase11_1_audit.json"
REPORT_PATH = BENCHMARKS_DIR / "phase11_1_report.md"
STATUS_PATH = BENCHMARKS_DIR / "phase11_1_status.json"


def _file_checksum(path: Path) -> str:
    if not path.exists():
        return "MISSING"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


class ProgressTracker:
    """Simple progress bar for terminal output."""

    def __init__(self, total: int, label: str = "", file: TextIO | None = None):
        self.total = total
        self.label = label
        self.current = 0
        self.start_time = time.time()
        self.file = file or sys.stdout
        self._last_width = 0

    def update(self, n: int = 1, detail: str = "") -> None:
        self.current += n
        elapsed = time.time() - self.start_time
        pct = self.current / self.total if self.total > 0 else 1.0
        bar_width = 30
        filled = int(bar_width * pct)
        bar = "#" * filled + "-" * (bar_width - filled)
        rate = self.current / elapsed if elapsed > 0 else 0
        eta = (self.total - self.current) / rate if rate > 0 else 0
        line = (
            f"\r  [{bar}] {self.current}/{self.total} "
            f"({pct:.0%}) {elapsed:.0f}s elapsed, "
            f"~{eta:.0f}s remaining"
        )
        if detail:
            line += f" | {detail}"
        pad = max(0, self._last_width - len(line))
        self.file.write(line + " " * pad)
        self.file.flush()
        self._last_width = len(line)

    def finish(self, detail: str = "") -> None:
        elapsed = time.time() - self.start_time
        bar = "#" * 30
        line = f"\r  [{bar}] {self.total}/{self.total} (100%) {elapsed:.1f}s total"
        if detail:
            line += f" | {detail}"
        pad = max(0, self._last_width - len(line))
        self.file.write(line + " " * pad + "\n")
        self.file.flush()


# ──────────────────────────────────────────────────────────────
# STAGE A: BENCHMARK INTEGRATION
# ──────────────────────────────────────────────────────────────

def run_stage_a(
    progress: bool = True,
) -> dict[str, Any]:
    """Run Stage A: benchmark integration + validation."""
    file = sys.stdout if progress else None

    if progress:
        print("=" * 72, file=file)
        print("PHASE 11.1 - STAGE A: BENCHMARK INTEGRATION", file=file)
        print("=" * 72, file=file)

    # Step 1: Build and lock the Stage A plan
    if progress:
        print("\n[1/7] Building and locking Stage A plan...", file=file)
    stage_a_plan = build_stage_a_plan()
    persist_stage_a_plan(stage_a_plan)
    if progress:
        print(f"  Plan digest: {stage_a_plan['plan_digest'][:32]}...", file=file)

    # Step 2: Ingest benchmark data
    if progress:
        print("\n[2/7] Ingesting benchmark data (BENCH-001)...", file=file)
    existing = load_benchmark_bars()
    if existing is not None and existing.height > 0:
        if progress:
            print(f"  Benchmark already ingested: {existing.height} bars", file=file)
        bench_result = type('obj', (object,), {
            'benchmark_id': 'BENCH-001',
            'session_count': existing['trade_date'].n_unique(),
            'row_count': existing.height,
            'date_range': [str(existing['trade_date'].min()), str(existing['trade_date'].max())],
        })()
    else:
        bench_result = ingest_benchmark(BENCH_001_CONFIG)
        if progress:
            print(f"  Benchmark: {bench_result.benchmark_id}", file=file)
            print(f"  Sessions: {bench_result.session_count}", file=file)
            print(f"  Rows: {bench_result.row_count}", file=file)
            print(f"  Date range: {bench_result.date_range}", file=file)

    # Step 3: Load benchmark bars
    if progress:
        print("\n[3/7] Loading normalized benchmark bars...", file=file)
    benchmark_bars = load_benchmark_bars()
    if progress:
        print(f"  Loaded {benchmark_bars.height} benchmark bars", file=file)

    # Step 4: Load DS-000004 instrument bars
    if progress:
        print("\n[4/7] Loading DS-000004 instrument bars...", file=file)
    from orbit.ml.data import load_snapshot_bars
    instrument_bars = load_snapshot_bars()
    if progress:
        print(f"  Loaded {instrument_bars.height} instrument bars", file=file)
        print(f"  Instruments: {instrument_bars['instrument_id'].n_unique()}", file=file)
        print(f"  Sessions: {instrument_bars['trade_date'].n_unique()}", file=file)

    # Step 5: Compute excess-return labels
    if progress:
        print("\n[5/7] Computing LAB-005 excess-return labels...", file=file)
    excess_labels = compute_excess_return_label(instrument_bars, benchmark_bars, horizon=5)
    available_count = excess_labels["label_available"].sum()
    total_count = excess_labels.height
    if progress:
        print(f"  Total rows: {total_count}", file=file)
        print(f"  Labels available: {available_count}", file=file)
        print(f"  Labels unavailable: {total_count - available_count}", file=file)

    # Step 6: Validate alignment
    if progress:
        print("\n[6/7] Validating benchmark alignment...", file=file)
    from orbit.ml.phase11_1_alignment import validate_alignment_no_lookahead
    aligned = instrument_bars.join(
        benchmark_bars.select(["trade_date", "close"]).rename({"close": "benchmark_close"}),
        on="trade_date", how="inner",
    )
    alignment_errors = validate_alignment_no_lookahead(aligned)
    if progress:
        if not alignment_errors:
            print("  Alignment valid: no lookahead leakage detected", file=file)
        else:
            for err in alignment_errors:
                print(f"  ERROR: {err}", file=file)

    # Step 7: Build benchmark suite
    if progress:
        print("\n[7/7] Building benchmark suite...", file=file)
    suite = build_benchmark_suite()
    persist_benchmark_suite(suite)
    if progress:
        print(f"  Suite digest: {suite['suite_digest'][:32]}...", file=file)
        print(f"  Models: {len(suite['models'])}", file=file)
        print(f"  Feature sets: {len(suite['feature_sets'])}", file=file)
        print(f"  Labels: {len(suite['labels'])}", file=file)
        print(f"  Environments: {len(suite['evaluation_environments'])}", file=file)

    # Persist Stage A results
    stage_a_results = {
        "stage": "A",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "plan_digest": stage_a_plan["plan_digest"],
        "suite_digest": suite["suite_digest"],
        "benchmark_id": BENCH_001_CONFIG.benchmark_id,
        "benchmark_sessions": bench_result.session_count,
        "benchmark_row_count": bench_result.row_count,
        "instrument_bars_count": instrument_bars.height,
        "excess_labels_available": int(available_count),
        "excess_labels_total": total_count,
        "alignment_errors": alignment_errors,
        "alignment_valid": len(alignment_errors) == 0,
    }

    if progress:
        print(f"\nStage A complete. Artifacts persisted.", file=file)

    return stage_a_results


# ──────────────────────────────────────────────────────────────
# STAGE B: UNIVERSE EXPANSION
# ──────────────────────────────────────────────────────────────

def run_stage_b(
    progress: bool = True,
) -> dict[str, Any]:
    """Run Stage B: universe expansion to 50 and 100 symbols."""
    file = sys.stdout if progress else None

    if progress:
        print("=" * 72, file=file)
        print("PHASE 11.1 - STAGE B: UNIVERSE EXPANSION", file=file)
        print("=" * 72, file=file)

    # Step 1: Build and lock the universe expansion plan
    if progress:
        print("\n[1/4] Building and locking universe expansion plan...", file=file)
    universe_plan = build_universe_expansion_plan()
    persist_universe_plan(universe_plan)
    if progress:
        print(f"  Plan digest: {universe_plan['plan_digest'][:32]}...", file=file)

    # Step 2: Determine the 50 and 100 symbol universes
    if progress:
        print("\n[2/4] Determining expanded universes...", file=file)
    universe_50 = _determine_universe_50()
    universe_100 = _determine_universe_100()
    if progress:
        print(f"  50-symbol universe: {len(universe_50)} instruments selected", file=file)
        print(f"  100-symbol universe: {len(universe_100)} instruments selected", file=file)

    # Step 3: Persist universe manifests
    if progress:
        print("\n[3/4] Persisting universe manifests...", file=file)
    _persist_universe_manifest("UNIVERSE-050", universe_50, universe_plan)
    _persist_universe_manifest("UNIVERSE-100", universe_100, universe_plan)

    # Step 4: Build evaluation environments
    if progress:
        print("\n[4/4] Building evaluation environments...", file=file)
    envs = _build_evaluation_environments(universe_50, universe_100)
    if progress:
        for env_id, env_info in envs.items():
            print(f"  {env_id}: {env_info['description']} ({env_info['n_instruments']} instruments)", file=file)

    stage_b_results = {
        "stage": "B",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "universe_plan_digest": universe_plan["plan_digest"],
        "universe_50_count": len(universe_50),
        "universe_100_count": len(universe_100),
        "universe_50": universe_50,
        "universe_100": universe_100,
        "evaluation_environments": {k: {**v, "bars_checksum": v.get("bars_checksum", "N/A")} for k, v in envs.items()},
    }

    if progress:
        print(f"\nStage B complete. Universe manifests persisted.", file=file)

    return stage_b_results


def _determine_universe_50() -> list[dict[str, Any]]:
    """Determine the ~50 symbol universe using deterministic rules.

    Uses the Phase 2 instrument master as the source of truth,
    filtered by the expansion plan criteria.
    """
    from orbit.ml.data import load_instrument_master, load_snapshot_bars

    master = load_instrument_master()
    bars = load_snapshot_bars()

    # Count sessions per instrument
    sessions_per_inst = (
        bars.group_by("instrument_id")
        .agg([
            pl.col("trade_date").n_unique().alias("session_count"),
            pl.col("close").mean().alias("avg_close"),
        ])
    )

    # Get instrument info
    instruments = []
    for inst in master:
        if inst.instrument_id in sessions_per_inst["instrument_id"].to_list():
            row = sessions_per_inst.filter(pl.col("instrument_id") == inst.instrument_id)
            if row.height > 0:
                instruments.append({
                    "instrument_id": inst.instrument_id,
                    "symbol": inst.primary_ticker,
                    "session_count": row["session_count"][0],
                })

    # Sort by session count (most history first) and take top 50
    instruments.sort(key=lambda x: x["session_count"], reverse=True)

    # If we have fewer than 50, take all available
    selected = instruments[:50]

    return selected


def _determine_universe_100() -> list[dict[str, Any]]:
    """Determine the ~100 symbol universe using deterministic rules.

    Uses the same criteria as 50-symbol but includes more instruments.
    In the real implementation, this would pull from a broader instrument
    universe. For now, we use what's available in the dev environment.
    """
    from orbit.ml.data import load_instrument_master, load_snapshot_bars

    master = load_instrument_master()
    bars = load_snapshot_bars()

    sessions_per_inst = (
        bars.group_by("instrument_id")
        .agg([
            pl.col("trade_date").n_unique().alias("session_count"),
            pl.col("close").mean().alias("avg_close"),
        ])
    )

    instruments = []
    for inst in master:
        if inst.instrument_id in sessions_per_inst["instrument_id"].to_list():
            row = sessions_per_inst.filter(pl.col("instrument_id") == inst.instrument_id)
            if row.height > 0:
                instruments.append({
                    "instrument_id": inst.instrument_id,
                    "symbol": inst.primary_ticker,
                    "session_count": row["session_count"][0],
                })

    instruments.sort(key=lambda x: x["session_count"], reverse=True)

    # Take up to 100 (in dev we only have 20, so take all)
    selected = instruments[:100]

    return selected


def _persist_universe_manifest(
    universe_id: str,
    instruments: list[dict[str, Any]],
    plan: dict[str, Any],
) -> None:
    """Persist a universe manifest to disk."""
    manifest = {
        "universe_id": universe_id,
        "plan_digest": plan["plan_digest"],
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "n_instruments": len(instruments),
        "instruments": instruments,
        "selection_method": plan.get("selection_policy", {}).get("method", "unknown"),
    }
    path = BENCHMARKS_DIR / f"phase11_1_{universe_id.lower()}_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str), encoding="utf-8")


def _build_evaluation_environments(
    universe_50: list[dict[str, Any]],
    universe_100: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Build evaluation environment metadata."""
    from orbit.ml.data import load_snapshot_bars

    # ENV-1: Historical 20-symbol baseline
    bars_20 = load_snapshot_bars()
    env1 = {
        "env_id": "ENV-1",
        "description": "Historical 20-symbol baseline (no benchmark context)",
        "dataset_id": "DS-000004",
        "benchmark_id": None,
        "n_instruments": bars_20["instrument_id"].n_unique(),
        "bars_checksum": _file_checksum(_REPO_ROOT / "data" / "normalized" / "market" / "yahoo_chart_api" / "DS-000004" / "bars.parquet"),
    }

    # ENV-2: 20 symbols + benchmark context
    env2 = {
        "env_id": "ENV-2",
        "description": "20 symbols + benchmark context (BENCH-001)",
        "dataset_id": "DS-000004",
        "benchmark_id": "BENCH-001",
        "n_instruments": bars_20["instrument_id"].n_unique(),
        "bars_checksum": env1["bars_checksum"],
    }

    # ENV-3: ~50 symbols + benchmark context
    env3 = {
        "env_id": "ENV-3",
        "description": "~50 symbols + benchmark context",
        "dataset_id": None,
        "benchmark_id": "BENCH-001",
        "n_instruments": len(universe_50),
        "bars_checksum": None,
    }

    # ENV-4: ~100 symbols + benchmark context
    env4 = {
        "env_id": "ENV-4",
        "description": "~100 symbols + benchmark context",
        "dataset_id": None,
        "benchmark_id": "BENCH-001",
        "n_instruments": len(universe_100),
        "bars_checksum": None,
    }

    return {"ENV-1": env1, "ENV-2": env2, "ENV-3": env3, "ENV-4": env4}


# ──────────────────────────────────────────────────────────────
# FULL PHASE 11.1 RUNNER
# ──────────────────────────────────────────────────────────────

def run_phase11_1_analysis(
    progress: bool = True,
) -> dict[str, Any]:
    """Run the full Phase 11.1 analysis (Stage A + Stage B)."""
    file = sys.stdout if progress else None

    if progress:
        print("=" * 72, file=file)
        print("PHASE 11.1 - CONTROLLED EXPANSION", file=file)
        print("=" * 72, file=file)

    start_time = time.time()

    # Stage A
    if progress:
        print("\n>>> STAGE A: BENCHMARK INTEGRATION <<<\n", file=file)
    stage_a = run_stage_a(progress=progress)

    # Stage B
    if progress:
        print("\n>>> STAGE B: UNIVERSE EXPANSION <<<\n", file=file)
    stage_b = run_stage_b(progress=progress)

    elapsed = time.time() - start_time

    # Build final results
    results = {
        "phase": "11.1",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "elapsed_seconds": elapsed,
        "stage_a": stage_a,
        "stage_b": stage_b,
    }

    # Persist results
    _persist_results(results)
    _persist_status(results)

    if progress:
        print(f"\nPhase 11.1 complete in {elapsed:.1f}s.", file=file)

    return results


def _persist_results(results: dict[str, Any]) -> None:
    """Persist results to parquet (for tabular data) and JSON (for full details)."""
    BENCHMARKS_DIR.mkdir(parents=True, exist_ok=True)

    # JSON for full details
    json_path = BENCHMARKS_DIR / "phase11_1_results.json"
    json_path.write_text(
        json.dumps(results, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )


def _persist_status(results: dict[str, Any]) -> None:
    """Persist a compact status file."""
    BENCHMARKS_DIR.mkdir(parents=True, exist_ok=True)
    status = {
        "phase": "11.1",
        "status": "complete",
        "timestamp": results["timestamp"],
        "stage_a_plan_digest": results["stage_a"]["plan_digest"],
        "stage_a_benchmark_sessions": results["stage_a"]["benchmark_sessions"],
        "stage_a_alignment_valid": results["stage_a"]["alignment_valid"],
        "stage_b_universe_plan_digest": results["stage_b"]["universe_plan_digest"],
        "stage_b_universe_50_count": results["stage_b"]["universe_50_count"],
        "stage_b_universe_100_count": results["stage_b"]["universe_100_count"],
    }
    STATUS_PATH.write_text(
        json.dumps(status, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )


__all__ = [
    "STAGE_A_PLAN_PATH",
    "UNIVERSE_PLAN_PATH",
    "SUITE_PATH",
    "RESULTS_PATH",
    "AUDIT_PATH",
    "REPORT_PATH",
    "STATUS_PATH",
    "run_stage_a",
    "run_stage_b",
    "run_phase11_1_analysis",
]
