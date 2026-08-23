"""Phase 11 runner: orchestrate the locked inference analysis.

Reads existing Phase 9/10 artifacts, runs the locked inference plan,
and produces permanent results. This module does NOT retrain models
or rerun backtests.
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

from orbit.ml.phase11_bootstrap import (
    bootstrap_ci,
    rule_of_thumb_block_length,
)
from orbit.ml.phase11_effects import compute_effect_size
from orbit.ml.phase11_inference import (
    ConfidenceInterval,
    InferenceResult,
    compute_ci,
    make_inference_result_id,
)
from orbit.ml.phase11_multiple_testing import multiple_testing_analysis
from orbit.ml.phase11_power import achieved_power_independent, min_detectable_effect
from orbit.ml.phase11_plan import (
    CONFIDENCE_LEVEL,
    SEED,
    load_plan,
    phase11_plan_digest,
    verify_plan,
)

_ANALYSIS_N_RESAMPLES = 1000

_REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_PARQUET = _REPO_ROOT / "benchmarks" / "phase11_inference_results.parquet"
RESULTS_MD = _REPO_ROOT / "benchmarks" / "phase11_inference_results.md"
RUNS_DIR = _REPO_ROOT / "benchmarks" / "phase11_inference_runs"


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
        # Pad to clear previous line
        pad = max(0, self._last_width - len(line))
        self.file.write(line + " " * pad)
        self.file.flush()
        self._last_width = len(line)

    def finish(self, detail: str = "") -> None:
        elapsed = time.time() - self.start_time
        bar = "#" * 30
        line = (
            f"\r  [{bar}] {self.total}/{self.total} "
            f"(100%) {elapsed:.1f}s total"
        )
        if detail:
            line += f" | {detail}"
        pad = max(0, self._last_width - len(line))
        self.file.write(line + " " * pad + "\n")
        self.file.flush()


def _file_checksum(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_phase9_metrics() -> list[dict[str, Any]]:
    parquet = _REPO_ROOT / "benchmarks" / "phase9_ml_benchmark.parquet"
    if not parquet.exists():
        return []
    df = pl.read_parquet(parquet)
    records = []
    for row in df.iter_rows(named=True):
        records.append({
            "experiment_id": row["experiment_id"],
            "source": "phase9",
            "metrics": {
                "oos_ic": row.get("oos_ic"),
                "rank_ic": row.get("rank_ic"),
                "ece": row.get("ece"),
                "brier": row.get("brier"),
                "mse": row.get("mse"),
                "hit_rate": row.get("hit_rate"),
                "after_cost_total_return": row.get("after_cost_total_return"),
                "turnover": row.get("turnover"),
                "total_costs": row.get("total_costs"),
            },
        })
    return records


def _load_phase10_metrics() -> list[dict[str, Any]]:
    parquet = _REPO_ROOT / "benchmarks" / "phase10_feature_research.parquet"
    if not parquet.exists():
        return []
    df = pl.read_parquet(parquet)
    records = []
    for row in df.iter_rows(named=True):
        records.append({
            "experiment_id": row["experiment_id"],
            "source": "phase10",
            "feature_set_id": row.get("feature_set_id"),
            "family": row.get("family"),
            "metrics": {
                "oos_ic": row.get("oos_ic"),
                "rank_ic": row.get("rank_ic"),
                "ece": row.get("ece"),
                "brier": row.get("brier"),
                "mse": row.get("mse"),
                "hit_rate": row.get("hit_rate"),
                "after_cost_total_return": row.get("after_cost_total_return"),
                "turnover": row.get("turnover"),
                "total_costs": row.get("total_costs"),
            },
        })
    return records


def _load_phase10_test_predictions(experiment_id: str) -> pl.DataFrame | None:
    runs_dir = _REPO_ROOT / "benchmarks" / "phase10_runs"
    pred_file = runs_dir / experiment_id / "test_predictions.parquet"
    if not pred_file.exists():
        return None
    return pl.read_parquet(pred_file)


def _load_phase9_test_predictions(experiment_id: str) -> pl.DataFrame | None:
    runs_dir = _REPO_ROOT / "benchmarks" / "phase9_runs"
    pred_file = runs_dir / experiment_id / "test_predictions.parquet"
    if not pred_file.exists():
        return None
    return pl.read_parquet(pred_file)


def run_single_experiment_inference(
    experiment_id: str,
    metrics: dict[str, Any],
    test_predictions: pl.DataFrame | None,
    source_checksum: str,
    plan_digest: str,
) -> list[InferenceResult]:
    results: list[InferenceResult] = []
    _bootstrap_failures = 0

    for metric in ("oos_ic", "rank_ic", "hit_rate", "mse", "brier",
                    "ece", "after_cost_total_return", "turnover", "total_costs"):
        value = metrics.get(metric)
        if value is None or (isinstance(value, float) and np.isnan(value)):
            continue

        ci: ConfidenceInterval | None = None

        if metric in ("oos_ic", "rank_ic") and test_predictions is not None:
            try:
                sessions = test_predictions["decision_session"].unique().sort().to_list()
                session_ics = []
                pred_col = "prediction"
                label_col = "outcome_value"
                if pred_col in test_predictions.columns and label_col in test_predictions.columns:
                    for sess in sessions:
                        sess_df = test_predictions.filter(pl.col("decision_session") == sess)
                        if sess_df.height < 3:
                            continue
                        x = sess_df[pred_col].to_numpy()
                        y = sess_df[label_col].to_numpy()
                        if np.std(x) == 0 or np.std(y) == 0:
                            continue
                        if metric == "rank_ic":
                            from scipy.stats import rankdata
                            x = rankdata(x, method="average")
                            y = rankdata(y, method="average")
                        ic = float(np.corrcoef(x, y)[0, 1])
                        if not np.isnan(ic):
                            session_ics.append(ic)
                if len(session_ics) >= 2:
                    session_arr = np.array(session_ics)
                    bl = rule_of_thumb_block_length(len(session_arr))
                    boot = bootstrap_ci(
                        session_arr,
                        statistic=lambda x: float(np.mean(x)),
                        method="moving_block",
                        n_resamples=_ANALYSIS_N_RESAMPLES,
                        block_length=bl,
                        seed=SEED,
                        confidence_level=CONFIDENCE_LEVEL,
                    )
                    ci = ConfidenceInterval(
                        point_estimate=boot.point_estimate,
                        lower=boot.ci_lower,
                        upper=boot.ci_upper,
                        confidence_level=CONFIDENCE_LEVEL,
                        method="moving_block_bootstrap",
                        assumptions="stationary time series with short-range dependence",
                        sample_size=len(session_arr),
                        effective_sample_size=None,
                        seed=SEED,
                        n_resamples=_ANALYSIS_N_RESAMPLES,
                    )
            except Exception:
                _bootstrap_failures += 1

        if ci is None:
            ci = ConfidenceInterval(
                point_estimate=float(value),
                lower=float(value),
                upper=float(value),
                confidence_level=CONFIDENCE_LEVEL,
                method="point_estimate_only",
                assumptions="insufficient data for CI; point estimate only",
                sample_size=1,
                seed=SEED,
            )

        effect = compute_effect_size(metric, value)

        p_value = None
        if ci.width() > 0:
            if ci.lower > 0 or ci.upper < 0:
                from scipy import stats as sp_stats
                if ci.sample_size > 1:
                    se = ci.width() / (2.0 * sp_stats.norm.ppf(0.5 + CONFIDENCE_LEVEL / 2.0))
                    if se > 0:
                        z_stat = ci.point_estimate / se
                        p_value = float(2 * sp_stats.norm.sf(abs(z_stat)))
            elif ci.lower <= 0 <= ci.upper:
                p_value = 1.0
        elif ci.point_estimate == 0:
            p_value = 1.0

        inf_id = make_inference_result_id([experiment_id], metric, "moving_block_bootstrap", SEED)
        results.append(InferenceResult(
            inference_result_id=inf_id,
            source_experiment_ids=[experiment_id],
            source_artifact_checksums={"metrics": source_checksum},
            metric=metric,
            ci=ci,
            p_value=p_value,
            effect_size=effect.magnitude,
            effect_size_method=effect.method,
            seed=SEED,
            inference_plan_digest=plan_digest,
        ))

    return results


def run_phase11_analysis(
    plan: dict[str, Any] | None = None,
    progress: bool = True,
) -> dict[str, Any]:
    """Run the full locked Phase 11 inference analysis."""
    file = sys.stdout if progress else None

    if progress:
        print("=" * 72, file=file)
        print("PHASE 11 - STATISTICAL INFERENCE ANALYSIS", file=file)
        print("=" * 72, file=file)

    # Step 1: plan verification
    if progress:
        print("\n[1/6] Verifying locked inference plan...", file=file)
    if plan is None:
        verify_plan()
        plan = load_plan()
    plan_digest = plan["plan_digest"]
    if progress:
        print(f"  Plan digest: {plan_digest[:32]}...", file=file)

    # Step 2: load source artifacts
    if progress:
        print("\n[2/6] Loading source artifacts...", file=file)
    phase9 = _load_phase9_metrics()
    phase10 = _load_phase10_metrics()
    if progress:
        print(f"  Phase 9: {len(phase9)} experiments from parquet", file=file)
        print(f"  Phase 10: {len(phase10)} experiments from parquet", file=file)

    p9_parquet = _REPO_ROOT / "benchmarks" / "phase9_ml_benchmark.parquet"
    p10_parquet = _REPO_ROOT / "benchmarks" / "phase10_feature_research.parquet"
    p9_checksum = _file_checksum(p9_parquet) if p9_parquet.exists() else "MISSING"
    p10_checksum = _file_checksum(p10_parquet) if p10_parquet.exists() else "MISSING"
    if progress:
        print(f"  Phase 9 checksum: {p9_checksum[:16]}...", file=file)
        print(f"  Phase 10 checksum: {p10_checksum[:16]}...", file=file)

    # Step 3: per-experiment inference with progress
    if progress:
        print(f"\n[3/6] Running inference on {len(phase10)} Phase 10 experiments...", file=file)
    all_results: list[InferenceResult] = []
    phase10_p_values: list[float] = []
    phase10_experiment_ids: list[str] = []

    tracker = ProgressTracker(len(phase10), "Phase 10", file=file) if progress else None
    for exp in phase10:
        eid = exp["experiment_id"]
        metrics = exp["metrics"]
        tp = _load_phase10_test_predictions(eid)
        results = run_single_experiment_inference(eid, metrics, tp, p10_checksum, plan_digest)
        all_results.extend(results)
        ic_result = next((r for r in results if r.metric == "oos_ic"), None)
        if ic_result and ic_result.p_value is not None:
            phase10_p_values.append(ic_result.p_value)
            phase10_experiment_ids.append(eid)
        if tracker:
            tracker.update(1, eid)
    if tracker:
        tracker.finish(f"{len(results)} metrics per experiment")

    # Phase 9
    if progress:
        print(f"\n[4/6] Running inference on {len(phase9)} Phase 9 experiments...", file=file)
    tracker9 = ProgressTracker(len(phase9), "Phase 9", file=file) if progress else None
    for exp in phase9:
        eid = exp["experiment_id"]
        metrics = exp["metrics"]
        tp = _load_phase9_test_predictions(eid)
        results = run_single_experiment_inference(eid, metrics, tp, p9_checksum, plan_digest)
        all_results.extend(results)
        if tracker9:
            tracker9.update(1, eid)
    if tracker9:
        tracker9.finish()

    # Multiple testing
    if progress:
        print(f"\n[5/6] Multiple-comparison analysis...", file=file)
    mt_results = None
    if phase10_p_values:
        mt_results = multiple_testing_analysis(
            phase10_p_values, phase10_experiment_ids, "phase10_grid"
        )
        if progress:
            holm = mt_results.get("holm_bonferroni", {})
            bh = mt_results.get("benjamini_hochberg", {})
            print(f"  Family: {mt_results['family']['n_members']} experiments", file=file)
            print(f"  Raw p<0.05: {mt_results.get('n_raw_significant_005', 0)}", file=file)
            print(f"  Holm p<0.05: {len(holm.get('significant_at', {}).get('0.05', []))}", file=file)
            print(f"  BH p<0.05: {len(bh.get('significant_at', {}).get('0.05', []))}", file=file)

    # Power: Note: sample_size here is the number of independent experiment-level
    # observations (52 Phase 10 experiments), not sessions. This gives the power
    # to detect an effect across experiments, accounting for experiment-level variation.
    # The actual per-session n_eff is much larger but depends on autocorrelation structure.
    if progress:
        print(f"\n[6/6] Power analysis...", file=file)
    power_results = {
        "min_detectable_ic_80power": min_detectable_effect(
            len(phase10_experiment_ids), alpha=0.05, power=0.80
        ).summary(),
        "achieved_power_ic_003": achieved_power_independent(
            0.03, len(phase10_experiment_ids), alpha=0.05
        ).summary(),
    }
    if progress:
        print(f"  Min detectable IC (80% power): {power_results['min_detectable_ic_80power']['assumed_effect_size']:.4f}", file=file)
        print(f"  Achieved power for IC=0.03: {power_results['achieved_power_ic_003']['achieved_power']:.1%}", file=file)

    return {
        "plan_digest": plan_digest,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "n_phase9_experiments": len(phase9),
        "n_phase10_experiments": len(phase10),
        "n_inference_results": len(all_results),
        "inference_results": all_results,
        "multiple_testing": mt_results,
        "power_analysis": power_results,
        "phase9_checksum": p9_checksum,
        "phase10_checksum": p10_checksum,
    }


def persist_results(analysis: dict[str, Any]) -> Path:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in analysis["inference_results"]:
        rows.append(r.summary())
    if rows:
        df = pl.DataFrame(rows)
        df.write_parquet(RESULTS_PARQUET)
    return RESULTS_PARQUET


__all__ = [
    "RESULTS_PARQUET",
    "RESULTS_MD",
    "RUNS_DIR",
    "ProgressTracker",
    "run_phase11_analysis",
    "persist_results",
]
