"""Phase 11.2 benchmark execution: run locked benchmark suite on expanded universes.

Runs the locked benchmark suite (4 models × 2 feature sets × 2 labels)
on the 50-symbol and 100-symbol datasets.

This module reuses existing ORBIT ML infrastructure but operates on new datasets.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[3]

from orbit.ml.phase11_1_plan import build_benchmark_suite, load_and_verify_plan
from orbit.ml.phase11_1_labels import compute_excess_return_label, LABEL_ID as LAB005_ID


def _file_checksum(path: Path) -> str:
    if not path.exists():
        return "MISSING"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def load_dataset(snapshot_id: str) -> tuple[pl.DataFrame, pl.DataFrame | None]:
    """Load normalized bars and events from a dataset."""
    from orbit.ingestion.paths import normalized_dir
    base = normalized_dir("market", "yahoo_chart_api", snapshot_id)
    bars_path = base / "bars.parquet"
    if not bars_path.exists():
        raise FileNotFoundError(f"Dataset not found: {bars_path}")
    bars = pl.read_parquet(bars_path)
    events_path = base / "events.parquet"
    events = pl.read_parquet(events_path) if events_path.exists() else None
    return bars, events


def load_benchmark_bars() -> pl.DataFrame:
    """Load BENCH-001 (SPY) benchmark bars."""
    from orbit.ingestion.paths import data_root
    bench_path = data_root() / "normalized" / "benchmark" / "BENCH-001" / "bars.parquet"
    if not bench_path.exists():
        raise FileNotFoundError(f"Benchmark not found: {bench_path}")
    return pl.read_parquet(bench_path)


def compute_features_fs001(bars: pl.DataFrame) -> pl.DataFrame:
    """Compute FS-001 features (8 features) for all instruments."""
    from orbit.ml.features import build_feature_frame
    return build_feature_frame(bars)


def compute_features_fs003(bars: pl.DataFrame) -> pl.DataFrame:
    """Compute FS-003 features (23 features) for all instruments."""
    from orbit.ml.features import build_phase10_all_feature_frame
    return build_phase10_all_feature_frame(bars)


def compute_labels_lab004(
    bars: pl.DataFrame,
    events: pl.DataFrame | None,
    feature_sessions: pl.DataFrame,
) -> pl.DataFrame:
    """Compute LAB-004 labels (5-session forward return)."""
    from orbit.ml.labels import build_phase9_label_snapshot
    from orbit.ml.data import load_instrument_master

    instruments = load_instrument_master()
    ls = build_phase9_label_snapshot(bars, events, instruments, feature_sessions)
    return ls.records


def compute_labels_lab005(
    bars: pl.DataFrame,
    benchmark_bars: pl.DataFrame,
    feature_sessions: pl.DataFrame,
) -> pl.DataFrame:
    """Compute LAB-005 labels (5-session excess return vs BENCH-001)."""
    excess = compute_excess_return_label(bars, benchmark_bars, horizon=5)
    return excess


def assemble_dataset(
    feature_frame: pl.DataFrame,
    label_frame: pl.DataFrame,
    feature_names: list[str],
    label_col: str = "outcome_value",
) -> dict[str, Any]:
    """Assemble train/val/test datasets from features and labels."""
    from orbit.ml.dataset import assemble_datasets
    from orbit.ml.features import FeatureSnapshot
    from orbit.ml.labels import LabelSnapshot

    # Create FeatureSnapshot
    fs = FeatureSnapshot(records=feature_frame, data_refs=["phase11_2"])

    # Create LabelSnapshot
    ls = LabelSnapshot(records=label_frame, data_refs=["phase11_2"])

    return assemble_datasets(fs, ls, feature_names=feature_names)


def train_and_evaluate(
    family: str,
    params: dict[str, Any],
    datasets: dict[str, Any],
    feature_names: list[str],
    seed: int = 42,
) -> dict[str, Any]:
    """Train a model and evaluate on test set."""
    from orbit.ml.models import train_model, predict_with_state
    from orbit.ml.metrics import oos_ic, rank_ic, hit_rate, mean_squared_error
    from orbit.ml.calibration import fit_platt, expected_calibration_error, brier_score

    X_train, y_train_reg, y_train_bin, meta_train = datasets["train"]
    X_val, y_val_reg, y_val_bin, meta_val = datasets["val"]
    X_test, y_test_reg, y_test_bin, meta_test = datasets["test"]

    if len(X_train) == 0 or len(X_test) == 0:
        return {"error": "insufficient data"}

    # Train
    model, state = train_model(family, params, X_train, y_train_reg,
                                feature_names=feature_names, seed=seed)

    # Predict
    pred_val = predict_with_state(model, state, X_val)
    pred_test = predict_with_state(model, state, X_test)

    # Calibrate on validation
    calibrator = fit_platt(pred_val, y_val_bin)
    pred_test_cal = calibrator.apply(pred_test)

    # Build test frame for metrics
    test_frame = meta_test.with_columns([
        pl.Series("prediction", pred_test_cal),
        pl.Series("outcome_value", y_test_reg),
    ])

    # Compute metrics
    metrics = {}
    try:
        metrics["oos_ic"] = float(oos_ic(test_frame))
    except Exception:
        metrics["oos_ic"] = None
    try:
        metrics["rank_ic"] = float(rank_ic(test_frame))
    except Exception:
        metrics["rank_ic"] = None
    try:
        metrics["hit_rate"] = float(hit_rate(test_frame))
    except Exception:
        metrics["hit_rate"] = None
    try:
        metrics["mse"] = float(mean_squared_error(y_test_reg, pred_test_cal))
    except Exception:
        metrics["mse"] = None
    try:
        metrics["brier"] = float(brier_score(pred_test_cal, y_test_bin))
    except Exception:
        metrics["brier"] = None
    try:
        metrics["ece"] = float(expected_calibration_error(pred_test_cal, y_test_bin))
    except Exception:
        metrics["ece"] = None

    return {
        "family": family,
        "params": params,
        "metrics": metrics,
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test),
        "predictions": test_frame,
    }


def run_benchmark_suite(
    snapshot_id: str,
    env_id: str,
    label_type: str = "both",
    progress: bool = True,
) -> dict[str, Any]:
    """Run the locked benchmark suite on a dataset.

    Parameters
    ----------
    snapshot_id : str
        Dataset snapshot ID (e.g., "DS-EXP-050")
    env_id : str
        Environment identifier (e.g., "ENV-3")
    label_type : str
        "absolute" (LAB-004), "excess" (LAB-005), or "both"
    progress : bool
        Print progress
    """
    file = sys.stdout if progress else None

    if progress:
        print(f"\n{'='*72}", file=file)
        print(f"BENCHMARK EXECUTION: {env_id} ({snapshot_id})", file=file)
        print(f"{'='*72}", file=file)

    # Load data
    if progress:
        print("\n[1/5] Loading data...", file=file)
    bars, events = load_dataset(snapshot_id)
    benchmark_bars = load_benchmark_bars()
    n_instruments = bars["instrument_id"].n_unique()
    n_sessions = bars["trade_date"].n_unique()
    if progress:
        print(f"  Instruments: {n_instruments}", file=file)
        print(f"  Sessions: {n_sessions}", file=file)
        print(f"  Events: {events.height if events is not None else 0}", file=file)

    # Compute features
    if progress:
        print("\n[2/5] Computing features...", file=file)
    fs001_frame = compute_features_fs001(bars)
    fs003_frame = compute_features_fs003(bars)
    if progress:
        print(f"  FS-001: {fs001_frame.height} rows", file=file)
        print(f"  FS-003: {fs003_frame.height} rows", file=file)

    # Compute labels
    if progress:
        print("\n[3/5] Computing labels...", file=file)

    # Get feature sessions for label computation
    feature_sessions = fs001_frame.select(["instrument_id", "decision_session"]).unique()
    # Label engine expects decision_time column
    feature_sessions = feature_sessions.rename({"decision_session": "decision_time"})

    # LAB-004 (absolute return)
    lab004_frame = None
    if label_type in ("absolute", "both"):
        from orbit.ml.labels import build_phase9_label_snapshot
        from orbit.ml.data import load_instrument_master
        instruments = load_instrument_master()
        ls = build_phase9_label_snapshot(bars, events, instruments, feature_sessions)
        lab004_frame = ls.records
        if progress:
            print(f"  LAB-004: {lab004_frame.height} rows", file=file)

    # LAB-005 (excess return)
    lab005_frame = None
    if label_type in ("excess", "both"):
        excess = compute_excess_return_label(bars, benchmark_bars, horizon=5)
        lab005_frame = excess
        if progress:
            available = lab005_frame["label_available"].sum() if "label_available" in lab005_frame.columns else "N/A"
            print(f"  LAB-005: {lab005_frame.height} rows, available={available}", file=file)

    # Run experiments
    if progress:
        print("\n[4/5] Running experiments...", file=file)

    suite = build_benchmark_suite()
    models = suite["models"]
    feature_sets = [
        {"id": "FS-001", "names": [c for c in fs001_frame.columns if c not in ("instrument_id", "decision_session", "window_end_session")]},
        {"id": "FS-003", "names": [c for c in fs003_frame.columns if c not in ("instrument_id", "decision_session", "window_end_session")]},
    ]
    labels_to_run = []
    if label_type in ("absolute", "both"):
        labels_to_run.append({"id": "LAB-004", "frame": lab004_frame, "col": "outcome_value"})
    if label_type in ("excess", "both"):
        labels_to_run.append({"id": "LAB-005", "frame": lab005_frame, "col": "excess_return"})

    results = []
    total = len(models) * len(feature_sets) * len(labels_to_run)
    done = 0

    for model_cfg in models:
        family = model_cfg["family"]
        params = model_cfg["params"]
        for fs_cfg in feature_sets:
            fs_id = fs_cfg["id"]
            fs_names = fs_cfg["names"]
            fs_frame = fs001_frame if fs_id == "FS-001" else fs003_frame
            for label_cfg in labels_to_run:
                lab_id = label_cfg["id"]
                lab_frame = label_cfg["frame"]
                lab_col = label_cfg["col"]

                done += 1
                if progress:
                    print(f"  [{done}/{total}] {family} + {fs_id} + {lab_id}...", file=file, end="")

                try:
                    # Assemble dataset
                    datasets = assemble_dataset(fs_frame, lab_frame, fs_names, lab_col)

                    # Train and evaluate
                    result = train_and_evaluate(family, params, datasets, fs_names)

                    # Record
                    exp_id = f"EXP-11-{env_id}-{fs_id}-{lab_id}-{family}"
                    result["experiment_id"] = exp_id
                    result["env_id"] = env_id
                    result["snapshot_id"] = snapshot_id
                    result["feature_set_id"] = fs_id
                    result["label_id"] = lab_id
                    results.append(result)

                    if progress:
                        ic = result["metrics"].get("oos_ic")
                        print(f" OOS_IC={ic:.4f}" if ic else " OOS_IC=N/A", file=file)

                except Exception as e:
                    if progress:
                        print(f" ERROR: {str(e)[:60]}", file=file)
                    results.append({
                        "experiment_id": f"EXP-11-{env_id}-{fs_id}-{lab_id}-{family}",
                        "env_id": env_id,
                        "snapshot_id": snapshot_id,
                        "feature_set_id": fs_id,
                        "label_id": lab_id,
                        "family": family,
                        "params": params,
                        "error": str(e),
                    })

    # Summary
    if progress:
        print(f"\n[5/5] Summarizing results...", file=file)
        successful = [r for r in results if "error" not in r]
        failed = [r for r in results if "error" in r]
        print(f"  Successful: {len(successful)}/{total}", file=file)
        print(f"  Failed: {len(failed)}/{total}", file=file)

        # Show IC summary
        ics = [r["metrics"]["oos_ic"] for r in successful if r["metrics"].get("oos_ic") is not None]
        if ics:
            print(f"  OOS IC: mean={np.mean(ics):.4f}, median={np.median(ics):.4f}, "
                  f"min={np.min(ics):.4f}, max={np.max(ics):.4f}", file=file)

    return {
        "env_id": env_id,
        "snapshot_id": snapshot_id,
        "n_instruments": n_instruments,
        "n_sessions": n_sessions,
        "results": results,
        "n_successful": len([r for r in results if "error" not in r]),
        "n_failed": len([r for r in results if "error" in r]),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }


def persist_results(execution_result: dict[str, Any], output_dir: Path | None = None) -> Path:
    """Persist benchmark execution results."""
    if output_dir is None:
        output_dir = REPO_ROOT / "benchmarks"
    output_dir.mkdir(parents=True, exist_ok=True)

    env_id = execution_result["env_id"]
    path = output_dir / f"phase11_2_{env_id}_results.json"
    path.write_text(json.dumps(execution_result, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return path


__all__ = [
    "load_dataset",
    "load_benchmark_bars",
    "compute_features_fs001",
    "compute_features_fs003",
    "run_benchmark_suite",
    "persist_results",
]
