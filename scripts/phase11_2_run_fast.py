"""Phase 11.2 optimized benchmark execution.

Runs experiments efficiently with progress tracking.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from orbit.ml.phase11_2_benchmark import (
    load_dataset,
    load_benchmark_bars,
    compute_features_fs001,
    compute_features_fs003,
    persist_results,
)
from orbit.ml.phase11_1_labels import compute_excess_return_label
from orbit.ml.phase11_1_plan import build_benchmark_suite


def run_single_experiment(
    family: str,
    params: dict[str, Any],
    fs_frame: pl.DataFrame,
    lab_frame: pl.DataFrame,
    lab_col: str,
    fs_names: list[str],
    env_id: str,
    fs_id: str,
    lab_id: str,
) -> dict[str, Any]:
    """Run a single experiment."""
    from orbit.ml.models import train_model, predict_with_state
    from orbit.ml.metrics import oos_ic, rank_ic, hit_rate, mean_squared_error
    from orbit.ml.calibration import fit_platt, expected_calibration_error, brier_score
    from orbit.ml.dataset import assemble_datasets
    from orbit.ml.features import FeatureSnapshot
    from orbit.ml.labels import LabelSnapshot

    exp_id = f"EXP-11-{env_id}-{fs_id}-{lab_id}-{family}"

    try:
        # Create snapshots with required fields
        from orbit.ml.features import FeatureSnapshot as FS
        from orbit.ml.labels import LabelSnapshot as LS

        fsnap = FS(
            feature_set_id=fs_id,
            feature_set_version="v1",
            feature_refs=fs_names,
            data_refs=["phase11_2"],
            records=fs_frame,
        )
        lsnap = LS(
            label_id=lab_id,
            version="v1",
            contract_digest="phase11_2",
            engine_version="v1",
            data_refs=["phase11_2"],
            records=lab_frame,
        )

        # Assemble datasets
        datasets = assemble_datasets(fsnap, lsnap, feature_names=fs_names)

        X_train, y_train_reg, y_train_bin, meta_train = datasets["train"]
        X_val, y_val_reg, y_val_bin, meta_val = datasets["val"]
        X_test, y_test_reg, y_test_bin, meta_test = datasets["test"]

        if len(X_train) == 0 or len(X_test) == 0:
            return {"experiment_id": exp_id, "error": "insufficient data"}

        # Train
        model, state = train_model(family, params, X_train, y_train_reg,
                                    feature_names=fs_names, seed=42)

        # Predict
        pred_val = predict_with_state(model, state, X_val)
        pred_test = predict_with_state(model, state, X_test)

        # Calibrate
        calibrator = fit_platt(pred_val, y_val_bin)
        pred_test_cal = calibrator.apply(pred_test)

        # Metrics
        test_frame = meta_test.with_columns([
            pl.Series("prediction", pred_test_cal),
            pl.Series("outcome_value", y_test_reg),
        ])

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

        return {
            "experiment_id": exp_id,
            "family": family,
            "params": params,
            "feature_set_id": fs_id,
            "label_id": lab_id,
            "env_id": env_id,
            "metrics": metrics,
            "n_train": len(X_train),
            "n_val": len(X_val),
            "n_test": len(X_test),
        }

    except Exception as e:
        return {"experiment_id": exp_id, "error": str(e)}


def run_benchmark_fast(
    snapshot_id: str,
    env_id: str,
    label_type: str = "both",
) -> dict[str, Any]:
    """Run benchmark suite efficiently."""
    print(f"\n{'='*72}")
    print(f"BENCHMARK: {env_id} ({snapshot_id})")
    print(f"{'='*72}")

    # Load data
    print("\n[1/4] Loading data...")
    bars, events = load_dataset(snapshot_id)
    benchmark_bars = load_benchmark_bars()
    n_inst = bars["instrument_id"].n_unique()
    n_sess = bars["trade_date"].n_unique()
    print(f"  {n_inst} instruments, {n_sess} sessions")

    # Compute features
    print("\n[2/4] Computing features...")
    t0 = time.time()
    fs001 = compute_features_fs001(bars)
    fs003 = compute_features_fs003(bars)
    print(f"  FS-001: {fs001.height} rows ({time.time()-t0:.1f}s)")

    fs001_names = [c for c in fs001.columns if c not in ("instrument_id", "decision_session", "window_end_session")]
    fs003_names = [c for c in fs003.columns if c not in ("instrument_id", "decision_session", "window_end_session")]

    # Compute labels
    print("\n[3/4] Computing labels...")
    t0 = time.time()

    # LAB-005 (excess return)
    lab005 = None
    if label_type in ("excess", "both"):
        lab005 = compute_excess_return_label(bars, benchmark_bars, horizon=5)
        print(f"  LAB-005: {lab005.height} rows ({time.time()-t0:.1f}s)")

    # LAB-004 (absolute return)
    lab004 = None
    if label_type in ("absolute", "both"):
        from orbit.ml.labels import build_phase9_label_snapshot
        from orbit.ml.data import load_instrument_master
        instruments = load_instrument_master()
        feature_sessions = fs001.select(["instrument_id", "decision_session"]).unique()
        feature_sessions = feature_sessions.rename({"decision_session": "decision_time"})
        t1 = time.time()
        ls = build_phase9_label_snapshot(bars, events, instruments, feature_sessions)
        lab004 = ls.records
        print(f"  LAB-004: {lab004.height} rows ({time.time()-t1:.1f}s)")

    # Run experiments
    print("\n[4/4] Running experiments...")
    suite = build_benchmark_suite()
    models = suite["models"]

    feature_sets = [
        ("FS-001", fs001, fs001_names),
        ("FS-003", fs003, fs003_names),
    ]

    labels = []
    if label_type in ("absolute", "both") and lab004 is not None:
        labels.append(("LAB-004", lab004, "outcome_value"))
    if label_type in ("excess", "both") and lab005 is not None:
        labels.append(("LAB-005", lab005, "excess_return"))

    results = []
    total = len(models) * len(feature_sets) * len(labels)
    done = 0
    t_start = time.time()

    for model_cfg in models:
        family = model_cfg["family"]
        params = model_cfg["params"]
        for fs_id, fs_frame, fs_names in feature_sets:
            for lab_id, lab_frame, lab_col in labels:
                done += 1
                t0 = time.time()
                print(f"  [{done}/{total}] {family}+{fs_id}+{lab_id}...", end="", flush=True)

                result = run_single_experiment(
                    family, params, fs_frame, lab_frame, lab_col,
                    fs_names, env_id, fs_id, lab_id,
                )
                results.append(result)

                dt = time.time() - t0
                if "error" in result:
                    print(f" ERROR ({dt:.1f}s)")
                else:
                    ic = result["metrics"].get("oos_ic")
                    print(f" IC={ic:.4f} ({dt:.1f}s)" if ic else f" IC=N/A ({dt:.1f}s)")

    elapsed = time.time() - t_start
    successful = [r for r in results if "error" not in r]
    failed = [r for r in results if "error" in r]

    print(f"\n  Completed: {len(successful)}/{total} successful in {elapsed:.1f}s")

    # IC summary
    for lab_id in ["LAB-004", "LAB-005"]:
        ics = [r["metrics"]["oos_ic"] for r in successful
               if r.get("label_id") == lab_id and r["metrics"].get("oos_ic") is not None]
        if ics:
            print(f"  {lab_id} IC: mean={np.mean(ics):.4f}, median={np.median(ics):.4f}, "
                  f"min={np.min(ics):.4f}, max={np.max(ics):.4f}")

    return {
        "env_id": env_id,
        "snapshot_id": snapshot_id,
        "n_instruments": n_inst,
        "n_sessions": n_sess,
        "results": results,
        "n_successful": len(successful),
        "n_failed": len(failed),
        "elapsed_seconds": elapsed,
    }


def main():
    print("=" * 72)
    print("PHASE 11.2 - BENCHMARK EXECUTION")
    print("=" * 72)

    results = {}

    # ENV-3: 50-symbol
    r3 = run_benchmark_fast("DS-EXP-050", "ENV-3", "both")
    persist_results(r3)
    results["ENV-3"] = r3

    # ENV-4: 100-symbol
    r4 = run_benchmark_fast("DS-EXP-100", "ENV-4", "both")
    persist_results(r4)
    results["ENV-4"] = r4

    # Summary
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    for env_id, res in results.items():
        print(f"\n{env_id}: {res['n_successful']}/{res['n_successful']+res['n_failed']} experiments, "
              f"{res['elapsed_seconds']:.1f}s")

    return results


if __name__ == "__main__":
    main()
