"""Phase 11.2 benchmark execution - uses proper snapshot builders."""

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

from orbit.ml.phase11_2_benchmark import load_dataset, persist_results
from orbit.ml.features import (
    build_feature_snapshot,
    build_phase10_all_feature_frame,
    attach_decision_times,
    _feature_names_for_ids,
)
from orbit.ml.labels import build_phase9_label_snapshot
from orbit.ml.data import load_instrument_master


def run_experiment(
    family: str,
    params: dict[str, Any],
    fs_snapshot: Any,
    lab_snapshot: Any,
    feature_names: list[str],
    env_id: str,
    fs_id: str,
) -> dict[str, Any]:
    """Run a single experiment."""
    from orbit.ml.models import train_model, predict_with_state
    from orbit.ml.metrics import oos_ic, rank_ic, hit_rate, mean_squared_error
    from orbit.ml.calibration import fit_platt
    from orbit.ml.dataset import assemble_datasets

    exp_id = f"EXP-11-{env_id}-{fs_id}-LAB-004-{family}"

    try:
        datasets = assemble_datasets(fs_snapshot, lab_snapshot, feature_names=feature_names)

        X_train, y_train_reg, y_train_bin, meta_train = datasets["train"]
        X_val, y_val_reg, y_val_bin, meta_val = datasets["val"]
        X_test, y_test_reg, y_test_bin, meta_test = datasets["test"]

        if len(X_train) == 0 or len(X_test) == 0:
            return {"experiment_id": exp_id, "error": "insufficient data"}

        model, state = train_model(family, params, X_train, y_train_reg,
                                    feature_names=feature_names)

        pred_val = predict_with_state(model, state, X_val)
        pred_test = predict_with_state(model, state, X_test)

        calibrator = fit_platt(pred_val, y_val_bin)
        pred_test_cal = calibrator.apply(pred_test)

        test_frame = meta_test.with_columns([
            pl.Series("prediction", pred_test_cal),
            pl.Series("outcome_value", y_test_reg),
        ])

        metrics = {}
        try:
            ic_result = oos_ic(test_frame, "prediction", "outcome_value")
            metrics["oos_ic"] = ic_result.get("value") if isinstance(ic_result, dict) else float(ic_result)
            metrics["oos_ic_sessions"] = ic_result.get("sessions_used", 0) if isinstance(ic_result, dict) else 0
        except Exception:
            metrics["oos_ic"] = None
            metrics["oos_ic_sessions"] = 0
        try:
            ric_result = rank_ic(test_frame, "prediction", "outcome_value")
            metrics["rank_ic"] = ric_result.get("value") if isinstance(ric_result, dict) else float(ric_result)
        except Exception:
            metrics["rank_ic"] = None
        try:
            hr_result = hit_rate(test_frame, "prediction", "outcome_value")
            metrics["hit_rate"] = float(hr_result) if not isinstance(hr_result, dict) else hr_result.get("value", None)
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
            "label_id": "LAB-004",
            "env_id": env_id,
            "metrics": metrics,
            "n_train": len(X_train),
            "n_val": len(X_val),
            "n_test": len(X_test),
        }

    except Exception as e:
        return {"experiment_id": exp_id, "error": str(e)}


def run_benchmark(snapshot_id: str, env_id: str) -> dict[str, Any]:
    """Run benchmark suite on one environment."""
    print(f"\n{'='*72}")
    print(f"BENCHMARK: {env_id} ({snapshot_id})")
    print(f"{'='*72}")

    # Load data
    print("\n[1/4] Loading data...")
    bars, events = load_dataset(snapshot_id)
    n_inst = bars["instrument_id"].n_unique()
    n_sess = bars["trade_date"].n_unique()
    print(f"  {n_inst} instruments, {n_sess} sessions")

    # Build feature snapshots (with decision_time)
    print("\n[2/4] Computing features...")
    t0 = time.time()

    # FS-001
    fs001_snap = build_feature_snapshot(bars, data_refs=[snapshot_id])
    fs001_names = _feature_names_for_ids(fs001_snap.feature_refs)
    print(f"  FS-001: {fs001_snap.records.height} rows ({time.time()-t0:.1f}s)")

    # FS-003
    t1 = time.time()
    all_frame = build_phase10_all_feature_frame(bars)
    all_frame = attach_decision_times(all_frame)
    from orbit.ml.features import phase10_set_identity, assert_features_point_in_time, assert_features_finite, FEATURE_NAMES_PHASE10
    identity = phase10_set_identity("FS-003")
    fs003_ids = identity["feature_refs"]
    fs003_names = _feature_names_for_ids(fs003_ids)
    # Project to FS-003 columns
    fs003_cols = ["instrument_id", "decision_session", "decision_time", "window_end_session"] + fs003_names
    fs003_frame = all_frame.select([c for c in fs003_cols if c in all_frame.columns])
    fs003_snap = __import__("orbit.ml.features", fromlist=["FeatureSnapshot"]).FeatureSnapshot(
        feature_set_id="FS-003",
        feature_set_version="v1",
        feature_refs=fs003_names,
        data_refs=[snapshot_id],
        records=fs003_frame,
    )
    print(f"  FS-003: {fs003_snap.records.height} rows ({time.time()-t1:.1f}s)")

    # Build LAB-004 label snapshot
    print("\n[3/4] Computing LAB-004 labels...")
    t0 = time.time()
    instruments = load_instrument_master()
    decisions = fs001_snap.records.select("instrument_id", "decision_time")
    lab_snapshot = build_phase9_label_snapshot(bars, events, instruments, decisions)
    print(f"  LAB-004: {lab_snapshot.records.height} rows, {lab_snapshot.available_count()} available ({time.time()-t0:.1f}s)")

    # Run experiments
    print("\n[4/4] Running experiments...")
    from orbit.ml.phase11_1_plan import build_benchmark_suite
    suite = build_benchmark_suite()
    models = suite["models"]

    feature_sets = [
        ("FS-001", fs001_snap, fs001_names),
        ("FS-003", fs003_snap, fs003_names),
    ]

    results = []
    total = len(models) * len(feature_sets)
    done = 0
    t_start = time.time()

    for model_cfg in models:
        family = model_cfg["family"]
        params = model_cfg["params"]
        for fs_id, fs_snap, fs_names in feature_sets:
            done += 1
            t0 = time.time()
            print(f"  [{done}/{total}] {family}+{fs_id}+LAB-004...", end="", flush=True)

            result = run_experiment(
                family, params, fs_snap, lab_snapshot,
                fs_names, env_id, fs_id,
            )
            results.append(result)

            dt = time.time() - t0
            if "error" in result:
                print(f" ERROR ({dt:.1f}s): {result['error'][:60]}")
            else:
                ic = result["metrics"].get("oos_ic")
                print(f" IC={ic:.4f} ({dt:.1f}s)" if ic else f" IC=N/A ({dt:.1f}s)")

    elapsed = time.time() - t_start
    successful = [r for r in results if "error" not in r]
    failed = [r for r in results if "error" in r]

    print(f"\n  Completed: {len(successful)}/{total} successful in {elapsed:.1f}s")

    ics = [r["metrics"]["oos_ic"] for r in successful if r["metrics"].get("oos_ic") is not None]
    if ics:
        print(f"  OOS IC: mean={np.mean(ics):.4f}, median={np.median(ics):.4f}, "
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

    r3 = run_benchmark("DS-EXP-050", "ENV-3")
    persist_results(r3)
    results["ENV-3"] = r3

    r4 = run_benchmark("DS-EXP-100", "ENV-4")
    persist_results(r4)
    results["ENV-4"] = r4

    # Summary
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    for env_id, res in results.items():
        ics = [r["metrics"]["oos_ic"] for r in res["results"]
               if "error" not in r and r["metrics"].get("oos_ic") is not None]
        print(f"\n{env_id}: {res['n_successful']}/{res['n_successful']+res['n_failed']} experiments")
        if ics:
            print(f"  OOS IC: mean={np.mean(ics):.4f}, median={np.median(ics):.4f}")

    return results


if __name__ == "__main__":
    main()
