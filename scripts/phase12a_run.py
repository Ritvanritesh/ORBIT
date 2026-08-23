"""Phase 12A benchmark execution runner."""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path
from typing import Any
import numpy as np
import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from orbit.ml.phase11_2_benchmark import load_dataset, load_benchmark_bars, persist_results
from orbit.ml.phase12a_plan import build_phase12a_plan, persist_phase12a_plan, PHASE12A_FEATURE_SETS, PHASE12A_FEATURE_NAMES
from orbit.ml.phase12a_features import build_phase12a_feature_snapshots
from orbit.ml.phase12a_validation import run_full_validation
from orbit.ml.features import _feature_names_for_ids, FeatureSnapshot
from orbit.ml.labels import build_phase9_label_snapshot
from orbit.ml.data import load_instrument_master
from orbit.ml.dataset import assemble_datasets


def run_single_experiment(family, params, fs_snapshot, lab_snapshot, feature_names, env_id, fs_id, lab_id):
    from orbit.ml.models import train_model, predict_with_state
    from orbit.ml.metrics import oos_ic, rank_ic, hit_rate, mean_squared_error
    from orbit.ml.calibration import fit_platt

    exp_id = f"EXP-12A-{env_id}-{fs_id}-{lab_id}-{family}"
    try:
        datasets = assemble_datasets(fs_snapshot, lab_snapshot, feature_names=feature_names)
        X_train, y_train_reg, y_train_bin, meta_train = datasets["train"]
        X_val, y_val_reg, y_val_bin, meta_val = datasets["val"]
        X_test, y_test_reg, y_test_bin, meta_test = datasets["test"]

        if len(X_train) == 0 or len(X_test) == 0:
            return {"experiment_id": exp_id, "error": "insufficient data"}

        model, state = train_model(family, params, X_train, y_train_reg, feature_names=feature_names)
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
            "experiment_id": exp_id, "family": family, "params": params,
            "feature_set_id": fs_id, "label_id": lab_id, "env_id": env_id,
            "metrics": metrics, "n_train": len(X_train), "n_val": len(X_val), "n_test": len(X_test),
        }
    except Exception as e:
        return {"experiment_id": exp_id, "error": str(e)}


def run_phase12a_benchmark(snapshot_id, env_id, plan):
    print(f"\n{'='*72}")
    print(f"PHASE 12A BENCHMARK: {env_id} ({snapshot_id})")
    print(f"{'='*72}")

    bars, events = load_dataset(snapshot_id)
    benchmark_bars = load_benchmark_bars()
    instruments = load_instrument_master()
    n_inst = bars["instrument_id"].n_unique()
    print(f"  {n_inst} instruments, {bars['trade_date'].n_unique()} sessions")

    print("\n[1/5] Building feature snapshots...")
    t0 = time.time()
    snapshots = build_phase12a_feature_snapshots(
        bars, benchmark_bars, instruments, data_refs=[snapshot_id]
    )
    print(f"  Built {len(snapshots)} feature sets in {time.time()-t0:.1f}s")
    for fs_id, snap in snapshots.items():
        print(f"    {fs_id}: {snap.records.height} rows, {len(snap.feature_refs)} features")

    print("\n[2/5] Computing LAB-004 labels...")
    t0 = time.time()
    fs001 = snapshots["FS-001"]
    decisions = fs001.records.select("instrument_id", "decision_time")
    lab_snapshot = build_phase9_label_snapshot(bars, events, instruments, decisions)
    print(f"  LAB-004: {lab_snapshot.records.height} rows ({time.time()-t0:.1f}s)")

    print("\n[3/5] Running validation...")
    from orbit.ml.phase12a_market import compute_market_features
    from orbit.ml.phase12a_sector import compute_sector_features, load_sector_mapping
    from orbit.ml.phase12a_cross_sectional import compute_cross_sectional_features
    universe_sessions = fs001.records.select("instrument_id", "decision_session").unique()
    market_feat = compute_market_features(benchmark_bars, universe_sessions)
    sector_map = load_sector_mapping(instruments)
    sector_feat = compute_sector_features(bars, sector_map, universe_sessions)
    fs001_names = _feature_names_for_ids(fs001.feature_refs)
    xs_feat = compute_cross_sectional_features(fs001.records, universe_sessions, fs001_names)

    validation = run_full_validation(
        bars, benchmark_bars, instruments, snapshots,
        market_feat, sector_feat, sector_map, xs_feat, snapshot_id,
    )
    if not validation["all_pass"]:
        print("\n  VALIDATION FAILED - ABORTING ML EXECUTION")
        return None

    print("\n[4/5] Running experiments...")
    LOCKED_MODELS = [
        ("ridge", {"alpha": 1.0}),
        ("lasso", {"alpha": 0.001}),
        ("random_forest", {"max_depth": 3, "n_estimators": 200}),
        ("xgboost", {"learning_rate": 0.1, "max_depth": 3, "n_estimators": 200}),
    ]

    results = []
    feature_sets = list(PHASE12A_FEATURE_SETS.keys())
    total = len(LOCKED_MODELS) * len(feature_sets)
    done = 0
    t_start = time.time()

    for family, params in LOCKED_MODELS:
        for fs_id in feature_sets:
            done += 1
            fs_snap = snapshots[fs_id]
            fs_names = [PHASE12A_FEATURE_NAMES.get(f, f) for f in fs_snap.feature_refs]
            t0 = time.time()
            print(f"  [{done}/{total}] {family}+{fs_id}+LAB-004...", end="", flush=True)
            result = run_single_experiment(family, params, fs_snap, lab_snapshot, fs_names, env_id, fs_id, "LAB-004")
            results.append(result)
            dt = time.time() - t0
            ic = result.get("metrics", {}).get("oos_ic")
            if "error" in result:
                print(f" ERROR ({dt:.1f}s): {result['error'][:60]}")
            else:
                print(f" IC={ic:.4f} ({dt:.1f}s)" if ic else f" IC=N/A ({dt:.1f}s)")

    elapsed = time.time() - t_start
    successful = [r for r in results if "error" not in r]
    failed = [r for r in results if "error" in r]
    ics = [r["metrics"]["oos_ic"] for r in successful if r["metrics"].get("oos_ic") is not None]

    print(f"\n  Completed: {len(successful)}/{total} in {elapsed:.1f}s")
    if ics:
        print(f"  OOS IC: mean={np.mean(ics):.4f}, median={np.median(ics):.4f}, min={np.min(ics):.4f}, max={np.max(ics):.4f}")

    res = {
        "env_id": env_id, "snapshot_id": snapshot_id,
        "n_instruments": n_inst, "n_sessions": int(bars["trade_date"].n_unique()),
        "results": results, "n_successful": len(successful),
        "n_failed": len(failed), "elapsed_seconds": elapsed,
        "validation": validation,
    }

    print("\n[5/5] Persisting results...")
    persist_results(res)
    out_path = REPO_ROOT / "benchmarks" / f"phase12a_{env_id}_results.json"
    out_path.write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    print(f"  Saved to {out_path.name}")

    return res


def main():
    print("=" * 72)
    print("PHASE 12A - BENCHMARK EXECUTION")
    print("=" * 72)

    plan = build_phase12a_plan()
    plan_path = persist_phase12a_plan(plan)
    print(f"Plan locked: {plan_path}")
    print(f"Plan digest: {plan['plan_digest'][:16]}...")
    print(f"Experiments: {plan['n_experiments']}")
    print(f"Feature sets: {list(PHASE12A_FEATURE_SETS.keys())}")

    results = {}
    for env_id, env_cfg in plan["environments"].items():
        r = run_phase12a_benchmark(env_cfg["dataset_id"], env_id, plan)
        if r:
            results[env_id] = r

    print("\n" + "=" * 72)
    print("PHASE 12A SUMMARY")
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
