"""Run ENV-1 and ENV-2 benchmarks on DS-000004 (historical 20-symbol baseline)."""

from __future__ import annotations
import sys, time, json
from pathlib import Path
import numpy as np
import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from orbit.ml.phase11_2_benchmark import load_dataset, persist_results
from orbit.ml.features import (
    build_feature_snapshot, build_phase10_all_feature_frame,
    attach_decision_times, _feature_names_for_ids, phase10_set_identity,
    FeatureSnapshot,
)
from orbit.ml.labels import build_phase9_label_snapshot
from orbit.ml.data import load_instrument_master
from phase11_2_run_final import run_experiment

bars, events = load_dataset("DS-000004")
n_inst = bars["instrument_id"].n_unique()
print(f"DS-000004: {n_inst} instruments, {bars['trade_date'].n_unique()} sessions")

fs001_snap = build_feature_snapshot(bars, data_refs=["DS-000004"])
fs001_names = _feature_names_for_ids(fs001_snap.feature_refs)
print(f"FS-001: {fs001_snap.records.height} rows")

all_frame = build_phase10_all_feature_frame(bars)
all_frame = attach_decision_times(all_frame)
identity = phase10_set_identity("FS-003")
fs003_names = _feature_names_for_ids(identity["feature_refs"])
fs003_cols = ["instrument_id", "decision_session", "decision_time", "window_end_session"] + fs003_names
fs003_frame = all_frame.select([c for c in fs003_cols if c in all_frame.columns])
fs003_snap = FeatureSnapshot(
    feature_set_id="FS-003", feature_set_version="v1",
    feature_refs=fs003_names, data_refs=["DS-000004"], records=fs003_frame,
)
print(f"FS-003: {fs003_snap.records.height} rows")

instruments = load_instrument_master()
decisions = fs001_snap.records.select("instrument_id", "decision_time")
lab = build_phase9_label_snapshot(bars, events, instruments, decisions)
print(f"LAB-004: {lab.records.height} rows, {lab.available_count()} available")

LOCKED_MODELS = [
    ("ridge", {"alpha": 1.0}),
    ("lasso", {"alpha": 0.001}),
    ("random_forest", {"max_depth": 3, "n_estimators": 200}),
    ("xgboost", {"learning_rate": 0.1, "max_depth": 3, "n_estimators": 200}),
]

feature_sets = [
    ("FS-001", fs001_snap, fs001_names),
    ("FS-003", fs003_snap, fs003_names),
]

for env_id in ["ENV-1", "ENV-2"]:
    print(f"\n{'='*72}")
    print(f"BENCHMARK: {env_id} (DS-000004)")
    print(f"{'='*72}")

    results = []
    total = len(LOCKED_MODELS) * len(feature_sets)
    for i, (family, params) in enumerate(LOCKED_MODELS):
        for fs_id, fs_snap, fs_names in feature_sets:
            idx = i * 2 + (1 if fs_id == "FS-001" else 2)
            t0 = time.time()
            print(f"  [{idx}/{total}] {family}+{fs_id}+LAB-004...", end="", flush=True)
            result = run_experiment(family, params, fs_snap, lab, fs_names, env_id, fs_id)
            results.append(result)
            dt = time.time() - t0
            ic = result.get("metrics", {}).get("oos_ic")
            if "error" in result:
                print(f" ERROR ({dt:.1f}s): {result['error'][:60]}")
            else:
                print(f" IC={ic:.4f} ({dt:.1f}s)" if ic else f" IC=N/A ({dt:.1f}s)")

    all_ics = [r["metrics"]["oos_ic"] for r in results if r.get("metrics", {}).get("oos_ic") is not None]
    print(f"\n{env_id}: {len(results)} experiments")
    print(f"OOS IC: mean={np.mean(all_ics):.4f}, median={np.median(all_ics):.4f}, min={np.min(all_ics):.4f}, max={np.max(all_ics):.4f}")

    res = {
        "env_id": env_id, "snapshot_id": "DS-000004",
        "n_instruments": n_inst, "n_sessions": int(bars["trade_date"].n_unique()),
        "results": results, "n_successful": len([r for r in results if "error" not in r]),
        "n_failed": len([r for r in results if "error" in r]), "elapsed_seconds": 0,
    }
    persist_results(res)
    with open(f"benchmarks/phase11_2_{env_id}_results.json", "w") as f:
        json.dump(res, f, indent=2, default=str)
    print(f"{env_id} saved")
