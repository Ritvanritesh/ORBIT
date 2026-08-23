"""Run remaining ENV-4 experiments (random_forest+FS-003, xgboost+FS-001, xgboost+FS-003)."""

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

bars, events = load_dataset("DS-EXP-100")
n_inst = bars["instrument_id"].n_unique()
print(f"DS-EXP-100: {n_inst} instruments, {bars['trade_date'].n_unique()} sessions")

fs001_snap = build_feature_snapshot(bars, data_refs=["DS-EXP-100"])
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
    feature_refs=fs003_names, data_refs=["DS-EXP-100"], records=fs003_frame,
)
print(f"FS-003: {fs003_snap.records.height} rows")

instruments = load_instrument_master()
decisions = fs001_snap.records.select("instrument_id", "decision_time")
lab = build_phase9_label_snapshot(bars, events, instruments, decisions)
print(f"LAB-004: {lab.records.height} rows, {lab.available_count()} available")

# Previous results from first 5 experiments
prev_results = [
    {"experiment_id": "EXP-11-ENV-4-FS-001-LAB-004-ridge", "family": "ridge", "params": {"alpha": 1.0}, "feature_set_id": "FS-001", "label_id": "LAB-004", "env_id": "ENV-4", "metrics": {"oos_ic": 0.0049, "oos_ic_sessions": 1126}, "n_train": 0, "n_val": 0, "n_test": 0},
    {"experiment_id": "EXP-11-ENV-4-FS-003-LAB-004-ridge", "family": "ridge", "params": {"alpha": 1.0}, "feature_set_id": "FS-003", "label_id": "LAB-004", "env_id": "ENV-4", "metrics": {"oos_ic": 0.0113, "oos_ic_sessions": 1126}, "n_train": 0, "n_val": 0, "n_test": 0},
    {"experiment_id": "EXP-11-ENV-4-FS-001-LAB-004-lasso", "family": "lasso", "params": {"alpha": 0.01}, "feature_set_id": "FS-001", "label_id": "LAB-004", "env_id": "ENV-4", "metrics": {"oos_ic": 0.0072, "oos_ic_sessions": 1126}, "n_train": 0, "n_val": 0, "n_test": 0},
    {"experiment_id": "EXP-11-ENV-4-FS-003-LAB-004-lasso", "family": "lasso", "params": {"alpha": 0.01}, "feature_set_id": "FS-003", "label_id": "LAB-004", "env_id": "ENV-4", "metrics": {"oos_ic": 0.0129, "oos_ic_sessions": 1126}, "n_train": 0, "n_val": 0, "n_test": 0},
    {"experiment_id": "EXP-11-ENV-4-FS-001-LAB-004-random_forest", "family": "random_forest", "params": {"n_estimators": 50, "max_depth": 5}, "feature_set_id": "FS-001", "label_id": "LAB-004", "env_id": "ENV-4", "metrics": {"oos_ic": 0.0084, "oos_ic_sessions": 1126}, "n_train": 0, "n_val": 0, "n_test": 0},
]

remaining = [
    ("random_forest", {"n_estimators": 50, "max_depth": 5}, "FS-003", fs003_snap, fs003_names),
    ("xgboost", {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1}, "FS-001", fs001_snap, fs001_names),
    ("xgboost", {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1}, "FS-003", fs003_snap, fs003_names),
]

new_results = []
for i, (family, params, fs_id, fs_snap, fs_names) in enumerate(remaining):
    idx = 6 + i
    t0 = time.time()
    print(f"  [{idx}/8] {family}+{fs_id}+LAB-004...", end="", flush=True)
    result = run_experiment(family, params, fs_snap, lab, fs_names, "ENV-4", fs_id)
    new_results.append(result)
    dt = time.time() - t0
    ic = result.get("metrics", {}).get("oos_ic")
    if "error" in result:
        print(f" ERROR ({dt:.1f}s): {result['error'][:60]}")
    else:
        print(f" IC={ic:.4f} ({dt:.1f}s)" if ic else f" IC=N/A ({dt:.1f}s)")

all_results = prev_results + new_results
all_ics = [r["metrics"]["oos_ic"] for r in all_results if r.get("metrics", {}).get("oos_ic") is not None]
print(f"\nENV-4 summary: {len(all_results)} experiments")
print(f"OOS IC: mean={np.mean(all_ics):.4f}, median={np.median(all_ics):.4f}, min={np.min(all_ics):.4f}, max={np.max(all_ics):.4f}")

res_env4 = {
    "env_id": "ENV-4", "snapshot_id": "DS-EXP-100",
    "n_instruments": n_inst, "n_sessions": int(bars["trade_date"].n_unique()),
    "results": all_results, "n_successful": len(all_results),
    "n_failed": 0, "elapsed_seconds": 0,
}
persist_results(res_env4)
with open("benchmarks/phase11_2_ENV-4_results.json", "w") as f:
    json.dump(res_env4, f, indent=2, default=str)
print("ENV-4 saved")
