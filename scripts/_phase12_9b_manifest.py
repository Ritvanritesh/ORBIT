"""Phase 12.9B - Environment manifest and data preparation."""
from __future__ import annotations
import hashlib, json, os, platform, sys, time
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl

REPO = Path(r"E:\Orbit (Optimized Research & Behavioral Intelligence Trading)")
BENCH = REPO / "benchmarks"
DATA = REPO / "data"
OUT = BENCH

sys.path.insert(0, str(REPO / "src"))

def sha256_file(path: Path) -> str:
    if not path.exists(): return "MISSING"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def save_json(name, data):
    with open(OUT / name, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str, ensure_ascii=False)
    print(f"  Saved: {name}")

# Environment
print("=" * 72)
print("PHASE 12.9B - ENVIRONMENT & MANIFEST")
print("=" * 72)

env = {
    "python": sys.version,
    "platform": platform.platform(),
    "machine": platform.machine(),
    "processor": platform.processor(),
    "numpy": np.__version__,
    "polars": pl.__version__,
    "timestamp": datetime.now().isoformat(),
}
try:
    import sklearn; env["sklearn"] = sklearn.__version__
except: pass
try:
    import xgboost; env["xgboost"] = xgboost.__version__
except: pass
try:
    import scipy; env["scipy"] = scipy.__version__
except: pass
print(f"  Python: {env['python']}")
print(f"  Platform: {env['platform']}")

# Dataset hashes
ds_hashes = {}
for ds_id in ["DS-000004", "DS-EXP-050", "DS-EXP-100"]:
    bp = DATA / "normalized" / "market" / "yahoo_chart_api" / ds_id / "bars.parquet"
    ds_hashes[ds_id] = sha256_file(bp)
    print(f"  {ds_id} hash: {ds_hashes[ds_id][:16]}...")

# Benchmark hash
bench_hash = sha256_file(DATA / "normalized" / "benchmark" / "BENCH-001" / "bars.parquet")
print(f"  BENCH-001 hash: {bench_hash[:16]}...")

# Plan digests
plan_digests = {}
for pf in sorted(BENCH.glob("*plan*.json")):
    with open(pf) as f:
        content = f.read()
    plan_digests[pf.name] = hashlib.sha256(content.encode()).hexdigest()[:16]
print(f"  Plan files: {len(plan_digests)}")

save_json("phase12_9b_environment.json", {
    "env": env, "dataset_hashes": ds_hashes,
    "benchmark_hash": bench_hash, "plan_digests": plan_digests
})

# Prepare data: build features and labels for all datasets
print("\n  Preparing clean intermediate artifacts...")
from orbit.ml.phase11_2_benchmark import load_dataset
from orbit.ml.features import build_feature_snapshot
from orbit.ml.labels import build_phase9_label_snapshot
from orbit.ml.data import load_instrument_master

instruments = load_instrument_master()
clean_data = {}

for ds_id in ["DS-000004", "DS-EXP-050", "DS-EXP-100"]:
    print(f"  Building {ds_id}...")
    t0 = time.time()
    bars, events = load_dataset(ds_id)
    fs = build_feature_snapshot(bars, data_refs=[ds_id])
    decisions = fs.records.select("instrument_id", "decision_time")
    lab = build_phase9_label_snapshot(bars, events, instruments, decisions, data_refs=[ds_id])
    elapsed = time.time() - t0
    clean_data[ds_id] = {"bars": bars, "events": events, "fs": fs, "lab": lab}
    print(f"    {ds_id}: {fs.records.height} features, {lab.records.height} labels ({elapsed:.1f}s)")

save_json("phase12_9b_manifest.json", {
    "datasets_prepared": list(clean_data.keys()),
    "feature_rows": {k: v["fs"].records.height for k, v in clean_data.items()},
    "label_rows": {k: v["lab"].records.height for k, v in clean_data.items()},
    "preparation_time": datetime.now().isoformat(),
})
print("  Manifest saved.")
