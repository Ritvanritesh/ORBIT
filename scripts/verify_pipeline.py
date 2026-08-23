"""Quick pipeline verification."""
import sys
sys.path.insert(0, ".")

import polars as pl
from orbit.ml.phase11_2_benchmark import load_dataset, load_benchmark_bars
from orbit.ml.phase11_1_labels import compute_excess_return_label
from orbit.ml.features import build_feature_frame

# Load small subset
bars, events = load_dataset("DS-EXP-050")
print(f"Loaded: {bars.height} rows, {bars['instrument_id'].n_unique()} instruments")

# Take first 5 instruments only for quick test
inst_ids = bars["instrument_id"].unique().to_list()[:5]
bars_small = bars.filter(pl.col("instrument_id").is_in(inst_ids))
print(f"Small: {bars_small.height} rows")

# Compute features
fs = build_feature_frame(bars_small)
print(f"Features: {fs.height} rows, cols={fs.columns}")

# Compute labels
bench = load_benchmark_bars()
excess = compute_excess_return_label(bars_small, bench, horizon=5)
print(f"Excess labels: {excess.height} rows")

# Test label computation with events
from orbit.ml.labels import build_phase9_label_snapshot
from orbit.ml.data import load_instrument_master
instruments = load_instrument_master()
feature_sessions = fs.select(["instrument_id", "decision_session"]).unique()
feature_sessions = feature_sessions.rename({"decision_session": "decision_time"})
ls = build_phase9_label_snapshot(bars_small, events, instruments, feature_sessions)
print(f"LAB-004 labels: {ls.records.height} rows")

print("\nPipeline verification PASSED!")
