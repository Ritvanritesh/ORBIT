"""Test Phase 12A feature building."""
import sys, time
sys.path.insert(0, ".")

from orbit.ml.phase11_2_benchmark import load_dataset, load_benchmark_bars
from orbit.ml.phase12a_features import build_phase12a_feature_snapshots
from orbit.ml.phase12a_market import compute_market_features
from orbit.ml.phase12a_sector import compute_sector_features, load_sector_mapping
from orbit.ml.phase12a_cross_sectional import compute_cross_sectional_features
from orbit.ml.features import _feature_names_for_ids, FeatureSnapshot
from orbit.ml.data import load_instrument_master
from orbit.ml.phase12a_validation import run_full_validation

print("Loading data...")
bars, events = load_dataset("DS-EXP-050")
benchmark_bars = load_benchmark_bars()
instruments = load_instrument_master()
print(f"  bars: {bars.height} rows, {bars['instrument_id'].n_unique()} instruments")
print(f"  benchmark: {benchmark_bars.height} rows")

print("\nBuilding feature snapshots...")
t0 = time.time()
snapshots = build_phase12a_feature_snapshots(
    bars, benchmark_bars, instruments, data_refs=["DS-EXP-050"]
)
print(f"  Done in {time.time()-t0:.1f}s")
for fs_id, snap in snapshots.items():
    print(f"  {fs_id}: {snap.records.height} rows, {len(snap.feature_refs)} features, cols={snap.records.columns[:5]}...")

print("\nValidation...")
universe_sessions = snapshots["FS-001"].records.select("instrument_id", "decision_session").unique()
market_feat = compute_market_features(benchmark_bars, universe_sessions)
sector_map = load_sector_mapping(instruments)
sector_feat = compute_sector_features(bars, sector_map, universe_sessions)
fs001_names = _feature_names_for_ids(snapshots["FS-001"].feature_refs)
xs_feat = compute_cross_sectional_features(snapshots["FS-001"].records, universe_sessions, fs001_names)

validation = run_full_validation(
    bars, benchmark_bars, instruments, snapshots,
    market_feat, sector_feat, sector_map, xs_feat, "DS-EXP-050",
)
print(f"\nValidation: {'PASS' if validation['all_pass'] else 'FAIL'}")
