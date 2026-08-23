"""Quick test of Phase 12A modules."""
import sys
sys.path.insert(0, ".")

from orbit.ml.phase12a_plan import build_phase12a_plan, PHASE12A_FEATURE_SETS
plan = build_phase12a_plan()
print(f"Plan built: {plan['n_experiments']} experiments")
print(f"Feature sets: {list(PHASE12A_FEATURE_SETS.keys())}")
print(f"Plan digest: {plan['plan_digest'][:16]}...")

from orbit.ml.phase12a_market import compute_market_features
from orbit.ml.phase12a_sector import compute_sector_features, load_sector_mapping
from orbit.ml.phase12a_cross_sectional import compute_cross_sectional_features
from orbit.ml.phase12a_validation import run_full_validation
from orbit.ml.phase12a_features import build_phase12a_feature_snapshots
print("All modules imported successfully")
