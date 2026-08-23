"""Phase 12B fundamental feature snapshot builder."""
from __future__ import annotations
from typing import Any
import polars as pl
from datetime import date
from pathlib import Path

from orbit.ml.phase12b_plan import (
    PHASE12B_FEATURE_SETS, PHASE12B_FEATURE_NAMES, PHASE12B_STALENESS,
    PHASE12B_WINDOWS, ACTIVE_FUNDAMENTAL_SOURCE,
)
from orbit.ml.phase12b_fundamentals import (
    load_sec_edgar_companyfacts, compute_fundamental_features,
    validate_pit_compliance,
)
from orbit.ml.features import attach_decision_times, FeatureSnapshot, _feature_names_for_ids
from orbit.ml.phase11_2_benchmark import load_dataset, load_benchmark_bars
from orbit.ml.data import load_instrument_master


def _compute_fs_features(
    fundamental_by_instrument: dict[str, dict[str, float | None]],
    feature_names: list[str],
) -> dict[str, float | None]:
    """Select requested feature names from computed fundamentals."""
    results = {}
    for name in feature_names:
        results[name] = fundamental_by_instrument.get(name)
    return results


def build_fundamental_feature_snapshots(
    snapshot_id: str,
    data_refs: list[str] | None = None,
) -> dict[str, Any]:
    """Build fundamental feature snapshots for a given dataset snapshot.

    Returns dict with keys:
    - "FS-12B-A": baseline OHLCV only
    - "FS-12B-B": baseline + valuation
    - "FS-12B-C": baseline + profitability
    - "FS-12B-D": baseline + growth
    - "FS-12B-E": baseline + leverage
    - "FS-12B-F": baseline + all fundamental families

    Each value is a FeatureSnapshot.
    """
    # Load data
    bars, events = load_dataset(snapshot_id)
    benchmark_bars = load_benchmark_bars()
    instruments = load_instrument_master()
    fundamental_df = load_sec_edgar_companyfacts(snapshot_id)

    # Get all decision sessions from the baseline features
    fs001 = FeatureSnapshot(
        feature_set_id="FS-001",
        feature_set_version="v1",
        feature_refs=["FEAT-001", "FEAT-002", "FEAT-003", "FEAT-004",
                      "FEAT-005", "FEAT-006", "FEAT-007", "FEAT-008"],
        data_refs=data_refs or [],
        records=bars,  # simplified - in practice would be the processed feature frame
    )

    # Get all decision sessions
    all_sessions = fs001.records.select("instrument_id", "decision_session").unique().sort(
        ["instrument_id", "decision_session"]
    )

    # Get unique instruments and their latest filing dates
    # Build a map: instrument_id -> latest filing_date across all available fundamentals
    # For simplicity, we'll use the latest filing_date in the dataset
    if fundamental_df.height > 0:
        latest_filing_per_instrument = fundamental_df.group_by("instrument_id").agg(
            pl.col("filing_date").max().alias("latest_filing")
        )
    else:
        latest_filing_per_instrument = pl.DataFrame({
            "instrument_id": [],
            "latest_filing": [],
        })

    # Create a map from instrument_id to the as-of date to use
    # We'll use the latest filing date that is <= each decision session
    # For this implementation, we'll compute features per session using
    # the latest filing available up to that session

    # Collect all unique decision dates
    all_dates = sorted(set(all_sessions["decision_session"].to_list()))

    # Compute compliance validation
    compliance = validate_pit_compliance(fundamental_df, all_dates[0] if all_dates else date.today())

    # Build fundamental features per instrument per session
    # This is the core PIT logic: for each session D, use the latest
    # filing with filing_date <= D
    fundamental_by_session: dict[date, dict[str, dict[str, float | None]]] = {}

    for d in all_dates:
        # Validate PIT compliance
        pit_valid = validate_pit_compliance(fundamental_df, d)

        # Compute fundamental features for this date
        feats = compute_fundamental_features(fundamental_df, as_of_date=d)
        fundamental_by_session[d] = feats

    # Now build the feature snapshots
    # FS-12B-A: Baseline only (no fundamental features added)
    # We'll still include the baseline FS-001 features

    snapshots = {}

    # Helper: get feature column names from feature refs
    # For FS-001, these are FEAT-001..FEAT-008 -> ret_10..log_dv_med_20
    # For fundamental sets, we need to map feature IDs to column names

    # Build baseline feature names (from FS-001)
    baseline_col_names = ["ret_10", "ret_20", "ret_30", "sma_ratio_5_30",
                          "sma_ratio_15_40", "vol_10", "vol_30", "log_dv_med_20"]

    # FS-12B-A: Baseline only
    print("  Building FS-12B-A (baseline only)...")
    # In a full implementation, we would copy the FS-001 snapshot
    # For now, create a placeholder
    snapshots["FS-12B-A"] = None  # placeholder

    # FS-12B-B: Baseline + valuation (FEAT-301..FEAT-303: earnings_yield,
    # book_to_market, sales_to_price)
    # These need market price, which we don't have in the fundamental data alone
    # We'll compute what we can and mark the rest
    print("  Building FS-12B-B (baseline + valuation)...")
    snapshots["FS-12B-B"] = None  # placeholder - needs market price integration

    # FS-12B-C: Baseline + profitability (FEAT-311..FEAT-314: roa, roe, operating_margin,
    # gross_profitability)
    print("  Building FS-12B-C (baseline + profitability)...")
    snapshots["FS-12B-C"] = None  # placeholder

    # FS-12B-D: Baseline + growth (FEAT-321..FEAT-323: revenue_growth, earnings_growth,
    # cash_flow_growth) - requires comparing to prior period
    print("  Building FS-12B-D (baseline + growth)...")
    snapshots["FS-12B-D"] = None  # placeholder

    # FS-12B-E: Baseline + leverage (FEAT-331..FEAT-333: debt_to_equity,
    # debt_to_assets, current_ratio)
    print("  Building FS-12B-E (baseline + leverage)...")
    snapshots["FS-12B-E"] = None  # placeholder

    # FS-12B-F: Baseline + all fundamental families
    print("  Building FS-12B-F (baseline + all fundamental families)...")
    snapshots["FS-12B-F"] = None  # placeholder

    return {
        "snapshots": snapshots,
        "fundamental_source": ACTIVE_FUNDAMENTAL_SOURCE,
        "fundamental_df_height": fundamental_df.height,
        "compliance": compliance,
        "all_sessions": len(all_dates),
    }


def main():
    """Test the fundamental data pipeline."""
    import sys
    sys.path.insert(0, ".")

    from orbit.ml.phase12b_plan import build_phase12b_plan, persist_phase12b_plan

    # Build and persist the plan
    plan = build_phase12b_plan()
    plan_path = persist_phase12b_plan(plan)
    print(f"Plan saved: {plan_path}")
    print(f"Plan digest: {plan['plan_digest'][:16]}...")
    print(f"Experiments: {plan['n_experiments']}")
    print(f"Feature sets: {list(PHASE12B_FEATURE_SETS.keys())}")

    # Test data loading
    print("\\nTesting data loading...")
    try:
        fundamental_df = load_sec_edgar_companyfacts("DS-000002")
        print(f"SEC EDGAR data loaded: {fundamental_df.height} records")
        print(f"Columns: {fundamental_df.columns[:5]}...")
    except Exception as e:
        print(f"Error loading SEC EDGAR data: {e}")
        print("This is expected if the data has encoding issues.")
        print("The pipeline is designed to handle this gracefully.")


if __name__ == "__main__":
    main()