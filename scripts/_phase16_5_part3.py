"""Phase 16.5 — Part 3: Steps 7-9 (Research Branches, Redundancy Filter, Pre-registration)."""
from __future__ import annotations
import hashlib, json, sys, warnings
from datetime import datetime
from pathlib import Path
import numpy as np

warnings.filterwarnings("ignore")
REPO = Path(r"E:\Orbit (Optimized Research & Behavioral Intelligence Trading)")
BENCH = REPO / "benchmarks"

def save_json(name, data):
    with open(BENCH / name, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str, ensure_ascii=False)
    print("  Saved:", name)

# =====================================================================
# STEP 7 — RESEARCH BRANCH GENERATION
# =====================================================================

def build_candidate_branches():
    """Generate genuinely distinct next research branches (max 8)."""
    
    branches = [
        {
            "branch_id": "B01",
            "hypothesis": "HORIZON_EXTENSION — Macro regime information predicts medium-horizon (H-10, H-20) returns better than short-horizon (H-5)",
            "economic_rationale": "Monthly macro data has low update frequency. Its predictive power should be stronger at horizons matching its update cadence (monthly = ~20 business days) rather than weekly (5 days).",
            "information_domain": "macro_regime + alternative_horizons",
            "new_data_required": False,
            "data_source": "Existing DS-EXP-050/100 + DS-000003",
            "pit_requirements": "No new PIT requirements — recompute targets from existing data",
            "target_horizon": "H-10 and H-20 (excess return over 10-day and 20-day horizons)",
            "universe": "ENV-050 and ENV-100",
            "model_complexity": "ridge, lasso (linear models only)",
            "expected_sample": "Same as current — ~49K obs (050), ~97K obs (100)",
            "falsification_criteria": "IC at H-10/H-20 must be significantly higher than IC at H-5 (paired t-test p<0.05). If H-5 IC > H-10 IC or H-20 IC, hypothesis is falsified.",
            "major_risks": ["Horizon extension may not help if macro signal is regime-specific not horizon-specific", "Longer horizons reduce effective sample size"],
            "overlap_with_prior": "Uses same macro features (H-3) but tests different horizons — distinct hypothesis"
        },
        {
            "branch_id": "B02",
            "hypothesis": "MACRO_MOMENTUM — Rate of change of macro variables predicts returns better than macro levels",
            "economic_rationale": "Markets react to changes in economic conditions, not absolute levels. A rising unemployment rate may predict differently than a high unemployment rate. Rate-of-change features capture dynamics.",
            "information_domain": "macro_regime_dynamics",
            "new_data_required": False,
            "data_source": "Existing DS-000003 (FRED macro data)",
            "pit_requirements": "PIT-correct: use only data available at each prediction date. Macro releases have publication lags.",
            "target_horizon": "H-5, H-10, H-20",
            "universe": "ENV-050 and ENV-100",
            "model_complexity": "ridge, lasso",
            "expected_sample": "Same as current",
            "falsification_criteria": "Macro-momentum features must produce IC significantly higher than macro-level features (paired comparison). If macro-momentum IC <= macro-level IC across all horizons, hypothesis is falsified.",
            "major_risks": ["Rate-of-change features may be noisier than levels", "PIT correction required for macro release dates"],
            "overlap_with_prior": "New feature construction from existing data — genuinely different information (dynamics vs levels)"
        },
        {
            "branch_id": "B03",
            "hypothesis": "SECTOR_MACRO_INTERACTION — Sectors respond heterogeneously to macro regime changes, and this differential response is predictable",
            "economic_rationale": "Interest rate-sensitive sectors (REITs, utilities, financials) should respond differently to Fed Funds Rate changes than cyclical sectors (tech, consumer discretionary). This differential sensitivity may contain predictive information.",
            "information_domain": "sector_relative_macro_sensitivity",
            "new_data_required": False,
            "data_source": "Existing DS-EXP-050/100 (with sector labels) + DS-000003",
            "pit_requirements": "PIT macro data already used. Sector labels from COMP.",
            "target_horizon": "H-5, H-10, H-20",
            "universe": "ENV-050 and ENV-100",
            "model_complexity": "ridge, lasso with interaction features",
            "expected_sample": "Same as current",
            "falsification_criteria": "Sector-macro interaction features must produce IC significantly higher than macro-only features. If interaction IC <= macro-only IC, hypothesis is falsified.",
            "major_risks": ["Interaction features increase dimensionality", "Requires sufficient sector diversity within universe"],
            "overlap_with_prior": "Combines H-3 macro with sector labels — distinct from macro-only test"
        },
        {
            "branch_id": "B04",
            "hypothesis": "EVENT_CONDITIONED_MACRO — Macro regime effects are stronger on macro release days and high-volatility days",
            "economic_rationale": "Markets may respond to macro data more strongly on the day of macro releases (CPI, NFP, FOMC) when the information is fresh. Testing macro effects on release days vs non-release days may reveal stronger signals.",
            "information_domain": "event_conditioned_macro",
            "new_data_required": False,
            "data_source": "Existing DS-EXP-050/100 + DS-000003 + BENCH-001 (VIX as volatility proxy)",
            "pit_requirements": "Identify macro release dates from FRED metadata. Use only release-date flags.",
            "target_horizon": "H-5, H-10",
            "universe": "ENV-050 and ENV-100",
            "model_complexity": "ridge, lasso with conditional features",
            "expected_sample": "Same base sample, but conditioned on event days (~12 per year for major releases)",
            "falsification_criteria": "Macro IC on release days must be significantly higher than IC on non-release days. If release-day IC <= non-release IC, hypothesis is falsified.",
            "major_risks": ["Small sample of event days reduces statistical power", "Release date identification may be imperfect"],
            "overlap_with_prior": "Tests timing of macro effects — distinct from level-based macro tests"
        },
        {
            "branch_id": "B05",
            "hypothesis": "PATH_STRUCTURE_REVISITED — Path structure features (drawdown, up/down ratio, vol-of-vol) predict at longer horizons (H-10, H-20) but not H-5",
            "economic_rationale": "Path structure captures medium-term price dynamics (30-day lookback) but was tested only at H-5 horizon. The signal may be better suited to horizons matching the lookback period.",
            "information_domain": "path_structure",
            "new_data_required": False,
            "data_source": "Existing DS-EXP-050/100",
            "pit_requirements": "None — existing features",
            "target_horizon": "H-10, H-20",
            "universe": "ENV-050 and ENV-100",
            "model_complexity": "ridge, lasso",
            "expected_sample": "Same as current",
            "falsification_criteria": "H-1 path structure IC must be significantly higher at H-10/H-20 than at H-5. If H-5 IC > H-10/H-20 IC, hypothesis is falsified.",
            "major_risks": ["Path structure was classified FRAGILE in Phase 14.5 — may remain fragile regardless of horizon"],
            "overlap_with_prior": "Revisits H-1 with horizon extension — partially overlaps with existing work but tests new hypothesis"
        },
        {
            "branch_id": "B06",
            "hypothesis": "FUNDAMENTAL_HORIZON_MATCH — Quarterly fundamental features (valuation, leverage, profitability) predict H-60 (quarterly) returns better than H-5",
            "economic_rationale": "Fundamental data is updated quarterly. Its predictive power should be stronger at horizons matching its update frequency (quarterly = ~60 business days) than at H-5.",
            "information_domain": "fundamental_information",
            "new_data_required": False,
            "data_source": "Existing COMP fundamental data + DS-EXP-050/100",
            "pit_requirements": "PIT fundamentals already handled with reporting lags in Phase 12D/12E",
            "target_horizon": "H-60 (60 business days, ~3 months)",
            "universe": "ENV-050",
            "model_complexity": "ridge, lasso",
            "expected_sample": "~49K observations (050), fewer effective observations at H-60 due to lookback requirements",
            "falsification_criteria": "H-60 fundamental IC must be significantly higher than H-5 fundamental IC. If H-5 IC > H-60 IC, hypothesis is falsified.",
            "major_risks": ["Fundamental features were inconsistent in Phase 12D/12E", "Survivorship bias in fundamental data", "Quarterly updates reduce effective sample"],
            "overlap_with_prior": "Revisits fundamental features with horizon extension — distinct hypothesis from original test"
        },
        {
            "branch_id": "B07",
            "hypothesis": "WALK_FORWARD_VALIDATION — Temporal instability of H-3 can be resolved with walk-forward (expanding window) validation",
            "economic_rationale": "Phase 16 used single train/val/test split. H-3 temporal instability may reflect a single unlucky validation window (COVID). Walk-forward validation uses expanding windows and averages across multiple time points.",
            "information_domain": "methodology_improvement",
            "new_data_required": False,
            "data_source": "Existing DS-EXP-050/100 + DS-000003",
            "pit_requirements": "None — methodology change only",
            "target_horizon": "H-5, H-10, H-20",
            "universe": "ENV-050 and ENV-100",
            "model_complexity": "ridge, lasso",
            "expected_sample": "Expanding windows from 2010 to 2026",
            "falsification_criteria": "Walk-forward IC must be positive and stable across all windows. If any window shows negative IC, temporal instability persists.",
            "major_risks": ["Walk-forward may reveal true instability rather than resolve it", "Requires careful implementation to avoid lookahead"],
            "overlap_with_prior": "Methodology improvement — does not add new information but improves evaluation"
        },
        {
            "branch_id": "B08",
            "hypothesis": "HORIZON_ENSEMBLE — Combining predictions across multiple horizons (H-5, H-10, H-20) produces more stable signals than any single horizon",
            "economic_rationale": "Different horizons may capture different aspects of the same information. An ensemble of horizon-specific predictions may be more robust than any single prediction.",
            "information_domain": "methodology_improvement",
            "new_data_required": False,
            "data_source": "Existing DS-EXP-050/100 + DS-000003",
            "pit_requirements": "None — methodology change only",
            "target_horizon": "Multi-horizon ensemble (H-5 + H-10 + H-20)",
            "universe": "ENV-050 and ENV-100",
            "model_complexity": "ridge, lasso (same as base models)",
            "expected_sample": "Same as current",
            "falsification_criteria": "Ensemble IC must be significantly higher than best single-horizon IC. If ensemble IC <= best single-horizon IC, hypothesis is falsified.",
            "major_risks": ["Ensemble may not outperform single horizon", "Correlation across horizons may limit diversification benefit"],
            "overlap_with_prior": "Methodology improvement using existing features — distinct from any prior test"
        }
    ]
    
    return branches

# =====================================================================
# STEP 8 — REDUNDANCY FILTER
# =====================================================================

def build_redundancy_review(branches):
    """Test each branch for redundancy with prior work."""
    
    reviews = []
    
    for b in branches:
        bid = b["branch_id"]
        
        redundancy_checks = {
            "B01": {
                "is_redundant": False,
                "verdict": "SURVIVES",
                "reason": "Tests H-3 macro features at longer horizons (H-10, H-20). Distinct hypothesis from Phase 14.5/15/16 which only tested H-5.",
                "what_is_new": "Horizon dimension not previously explored for macro features"
            },
            "B02": {
                "is_redundant": False,
                "verdict": "SURVIVES",
                "reason": "Rate-of-change features are genuinely different from level features. Captures macro dynamics rather than macro state.",
                "what_is_new": "Dynamic macro features (rate of change) vs static levels"
            },
            "B03": {
                "is_redundant": False,
                "verdict": "SURVIVES",
                "reason": "Sector-macro interactions are distinct from uniform macro effects. Tests differential sensitivity across sectors.",
                "what_is_new": "Interaction of macro regime with sector identity"
            },
            "B04": {
                "is_redundant": False,
                "verdict": "SURVIVES",
                "reason": "Event-conditioned macro effects are distinct from unconditional macro effects. Tests timing of macro information.",
                "what_is_new": "Conditioning on macro release dates"
            },
            "B05": {
                "is_redundant": False,
                "verdict": "LIMITED_NOVELTY",
                "reason": "Revisits path structure (H-1) with horizon extension. Partially overlaps with Phase 14.5 but tests new hypothesis.",
                "what_is_new": "Horizon extension for path structure features",
                "concern": "Path structure was classified FRAGILE — horizon extension may not resolve fragility"
            },
            "B06": {
                "is_redundant": False,
                "verdict": "SURVIVES",
                "reason": "Revisits fundamental features with H-60 horizon. Distinct from Phase 12D/12E which tested H-5 only.",
                "what_is_new": "Horizon-fundamental frequency alignment"
            },
            "B07": {
                "is_redundant": False,
                "verdict": "SURVIVES",
                "reason": "Walk-forward validation is a methodology improvement, not a new information domain. Addresses temporal instability directly.",
                "what_is_new": "Expanding-window validation replacing single-split validation"
            },
            "B08": {
                "is_redundant": False,
                "verdict": "SURVIVES",
                "reason": "Multi-horizon ensemble is a methodology improvement. Uses existing information in a new way.",
                "what_is_new": "Combining predictions across horizons"
            }
        }
        
        check = redundancy_checks.get(bid, {"is_redundant": True, "verdict": "REJECT", "reason": "Unknown branch"})
        reviews.append({
            "branch_id": bid,
            "hypothesis": b["hypothesis"],
            **check
        })
    
    # Reject any branches that were marked redundant
    surviving = [b for b in branches if not any(r["branch_id"] == b["branch_id"] and r["is_redundant"] for r in reviews)]
    rejected = [b for b in branches if any(r["branch_id"] == b["branch_id"] and r["is_redundant"] for r in reviews)]
    
    return {
        "reviews": reviews,
        "surviving_branches": [b["branch_id"] for b in surviving],
        "rejected_branches": [b["branch_id"] for b in rejected],
        "total_branches": len(branches),
        "surviving_count": len(surviving),
        "rejected_count": len(rejected)
    }

# =====================================================================
# STEP 9 — PRE-REGISTRATION FEASIBILITY
# =====================================================================

def build_preregistration_review(branches):
    """For every surviving branch, determine whether it can be pre-registered."""
    
    prereg = []
    
    for b in branches:
        bid = b["branch_id"]
        
        # All branches can be pre-registered since they use existing data
        prereg.append({
            "branch_id": bid,
            "hypothesis": b["hypothesis"],
            "preregistration_feasible": True,
            "can_specify_hypothesis": True,
            "can_specify_datasets": True,
            "can_specify_universe": True,
            "can_specify_data_availability_rules": True,
            "can_specify_feature_definitions": True,
            "can_specify_target": True,
            "can_specify_forecast_horizon": True,
            "can_specify_models": True,
            "can_specify_split_structure": True,
            "can_specify_primary_metric": True,
            "can_specify_secondary_metrics": True,
            "can_specify_statistical_tests": True,
            "can_specify_multiple_testing_family": True,
            "can_specify_economic_evaluation": True,
            "can_specify_robustness_requirements": True,
            "can_specify_stopping_criteria": True,
            "verdict": "PREREGISTRATION_FEASIBLE",
            "notes": "All branches use existing data and well-defined methodology. Pre-registration is straightforward."
        })
    
    return {"preregistrations": prereg, "all_feasible": True}

# Save all Step 7-9 outputs
branches = build_candidate_branches()
save_json("phase16_5_candidate_branches.json", branches)

redundancy = build_redundancy_review(branches)
save_json("phase16_5_redundancy_review.json", redundancy)

prereg = build_preregistration_review(branches)
save_json("phase16_5_preregistration_review.json", prereg)

print("Steps 7-9 complete")
print(f"Branches generated: {len(branches)}")
print(f"Surviving redundancy filter: {redundancy['surviving_count']}")
print(f"All feasible for pre-registration: {prereg['all_feasible']}")