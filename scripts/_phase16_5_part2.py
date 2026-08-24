"""Phase 16.5 — Part 2: Steps 3-6 (Information Gaps, Target/Horizon, Temporal, Data Limitations)."""
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

def load_json(name):
    with open(BENCH / name, encoding="utf-8") as f:
        return json.load(f)

def canonical(obj):
    return json.dumps(obj, sort_keys=True, default=str, ensure_ascii=False)

def digest_full(obj):
    return hashlib.sha256(canonical(obj).encode()).hexdigest()

# Record digests of all source artifacts used
source_artifacts = {}
for a in [
    "phase14_5_results.json", "phase14_5_robustness.json",
    "phase15_audit.json", "phase15_1_audit.json", "phase15_2_audit.json",
    "phase16_results.json", "phase16_temporal_stability.json",
    "phase16_baselines.json", "phase16_robustness.json", "phase16_adversarial.json"
]:
    try:
        source_artifacts[a] = digest_full(load_json(a))
    except Exception:
        source_artifacts[a] = "NOT_FOUND"

# =====================================================================
# STEP 3 — INFORMATION GAP ANALYSIS
# =====================================================================

def build_information_gaps():
    """Identify information domains NOT yet tested by ORBIT."""
    
    gaps = [
        {
            "domain": "earnings_surprises_and_revisions",
            "description": "Post-earnings-announcement drift, earnings surprise magnitude, analyst revision direction",
            "genuinely_different": True,
            "data_available": False,
            "pit_feasible": False,
            "historical_coverage": "Limited — EPS surprise databases (IBES/Refinitiv) required",
            "economic_mechanism": True,
            "falsifiable": True,
            "without_data_snooping": True,
            "new_data_required": True,
            "data_source_type": "IBES, Refinitiv, or FactSet earnings surprise data",
            "verdict": "REQUIRES_NEW_DATA",
            "reason": "ORBIT does not have earnings surprise or analyst revision data. Not available in DS-EXP-050/100."
        },
        {
            "domain": "options_implied_volatility",
            "description": "Implied volatility surface, skew, term structure, volatility risk premium",
            "genuinely_different": True,
            "data_available": False,
            "pit_feasible": True,
            "historical_coverage": "Requires CBOE options data back to 2010",
            "economic_mechanism": True,
            "falsifiable": True,
            "without_data_snooping": True,
            "new_data_required": True,
            "data_source_type": "CBOE options chains or IVIX data",
            "verdict": "REQUIRES_NEW_DATA",
            "reason": "ORBIT has no options data. Would require CBOE or OptionsMetrics data acquisition."
        },
        {
            "domain": "liquidity_dynamics",
            "description": "Bid-ask spread, market depth, order flow imbalance, Amihud illiquidity",
            "genuinely_different": True,
            "data_available": False,
            "pit_feasible": True,
            "historical_coverage": "Level 2 data required; TAQ for US equities back to 2010",
            "economic_mechanism": True,
            "falsifiable": True,
            "without_data_snooping": True,
            "new_data_required": True,
            "data_source_type": "TAQ or proprietary order book data",
            "verdict": "REQUIRES_NEW_DATA",
            "reason": "ORBIT OHLCV data lacks bid-ask or order book information."
        },
        {
            "domain": "corporate_events_and_actions",
            "description": "M&A announcements, buybacks, dividends, insider transactions, 13F filings",
            "genuinely_different": True,
            "data_available": False,
            "pit_feasible": True,
            "historical_coverage": "Event databases (WRDS Eventus, SecApi) required",
            "economic_mechanism": True,
            "falsifiable": True,
            "without_data_snooping": True,
            "new_data_required": True,
            "data_source_type": "SEC filings, exchange event data, or event databases",
            "verdict": "REQUIRES_NEW_DATA",
            "reason": "ORBIT does not have structured corporate event data."
        },
        {
            "domain": "cross_asset_relationships",
            "description": "Equity-bond correlation, credit spreads, commodity-equity relationships, FX-equity linkages",
            "genuinely_different": True,
            "data_available": False,
            "pit_feasible": True,
            "historical_coverage": "Multi-asset class data required",
            "economic_mechanism": True,
            "falsifiable": True,
            "without_data_snooping": True,
            "new_data_required": True,
            "data_source_type": "Bloomberg, FRED, or multi-asset data providers",
            "verdict": "PARTIALLY_AVAILABLE",
            "reason": "FRED DS-000003 has some cross-asset data (rates, unemployment, CPI) but lacks credit spreads, commodity prices, and FX."
        },
        {
            "domain": "alternative_prediction_horizons",
            "description": "Test H-1 (1-day), H-2 (2-day), H-10 (10-day), H-20 (20-day) horizons systematically",
            "genuinely_different": True,
            "data_available": True,
            "pit_feasible": True,
            "historical_coverage": "Existing OHLCV data supports all horizons up to 60 days",
            "economic_mechanism": True,
            "falsifiable": True,
            "without_data_snooping": True,
            "new_data_required": False,
            "data_source_type": "Existing DS-EXP-050 and DS-EXP-100",
            "verdict": "ACTIONABLE_NOW",
            "reason": "Current data supports multiple horizons. LAB-006 only tested H-5 (5-day). Different horizons may capture different information dynamics."
        },
        {
            "domain": "event_conditioned_returns",
            "description": "Returns conditioned on high-vol, high-volume, earnings-season, or macro-release days",
            "genuinely_different": True,
            "data_available": True,
            "pit_feasible": True,
            "historical_coverage": "Existing OHLCV + macro data supports this",
            "economic_mechanism": True,
            "falsifiable": True,
            "without_data_snooping": True,
            "new_data_required": False,
            "data_source_type": "Existing DS-EXP-050, DS-EXP-100, BENCH-001, DS-000003",
            "verdict": "ACTIONABLE_NOW",
            "reason": "Conditioned predictions on macro-release days or high-volatility regimes may capture regime-specific signal."
        },
        {
            "domain": "cross_sectional_ranking_improvements",
            "description": "Improve cross-sectional features with percentile ranks, z-scores, winsorization, rolling windows",
            "genuinely_different": False,
            "data_available": True,
            "pit_feasible": True,
            "historical_coverage": "Full coverage with existing data",
            "economic_mechanism": False,
            "falsifiable": True,
            "without_data_snooping": False,
            "new_data_required": False,
            "data_source_type": "Existing data",
            "verdict": "REJECT_REDUNDANT",
            "reason": "Cross-sectional features already tested in Phases 10-11. Feature engineering without new information is redundant."
        },
        {
            "domain": "momentum_and_reversal_alternatives",
            "description": "Test momentum/reversal at different lookback windows, skip-day variants, or industry-neutralized variants",
            "genuinely_different": False,
            "data_available": True,
            "pit_feasible": True,
            "historical_coverage": "Full coverage",
            "economic_mechanism": True,
            "falsifiable": True,
            "without_data_snooping": False,
            "new_data_required": False,
            "data_source_type": "Existing OHLCV data",
            "verdict": "REJECT_REDUNDANT",
            "reason": "Momentum and reversal are subcategories of the OHLCV technical features already extensively tested in Phases 9-13C."
        },
        {
            "domain": "factor_exposures",
            "description": "Decompose returns into Fama-French factors, test factor momentum, factor timing",
            "genuinely_different": True,
            "data_available": False,
            "pit_feasible": True,
            "historical_coverage": "Fama-French data available from Kenneth French website back to 1926",
            "economic_mechanism": True,
            "falsifiable": True,
            "without_data_snooping": True,
            "new_data_required": True,
            "data_source_type": "Kenneth French Data Library or CRSP",
            "verdict": "REQUIRES_NEW_DATA",
            "reason": "ORBIT does not have factor exposure data. Would require Fama-French data acquisition."
        },
        {
            "domain": "sector_relative_macro_sensitivity",
            "description": "Test whether sectors respond differently to macro regime — differential macro sensitivity",
            "genuinely_different": True,
            "data_available": True,
            "pit_feasible": True,
            "historical_coverage": "Existing data supports this (sector labels + macro + returns)",
            "economic_mechanism": True,
            "falsifiable": True,
            "without_data_snooping": True,
            "new_data_required": False,
            "data_source_type": "Existing DS-EXP-050/100 (with sector labels) + DS-000003",
            "verdict": "ACTIONABLE_NOW",
            "reason": "H-3 macro regime was tested uniformly across all sectors. Differential sensitivity by sector may capture heterogeneous effects."
        },
        {
            "domain": "macro_momentum_features",
            "description": "Rate of change of macro variables, macro surprise (actual vs expected), macro momentum",
            "genuinely_different": True,
            "data_available": True,
            "pit_feasible": True,
            "historical_coverage": "FRED data supports rate-of-change features",
            "economic_mechanism": True,
            "falsifiable": True,
            "without_data_snooping": True,
            "new_data_required": False,
            "data_source_type": "Existing DS-000003",
            "verdict": "ACTIONABLE_NOW",
            "reason": "Current H-3 uses only levels of macro variables. Rate of change and surprise features may contain distinct information."
        }
    ]
    return gaps

# =====================================================================
# STEP 4 — TARGET & HORIZON AUDIT
# =====================================================================

def build_target_horizon_audit():
    """Audit the prediction targets and horizons used across Phases 9-16."""
    
    audit = {
        "current_target": "LAB-006 (5-day excess return, corrected for dividends)",
        "current_horizon": "H-5 (5 business days)",
        "target_formulation": "Excess return vs equal-weight universe median",
        "classification": {
            "cross_sectional_ranking": {
                "status": "TESTED",
                "details": "Used throughout Phases 9-13C. Cross-sectional ranking within universe is the primary target.",
                "coverage": "Comprehensive across all phases"
            },
            "directional_prediction": {
                "status": "TESTED",
                "details": "Implicit in IC evaluation — positive IC means model assigns higher scores to stocks that outperform.",
                "coverage": "All models tested"
            },
            "absolute_return_prediction": {
                "status": "UNTESTED",
                "details": "ORBIT uses relative (cross-sectional) prediction, not absolute return forecasting.",
                "coverage": "Never tested"
            },
            "conditional_prediction": {
                "status": "PARTIALLY_TESTED",
                "details": "H-3 macro regime conditions predictions on macro state, but prediction itself is unconditional within each regime.",
                "coverage": "Partial — regime conditioning only"
            },
            "event_conditioned_prediction": {
                "status": "UNTESTED",
                "details": "No predictions conditioned on specific events (earnings, macro releases, corporate actions).",
                "coverage": "Never tested"
            },
            "short_horizon_1_day": {
                "status": "UNTESTED",
                "details": "LAB-006 only computed for H-5. No 1-day horizon tested.",
                "coverage": "Never tested",
                "feasibility": "Can be computed from existing OHLCV data"
            },
            "short_horizon_2_day": {
                "status": "UNTESTED",
                "details": "No 2-day horizon tested.",
                "coverage": "Never tested",
                "feasibility": "Can be computed from existing data"
            },
            "medium_horizon_10_day": {
                "status": "UNTESTED",
                "details": "No 10-day horizon tested.",
                "coverage": "Never tested",
                "feasibility": "Can be computed from existing data"
            },
            "long_horizon_20_day": {
                "status": "UNTESTED",
                "details": "No 20-day horizon tested.",
                "coverage": "Never tested",
                "feasibility": "Can be computed from existing data"
            },
            "long_horizon_60_day": {
                "status": "UNTESTED",
                "details": "No 60-day horizon tested.",
                "coverage": "Never tested",
                "feasibility": "Can be computed from existing data"
            }
        },
        "horizon_mismatch_analysis": {
            "hypothesis": "Macro regime information (low-frequency, monthly/quarterly updates) may be better suited to longer horizons than the 5-day prediction window currently used.",
            "evidence": "H-3 uses FRED macro data (monthly release frequency, quarterly for some). Predicting 5-day returns from monthly macro updates creates a frequency mismatch.",
            "testable": True,
            "test_proposal": "Recompute targets for H-10, H-20 horizons and test H-3 macro regime features against longer horizons",
            "prior_evidence": "Phase 15.2 temporal instability (val IC negative, test IC positive) may partly reflect horizon mismatch"
        },
        "target_formulation_audit": {
            "LAB_005_defect": "Excess return calculated vs individual stock historical mean — introduced bias",
            "LAB_006_correction": "Excess return calculated vs equal-weight universe median — correct formulation",
            "current_status": "LAB-006 is correct and should be retained",
            "remaining_concern": "Cross-sectional median subtraction removes market beta but does not control for sector or factor exposures"
        }
    }
    return audit

# =====================================================================
# STEP 5 — TEMPORAL COVERAGE AUDIT
# =====================================================================

def build_temporal_audit():
    """Audit temporal coverage across market regimes."""
    
    periods = [
        {"period": "2010-2013", "regime": "Post-GFC recovery, low rates, low volatility", "split": "train", "coverage": "Full"},
        {"period": "2014-2015", "regime": "Bull market, low volatility, oil crash", "split": "train", "coverage": "Full"},
        {"period": "2016-2017", "regime": "Low volatility, synchronized global growth", "split": "train", "coverage": "Full"},
        {"period": "2018", "regime": "Q4 correction, rising rates, vol spike", "split": "train", "coverage": "Full"},
        {"period": "2019", "regime": "Recovery, Fed pivot, low rates", "split": "val", "coverage": "Full"},
        {"period": "2020", "regime": "COVID crash and recovery — extreme regime", "split": "val", "coverage": "Full but extreme"},
        {"period": "2021", "regime": "Bull market, meme stocks, reopening", "split": "val", "coverage": "Full"},
        {"period": "2022", "regime": "Bear market, rising rates, inflation spike", "split": "test", "coverage": "Full"},
        {"period": "2023", "regime": "Recovery, AI rally, rate plateau", "split": "test", "coverage": "Full"},
        {"period": "2024-2025", "regime": "Continued recovery, rate cuts expected", "split": "test", "coverage": "Partial — latest data"}
    ]
    
    regime_coverage = {
        "bull_markets": "Covered in train (2010-2017) and test (2023-2025)",
        "bear_markets": "Covered in train (Q4 2018) and test (2022)",
        "high_volatility": "Partially covered (Q4 2018, COVID 2020, Q4 2022)",
        "low_volatility": "Covered in train (2016-2017) and val (2019)",
        "crisis_periods": "COVID 2020 (val) — single observation of extreme regime",
        "inflationary_periods": "Covered only in test (2022-2023) — no train/validation inflation data",
        "low_rate_periods": "Covered in train (2010-2021)",
        "rising_rate_periods": "2022 test only — limited coverage",
        "post_crisis_periods": "Train period (2010-2013)"
    }
    
    h3_temporal_analysis = {
        "observation": "H-3 macro regime showed strong IC in test but weak/negative IC in validation",
        "possible_explanation": "COVID-2020 regime in validation period corrupted macro regime signal",
        "implication": "Single validation period (2019-2021) dominated by COVID is not representative of typical macro conditions",
        "recommendation": "Walk-forward or expanding-window validation needed; COVID period may need separate treatment",
        "severity": "MATERIAL — H-3 temporal instability cannot be assessed with single validation window"
    }
    
    test_period_specifics = {
        "inflation_regime_2022_2023": "Unique in sample — first sustained inflation since 1980s",
        "rate_regime_2022_2025": "First sustained high-rate period since 2007",
        "regime_novelty": "Test period contains macro conditions unseen in training data",
        "implication": "Model performance in test may reflect regime adaptation rather than robust signal",
        "concern_level": "HIGH — train/val do not contain inflation or rising-rate regimes"
    }
    
    return {
        "periods": periods,
        "regime_coverage": regime_coverage,
        "h3_temporal_analysis": h3_temporal_analysis,
        "test_period_specifics": test_period_specifics,
        "overall_assessment": "ORBIT train+val period (2010-2021) is dominated by low-rate, low-inflation, post-GFC conditions. Test period (2022-2026) includes inflation, rate hikes, and geopolitical shocks. Regime diversity is inadequate for validating macro-dependent models."
    }

# =====================================================================
# STEP 6 — DATA LIMITATION AUDIT
# =====================================================================

def build_data_limitations():
    """Audit structural data limitations."""
    
    limitations = [
        {
            "dimension": "number_of_instruments",
            "status": "MODERATE",
            "details": "ENV-050 (50 stocks), ENV-100 (97 stocks). Both are subsets of S&P 500.",
            "concern": "Limited cross-sectional diversity. Sector representation may be skewed.",
            "impact_on_research": "Limits ability to detect cross-sectional patterns requiring broad diversification"
        },
        {
            "dimension": "survivorship_concerns",
            "status": "MINOR",
            "details": "DS-EXP-050/100 use current constituents — survivorship bias possible for pre-2020 delistings",
            "concern": "Delisted stocks excluded from early periods may bias results upward",
            "impact_on_research": "Minor for most features; more concerning for fundamental features (Phase 12D/12E)"
        },
        {
            "dimension": "historical_coverage",
            "status": "MINOR",
            "details": "Train starts 2010. 14 years of history (2010-2024) split into train/val/test.",
            "concern": "Pre-2010 data not available — misses GFC, dot-com bust",
            "impact_on_research": "Limits ability to test across multiple complete market cycles"
        },
        {
            "dimension": "delisted_instruments",
            "status": "MINOR",
            "details": "Delisted stocks not included in universe composition",
            "concern": "Survivorship bias for early periods",
            "impact_on_research": "Minor impact for most analyses"
        },
        {
            "dimension": "missing_corporate_actions",
            "status": "MINOR",
            "details": "Dividends, splits, and other corporate actions handled in data construction",
            "concern": "Minor data quality concern",
            "impact_on_research": "Already addressed in LAB-006 label correction"
        },
        {
            "dimension": "benchmark_construction",
            "status": "MINOR",
            "details": "Equal-weight universe median used as benchmark",
            "concern": "Does not account for market cap weighting or sector neutrality",
            "impact_on_research": "Minor — label may capture sector effects rather than pure stock selection"
        },
        {
            "dimension": "macro_release_timing",
            "status": "MATERIAL",
            "details": "FRED macro data used with same-day release assumption. Macro releases have specific timing (e.g., CPI first Friday, NFP first Friday).",
            "concern": "Look-ahead bias possible if macro data not aligned with correct release dates",
            "impact_on_research": "Could invalidate H-3 macro regime results if macro data release dates not handled correctly",
            "note": "Phase 15.2 did not audit macro release timing alignment"
        },
        {
            "dimension": "fundamental_data_availability_timing",
            "status": "MODERATE",
            "details": "Fundamental data frequency is quarterly with reporting lag (45-90 days). Phase 12D/12E used COMP (Compustat) with lags.",
            "concern": "Quarterly fundamentals are low-frequency relative to 5-day prediction horizon",
            "impact_on_research": "Fundamental features may be better suited to longer horizons (monthly/quarterly)"
        },
        {
            "dimension": "data_frequency",
            "status": "MODERATE",
            "details": "Daily OHLCV for all assets. Monthly macro data. Quarterly fundamentals.",
            "concern": "Multi-frequency data requires careful temporal alignment",
            "impact_on_research": "Feature construction must handle mixed frequencies correctly"
        },
        {
            "dimension": "observation_count_versus_complexity",
            "status": "MODERATE",
            "details": "ENV-050 has ~49K observations. ENV-100 has ~97K. H-3 macro features tested with ridge/lasso.",
            "concern": "Sample size adequate for linear models but may be insufficient for complex nonlinear interactions with macro features",
            "impact_on_research": "Limits model complexity for macro-dependent hypotheses"
        },
        {
            "dimension": "options_data",
            "status": "CRITICAL",
            "details": "No options or derivatives data available",
            "concern": "Cannot test implied volatility, volatility risk premium, or options-based signals",
            "impact_on_research": "Excludes entire information domain from research"
        },
        {
            "dimension": "earnings_data",
            "status": "CRITICAL",
            "details": "No earnings surprise, analyst revisions, or earnings calendar data",
            "concern": "Cannot test post-earnings drift, analyst expectation changes",
            "impact_on_research": "Excludes fundamental information domain from research"
        },
        {
            "dimension": "order_flow_data",
            "status": "CRITICAL",
            "details": "No Level 2, order book, or tick data",
            "concern": "Cannot test liquidity dynamics, order flow imbalance, market microstructure",
            "impact_on_research": "Excludes market microstructure information domain"
        }
    ]
    
    overall_summary = {
        "critical_limitations": 3,
        "material_limitations": 1,
        "moderate_limitations": 4,
        "minor_limitations": 5,
        "verdict": "ORBIT has sufficient OHLCV, macro, and sector data for most analyses. However, critical gaps exist in options, earnings, and order flow data. These are structural limitations that cannot be resolved through feature engineering."
    }
    
    return {"limitations": limitations, "overall_summary": overall_summary}

# Save all Step 3-6 outputs
save_json("phase16_5_information_gaps.json", build_information_gaps())
save_json("phase16_5_target_horizon_audit.json", build_target_horizon_audit())
save_json("phase16_5_temporal_audit.json", build_temporal_audit())
save_json("phase16_5_data_limitations.json", build_data_limitations())
save_json("phase16_5_source_digests.json", source_artifacts)

print("Steps 3-6 complete")