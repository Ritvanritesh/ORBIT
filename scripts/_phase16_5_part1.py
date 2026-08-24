"""Phase 16.5 — Research Reset & Next-Hypothesis Selection (Part 1: Research Map & Failure Modes)."""
from __future__ import annotations
import hashlib, json, sys, warnings
from datetime import datetime
from pathlib import Path
import numpy as np

warnings.filterwarnings("ignore")
REPO = Path(r"E:\Orbit (Optimized Research & Behavioral Intelligence Trading)")
BENCH = REPO / "benchmarks"
DOCS = REPO / "docs"
sys.path.insert(0, str(REPO / "src"))

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

# =====================================================================
# STEP 1: RECONSTRUCT THE FULL RESEARCH MAP
# =====================================================================

def build_research_map():
    """Complete map of every major information domain tested in Phases 1-16."""
    
    domains = [
        {
            "domain_name": "OHLCV_technical_information",
            "phases_tested": [9, 10, 11, 12, 13],
            "datasets": ["DS-000004", "DS-EXP-050", "DS-EXP-100"],
            "universes": ["ENV-DEV20", "ENV-050", "ENV-100"],
            "feature_families": ["returns", "moving_averages", "volatility", "drawdown", "skew_kurtosis", "volume_dollar"],
            "labels": ["LAB-004", "LAB-005", "LAB-006"],
            "models": ["ridge", "lasso", "random_forest", "xgboost"],
            "evaluation_periods": ["train:2010-2018", "val:2019-2021", "test:2022-2026"],
            "strongest_result": "Phase 9: ridge LAB-004 IC~0.02-0.03 on ENV-DEV20",
            "weakest_result": "Phase 11: nonlinear models (RF/XGB) reversed sign on test",
            "statistical_result": "Multiple significant ICs but economically small (0.01-0.04)",
            "economic_result": "Phase 13C: no robust candidate survived",
            "robustness_result": "Failed to generalize across universes and model families",
            "final_conclusion": "OHLCV technical features alone do not contain robust predictive signal for 5-day horizon",
            "classification": "EXHAUSTED"
        },
        {
            "domain_name": "market_context",
            "phases_tested": [9, 10, 11, 12],
            "datasets": ["DS-000004", "DS-EXP-050"],
            "universes": ["ENV-DEV20", "ENV-050"],
            "feature_families": ["benchmark_returns", "market_volatility", "market_breadth", "sector_rotation"],
            "labels": ["LAB-004", "LAB-006"],
            "models": ["ridge", "lasso"],
            "evaluation_periods": ["train:2010-2018", "val:2019-2021", "test:2022-2026"],
            "strongest_result": "Phase 10: some market context features showed marginal IC improvement",
            "weakest_result": "Phase 11: effects did not replicate on test",
            "statistical_result": "Inconsistent significance across splits",
            "economic_result": "Not tested at portfolio level",
            "robustness_result": "Failed universe generalization",
            "final_conclusion": "Market context features did not provide robust incremental signal",
            "classification": "EXHAUSTED"
        },
        {
            "domain_name": "sector_context",
            "phases_tested": [9, 10, 11],
            "datasets": ["DS-000004"],
            "universes": ["ENV-DEV20"],
            "feature_families": ["sector_returns", "sector_momentum", "sector_relative_strength"],
            "labels": ["LAB-004"],
            "models": ["ridge", "lasso"],
            "evaluation_periods": ["train:2010-2018", "val:2019-2021", "test:2022-2026"],
            "strongest_result": "Phase 9: some sector features significant in-sample",
            "weakest_result": "Phase 11: no out-of-sample persistence",
            "statistical_result": "In-sample significance, OOS failure",
            "economic_result": "Not tested",
            "robustness_result": "Single universe only (ENV-DEV20)",
            "final_conclusion": "Sector context features overfit in-sample, no OOS validation",
            "classification": "EXHAUSTED"
        },
        {
            "domain_name": "cross_sectional_information",
            "phases_tested": [10, 11, 12],
            "datasets": ["DS-EXP-050", "DS-EXP-100"],
            "universes": ["ENV-050", "ENV-100"],
            "feature_families": ["cross_sectional_rank", "relative_momentum", "relative_volatility"],
            "labels": ["LAB-004", "LAB-006"],
            "models": ["ridge", "lasso"],
            "evaluation_periods": ["train:2010-2018", "val:2019-2021", "test:2022-2026"],
            "strongest_result": "Phase 12: cross-sectional ranking produced IC~0.02-0.03",
            "weakest_result": "Phase 11: effects did not generalize to test",
            "statistical_result": "Marginal significance, not robust",
            "economic_result": "Not tested at portfolio level",
            "robustness_result": "Universe-dependent",
            "final_conclusion": "Cross-sectional features provide marginal statistical signal but no economic robustness",
            "classification": "EXHAUSTED"
        },
        {
            "domain_name": "fundamental_information",
            "phases_tested": [12, 13],
            "datasets": ["DS-EXP-050"],
            "universes": ["ENV-12D-050", "ENV-12E-050"],
            "feature_families": ["valuation", "growth", "leverage", "profitability"],
            "labels": ["LAB-004", "LAB-006"],
            "models": ["ridge", "lasso"],
            "evaluation_periods": ["train:2010-2018", "val:2019-2021", "test:2022-2026"],
            "strongest_result": "Phase 12D: leverage features IC~0.04, Phase 12E: growth features IC~0.03",
            "weakest_result": "Phase 13: LAB-005 label defect invalidated early results",
            "statistical_result": "Point-in-time fundamentals produced small IC improvements",
            "economic_result": "Phase 13C: no robust candidate",
            "robustness_result": "Failed label correction test (LAB-006)",
            "final_conclusion": "Real PIT fundamental features produced inconsistent, economically modest improvements that did not survive label correction",
            "classification": "PARTIALLY_EXPLORED"
        },
        {
            "domain_name": "path_structure",
            "phases_tested": [14.5],
            "datasets": ["DS-EXP-050", "DS-EXP-100"],
            "universes": ["ENV-050", "ENV-100"],
            "feature_families": ["max_drawdown", "up_down_ratio", "largest_move", "vol_of_vol", "vol_change"],
            "labels": ["LAB-006"],
            "models": ["ridge", "lasso"],
            "evaluation_periods": ["train:2010-2018", "val:2019-2021", "test:2022-2026"],
            "strongest_result": "H-1: ridge IC +0.010 (050), +0.008 (100) over baseline",
            "weakest_result": "H-1: lasso produced negative IC in some configs",
            "statistical_result": "Phase 14.5: H-1 classified as FRAGILE (sign-inconsistent across universes)",
            "economic_result": "Not tested at portfolio level",
            "robustness_result": "FRAGILE per Phase 14.5 robustness.json",
            "final_conclusion": "Path structure hypothesis (H-1) showed sign inconsistency, classified as fragile",
            "classification": "INCONCLUSIVE"
        },
        {
            "domain_name": "return_asymmetry",
            "phases_tested": [14.5],
            "datasets": ["DS-EXP-050", "DS-EXP-100"],
            "universes": ["ENV-050", "ENV-100"],
            "feature_families": ["skew", "kurtosis", "downside_vol", "upside_vol", "asymmetry_ratio"],
            "labels": ["LAB-006"],
            "models": ["ridge", "lasso"],
            "evaluation_periods": ["train:2010-2018", "val:2019-2021", "test:2022-2026"],
            "strongest_result": "H-2: ridge IC +0.008 (050), +0.012 (100) over baseline",
            "weakest_result": "H-2: lasso effects inconsistent",
            "statistical_result": "Phase 14.5: H-2 classified as FRAGILE",
            "economic_result": "Not tested at portfolio level",
            "robustness_result": "FRAGILE per Phase 14.5",
            "final_conclusion": "Return asymmetry hypothesis (H-2) showed sign inconsistency across universes",
            "classification": "INCONCLUSIVE"
        },
        {
            "domain_name": "volatility_dynamics",
            "phases_tested": [14.5],
            "datasets": ["DS-EXP-050", "DS-EXP-100"],
            "universes": ["ENV-050", "ENV-100"],
            "feature_families": ["vol_of_vol", "vol_change", "regime_vol", "vol_persistence"],
            "labels": ["LAB-006"],
            "models": ["ridge", "lasso"],
            "evaluation_periods": ["train:2010-2018", "val:2019-2021", "test:2022-2026"],
            "strongest_result": "H-4: ridge IC +0.015 (050), +0.018 (100)",
            "weakest_result": "H-4: effects regime-dependent",
            "statistical_result": "Phase 14.5: H-4 classified as FRAGILE",
            "economic_result": "Not tested at portfolio level",
            "robustness_result": "FRAGILE per Phase 14.5",
            "final_conclusion": "Volatility dynamics hypothesis (H-4) classified as fragile",
            "classification": "INCONCLUSIVE"
        },
        {
            "domain_name": "macro_regime_information",
            "phases_tested": [14.5, 15, 15.1, 15.2, 16],
            "datasets": ["DS-EXP-050", "DS-EXP-100", "BENCH-001", "DS-000003"],
            "universes": ["ENV-050", "ENV-100"],
            "feature_families": ["fed_funds_rate", "unemployment", "cpi_yoy", "dff_change_3m"],
            "labels": ["LAB-006"],
            "models": ["ridge", "lasso"],
            "evaluation_periods": ["train:2010-2018", "val:2019-2021", "test:2022-2026"],
            "strongest_result": "Phase 14.5/15.2: H3-RIDGE-100 IC=0.1316, H3-LASSO-100 IC=0.1464",
            "weakest_result": "Phase 15.2: temporal instability (val IC negative, test IC positive)",
            "statistical_result": "Phase 15.2: orthogonalized REP-D IC up to 0.1848, but CLIFF sensitivity on all 4 macro features",
            "economic_result": "Phase 16: best Sharpe 0.405 vs equal-weight baseline 0.389 (delta +0.016)",
            "robustness_result": "Phase 15.2: RESEARCH tier; Phase 16: D/RED",
            "final_conclusion": "H-3 macro regime is the strongest signal found but suffers from temporal instability, regime dependence, and fails to produce economically meaningful portfolio alpha",
            "classification": "PROMISING_BUT_UNVALIDATED"
        },
        {
            "domain_name": "portfolio_construction_variants",
            "phases_tested": [16],
            "datasets": ["DS-EXP-050", "DS-EXP-100"],
            "universes": ["ENV-050", "ENV-100"],
            "feature_families": ["equal_weight_top10/20/30", "rank_proportional", "score_proportional", "capped_score"],
            "labels": ["LAB-006"],
            "models": ["ridge", "lasso"],
            "evaluation_periods": ["test:2022-2026 (validation missing)"],
            "strongest_result": "H3-LASSO-050 RP_TOP20 Sharpe=0.405",
            "weakest_result": "Multiple configurations negative Sharpe; equal-weight baseline=0.389",
            "statistical_result": "No configuration robustly beat naive equal-weight across universes",
            "economic_result": "Incremental Sharpe +0.016 vs baseline; costs reduce returns 1-5% annually",
            "robustness_result": "Method-dependent, universe-dependent, temporal validation missing",
            "final_conclusion": "Portfolio construction does not transform weak predictive signal into robust economic alpha",
            "classification": "PARTIALLY_EXPLORED"
        }
    ]
    return domains

# =====================================================================
# STEP 2: FAILURE MODE DECOMPOSITION
# =====================================================================

def build_failure_modes():
    """Classify every major failure from Phases 9-16."""
    
    failures = [
        {
            "failure_id": "F01",
            "description": "OHLCV technical features produce statistically significant but economically negligible ICs that fail to generalize across universes",
            "primary_mode": "WEAK_INFORMATION",
            "secondary_modes": ["UNIVERSE_DEPENDENCE", "MODEL_DEPENDENCE"],
            "evidence": "Phase 9-13C: ridge ICs 0.02-0.03 on ENV-DEV20; failed on ENV-050/100; nonlinear models reversed sign",
            "confidence": 0.9,
            "alternative_explanations": ["Target/horizon mismatch for technical features", "Insufficient feature engineering"],
            "distinguishable_with_additional_testing": False,
            "notes": "Multiple independent tests across phases consistently show no robust signal"
        },
        {
            "failure_id": "F02",
            "description": "Market context and sector context features show in-sample significance but complete OOS failure",
            "primary_mode": "FEATURE_FRAGILITY",
            "secondary_modes": ["INSUFFICIENT_EVALUATION_DIVERSITY"],
            "evidence": "Phase 10-11: features significant in train/val but sign-reversed or zero on test",
            "confidence": 0.85,
            "alternative_explanations": ["Regime change in 2019-2021", "Data leakage in feature construction"],
            "distinguishable_with_additional_testing": True,
            "notes": "Could test with walk-forward validation across more regimes"
        },
        {
            "failure_id": "F03",
            "description": "Point-in-time fundamental features (valuation, growth, leverage) produced small ICs that did not survive label correction from LAB-005 to LAB-006",
            "primary_mode": "TARGET_MISMATCH",
            "secondary_modes": ["WEAK_INFORMATION", "IMPLEMENTATION_LIMITATION"],
            "evidence": "Phase 12D/12E: ICs ~0.03-0.04; Phase 13C: LAB-005 defect; Phase 13C: LAB-006 correction eliminated candidates",
            "confidence": 0.8,
            "alternative_explanations": ["Fundamental data frequency (quarterly) mismatched to 5-day horizon", "Survivorship bias in PIT data"],
            "distinguishable_with_additional_testing": True,
            "notes": "Quarterly fundamentals vs daily prediction horizon is a structural mismatch"
        },
        {
            "failure_id": "F04",
            "description": "H-1 (path structure), H-2 (return asymmetry), H-4 (volatility dynamics) all classified as FRAGILE in Phase 14.5 - sign inconsistent across universes",
            "primary_mode": "UNIVERSE_DEPENDENCE",
            "secondary_modes": ["REGIME_DEPENDENCE", "FEATURE_FRAGILITY"],
            "evidence": "Phase 14.5 robustness.json: H-1, H-2, H-4 all FRAGILE; H-3 PARTIALLY_ROBUST",
            "confidence": 0.85,
            "alternative_explanations": ["Insufficient sample per universe", "Feature construction not PIT-correct"],
            "distinguishable_with_additional_testing": True,
            "notes": "All three failed the same way - sign flip between 050 and 100 universes"
        },
        {
            "failure_id": "F05",
            "description": "H-3 macro regime: high ICs (0.09-0.15) but temporal instability - validation IC negative, test IC positive",
            "primary_mode": "TEMPORAL_INSTABILITY",
            "secondary_modes": ["REGIME_DEPENDENCE", "HORIZON_MISMATCH"],
            "evidence": "Phase 15.2 temporal.json: val IC -0.16 to -0.04, test IC +0.09 to +0.15; Phase 16: no validation period predictions",
            "confidence": 0.95,
            "alternative_explanations": ["COVID-19 regime shift in validation period", "Macro features only predictive in post-2022 inflationary regime"],
            "distinguishable_with_additional_testing": True,
            "notes": "Critical failure - model appears to capture regime-specific effects, not persistent signal"
        },
        {
            "failure_id": "F06",
            "description": "H-3 macro features exhibit CLIFF sensitivity (all 4 features) and high collinearity (r=0.818)",
            "primary_mode": "FEATURE_FRAGILITY",
            "secondary_modes": ["COLLINEARITY"],
            "evidence": "Phase 15.2 cliff.json: all 4 macro features CLIFF; collinearity.json: max r=0.818",
            "confidence": 0.9,
            "alternative_explanations": ["Low variance of macro features makes them fragile to perturbations", "Standardization amplifies noise"],
            "distinguishable_with_additional_testing": False,
            "notes": "Structural property of low-frequency macro data vs high-frequency prediction"
        },
        {
            "failure_id": "F07",
            "description": "Phase 16 portfolio construction: best H-3 portfolio Sharpe 0.405 vs equal-weight baseline 0.389 (delta +0.016)",
            "primary_mode": "PORTFOLIO_TRANSLATION_FAILURE",
            "secondary_modes": ["WEAK_INFORMATION"],
            "evidence": "Phase 16 results: 24 configurations tested; none robustly beat equal-weight; costs reduce returns",
            "confidence": 0.9,
            "alternative_explanations": ["Portfolio methods not optimal", "Transaction cost model too conservative"],
            "distinguishable_with_additional_testing": True,
            "notes": "Even with multiple portfolio methods, signal does not translate to economic alpha"
        },
        {
            "failure_id": "F08",
            "description": "No validation period predictions available in Phase 16 - models only trained on train, predicted on test",
            "primary_mode": "IMPLEMENTATION_LIMITATION",
            "secondary_modes": ["INSUFFICIENT_EVALUATION_DIVERSITY"],
            "evidence": "Phase 16 temporal_stability.json: all val periods show NO_DATA",
            "confidence": 1.0,
            "alternative_explanations": ["Design choice to simplify implementation"],
            "distinguishable_with_additional_testing": True,
            "notes": "This is a fixable implementation gap, not a fundamental failure"
        },
        {
            "failure_id": "F09",
            "description": "Lasso degeneracy at alpha=0.001 (5 models all-zero coefficients) - resolved only at lower alphas",
            "primary_mode": "IMPLEMENTATION_LIMITATION",
            "secondary_modes": ["MODEL_DEPENDENCE"],
            "evidence": "Phase 15.1 lasso_diagnostic.json: 5 DEGENERATE at alpha=0.001; functional at 1e-4, 1e-5",
            "confidence": 0.95,
            "alternative_explanations": ["Alpha chosen without cross-validation", "Feature scaling issue"],
            "distinguishable_with_additional_testing": True,
            "notes": "This is a hyperparameter selection issue, not fundamental"
        },
        {
            "failure_id": "F10",
            "description": "Random Forest and XGBoost consistently degraded or reversed sign vs linear models across Phases 9-13",
            "primary_mode": "MODEL_DEPENDENCE",
            "secondary_modes": ["FEATURE_FRAGILITY", "INSUFFICIENT_DATA_COVERAGE"],
            "evidence": "Phase 11, 13C: nonlinear models failed to generalize; overfit in-sample",
            "confidence": 0.85,
            "alternative_explanations": ["Sample size insufficient for nonlinear models", "Feature engineering inadequate for trees"],
            "distinguishable_with_additional_testing": True,
            "notes": "Consistent pattern: linear > nonlinear for this data/hypothesis space"
        }
    ]
    return failures

# Save Step 1 and 2
research_map = build_research_map()
save_json("phase16_5_research_map.json", research_map)

failure_modes = build_failure_modes()
save_json("phase16_5_failure_modes.json", failure_modes)

print("Steps 1-2 complete")
print(f"Research domains: {len(research_map)}")
print(f"Failure modes: {len(failure_modes)}")