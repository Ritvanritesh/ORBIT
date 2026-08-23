"""Phase 12B locked information plan.

Phase 12B asks: "Does point-in-time fundamental information provide
predictive evidence beyond price, volume, market-context, sector-context,
and cross-sectional information already tested in ORBIT?"

This plan is immutable once locked (digest verified).
"""

from __future__ import annotations
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]

# =====================================================================
# PHASE 12B FUNDAMENTAL DEFINITIONS
# =====================================================================

# Family A: Valuation (FEAT-301..FEAT-303)
VALUATION_FEATURES: list[dict[str, Any]] = [
    {
        "feature_id": "FEAT-301",
        "name": "earnings_yield",
        "family": "valuation",
        "formula": "EPS(D) / price(D)",
        "window": None,
        "kind": "valuation",
        "source": "SEC EDGAR income statement",
        "period_type": "fiscal_quarter",
        "availability_lag": "filing_after_market_close",
    },
    {
        "feature_id": "FEAT-302",
        "name": "book_to_market",
        "family": "valuation",
        "formula": "book_value_share(D) / price(D)",
        "window": None,
        "kind": "valuation",
        "source": "SEC EDGAR balance sheet",
        "period_type": "fiscal_year",
        "availability_lag": "filing_after_market_close",
    },
    {
        "feature_id": "FEAT-303",
        "name": "sales_to_price",
        "family": "valuation",
        "formula": "trailing_12m_sales / price(D)",
        "window": None,
        "kind": "valuation",
        "source": "SEC EDGAR income statement",
        "period_type": "fiscal_year",
        "availability_lag": "filing_after_market_close",
    },
]

# Family B: Profitability / Quality (FEAT-311..FEAT-314)
PROFITABILITY_FEATURES: list[dict[str, Any]] = [
    {
        "feature_id": "FEAT-311",
        "name": "roa",
        "family": "profitability",
        "formula": "net_income / total_assets",
        "window": None,
        "kind": "profitability",
        "source": "SEC EDGAR income statement + balance sheet",
        "period_type": "fiscal_year",
        "availability_lag": "filing_after_market_close",
    },
    {
        "feature_id": "FEAT-312",
        "name": "roe",
        "family": "profitability",
        "formula": "net_income / shareholders_equity",
        "window": None,
        "kind": "profitability",
        "source": "SEC EDGAR income statement + equity",
        "period_type": "fiscal_year",
        "availability_lag": "filing_after_market_close",
    },
    {
        "feature_id": "FEAT-313",
        "name": "operating_margin",
        "family": "profitability",
        "formula": "operating_income / revenue",
        "window": None,
        "kind": "profitability",
        "source": "SEC EDGAR income statement",
        "period_type": "fiscal_year",
        "availability_lag": "filing_after_market_close",
    },
    {
        "feature_id": "FEAT-314",
        "name": "gross_profitability",
        "family": "profitability",
        "formula": "gross_profit / revenue",
        "window": None,
        "kind": "profitability",
        "source": "SEC EDGAR income statement",
        "period_type": "fiscal_year",
        "availability_lag": "filing_after_market_close",
    },
]

# Family C: Growth (FEAT-321..FEAT-323)
GROWTH_FEATURES: list[dict[str, Any]] = [
    {
        "feature_id": "FEAT-321",
        "name": "revenue_growth",
        "family": "growth",
        "formula": "(revenue(D) - revenue(D-1)) / revenue(D-1)",
        "window": None,
        "kind": "growth",
        "source": "SEC EDGAR income statement",
        "period_type": "fiscal_year",
        "availability_lag": "filing_after_market_close",
    },
    {
        "feature_id": "FEAT-322",
        "name": "earnings_growth",
        "family": "growth",
        "formula": "(EPS(D) - EPS(D-1)) / EPS(D-1)",
        "window": None,
        "kind": "growth",
        "source": "SEC EDGAR income statement",
        "period_type": "fiscal_year",
        "availability_lag": "filing_after_market_close",
    },
    {
        "feature_id": "FEAT-323",
        "name": "cash_flow_growth",
        "family": "growth",
        "formula": "(operating_cash_flow(D) - operating_cash_flow(D-1)) / operating_cash_flow(D-1)",
        "window": None,
        "kind": "growth",
        "source": "SEC EDGAR cash flow statement",
        "period_type": "fiscal_year",
        "availability_lag": "filing_after_market_close",
    },
]

# Family D: Leverage / Balance Sheet (FEAT-331..FEAT-333)
LEVERAGE_FEATURES: list[dict[str, Any]] = [
    {
        "feature_id": "FEAT-331",
        "name": "debt_to_equity",
        "family": "leverage",
        "formula": "total_debt / shareholders_equity",
        "window": None,
        "kind": "leverage",
        "source": "SEC EDGAR balance sheet + income statement",
        "period_type": "fiscal_year",
        "availability_lag": "filing_after_market_close",
    },
    {
        "feature_id": "FEAT-332",
        "name": "debt_to_assets",
        "family": "leverage",
        "formula": "total_debt / total_assets",
        "window": None,
        "kind": "leverage",
        "source": "SEC EDGAR balance sheet",
        "period_type": "fiscal_year",
        "availability_lag": "filing_after_market_close",
    },
    {
        "feature_id": "FEAT-333",
        "name": "current_ratio",
        "family": "leverage",
        "formula": "current_assets / current_liabilities",
        "window": None,
        "kind": "leverage",
        "source": "SEC EDGAR balance sheet",
        "period_type": "fiscal_year",
        "availability_lag": "filing_after_market_close",
    },
]

ALL_PHASE12B_DEFINITIONS = (
    VALUATION_FEATURES + PROFITABILITY_FEATURES + GROWTH_FEATURES + LEVERAGE_FEATURES
)

# Map feature_id -> feature name
PHASE12B_FEATURE_NAMES: dict[str, str] = {
    f["feature_id"]: f["name"] for f in ALL_PHASE12B_DEFINITIONS
}

# =====================================================================
# FUNDAMENTAL DATA SOURCE CONFIGURATION
# =====================================================================

# SEC EDGAR company facts source
PHASE12B_SOURCE_SEC_EDGAR = {
    "source_id": "SEC-EDGAR",
    "display_name": "SEC EDGAR Company Facts (via API)",
    "description": "Fundamental data extracted from SEC Form 10-Q/10-K filings",
    "api_endpoint": None,  # Data pre-extracted and stored in repository
    "data_path_template": "data/normalized/fundamentals/sec_edgar_companyfacts/{snapshot_id}/",
    "schema_version": "v1",
    "ingestion_timestamp": "2026-08-21T00:00:00",
    "lineage": "SEC EDGAR full-text index -> companyfacts JSON -> extracted fundamentals",
    "revisions_metadata": True,
    "availability_policy": "earliest_filing_date",  # use first public filing
    "staleness_max_age_years": 2,
}

# Alternative: Yahoo Finance fundamental data (if available via chart API)
PHASE12B_SOURCE_YAHOO_FUNDAMENTAL = {
    "source_id": "YAHOO-FUNDAMENTAL",
    "display_name": "Yahoo Finance Fundamental Data",
    "description": "Fundamental data from Yahoo Finance API (EPS, book value, etc.)",
    "api_endpoint": None,
    "data_path_template": None,
    "schema_version": "v1",
    "ingestion_timestamp": None,
    "lineage": "Yahoo Finance chart API fundamental fields",
    "revisions_metadata": False,
    "availability_policy": "unknown",  # Yahoo does not guarantee PIT availability
    "staleness_max_age_years": None,
}

ACTIVE_FUNDAMENTAL_SOURCE = PHASE12B_SOURCE_SEC_EDGAR

# =====================================================================
# AS-OF JOIN POLICY
# =====================================================================

# For each trading session D, use the latest fundamental observation
# whose public availability timestamp <= D.

# Policy rules:
# - filing_on_trading_day: filing submitted on day D, available at market close D
# - filing_after_market_close: filed after close of day D, available next day open
# - filing_non_trading_day: filed on non-trading day, available at next trading day open
# - missing_availability: if no availability timestamp, feature is null for that session
# - overlapping_revisions: use the revision that was publicly available earliest
# - multiple_filings: prefer the more recent filing with later availability
# - long_gaps: if no new fundamental for > staleness_max_age_years, feature invalidated

AS_OF_POLICY = {
    "availability_boundary": "public_availability_timestamp <= feature_boundary",
    "future_filing_block": True,  # never use future filings
    "revision_policy": "earliest_public_availability",
    "missing_data": "null_invalidated",  # session becomes null if no fundamental available
    "staleness_max_age_years": 2,
    "timezone_assumption": "America/New_York (filing dates in SEC submissions)",
}

# =====================================================================
# FEATURE SET ABLATION DESIGN
# =====================================================================

# SET A: Baseline OHLCV (FS-001, frozen from Phase 9/10/11)
# SET B: Baseline + valuation
# SET C: Baseline + profitability
# SET D: Baseline + growth
# SET E: Baseline + leverage
# SET F: Baseline + all fundamental families (A+B+C+D)

PHASE12B_FEATURE_SETS: dict[str, dict[str, Any]] = {
    "FS-12B-A": {
        "feature_set_id": "FS-12B-A",
        "feature_set_version": "v1",
        "description": "Baseline OHLCV only (FS-001)",
        "role": "baseline",
        "feature_refs": None,  # will use FS-001 refs
        "n_features": 8,
        "families": ["baseline_ohlcv"],
    },
    "FS-12B-B": {
        "feature_set_id": "FS-12B-B",
        "feature_set_version": "v1",
        "description": "Baseline + valuation family (FEAT-301..FEAT-303)",
        "role": "valuation",
        "feature_refs": None,  # populated at build time
        "n_features": 8 + 3,  # 8 baseline + 3 valuation
        "families": ["baseline_ohlcv", "valuation"],
    },
    "FS-12B-C": {
        "feature_set_id": "FS-12B-C",
        "feature_set_version": "v1",
        "description": "Baseline + profitability family (FEAT-311..FEAT-314)",
        "role": "profitability",
        "feature_refs": None,
        "n_features": 8 + 4,
        "families": ["baseline_ohlcv", "profitability"],
    },
    "FS-12B-D": {
        "feature_set_id": "FS-12B-D",
        "feature_set_version": "v1",
        "description": "Baseline + growth family (FEAT-321..FEAT-323)",
        "role": "growth",
        "feature_refs": None,
        "n_features": 8 + 3,
        "families": ["baseline_ohlcv", "growth"],
    },
    "FS-12B-E": {
        "feature_set_id": "FS-12B-E",
        "feature_set_version": "v1",
        "description": "Baseline + leverage family (FEAT-331..FEAT-333)",
        "role": "leverage",
        "feature_refs": None,
        "n_features": 8 + 3,
        "families": ["baseline_ohlcv", "leverage"],
    },
    "FS-12B-F": {
        "feature_set_id": "FS-12B-F",
        "feature_set_version": "v1",
        "description": "Baseline + all fundamental families",
        "role": "all_fundamentals",
        "feature_refs": None,
        "n_features": 8 + 3 + 4 + 3,  # 8 baseline + all fundamental families
        "families": ["baseline_ohlcv", "valuation", "profitability", "growth", "leverage"],
    },
}

# =====================================================================
# DATASET ENVIRONMENTS
# =====================================================================

PHASE12B_ENVIRONMENTS = {
    "ENV-12B-050": {
        "env_id": "ENV-12B-050",
        "description": "50-symbol universe with fundamental data",
        "dataset_id": "DS-EXP-050",
        "fundamental_source": "SEC-EDGAR",
        "n_instruments_target": 50,
    },
    "ENV-12B-100": {
        "env_id": "ENV-12B-100",
        "description": "100-symbol universe with fundamental data",
        "dataset_id": "DS-EXP-100",
        "fundamental_source": "SEC-EDGAR",
        "n_instruments_target": 100,
    },
}

# =====================================================================
# MODEL CONFIGURATION (same as Phase 10/11/12A)
# =====================================================================

PHASE12B_MODEL_POINTS = [
    {"family": "ridge", "params": {"alpha": 1.0}},
    {"family": "lasso", "params": {"alpha": 0.001}},
    {"family": "random_forest", "params": {"n_estimators": 200, "max_depth": 3}},
    {"family": "xgboost", "params": {"learning_rate": 0.1, "max_depth": 3, "n_estimators": 200}},
]

# =====================================================================
# WINDOWS (same as Phase 9/10/11/12A)
# =====================================================================

PHASE12B_WINDOWS = {
    "train": ("2010-01-04", "2018-12-31"),
    "val": ("2019-01-02", "2021-12-31"),
    "test": ("2022-01-03", "2026-06-30"),
}

# =====================================================================
# LABEL POLICY (same as Phase 12A)
# =====================================================================

PHASE12B_LABELS = {
    "LAB-004": {"label_id": "LAB-004", "version": "v1", "type": "forward_return",
                "description": "5-session forward total return (absolute)"},
    "LAB-005": {"label_id": "LAB-005", "version": "v1", "type": "excess_return",
                "description": "5-session benchmark-relative excess return"},
}

# =====================================================================
# STALENESS POLICY
# =====================================================================

PHASE12B_STALENESS = {
    "max_age_years": 2,
    "invalidation_behavior": "null_out_session",  # session becomes null if fundamental older than threshold
    "record_age": "compute_age_from_availability_timestamp",
    "policy_enforced_before": "model_training",
}

# =====================================================================
# INFERENCE PLAN
# =====================================================================

PHASE12B_INFERENCE = {
    "seed": 42,
    "confidence_level": 0.95,
    "bootstrap_resamples": 10000,
    "block_length": "auto_rule_of_thumb",
    "multiple_testing": ["holm_bonferroni", "benjamini_hochberg"],
    "effect_size_thresholds": {
        "small": 0.02,
        "medium": 0.05,
        "large": 0.10,
    },
}

# =====================================================================
# BENCHMARK GRID
# =====================================================================

def build_phase12b_experiment_grid() -> list[dict[str, Any]]:
    """Build the complete pre-registered experiment grid."""
    experiments = []
    exp_num = 0

    for env_id, env_cfg in PHASE12B_ENVIRONMENTS.items():
        for fs_id, fs_cfg in PHASE12B_FEATURE_SETS.items():
            for model_cfg in PHASE12B_MODEL_POINTS:
                for lab_id, lab_cfg in PHASE12B_LABELS.items():
                    exp_num += 1
                    experiments.append({
                        "experiment_id": f"EXP-12B-{exp_num:04d}",
                        "env_id": env_id,
                        "dataset_id": env_cfg["dataset_id"],
                        "feature_set_id": fs_id,
                        "feature_set_version": fs_cfg["feature_set_version"],
                        "n_features": fs_cfg["n_features"],
                        "families": fs_cfg["families"],
                        "family": model_cfg["family"],
                        "params": model_cfg["params"],
                        "label_id": lab_id,
                        "label_type": lab_cfg["type"],
                        "windows": PHASE12B_WINDOWS,
                        "cost_model": {"spread_bps": 2.0, "fees_bps": 1.0, "slippage_bps": 2.0},
                        "seed": PHASE12B_INFERENCE["seed"],
                    })

    return experiments


def _sha256_json(obj: Any) -> str:
    """Deterministic SHA-256 of a JSON-serializable object."""
    canonical = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def build_phase12b_plan() -> dict[str, Any]:
    """Build and lock the complete Phase 12B plan."""
    experiments = build_phase12b_experiment_grid()

    plan = {
        "phase": "12B",
        "version": "v1",
        "created_at": datetime.now().isoformat(),
        "research_question": (
            "Does point-in-time fundamental information provide predictive "
            "evidence beyond the price, volume, market-context, sector-context, "
            "and cross-sectional information already tested in ORBIT?"
        ),
        "fundamental_source": ACTIVE_FUNDAMENTAL_SOURCE,
        "fundamental_definitions": {
            "A_valuation": {
                "features": VALUATION_FEATURES,
                "n_features": len(VALUATION_FEATURES),
            },
            "B_profitability": {
                "features": PROFITABILITY_FEATURES,
                "n_features": len(PROFITABILITY_FEATURES),
            },
            "C_growth": {
                "features": GROWTH_FEATURES,
                "n_features": len(GROWTH_FEATURES),
            },
            "D_leverage": {
                "features": LEVERAGE_FEATURES,
                "n_features": len(LEVERAGE_FEATURES),
            },
        },
        "feature_sets": PHASE12B_FEATURE_SETS,
        "environments": PHASE12B_ENVIRONMENTS,
        "models": PHASE12B_MODEL_POINTS,
        "labels": PHASE12B_LABELS,
        "windows": PHASE12B_WINDOWS,
        "staleness": PHASE12B_STALENESS,
        "inference": PHASE12B_INFERENCE,
        "experiments": experiments,
        "n_experiments": len(experiments),
        "n_environments": len(PHASE12B_ENVIRONMENTS),
        "n_feature_sets": len(PHASE12B_FEATURE_SETS),
        "n_models": len(PHASE12B_MODEL_POINTS),
        "n_labels": len(PHASE12B_LABELS),
    }

    plan["plan_digest"] = _sha256_json({k: v for k, v in plan.items() if k != "plan_digest"})
    return plan


def persist_phase12b_plan(plan: dict[str, Any]) -> Path:
    """Persist the locked plan to disk."""
    out_dir = REPO_ROOT / "benchmarks"
    out_dir.mkdir(exist_ok=True)
    path = out_dir / "phase12b_plan.json"
    path.write_text(json.dumps(plan, indent=2, default=str), encoding="utf-8")
    return path


def load_phase12b_plan() -> dict[str, Any]:
    """Load the locked plan from disk."""
    path = REPO_ROOT / "benchmarks" / "phase12b_plan.json"
    return json.loads(path.read_text(encoding="utf-8"))