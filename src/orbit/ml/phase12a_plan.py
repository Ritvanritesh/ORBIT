"""Phase 12A locked information plan.

This plan defines ALL Phase 12A features, datasets, models, and experiments
before any execution. It is immutable once locked (digest verified).

Phase 12A asks: "Does market regime, sector context, and cross-sectional
information provide predictive information that isolated instrument-level
OHLCV representations do not?"
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]

# =====================================================================
# PHASE 12A FEATURE DEFINITIONS
# =====================================================================

# Family A: Market Regime Context (FEAT-201..FEAT-206)
MARKET_CONTEXT_FEATURES: list[dict[str, Any]] = [
    {
        "feature_id": "FEAT-201",
        "name": "mkt_ret_5",
        "family": "market_regime",
        "formula": "close(D-1)/close(D-5) - 1 on SPY",
        "window": 5,
        "kind": "momentum_return",
        "source": "BENCH-001 (SPY)",
        "pit_boundary": "window_end_session < decision_session",
    },
    {
        "feature_id": "FEAT-202",
        "name": "mkt_ret_20",
        "family": "market_regime",
        "formula": "close(D-1)/close(D-20) - 1 on SPY",
        "window": 20,
        "kind": "momentum_return",
        "source": "BENCH-001 (SPY)",
        "pit_boundary": "window_end_session < decision_session",
    },
    {
        "feature_id": "FEAT-203",
        "name": "mkt_vol_20",
        "family": "market_regime",
        "formula": "std(daily_returns, 20) on SPY",
        "window": 20,
        "kind": "realized_volatility",
        "source": "BENCH-001 (SPY)",
        "pit_boundary": "window_end_session < decision_session",
    },
    {
        "feature_id": "FEAT-204",
        "name": "mkt_vol_60",
        "family": "market_regime",
        "formula": "std(daily_returns, 60) on SPY",
        "window": 60,
        "kind": "realized_volatility",
        "source": "BENCH-001 (SPY)",
        "pit_boundary": "window_end_session < decision_session",
    },
    {
        "feature_id": "FEAT-205",
        "name": "mkt_trend_20_50",
        "family": "market_regime",
        "formula": "sma(SPY, 20) / sma(SPY, 50) - 1",
        "short": 20,
        "long": 50,
        "kind": "moving_average_ratio",
        "source": "BENCH-001 (SPY)",
        "pit_boundary": "window_end_session < decision_session",
    },
    {
        "feature_id": "FEAT-206",
        "name": "mkt_drawdown_from_peak_60",
        "family": "market_regime",
        "formula": "close(D-1)/max(close, 60) - 1 on SPY",
        "window": 60,
        "kind": "drawdown",
        "source": "BENCH-001 (SPY)",
        "pit_boundary": "window_end_session < decision_session",
    },
]

# Family B: Sector Context (FEAT-211..FEAT-215)
SECTOR_CONTEXT_FEATURES: list[dict[str, Any]] = [
    {
        "feature_id": "FEAT-211",
        "name": "sector_ret_20",
        "family": "sector_context",
        "formula": "mean(instrument_returns, sector, 20)",
        "window": 20,
        "kind": "sector_momentum",
        "source": "instrument returns within sector",
        "pit_boundary": "window_end_session < decision_session",
    },
    {
        "feature_id": "FEAT-212",
        "name": "sector_vol_20",
        "family": "sector_context",
        "formula": "std(daily_returns, sector, 20)",
        "window": 20,
        "kind": "sector_volatility",
        "source": "instrument returns within sector",
        "pit_boundary": "window_end_session < decision_session",
    },
    {
        "feature_id": "FEAT-213",
        "name": "sector_ret_5",
        "family": "sector_context",
        "formula": "mean(instrument_returns, sector, 5)",
        "window": 5,
        "kind": "sector_momentum",
        "source": "instrument returns within sector",
        "pit_boundary": "window_end_session < decision_session",
    },
    {
        "feature_id": "FEAT-214",
        "name": "sector_trend_5_30",
        "family": "sector_context",
        "formula": "sma(sector_return, 5) / sma(sector_return, 30) - 1",
        "short": 5,
        "long": 30,
        "kind": "sector_trend",
        "source": "instrument returns within sector",
        "pit_boundary": "window_end_session < decision_session",
    },
    {
        "feature_id": "FEAT-215",
        "name": "sector_dispersion_20",
        "family": "sector_context",
        "formula": "std(sector_member_returns, 20) - mean(std(member_returns, 20))",
        "window": 20,
        "kind": "sector_dispersion",
        "source": "instrument returns within sector",
        "pit_boundary": "window_end_session < decision_session",
    },
]

# Family C: Cross-Sectional (FEAT-221..FEAT-225)
CROSS_SECTIONAL_FEATURES: list[dict[str, Any]] = [
    {
        "feature_id": "FEAT-221",
        "name": "xs_rank_ret_20",
        "family": "cross_sectional",
        "formula": "percentile_rank(ret_20, universe, D)",
        "window": 20,
        "kind": "cross_sectional_rank",
        "source": "instrument ret_20 within universe",
        "pit_boundary": "window_end_session < decision_session",
        "min_population": 5,
    },
    {
        "feature_id": "FEAT-222",
        "name": "xs_rank_vol_10",
        "family": "cross_sectional",
        "formula": "percentile_rank(vol_10, universe, D)",
        "window": 20,
        "kind": "cross_sectional_rank",
        "source": "instrument vol_20 within universe",
        "pit_boundary": "window_end_session < decision_session",
        "min_population": 5,
    },
    {
        "feature_id": "FEAT-223",
        "name": "xs_ret_vs_median_20",
        "family": "cross_sectional",
        "formula": "ret_20(D) - median(ret_20, universe, D)",
        "window": 20,
        "kind": "cross_sectional_relative",
        "source": "instrument ret_20 within universe",
        "pit_boundary": "window_end_session < decision_session",
        "min_population": 5,
    },
    {
        "feature_id": "FEAT-224",
        "name": "xs_ret_vs_mean_20",
        "family": "cross_sectional",
        "formula": "ret_20(D) - mean(ret_20, universe, D)",
        "window": 20,
        "kind": "cross_sectional_relative",
        "source": "instrument ret_20 within universe",
        "pit_boundary": "window_end_session < decision_session",
        "min_population": 5,
    },
    {
        "feature_id": "FEAT-225",
        "name": "xs_dispersion_ret_20",
        "family": "cross_sectional",
        "formula": "std(ret_20, universe, D)",
        "window": 20,
        "kind": "cross_sectional_dispersion",
        "source": "instrument ret_20 within universe",
        "pit_boundary": "window_end_session < decision_session",
        "min_population": 5,
    },
]

ALL_PHASE12A_DEFINITIONS = (
    MARKET_CONTEXT_FEATURES + SECTOR_CONTEXT_FEATURES + CROSS_SECTIONAL_FEATURES
)

PHASE12A_FEATURE_NAMES: dict[str, str] = {
    f["feature_id"]: f["name"] for f in ALL_PHASE12A_DEFINITIONS
}

# =====================================================================
# FEATURE SET DEFINITIONS (ABLADESIGN)
# =====================================================================

# FS-001: Existing baseline (8 features, frozen)
# FS-101: Baseline + market context
# FS-102: Baseline + sector context
# FS-103: Baseline + cross-sectional context
# FS-104: Baseline + all Phase 12A context

PHASE12A_FEATURE_SETS: dict[str, dict[str, Any]] = {
    "FS-101": {
        "feature_set_id": "FS-101",
        "feature_set_version": "v1",
        "description": "Baseline OHLCV (FS-001) + market regime context",
        "role": "market_context",
        "feature_refs": ["FEAT-001", "FEAT-002", "FEAT-003", "FEAT-004",
                         "FEAT-005", "FEAT-006", "FEAT-007", "FEAT-008",
                         "FEAT-201", "FEAT-202", "FEAT-203", "FEAT-204",
                         "FEAT-205", "FEAT-206"],
        "n_features": 14,
        "families": ["baseline_ohlcv", "market_regime"],
    },
    "FS-102": {
        "feature_set_id": "FS-102",
        "feature_set_version": "v1",
        "description": "Baseline OHLCV (FS-001) + sector context",
        "role": "sector_context",
        "feature_refs": ["FEAT-001", "FEAT-002", "FEAT-003", "FEAT-004",
                         "FEAT-005", "FEAT-006", "FEAT-007", "FEAT-008",
                         "FEAT-211", "FEAT-212", "FEAT-213", "FEAT-214",
                         "FEAT-215"],
        "n_features": 13,
        "families": ["baseline_ohlcv", "sector_context"],
    },
    "FS-103": {
        "feature_set_id": "FS-103",
        "feature_set_version": "v1",
        "description": "Baseline OHLCV (FS-001) + cross-sectional context",
        "role": "cross_sectional",
        "feature_refs": ["FEAT-001", "FEAT-002", "FEAT-003", "FEAT-004",
                         "FEAT-005", "FEAT-006", "FEAT-007", "FEAT-008",
                         "FEAT-221", "FEAT-222", "FEAT-223", "FEAT-224",
                         "FEAT-225"],
        "n_features": 13,
        "families": ["baseline_ohlcv", "cross_sectional"],
    },
    "FS-104": {
        "feature_set_id": "FS-104",
        "feature_set_version": "v1",
        "description": "Baseline OHLCV + all Phase 12A context families",
        "role": "all_context",
        "feature_refs": ["FEAT-001", "FEAT-002", "FEAT-003", "FEAT-004",
                         "FEAT-005", "FEAT-006", "FEAT-007", "FEAT-008",
                         "FEAT-201", "FEAT-202", "FEAT-203", "FEAT-204",
                         "FEAT-205", "FEAT-206",
                         "FEAT-211", "FEAT-212", "FEAT-213", "FEAT-214",
                         "FEAT-215",
                         "FEAT-221", "FEAT-222", "FEAT-223", "FEAT-224",
                         "FEAT-225"],
        "n_features": 23,
        "families": ["baseline_ohlcv", "market_regime", "sector_context", "cross_sectional"],
    },
}

# =====================================================================
# DATASET ENVIRONMENTS
# =====================================================================

PHASE12A_ENVIRONMENTS = {
    "ENV-12A-050": {
        "env_id": "ENV-12A-050",
        "description": "50-symbol universe with benchmark and sector context",
        "dataset_id": "DS-EXP-050",
        "benchmark_id": "BENCH-001",
        "n_instruments_target": 50,
        "has_sector_classification": True,
    },
    "ENV-12A-100": {
        "env_id": "ENV-12A-100",
        "description": "100-symbol universe with benchmark and sector context",
        "dataset_id": "DS-EXP-100",
        "benchmark_id": "BENCH-001",
        "n_instruments_target": 100,
        "has_sector_classification": True,
    },
}

# =====================================================================
# MODEL CONFIGURATION (LOCKED - same as Phase 10/11)
# =====================================================================

PHASE12A_MODEL_POINTS = [
    {"family": "ridge", "params": {"alpha": 1.0}, "phase10_parent": "ridge+FS-003"},
    {"family": "lasso", "params": {"alpha": 0.001}, "phase10_parent": "lasso+FS-003"},
    {"family": "random_forest", "params": {"n_estimators": 200, "max_depth": 3}, "phase10_parent": "rf+FS-003"},
    {"family": "xgboost", "params": {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.1}, "phase10_parent": "xgb+FS-003"},
]

# =====================================================================
# WINDOWS (LOCKED - same as Phase 9/10/11)
# =====================================================================

PHASE12A_WINDOWS = {
    "train": ("2010-01-04", "2018-12-31"),
    "val": ("2019-01-02", "2021-12-31"),
    "test": ("2022-01-03", "2026-06-30"),
}

# =====================================================================
# LABEL POLICY
# =====================================================================

PHASE12A_LABELS = {
    "LAB-004": {"label_id": "LAB-004", "version": "v1", "type": "forward_return",
                "description": "5-session forward total return (absolute)"},
    "LAB-005": {"label_id": "LAB-005", "version": "v1", "type": "excess_return",
                "description": "5-session benchmark-relative excess return"},
}

# =====================================================================
# CROSS-SECTIONAL CONFIGURATION
# =====================================================================

CROSS_SECTIONAL_CONFIG = {
    "min_universe_population": 5,
    "missing_policy": "drop_session_for_instrument",
    "new_listing_warmup": 20,
    "tie_breaking": "average",
    "universe_source": "instrument_master",
}

# =====================================================================
# SECTOR CONFIGURATION
# =====================================================================

SECTOR_CONFIG = {
    "taxonomy": "GICS_depth1",
    "sector_codes": ["S10", "S20", "S25", "S30", "S35", "S40", "S45", "S50", "S55"],
    "sector_min_members": 2,
    "sector_missing_policy": "drop_sector_features",
    "sector_source": "instrument_master_sector_field",
    "pit_note": "sector membership treated as time-invariant (documented limitation)",
}

# =====================================================================
# COST MODEL (LOCKED)
# =====================================================================

PHASE12A_COST_MODEL = {
    "spread_bps": 2.0,
    "fees_bps": 1.0,
    "slippage_bps": 2.0,
}

# =====================================================================
# INFERENCE PLAN
# =====================================================================

PHASE12A_INFERENCE = {
    "seed": 42,
    "confidence_level": 0.95,
    "bootstrap_resamples": 10000,
    "block_length": "auto_rule_of_thumb",
    "multiple_testing": ["holm_bonferroni", "benjamini_hochberg"],
    "effect_size_thresholds": {
        "ic_small": 0.02,
        "ic_medium": 0.05,
        "ic_large": 0.10,
    },
}

# =====================================================================
# EXPERIMENT GRID
# =====================================================================

def build_phase12a_experiment_grid() -> list[dict[str, Any]]:
    """Build the complete pre-registered experiment grid."""
    experiments = []
    exp_num = 0

    for env_id, env_cfg in PHASE12A_ENVIRONMENTS.items():
        for fs_id, fs_cfg in PHASE12A_FEATURE_SETS.items():
            for model_cfg in PHASE12A_MODEL_POINTS:
                for lab_id, lab_cfg in PHASE12A_LABELS.items():
                    exp_num += 1
                    exp_id = f"EXP-12A-{exp_num:04d}"
                    experiments.append({
                        "experiment_id": exp_id,
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
                        "windows": PHASE12A_WINDOWS,
                        "cost_model": PHASE12A_COST_MODEL,
                        "seed": PHASE12A_INFERENCE["seed"],
                    })

    return experiments


def _sha256_json(obj: Any) -> str:
    """Deterministic SHA-256 of a JSON-serializable object."""
    canonical = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def build_phase12a_plan() -> dict[str, Any]:
    """Build and lock the complete Phase 12A plan."""
    experiments = build_phase12a_experiment_grid()

    plan = {
        "phase": "12A",
        "version": "v1",
        "created_at": datetime.now().isoformat(),
        "research_question": (
            "Does adding market regime, sector context, and cross-sectional "
            "information provide predictive information that isolated "
            "instrument-level OHLCV representations do not?"
        ),
        "information_families": {
            "A_market_regime": {
                "features": MARKET_CONTEXT_FEATURES,
                "n_features": len(MARKET_CONTEXT_FEATURES),
                "source": "BENCH-001 (SPY)",
            },
            "B_sector_context": {
                "features": SECTOR_CONTEXT_FEATURES,
                "n_features": len(SECTOR_CONTEXT_FEATURES),
                "source": "instrument returns within sector",
            },
            "C_cross_sectional": {
                "features": CROSS_SECTIONAL_FEATURES,
                "n_features": len(CROSS_SECTIONAL_FEATURES),
                "source": "instrument features within universe",
            },
        },
        "feature_sets": PHASE12A_FEATURE_SETS,
        "environments": PHASE12A_ENVIRONMENTS,
        "models": PHASE12A_MODEL_POINTS,
        "labels": PHASE12A_LABELS,
        "windows": PHASE12A_WINDOWS,
        "cost_model": PHASE12A_COST_MODEL,
        "cross_sectional_config": CROSS_SECTIONAL_CONFIG,
        "sector_config": SECTOR_CONFIG,
        "inference": PHASE12A_INFERENCE,
        "experiments": experiments,
        "n_experiments": len(experiments),
        "n_environments": len(PHASE12A_ENVIRONMENTS),
        "n_feature_sets": len(PHASE12A_FEATURE_SETS),
        "n_models": len(PHASE12A_MODEL_POINTS),
        "n_labels": len(PHASE12A_LABELS),
    }

    plan["plan_digest"] = _sha256_json({k: v for k, v in plan.items() if k != "plan_digest"})
    return plan


def persist_phase12a_plan(plan: dict[str, Any]) -> Path:
    """Persist the locked plan to disk."""
    out_dir = REPO_ROOT / "benchmarks"
    out_dir.mkdir(exist_ok=True)
    path = out_dir / "phase12a_plan.json"
    path.write_text(json.dumps(plan, indent=2, default=str), encoding="utf-8")
    return path


def load_phase12a_plan() -> dict[str, Any]:
    """Load the locked plan from disk."""
    path = REPO_ROOT / "benchmarks" / "phase12a_plan.json"
    return json.loads(path.read_text(encoding="utf-8"))
