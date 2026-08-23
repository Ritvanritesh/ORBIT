"""Phase 11.1 locked plans and configuration.

Implements:
- Stage A plan (benchmark integration)
- Universe expansion plan (20 -> 50 -> 100)
- Benchmark suite registration

All plans are serialized, hashed, and persisted BEFORE execution.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any


def _sha256_json(obj: Any) -> str:
    raw = json.dumps(obj, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ──────────────────────────────────────────────────────────────
# STAGE A PLAN
# ──────────────────────────────────────────────────────────────

def build_stage_a_plan() -> dict[str, Any]:
    """Build the locked Stage A plan (benchmark integration)."""
    plan = {
        "phase": "11.1",
        "stage": "A",
        "protocol": "phase11_1_stage_a_v1",
        "version": "1.0.0",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "purpose": (
            "Add benchmark data and benchmark-relative labels. "
            "This stage must pass all validation gates before Stage B begins."
        ),
        "benchmark_configuration": {
            "benchmark_id": "BENCH-001",
            "benchmark_symbol": "SPY",
            "benchmark_role": "broad_market",
            "market": "US",
            "currency": "USD",
            "return_definition": "simple_total_return",
            "alignment_policy": "same_day",
            "adjusted_price_policy": "split_continuous",
            "source": "yahoo_chart_api",
            "tradable": False,
        },
        "label_configuration": {
            "label_id": "LAB-005",
            "version": "v1",
            "target_type": "excess_return",
            "horizon": 5,
            "anchor_mode": "decision_instant",
            "return_convention": "simple_total_return",
            "benchmark": "BENCH-001",
        },
        "validation_gates": [
            "benchmark configuration locked",
            "benchmark role explicit",
            "deterministic ingestion",
            "schema valid",
            "source lineage recorded",
            "no lookahead leakage",
            "alignment tested",
            "excess-return labels validated",
            "LAB-004 unchanged",
            "new label artifact created",
            "manifests immutable",
            "DS-000004 unchanged",
            "benchmark-relative metrics reproducible",
            "adversarial tests pass",
            "independent Stage A audit passes",
            "full regression suite passes",
        ],
        "adversarial_tests": [
            "A1: benchmark shifted by one session",
            "A2: future benchmark value leaked into label",
            "A3: missing benchmark observation",
            "A4: existing DS-000004 modified",
            "A5: existing LAB-004 modified",
        ],
    }
    plan["plan_digest"] = _sha256_json({k: v for k, v in plan.items() if k != "plan_digest"})
    return plan


# ──────────────────────────────────────────────────────────────
# UNIVERSE EXPANSION PLAN
# ──────────────────────────────────────────────────────────────

def build_universe_expansion_plan() -> dict[str, Any]:
    """Build the locked universe expansion plan.

    This plan MUST be created BEFORE any model performance from
    the new universe is examined.
    """
    plan = {
        "phase": "11.1",
        "protocol": "phase11_1_universe_v1",
        "version": "1.0.0",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "purpose": (
            "Controlled expansion of the tradable universe from ~20 to ~50 to ~100 symbols. "
            "The expansion policy is locked before any model performance is observed."
        ),
        "expansion_stages": {
            "stage_0": {
                "name": "historical_baseline",
                "description": "20 symbols without benchmark context (existing DS-000004)",
                "target_size": 20,
                "dataset_id": "DS-000004",
            },
            "stage_1": {
                "name": "benchmark_context",
                "description": "20 symbols + benchmark context (BENCH-001)",
                "target_size": 20,
                "dataset_id": "DS-000004",
                "benchmark_id": "BENCH-001",
            },
            "stage_2": {
                "name": "expanded_50",
                "description": "~50 symbols + benchmark context",
                "target_size_min": 45,
                "target_size_max": 55,
            },
            "stage_3": {
                "name": "expanded_100",
                "description": "~100 symbols + benchmark context",
                "target_size_min": 90,
                "target_size_max": 110,
            },
        },
        "selection_policy": {
            "method": "deterministic_rule_based",
            "criteria": [
                "US-listed equities",
                "minimum 5 years of trading history by selection date",
                "minimum average daily dollar volume > $10M (trailing 200 sessions)",
                "must have point-in-time data available",
                "no OTC/pink-sheet securities",
            ],
            "exclusion_criteria": [
                "securities with < 5 years history",
                "securities with < $10M average daily dollar volume",
                "securities without point-in-time data",
                "OTC/pink-sheet securities",
            ],
            "selection_date": "2026-01-01",
            "lookback_window": 200,
            "min_history_years": 5,
        },
        "sector_representation": {
            "method": "broad_market_coverage",
            "description": (
                "Selection follows deterministic liquidity and history rules. "
                "Sector diversity is a natural consequence, not a hard constraint."
            ),
        },
        "survivorship_bias": {
            "status": "NOT FULLY CONTROLLED",
            "description": (
                "The development universe is constructed from currently available "
                "instruments with sufficient history. Historical delistings that "
                "occurred before the selection date may not be fully represented. "
                "This limitation is explicitly documented."
            ),
        },
        "identity_mapping": {
            "method": "canonical_ins_id",
            "description": (
                "New instruments receive INS-XXXXXX IDs following ORBIT conventions. "
                "Ticker changes are tracked via SymbolHistory."
            ),
        },
        "gate_between_50_and_100": {
            "permitted_criteria": [
                "infrastructure validity",
                "data quality",
                "identity integrity",
                "universe-plan compliance",
                "reproducibility",
            ],
            "forbidden_criteria": [
                "model performance",
                "returns",
                "IC",
                "statistical significance",
            ],
        },
    }
    plan["plan_digest"] = _sha256_json({k: v for k, v in plan.items() if k != "plan_digest"})
    return plan


# ──────────────────────────────────────────────────────────────
# BENCHMARK SUITE
# ──────────────────────────────────────────────────────────────

def build_benchmark_suite() -> dict[str, Any]:
    """Build the locked benchmark suite for Phase 11.1.

    This suite defines the EXACT set of experiments to run across
    all universe sizes. It is locked before execution.
    """
    suite = {
        "phase": "11.1",
        "protocol": "phase11_1_benchmark_suite_v1",
        "version": "1.0.0",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "description": (
            "Locked benchmark suite for Phase 11.1. Runs ridge, lasso, "
            "random_forest, and xgboost on FS-001 and FS-003 feature sets, "
            "evaluating both absolute (LAB-004) and excess-return (LAB-005) labels."
        ),
        "models": [
            {"family": "ridge", "params": {"alpha": 1.0}},
            {"family": "lasso", "params": {"alpha": 0.001}},
            {"family": "random_forest", "params": {"max_depth": 3, "n_estimators": 200}},
            {"family": "xgboost", "params": {"learning_rate": 0.1, "max_depth": 3, "n_estimators": 200}},
        ],
        "feature_sets": [
            {"feature_set_id": "FS-001", "role": "base", "n_features": 8},
            {"feature_set_id": "FS-003", "role": "all", "n_features": 23},
        ],
        "labels": [
            {"label_id": "LAB-004", "version": "v1", "type": "forward_return"},
            {"label_id": "LAB-005", "version": "v1", "type": "excess_return"},
        ],
        "windows": {
            "train": "2010-01-04..2018-12-31",
            "val": "2019-01-02..2021-12-31",
            "test": "2022-01-03..2026-06-30",
        },
        "cost_model": {"spread_bps": 2, "fees_bps": 1, "slippage_bps": 2},
        "seed": 42,
        "signal_construction": "top-3 long, equal weight 1/3",
        "evaluation_environments": [
            {"env_id": "ENV-1", "description": "Historical 20-symbol baseline", "dataset_id": "DS-000004", "benchmark_id": None},
            {"env_id": "ENV-2", "description": "20 symbols + benchmark context", "dataset_id": "DS-000004", "benchmark_id": "BENCH-001"},
            {"env_id": "ENV-3", "description": "~50 symbols + benchmark context", "dataset_id": None, "benchmark_id": "BENCH-001"},
            {"env_id": "ENV-4", "description": "~100 symbols + benchmark context", "dataset_id": None, "benchmark_id": "BENCH-001"},
        ],
    }
    suite["suite_digest"] = _sha256_json({k: v for k, v in suite.items() if k != "suite_digest"})
    return suite


# ──────────────────────────────────────────────────────────────
# PERSISTENCE
# ──────────────────────────────────────────────────────────────

_REPO_ROOT = Path(__file__).resolve().parents[3] if False else None  # avoid circular

def _benchmarks_dir():
    from pathlib import Path
    return Path(__file__).resolve().parents[3] / "benchmarks"


def persist_stage_a_plan(plan: dict[str, Any]) -> str:
    import os
    out = _benchmarks_dir() / "phase11_1_stage_a_plan.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(plan, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return str(out)


def persist_universe_plan(plan: dict[str, Any]) -> str:
    out = _benchmarks_dir() / "phase11_1_universe_plan_v1.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(plan, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return str(out)


def persist_benchmark_suite(suite: dict[str, Any]) -> str:
    out = _benchmarks_dir() / "phase11_1_benchmark_suite.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(suite, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return str(out)


def load_and_verify_plan(path: str) -> dict[str, Any]:
    from pathlib import Path as _P
    data = json.loads(_P(path).read_text(encoding="utf-8"))
    stored = data.get("plan_digest") or data.get("suite_digest")
    key = "plan_digest" if "plan_digest" in data else "suite_digest"
    payload = {k: v for k, v in data.items() if k != key}
    computed = _sha256_json(payload)
    if stored != computed:
        raise ValueError(f"Plan digest mismatch: stored={stored[:16]}... computed={computed[:16]}...")
    return data


__all__ = [
    "build_stage_a_plan", "build_universe_expansion_plan", "build_benchmark_suite",
    "persist_stage_a_plan", "persist_universe_plan", "persist_benchmark_suite",
    "load_and_verify_plan",
]
