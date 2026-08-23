"""Phase 11 locked inference plan (versioned, digest-verified).

The plan defines the EXACT set of analysis tasks Phase 11 is allowed to
perform on the existing Phase 9 and Phase 10 artifacts. It must be
serialized, hashed, and persisted BEFORE any real-data analysis runs.

Any modification after results exist must create a new version rather than
silently changing the old plan.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[3]
PLAN_JSON = _REPO_ROOT / "benchmarks" / "phase11_inference_plan.json"

SEED = 42
CONFIDENCE_LEVEL = 0.95
N_BOOTSTRAP_RESAMPLES = 1000
BLOCK_LENGTH_POLICY = "rule_of_thumb"
BLOCK_LENGTH_DIAGNOSTICS = True

# Economic interpretation thresholds (documented, not arbitrary)
IC_THRESHOLD_NEGIGLIBLE = 0.01
IC_THRESHOLD_MEANINGFUL = 0.03
RETURN_THRESHOLD_NEGIGLIBLE = 0.0  # zero net of costs
RETURN_THRESHOLD_MEANINGFUL = 0.10  # 10% annualized

PHASE10_EXPERIMENT_FAMILY = [f"EXP-{i}" for i in range(10001, 10053)]


def _plan_payload() -> dict[str, Any]:
    return {
        "phase": 11,
        "protocol": "phase11_v1",
        "version": "1.0.0",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "seed": SEED,
        "confidence_level": CONFIDENCE_LEVEL,
        "n_bootstrap_resamples": N_BOOTSTRAP_RESAMPLES,
        "block_length_policy": BLOCK_LENGTH_POLICY,
        "block_length_diagnostics": BLOCK_LENGTH_DIAGNOSTICS,
        "source_experiments": {
            "phase9": {
                "description": "Phase 9 baseline ML benchmark",
                "experiment_ids": "EXP-90001..EXP-90036",
                "artifact_parquet": "benchmarks/phase9_ml_benchmark.parquet",
                "artifact_runs_dir": "benchmarks/phase9_runs/",
            },
            "phase10": {
                "description": "Phase 10 feature ablation grid (13 feature sets x 4 models = 52)",
                "experiment_ids": "EXP-10001..EXP-10052",
                "artifact_parquet": "benchmarks/phase10_feature_research.parquet",
                "artifact_runs_dir": "benchmarks/phase10_runs/",
                "locked_plan": "benchmarks/phase10_plan.json",
                "locked_plan_digest": _read_phase10_plan_digest(),
            },
        },
        "hypothesis_families": {
            "phase10_grid": {
                "description": (
                    "The full 52-experiment Phase 10 grid (13 feature sets x 4 models) "
                    "is treated as a SINGLE locked comparison family. No post-hoc "
                    "exclusion of inconvenient experiments is permitted."
                ),
                "members": PHASE10_EXPERIMENT_FAMILY,
                "n_members": len(PHASE10_EXPERIMENT_FAMILY),
                "rationale": (
                    "Phase 10 is an ablation study over a pre-registered grid. "
                    "All 52 experiments belong to the same researcher search space. "
                    "Treating them as a single family is the conservative choice."
                ),
            }
        },
        "metrics_in_scope": {
            "prediction": ["oos_ic", "rank_ic", "hit_rate", "mse"],
            "calibration": ["brier", "ece"],
            "economic": [
                "after_cost_total_return",
                "turnover",
                "total_costs",
            ],
        },
        "inference_methods": {
            "confidence_intervals": {
                "iid_bootstrap": {
                    "description": "Ordinary bootstrap; valid only for i.i.d. observations",
                    "assumptions": "observations are independent and identically distributed",
                },
                "moving_block_bootstrap": {
                    "description": "Moving block bootstrap with deterministic block length",
                    "assumptions": "stationary time series with short-range dependence",
                    "block_length_policy": BLOCK_LENGTH_POLICY,
                },
            },
            "multiple_testing": {
                "holm_bonferroni": {
                    "description": "Family-wise error rate control (conservative)",
                },
                "benjamini_hochberg": {
                    "description": "False discovery rate control (less conservative)",
                },
            },
            "effect_sizes": {
                "ic_magnitude": "raw IC value",
                "return_magnitude": "raw after-cost return",
                "relative_improvement": "fractional change vs. zero-effect baseline",
            },
        },
        "economic_interpretation_thresholds": {
            "ic_negligible": IC_THRESHOLD_NEGIGLIBLE,
            "ic_meaningful": IC_THRESHOLD_MEANINGFUL,
            "return_negligible": RETURN_THRESHOLD_NEGIGLIBLE,
            "return_meaningful": RETURN_THRESHOLD_MEANINGFUL,
        },
        "known_limitations": [
            "Inference is on a 20-symbol development universe; generalization to larger universes is not tested.",
            "Block length is set by rule of thumb, not optimized against outcomes.",
            "The label (5-session forward return) creates overlapping outcomes; dependence diagnostics will disclose this.",
            "Backtest metrics inherit all backtester assumptions (costs, execution, sizing).",
            "Bootstrap CIs are approximate; they do not provide exact finite-sample coverage.",
            "The inference layer analyzes evidence; it does not generate new signals or strategies.",
        ],
        "statistical_vs_economic_separation": (
            "Every result must separately report: (1) statistical evidence, "
            "(2) effect magnitude, (3) uncertainty bounds, and (4) economic "
            "interpretation. Statistical significance is NEVER treated as "
            "equivalent to deployable trading edge."
        ),
    }


def _read_phase10_plan_digest() -> str:
    """Read the locked Phase 10 plan digest from the stored plan file."""
    plan_path = _REPO_ROOT / "benchmarks" / "phase10_plan.json"
    if plan_path.exists():
        try:
            data = json.loads(plan_path.read_text(encoding="utf-8"))
            return data.get("plan_digest", "UNKNOWN")
        except Exception:
            return "READ_ERROR"
    return "FILE_NOT_FOUND"


def phase11_plan_digest() -> str:
    """sha256 over the full locked plan (identity fingerprint)."""
    raw = json.dumps(_plan_payload(), sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def phase11_plan() -> dict[str, Any]:
    """The complete, locked plan (payload + digest)."""
    payload = _plan_payload()
    payload["plan_digest"] = phase11_plan_digest()
    return payload


def write_plan(plan: dict[str, Any] | None = None) -> Path:
    """Serialize and persist the locked inference plan."""
    if plan is None:
        plan = phase11_plan()
    PLAN_JSON.parent.mkdir(parents=True, exist_ok=True)
    PLAN_JSON.write_text(
        json.dumps(plan, sort_keys=True, indent=2, default=str),
        encoding="utf-8",
    )
    return PLAN_JSON


def load_plan() -> dict[str, Any]:
    """Load and verify the persisted inference plan."""
    if not PLAN_JSON.exists():
        raise FileNotFoundError(f"no inference plan at {PLAN_JSON}")
    data = json.loads(PLAN_JSON.read_text(encoding="utf-8"))
    stored_digest = data.get("plan_digest")
    if stored_digest is None:
        raise ValueError("plan file has no plan_digest field")
    # Re-compute digest from payload (exclude plan_digest itself)
    payload = {k: v for k, v in data.items() if k != "plan_digest"}
    raw = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    computed = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    if computed != stored_digest:
        raise ValueError(
            f"plan digest mismatch: stored={stored_digest[:16]}... "
            f"computed={computed[:16]}..."
        )
    return data


def verify_plan() -> None:
    """Adversarial verification that the plan is intact."""
    data = load_plan()
    if data.get("phase") != 11:
        raise ValueError(f"unexpected phase field: {data.get('phase')}")
    family = data.get("hypothesis_families", {}).get("phase10_grid", {})
    members = family.get("members", [])
    if len(members) != 52:
        raise ValueError(
            f"Phase 10 family has {len(members)} members; expected 52"
        )
    expected_range = [f"EXP-{i}" for i in range(10001, 10053)]
    if sorted(members) != sorted(expected_range):
        raise ValueError("Phase 10 family membership does not match EXP-10001..10052")


__all__ = [
    "SEED",
    "CONFIDENCE_LEVEL",
    "N_BOOTSTRAP_RESAMPLES",
    "BLOCK_LENGTH_POLICY",
    "IC_THRESHOLD_NEGIGLIBLE",
    "IC_THRESHOLD_MEANINGFUL",
    "RETURN_THRESHOLD_NEGIGLIBLE",
    "RETURN_THRESHOLD_MEANINGFUL",
    "PHASE10_EXPERIMENT_FAMILY",
    "PLAN_JSON",
    "phase11_plan",
    "phase11_plan_digest",
    "write_plan",
    "load_plan",
    "verify_plan",
]
