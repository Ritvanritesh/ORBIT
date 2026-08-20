"""Pre-registered hyperparameter grids for the Phase 9 benchmark.

Every model family has a small, fixed grid (documented in
docs/phase9_ml_benchmark.md). A Phase 9 experiment pins exactly ONE
hyperparameter set from its family's grid; different sets are different
experiments, never a post-hoc search. The registry validator rejects any
parameter combination that is not in the grid, so "tuning after seeing
results" is structurally impossible.

Seed: every Phase 9 experiment uses seed 42 (the ORBIT default); a different
seed is a different, separately registered experiment.
"""

from __future__ import annotations

from typing import Any

MODEL_FAMILIES = ("ridge", "lasso", "logistic", "random_forest", "xgboost")

PHASE9_GRIDS: dict[str, list[dict[str, Any]]] = {
    "ridge": [{"alpha": a} for a in (0.01, 0.1, 1.0, 10.0)],
    "lasso": [{"alpha": a} for a in (0.0001, 0.001, 0.01, 0.1)],
    "logistic": [{"C": c} for c in (0.01, 0.1, 1.0, 10.0)],
    "random_forest": [
        {"n_estimators": n, "max_depth": d}
        for n in (50, 200)
        for d in (3, 6)
    ],
    "xgboost": [
        {"n_estimators": n, "max_depth": d, "learning_rate": 0.1}
        for n in (50, 200)
        for d in (3, 6)
    ],
}

SEED = 42

MODEL_PARAM_SCHEMA: dict[str, tuple[str, ...]] = {
    "ridge": ("alpha",),
    "lasso": ("alpha",),
    "logistic": ("C",),
    "random_forest": ("n_estimators", "max_depth"),
    "xgboost": ("n_estimators", "max_depth", "learning_rate"),
}


def validate_model_parameters(family: str, params: dict[str, Any]) -> dict[str, Any]:
    """Validate that `params` is exactly one pre-registered grid point.

    Raises ValueError on unknown family, missing/extra parameters, or a
    combination not present in the family's grid.
    """
    if family not in MODEL_FAMILIES:
        raise ValueError(
            f"unknown model family {family!r}; allowed: {sorted(MODEL_FAMILIES)}"
        )
    expected = MODEL_PARAM_SCHEMA[family]
    if set(params) != set(expected):
        raise ValueError(
            f"family {family!r} requires exactly {expected} parameters; got {sorted(params)}"
        )
    if params not in PHASE9_GRIDS[family]:
        raise ValueError(
            f"parameter set {params} is not a pre-registered {family} grid point; "
            f"grid: {PHASE9_GRIDS[family]}"
        )
    return dict(params)


def grid_points(family: str) -> list[dict[str, Any]]:
    if family not in MODEL_FAMILIES:
        raise ValueError(f"unknown model family {family!r}")
    return [dict(p) for p in PHASE9_GRIDS[family]]


def params_identity(family: str, params: dict[str, Any]) -> str:
    """Short deterministic identity of a grid point (used in signal ids)."""
    return "-".join(f"{k}{v}" for k, v in sorted(validate_model_parameters(family, params).items()))


__all__ = [
    "MODEL_FAMILIES",
    "PHASE9_GRIDS",
    "SEED",
    "MODEL_PARAM_SCHEMA",
    "validate_model_parameters",
    "grid_points",
    "params_identity",
]