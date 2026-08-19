"""Small pre-registered parameter grids for Phase 8 baseline strategies.

The purpose is controlled comparison, not optimization.  Grids are explicit,
small, deterministic, and registered with each experiment.

Grids follow the roadmap requirement: "small pre-parameter grids" - no huge
sweeps, no automatic hyperparameter optimization, no genetic search, no
Bayesian optimization, no repeated tuning against the same evaluation result.
"""

from __future__ import annotations

from typing import Any

from orbit.schemas.common import CostModel


# ---------------------------------------------------------------------------
# Buy-and-hold: no parameters (trivially one configuration)
# ---------------------------------------------------------------------------

BH_GRID: dict[str, Any] = {
    "strategy": "buy_and_hold",
    "parameters": {},
    "description": "One configuration: always hold the asset.",
}


# ---------------------------------------------------------------------------
# Equal weight: no free parameters (rebalance frequency is explicit:
# session-by-session or one-time, documented in the strategy function)
# ---------------------------------------------------------------------------

EW_GRID: dict[str, Any] = {
    "strategy": "equal_weight",
    "parameters": {
        "rebalance": True,  # session-by-session; documented choice
    },
    "description": "Equal allocation across universe; rebalance=True means session-by-session rebalancing.",
}


# ---------------------------------------------------------------------------
# Momentum: small pre-registered lookback grid
# ---------------------------------------------------------------------------

MOMENTUM_GRID: dict[str, Any] = {
    "strategy": "momentum",
    "parameters": {
        "lookback_days": [10, 20, 30],  # small grid, no optimization
    },
    "description": "Rank instruments by total return over lookback window.",
}


# ---------------------------------------------------------------------------
# Mean reversion: small pre-registered lookback grid
# ---------------------------------------------------------------------------

MEAN_REVERSION_GRID: dict[str, Any] = {
    "strategy": "mean_reversion",
    "parameters": {
        "lookback_days": [10, 20, 30],  # small grid, no optimization
    },
    "description": "Distance from rolling mean; signal direction based on deviation.",
}


# ---------------------------------------------------------------------------
# Moving-average rule: exactly 3 pre-registered parameter combinations
# ---------------------------------------------------------------------------

MOVING_AVERAGE_GRID: dict[str, Any] = {
    "strategy": "moving_average",
    "parameters": [
        {"short_window": 5, "long_window": 30},
        {"short_window": 10, "long_window": 30},
        {"short_window": 15, "long_window": 40},
    ],
    "description": "3 pre-registered SMA crossover combinations. No tuning against final results.",
}


# ---------------------------------------------------------------------------
# Volatility-targeted control: small pre-registered grid
# ---------------------------------------------------------------------------

VOLATILITY_TARGETED_GRID: dict[str, Any] = {
    "strategy": "volatility_targeted",
    "parameters": {
        "target_volatility": [0.10, 0.15, 0.20],
        "estimation_window": [10, 30, 60],
    },
    "description": "3x3 = 9 pre-registered combinations. Volatility estimation uses only past closes.",
}


# ---------------------------------------------------------------------------
# Random/null control: seed is the only parameter; grid is {seed: 42}
# ---------------------------------------------------------------------------

RANDOM_NULL_GRID: dict[str, Any] = {
    "strategy": "random_null",
    "parameters": {
        "seed": [42],  # single seeded trial for reproducibility
    },
    "description": "Single seeded random trial. Multiple trials keep number small and explicitly recorded.",
}


# ---------------------------------------------------------------------------
# Grid registry: look up grid by strategy name
# ---------------------------------------------------------------------------

def get_grid(strategy_name: str) -> dict[str, Any] | None:
    """Return the pre-registered parameter grid for *strategy_name*, or None."""
    grids: dict[str, dict[str, Any]] = {
        "buy_and_hold": BH_GRID,
        "equal_weight": EW_GRID,
        "momentum": MOMENTUM_GRID,
        "mean_reversion": MEAN_REVERSION_GRID,
        "moving_average": MOVING_AVERAGE_GRID,
        "volatility_targeted": VOLATILITY_TARGETED_GRID,
        "random_null": RANDOM_NULL_GRID,
    }
    return grids.get(strategy_name)


# ---------------------------------------------------------------------------
# Parameter validation: ensure a given parameter set belongs to the grid
# ---------------------------------------------------------------------------

def validate_parameters(strategy_name: str, parameters: dict[str, Any]) -> bool:
    """Return True if *parameters* is an allowed combination for the grid.

    This prevents hidden parameter tuning.  A parameter set is valid only
    if every key is known and every value matches one of the grid entries.
    """
    grid = get_grid(strategy_name)
    if grid is None:
        return False

    params = grid.get("parameters", {})
    if isinstance(params, list):
        # Grid is a list of dicts (moving average crossover combinations)
        return any(_dict_matches(params_entry, parameters) for params_entry in params)
    else:
        # Grid is a dict of {param_name: [allowed_values]} or {param_name: bool_allowed}
        for key, allowed in params.items():
            if key not in parameters:
                return False
            # Handle both list allowed values and scalar (bool/str/number) allowed values
            if isinstance(allowed, list):
                if parameters[key] not in allowed:
                    return False
            else:
                # Scalar allowed value: exact match required
                if parameters[key] != allowed:
                    return False
        # Allow extra keys only if the grid is permissive (none here)
        return True


def _dict_matches(grid_entry: dict[str, Any], parameters: dict[str, Any]) -> bool:
    """Check if a single grid entry matches the given parameters."""
    if set(grid_entry.keys()) != set(parameters.keys()):
        return False
    for key in grid_entry:
        if grid_entry[key] != parameters[key]:
            return False
    return True