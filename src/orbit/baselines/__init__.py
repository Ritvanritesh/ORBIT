"""ORBIT Phase 8: Transparent Baseline Strategy Lab.

This package implements the canonical baseline strategies specified in the
ORBIT Phase 8 roadmap. Every strategy produces signal rows that pass through
the same Phase 7 backtester with identical cost assumptions and execution
semantics.

The canonical chain per strategy:

    STRATEGY LOGIC
        ↓
    SIGNAL ROWS (standard format)
        ↓
    PHASE 7 BACKTESTER (canonical chain: market -> fill -> signal -> order -> ledger -> valuation)
        ↓
    AFTER-COST METRICS
        ↓
    BENCHMARK REPORT
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from polars import DataFrame as PlDataFrame

from orbit.backtest import BacktestConfig, Backtester
from orbit.baselines.parameter_grids import get_grid, validate_parameters
from orbit.baselines.runner import run_strategy
from orbit.baselines.registry_integration import (
    register_baseline_experiment,
    record_baseline_result,
    _default_temporal_config,
)
from orbit.baselines.benchmark_report import BenchmarkReport, json_serialize

# Strategy functions re-exported from strategies module
from orbit.baselines.strategies import (
    buy_and_hold_signals,
    equal_weight_signals,
    momentum_signals,
    mean_reversion_signals,
    moving_average_signals,
    random_null_signals,
    volatility_targeted_signals,
)

# Module-level constants
DEFAULT_UNIVERSE = ["INS-000001", "INS-000002"]
INITIAL_CASH = 1_000_000.0


# ---------------------------------------------------------------------------
# Default cost model (roadmap-consistent: spread=2bps, fees=1bps, slippage=2bps)
# ---------------------------------------------------------------------------

def default_cost_model() -> Any:
    """Return the default cost model used across all Phase 8 baselines.

    This is the cost assumption that every baseline strategy must use for
    fair comparison.  The same cost_model instance must be used for every
    strategy comparison so that differences arise only from strategy logic,
    not from hidden cost assumptions.
    """
    from orbit.schemas.common import CostModel as _CM
    return _CM(spread_bps=2.0, fees_bps=1.0, slippage_bps=2.0)


DEFAULT_COST_MODEL = default_cost_model()
DEFAULT_SEED = 42

__all__ = [
    "buy_and_hold_signals",
    "equal_weight_signals",
    "momentum_signals",
    "mean_reversion_signals",
    "moving_average_signals",
    "volatility_targeted_signals",
    "random_null_signals",
    "run_strategy",
    "register_experiment",
    "BenchmarkReport",
    "json_serialize",
    "DEFAULT_UNIVERSE",
    "INITIAL_CASH",
    "DEFAULT_COST_MODEL",
    "DEFAULT_SEED",
]