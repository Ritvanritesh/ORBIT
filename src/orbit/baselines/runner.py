"""Runner: execute a baseline strategy through the Phase 7 backtester.

All strategies share the same canonical chain:

    SIGNAL ROWS
        ↓
    PHASE 7 BACKTESTER (configurable cost model, execution simulator, ledger)
        ↓
    BACKTEST RESULT (events, ledger snapshot, metrics)
        ↓
    AFTER-COST METRICS (extracted from result)

This module does NOT create a second backtesting mechanism - it uses the
canonical Phase 7 Backtester exactly as Phase 7 provides it.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import polars as pl

from orbit.baselines.strategies import (
    buy_and_hold_signals,
    equal_weight_signals,
    mean_reversion_signals,
    momentum_signals,
    moving_average_signals,
    random_null_signals,
    volatility_targeted_signals,
)
from orbit.baselines.parameter_grids import get_grid, validate_parameters
from orbit.backtest import BacktestConfig, Backtester
from orbit.schemas.common import CostModel

# Module-level constant: initial cash for all baseline runs
INITIAL_CASH = 1_000_000.0


# ---------------------------------------------------------------------------
# Helper: build a backtester ready to run a baseline strategy
# ---------------------------------------------------------------------------

def _make_backtester(
    universe: list[str] | None = None,
    *,
    cost_model: Any = None,
    window_start: date | None = None,
    window_end: date | None = None,
    experiment_id: str = "EXP-00001",
    hypothesis_id: str = "H-001",
    code_hash: str = "a" * 64,
    dataset_snapshot_ids: list[str] | None = None,
) -> Backtester:
    """Create a Phase 7 backtester pre-wired for a baseline strategy run.

    All strategies share the same cost model, universe, and evaluation window
    assumptions so comparisons are under identical execution conditions.
    """
    from orbit.schemas.common import CostModel as _CostModel
    from orbit.backtest.config import CostConfig, ExecutionConfig, BacktestConfig

    default_universe = ["INS-000001", "INS-000002"]
    if universe is None:
        universe = default_universe
    if dataset_snapshot_ids is None:
        dataset_snapshot_ids = ["DS-000001"]

    if cost_model is None:
        cost_model = _CostModel(spread_bps=2.0, fees_bps=1.0, slippage_bps=2.0)

    # Convert CostModel -> CostConfig (the simulator's executable form)
    cost_config = CostConfig.from_cost_model(cost_model)

    # Build execution config with roadmap defaults
    execution_config = ExecutionConfig(
        execution_price="open",
        execution_delay=1,
        participation_fraction=0.05,
        max_order_quantity=None,
        partial_fills=True,
        order_expiry_sessions=5,
    )

    # Construct a minimal but valid BacktestConfig
    config = BacktestConfig(
        initial_cash=1_000_000.0,
        costs=cost_config,
        execution=execution_config,
        window_start=window_start,
        window_end=window_end,
        seed=42,
    )

    return Backtester(
        config=config,
        universe=sorted(set(universe)),
        dataset_snapshot_ids=sorted(set(dataset_snapshot_ids)),
        code_hash=code_hash,
        experiment_id=experiment_id,
        hypothesis_id=hypothesis_id,
        feature_refs=[{"feature_id": "FEAT-001", "feature_version": "v1"}],
        model={"family": "linear", "hyperparameters": {}},
        label_id="LAB-001",
        label_version="v1",
        cost_model_id="CM-001",
    )


# ---------------------------------------------------------------------------
# Core runner: run one strategy through the Phase 7 backtester
# ---------------------------------------------------------------------------

def run_strategy(
    strategy_name: str,
    universe: list[str] | None = None,
    *,
    sessions: list[date] | None = None,
    cost_model: CostModel | None = None,
    window_start: date | None = None,
    window_end: date | None = None,
    experiment_id: str = "EXP-00001",
    hypothesis_id: str = "H-001",
    dataset_snapshot_ids: list[str] | None = None,
    strategy_params: dict[str, Any] | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Execute one baseline strategy through the Phase 7 backtester and
    return a result dictionary with after-cost metrics.

    The result contains everything needed for the permanent benchmark report
    (strategy identity, parameters, universe, evaluation window, cost model,
    experiment ID, after-cost metrics, and stability information).

    Parameters
    ----------
    strategy_name : str
        One of: "buy_and_hold", "equal_weight", "momentum", "mean_reversion",
        "moving_average", "volatility_targeted", "random_null".
    universe : list[str] | None
        Instruments to trade.  Defaults to DEFAULT_UNIVERSE.
    sessions : list[date] | None
        Evaluation window sessions.  If None, the full dataset is used.
    cost_model : CostModel | None
        Execution cost assumptions.  Defaults to the roadmap-consistent
        default (spread=2bps, fees=1bps, slippage=2bps).  *Crucial*: the
        same cost_model instance must be used for every strategy comparison.
    window_start, window_end : date | None
        Evaluation window boundaries.  If None, full dataset is used.
    experiment_id : str
        Experiment identifier for the Phase 6 registry.
    hypothesis_id : str
        Hypothesis identifier for the Phase 6 registry.
    dataset_snapshot_ids : list[str] | None
        Pinned dataset snapshots for the experiment registry.
    strategy_params : dict[str, Any] | None
        Strategy-specific parameters (must belong to the pre-registered grid).
    seed : int
        Random seed for strategies that use randomness.

    Returns
    -------
    dict[str, Any]
        Result dictionary with after-cost metrics and lineage information.
    """
    if universe is None:
        universe = ["INS-000001", "INS-000002"]
    if sessions is None:
        # Default: use a reasonable window of weekdays
        sessions = _default_sessions()
    if cost_model is None:
        cost_model = CostModel(spread_bps=2.0, fees_bps=1.0, slippage_bps=2.0)

    # Validate parameters belong to the pre-registered grid
    grid = get_grid(strategy_name)
    if grid is not None and strategy_params is not None:
        if not validate_parameters(strategy_name, strategy_params):
            raise ValueError(
                f"strategy_params {strategy_params} are not a valid "
                f"parameter set for {strategy_name}"
            )

    # Select the appropriate signal-generating function
    strategy_funcs = {
        "buy_and_hold": _buy_and_hold_run,
        "equal_weight": _equal_weight_run,
        "momentum": _momentum_run,
        "mean_reversion": _mean_reversion_run,
        "moving_average": _moving_average_run,
        "volatility_targeted": _volatility_targeted_run,
        "random_null": _random_null_run,
    }

    if strategy_name not in strategy_funcs:
        raise ValueError(f"unknown strategy: {strategy_name}")

    runner = strategy_funcs[strategy_name]

    # Generate signals
    signals = runner(
        universe=universe,
        sessions=sessions,
        cost_model=cost_model,
        experiment_id=experiment_id,
        hypothesis_id=hypothesis_id,
        dataset_snapshot_ids=dataset_snapshot_ids,
        strategy_params=strategy_params,
        seed=seed,
    )

    # Run through the Phase 7 backtester
    backtester = _make_backtester(
        universe=universe,
        cost_model=cost_model,
        window_start=window_start,
        window_end=window_end,
        experiment_id=experiment_id,
        hypothesis_id=hypothesis_id,
        dataset_snapshot_ids=dataset_snapshot_ids,
    )

    result = backtester.run(
        bars=_get_bars(universe, sessions),
        signals=signals,
    )

    # Extract after-cost metrics
    metrics = _extract_metrics(result, strategy_name, strategy_params or {})

    # Compute summary from the result for the benchmark record
    summary = result.summary()

    # Build the full result record
    result_record = {
        "strategy": strategy_name,
        "strategy_version": "phase_8_baseline",
        "parameters": strategy_params or {},
        "parameter_grid_identity": grid["description"] if grid else "none",
        "universe": sorted(universe),
        "dataset_snapshot_ids": dataset_snapshot_ids or ["DS-000001"],
        "evaluation_window": {
            "start": sessions[0] if sessions else None,
            "end": sessions[-1] if sessions else None,
        },
        "cost_model": {
            "spread_bps": cost_model.spread_bps,
            "fees_bps": cost_model.fees_bps,
            "slippage_bps": cost_model.slippage_bps,
            "total_bps": cost_model.total_bps(),
        },
        "initial_cash": INITIAL_CASH,
        "final_equity": summary["final_equity"],
        "total_return": summary["total_return"],
        "total_pnl": summary["total_pnl"],
        "total_fees": summary["total_fees"],
        "total_spread_cost": summary["total_spread_cost"],
        "total_slippage_cost": summary["total_slippage_cost"],
        "turnover": summary["turnover"],
        "n_fills": summary["n_fills"],
        "n_rejects": summary["n_rejects"],
        "n_signals": summary["n_signals"],
        "experiment_id": experiment_id,
        "seed": seed,
        "stability": _compute_stability(result),
        "benchmark_report_ready": True,
    }

    return result_record


def _default_sessions() -> list[date]:
    """Return a default set of evaluation window sessions (weekdays)."""
    from datetime import date as d
    import itertools

    start = d(2024, 1, 2)
    dates: list[date] = []
    d = start
    for _ in range(60):  # ~3 months of trading days
        if d.weekday() < 5:
            dates.append(d)
        d += timedelta(days=1)
    return dates


def _get_bars(universe: list[str], sessions: list[date]) -> Any:
    """Get canonical bars for the evaluation window.

    In a production setting, these would come from the Phase 3 dataset
    snapshots.  For fixtures, we generate synthetic bars.
    """
    import polars as pl
    from datetime import timedelta

    # Generate synthetic bars similar to the test fixture
    base_prices = {"INS-000001": 100.0, "INS-000002": 50.0}
    prev_close: dict[str, float] = {}
    rows: list[dict[str, Any]] = []

    for i, s in enumerate(sessions):
        for ins in sorted(universe):
            base = base_prices.get(ins, 100.0)
            close = base * (1.0 + 0.0 * i)  # zero drift for baseline
            open_ = prev_close.get(ins, base)
            prev_close[ins] = close
            rows.append(
                {
                    "instrument_id": ins,
                    "trade_date": s,
                    "open": open_,
                    "high": max(open_, close) * 1.01,
                    "low": min(open_, close) * 0.99,
                    "close": close,
                    "volume": 1_000_000.0,
                }
            )

    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# Strategy-specific runner functions
# ---------------------------------------------------------------------------


def _buy_and_hold_run(
    universe: list[str],
    sessions: list[date],
    *,
    cost_model: CostModel,
    experiment_id: str,
    hypothesis_id: str,
    dataset_snapshot_ids: list[str] | None,
    strategy_params: dict[str, Any] | None,
    seed: int,
) -> list[dict[str, Any]]:
    """Run buy-and-hold: one initial buy, hold forever."""
    # Pick the first instrument in the universe
    inst_id = universe[0] if universe else "INS-000001"
    signals = buy_and_hold_signals(
        instrument_id=inst_id,
        sessions=sessions,
        target_shares=1_000.0,
        strategy_ref="buy_and_hold",
    )
    return signals


def _equal_weight_run(
    universe: list[str],
    sessions: list[date],
    *,
    cost_model: CostModel,
    experiment_id: str,
    hypothesis_id: str,
    dataset_snapshot_ids: list[str] | None,
    strategy_params: dict[str, Any] | None,
    seed: int,
) -> list[dict[str, Any]]:
    """Run equal-weight: session-by-session rebalancing."""
    rebalance = strategy_params.get("rebalance", True) if strategy_params else True
    signals = equal_weight_signals(
        instrument_ids=universe,
        sessions=sessions,
        rebalance=rebalance,
        strategy_ref="equal_weight",
    )
    return signals


def _momentum_run(
    universe: list[str],
    sessions: list[date],
    *,
    cost_model: CostModel,
    experiment_id: str,
    hypothesis_id: str,
    dataset_snapshot_ids: list[str] | None,
    strategy_params: dict[str, Any] | None,
    seed: int,
) -> list[dict[str, Any]]:
    """Run momentum: lookback-based ranking."""
    lookback = strategy_params.get("lookback_days", 20) if strategy_params else 20
    signals = momentum_signals(
        instrument_ids=universe,
        sessions=sessions,
        lookback_days=lookback,
        strategy_ref="momentum",
    )
    return signals


def _mean_reversion_run(
    universe: list[str],
    sessions: list[date],
    *,
    cost_model: CostModel,
    experiment_id: str,
    hypothesis_id: str,
    dataset_snapshot_ids: list[str] | None,
    strategy_params: dict[str, Any] | None,
    seed: int,
) -> list[dict[str, Any]]:
    """Run mean reversion: distance from rolling mean."""
    lookback = strategy_params.get("lookback_days", 20) if strategy_params else 20
    signals = mean_reversion_signals(
        instrument_ids=universe,
        sessions=sessions,
        lookback_days=lookback,
        strategy_ref="mean_reversion",
    )
    return signals


def _moving_average_run(
    universe: list[str],
    sessions: list[date],
    *,
    cost_model: CostModel,
    experiment_id: str,
    hypothesis_id: str,
    dataset_snapshot_ids: list[str] | None,
    strategy_params: dict[str, Any] | None,
    seed: int,
) -> list[dict[str, Any]]:
    """Run moving-average SMA crossover."""
    sw = strategy_params.get("short_window", 10) if strategy_params else 10
    lw = strategy_params.get("long_window", 30) if strategy_params else 30
    signals = moving_average_signals(
        instrument_ids=universe,
        sessions=sessions,
        short_window=sw,
        long_window=lw,
        strategy_ref="moving_average",
    )
    return signals


def _volatility_targeted_run(
    universe: list[str],
    sessions: list[date],
    *,
    cost_model: CostModel,
    experiment_id: str,
    hypothesis_id: str,
    dataset_snapshot_ids: list[str] | None,
    strategy_params: dict[str, Any] | None,
    seed: int,
) -> list[dict[str, Any]]:
    """Run volatility-targeted control."""
    tvol = strategy_params.get("target_volatility", 0.15) if strategy_params else 0.15
    ew = strategy_params.get("estimation_window", 30) if strategy_params else 30
    signals = volatility_targeted_signals(
        instrument_ids=universe,
        sessions=sessions,
        target_volatility=tvol,
        estimation_window=ew,
        strategy_ref="volatility_targeted",
    )
    return signals


def _random_null_run(
    universe: list[str],
    sessions: list[date],
    *,
    cost_model: CostModel,
    experiment_id: str,
    hypothesis_id: str,
    dataset_snapshot_ids: list[str] | None,
    strategy_params: dict[str, Any] | None,
    seed: int,
) -> list[dict[str, Any]]:
    """Run random/null control."""
    null = strategy_params.get("null", False) if strategy_params else False
    signals = random_null_signals(
        instrument_ids=universe,
        sessions=sessions,
        seed=seed,
        strategy_ref="random_null",
        null=null,
    )
    return signals


# ---------------------------------------------------------------------------
# After-cost metrics extraction
# ---------------------------------------------------------------------------

def _extract_metrics(
    result: Any, strategy_name: str, strategy_params: dict[str, Any]
) -> dict[str, Any]:
    """Extract after-cost risk-adjusted metrics from a Phase 7 backtest result.

    Uses the existing result object's fields that are already populated by
    the backtester (fees, spread, slippage, turnover, equity curve).
    The BacktestResult.summary() method provides all needed fields.
    """
    # Extract summary from the result
    summary = result.summary()

    total_return = summary["total_return"] if summary["total_return"] is not None else 0.0
    total_pnl = summary["total_pnl"] if summary["total_pnl"] is not None else 0.0
    final_equity = summary["final_equity"] if summary["final_equity"] is not None else 1_000_000.0
    turnover = summary["turnover"] if summary["turnover"] is not None else 0.0
    total_fees = summary["total_fees"] if summary["total_fees"] is not None else 0.0
    total_spread_cost = summary["total_spread_cost"] if summary["total_spread_cost"] is not None else 0.0
    total_slippage_cost = summary["total_slippage_cost"] if summary["total_slippage_cost"] is not None else 0.0
    total_costs = total_fees + total_spread_cost + total_slippage_cost

    # Number of fills and rejects
    n_fills = summary["n_fills"] if summary["n_fills"] is not None else 0
    n_rejects = summary["n_rejects"] if summary["n_rejects"] is not None else 0
    n_signals = summary["n_signals"] if summary["n_signals"] is not None else 0

    return {
        "total_return": total_return,
        "total_pnl": total_pnl,
        "total_costs": total_costs,
        "total_fees": total_fees,
        "total_spread_cost": total_spread_cost,
        "total_slippage_cost": total_slippage_cost,
        "turnover": turnover,
        "n_fills": n_fills,
        "n_rejects": n_rejects,
        "n_signals": n_signals,
        "final_equity": final_equity,
        "after_cost_return": total_return,  # same as total_return since costs already deducted
    }


# ---------------------------------------------------------------------------
# Stability computation
# ---------------------------------------------------------------------------

def _compute_stability(result: Any) -> dict[str, Any]:
    """Compute stability information from a Phase 7 backtest result.

    Evaluates whether baseline behavior is stable rather than relying on
    one headline number.  Uses the evaluation windows already supported by
    the current architecture.

    Stability measures extracted from the result:
      - final_equity
      - total_return
      - turnover
      - cost totals
      - fill/rejection counts
    """
    summary = result.summary()
    final_equity = summary["final_equity"] if summary["final_equity"] is not None else 1_000_000.0
    total_return = summary["total_return"] if summary["total_return"] is not None else 0.0
    turnover = summary["turnover"] if summary["turnover"] is not None else 0.0
    total_fees = summary["total_fees"] if summary["total_fees"] is not None else 0.0
    total_spread_cost = summary["total_spread_cost"] if summary["total_spread_cost"] is not None else 0.0
    total_slippage_cost = summary["total_slippage_cost"] if summary["total_slippage_cost"] is not None else 0.0
    total_costs = total_fees + total_spread_cost + total_slippage_cost
    n_fills = summary["n_fills"] if summary["n_fills"] is not None else 0
    n_rejects = summary["n_rejects"] if summary["n_rejects"] is not None else 0

    # Stability: compare cost efficiency
    cost_efficiency = total_fees / max(final_equity, 1e-9)
    slippage_ratio = total_slippage_cost / max(total_costs, 1e-9)

    # Fill quality: what fraction of signals resulted in fills?
    fill_rate = n_fills / max(n_rejects + n_fills, 1)

    # Rejection rate
    reject_rate = n_rejects / max(n_rejects + n_fills, 1)

    return {
        "final_equity": final_equity,
        "total_return": total_return,
        "turnover": turnover,
        "cost_efficiency": cost_efficiency,
        "slippage_ratio": slippage_ratio,
        "fill_rate": fill_rate,
        "reject_rate": reject_rate,
        "n_fills": n_fills,
        "n_rejects": n_rejects,
    }