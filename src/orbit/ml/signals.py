"""Phase 9 signal bridge: model predictions -> canonical Phase 7 signals.

The bridge emits the canonical signal row contract
(orbit.backtest.clock.MarketEventClock.normalize_signals):
  signal_id, instrument_id, signal_session (date), decision_time
  (= session_close_utc of the signal session), direction ('long'|'flat'),
  target (WEIGHT fraction), signal_metric, strategy_ref.

Protocol:
  - top-k long-only, equal weight 1/k over the k highest-scored names per
    session; every other name of the cross-section receives an explicit
    FLAT signal with target 0.0 so the Phase 7 rebalance semantics sell
    names that drop out of the top-k (no stale positions).
  - decision_time always equals the strict session close, so the Phase 4
    temporal gate (validate_signal_temporality) passes.
  - identical path and cost model (CM-001) as the Phase 8 controls.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import polars as pl

from orbit.backtest import Backtester
from orbit.backtest.config import (
    BacktestConfig,
    CostConfig,
    ExecutionConfig,
    ExecutionPrice,
    SizingPolicy,
)
from orbit.ml.features import close_utc
from orbit.ml.grids import params_identity
from orbit.ml.ranking import cross_sectional_rank, top_k_long
from orbit.schemas.experiment import CostModel

PHASE9_COST_MODEL = CostModel(spread_bps=2.0, fees_bps=1.0, slippage_bps=2.0)
PHASE9_COST_MODEL_ID = "CM-001"


def predictions_to_signals(
    pred_frame: pl.DataFrame,
    *,
    family: str,
    params: dict[str, Any],
    top_k: int = 3,
    strategy_ref: str | None = None,
) -> pl.DataFrame:
    """Convert a prediction frame into complete per-session canonical signals.

    `pred_frame` requires: instrument_id, decision_session (Date), prediction
    (Float). Every (instrument, session) row yields one signal row: LONG with
    target weight 1/k for top-k names, FLAT with target 0.0 otherwise.
    """
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    ranked = cross_sectional_rank(pred_frame, score_col="prediction")
    selected = top_k_long(ranked, k=top_k)
    base_ref = strategy_ref or f"phase9:{family}:{params_identity(family, params)}:topk{top_k}"

    out = pred_frame.sort(["decision_session", "instrument_id"]).with_columns(
        pl.lit("flat", dtype=pl.Utf8).alias("direction"),
        pl.lit(0.0, dtype=pl.Float64).alias("target"),
    )
    longs = selected.select(
        "instrument_id",
        "decision_session",
        pl.lit("long", dtype=pl.Utf8).alias("direction"),
        "target_weight",
    ).rename({"target_weight": "target"})
    out = out.join(longs, on=["instrument_id", "decision_session"], how="left")
    out = out.with_columns(
        pl.coalesce(pl.col("direction_right"), pl.col("direction")).alias("direction"),
        pl.coalesce(pl.col("target_right"), pl.col("target")).alias("target"),
    ).drop(["direction_right", "target_right"])

    out = out.with_columns(
        pl.col("decision_session").map_elements(close_utc, return_dtype=pl.Datetime("us", None)).alias("decision_time"),
        pl.lit(base_ref, dtype=pl.Utf8).alias("strategy_ref"),
        pl.col("prediction").alias("signal_metric"),
    )
    out = out.with_columns(
        (
            "SIG-P9-"
            + pl.col("instrument_id")
            + "-"
            + pl.col("decision_session").dt.strftime("%Y%m%d")
        ).alias("signal_id")
    )
    return out.select(
        "signal_id",
        "instrument_id",
        pl.col("decision_session").alias("signal_session"),
        "decision_time",
        "direction",
        "target",
        "signal_metric",
        "strategy_ref",
    )


def build_backtest_config(
    *,
    window_start: date,
    window_end: date,
    initial_cash: float = 1_000_000.0,
    seed: int = 42,
) -> BacktestConfig:
    """Canonical Phase 9 backtest configuration (WEIGHT sizing, CM-001 costs,
    open/delay=1 execution - identical to the Phase 8 control runs)."""
    return BacktestConfig(
        initial_cash=initial_cash,
        costs=CostConfig.from_cost_model(PHASE9_COST_MODEL),
        execution=ExecutionConfig(
            execution_price=ExecutionPrice.OPEN,
            execution_delay=1,
            partial_fills=True,
            order_expiry_sessions=5,
        ),
        sizing=SizingPolicy.WEIGHT,
        long_only=True,
        seed=seed,
        randomness_policy="seeded",
        window_start=window_start,
        window_end=window_end,
    )


def run_backtest(
    bars: pl.DataFrame,
    signals: pl.DataFrame,
    *,
    window_start: date,
    window_end: date,
    experiment_id: str,
    hypothesis_id: str,
    events: pl.DataFrame | None = None,
    feature_refs: list[dict[str, Any]] | None = None,
    model: dict[str, Any] | None = None,
    label_id: str = "LAB-004",
    label_version: str = "v1",
    temporal_config_digest: str | None = None,
    seed: int = 42,
    initial_cash: float = 1_000_000.0,
) -> Any:
    """Run the canonical Phase 7 backtester on a Phase 9 signal set."""
    config = build_backtest_config(
        window_start=window_start, window_end=window_end, initial_cash=initial_cash, seed=seed
    )
    from orbit.ml.registry import ml_code_hash

    backtester = Backtester(
        config=config,
        universe=sorted(bars["instrument_id"].unique().to_list()),
        dataset_snapshot_ids=["DS-000004"],
        code_hash=ml_code_hash(),
        experiment_id=experiment_id,
        hypothesis_id=hypothesis_id,
        feature_refs=feature_refs,
        model=model,
        label_id=label_id,
        label_version=label_version,
        temporal_config_digest=temporal_config_digest,
        cost_model_id=PHASE9_COST_MODEL_ID,
    )
    return backtester.run(bars, signals, events_artifact=events)


__all__ = [
    "PHASE9_COST_MODEL",
    "PHASE9_COST_MODEL_ID",
    "predictions_to_signals",
    "build_backtest_config",
    "run_backtest",
]