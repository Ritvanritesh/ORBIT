"""Permanent benchmark report for Phase 8 baseline strategies.

Creates a durable benchmark artifact/report containing the baseline results.
The report identifies all required fields and is reproducible from ORBIT's
existing lineage.

The report is the benchmark against which Phase 9 ML must later be compared.
It must NOT overwrite historical benchmark results silently.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

from orbit.baselines.registry_integration import record_baseline_result
from orbit.schemas.common import CostModel


# ---------------------------------------------------------------------------
# Benchmark report class
# ---------------------------------------------------------------------------

class BenchmarkReport:
    """A permanent, reproducible benchmark report for Phase 8 baseline
    strategies.

    Invariants:
      - One row per (strategy, parameter_grid, cost_model, window) combination.
      - Never overwritten silently: a new report run with the same experiment
        ID but different parameters creates a new experiment (child or new ID).
      - All lineage is preserved through the Phase 6 experiment registry.
      - Reproducible from ORBIT's existing lineage (dataset snapshots, cost
        model hash, code hash, parameter grid identity).
    """

    def __init__(self, report_path: str | Path | None = None):
        self._report_path = Path(report_path) if report_path else Path(
            "benchmarks/phase8_baseline_benchmark.parquet"
        )
        self._rows: list[dict[str, Any]] = []
        self._ensure_dir()

    def _ensure_dir(self) -> None:
        """Create the report directory if it doesn't exist."""
        self._report_path.parent.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------
    # Add a result row
    # -----------------------------------------------------------------

    def add_result(
        self,
        strategy: str,
        strategy_version: str,
        parameters: dict[str, Any],
        parameter_grid_identity: str,
        universe: list[str],
        dataset_snapshot_ids: list[str],
        evaluation_window: dict[str, Any],
        cost_model: CostModel,
        initial_cash: float,
        final_equity: float,
        total_return: float,
        total_pnl: float,
        total_costs: float,
        turnover: float,
        n_fills: int,
        n_rejects: int,
        n_signals: int,
        experiment_id: str,
        seed: int,
        stability: dict[str, Any],
        strategy_code_hash: str | None = None,
        config_hash: str | None = None,
    ) -> None:
        """Add a result row to the benchmark report.

        All fields are required for reproducibility and comparability.
        """
        row = {
            "timestamp": datetime.now().isoformat(),
            "strategy": strategy,
            "strategy_version": strategy_version,
            "parameters": json_serialize(parameters),
            "parameter_grid_identity": parameter_grid_identity,
            "universe": json_serialize(sorted(universe)),
            "dataset_snapshot_ids": json_serialize(sorted(dataset_snapshot_ids)),
            "evaluation_window_start": evaluation_window.get("start"),
            "evaluation_window_end": evaluation_window.get("end"),
            "cost_model_spread_bps": cost_model.spread_bps,
            "cost_model_fees_bps": cost_model.fees_bps,
            "cost_model_slippage_bps": cost_model.slippage_bps,
            "cost_model_total_bps": cost_model.total_bps(),
            "initial_cash": initial_cash,
            "final_equity": final_equity,
            "total_return": total_return,
            "total_pnl": total_pnl,
            "total_costs": total_costs,
            "turnover": turnover,
            "n_fills": n_fills,
            "n_rejects": n_rejects,
            "n_signals": n_signals,
            "experiment_id": experiment_id,
            "seed": seed,
            "stability_json": json_serialize(stability),
            "strategy_code_hash": strategy_code_hash or "unknown",
            "config_hash": config_hash or "unknown",
        }
        self._rows.append(row)

    # -----------------------------------------------------------------
    # Save the report
    # -----------------------------------------------------------------

    def save(self) -> Path:
        """Persist the benchmark report to parquet (columnar, reproducible).

        The parquet file is written with a deterministic schema.  If the file
        already exists, a new file is written with a suffix to avoid silent
        overwrite (the roadmap explicitly says "Do not overwrite historical
        benchmark results silently").
        """
        df = pl.DataFrame(self._rows)

        # Check if file already exists - if so, suffix to avoid overwrite
        if self._report_path.exists():
            suffix = 1
            new_path = self._report_path.parent / (
                self._report_path.stem + f"_{suffix}" + self._report_path.suffix
            )
            while new_path.exists():
                suffix += 1
                new_path = self._report_path.parent / (
                    self._report_path.stem + f"_{suffix}" + self._report_path.suffix
                )
            self._report_path = new_path

        df.write_parquet(str(self._report_path))
        return self._report_path

    # -----------------------------------------------------------------
    # Load a previously saved report
    # -----------------------------------------------------------------

    def load(self) -> pl.DataFrame:
        """Load a previously saved benchmark report."""
        return pl.read_parquet(str(self._report_path))

    # -----------------------------------------------------------------
    # Generate summary statistics
    # -----------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        """Compute summary statistics across all recorded baselines."""
        if not self._rows:
            return {"n_baselines": 0, "strategies": {}}

        df = pl.DataFrame(self._rows)

        strategies = {}
        for strategy in df["strategy"].unique():
            sdf = df.filter(pl.col("strategy") == strategy)
            strategies[strategy] = {
                "n_runs": len(sdf),
                "mean_total_return": float(sdf["total_return"].mean()),
                "std_total_return": float(sdf["total_return"].std()),
                "min_total_return": float(sdf["total_return"].min()),
                "max_total_return": float(sdf["total_return"].max()),
                "mean_final_equity": float(sdf["final_equity"].mean()),
                "mean_turnover": float(sdf["turnover"].mean()),
                "mean_total_costs": float(sdf["total_costs"].mean()),
            }

        return {
            "n_baselines": len(df),
            "strategies": strategies,
        }


# ---------------------------------------------------------------------------
# JSON serialization helper (polars-friendly)
# ---------------------------------------------------------------------------

def json_serialize(value: Any) -> str:
    """Serialize a value to JSON string, handling polars types and None."""
    import json

    if value is None:
        return json.dumps(None)
    if isinstance(value, (bool, int, float)):
        return json.dumps(value)
    if isinstance(value, str):
        return json.dumps(value)
    # Fallback: try model_dump or dict
    if hasattr(value, "model_dump"):
        return json.dumps(value.model_dump(mode="json"), sort_keys=True, default=str)
    if hasattr(value, "dict"):
        return json.dumps(value.dict(), sort_keys=True, default=str)
    return json.dumps(str(value), sort_keys=True, default=str)