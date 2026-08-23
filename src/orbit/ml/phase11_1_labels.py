"""Phase 11.1 excess-return labels (LAB-005).

Defines and computes benchmark-relative excess return labels.
LAB-005 v1: 5-session forward excess return vs BENCH-001 (SPY).

This is a NEW label generation - LAB-004 remains unchanged.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date
from typing import Any

import polars as pl

from orbit.labels.contract import (
    AnchorMode,
    CorporateActionPolicy,
    DelistingPolicy,
    LabelContract,
    LabelType,
    MissingDataPolicy,
    OverlapPolicy,
    ReturnConvention,
)

LABEL_ID = "LAB-005"
LABEL_VERSION = "v1"
LABEL_HORIZON_SESSIONS = 5

LAB005_FORMULA = (
    "5-session forward excess return vs BENCH-001 (SPY): "
    "instrument_forward_return(t, h) - benchmark_forward_return(t, h), "
    "where both are SIMPLE_TOTAL_RETURN over h=5 sessions anchored at "
    "the decision instant. Entry is the last completed session strictly "
    "before the decision instant; outcome closes over the next 5 sessions."
)


def build_phase11_1_label_contract() -> LabelContract:
    """Build the LAB-005 v1 excess-return label contract."""
    return LabelContract(
        label_id=LABEL_ID,
        version=LABEL_VERSION,
        target_type=LabelType.EXCESS_RETURN,
        horizon=LABEL_HORIZON_SESSIONS,
        anchor_mode=AnchorMode.DECISION_INSTANT,
        price_field="close",
        return_convention=ReturnConvention.SIMPLE_TOTAL_RETURN,
        benchmark="BENCH-001",
        overlap_policy=OverlapPolicy.WINDOWS_TRACKED,
        missing_data_policy=MissingDataPolicy.EXPLICIT_UNAVAILABLE,
        delisting_policy=DelistingPolicy.UNAVAILABLE_WITH_REASON,
        corporate_action_policy=CorporateActionPolicy.CANONICAL_PHASE3,
        formula=LAB005_FORMULA,
        description="Phase 11.1 benchmark-relative excess return target",
        author="orbit-research",
    )


def compute_excess_return_label(
    instrument_bars: pl.DataFrame,
    benchmark_bars: pl.DataFrame,
    horizon: int = LABEL_HORIZON_SESSIONS,
) -> pl.DataFrame:
    """Compute excess-return labels for all instruments.

    Parameters
    ----------
    instrument_bars : pl.DataFrame
        Must have: trade_date, instrument_id, close
    benchmark_bars : pl.DataFrame
        Must have: trade_date, close (will be renamed to benchmark_close)
    horizon : int
        Forward return horizon in sessions (default 5)

    Returns
    -------
    pl.DataFrame
        Rows with: instrument_id, trade_date, close, benchmark_close,
        instrument_return, benchmark_return, excess_return, label_available
    """
    from orbit.ml.phase11_1_alignment import align_instrument_benchmark, compute_forward_returns

    # Align instrument and benchmark on same trading dates
    aligned = align_instrument_benchmark(instrument_bars, benchmark_bars)

    # Compute forward returns for both
    with_returns = compute_forward_returns(aligned, horizon=horizon)

    # Determine label availability
    with_returns = with_returns.with_columns([
        pl.when(
            pl.col("instrument_forward_return").is_null()
            | pl.col("benchmark_forward_return").is_null()
        ).then(pl.lit(False)).otherwise(pl.lit(True)).alias("label_available"),
    ])

    return with_returns


def toy_example_simple() -> dict[str, Any]:
    """Hand-computed toy example for validation.

    Scenario:
        Session 0: instrument close=100, benchmark close=100
        Session 1: instrument close=105, benchmark close=102
        Session 2: instrument close=103, benchmark close=101
        Session 3: instrument close=108, benchmark close=104
        Session 4: instrument close=110, benchmark close=105
        Session 5: instrument close=112, benchmark close=106

    For horizon=5, entry at session 0, outcome at session 5:
        instrument_return = 112/100 - 1 = 0.12
        benchmark_return = 106/100 - 1 = 0.06
        excess_return = 0.12 - 0.06 = 0.06
    """
    return {
        "scenario": "toy_simple",
        "horizon": 5,
        "entries": [
            {"session": 0, "instrument_close": 100.0, "benchmark_close": 100.0},
            {"session": 1, "instrument_close": 105.0, "benchmark_close": 102.0},
            {"session": 2, "instrument_close": 103.0, "benchmark_close": 101.0},
            {"session": 3, "instrument_close": 108.0, "benchmark_close": 104.0},
            {"session": 4, "instrument_close": 110.0, "benchmark_close": 105.0},
            {"session": 5, "instrument_close": 112.0, "benchmark_close": 106.0},
        ],
        "expected": {
            "instrument_return_at_0": 0.12,
            "benchmark_return_at_0": 0.06,
            "excess_return_at_0": 0.06,
        },
    }


def toy_example_instrument_equals_benchmark() -> dict[str, Any]:
    """When instrument equals benchmark, excess return should be ~0."""
    return {
        "scenario": "instrument_equals_benchmark",
        "horizon": 5,
        "entries": [
            {"session": i, "instrument_close": 100.0 + i, "benchmark_close": 100.0 + i}
            for i in range(10)
        ],
        "expected": {
            "excess_returns_all_zero": True,
        },
    }


def toy_example_benchmark_rises_instrument_flat() -> dict[str, Any]:
    """Benchmark rises while instrument is flat -> negative excess return."""
    return {
        "scenario": "benchmark_rises_instrument_flat",
        "horizon": 5,
        "entries": [
            {"session": 0, "instrument_close": 100.0, "benchmark_close": 100.0},
            {"session": 5, "instrument_close": 100.0, "benchmark_close": 110.0},
        ],
        "expected": {
            "instrument_return": 0.0,
            "benchmark_return": 0.10,
            "excess_return": -0.10,
        },
    }


def toy_example_instrument_outperforms() -> dict[str, Any]:
    """Instrument outperforms benchmark -> positive excess return."""
    return {
        "scenario": "instrument_outperforms",
        "horizon": 5,
        "entries": [
            {"session": 0, "instrument_close": 100.0, "benchmark_close": 100.0},
            {"session": 5, "instrument_close": 120.0, "benchmark_close": 110.0},
        ],
        "expected": {
            "instrument_return": 0.20,
            "benchmark_return": 0.10,
            "excess_return": 0.10,
        },
    }


__all__ = [
    "LABEL_ID", "LABEL_VERSION", "LABEL_HORIZON_SESSIONS", "LAB005_FORMULA",
    "build_phase11_1_label_contract", "compute_excess_return_label",
    "toy_example_simple", "toy_example_instrument_equals_benchmark",
    "toy_example_benchmark_rises_instrument_flat", "toy_example_instrument_outperforms",
]
