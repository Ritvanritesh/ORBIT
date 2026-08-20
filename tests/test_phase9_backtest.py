"""Phase 9 signal bridge and backtest tests: canonical path, identical costs."""

from __future__ import annotations

import polars as pl
import pytest
from datetime import date

from orbit.ml.signals import (
    PHASE9_COST_MODEL,
    PHASE9_COST_MODEL_ID,
    build_backtest_config,
    predictions_to_signals,
    run_backtest,
)
from tests.phase9_testutils import TEST_WINDOWS, make_canonical_bars, make_events, session_close_utc, weekdays


def _pred_frame(n_sessions: int = 30):
    bars = make_canonical_bars(sessions=weekdays(date(2014, 7, 1), n_sessions))
    sessions = bars["trade_date"].unique().sort().to_list()
    instruments = sorted(bars["instrument_id"].unique().to_list())
    rows = []
    i = 0
    for s in sessions:
        for ins in instruments:
            rows.append(
                {
                    "instrument_id": ins,
                    "decision_session": s,
                    "prediction": float(i % 7) / 10.0,
                }
            )
            i += 1
    return pl.DataFrame(rows)


def test_signals_are_complete_per_session():
    sig = predictions_to_signals(_pred_frame(), family="ridge", params={"alpha": 1.0}, top_k=3)
    n_sessions = sig["signal_session"].n_unique()
    n_instruments = sig["instrument_id"].n_unique()
    assert sig.height == n_sessions * n_instruments
    assert sig.filter(pl.col("direction") == "long").height == n_sessions * 3
    assert sig.filter(pl.col("direction") == "flat").height == n_sessions * (n_instruments - 3)


def test_signal_decision_time_is_session_close():
    sig = predictions_to_signals(_pred_frame(), family="ridge", params={"alpha": 1.0}, top_k=3)
    for r in sig.head(20).iter_rows(named=True):
        assert r["decision_time"] == session_close_utc(r["signal_session"])


def test_topk_targets_sum_to_one():
    sig = predictions_to_signals(_pred_frame(), family="ridge", params={"alpha": 1.0}, top_k=4)
    first = sig["signal_session"].min()
    weights = sig.filter(pl.col("signal_session") == first)["target"].sum()
    assert weights == pytest.approx(1.0)


def test_signal_metric_is_prediction_and_strategy_ref_is_pinned():
    sig = predictions_to_signals(_pred_frame(), family="lasso", params={"alpha": 0.01}, top_k=3)
    assert sig["strategy_ref"].n_unique() == 1
    assert "lasso" in sig["strategy_ref"][0]


def test_backtest_config_uses_cm001_and_weight_sizing():
    cfg = build_backtest_config(window_start=TEST_WINDOWS["test_start"], window_end=TEST_WINDOWS["test_end"])
    assert cfg.costs.spread_bps == 2.0
    assert cfg.costs.fees_bps == 1.0
    assert cfg.costs.slippage_bps == 2.0
    assert cfg.sizing.value == "weight"
    assert cfg.long_only is True
    assert cfg.execution.execution_price.value == "open"
    assert cfg.execution.execution_delay == 1


def test_cost_model_matches_phase8_default():
    assert PHASE9_COST_MODEL.spread_bps == 2.0
    assert PHASE9_COST_MODEL.fees_bps == 1.0
    assert PHASE9_COST_MODEL.slippage_bps == 2.0
    assert PHASE9_COST_MODEL_ID == "CM-001"


def test_ml_backtest_runs_through_canonical_phase7():
    bars = make_canonical_bars()
    sig = predictions_to_signals(_pred_frame(40), family="ridge", params={"alpha": 1.0}, top_k=3)
    res = run_backtest(
        bars, sig,
        window_start=TEST_WINDOWS["test_start"],
        window_end=TEST_WINDOWS["test_end"],
        experiment_id="EXP-90005",
        hypothesis_id="H-001",
        events=make_events(),
        model={"family": "ridge", "hyperparameters": {"alpha": 1.0}},
    )
    summary = res.summary()
    assert summary["n_fills"] >= 0
    assert summary["final_equity"] > 0
    assert res.invariant_violations() == []


def test_backtest_uses_test_window_only():
    bars = make_canonical_bars()
    sig = predictions_to_signals(_pred_frame(40), family="ridge", params={"alpha": 1.0}, top_k=3)
    res = run_backtest(
        bars, sig,
        window_start=TEST_WINDOWS["test_start"],
        window_end=TEST_WINDOWS["test_end"],
        experiment_id="EXP-90005",
        hypothesis_id="H-001",
        events=make_events(),
        model={"family": "ridge", "hyperparameters": {"alpha": 1.0}},
    )
    session_dates = pl.DataFrame(res.signals)["signal_session"].unique().to_list()
    assert all(TEST_WINDOWS["test_start"] <= s <= TEST_WINDOWS["test_end"] for s in session_dates)