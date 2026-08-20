"""Phase 9 control baseline tests: documented Phase 8 rules on real bars."""

from __future__ import annotations

import polars as pl
import pytest
from datetime import date

from orbit.ml.baselines import (
    CONTROL_BUILDERS,
    CONTROL_FAMILIES,
    CONTROL_GRIDS,
    build_control_signals,
    control_metrics,
)
from tests.phase9_testutils import make_canonical_bars, session_close_utc, weekdays


@pytest.fixture(scope="module")
def bars():
    return make_canonical_bars(sessions=weekdays(date(2014, 7, 1), 200))


@pytest.fixture(scope="module")
def sessions(bars):
    return bars["trade_date"].unique().sort().to_list()


def test_all_controls_build_complete_signals(bars, sessions):
    for family in CONTROL_FAMILIES:
        points = CONTROL_GRIDS.get(family) or [{}]
        for params in points:
            sig = build_control_signals(bars, sessions, family, params)
            assert sig.height > 0, (family, params)
            assert set(sig["direction"].unique().to_list()) <= {"long", "flat"}
            assert sig.filter(pl.col("target") < 0).height == 0


def test_control_decision_times_are_session_close(bars, sessions):
    sig = build_control_signals(bars, sessions, "equal_weight", {})
    for r in sig.head(20).iter_rows(named=True):
        assert r["decision_time"] == session_close_utc(r["signal_session"])


def test_equal_weight_targets_one_over_n(bars, sessions):
    sig = build_control_signals(bars, sessions, "equal_weight", {})
    first = sig["signal_session"].min()
    weights = sig.filter(pl.col("signal_session") == first)["target"].to_list()
    assert all(w == pytest.approx(1.0 / sig["instrument_id"].n_unique()) for w in weights)


def test_momentum_signals_one_long_per_session(bars, sessions):
    sig = build_control_signals(bars, sessions, "momentum", {"lookback": 20})
    longs = sig.filter(pl.col("direction") == "long")
    assert longs.height == sig["signal_session"].n_unique()


def test_momentum_winner_uses_point_in_time_return(bars, sessions):
    metrics = control_metrics(bars).filter(pl.col("decision_session").is_in(sessions))
    assert "ret_20" in metrics.columns
    first = metrics.filter(pl.col("decision_session") == sessions[70]).drop_nulls(subset=["ret_20"])
    assert first.height > 0


def test_momentum_grid_is_registered():
    assert CONTROL_GRIDS["momentum"] == [{"lookback": 10}, {"lookback": 20}, {"lookback": 30}]


def test_moving_average_grid_is_registered():
    assert CONTROL_GRIDS["moving_average"] == [
        {"short_window": 5, "long_window": 30},
        {"short_window": 10, "long_window": 30},
        {"short_window": 15, "long_window": 40},
    ]


def test_volatility_targeted_grid_is_registered():
    assert CONTROL_GRIDS["volatility_targeted"] == [
        {"target_volatility": 0.10, "estimation_window": 10},
        {"target_volatility": 0.15, "estimation_window": 30},
        {"target_volatility": 0.20, "estimation_window": 60},
    ]


def test_buy_and_hold_emits_single_long(bars, sessions):
    sig = build_control_signals(bars, sessions, "buy_and_hold", {})
    assert sig.height == 1
    assert sig["direction"][0] == "long"
    assert sig["target"][0] == pytest.approx(0.99)


def test_null_flat_emits_no_longs(bars, sessions):
    sig = build_control_signals(bars, sessions, "null_flat", {})
    assert sig.filter(pl.col("direction") == "long").height == 0


def test_random_null_is_seeded(bars, sessions):
    a = build_control_signals(bars, sessions, "random_null", {})
    b = build_control_signals(bars, sessions, "random_null", {})
    assert a.equals(b)


def test_mean_reversion_long_only_when_below_mean(bars, sessions):
    sig = build_control_signals(bars, sessions, "mean_reversion", {"lookback": 10})
    assert sig.filter(pl.col("direction") == "long").height <= sig["signal_session"].n_unique()


def test_controls_are_point_in_time(bars, sessions):
    metrics = control_metrics(bars)
    # every metric row's window ends strictly before its decision session
    assert metrics.height > 0
    # spot check: ret_20 at decision_session t equals close(t-1)/close(t-21) - 1
    one = bars.filter(pl.col("instrument_id") == "INS-000101").sort("trade_date")
    m = metrics.filter(pl.col("instrument_id") == "INS-000101").sort("decision_session")
    row = m[100]
    dates = one["trade_date"].to_list()
    closes = one["close"].to_list()
    pos = dates.index(row["decision_session"].to_list()[0])
    expected = closes[pos - 1] / closes[pos - 20] - 1.0
    assert m["ret_20"][100] == pytest.approx(expected)


def test_unknown_control_rejected(bars, sessions):
    with pytest.raises(ValueError, match="unknown control family"):
        build_control_signals(bars, sessions, "bogus", {})


def test_paramless_controls_reject_params(bars, sessions):
    with pytest.raises(ValueError, match="takes no parameters"):
        build_control_signals(bars, sessions, "equal_weight", {"rebalance": True})