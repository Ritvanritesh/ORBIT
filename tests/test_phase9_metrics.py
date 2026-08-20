"""Phase 9 metric tests: OOS IC / rank IC semantics and aggregation."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from datetime import date

from orbit.ml.metrics import hit_rate, mean_squared_error, oos_ic, rank_ic


def _frame():
    return pl.DataFrame(
        {
            "decision_session": [date(2024, 1, 2)] * 4 + [date(2024, 1, 3)] * 4,
            "instrument_id": ["A", "B", "C", "D"] * 2,
            "prediction": [1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0],
            "outcome_value": [2.0, 3.0, 4.0, 5.0, -0.1, -0.2, -0.3, -0.4],
        }
    )


def test_oos_ic_perfect_positive_correlation():
    f = _frame()
    ic = oos_ic(f, "prediction")
    assert ic["value"] == pytest.approx(1.0)
    assert ic["sessions_used"] == 2


def test_rank_ic_matches_spearman():
    f = _frame()
    ric = rank_ic(f, "prediction")
    assert ric["value"] == pytest.approx(1.0)


def test_ic_is_mean_over_sessions():
    f = _frame()
    ic = oos_ic(f, "prediction")
    assert ic["value"] == pytest.approx(1.0)


def test_ic_can_average_mixed_sessions():
    f = _frame().with_columns(
        pl.when(pl.col("decision_session") == date(2024, 1, 3))
        .then(-pl.col("outcome_value"))
        .otherwise(pl.col("outcome_value"))
        .alias("outcome_value")
    )
    # session 1: +1, session 2: -1 -> mean 0
    assert oos_ic(f, "prediction")["value"] == pytest.approx(0.0, abs=1e-9)


def test_short_sessions_are_skipped_and_counted():
    f = pl.DataFrame(
        {
            "decision_session": [date(2024, 1, 2)] * 2,
            "instrument_id": ["A", "B"],
            "prediction": [1.0, 2.0],
            "outcome_value": [2.0, 3.0],
        }
    )
    ic = oos_ic(f, "prediction", min_obs=3)
    assert ic["sessions_used"] == 0
    assert ic["sessions_skipped_short"] == 1


def test_zero_variance_sessions_are_skipped():
    f = pl.DataFrame(
        {
            "decision_session": [date(2024, 1, 2)] * 4,
            "instrument_id": ["A", "B", "C", "D"],
            "prediction": [1.0, 1.0, 1.0, 1.0],
            "outcome_value": [2.0, 3.0, 4.0, 5.0],
        }
    )
    ic = oos_ic(f, "prediction")
    assert ic["sessions_used"] == 0
    assert ic["sessions_skipped_variance"] == 1


def test_null_rows_dropped_before_ic():
    f = _frame().with_columns(
        pl.when(pl.col("instrument_id") == "A").then(pl.lit(None)).otherwise(pl.col("prediction")).alias("prediction")
    )
    ic = oos_ic(f, "prediction")
    assert ic["sessions_used"] == 2
    assert np.isfinite(ic["value"])


def test_mean_squared_error():
    assert mean_squared_error(np.array([1.0, 2.0]), np.array([2.0, 2.0])) == pytest.approx(0.5)


def test_hit_rate_sign_agreement():
    f = pl.DataFrame(
        {
            "prediction": [0.1, -0.1, 0.2, -0.2],
            "outcome_value": [0.3, 0.1, -0.1, -0.4],
        }
    )
    # agreements: row0 +, row1 +->(pred -) mismatch, row2 mismatch, row3 agree -> 0.5
    assert hit_rate(f, "prediction") == pytest.approx(0.5)


def test_hit_rate_empty_is_nan():
    f = pl.DataFrame({"prediction": [], "outcome_value": []})
    assert np.isnan(hit_rate(f, "prediction"))