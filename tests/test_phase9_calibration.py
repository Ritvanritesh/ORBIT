"""Phase 9 calibration tests: validation-only fitting, deterministic ECE."""

from __future__ import annotations

import numpy as np
import pytest

from orbit.ml.calibration import (
    assert_no_test_fit,
    brier_score,
    expected_calibration_error,
    fit_platt,
)


def _scores():
    rng = np.random.RandomState(7)
    scores = rng.uniform(-1.0, 1.0, 400)
    binary = (scores > 0).astype(int)
    return scores, binary


def test_platt_map_returns_probabilities():
    scores, binary = _scores()
    m = fit_platt(scores, binary)
    probs = m.apply(scores)
    assert ((probs >= 0.0) & (probs <= 1.0)).all()
    assert m.fitted_on == "val"


def test_platt_fit_is_deterministic():
    scores, binary = _scores()
    a = fit_platt(scores, binary)
    b = fit_platt(scores, binary)
    assert a.slope == b.slope and a.intercept == b.intercept


def test_platt_requires_both_classes():
    with pytest.raises(ValueError, match="both outcome classes"):
        fit_platt(np.zeros(10), np.zeros(10, dtype=int))


def test_platt_requires_rows():
    with pytest.raises(ValueError, match="at least 2"):
        fit_platt(np.zeros(1), np.array([0]))


def test_ece_perfect_calibration_is_zero():
    """Scores equal to the realized binary outcomes are perfectly
    calibrated: in every bin the mean prediction equals the mean outcome."""
    scores = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=float)
    binary = scores.astype(int)
    res = expected_calibration_error(scores, binary, n_bins=10)
    assert res["ece"] == pytest.approx(0.0, abs=1e-9)


def test_ece_deterministic_and_bounded():
    scores, binary = _scores()
    a = expected_calibration_error(scores, binary, n_bins=10)
    b = expected_calibration_error(scores, binary, n_bins=10)
    assert a["ece"] == b["ece"]
    assert 0.0 <= a["ece"] <= 1.0
    assert a["n_bins"] == 10


def test_brier_score():
    scores = np.array([0.9, 0.1, 0.6])
    binary = np.array([1, 0, 1])
    assert brier_score(scores, binary) == pytest.approx((0.01 + 0.01 + 0.16) / 3)


def test_brier_perfect_is_zero():
    assert brier_score(np.array([1.0, 0.0]), np.array([1, 0])) == 0.0


def test_assert_no_test_fit_blocks_test_fitted_maps():
    scores, binary = _scores()
    m = fit_platt(scores, binary)
    assert_no_test_fit(m)  # fitted_on == "val"
    m.fitted_on = "test"
    with pytest.raises(AssertionError, match="must be 'val'"):
        assert_no_test_fit(m)


def test_apply_transforms_scores_monotonically():
    scores, binary = _scores()
    m = fit_platt(scores, binary)
    probs = m.apply(np.array([-0.5, 0.0, 0.5]))
    assert probs[0] < probs[1] < probs[2]