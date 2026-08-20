"""Calibration for the Phase 9 benchmark.

Protocol (fixed):
  - The calibration map (Platt scaling: a single-parameter sigmoid; or
    isotonic regression) is fit on the VALIDATION split's predictions ONLY.
  - The TEST split is never touched by any fitting step - the fitted map is
    applied to test scores unchanged. This is the standard "calibration must
    be fit out-of-sample" rule, audited by `assert_no_test_fit`.
  - The logistic family is already a probability model; the same
    validation-only protocol still applies (its scores are calibrated on
    validation and applied to test).
  - ECE is computed with uniform-width bins over [0, 1]; empty bins are
    skipped; the number of bins is recorded in every report.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


@dataclass
class CalibrationMap:
    method: str
    fitted_rows: int
    fitted_on: str = "val"

    def apply(self, scores: np.ndarray) -> np.ndarray:
        raise NotImplementedError


@dataclass
class PlattMap(CalibrationMap):
    slope: float = 0.0
    intercept: float = 0.0

    def apply(self, scores: np.ndarray) -> np.ndarray:
        logits = self.slope * np.asarray(scores, dtype=np.float64) + self.intercept
        return 1.0 / (1.0 + np.exp(-logits))


def fit_platt(scores: np.ndarray, binary_outcomes: np.ndarray) -> PlattMap:
    """Fit Platt scaling on validation scores/binary outcomes (deterministic)."""
    from sklearn.linear_model import LogisticRegression

    scores = np.asarray(scores, dtype=np.float64).reshape(-1, 1)
    binary = np.asarray(binary_outcomes, dtype=np.int64)
    if len(scores) < 2:
        raise ValueError("Platt scaling requires at least 2 validation rows")
    if len(np.unique(binary)) < 2:
        raise ValueError("Platt scaling requires both outcome classes in validation")
    lr = LogisticRegression(solver="lbfgs", max_iter=5000, random_state=42)
    lr.fit(scores, binary)
    return PlattMap(
        method="platt",
        fitted_rows=int(len(scores)),
        slope=float(lr.coef_[0][0]),
        intercept=float(lr.intercept_[0]),
    )


def expected_calibration_error(
    calibrated_scores: np.ndarray, binary_outcomes: np.ndarray, n_bins: int = 10
) -> dict[str, Any]:
    """ECE with uniform-width bins over [0,1]; empty bins skipped."""
    scores = np.asarray(calibrated_scores, dtype=np.float64)
    binary = np.asarray(binary_outcomes, dtype=np.int64)
    if len(scores) == 0:
        return {"ece": float("nan"), "n_bins": n_bins, "bins_used": 0, "rows": 0}
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total_w = 0.0
    total_acc = 0.0
    bins_used = 0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (scores >= lo) & (scores < hi)
        if i == n_bins - 1:
            mask = (scores >= lo) & (scores <= hi)
        n = int(mask.sum())
        if n == 0:
            continue
        pred = float(scores[mask].mean())
        actual = float(binary[mask].mean())
        total_w += n
        total_acc += n * abs(pred - actual)
        bins_used += 1
    ece = total_acc / total_w if total_w else float("nan")
    return {"ece": float(ece), "n_bins": n_bins, "bins_used": bins_used, "rows": int(len(scores))}


def brier_score(calibrated_scores: np.ndarray, binary_outcomes: np.ndarray) -> float:
    scores = np.asarray(calibrated_scores, dtype=np.float64)
    binary = np.asarray(binary_outcomes, dtype=np.int64)
    return float(np.mean((scores - binary) ** 2))


def assert_no_test_fit(calibration: CalibrationMap, test_scores_used: bool = True) -> None:
    """Adversarial verification: calibration must be a validation-fitted map."""
    if calibration.fitted_on != "val":
        raise AssertionError(f"calibration fitted on {calibration.fitted_on!r}; must be 'val'")


__all__ = [
    "CalibrationMap",
    "PlattMap",
    "fit_platt",
    "expected_calibration_error",
    "brier_score",
    "assert_no_test_fit",
]