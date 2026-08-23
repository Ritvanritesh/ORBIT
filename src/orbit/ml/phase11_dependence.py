"""Phase 11 dependence diagnostics.

Builds autocorrelation awareness, effective sample-size diagnostics,
overlap detection, and warnings when naive i.i.d. assumptions would be
misleading.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DependenceReport:
    """Immutable dependence diagnostic record."""

    autocorrelations: dict[int, float]
    ljung_box_statistic: float | None
    ljung_box_p_value: float | None
    effective_sample_size: int | None
    effective_sample_size_method: str | None
    overlapping_outcomes: bool
    overlap_fraction: float
    overlap_description: str
    warnings: list[str]

    def summary(self) -> dict[str, Any]:
        return {
            "autocorrelations": self.autocorrelations,
            "ljung_box_statistic": self.ljung_box_statistic,
            "ljung_box_p_value": self.ljung_box_p_value,
            "effective_sample_size": self.effective_sample_size,
            "effective_sample_size_method": self.effective_sample_size_method,
            "overlapping_outcomes": self.overlapping_outcomes,
            "overlap_fraction": self.overlap_fraction,
            "overlap_description": self.overlap_description,
            "warnings": self.warnings,
        }


def autocorrelation_estimates(
    values: np.ndarray,
    max_lag: int = 10,
) -> dict[int, float]:
    """Sample autocorrelation function estimates at lags 1..max_lag."""
    n = len(values)
    vals = values[np.isfinite(values)]
    n_valid = len(vals)
    if n_valid < 3:
        return {}
    mean = np.mean(vals)
    var = np.var(vals, ddof=1)
    if var == 0.0:
        return {lag: 0.0 for lag in range(1, max_lag + 1)}
    acf: dict[int, float] = {}
    for lag in range(1, max_lag + 1):
        if lag >= n_valid:
            break
        cov = np.sum((vals[: n_valid - lag] - mean) * (vals[lag:] - mean)) / (n_valid - lag)
        acf[lag] = float(cov / var)
    return acf


def ljung_box_test(
    values: np.ndarray,
    max_lag: int = 10,
) -> tuple[float, float]:
    """Ljung-Box test statistic and approximate p-value.

    H0: no autocorrelation up to max_lag.
    """
    from scipy import stats as sp_stats

    n = len(values)
    vals = values[np.isfinite(values)]
    n_valid = len(vals)
    if n_valid < max_lag + 2:
        return 0.0, 1.0
    acf = autocorrelation_estimates(vals, max_lag)
    q = n_valid * (n_valid + 2) * sum(
        acf.get(lag, 0.0) ** 2 / (n_valid - lag)
        for lag in range(1, max_lag + 1)
    )
    p_value = 1.0 - sp_stats.chi2.cdf(q, df=max_lag)
    return float(q), float(p_value)


def effective_sample_size_iid(
    values: np.ndarray,
) -> tuple[int | None, str | None]:
    """Effective sample size for i.i.d. samples (trivial: just count valid).

    Returns None if the concept does not apply.
    """
    n_valid = int(np.sum(np.isfinite(values)))
    if n_valid < 2:
        return None, None
    return n_valid, "count_finite"


def effective_sample_size_autocorrelation(
    values: np.ndarray,
    max_lag: int = 10,
) -> tuple[int | None, str | None]:
    """Effective sample size accounting for autocorrelation.

    Uses the approximation:
        n_eff = n / (1 + 2 * sum_{k=1}^{K} (1 - k/n) * rho(k))

    where rho(k) are sample autocorrelations. If the sum of
    autocorrelations is negative (antithetic), n_eff > n and we cap at n.

    This is an APPROXIMATE method; the result is labeled as such.
    """
    vals = values[np.isfinite(values)]
    n = len(vals)
    if n < 3:
        return None, None
    acf = autocorrelation_estimates(vals, max_lag)
    cum = 0.0
    for lag in range(1, max_lag + 1):
        rho = acf.get(lag, 0.0)
        weight = 1.0 - lag / n
        cum += 2.0 * weight * rho
    denominator = 1.0 + cum
    if denominator <= 0:
        return n, "autocorrelation_corrected_capped"
    n_eff = n / denominator
    return max(1, int(round(n_eff))), "autocorrelation_corrected_approximate"


def detect_overlapping_outcomes(
    decision_sessions: np.ndarray,
    horizon_sessions: int = 5,
) -> dict[str, Any]:
    """Detect whether outcomes overlap due to multi-session label horizon.

    For LAB-004 (5-session forward return), the label at session D
    overlaps with the label at session D+1 through D+4 (each depends
    on overlapping future price observations).

    Returns a report of the overlap structure.
    """
    sessions = np.sort(np.unique(decision_sessions))
    n_sessions = len(sessions)
    if n_sessions < 2:
        return {
            "overlapping": False,
            "horizon_sessions": horizon_sessions,
            "overlap_fraction": 0.0,
            "description": "insufficient sessions to assess overlap",
        }
    # Compute overlap fraction: what fraction of consecutive sessions are
    # within horizon_sessions of each other
    diffs = np.diff(sessions)
    within_horizon = np.sum(diffs < horizon_sessions)
    overlap_frac = float(within_horizon / len(diffs)) if len(diffs) > 0 else 0.0
    return {
        "overlapping": overlap_frac > 0.0,
        "horizon_sessions": horizon_sessions,
        "overlap_fraction": overlap_frac,
        "n_sessions": n_sessions,
        "session_gap_distribution": {
            "min": float(np.min(diffs)) if len(diffs) > 0 else None,
            "max": float(np.max(diffs)) if len(diffs) > 0 else None,
            "median": float(np.median(diffs)) if len(diffs) > 0 else None,
        },
        "description": (
            f"horizon={horizon_sessions} sessions; "
            f"{overlap_frac:.1%} of consecutive session pairs are within "
            f"the {horizon_sessions}-session window"
        ),
    }


def run_dependence_diagnostics(
    values: np.ndarray,
    decision_sessions: np.ndarray | None = None,
    horizon_sessions: int = 5,
    max_lag: int = 10,
    label: str = "",
) -> DependenceReport:
    """Full dependence diagnostic suite for a metric series."""
    warnings: list[str] = []
    vals = values[np.isfinite(values)]

    # Autocorrelation
    acf = autocorrelation_estimates(vals, max_lag)
    significant_lags = [lag for lag, rho in acf.items() if abs(rho) > 0.2]
    if significant_lags:
        warnings.append(
            f"significant autocorrelation at lags {significant_lags} "
            f"(|rho| > 0.2); i.i.d. assumption may be violated"
        )

    # Ljung-Box
    q_stat, p_value = ljung_box_test(vals, max_lag)
    if p_value < 0.05:
        warnings.append(
            f"Ljung-Box test rejects no-autocorrelation at 5% "
            f"(Q={q_stat:.2f}, p={p_value:.3f})"
        )

    # Effective sample size
    n_eff_iid, method_iid = effective_sample_size_iid(vals)
    n_eff_ac, method_ac = effective_sample_size_autocorrelation(vals, max_lag)
    if n_eff_ac is not None and n_eff_iid is not None:
        reduction = 1.0 - n_eff_ac / n_eff_iid
        if reduction > 0.2:
            warnings.append(
                f"autocorrelation reduces effective sample size by "
                f"{reduction:.1%} ({n_eff_iid} -> {n_eff_ac})"
            )

    # Overlap detection
    if decision_sessions is not None:
        overlap = detect_overlapping_outcomes(decision_sessions, horizon_sessions)
    else:
        overlap = {
            "overlapping": False,
            "horizon_sessions": horizon_sessions,
            "overlap_fraction": 0.0,
            "description": "no session data provided; overlap not assessed",
        }

    if overlap["overlapping"] and overlap["overlap_fraction"] > 0.5:
        warnings.append(
            f"high overlap fraction ({overlap['overlap_fraction']:.1%}) "
            f"due to {horizon_sessions}-session label horizon; "
            f"observations are NOT independent"
        )

    return DependenceReport(
        autocorrelations=acf,
        ljung_box_statistic=q_stat,
        ljung_box_p_value=p_value,
        effective_sample_size=n_eff_ac if n_eff_ac is not None else n_eff_iid,
        effective_sample_size_method=method_ac if n_eff_ac is not None else method_iid,
        overlapping_outcomes=overlap["overlapping"],
        overlap_fraction=overlap["overlap_fraction"],
        overlap_description=overlap.get("description", ""),
        warnings=warnings,
    )


__all__ = [
    "DependenceReport",
    "autocorrelation_estimates",
    "ljung_box_test",
    "effective_sample_size_iid",
    "effective_sample_size_autocorrelation",
    "detect_overlapping_outcomes",
    "run_dependence_diagnostics",
]
