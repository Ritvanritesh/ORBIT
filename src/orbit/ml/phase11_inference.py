"""Phase 11 confidence interval primitives.

Provides reproducible confidence intervals appropriate to each metric.
Every result records the method used, assumptions, sample size, and
confidence level. No method silently assumes i.i.d. observations.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np


@dataclass(frozen=True)
class ConfidenceInterval:
    """Immutable confidence interval record with full lineage."""

    point_estimate: float
    lower: float
    upper: float
    confidence_level: float
    method: str
    assumptions: str
    sample_size: int
    effective_sample_size: int | None = None
    seed: int | None = None
    n_resamples: int | None = None

    def contains(self, value: float) -> bool:
        return self.lower <= value <= self.upper

    def width(self) -> float:
        return self.upper - self.lower

    def summary(self) -> dict[str, Any]:
        return {
            "point_estimate": self.point_estimate,
            "lower": self.lower,
            "upper": self.upper,
            "confidence_level": self.confidence_level,
            "method": self.method,
            "assumptions": self.assumptions,
            "sample_size": self.sample_size,
            "effective_sample_size": self.effective_sample_size,
            "seed": self.seed,
            "n_resamples": self.n_resamples,
        }


@dataclass(frozen=True)
class InferenceResult:
    """A complete inference record for one metric on one source."""

    inference_result_id: str
    source_experiment_ids: list[str]
    source_artifact_checksums: dict[str, str]
    metric: str
    ci: ConfidenceInterval
    p_value: float | None = None
    adjusted_p_value: float | None = None
    effect_size: float | None = None
    effect_size_method: str | None = None
    dependence_diagnostics: dict[str, Any] = field(default_factory=dict)
    bootstrap_config: dict[str, Any] = field(default_factory=dict)
    seed: int = 42
    inference_plan_digest: str = ""

    def summary(self) -> dict[str, Any]:
        return {
            "inference_result_id": self.inference_result_id,
            "source_experiment_ids": self.source_experiment_ids,
            "source_artifact_checksums": self.source_artifact_checksums,
            "metric": self.metric,
            "point_estimate": self.ci.point_estimate,
            "ci_lower": self.ci.lower,
            "ci_upper": self.ci.upper,
            "confidence_level": self.ci.confidence_level,
            "method": self.ci.method,
            "p_value": self.p_value,
            "adjusted_p_value": self.adjusted_p_value,
            "effect_size": self.effect_size,
            "effect_size_method": self.effect_size_method,
            "sample_size": self.ci.sample_size,
            "effective_sample_size": self.ci.effective_sample_size,
            "seed": self.seed,
            "inference_plan_digest": self.inference_plan_digest,
        }


def make_inference_result_id(
    source_experiment_ids: list[str],
    metric: str,
    method: str,
    seed: int,
) -> str:
    """Deterministic inference result ID from lineage components."""
    raw = f"INF|{','.join(sorted(source_experiment_ids))}|{metric}|{method}|{seed}"
    h = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
    return f"INF-{h}"


def normal_ci(
    values: np.ndarray,
    confidence_level: float = 0.95,
) -> tuple[float, float]:
    """Simple normal-approximation CI from a sample of statistics.

    This is appropriate ONLY when the sampling distribution of the statistic
    is approximately normal (e.g., sample mean of a non-pathological variable
    with sufficient sample size).
    """
    n = len(values)
    if n < 2:
        raise ValueError("need at least 2 values for normal CI")
    from scipy import stats as sp_stats

    alpha = 1.0 - confidence_level
    z = float(sp_stats.norm.ppf(1.0 - alpha / 2.0))
    se = float(np.std(values, ddof=1)) / np.sqrt(n)
    mean = float(np.mean(values))
    return mean - z * se, mean + z * se


def percentile_ci(
    values: np.ndarray,
    confidence_level: float = 0.95,
) -> tuple[float, float]:
    """Percentile bootstrap CI from a sample of bootstrap statistics.

    This does NOT require normality of the sampling distribution.
    """
    n = len(values)
    if n < 2:
        raise ValueError("need at least 2 values for percentile CI")
    alpha = 1.0 - confidence_level
    lo = float(np.percentile(values, 100 * alpha / 2))
    hi = float(np.percentile(values, 100 * (1.0 - alpha / 2)))
    return lo, hi


def t_ci(
    values: np.ndarray,
    confidence_level: float = 0.95,
) -> tuple[float, float]:
    """Student-t CI from a sample of observations (assumes i.i.d.)."""
    from scipy import stats as sp_stats

    n = len(values)
    if n < 2:
        raise ValueError("need at least 2 values for t CI")
    alpha = 1.0 - confidence_level
    mean = float(np.mean(values))
    se = float(np.std(values, ddof=1)) / np.sqrt(n)
    t_crit = float(sp_stats.t.ppf(1.0 - alpha / 2, df=n - 1))
    return mean - t_crit * se, mean + t_crit * se


def compute_ci(
    values: np.ndarray,
    method: str,
    confidence_level: float = 0.95,
    **kwargs: Any,
) -> ConfidenceInterval:
    """Dispatch to a named CI method with full lineage."""
    methods: dict[str, Callable[..., tuple[float, float]]] = {
        "normal": lambda v, cl, **kw: normal_ci(v, cl),
        "percentile": lambda v, cl, **kw: percentile_ci(v, cl),
        "t": lambda v, cl, **kw: t_ci(v, cl),
    }
    if method not in methods:
        raise ValueError(f"unsupported CI method: {method!r}")
    lo, hi = methods[method](values, confidence_level, **kwargs)
    point = float(np.mean(values))
    return ConfidenceInterval(
        point_estimate=point,
        lower=lo,
        upper=hi,
        confidence_level=confidence_level,
        method=method,
        assumptions=kwargs.get("assumptions", f"{method} approximation"),
        sample_size=int(len(values)),
        effective_sample_size=kwargs.get("effective_sample_size"),
        seed=kwargs.get("seed"),
        n_resamples=kwargs.get("n_resamples"),
    )


def ic_ci_from_sessions(
    session_ics: np.ndarray,
    confidence_level: float = 0.95,
    method: str = "t",
) -> ConfidenceInterval:
    """CI for mean IC from per-session IC values.

    When per-session ICs are available, this is the most appropriate
    approach because the per-session aggregation already accounts for
    cross-sectional correlation within sessions.
    """
    values = session_ics[np.isfinite(session_ics)]
    if len(values) < 2:
        raise ValueError("need at least 2 valid session ICs for CI")
    ci = compute_ci(values, method=method, confidence_level=confidence_level)
    return ConfidenceInterval(
        point_estimate=float(np.mean(values)),
        lower=ci.lower,
        upper=ci.upper,
        confidence_level=confidence_level,
        method=f"session_mean_{method}",
        assumptions=f"per-session ICs are approximately {('i.i.d.' if method == 't' else 'normal')}",
        sample_size=int(len(values)),
        effective_sample_size=None,
    )


__all__ = [
    "ConfidenceInterval",
    "InferenceResult",
    "make_inference_result_id",
    "normal_ci",
    "percentile_ci",
    "t_ci",
    "compute_ci",
    "ic_ci_from_sessions",
]
