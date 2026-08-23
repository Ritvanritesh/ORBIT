"""Phase 11 effect size reporting.

Metric-specific effect sizes (not blindly applying Cohen's d to every
financial metric). Each result includes an interpretation category
where defensible, with documented thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from orbit.ml.phase11_plan import (
    IC_THRESHOLD_MEANINGFUL,
    IC_THRESHOLD_NEGIGLIBLE,
    RETURN_THRESHOLD_MEANINGFUL,
    RETURN_THRESHOLD_NEGIGLIBLE,
)


@dataclass(frozen=True)
class EffectSize:
    """Effect size record for a single metric."""

    metric: str
    magnitude: float
    method: str
    interpretation: str
    thresholds: dict[str, float]
    raw_value: float | None = None
    standardized_value: float | None = None

    def summary(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "magnitude": self.magnitude,
            "method": self.method,
            "interpretation": self.interpretation,
            "thresholds": self.thresholds,
            "raw_value": self.raw_value,
            "standardized_value": self.standardized_value,
        }


def ic_effect_size(ic: float) -> EffectSize:
    """IC-based effect size with documented interpretation.

    Thresholds (documented in phase11_plan.py):
      |IC| < 0.01: negligible
      0.01 <= |IC| < 0.03: small
      |IC| >= 0.03: potentially meaningful

    These are financial IC thresholds, not psychological Cohen's d.
    """
    magnitude = abs(ic)
    if magnitude < IC_THRESHOLD_NEGIGLIBLE:
        interp = "negligible"
    elif magnitude < IC_THRESHOLD_MEANINGFUL:
        interp = "small"
    else:
        interp = "potentially_meaningful"
    return EffectSize(
        metric="ic",
        magnitude=magnitude,
        method="raw_ic_magnitude",
        interpretation=interp,
        thresholds={
            "negligible": IC_THRESHOLD_NEGIGLIBLE,
            "meaningful": IC_THRESHOLD_MEANINGFUL,
        },
        raw_value=ic,
    )


def return_effect_size(
    after_cost_return: float,
    zero_cost_return: float | None = None,
) -> EffectSize:
    """After-cost return effect size.

    Thresholds:
      |return| < 0%: negligible (net loss)
      0% <= |return| < 10%: small
      |return| >= 10%: potentially meaningful
    """
    magnitude = abs(after_cost_return)
    if magnitude < abs(RETURN_THRESHOLD_NEGIGLIBLE):
        interp = "negligible"
    elif magnitude < RETURN_THRESHOLD_MEANINGFUL:
        interp = "small"
    else:
        interp = "potentially_meaningful"
    standardized = None
    if zero_cost_return is not None and abs(zero_cost_return) > 1e-12:
        standardized = after_cost_return / zero_cost_return
    return EffectSize(
        metric="after_cost_return",
        magnitude=magnitude,
        method="raw_return_magnitude",
        interpretation=interp,
        thresholds={
            "negligible": RETURN_THRESHOLD_NEGIGLIBLE,
            "meaningful": RETURN_THRESHOLD_MEANINGFUL,
        },
        raw_value=after_cost_return,
        standardized_value=standardized,
    )


def hit_rate_effect_size(hr: float) -> EffectSize:
    """Hit rate effect size.

    For a binary classification task:
      hr = 0.5: no skill (random)
      hr > 0.53: small skill
      hr > 0.55: moderate skill
    """
    magnitude = abs(hr - 0.5)
    if magnitude < 0.01:
        interp = "negligible"
    elif magnitude < 0.03:
        interp = "small"
    else:
        interp = "potentially_meaningful"
    return EffectSize(
        metric="hit_rate",
        magnitude=magnitude,
        method="hit_rate_excess_over_random",
        interpretation=interp,
        thresholds={
            "negligible": 0.01,
            "meaningful": 0.03,
        },
        raw_value=hr,
        standardized_value=magnitude,
    )


def ic_difference_effect(
    ic_a: float, ic_b: float, label_a: str = "A", label_b: str = "B"
) -> EffectSize:
    """Effect size for the difference between two ICs."""
    diff = ic_a - ic_b
    magnitude = abs(diff)
    if magnitude < 0.005:
        interp = "negligible"
    elif magnitude < 0.015:
        interp = "small"
    else:
        interp = "potentially_meaningful"
    return EffectSize(
        metric="ic_difference",
        magnitude=magnitude,
        method="raw_ic_difference",
        interpretation=interp,
        thresholds={
            "negligible": 0.005,
            "meaningful": 0.015,
        },
        raw_value=diff,
        standardized_value=None,
    )


def compute_effect_size(
    metric: str, value: float, **kwargs: Any
) -> EffectSize:
    """Unified effect-size dispatcher."""
    if metric in ("oos_ic", "rank_ic"):
        return ic_effect_size(value)
    elif metric == "after_cost_total_return":
        return return_effect_size(value, kwargs.get("zero_cost_return"))
    elif metric == "hit_rate":
        return hit_rate_effect_size(value)
    else:
        return EffectSize(
            metric=metric,
            magnitude=abs(value),
            method="raw_magnitude",
            interpretation="not_assessed",
            thresholds={},
            raw_value=value,
        )


def classify_statistical_evidence(
    p_value: float | None,
    confidence_level: float = 0.95,
) -> str:
    """Classify statistical evidence along a 3-level axis."""
    if p_value is None:
        return "inconclusive"
    alpha = 1.0 - confidence_level
    if p_value < alpha:
        return "evidence_under_stated_assumptions"
    elif p_value < alpha * 3:
        return "inconclusive"
    else:
        return "insufficient_evidence"


def classify_economic_meaning(
    effect: EffectSize,
) -> str:
    """Classify economic meaning along a 3-level axis."""
    if effect.interpretation == "negligible":
        return "negligible"
    elif effect.interpretation in ("small", "potentially_meaningful"):
        return "potentially_meaningful"
    else:
        return "not_assessable"


def significance_economy_matrix(
    p_value: float | None,
    effect: EffectSize,
    confidence_level: float = 0.95,
) -> dict[str, str]:
    """The explicit 2-axis interpretation matrix."""
    stat = classify_statistical_evidence(p_value, confidence_level)
    econ = classify_economic_meaning(effect)
    return {
        "statistical_evidence": stat,
        "economic_meaning": econ,
        "effect_size": effect.summary(),
        "p_value": p_value,
        "confidence_level": confidence_level,
    }


__all__ = [
    "EffectSize",
    "ic_effect_size",
    "return_effect_size",
    "hit_rate_effect_size",
    "ic_difference_effect",
    "compute_effect_size",
    "classify_statistical_evidence",
    "classify_economic_meaning",
    "significance_economy_matrix",
]
