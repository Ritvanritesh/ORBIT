"""Phase 11 practical power analysis.

NOT a license to retrospectively manipulate sample size. Provides
approximate answers to:
  - What sample size is needed to detect an effect of size d?
  - What effect size can the current design realistically detect?

All calculations are labeled APPROXIMATE where assumptions are
simplified (e.g., independent-sample formulas for dependent data).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PowerResult:
    """Power analysis result with explicit assumptions."""

    scenario: str
    assumed_effect_size: float
    sample_size: int
    effective_sample_size: int | None
    alpha: float
    target_power: float
    achieved_power: float | None
    method: str
    assumptions: list[str]
    label: str  # "APPROXIMATE" or "EXACT"

    def summary(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "assumed_effect_size": self.assumed_effect_size,
            "sample_size": self.sample_size,
            "effective_sample_size": self.effective_sample_size,
            "alpha": self.alpha,
            "target_power": self.target_power,
            "achieved_power": self.achieved_power,
            "method": self.method,
            "assumptions": self.assumptions,
            "label": self.label,
        }


def required_sample_size_independent(
    effect_size: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> PowerResult:
    """Sample size for a two-sided z-test (i.i.d. normal observations).

    This is APPROXIMATE for financial data where observations are
    typically dependent and non-normal. The result is labeled as such.
    """
    from scipy import stats as sp_stats

    if effect_size == 0:
        return PowerResult(
            scenario="required_sample_size",
            assumed_effect_size=0.0,
            sample_size=0,
            effective_sample_size=None,
            alpha=alpha,
            target_power=power,
            achieved_power=None,
            method="normal_two_sided",
            assumptions=["i.i.d. normal observations", "known variance"],
            label="APPROXIMATE",
        )
    z_alpha = sp_stats.norm.ppf(1.0 - alpha / 2)
    z_beta = sp_stats.norm.ppf(power)
    n = int(np.ceil(((z_alpha + z_beta) / effect_size) ** 2))
    return PowerResult(
        scenario="required_sample_size",
        assumed_effect_size=effect_size,
        sample_size=n,
        effective_sample_size=None,
        alpha=alpha,
        target_power=power,
        achieved_power=None,
        method="normal_two_sided",
        assumptions=[
            "i.i.d. normal observations",
            "known variance",
            "effect is a standardized mean difference",
            "APPROXIMATE for financial data with dependent observations",
        ],
        label="APPROXIMATE",
    )


def achieved_power_independent(
    effect_size: float,
    sample_size: int,
    alpha: float = 0.05,
) -> PowerResult:
    """Achieved power given sample size and effect size (i.i.d. normal)."""
    from scipy import stats as sp_stats

    if effect_size == 0 or sample_size == 0:
        return PowerResult(
            scenario="achieved_power",
            assumed_effect_size=effect_size,
            sample_size=sample_size,
            effective_sample_size=None,
            alpha=alpha,
            target_power=0.0,
            achieved_power=alpha,
            method="normal_two_sided",
            assumptions=["i.i.d. normal observations"],
            label="APPROXIMATE",
        )
    z_alpha = sp_stats.norm.ppf(1.0 - alpha / 2)
    ncp = effect_size * np.sqrt(sample_size)
    power = 1.0 - sp_stats.norm.cdf(z_alpha - ncp) + sp_stats.norm.cdf(-z_alpha - ncp)
    return PowerResult(
        scenario="achieved_power",
        assumed_effect_size=effect_size,
        sample_size=sample_size,
        effective_sample_size=None,
        alpha=alpha,
        target_power=power,
        achieved_power=float(power),
        method="normal_two_sided",
        assumptions=[
            "i.i.d. normal observations",
            "known variance",
            "APPROXIMATE for financial data",
        ],
        label="APPROXIMATE",
    )


def min_detectable_effect(
    sample_size: int,
    alpha: float = 0.05,
    power: float = 0.80,
) -> PowerResult:
    """Minimum detectable effect size given sample size and target power."""
    from scipy import stats as sp_stats

    if sample_size < 2:
        return PowerResult(
            scenario="min_detectable_effect",
            assumed_effect_size=float("inf"),
            sample_size=sample_size,
            effective_sample_size=None,
            alpha=alpha,
            target_power=power,
            achieved_power=None,
            method="normal_two_sided",
            assumptions=["insufficient sample size"],
            label="APPROXIMATE",
        )
    z_alpha = sp_stats.norm.ppf(1.0 - alpha / 2)
    z_beta = sp_stats.norm.ppf(power)
    mde = (z_alpha + z_beta) / np.sqrt(sample_size)
    return PowerResult(
        scenario="min_detectable_effect",
        assumed_effect_size=float(mde),
        sample_size=sample_size,
        effective_sample_size=None,
        alpha=alpha,
        target_power=power,
        achieved_power=None,
        method="normal_two_sided",
        assumptions=[
            "i.i.d. normal observations",
            "known variance",
            f"MDE = {mde:.4f} at n={sample_size}, alpha={alpha}, power={power}",
            "APPROXIMATE for financial data",
        ],
        label="APPROXIMATE",
    )


def power_with_autocorrelation_adjustment(
    effect_size: float,
    sample_size: int,
    effective_sample_size: int,
    alpha: float = 0.05,
) -> PowerResult:
    """Approximate power accounting for autocorrelation via n_eff.

    Uses the effective sample size in place of the nominal sample size.
    This is a simple correction that ignores distributional departures.
    """
    achieved = achieved_power_independent(
        effect_size, effective_sample_size, alpha
    )
    return PowerResult(
        scenario="achieved_power_autocorrelation_adjusted",
        assumed_effect_size=effect_size,
        sample_size=sample_size,
        effective_sample_size=effective_sample_size,
        alpha=alpha,
        target_power=achieved.target_power,
        achieved_power=achieved.achieved_power,
        method="normal_two_sided_n_eff",
        assumptions=[
            f"autocorrelation reduces effective n from {sample_size} to {effective_sample_size}",
            "otherwise same as i.i.d. normal",
            "APPROXIMATE",
        ],
        label="APPROXIMATE",
    )


__all__ = [
    "PowerResult",
    "required_sample_size_independent",
    "achieved_power_independent",
    "min_detectable_effect",
    "power_with_autocorrelation_adjustment",
]
