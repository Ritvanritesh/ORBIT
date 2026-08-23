"""Phase 11 dependence-aware bootstrap framework.

Implements:
  1. I.i.d. bootstrap (for controlled synthetic tests)
  2. Moving block bootstrap (time-series-aware)
  3. Deterministic seeded sampling
  4. Explicit block-length configuration
  5. Block-length diagnostics

Block-length policy: rule-of-thumb based on the data's autocorrelation
structure (not optimized against Phase 10 outcomes).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np


@dataclass(frozen=True)
class BootstrapConfig:
    """Immutable bootstrap configuration record."""

    method: str  # "iid" | "moving_block"
    n_resamples: int
    block_length: int | None = None
    seed: int = 42
    confidence_level: float = 0.95

    def summary(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "n_resamples": self.n_resamples,
            "block_length": self.block_length,
            "seed": self.seed,
            "confidence_level": self.confidence_level,
        }


@dataclass(frozen=True)
class BootstrapResult:
    """Immutable bootstrap result with full lineage."""

    config: BootstrapConfig
    bootstrap_distribution: np.ndarray
    point_estimate: float
    ci_lower: float
    ci_upper: float
    se: float
    bias: float
    block_length_diagnostics: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return {
            "config": self.config.summary(),
            "point_estimate": self.point_estimate,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "se": self.se,
            "bias": self.bias,
            "n_resamples": len(self.bootstrap_distribution),
            "block_length_diagnostics": self.block_length_diagnostics,
        }


def _rng(seed: int) -> np.random.Generator:
    """Deterministic RNG from a seed."""
    return np.random.default_rng(seed)


def rule_of_thumb_block_length(n: int, autocorrelation: float | None = None) -> int:
    """Deterministic block-length selection using rule of thumb.

    For a sample of size n, the rule-of-thumb block length is:
        b = ceil(n^(1/3))

    This is a standard choice in the block bootstrap literature
    (Politis & White, 2004; Patton, Politis & White, 2009).

    If an autocorrelation estimate is provided, it is recorded in
    diagnostics but does not change the block length (to avoid
    optimizing against outcomes).
    """
    b = max(1, int(np.ceil(n ** (1.0 / 3.0))))
    return b


def block_length_diagnostics(
    data: np.ndarray,
    block_length: int,
) -> dict[str, Any]:
    """Report diagnostics about the chosen block length.

    Includes:
    - number of blocks
    - overlap fraction
    - effective coverage
    - sample size
    """
    n = len(data)
    if block_length <= 0:
        raise ValueError(f"block_length must be positive, got {block_length}")
    n_blocks = max(1, n - block_length + 1)
    coverage = n_blocks / n if n > 0 else 0.0
    return {
        "sample_size": n,
        "block_length": block_length,
        "n_potential_blocks": n_blocks,
        "coverage_ratio": coverage,
        "policy": "rule_of_thumb",
    }


def iid_bootstrap(
    data: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    n_resamples: int = 10000,
    seed: int = 42,
    confidence_level: float = 0.95,
) -> BootstrapResult:
    """Ordinary i.i.d. bootstrap.

    Valid ONLY when observations are independent and identically distributed.
    """
    rng = _rng(seed)
    n = len(data)
    boot_stats = np.empty(n_resamples, dtype=np.float64)
    for b in range(n_resamples):
        idx = rng.choice(n, size=n, replace=True)
        boot_stats[b] = statistic(data[idx])
    point = statistic(data)
    alpha = 1.0 - confidence_level
    ci_lo = float(np.percentile(boot_stats, 100 * alpha / 2))
    ci_hi = float(np.percentile(boot_stats, 100 * (1.0 - alpha / 2)))
    se = float(np.std(boot_stats, ddof=1))
    bias = float(np.mean(boot_stats) - point)
    return BootstrapResult(
        config=BootstrapConfig(
            method="iid",
            n_resamples=n_resamples,
            block_length=None,
            seed=seed,
            confidence_level=confidence_level,
        ),
        bootstrap_distribution=boot_stats,
        point_estimate=point,
        ci_lower=ci_lo,
        ci_upper=ci_hi,
        se=se,
        bias=bias,
        block_length_diagnostics={"method": "iid", "no_blocks": True},
    )


def moving_block_bootstrap(
    data: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    n_resamples: int = 10000,
    block_length: int | None = None,
    seed: int = 42,
    confidence_level: float = 0.95,
) -> BootstrapResult:
    """Moving block bootstrap for time-series data.

    Draws overlapping blocks of consecutive observations to preserve
    local dependence structure. Block length defaults to rule-of-thumb
    if not provided.
    """
    n = len(data)
    if block_length is None:
        block_length = rule_of_thumb_block_length(n)
    if block_length <= 0:
        raise ValueError(f"block_length must be positive, got {block_length}")
    if block_length > n:
        raise ValueError(
            f"block_length {block_length} exceeds sample size {n}"
        )

    rng = _rng(seed)
    n_blocks_needed = int(np.ceil(n / block_length))
    n_starts = n - block_length + 1  # number of valid block start positions

    boot_stats = np.empty(n_resamples, dtype=np.float64)
    for b in range(n_resamples):
        # Draw block start positions (with replacement)
        starts = rng.choice(n_starts, size=n_blocks_needed, replace=True)
        # Assemble the resampled series
        resampled_parts: list[np.ndarray] = []
        for start in starts:
            resampled_parts.append(data[start : start + block_length])
        resampled = np.concatenate(resampled_parts)[:n]  # trim to original length
        boot_stats[b] = statistic(resampled)

    point = statistic(data)
    alpha = 1.0 - confidence_level
    ci_lo = float(np.percentile(boot_stats, 100 * alpha / 2))
    ci_hi = float(np.percentile(boot_stats, 100 * (1.0 - alpha / 2)))
    se = float(np.std(boot_stats, ddof=1))
    bias = float(np.mean(boot_stats) - point)

    bl_diag = block_length_diagnostics(data, block_length)

    return BootstrapResult(
        config=BootstrapConfig(
            method="moving_block",
            n_resamples=n_resamples,
            block_length=block_length,
            seed=seed,
            confidence_level=confidence_level,
        ),
        bootstrap_distribution=boot_stats,
        point_estimate=point,
        ci_lower=ci_lo,
        ci_upper=ci_hi,
        se=se,
        bias=bias,
        block_length_diagnostics=bl_diag,
    )


def bootstrap_ci(
    data: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    method: str = "iid",
    n_resamples: int = 10000,
    block_length: int | None = None,
    seed: int = 42,
    confidence_level: float = 0.95,
) -> BootstrapResult:
    """Unified bootstrap dispatcher."""
    if method == "iid":
        return iid_bootstrap(
            data, statistic, n_resamples, seed, confidence_level
        )
    elif method == "moving_block":
        return moving_block_bootstrap(
            data, statistic, n_resamples, block_length, seed, confidence_level
        )
    else:
        raise ValueError(f"unsupported bootstrap method: {method!r}")


__all__ = [
    "BootstrapConfig",
    "BootstrapResult",
    "rule_of_thumb_block_length",
    "block_length_diagnostics",
    "iid_bootstrap",
    "moving_block_bootstrap",
    "bootstrap_ci",
]
