"""Phase 11 multiple-comparison awareness and control.

Phase 10 contains 52 experiment outcomes. This module explicitly analyzes
the researcher search space represented by those experiments and provides
appropriate multiple-testing corrections.

The hypothesis family is LOCKED: all 52 Phase 10 experiments form a
single family. No post-hoc exclusion of inconvenient experiments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import stats as sp_stats


@dataclass(frozen=True)
class MultipleTestingResult:
    """Result of multiple-testing correction on a family of hypotheses."""

    family_id: str
    method: str
    n_hypotheses: int
    raw_p_values: list[float]
    adjusted_p_values: list[float]
    experiment_ids: list[str]
    significant_at: dict[str, list[str]]
    assumptions: str

    def summary(self) -> dict[str, Any]:
        return {
            "family_id": self.family_id,
            "method": self.method,
            "n_hypotheses": self.n_hypotheses,
            "n_significant_005": len(self.significant_at.get("0.05", [])),
            "n_significant_001": len(self.significant_at.get("0.01", [])),
            "experiment_ids_significant_005": self.significant_at.get("0.05", []),
            "assumptions": self.assumptions,
        }


def holm_bonferroni(
    p_values: list[float],
    experiment_ids: list[str],
    family_id: str = "phase10_grid",
) -> MultipleTestingResult:
    """Holm-Bonferroni step-down correction (family-wise error rate).

    Controls the probability of making ANY false rejection among the family.
    More powerful than Bonferroni but still conservative.
    """
    n = len(p_values)
    if n == 0:
        return MultipleTestingResult(
            family_id=family_id,
            method="holm_bonferroni",
            n_hypotheses=0,
            raw_p_values=[],
            adjusted_p_values=[],
            experiment_ids=[],
            significant_at={"0.05": [], "0.01": []},
            assumptions="empty family",
        )
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = np.full(n, np.nan)
    for rank, (orig_idx, p) in enumerate(indexed):
        adj = min(1.0, p * (n - rank))
        # Step-down: must be at least as large as previous
        if rank > 0:
            prev_adj = adjusted[indexed[rank - 1][0]]
            adj = max(adj, prev_adj)
        adjusted[orig_idx] = adj

    sig_005 = [
        experiment_ids[i]
        for i in range(n)
        if adjusted[i] <= 0.05
    ]
    sig_001 = [
        experiment_ids[i]
        for i in range(n)
        if adjusted[i] <= 0.01
    ]

    return MultipleTestingResult(
        family_id=family_id,
        method="holm_bonferroni",
        n_hypotheses=n,
        raw_p_values=p_values,
        adjusted_p_values=adjusted.tolist(),
        experiment_ids=experiment_ids,
        significant_at={"0.05": sig_005, "0.01": sig_001},
        assumptions=(
            "tests are independent or positively dependent; "
            "controls family-wise error rate at 5%"
        ),
    )


def benjamini_hochberg(
    p_values: list[float],
    experiment_ids: list[str],
    family_id: str = "phase10_grid",
) -> MultipleTestingResult:
    """Benjamini-Hochberg procedure (false discovery rate).

    Controls the expected fraction of false rejections among rejections.
    Less conservative than Holm-Bonferroni, especially when many tests
    are truly non-significant.
    """
    n = len(p_values)
    if n == 0:
        return MultipleTestingResult(
            family_id=family_id,
            method="benjamini_hochberg",
            n_hypotheses=0,
            raw_p_values=[],
            adjusted_p_values=[],
            experiment_ids=[],
            significant_at={"0.05": [], "0.01": []},
            assumptions="empty family",
        )
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = np.full(n, np.nan)
    # BH step-up
    prev_adj = 1.0
    for rank in range(n - 1, -1, -1):
        orig_idx, p = indexed[rank]
        adj = min(1.0, p * n / (rank + 1))
        adj = min(adj, prev_adj)
        adjusted[orig_idx] = adj
        prev_adj = adj

    sig_005 = [
        experiment_ids[i]
        for i in range(n)
        if adjusted[i] <= 0.05
    ]
    sig_001 = [
        experiment_ids[i]
        for i in range(n)
        if adjusted[i] <= 0.01
    ]

    return MultipleTestingResult(
        family_id=family_id,
        method="benjamini_hochberg",
        n_hypotheses=n,
        raw_p_values=p_values,
        adjusted_p_values=adjusted.tolist(),
        experiment_ids=experiment_ids,
        significant_at={"0.05": sig_005, "0.01": sig_001},
        assumptions=(
            "tests are independent or positively correlated; "
            "controls expected false discovery rate at 5%"
        ),
    )


def define_phase10_family() -> dict[str, Any]:
    """Define and lock the Phase 10 hypothesis family.

    The full 52-experiment grid (13 feature sets x 4 models) forms
    a single comparison family. No post-hoc exclusion is permitted.
    """
    from orbit.ml.phase11_plan import PHASE10_EXPERIMENT_FAMILY

    return {
        "family_id": "phase10_grid",
        "description": (
            "The full Phase 10 ablation grid: 13 feature sets x 4 model "
            "families = 52 experiments. All belong to the same researcher "
            "search space; no exclusion permitted."
        ),
        "members": PHASE10_EXPERIMENT_FAMILY,
        "n_members": len(PHASE10_EXPERIMENT_FAMILY),
        "locked": True,
        "rationale": (
            "Phase 10 is a pre-registered ablation study. All 52 experiments "
            "were part of the original design. Removing any post-hoc would "
            "introduce selection bias into the inference."
        ),
    }


def multiple_testing_analysis(
    p_values: list[float],
    experiment_ids: list[str],
    family_id: str = "phase10_grid",
) -> dict[str, Any]:
    """Run multiple-testing corrections using both Holm and BH methods."""
    holm = holm_bonferroni(p_values, experiment_ids, family_id)
    bh = benjamini_hochberg(p_values, experiment_ids, family_id)
    return {
        "family": define_phase10_family(),
        "holm_bonferroni": holm.summary(),
        "benjamini_hochberg": bh.summary(),
        "n_raw_significant_005": sum(1 for p in p_values if p <= 0.05),
        "n_raw_significant_001": sum(1 for p in p_values if p <= 0.01),
    }


__all__ = [
    "MultipleTestingResult",
    "holm_bonferroni",
    "benjamini_hochberg",
    "define_phase10_family",
    "multiple_testing_analysis",
]
