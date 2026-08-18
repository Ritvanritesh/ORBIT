"""Seed label contracts for the Phase 1 registered hypotheses.

The three seed hypotheses promise specific prediction targets; Phase 5 makes
those targets operationally computable. The registry here registers the
matching contracts:

    LAB-001  H-001 momentum     5-session forward excess TOTAL return vs SPY
                                (decision-instant anchor)
    LAB-003  H-003 PEAD         5-session forward excess TOTAL return vs SPY
                                (POST_EVENT anchor: entry is the first
                                session after the filing's point-in-time
                                availability instant)

H-002's risk-adjusted-return label is NOT registered: it is a composite
label (future outcome divided by a point-in-time trailing-volatility
denominator) whose assembly is deferred to a later phase; the components it
needs (forward return, volatility outcomes) exist in Phase 5.

The benchmark field identifies the benchmark series in the engine's bars
universe; Phase 2's benchmark instrument set (SPY + broad/sector ETFs) is a
documented follow-up, so excess-return labels resolve only once the
benchmark bars exist.
"""

from __future__ import annotations

from orbit.labels.contract import (
    AnchorMode,
    LabelContract,
    ReturnConvention,
)
from orbit.labels.registry import LabelVersionRegistry

SPY_BENCHMARK = "SPY"


def build_seed_label_registry() -> LabelVersionRegistry:
    reg = LabelVersionRegistry()

    reg.register(
        LabelContract(
            label_id="LAB-001",
            version="v1",
            target_type="excess_return",
            horizon=5,
            anchor_mode=AnchorMode.DECISION_INSTANT,
            return_convention=ReturnConvention.SIMPLE_TOTAL_RETURN,
            benchmark=SPY_BENCHMARK,
            formula=(
                "5-session forward total return (split-continuous closes with "
                "ex-date dividends reinvested at the ex-date close) minus SPY "
                "total return over its own 5 sessions after the same decision "
                "instant, close-to-close."
            ),
            description="seed H-001 momentum prediction target",
        ),
        note="seed hypothesis H-001 (12-1 momentum)",
    )

    reg.register(
        LabelContract(
            label_id="LAB-003",
            version="v1",
            target_type="excess_return",
            horizon=5,
            anchor_mode=AnchorMode.POST_EVENT,
            return_convention=ReturnConvention.SIMPLE_TOTAL_RETURN,
            benchmark=SPY_BENCHMARK,
            formula=(
                "5-session forward total return (split-continuous closes with "
                "ex-date dividends reinvested at the ex-date close) minus SPY "
                "total return over its own 5 sessions after the same anchor, "
                "measured from the trading day after the earliest point-in-"
                "time publication timestamp of the filing (POST_EVENT anchor)."
            ),
            description="seed H-003 PEAD prediction target",
        ),
        note="seed hypothesis H-003 (post-earnings announcement drift)",
    )

    return reg


__all__ = ["SPY_BENCHMARK", "build_seed_label_registry"]