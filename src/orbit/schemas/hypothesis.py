"""HypothesisSpec: pre-registered, falsifiable research hypotheses (Phase 1).

Every hypothesis must be fully described *before* feature exploration. The
falsification criteria are binding: results cannot move the criteria.
"""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, ConfigDict, Field, model_validator

from orbit.schemas.common import (
    CostModel,
    EvidenceType,
    Horizon,
    LabelType,
    LeakageClass,
    HypothesisStatus,
    UniverseScope,
)


class LabelSpec(BaseModel):
    """The exact outcome a hypothesis predicts. Frozen once registered."""

    label_type: LabelType
    horizon: Horizon
    benchmark: str | None = Field(
        default=None,
        description="Benchmark for EXCESS_RETURN labels, e.g. SPY.",
    )
    label_version: str = Field(default="v1")
    definition: str = Field(
        description="Exact mathematical definition. Must be unambiguous."
    )
    overlap_policy: str = Field(
        default="purge+embargo",
        description="How overlapping horizons are handled in validation.",
    )


class EconomicEvidence(BaseModel):
    """Pre-registered thresholds that constitute *economic* evidence.

    Roadmap 12: statistical significance is not economic significance.
    These thresholds are binding before any experiment runs.
    """

    oos_rank_ic: float | None = Field(
        default=None,
        ge=-1.0,
        le=1.0,
        description="Minimum out-of-sample rank IC to count as economic evidence.",
    )
    after_cost_annual_excess: float | None = Field(
        default=None,
        description="Minimum after-cost annualized excess return vs benchmark (e.g. 0.03 = 3%).",
    )
    min_regimes: int = Field(
        default=3,
        ge=1,
        description="Minimum distinct market regimes where the effect must survive.",
    )
    min_walkforward_windows: int = Field(
        default=4,
        ge=1,
        description="Minimum sequential OOS windows where the effect must survive.",
    )
    cost_model: CostModel = Field(
        default_factory=CostModel,
        description="Conservative cost assumptions for economic evidence.",
    )


class ResearchBudget(BaseModel):
    """Maximum research spend per hypothesis family (governance control)."""

    max_trials: int = Field(default=20, ge=1)
    max_parameter_sets: int = Field(default=50, ge=1)
    review_after_trials: int = Field(
        default=10, ge=1,
        description="Trials after which a review is mandatory.",
    )


class HypothesisSpec(BaseModel):
    """A falsifiable research claim, fully specified before execution.

    Immutable: criteria cannot change once constructed. To change a
    registered hypothesis, create a new version.
    """

    model_config = ConfigDict(frozen=True)

    hypothesis_id: str = Field(pattern=r"^H-\d{3}$")
    title: str
    version: str = Field(default="v1")
    author: str = "orbit-research"

    statement: str = Field(
        description="The falsifiable claim, one or two sentences, no hedging."
    )
    mechanism: str = Field(
        description="Why the effect might exist (economic/behavioral rationale)."
    )
    baseline: list[str] = Field(
        min_length=1,
        description="Control strategies the hypothesis must beat after costs.",
    )

    universe: UniverseScope
    label: LabelSpec
    feature_families: list[str] = Field(
        min_length=1,
        description="Feature namespaces allowed: momentum, reversal, volatility, liquidity, relative_strength, fundamentals, market_regime.",
    )
    leakage_class: LeakageClass = LeakageClass.NONE
    data_sources: list[str] = Field(
        min_length=1,
        description="Approved data sources (e.g. licensed market data, SEC EDGAR/XBRL, FRED/ALFRED).",
    )

    economic_evidence: EconomicEvidence
    falsification_criteria: str = Field(
        description="The exact conditions under which this hypothesis is rejected."
    )
    non_goals: list[str] = Field(
        default_factory=list,
        description="What this hypothesis explicitly is NOT trying to show.",
    )

    evidence_type: EvidenceType = EvidenceType.ECONOMIC
    status: HypothesisStatus = HypothesisStatus.DRAFT
    registration_date: date | None = None

    research_budget: ResearchBudget = Field(default_factory=ResearchBudget)

    @model_validator(mode="after")
    def _status_vs_registration(self) -> "HypothesisSpec":
        if (
            self.status in (HypothesisStatus.REGISTERED, HypothesisStatus.ACTIVE)
            and self.registration_date is None
        ):
            raise ValueError("registered/active hypotheses require registration_date")
        return self

    def register(self) -> "HypothesisSpec":
        """Promote to REGISTERED, freezing the spec. Criteria cannot change after this."""
        if self.status not in (HypothesisStatus.DRAFT, HypothesisStatus.PROPOSED):
            raise ValueError(f"cannot register hypothesis in status {self.status.value}")
        return self.model_copy(
            update={
                "status": HypothesisStatus.REGISTERED,
                "registration_date": date.today(),
            }
        )


class HypothesisRegistry(BaseModel):
    """Registry of all pre-registered hypotheses (Phase 1 governance)."""

    hypotheses: list[HypothesisSpec] = Field(default_factory=list)

    def get(self, hypothesis_id: str) -> HypothesisSpec:
        for h in self.hypotheses:
            if h.hypothesis_id == hypothesis_id:
                return h
        raise KeyError(f"unknown hypothesis: {hypothesis_id}")

    def register(self, spec: HypothesisSpec) -> HypothesisSpec:
        if spec.hypothesis_id in {h.hypothesis_id for h in self.hypotheses}:
            raise ValueError(f"duplicate hypothesis id: {spec.hypothesis_id}")
        registered = spec.register()
        self.hypotheses.append(registered)
        return registered

    def register_all(self) -> "HypothesisRegistry":
        """Freeze every draft/proposed hypothesis (idempotent, strict on status)."""
        return HypothesisRegistry(
            hypotheses=[
                h.register()
                if h.status in (HypothesisStatus.DRAFT, HypothesisStatus.PROPOSED)
                else h
                for h in self.hypotheses
            ]
        )