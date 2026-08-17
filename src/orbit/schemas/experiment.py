"""ExperimentSpec: one fully-specified experiment inside a hypothesis family.

Every experiment is a first-class object with genealogy (parent_id), pinned
datasets/features/model/windows/cost model, and a status lifecycle.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from orbit.schemas.common import CostModel, ExperimentStatus
from orbit.schemas.hypothesis import HypothesisRegistry


class WindowSpec(BaseModel):
    """Train / validation / test windows. Frozen at registration."""

    train_start: date
    train_end: date
    val_start: date
    val_end: date
    test_start: date
    test_end: date
    embargo_days: int = Field(default=5, ge=0)
    purge_days: int = Field(default=0, ge=0)

    @field_validator("train_end", "val_start", "val_end", "test_start", "test_end")
    @classmethod
    def _check_order(cls, v: date, info) -> date:
        field = info.field_name
        field_to_min = {
            "train_end": "train_start",
            "val_start": "train_end",
            "val_end": "val_start",
            "test_start": "val_end",
            "test_end": "test_start",
        }
        other = info.data.get(field_to_min.get(field, ""))
        if other is not None and v <= other:
            raise ValueError(f"{field} must be after {field_to_min[field]}")
        return v


class FeaturePin(BaseModel):
    """A pinned feature set version. Features are immutable once referenced."""

    feature_names: list[str] = Field(min_length=1)
    feature_version: str = Field(pattern=r"^v\d+(\.\d+)*$")


_KNOWN_MODEL_FAMILIES = frozenset(
    {
        "heuristic", "linear", "ridge", "logistic", "random_forest",
        "xgboost", "lightgbm", "null_shuffle",
    }
)


class ModelSpec(BaseModel):
    """Pinned model specification. No anonymous weights."""

    family: str = Field(
        description="e.g. heuristic, linear, ridge, logistic, random_forest, xgboost, lightgbm, null_shuffle"
    )
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    target_transform: str | None = None
    uncertainty: bool = Field(default=False)

    @field_validator("family")
    @classmethod
    def _family_known(cls, v: str) -> str:
        if v not in _KNOWN_MODEL_FAMILIES:
            raise ValueError(f"unknown model family: {v}")
        return v


class ExperimentSpec(BaseModel):
    """A single registered experiment. Immutable after registration."""

    model_config = ConfigDict(frozen=True)

    experiment_id: str = Field(pattern=r"^EXP-\d{5}$")
    hypothesis_id: str = Field(pattern=r"^H-\d{3}$")
    title: str
    parent_id: str | None = Field(
        default=None,
        pattern=r"^EXP-\d{5}$",
        description="Genealogy: variant experiments reference a parent.",
    )

    datasets: list[str] = Field(min_length=1)
    features: FeaturePin
    model: ModelSpec
    windows: WindowSpec
    cost_model: CostModel = Field(default_factory=CostModel)

    seed: int = Field(default=42)
    status: ExperimentStatus = ExperimentStatus.DRAFT
    code_hash: str | None = Field(
        default=None,
        description="Hash of the executing code/config; set at run time.",
    )
    notes: str | None = None

    def content_hash(self) -> str:
        """Deterministic hash of the full spec (excluding status/hash fields)."""
        payload = self.model_dump(exclude={"status", "code_hash"})
        raw = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class ExperimentRegistry(BaseModel):
    """Registry of all experiments with genealogy and trial budgets."""

    experiments: list[ExperimentSpec] = Field(default_factory=list)

    def get(self, experiment_id: str) -> ExperimentSpec:
        for e in self.experiments:
            if e.experiment_id == experiment_id:
                return e
        raise KeyError(f"unknown experiment: {experiment_id}")

    def trials_for(self, hypothesis_id: str) -> int:
        return sum(
            1
            for e in self.experiments
            if e.hypothesis_id == hypothesis_id
            and e.status not in (ExperimentStatus.DRAFT, ExperimentStatus.RETIRED)
        )

    def register(
        self,
        spec: ExperimentSpec,
        hypothesis_registry: HypothesisRegistry | None = None,
    ) -> ExperimentSpec:
        """Register an experiment, optionally enforcing hypothesis lineage and
        the hypothesis family's research budget.

        When a hypothesis registry is supplied, the experiment's
        hypothesis_id must exist there (governance: no orphaned results) and
        the hypothesis's declared max_trials budget is enforced.
        """
        if spec.experiment_id in {e.experiment_id for e in self.experiments}:
            raise ValueError(f"duplicate experiment id: {spec.experiment_id}")
        if spec.parent_id is not None and spec.parent_id not in {
            e.experiment_id for e in self.experiments
        }:
            raise ValueError(f"unknown parent experiment: {spec.parent_id}")
        if spec.parent_id is not None:
            parent = self.get(spec.parent_id)
            if parent.hypothesis_id != spec.hypothesis_id:
                raise ValueError(
                    f"parent {spec.parent_id} belongs to {parent.hypothesis_id}, "
                    f"not {spec.hypothesis_id}: genealogy is hypothesis-scoped"
                )
        if hypothesis_registry is not None:
            try:
                hyp = hypothesis_registry.get(spec.hypothesis_id)
            except KeyError:
                raise ValueError(
                    f"experiment references unknown hypothesis: {spec.hypothesis_id}"
                ) from None
            if self.trials_for(spec.hypothesis_id) >= hyp.research_budget.max_trials:
                raise ValueError(
                    f"research budget exhausted for {spec.hypothesis_id}: "
                    f"{hyp.research_budget.max_trials} trials maximum"
                )
        registered = spec.model_copy(update={"status": ExperimentStatus.REGISTERED})
        self.experiments.append(registered)
        return registered