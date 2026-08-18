"""ExperimentSpec: one fully-specified experiment inside a hypothesis family.

Every experiment is a first-class, immutable research object with genealogy
(parent_id), pinned datasets/features/model/windows/cost model, and a
validated status lifecycle (Phase 6).

Phase 6 lineage fields: `dataset_snapshot_ids` (exact Phase 3 DS- snapshots),
`label_id`/`label_version` (Phase 5 contracts), `temporal_config` (Phase 4
temporal identity), `cost_model_id`, feature refs, code/config hashes, trial
metadata, research epoch, and researcher identity. Operational fields
(status, code/config hashes, timestamps, trial counters) are excluded from
`content_hash`; everything else is the scientific identity, frozen at
registration and immutable thereafter. Changing any identity field requires a
NEW experiment (child), never an edit.

The Phase 1 in-memory `ExperimentRegistry` below remains the governance
helper for hypothesis budgets. The operational, persistent Phase 6 registry
(with database constraints, lineage validation, reproduction specs and the
decision log) lives in `orbit.experiments` (ExperimentRegistry + service).
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from orbit.schemas.common import CostModel, ExperimentStatus
from orbit.schemas.hypothesis import HypothesisRegistry

_OPERATIONAL_FIELDS = frozenset(
    {
        "status",
        "code_hash",
        "config_hash",
        "created_at",
        "registered_at",
        "trial_number",
        "number_of_prior_trials",
    }
)


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


class FeatureRef(BaseModel):
    """One feature family pinned to an exact immutable version.

    Resolution target for Phase 10 feature ablation: every feature in an
    experiment must ultimately resolve to (feature_id, feature_version,
    transformation/configuration), not to a bare name.
    """

    feature_id: str = Field(pattern=r"^FEAT-\d{3,}$")
    feature_version: str = Field(pattern=r"^v\d+(\.\d+)*$")
    transformation: str | None = Field(
        default=None,
        description="transformation/configuration identity (config hash or ref)",
    )


class FeaturePin(BaseModel):
    """A pinned feature set version. Features are immutable once referenced."""

    feature_names: list[str] = Field(min_length=1)
    feature_version: str = Field(pattern=r"^v\d+(\.\d+)*$")
    feature_set_id: str | None = Field(
        default=None, pattern=r"^FS-\d{3,}$",
        description="id of the registered feature set, when one exists",
    )
    feature_refs: list[FeatureRef] = Field(
        default_factory=list,
        description="per-feature immutable version refs (Phase 6 / 10)",
    )


class TemporalConfigRef(BaseModel):
    """Reference to the exact Phase 4 temporal configuration used.

    The experiment pins (engine_version, config_digest): a future rerun must
    resolve the same temporal policy. The engine itself is not duplicated -
    this is a reference to it. `config_digest` is the sha256 of the loaded
    TemporalContract's canonical JSON.
    """

    engine_version: str = Field(pattern=r"^v\d+(\.\d+)*$")
    config_digest: str = Field(
        min_length=32,
        description="sha256 of the loaded temporal contract (configs/temporal.json)",
    )
    as_of_semantics: str = "strict_publication_less_than_asof"
    timezone: str = "America/New_York"


_KNOWN_MODEL_FAMILIES = frozenset(
    {
        "heuristic", "linear", "ridge", "logistic", "random_forest",
        "xgboost", "lightgbm", "null_shuffle",
    }
)


class ModelSpec(BaseModel):
    """Pinned model specification. No anonymous weights.

    Two models with different versions, hyperparameters or preprocessing are
    different experiments: all of it is part of the scientific identity.
    """

    family: str = Field(
        description="e.g. heuristic, linear, ridge, logistic, random_forest, xgboost, lightgbm, null_shuffle"
    )
    model_version: str | None = Field(default=None, pattern=r"^v\d+(\.\d+)*$")
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    preprocessing: dict[str, Any] = Field(default_factory=dict)
    training_config: dict[str, Any] = Field(default_factory=dict)
    target_transform: str | None = None
    uncertainty: bool = Field(default=False)

    @field_validator("family")
    @classmethod
    def _family_known(cls, v: str) -> str:
        if v not in _KNOWN_MODEL_FAMILIES:
            raise ValueError(f"unknown model family: {v}")
        return v


class ExperimentSpec(BaseModel):
    """A single registered experiment. Scientific identity frozen after
    registration; only the operational fields (status, code/config hash,
    timestamps, trial counters) may differ between the registration snapshot
    and later reads of the same experiment."""

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
    dataset_snapshot_ids: list[str] = Field(
        default_factory=list,
        description="Exact Phase 3 dataset snapshot ids (DS-xxxxxx) consumed; "
        "required by the Phase 6 registry. Never 'latest data'.",
    )
    features: FeaturePin
    model: ModelSpec
    windows: WindowSpec
    cost_model: CostModel = Field(default_factory=CostModel)
    cost_model_id: str | None = Field(
        default=None, pattern=r"^CM-\d{3,}$",
        description="Registered cost-model identity; two experiments with "
        "different cost assumptions must be distinguishable.",
    )

    label_id: str | None = Field(default=None, pattern=r"^LAB-\d{3}$")
    label_version: str | None = Field(
        default=None, pattern=r"^v\d+(\.\d+)*$",
        description="Phase 5 label contract version, pinned - never 'latest label'.",
    )
    temporal_config: TemporalConfigRef | None = Field(
        default=None,
        description="Phase 4 temporal identity: engine version + config digest.",
    )

    seed: int = Field(default=42)
    randomness_policy: Literal["seeded", "nondeterministic"] = Field(
        default="seeded",
        description="'nondeterministic' records explicitly that reproducibility "
        "is NOT claimed.",
    )

    status: ExperimentStatus = ExperimentStatus.DRAFT
    code_hash: str | None = Field(
        default=None,
        description="Hash of the executing code; set at registration or run start.",
    )
    config_hash: str | None = Field(
        default=None,
        description="Hash of the executing configuration; set at registration or run start.",
    )

    # Trial / search-depth facts. The counters (trial_number,
    # number_of_prior_trials) are computed by the registry at registration and
    # cannot be declared by the researcher; the rest are grouping/stage facts.
    hypothesis_family: str | None = Field(
        default=None,
        description="Optional family label grouping experiments (possibly across "
        "hypotheses) for search-depth accounting; defaults to hypothesis_id.",
    )
    research_epoch: str | None = Field(default=None, pattern=r"^EPOCH-\d{3,}$")
    selection_stage: str | None = Field(
        default=None,
        description="e.g. exploration, validation, finalist - a declared stage, "
        "never a promotion claim.",
    )
    parameter_count: int | None = Field(default=None, ge=0)
    feature_count: int | None = Field(default=None, ge=0)
    trial_number: int | None = Field(default=None, ge=1)
    number_of_prior_trials: int | None = Field(default=None, ge=0)

    researcher: str = "orbit-research"
    evaluation_protocol: str | None = Field(
        default=None,
        description="e.g. walkforward_v1, fixed_split_v1 - the evaluation protocol identity.",
    )
    notes: str | None = None

    created_at: datetime | None = Field(
        default=None, description="when the spec was created (set by the registry)"
    )
    registered_at: datetime | None = Field(
        default=None, description="when the experiment entered the registry"
    )

    # ------------------------------------------------------------ validators

    @model_validator(mode="after")
    def _no_self_parenting(self) -> "ExperimentSpec":
        if self.parent_id == self.experiment_id:
            raise ValueError("an experiment cannot be its own parent")
        return self

    @model_validator(mode="after")
    def _label_version_requires_label(self) -> "ExperimentSpec":
        if self.label_version is not None and self.label_id is None:
            raise ValueError("label_version requires label_id")
        return self

    @model_validator(mode="after")
    def _label_id_requires_version(self) -> "ExperimentSpec":
        if self.label_id is not None and self.label_version is None:
            raise ValueError(
                "label_id requires label_version: an experiment must pin an "
                "exact label version, never 'latest'"
            )
        return self

    @model_validator(mode="after")
    def _seeded_requires_seed(self) -> "ExperimentSpec":
        if self.randomness_policy == "seeded" and self.seed is None:
            raise ValueError("seeded experiments require a seed")
        return self

    @model_validator(mode="after")
    def _dataset_snapshot_ids_pattern(self) -> "ExperimentSpec":
        for ds in self.dataset_snapshot_ids:
            if not re.match(r"^DS-\d{6}$", ds):
                raise ValueError(f"invalid dataset snapshot id: {ds!r} (expected DS-xxxxxx)")
        return self

    # ------------------------------------------------------------ identity

    def content_hash(self) -> str:
        """Deterministic hash of the scientific identity.

        Excludes the operational fields (status, code/config hashes,
        timestamps, trial counters): the identity is what the researcher
        chose, not what happened during registration/execution.
        """
        payload = self.model_dump(exclude=_OPERATIONAL_FIELDS)
        raw = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class ExperimentRegistry(BaseModel):
    """Phase 1 governance registry of experiments with genealogy and trial
    budgets (in-memory). The operational Phase 6 registry - persistent,
    constraint-enforced, with lineage and reproduction specs - lives in
    `orbit.experiments`; this class remains the lightweight budget-check
    helper used by the Phase 1 tests and research-planning code."""

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