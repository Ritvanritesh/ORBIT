"""Reproduction specifications (Phase 6, section 22).

`build_reproduction_spec` resolves EVERYTHING a future researcher needs to
reconstruct an experiment:

    experiment
      -> dataset snapshots (Phase 3, exact DS- ids + checksums)
      -> temporal configuration (Phase 4: engine version + config digest)
      -> label contract (Phase 5: pinned version + content hash)
      -> feature refs, model spec, evaluation windows, cost model
      -> code/config hashes, seed
      -> trial/search depth, genealogy
      -> result and decision records

The `reproduction_digest` is a stable hash over the immutable core: it must
not change after completion. Two rebuilds of the same experiment must yield
the same digest; any drift is a lineage violation.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict

from orbit.schemas.common import ExperimentStatus
from orbit.schemas.experiment import ExperimentSpec


def _stable_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, default=str, separators=(",", ":"))


class ReproductionSpec(BaseModel):
    """Everything needed to reproduce one experiment."""

    model_config = ConfigDict(frozen=True)

    experiment_id: str
    status: ExperimentStatus
    registered_at: datetime | None
    trial_number: int | None
    number_of_prior_trials: int | None
    research_epoch: str | None
    parent_id: str | None

    spec: ExperimentSpec
    content_hash: str
    code_hash: str | None
    config_hash: str | None

    hypothesis: dict[str, Any] | None = None
    datasets: list[dict[str, Any]] = []
    temporal: dict[str, Any] | None = None
    label: dict[str, Any] | None = None
    features: list[dict[str, Any]] = []

    result: dict[str, Any] | None = None
    decision: dict[str, Any] | None = None
    artifacts: list[dict[str, Any]] = []

    reproduction_digest: str

    def _immutable_core(self) -> dict[str, Any]:
        """The scientific core that defines reproduction identity."""
        return {
            "experiment_id": self.experiment_id,
            "spec_identity": self.content_hash,
            "code_hash": self.code_hash,
            "config_hash": self.config_hash,
            "seed": self.spec.seed,
            "randomness_policy": self.spec.randomness_policy,
            "dataset_snapshot_ids": sorted(self.spec.dataset_snapshot_ids),
            "datasets": [
                {
                    "snapshot_id": d["snapshot_id"],
                    "checksum": d.get("checksum"),
                    "schema_version": d.get("schema_version"),
                    "provider": d.get("provider"),
                    "validation_status": d.get("validation_status"),
                }
                for d in sorted(self.datasets, key=lambda d: d["snapshot_id"])
            ],
            "label_id": self.spec.label_id,
            "label_version": self.spec.label_version,
            "label_digest": (self.label or {}).get("content_hash"),
            "temporal": (
                self.spec.temporal_config.model_dump(mode="json")
                if self.spec.temporal_config
                else None
            ),
            "cost_model_id": self.spec.cost_model_id,
            "cost_model": self.spec.cost_model.model_dump(mode="json"),
            "features": sorted(
                (f.get("feature_id"), f.get("feature_version"), f.get("transformation"))
                for f in self.features
            ),
            "model": self.spec.model.model_dump(mode="json"),
            "windows": self.spec.windows.model_dump(mode="json"),
            "evaluation_protocol": self.spec.evaluation_protocol,
            "randomness": self.spec.randomness_policy,
        }

    def reproduction_digest_computed(self) -> str:
        raw = _stable_json(self._immutable_core())
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def verify_digest(self) -> bool:
        """The stored digest must match a recomputation over the core."""
        return self.reproduction_digest == self.reproduction_digest_computed()


def build_reproduction_spec(
    *,
    spec: ExperimentSpec,
    status: ExperimentStatus,
    code_hash: str | None,
    config_hash: str | None,
    registered_at: datetime | None,
    hypothesis: dict[str, Any] | None,
    datasets: list[dict[str, Any]],
    temporal: dict[str, Any] | None,
    label: dict[str, Any] | None,
    features: list[dict[str, Any]],
    result: dict[str, Any] | None,
    decision: dict[str, Any] | None,
    artifacts: list[dict[str, Any]],
) -> ReproductionSpec:
    spec_obj = ReproductionSpec(
        experiment_id=spec.experiment_id,
        status=status,
        registered_at=registered_at,
        trial_number=spec.trial_number,
        number_of_prior_trials=spec.number_of_prior_trials,
        research_epoch=spec.research_epoch,
        parent_id=spec.parent_id,
        spec=spec,
        content_hash=spec.content_hash(),
        code_hash=code_hash,
        config_hash=config_hash,
        hypothesis=hypothesis,
        datasets=datasets,
        temporal=temporal,
        label=label,
        features=features,
        result=result,
        decision=decision,
        artifacts=artifacts,
        reproduction_digest="",
    )
    digest = spec_obj.reproduction_digest_computed()
    return spec_obj.model_copy(update={"reproduction_digest": digest})


__all__ = ["ReproductionSpec", "build_reproduction_spec"]