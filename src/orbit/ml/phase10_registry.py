"""Phase 10 experiment registry integration (register-before-run).

Mirrors the Phase 9 registry contract: every Phase 10 experiment is
registered BEFORE any training or evaluation happens, its identity pins the
full lineage (dataset snapshot DS-000004, feature set id/version + exact
feature refs + definitions digest, label LAB-004 v1, temporal config digest,
cost model CM-001, model family + hyperparameters, windows, seed, protocol),
and code/config hashes are pinned at run start. A different feature set,
grid point, seed, or data pin is a different experiment.

The config hash deliberately includes the feature-set id, its version, the
exact feature refs and the definitions digest, so changing any feature
definition or set membership after registration produces a config-hash
mismatch - making feature mutation after registration structurally
detectable (adversarial requirement A8/A9/A15).
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

from orbit.experiments import ExperimentService
from orbit.ml.features import (
    FEATURE_SET_VERSION,
    _feature_definitions_digest,
    phase10_set_identity,
)
from orbit.ml.grids import SEED, validate_model_parameters
from orbit.ml.labels import LABEL_ID, LABEL_VERSION
from orbit.ml.phase10_plan import phase10_model_point_for
from orbit.ml.registry import ml_code_hash
from orbit.ml.splits import PHASE9_WINDOWS, window_identity
from orbit.schemas.experiment import (
    CostModel,
    ExperimentSpec,
    FeatureRef,
    TemporalConfigRef,
)


def phase10_config_hash(
    feature_set_id: str,
    family: str,
    params: dict[str, Any],
    seed: int = SEED,
) -> str:
    """Deterministic config identity of one Phase 10 run.

    Covers the full lineage including the feature definitions digest, so a
    feature definition or membership change invalidates the hash.
    """
    identity = phase10_set_identity(feature_set_id)
    payload = {
        "protocol": "phase10_v1",
        "model_family": family,
        "hyperparameters": dict(params),
        "seed": seed,
        "feature_set_id": feature_set_id,
        "feature_set_version": identity["feature_set_version"],
        "feature_refs": identity["feature_refs"],
        "feature_definitions_digest": _feature_definitions_digest(
            identity["feature_refs"]
        ),
        "label_id": LABEL_ID,
        "label_version": LABEL_VERSION,
        "windows": window_identity(),
        "cost_model_id": "CM-001",
        "dataset_snapshot_ids": ["DS-000004"],
    }
    raw = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _temporal_config_ref() -> TemporalConfigRef:
    from orbit.temporal.contracts import load_temporal_contract

    temporal = load_temporal_contract()
    raw = json.dumps(temporal.model_dump(mode="json"), sort_keys=True, default=str)
    return TemporalConfigRef(
        engine_version=temporal.engine_version,
        config_digest=hashlib.sha256(raw.encode("utf-8")).hexdigest(),
    )


def _windows_spec() -> dict[str, Any]:
    return {
        "train_start": PHASE9_WINDOWS["train_start"],
        "train_end": PHASE9_WINDOWS["train_end"],
        "val_start": PHASE9_WINDOWS["val_start"],
        "val_end": PHASE9_WINDOWS["val_end"],
        "test_start": PHASE9_WINDOWS["test_start"],
        "test_end": PHASE9_WINDOWS["test_end"],
        "embargo_days": PHASE9_WINDOWS["embargo_days"],
        "purge_days": PHASE9_WINDOWS["purge_days"],
    }


def _build_label_registry():
    from orbit.labels.registry import LabelVersionRegistry
    from orbit.labels.seeds import build_seed_label_registry

    registry = build_seed_label_registry()
    from orbit.ml.labels import register_phase9_label

    register_phase9_label(registry)
    return registry


def _build_hypothesis_registry():
    from hypotheses.seeds import register_seeds

    return register_seeds()


def register_phase10_experiment(
    *,
    experiment_id: str,
    hypothesis_id: str,
    feature_set_id: str,
    feature_set_version: str,
    family: str,
    params: dict[str, Any],
    seed: int = SEED,
    researcher: str = "orbit-research",
    notes: str | None = None,
    plan_digest: str | None = None,
) -> tuple[ExperimentService, Any]:
    """Register one Phase 10 experiment BEFORE any training/evaluation.

    `family`/`params` must be one of the pre-registered Phase 10 model points
    (itself a subset of the Phase 9 grids), so an unregistered hyperparameter
    set can never run. `feature_set_id` must be a registered Phase 10 set
    (or the frozen FS-001 base). Returns (service, registered spec).
    """
    point = phase10_model_point_for(family, params)
    params = validate_model_parameters(family, params)
    identity = phase10_set_identity(feature_set_id)
    if feature_set_version != identity["feature_set_version"]:
        raise ValueError(
            f"feature set {feature_set_id} is version "
            f"{identity['feature_set_version']}, not {feature_set_version!r}"
        )

    label_registry = _build_label_registry()
    hypotheses = _build_hypothesis_registry()

    tmp_dir = tempfile.mkdtemp(prefix="phase10_exp_")
    db_path = Path(tmp_dir) / "experiments.duckdb"
    service = ExperimentService(
        registry=None,
        db_path=db_path,
        hypothesis_registry=hypotheses,
        label_registry=label_registry,
    )

    spec = ExperimentSpec(
        experiment_id=experiment_id,
        hypothesis_id=hypothesis_id,
        title=(
            f"Phase 10 feature ablation: {feature_set_id} "
            f"{feature_set_version} {family} {params}"
        ),
        parent_id=None,
        datasets=["market_daily_v1"],
        dataset_snapshot_ids=["DS-000004"],
        features={
            "feature_names": [_feature_name_for_id(fid) for fid in identity["feature_refs"]],
            "feature_version": feature_set_version,
            "feature_set_id": feature_set_id,
            "feature_refs": [
                FeatureRef(
                    feature_id=fid,
                    feature_version="v1",
                    transformation=identity["transformation"],
                )
                for fid in identity["feature_refs"]
            ],
        },
        model={
            "family": family,
            "model_version": "v1",
            "hyperparameters": params,
            "preprocessing": {"standardized": family in ("ridge", "lasso")},
            "training_config": {"protocol": "fixed_chronological_v1", "seed": seed},
            "target_transform": "identity",
            "uncertainty": False,
        },
        windows=_windows_spec(),
        cost_model=CostModel(spread_bps=2.0, fees_bps=1.0, slippage_bps=2.0),
        cost_model_id="CM-001",
        label_id=LABEL_ID,
        label_version=LABEL_VERSION,
        temporal_config=_temporal_config_ref(),
        seed=seed,
        randomness_policy="seeded",
        hypothesis_family="phase10_feature_ablation",
        researcher=researcher,
        evaluation_protocol="fixed_split_v1",
        notes=(
            f"Phase 10 ablation set {feature_set_id} {feature_set_version}; "
            f"model point phase9_parent={point['phase9_parent']}; "
            f"plan_digest={plan_digest}"
            + (f"; {notes}" if notes else "")
        ),
    )
    registered = service.register(spec)
    return service, registered


def _feature_name_for_id(feature_id: str) -> str:
    from orbit.ml.features import (
        ALL_PHASE10_DEFINITIONS,
        FEATURE_DEFINITIONS,
    )

    for f in FEATURE_DEFINITIONS + ALL_PHASE10_DEFINITIONS:
        if f["feature_id"] == feature_id:
            return f["name"]
    raise ValueError(f"unknown feature id {feature_id!r}")


def run_registered_phase10_experiment(
    service: ExperimentService,
    experiment_id: str,
    *,
    feature_set_id: str,
    family: str,
    params: dict[str, Any],
    seed: int,
    artifacts_dir: str | Path,
    result_summary: str,
    result_metrics: dict[str, Any],
    artifact_files: dict[str, Path],
    result_kind: str = "supported",
) -> str:
    """Complete lifecycle: mark_running -> attach artifacts -> complete ->
    record the single immutable result. Returns the result_id.

    The caller must have already performed training/evaluation (registration
    happened before that). Artifact files are checksummed on attach.
    """
    code_hash = ml_code_hash()
    config_hash = phase10_config_hash(feature_set_id, family, params, seed)
    service.mark_running(
        experiment_id, code_hash=code_hash, config_hash=config_hash
    )

    out_dir = Path(artifacts_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stored: dict[str, Path] = {}
    for kind, path in artifact_files.items():
        dest = out_dir / f"{experiment_id}_{Path(path).name}"
        dest.write_bytes(Path(path).read_bytes())
        stored[kind] = dest

    service.complete(experiment_id, note=result_summary)
    for kind, path in stored.items():
        service.attach_artifact(
            experiment_id,
            kind=kind,
            path=str(path),
            checksum=_sha256_file(path),
        )
    return service.record_result(
        experiment_id,
        kind=result_kind,
        summary=result_summary,
        metrics=result_metrics,
        recorded_by="orbit-research",
    )


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    hasher.update(Path(path).read_bytes())
    return hasher.hexdigest()


__all__ = [
    "phase10_config_hash",
    "register_phase10_experiment",
    "run_registered_phase10_experiment",
]