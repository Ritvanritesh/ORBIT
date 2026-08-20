"""Phase 6 experiment registry integration for the Phase 9 ML benchmark.

Every Phase 9 experiment follows the Phase 6 contract: the experiment is
registered BEFORE any training or evaluation happens, its identity pins the
full lineage (dataset snapshot, feature refs + version, label id/version,
temporal config digest, cost model id, model family + hyperparameters,
windows, seed, protocol), and code/config hashes are pinned at run start.
A different grid point, seed, or data pin is a different experiment.

Lifecycle for one ML run:
  register_ml_experiment(...)   -> REGISTERED (identity locked)
  run_registered_experiment(...)-> mark_running(code_hash, config_hash)
                                   -> train -> predict -> backtest
                                   -> attach artifacts -> complete
                                   -> record_result (single, immutable)
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl

from orbit.experiments import ExperimentService
from orbit.ml.features import FEATURE_DEFINITIONS, FEATURE_SET_ID, FEATURE_SET_VERSION
from orbit.ml.grids import SEED, validate_model_parameters
from orbit.ml.labels import LABEL_ID, LABEL_VERSION
from orbit.ml.splits import PHASE9_WINDOWS, window_identity
from orbit.schemas.experiment import (
    CostModel,
    ExperimentSpec,
    FeatureRef,
    TemporalConfigRef,
    WindowSpec,
)

ML_PACKAGE_DIR = Path(__file__).resolve().parent

_EXPERIMENT_ID_BASE = 90001
_CONTROL_ID_BASE = 90021


def control_experiment_id_for(family: str, params: dict[str, Any]) -> str:
    """Deterministic Phase 9 control experiment id (EXP-9xxxx range).

    Canonical order: buy_and_hold, equal_weight, then CONTROL_GRIDS in
    family order, then random_null and null_flat (parameterless).
    """
    from orbit.ml.baselines import CONTROL_GRIDS

    ordered: list[tuple[str, dict[str, Any]]] = [("buy_and_hold", {}), ("equal_weight", {})]
    for fam, grid in CONTROL_GRIDS.items():
        for point in grid:
            ordered.append((fam, dict(point)))
    ordered.append(("random_null", {}))
    ordered.append(("null_flat", {}))
    index = ordered.index((family, dict(params)))
    return f"EXP-{_CONTROL_ID_BASE + index}"


def experiment_id_for(family: str, params: dict[str, Any]) -> str:
    """Deterministic Phase 9 experiment id (EXP-9xxxx range) for a grid point.

    The canonical order is the model-family order of MODEL_FAMILIES and each
    family's pre-registered grid order, so the id is stable across runs.
    """
    validate_model_parameters(family, params)
    ordered: list[tuple[str, dict[str, Any]]] = []
    from orbit.ml.grids import MODEL_FAMILIES, PHASE9_GRIDS

    for fam in MODEL_FAMILIES:
        for point in PHASE9_GRIDS[fam]:
            ordered.append((fam, dict(point)))
    index = ordered.index((family, dict(params)))
    return f"EXP-{_EXPERIMENT_ID_BASE + index}"


def ml_code_hash() -> str:
    """sha256 over the Phase 9 ml package sources (deterministic identity)."""
    hasher = hashlib.sha256()
    for path in sorted(ML_PACKAGE_DIR.glob("*.py")):
        hasher.update(path.name.encode("utf-8"))
        hasher.update(path.read_bytes())
    return hasher.hexdigest()


def phase9_config_hash(family: str, params: dict[str, Any], seed: int) -> str:
    """Deterministic config identity of one Phase 9 run."""
    payload = {
        "model_family": family,
        "hyperparameters": dict(params),
        "seed": seed,
        "feature_set_id": FEATURE_SET_ID,
        "feature_set_version": FEATURE_SET_VERSION,
        "label_id": LABEL_ID,
        "label_version": LABEL_VERSION,
        "windows": window_identity(),
        "cost_model_id": "CM-001",
        "dataset_snapshot_ids": ["DS-000004"],
        "protocol": "phase9_v1",
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


def register_ml_experiment(
    *,
    experiment_id: str,
    hypothesis_id: str,
    family: str,
    params: dict[str, Any],
    seed: int = SEED,
    researcher: str = "orbit-research",
    notes: str | None = None,
) -> tuple[ExperimentService, Any]:
    """Register one Phase 9 ML experiment BEFORE any training/evaluation.

    Validates the grid point at registration time, so an unregistered
    hyperparameter set can never run. Returns (service, registered spec).
    """
    params = validate_model_parameters(family, params)
    label_registry = _build_label_registry()
    hypotheses = _build_hypothesis_registry()

    tmp_dir = tempfile.mkdtemp(prefix="phase9_exp_")
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
        title=f"Phase 9 ML benchmark: {family} {params}",
        parent_id=None,
        datasets=["market_daily_v1"],
        dataset_snapshot_ids=["DS-000004"],
        features={
            "feature_names": [f["name"] for f in FEATURE_DEFINITIONS],
            "feature_version": FEATURE_SET_VERSION,
            "feature_refs": [
                {"feature_id": f["feature_id"], "feature_version": "v1"}
                for f in FEATURE_DEFINITIONS
            ],
        },
        model={
            "family": family,
            "model_version": "v1",
            "hyperparameters": params,
            "preprocessing": {"standardized": family in ("ridge", "lasso", "logistic")},
            "training_config": {"protocol": "fixed_chronological_v1", "seed": seed},
            "target_transform": "identity" if family != "logistic" else "positive_return_binary",
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
        hypothesis_family=f"phase9_{family}",
        researcher=researcher,
        evaluation_protocol="fixed_split_v1",
        notes=notes,
    )
    registered = service.register(spec)
    return service, registered


def _build_label_registry():
    from orbit.labels.registry import LabelVersionRegistry
    from orbit.labels.seeds import build_seed_label_registry

    registry = build_seed_label_registry()
    from orbit.ml.labels import register_phase9_label

    register_phase9_label(registry)
    return registry


def register_control_experiment(
    *,
    experiment_id: str,
    hypothesis_id: str,
    family: str,
    params: dict[str, Any],
    seed: int = SEED,
    researcher: str = "orbit-research",
    notes: str | None = None,
) -> tuple[ExperimentService, Any]:
    """Register one Phase 8 control run on the real dataset (Phase 9 use).

    `family` must be a Phase 8 control family; `params` must be a
    pre-registered control grid point (or empty for the parameterless
    controls). Same register-before-run contract as the ML experiments.
    """
    from orbit.ml.baselines import CONTROL_GRIDS

    if family in CONTROL_GRIDS:
        if params not in CONTROL_GRIDS[family]:
            raise ValueError(
                f"control {family!r} params {params} are not pre-registered; "
                f"grid: {CONTROL_GRIDS[family]}"
            )
    elif family not in ("buy_and_hold", "equal_weight", "random_null", "null_flat"):
        raise ValueError(f"unknown control family {family!r}")
    elif params:
        raise ValueError(f"control {family!r} takes no parameters")

    label_registry = _build_label_registry()
    hypotheses = _build_hypothesis_registry()

    tmp_dir = tempfile.mkdtemp(prefix="phase9_ctl_")
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
        title=f"Phase 9 control: {family} {params or ''}",
        parent_id=None,
        datasets=["market_daily_v1"],
        dataset_snapshot_ids=["DS-000004"],
        features={
            "feature_names": ["phase8_control_rules"],
            "feature_version": "v1",
            "feature_refs": [
                {"feature_id": "FEAT-901", "feature_version": "v1", "transformation": "phase8_documented_rules"}
            ],
        },
        model={
            "family": "heuristic",
            "model_version": "v1",
            "hyperparameters": params,
            "preprocessing": {},
            "training_config": {"protocol": "fixed_chronological_v1", "seed": seed},
            "target_transform": None,
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
        hypothesis_family=f"phase9_control_{family}",
        researcher=researcher,
        evaluation_protocol="fixed_split_v1",
        notes=notes,
    )
    registered = service.register(spec)
    return service, registered


def _build_hypothesis_registry():
    from hypotheses.seeds import register_seeds

    return register_seeds()


def run_registered_experiment(
    service: ExperimentService,
    experiment_id: str,
    *,
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
    config_hash = phase9_config_hash(family, params, seed)
    service.mark_running(experiment_id, code_hash=code_hash, config_hash=config_hash)

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
    "ml_code_hash",
    "phase9_config_hash",
    "experiment_id_for",
    "control_experiment_id_for",
    "register_ml_experiment",
    "register_control_experiment",
    "run_registered_experiment",
]