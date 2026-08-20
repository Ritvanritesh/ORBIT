"""Experiment registry integration for Phase 8 baseline strategies.

Every baseline run must use the Phase 6 experiment registry (Section 11 of
the roadmap).  This module handles:

  - Registering the experiment BEFORE execution.
  - Recording the hypothesis/strategy definition, parameters, dataset snapshot,
    universe, evaluation window, cost model, code/config identity, and random seed.
  - Recording result artifacts and lineage.
  - Ensuring reproducibility through immutable experiment identities.

The pattern follows the ExperimentService API (Section 12 of the roadmap):
all strategies pass through the same canonical Phase 7 backtester, and the
experiment is registered before the backtest run begins.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from orbit.experiments import ExperimentService
from orbit.experiments.registry import ExperimentRegistry
from orbit.schemas.experiment import ExperimentSpec, TemporalConfigRef, CostModel
from orbit.schemas.common import ExperimentStatus


# ---------------------------------------------------------------------------
# Default temporal config (Phase 4) - shared across all baselines
# ---------------------------------------------------------------------------

def _default_temporal_config() -> TemporalConfigRef:
    """Return the default Phase 4 temporal configuration reference.

    The experiment pins (engine_version, config_digest): a future rerun must
    resolve the same temporal policy. The config_digest is the sha256 of the
    loaded TemporalContract's canonical JSON.
    """
    from orbit.temporal.contracts import load_temporal_contract

    temporal = load_temporal_contract()
    # Compute the config digest from the loaded contract
    import hashlib
    import json

    payload = temporal.model_dump(mode="json")
    raw = json.dumps(payload, sort_keys=True, default=str)
    config_digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()

    return TemporalConfigRef(
        engine_version=temporal.engine_version,
        config_digest=config_digest,
    )


# ---------------------------------------------------------------------------
# Register a baseline experiment before execution
# ---------------------------------------------------------------------------

def register_baseline_experiment(
    *,
    experiment_id: str,
    hypothesis_id: str,
    strategy_name: str,
    strategy_params: dict[str, Any],
    universe: list[str],
    dataset_snapshot_ids: list[str],
    cost_model: CostModel,
    window_start: date | None,
    window_end: date | None,
    temporal_contract: Any | None = None,
    label_id: str = "LAB-001",
    label_version: str = "v1",
    researcher: str = "orbit-research",
    seed: int = 42,
) -> ExperimentService:
    """Register a Phase 6 experiment for a baseline strategy run.

    Validation performed by the service:
      - the spec is a canonical ExperimentSpec (no raw-dict bypass);
      - the experiment id is unique;
      - the hypothesis exists and has budget;
      - every dataset_snapshot_id resolves in the Phase 3 registry;
      - the label resolves in the Phase 5 registry;
      - the temporal configuration pins the loaded Phase 4 contract;
      - trial counters are computed by the registry.

    Returns the registered ExperimentSpec (now in REGISTERED status).
    """
    from orbit.labels.seeds import build_seed_label_registry

    # Build the experiment service with all required registries
    label_registry = build_seed_label_registry()
    hypotheses = _build_seed_hypothesis_registry()

    # Use a temporary DB path for the experiment ledger
    import tempfile

    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "experiments.duckdb"

        service = ExperimentService(
            registry=None,  # will use default or create one
            db_path=db_path,
            hypothesis_registry=hypotheses,
            label_registry=label_registry,
        )

        # Build the temporal config
        if temporal_contract is None:
            temporal_contract = _default_temporal_config()

        # Build the ExperimentSpec
        spec = ExperimentSpec(
            experiment_id=experiment_id,
            hypothesis_id=hypothesis_id,
            title=f"{strategy_name} baseline",
            parent_id=None,
            datasets=["market_daily_v1"],
            dataset_snapshot_ids=sorted(set(dataset_snapshot_ids)),
            features={
                "feature_names": ["ret_12m_1m"],
                "feature_version": "v1",
                "feature_refs": [
                    {"feature_id": "FEAT-001", "feature_version": "v1"}
                ],
            },
            model={
                "family": "linear",
                "model_version": None,
                "hyperparameters": {},
                "preprocessing": {},
                "training_config": {},
                "target_transform": None,
                "uncertainty": False,
            },
            windows={  # minimal window spec; actual values come from run context
                "train_start": window_start or datetime(2015, 1, 1).date(),
                "train_end": window_end or datetime(2020, 1, 1).date(),
                "val_start": (window_end or datetime(2020, 1, 1)).date()
                if window_end
                else datetime(2020, 1, 2).date(),
                "val_end": window_start or datetime(2021, 1, 1).date()
                if window_start
                else datetime(2021, 1, 1).date(),
                "test_start": window_start or datetime(2021, 1, 2).date()
                if window_start
                else datetime(2021, 1, 2).date(),
                "test_end": window_end or datetime(2022, 1, 1).date()
                if window_end
                else datetime(2022, 1, 1).date(),
                "embargo_days": 5,
                "purge_days": 0,
            },
            cost_model=cost_model,
            cost_model_id="CM-001",
            label_id=label_id,
            label_version=label_version,
            temporal_config=temporal_contract,
            seed=seed,
            randomness_policy="seeded",
            hypothesis_family=strategy_name,
            researcher=researcher,
            evaluation_protocol="fixed_split_v1",
        )

        # Register the experiment
        registered = service.register(spec)

        return service, registered


def _build_seed_hypothesis_registry():
    """Build a seed hypothesis registry for experiment validation."""
    from hypotheses.seeds import build_seed_registry, register_seeds

    hyp_registry = build_seed_registry()
    return hyp_registry


# ---------------------------------------------------------------------------
# Record result after backtest execution
# ---------------------------------------------------------------------------

def record_baseline_result(
    service: ExperimentService,
    experiment_id: str,
    *,
    summary: str,
    metrics: dict[str, Any],
    decision: str | None = None,
    decision_reason: str | None = None,
    decision_maker: str = "orbit-research",
) -> str:
    """Record the single immutable result of a baseline experiment.

    One result per experiment: a second recording is refused loudly.
    The result is FK-bound to the experiment, and the experiment status
    is moved to REJECTED or PROMOTED if a decision is recorded.

    Returns the result_id.
    """
    recorded_at = datetime.now()

    result_id = service.record_result(
        experiment_id=experiment_id,
        kind="supported",  # baseline results are "supported" controls
        summary=summary,
        metrics=metrics,
        recorded_by=decision_maker,
        recorded_at=recorded_at.isoformat(),
    )

    # If a decision was requested, record it too
    if decision is not None:
        service.record_decision(
            experiment_id=experiment_id,
            decision=decision,
            reason=decision_reason or f"{strategy_name} baseline evaluation",
            decision_maker=decision_maker,
            decided_at=recorded_at.isoformat(),
        )

    return result_id