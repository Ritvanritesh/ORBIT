"""Shared Phase 6 test fixtures.

A hermetic research-control stack: registered seed hypotheses (Phase 1),
seed label contracts (Phase 5), the loaded temporal contract (Phase 4), and
a fake Phase 3 dataset registry. `make_service` wires them together exactly
as production would.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from hypotheses.seeds import build_seed_registry, register_seeds
from orbit.experiments import ExperimentService, temporal_config_digest
from orbit.labels.seeds import build_seed_label_registry
from orbit.schemas.experiment import ExperimentSpec, TemporalConfigRef
from orbit.temporal.contracts import load_temporal_contract

WINDOWS = {
    "train_start": "2015-01-01",
    "train_end": "2020-01-01",
    "val_start": "2020-01-02",
    "val_end": "2021-01-01",
    "test_start": "2021-01-02",
    "test_end": "2022-01-01",
}

DS_000001 = {
    "snapshot_id": "DS-000001",
    "domain": "market",
    "provider": "yahoo_chart_api",
    "source_uri": "https://query1.finance.yahoo.com/v8/finance/chart/SPY",
    "request_fingerprint": "f" * 64,
    "checksum": "c" * 64,
    "file_count": 1,
    "row_count": 1000,
    "downloaded_at": "2026-01-01T00:00:00Z",
    "schema_version": "v1.0.0",
    "license_ref": None,
    "validation_status": "ok",
    "manifest_path": "data/manifests/DS-000001.json",
}

DS_000002 = {
    **DS_000001,
    "snapshot_id": "DS-000002",
    "checksum": "d" * 64,
    "source_uri": "https://query1.finance.yahoo.com/v8/finance/chart/AAPL",
}


class FakeDatasetRegistry:
    """Phase 3 IngestionRegistry stand-in (only snapshot() is used)."""

    def __init__(self, snapshots: dict[str, dict[str, Any]] | None = None):
        self._snapshots = dict(snapshots or {})
        for ds in (DS_000001, DS_000002):
            self._snapshots.setdefault(ds["snapshot_id"], dict(ds))

    def snapshot(self, snapshot_id: str) -> dict[str, Any] | None:
        rec = self._snapshots.get(snapshot_id)
        return dict(rec) if rec else None


@pytest.fixture
def hypotheses():
    return register_seeds()


@pytest.fixture
def labels():
    return build_seed_label_registry()


@pytest.fixture
def temporal():
    return load_temporal_contract()


@pytest.fixture
def datasets():
    return FakeDatasetRegistry()


@pytest.fixture
def temporal_digest(temporal) -> str:
    return temporal_config_digest(temporal)


def make_service(
    tmp_path: Path,
    *,
    hypotheses=None,
    labels=None,
    temporal=None,
    datasets=None,
    **kwargs,
) -> ExperimentService:
    return ExperimentService(
        db_path=tmp_path / "experiments.duckdb",
        hypothesis_registry=hypotheses,
        label_registry=labels,
        temporal_contract=temporal,
        dataset_registry=datasets,
        **kwargs,
    )


@pytest.fixture
def service(tmp_path, hypotheses, labels, temporal, datasets) -> ExperimentService:
    return make_service(tmp_path, hypotheses=hypotheses, labels=labels, temporal=temporal, datasets=datasets)


def make_spec(
    experiment_id: str = "EXP-00001",
    hypothesis_id: str = "H-001",
    *,
    temporal_digest: str,
    **overrides,
) -> ExperimentSpec:
    """A fully-specified Phase 6 experiment (all lineage pins present)."""
    base: dict[str, Any] = dict(
        title="momentum baseline",
        datasets=["market_daily_v1"],
        dataset_snapshot_ids=["DS-000001"],
        features={
            "feature_names": ["ret_12m_1m"],
            "feature_version": "v1",
            "feature_refs": [
                {"feature_id": "FEAT-001", "feature_version": "v1", "transformation": "xform-v1"}
            ],
        },
        model={"family": "linear", "hyperparameters": {"alpha": 0.1}},
        windows=dict(WINDOWS),
        label_id="LAB-001",
        label_version="v1",
        cost_model_id="CM-001",
        temporal_config=TemporalConfigRef(
            engine_version="v1.0.0", config_digest=temporal_digest
        ),
        seed=42,
        researcher="orbit-research",
        evaluation_protocol="walkforward_v1",
    )
    base.update(overrides)
    return ExperimentSpec(
        experiment_id=experiment_id, hypothesis_id=hypothesis_id, **base
    )