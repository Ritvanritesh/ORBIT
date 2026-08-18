"""Phase 6 reproducibility tests: reproduction specs and the replay test.

The replay test (section 23) proves an experiment can be reconstructed:

    Register -> Run -> Store result -> Load experiment id
        -> Reconstruct configuration -> Run again -> Compare result

Because ORBIT's real model execution engine is a later phase, the "pipeline"
here is a deterministic toy function whose output depends ONLY on the
resolved lineage (identity hash, code/config hashes, dataset checksums, label
digest, temporal digest, seed, windows, cost model). Identical lineage must
produce identical output; the recorded result must match the rerun within
documented numerical tolerances.
"""

from __future__ import annotations

import hashlib
import json
import random
from datetime import datetime, timezone

import pytest

from conftest import make_service, make_spec

from orbit.experiments import Decision, ExperimentService, ReproductionSpec, ResultKind
from orbit.labels.seeds import build_seed_label_registry
from hypotheses.seeds import register_seeds


def _at(y, m=1, d=1):
    return datetime(y, m, d, tzinfo=timezone.utc)


def replay_fn(spec: ReproductionSpec) -> dict:
    """Deterministic toy pipeline: output depends only on the resolved
    lineage, so the replay property is decidable without a real model."""
    rng = random.Random(spec.spec.seed)
    core = {
        "content_hash": spec.content_hash,
        "code_hash": spec.code_hash,
        "config_hash": spec.config_hash,
        "datasets": sorted(
            (d["snapshot_id"], d.get("checksum"), d.get("schema_version"))
            for d in spec.datasets
        ),
        "label_digest": (spec.label or {}).get("content_hash"),
        "temporal_digest": spec.spec.temporal_config.config_digest,
        "windows": spec.spec.windows.model_dump(),
        "cost": spec.spec.cost_model.model_dump(),
        "model": spec.spec.model.model_dump(),
        "features": spec.spec.features.model_dump(),
    }
    digest = hashlib.sha256(
        json.dumps(core, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    return {
        "output_digest": digest,
        "oos_rank_ic": round(0.03 + rng.random() * 0.02, 6),
        "after_cost_excess": round(0.02 + rng.random() * 0.02, 6),
    }


def _register_run_complete(service: ExperimentService, spec):
    service.register(spec, registered_at=_at(2026, 1, 1))
    service.mark_running(spec.experiment_id, code_hash="c" * 64, config_hash="g" * 64)
    service.complete(spec.experiment_id)


def test_reproduction_spec_resolves_every_element(service, temporal_digest):
    _register_run_complete(service, make_spec(temporal_digest=temporal_digest))
    spec = service.reproduction_spec("EXP-00001")
    assert spec.experiment_id == "EXP-00001"
    assert spec.code_hash == "c" * 64
    assert spec.config_hash == "g" * 64
    assert spec.spec.seed == 42
    assert spec.datasets[0]["snapshot_id"] == "DS-000001"
    assert spec.datasets[0]["checksum"] == "c" * 64
    assert spec.datasets[0]["provider"] == "yahoo_chart_api"
    assert spec.datasets[0]["schema_version"] == "v1.0.0"
    assert spec.label["label_id"] == "LAB-001"
    assert spec.label["version"] == "v1"
    assert spec.label["content_hash"]
    assert spec.temporal["engine_version"] == "v1.0.0"
    assert spec.temporal["boundary"] == "strict_publication_less_than_asof"
    assert spec.hypothesis["hypothesis_id"] == "H-001"
    assert spec.features[0]["feature_id"] == "FEAT-001"
    assert spec.spec.model.family == "linear"
    assert str(spec.spec.windows.train_start) == "2015-01-01"
    assert spec.spec.cost_model.total_bps() == 5.0
    assert spec.spec.cost_model_id == "CM-001"
    assert spec.verify_digest()


def test_replay_register_run_store_load_rerun_compare(service, temporal_digest):
    spec_obj = make_spec(temporal_digest=temporal_digest)
    service.register(spec_obj, registered_at=_at(2026, 1, 1))
    service.mark_running("EXP-00001", code_hash="c" * 64, config_hash="g" * 64)
    service.complete("EXP-00001")

    # run #1
    first = replay_fn(service.reproduction_spec("EXP-00001"))
    service.record_result(
        "EXP-00001",
        kind=ResultKind.SUPPORTED,
        summary="replay pipeline output",
        metrics=first,
    )

    # reconstruction: load by experiment id, resolve the full configuration
    rebuilt = service.reproduction_spec("EXP-00001")
    second = replay_fn(rebuilt)

    # compare the research artifacts, not one headline number
    assert first["output_digest"] == second["output_digest"]
    assert first["oos_rank_ic"] == pytest.approx(second["oos_rank_ic"], abs=1e-6)
    assert first["after_cost_excess"] == pytest.approx(second["after_cost_excess"], abs=1e-6)
    stored = service.result("EXP-00001")
    assert stored["kind"] == "supported"
    assert json.loads(stored["metrics_json"])["output_digest"] == second["output_digest"]


def test_reproduction_digest_is_stable_across_rebuilds(service, temporal_digest):
    _register_run_complete(service, make_spec(temporal_digest=temporal_digest))
    a = service.reproduction_spec("EXP-00001")
    b = service.reproduction_spec("EXP-00001")
    assert a.reproduction_digest == b.reproduction_digest
    assert a.verify_digest() and b.verify_digest()


def test_reproduction_digest_stable_after_completion(service, temporal_digest):
    # pin the code/config identity at registration: the digest is frozen
    # before any execution and never changes afterwards
    service.register(
        make_spec(temporal_digest=temporal_digest, code_hash="c" * 64, config_hash="g" * 64),
        registered_at=_at(2026, 1, 1),
    )
    before = service.reproduction_spec("EXP-00001")
    service.mark_running("EXP-00001", code_hash="c" * 64, config_hash="g" * 64)
    service.complete("EXP-00001")
    after = service.reproduction_spec("EXP-00001")
    assert before.reproduction_digest == after.reproduction_digest
    assert after.status.value == "completed"


def test_missing_dataset_lineage_is_a_loud_violation(tmp_path, hypotheses, labels, temporal, temporal_digest):
    # register with a working dataset registry, then resolve with a broken one
    full = make_service(tmp_path, hypotheses=hypotheses, labels=labels, temporal=temporal, datasets=_DatasetRegistryWith({}))
    full.register(make_spec(temporal_digest=temporal_digest), registered_at=_at(2026, 1, 1))
    full._registry.close()
    broken = make_service(tmp_path, hypotheses=hypotheses, labels=labels, temporal=temporal, datasets=_DatasetRegistryWith({"DS-000001": None}))
    with pytest.raises(ValueError, match="lineage violation"):
        broken.reproduction_spec("EXP-00001")


class _DatasetRegistryWith:
    """Fake dataset registry returning None for the given snapshot."""

    def __init__(self, snapshots):
        from conftest import DS_000001

        self._snapshots = {"DS-000001": DS_000001, **snapshots}

    def snapshot(self, snapshot_id):
        rec = self._snapshots.get(snapshot_id)
        return dict(rec) if rec else None


def test_missing_label_lineage_is_a_loud_violation(tmp_path, hypotheses, temporal, datasets, temporal_digest):
    from orbit.labels.contract import AnchorMode, LabelContract, ReturnConvention
    from orbit.labels.registry import LabelVersionRegistry

    # register against a label registry that HAS LAB-001
    full_labels = build_seed_label_registry()
    full = make_service(tmp_path, hypotheses=hypotheses, labels=full_labels, temporal=temporal, datasets=datasets)
    full.register(make_spec(temporal_digest=temporal_digest), registered_at=_at(2026, 1, 1))
    full._registry.close()

    # resolve against one that no longer resolves LAB-001
    reduced = LabelVersionRegistry()
    reduced.register(
        LabelContract(
            label_id="LAB-007", version="v1", target_type="forward_return",
            horizon=5, anchor_mode=AnchorMode.DECISION_INSTANT,
            return_convention=ReturnConvention.SIMPLE_PRICE_RETURN,
            formula="unrelated label",
        )
    )
    broken = make_service(tmp_path, hypotheses=hypotheses, labels=reduced, temporal=temporal, datasets=datasets)
    with pytest.raises(ValueError, match="lineage violation"):
        broken.reproduction_spec("EXP-00001")


def test_missing_hypothesis_lineage_is_a_loud_violation(tmp_path, labels, temporal, datasets, temporal_digest):
    from orbit.schemas.hypothesis import HypothesisRegistry

    full = make_service(tmp_path, hypotheses=register_seeds(), labels=labels, temporal=temporal, datasets=datasets)
    full.register(make_spec(temporal_digest=temporal_digest), registered_at=_at(2026, 1, 1))
    full._registry.close()

    empty = make_service(tmp_path, hypotheses=HypothesisRegistry(hypotheses=[]), labels=labels, temporal=temporal, datasets=datasets)
    with pytest.raises(ValueError, match="lineage violation"):
        empty.reproduction_spec("EXP-00001")


def test_pinned_label_version_never_resolves_to_latest(service, labels, temporal_digest):
    # register LAB-001 v2 after the experiment pinned v1
    from orbit.labels.contract import AnchorMode, LabelContract, ReturnConvention

    labels.register(
        LabelContract(
            label_id="LAB-001", version="v2", target_type="forward_return",
            horizon=10, anchor_mode=AnchorMode.DECISION_INSTANT,
            return_convention=ReturnConvention.SIMPLE_PRICE_RETURN,
            formula="a different, later definition",
        )
    )
    _register_run_complete(service, make_spec(temporal_digest=temporal_digest))
    spec = service.reproduction_spec("EXP-00001")
    assert spec.label["version"] == "v1"
    assert spec.label["horizon"] == 5
    assert spec.label["content_hash"] != labels.definition_digest("LAB-001", "v2")


def test_different_label_versions_are_different_experiments(service, labels, temporal_digest):
    from orbit.labels.contract import AnchorMode, LabelContract, ReturnConvention

    labels.register(
        LabelContract(
            label_id="LAB-001", version="v2", target_type="forward_return",
            horizon=10, anchor_mode=AnchorMode.DECISION_INSTANT,
            return_convention=ReturnConvention.SIMPLE_PRICE_RETURN,
            formula="a different, later definition",
        )
    )
    _register_run_complete(service, make_spec("EXP-00001", temporal_digest=temporal_digest, label_version="v1"))
    _register_run_complete(
        service,
        make_spec("EXP-00002", temporal_digest=temporal_digest, label_version="v2"),
    )
    a = service.reproduction_spec("EXP-00001")
    b = service.reproduction_spec("EXP-00002")
    assert a.reproduction_digest != b.reproduction_digest
    assert a.label["version"] == "v1"
    assert b.label["version"] == "v2"


def test_different_code_hash_changes_reproduction_identity(service, temporal_digest):
    service.register(make_spec("EXP-00001", temporal_digest=temporal_digest), registered_at=_at(2026, 1, 1))
    service.mark_running("EXP-00001", code_hash="c" * 64, config_hash="g" * 64)
    service.complete("EXP-00001")
    a = service.reproduction_spec("EXP-00001")
    service.register(make_spec("EXP-00002", temporal_digest=temporal_digest), registered_at=_at(2026, 1, 1))
    service.mark_running("EXP-00002", code_hash="e" * 64, config_hash="g" * 64)
    service.complete("EXP-00002")
    b = service.reproduction_spec("EXP-00002")
    assert a.reproduction_digest != b.reproduction_digest


def test_reproduction_spec_marks_unresolved_when_no_registries(tmp_path, temporal_digest):
    svc = ExperimentService(db_path=tmp_path / "bare.duckdb")
    svc.register(make_spec(temporal_digest=temporal_digest), registered_at=_at(2026, 1, 1))
    spec = svc.reproduction_spec("EXP-00001")
    assert spec.datasets[0] == {"snapshot_id": "DS-000001", "resolved": False}
    assert spec.label == {"label_id": "LAB-001", "version": "v1", "resolved": False}
    assert spec.hypothesis is None
    assert spec.temporal is None
    assert spec.verify_digest()


def test_reproduction_spec_stable_with_null_and_decision_records(service, temporal_digest):
    _register_run_complete(service, make_spec(temporal_digest=temporal_digest))
    before = service.reproduction_spec("EXP-00001")
    service.record_result("EXP-00001", kind=ResultKind.NULL, summary="no evidence")
    service.record_decision(
        "EXP-00001", decision=Decision.REJECTED, reason="Failed after-cost OOS threshold.",
        decision_maker="orbit-research",
    )
    after = service.reproduction_spec("EXP-00001")
    assert before.reproduction_digest == after.reproduction_digest
    assert after.result["kind"] == "null"
    assert after.decision["decision"] == "rejected"