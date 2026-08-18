"""Phase 6 property/invariant tests (section 33).

The invariant report (`ExperimentService.validate_invariants`) is the machine
half of the audit: content-hash integrity, acyclicity, lineage completeness,
decision/result consistency and orphan-free records are all checked across
the whole ledger, not per-test.
"""

from __future__ import annotations

from datetime import datetime, timezone

import duckdb
import pytest

from conftest import make_spec

from orbit.experiments import Decision, ExperimentService, ResultKind


def _at(y, m=1, d=1):
    return datetime(y, m, d, tzinfo=timezone.utc)


def _scenario(service: ExperimentService, temporal_digest: str):
    """A rich ledger: roots, branches, a failure, a null rejection, a
    promotion, an archival - with artifacts, results and decisions."""
    # root momentum family
    service.register(make_spec("EXP-00001", temporal_digest=temporal_digest), registered_at=_at(2026, 1, 1))
    service.mark_running("EXP-00001", code_hash="a" * 64, config_hash="b" * 64)
    service.complete("EXP-00001")
    service.record_result("EXP-00001", kind=ResultKind.SUPPORTED, summary="IC 0.041, 4.2% excess")
    service.record_decision(
        "EXP-00001", decision=Decision.PROMOTED,
        reason="After-cost annual excess 4.2% exceeds Gate-C threshold 3%.",
        policy_version="Gate-C-v1", decision_maker="orbit-research",
    )
    service.attach_artifact("EXP-00001", kind="evaluation_table", path="tables/EXP-00001.csv")

    # branch A: null + rejection
    service.register(
        make_spec("EXP-00002", temporal_digest=temporal_digest, parent_id="EXP-00001", model={"family": "ridge"}),
        registered_at=_at(2026, 1, 2),
    )
    service.mark_running("EXP-00002", code_hash="a" * 64, config_hash="b" * 64)
    service.complete("EXP-00002")
    service.record_result("EXP-00002", kind=ResultKind.NULL, summary="NO SIGNIFICANT / NO ECONOMIC EVIDENCE")
    service.record_decision(
        "EXP-00002", decision=Decision.REJECTED,
        reason="OOS rank IC 0.008 below the 0.03 Gate-C threshold.",
        policy_version="Gate-C-v1", decision_maker="orbit-research",
    )

    # branch B: infrastructure failure + archival
    service.register(
        make_spec("EXP-00003", temporal_digest=temporal_digest, parent_id="EXP-00001", model={"family": "xgboost"}),
        registered_at=_at(2026, 1, 3),
    )
    service.mark_running("EXP-00003", code_hash="a" * 64, config_hash="b" * 64)
    service.fail("EXP-00003", note="node lost")
    service.record_result("EXP-00003", kind=ResultKind.INFRASTRUCTURE_FAILURE, summary="no outputs")
    service.retire("EXP-00003")

    # negative result on a completed experiment
    service.register(make_spec("EXP-00004", temporal_digest=temporal_digest), registered_at=_at(2026, 1, 4))
    service.mark_running("EXP-00004", code_hash="a" * 64, config_hash="b" * 64)
    service.complete("EXP-00004")
    service.record_result("EXP-00004", kind=ResultKind.NEGATIVE, summary="effect reverses OOS")
    return service


def test_full_scenario_passes_all_invariants(service, temporal_digest):
    _scenario(service, temporal_digest)
    report = service.validate_invariants()
    assert report["ok"], report["violations"]
    assert report["experiments"] == 4
    assert report["orphan_counts"] == {
        "experiment_state": 0,
        "artifacts": 0,
        "results": 0,
        "decisions": 0,
        "transitions": 0,
        "experiment_datasets": 0,
        "experiment_features": 0,
    }


def test_stored_identity_is_immutable_even_via_raw_sql(service, temporal_digest, tmp_path):
    _scenario(service, temporal_digest)
    # an attacker with raw DB access cannot rewrite the stored identity:
    # experiments is an FK-referenced, secondary-indexed table, so DuckDB
    # refuses EVERY update to it (belt) ...
    con = duckdb.connect(tmp_path / "experiments.duckdb")
    with pytest.raises(duckdb.ConstraintException):
        con.execute(
            "UPDATE experiments SET spec_json = ? WHERE experiment_id = ?",
            ["{}", "EXP-00001"],
        )
    with pytest.raises(duckdb.ConstraintException):
        con.execute(
            "UPDATE experiments SET content_hash = ? WHERE experiment_id = ?",
            ["0" * 64, "EXP-00001"],
        )
    con.close()
    # ... and validate_invariants recomputes every content hash (suspenders)
    report = service.validate_invariants()
    assert report["ok"], report["violations"]


def test_orphaned_state_row_is_impossible(service, temporal_digest, tmp_path):
    _scenario(service, temporal_digest)
    # a state row without an experiment violates the FK, so orphans cannot
    # even be injected with raw SQL
    con = duckdb.connect(tmp_path / "experiments.duckdb")
    with pytest.raises(duckdb.ConstraintException):
        con.execute(
            "INSERT INTO experiment_state VALUES "
            "('EXP-99999', 'running', NULL, NULL, '2026-01-01T00:00:00Z')"
        )
    con.close()
    report = service.validate_invariants()
    assert report["ok"], report["violations"]
    assert report["orphan_counts"]["experiment_state"] == 0


def test_no_parent_cycles_across_the_graph(service, temporal_digest):
    _scenario(service, temporal_digest)
    for exp in service.list():
        # walks without error and never revisits an ancestor
        seen: set[str] = set()
        for node in service.ancestry(exp.experiment_id):
            assert node.experiment_id not in seen
            seen.add(node.experiment_id)
        service.descendants(exp.experiment_id)


def test_completed_experiments_have_code_and_config_identity(service, temporal_digest):
    _scenario(service, temporal_digest)
    for exp in service.list(status="completed"):
        assert exp.code_hash
        assert exp.config_hash


def test_promoted_and_rejected_require_result_and_decision(service, temporal_digest):
    _scenario(service, temporal_digest)
    for status in ("promoted", "rejected"):
        for exp in service.list(status=status):
            assert service.result(exp.experiment_id) is not None
            assert service.decisions(exp.experiment_id)
            assert service.decisions(exp.experiment_id)[-1]["decision"] == status


def test_failure_does_not_erase_history(service, temporal_digest):
    _scenario(service, temporal_digest)
    statuses = service._registry.status_counts()
    assert statuses.get("promoted") == 1
    assert statuses.get("rejected") == 1
    assert statuses.get("completed") == 1
    assert statuses.get("retired") == 1
    # the failed run was archived, not erased: full history survives
    assert service.result("EXP-00003")["kind"] == "infrastructure_failure"
    assert [t["to_status"] for t in service.transitions("EXP-00003")] == [
        "running", "failed", "retired"
    ]


def test_reproduction_digest_stable_across_all_experiments(service, temporal_digest):
    _scenario(service, temporal_digest)
    for exp in service.list():
        spec = service.reproduction_spec(exp.experiment_id)
        assert spec.verify_digest()
        again = service.reproduction_spec(exp.experiment_id)
        assert spec.reproduction_digest == again.reproduction_digest


def test_registered_without_run_can_still_be_archived(service, temporal_digest):
    service.register(make_spec(temporal_digest=temporal_digest), registered_at=_at(2026, 1, 1))
    service.retire("EXP-00001")
    report = service.validate_invariants()
    assert report["ok"], report["violations"]  # retired before running is legal


def test_every_experiment_is_reconstructable(service, temporal_digest):
    _scenario(service, temporal_digest)
    for row in service._registry.dump():
        exp = service._reconstruct(row)
        assert exp.experiment_id == row["experiment_id"]
        assert exp.status.value == row["status"]
        assert exp.content_hash() == row["content_hash"]