"""Phase 6 revision regression tests (pre-completion audit).

Each test in this file pins a genuine research-record hole found during the
pre-completion three-review pass. If any of them ever starts failing because
the guard was removed, the hole is back:

  R1  label_id without label_version is refused ("never latest label")
  R2  registration requires pinned feature_refs (feature lineage)
  R3  record_decision requires a recorded result (decisions cite evidence)
  R4  research budget is hypothesis-scoped; a new hypothesis_family label
      cannot launder search history or escape the budget
  R5  the registry itself enforces the lifecycle (no direct-registry bypass)
  R6  the DB refuses garbage transition statuses and bad lineage ids
  R7  trial-number collisions (tampered counters) are detected by invariants
  R8  list() can filter by label_version
  R9  the budget check is atomic under concurrency (no overshoot)
  R10 the registry itself requires a recorded result before a decision
  R11 decision-vs-retire races have exactly one winner and never leak raw
      DuckDB transaction errors
  R12 state/audit-trail consistency: every status must be explained by the
      recorded transition chain and decisions; register() can only birth
      experiments (draft/registered), never teleport them into later states
  R13 child-record content integrity: results/decisions/artifacts/
      transitions carry a write-time content hash; raw-SQL rewrites of
      summary/reason/checksum/metrics/note are detected by invariants
  R14 lineage join rows must match the pinned spec: a raw-SQL UPDATE or
      DELETE of experiment_datasets / experiment_features rows is detected
"""

from __future__ import annotations

from datetime import datetime, timezone

import duckdb
import pytest
from pydantic import ValidationError

from conftest import make_spec, make_service

from orbit.experiments import Decision, ExperimentRegistry, ExperimentService, ResultKind
from orbit.experiments.registry import ExperimentRegistry as RawRegistry
from orbit.schemas.common import ExperimentStatus
from orbit.schemas.hypothesis import HypothesisSpec


def _at(y, m=1, d=1):
    return datetime(y, m, d, tzinfo=timezone.utc)


def _run(service: ExperimentService, experiment_id: str, code="c" * 64, config="g" * 64):
    service.mark_running(experiment_id, code_hash=code, config_hash=config)
    return service.complete(experiment_id)


# --------------------------------------------------------------------- R1


def test_label_id_without_label_version_is_refused_at_schema(service, temporal_digest):
    # 'latest label' must never be pinnable: label_id alone resolves to the
    # newest version at reproduction time, so the same experiment id could
    # silently mean different labels across time.
    with pytest.raises(ValidationError, match="label_version"):
        make_spec(temporal_digest=temporal_digest, label_id="LAB-001", label_version=None)


def test_label_id_without_label_version_is_refused_at_registration(service, temporal_digest):
    # model_copy(update=...) skips schema validation: the DATABASE CHECK
    # constraint is the second line of defense and must fire on its own
    spec = make_spec(temporal_digest=temporal_digest)
    spec = spec.model_copy(update={"label_id": "LAB-001", "label_version": None})
    with pytest.raises(ValueError, match="constraint violation"):
        service.register(spec, registered_at=_at(2026, 1, 1))
    assert service._registry.count() == 0


def test_latest_label_never_resolves_in_reproduction(service, temporal_digest):
    # after R1 the only way in is a pinned version; a pinned v1 experiment
    # must resolve to v1 even when v2 exists later
    from orbit.labels.contract import AnchorMode, LabelContract, ReturnConvention

    service._labels.register(
        LabelContract(
            label_id="LAB-001", version="v2", target_type="excess_return",
            horizon=10, anchor_mode=AnchorMode.DECISION_INSTANT,
            return_convention=ReturnConvention.SIMPLE_PRICE_RETURN,
            benchmark="SPY", formula="a later, different definition",
        )
    )
    service.register(make_spec(temporal_digest=temporal_digest), registered_at=_at(2026, 1, 1))
    spec = service.reproduction_spec("EXP-00001")
    assert spec.label["version"] == "v1"
    assert spec.spec.label_version == "v1"


# --------------------------------------------------------------------- R2


def test_registration_requires_pinned_feature_refs(service, temporal_digest):
    spec = make_spec(
        temporal_digest=temporal_digest,
        features={"feature_names": ["ret_12m_1m"], "feature_version": "v1", "feature_refs": []},
    )
    with pytest.raises(ValueError, match="feature_refs"):
        service.register(spec, registered_at=_at(2026, 1, 1))
    assert service._registry.count() == 0


def test_feature_lineage_rows_exist_for_every_registered_experiment(service, temporal_digest):
    service.register(make_spec(temporal_digest=temporal_digest), registered_at=_at(2026, 1, 1))
    con = duckdb.connect(service._registry.path)
    n = con.execute(
        "SELECT COUNT(*) FROM experiment_features WHERE experiment_id = 'EXP-00001'"
    ).fetchone()[0]
    con.close()
    assert n == 1


# --------------------------------------------------------------------- R3


def test_decision_without_recorded_result_is_refused(service, temporal_digest):
    service.register(make_spec(temporal_digest=temporal_digest), registered_at=_at(2026, 1, 1))
    _run(service, "EXP-00001")
    with pytest.raises(ValueError, match="recorded result"):
        service.record_decision(
            "EXP-00001", decision=Decision.PROMOTED,
            reason="After-cost excess beats Gate-C threshold.",
            decision_maker="orbit-research",
        )
    # the state machine must not have moved
    assert service.get("EXP-00001").status == ExperimentStatus.COMPLETED


def test_decision_after_result_still_works(service, temporal_digest):
    service.register(make_spec(temporal_digest=temporal_digest), registered_at=_at(2026, 1, 1))
    _run(service, "EXP-00001")
    service.record_result("EXP-00001", kind=ResultKind.SUPPORTED, summary="IC 0.04, 4.1% excess")
    exp = service.record_decision(
        "EXP-00001", decision=Decision.PROMOTED,
        reason="After-cost excess beats Gate-C threshold.",
        decision_maker="orbit-research",
    )
    assert exp.status == ExperimentStatus.PROMOTED
    assert service.validate_invariants()["ok"]


# --------------------------------------------------------------------- R4


def test_budget_is_hypothesis_scoped_and_family_labels_cannot_bypass(
    tmp_path, temporal_digest
):
    from hypotheses.seeds import register_seeds

    hypotheses = register_seeds()
    hypotheses.hypotheses[0] = HypothesisSpec.model_validate(
        {**hypotheses.hypotheses[0].model_dump(), "research_budget": {"max_trials": 1}}
    )
    svc = make_service(tmp_path, hypotheses=hypotheses)
    svc.register(
        make_spec("EXP-00001", temporal_digest=temporal_digest, hypothesis_family="fam-a"),
        registered_at=_at(2026, 1, 1),
    )
    # a new family label is a search-depth disguise, not a budget escape
    with pytest.raises(ValueError, match="research budget exhausted"):
        svc.register(
            make_spec("EXP-00002", temporal_digest=temporal_digest, hypothesis_family="fam-b"),
            registered_at=_at(2026, 1, 2),
        )
    assert svc._registry.count() == 1


def test_count_trials_counts_hypothesis_regardless_of_family(service, temporal_digest):
    service.register(
        make_spec("EXP-00001", temporal_digest=temporal_digest, hypothesis_family="momentum"),
        registered_at=_at(2026, 1, 1),
    )
    service.fail("EXP-00001")
    service.register(
        make_spec("EXP-00002", temporal_digest=temporal_digest, hypothesis_family="momentum"),
        registered_at=_at(2026, 1, 2),
    )
    assert service.count_trials("H-001") == 2
    assert service._registry.trial_count("momentum") == 2


# --------------------------------------------------------------------- R5


def test_registry_itself_enforces_the_lifecycle(tmp_path, temporal_digest):
    svc = make_service(tmp_path)
    svc.register(make_spec(temporal_digest=temporal_digest), registered_at=_at(2026, 1, 1))
    svc._registry.close()

    raw = RawRegistry(db_path=str(tmp_path / "experiments.duckdb"))
    try:
        # direct-registry users get the same state machine as the service
        with pytest.raises(ValueError, match="invalid experiment transition"):
            raw.transition(
                experiment_id="EXP-00001", from_status="registered", to_status="completed",
                transitioned_at=_at(2026, 1, 2).isoformat(),
            )
        # and decision states can never be entered by a bare transition
        with pytest.raises(ValueError, match="decision state"):
            raw.transition(
                experiment_id="EXP-00001", from_status="completed", to_status="rejected",
                transitioned_at=_at(2026, 1, 2).isoformat(),
            )
        assert raw.get("EXP-00001")["status"] == "registered"
    finally:
        raw.close()


def test_registry_rejects_unknown_statuses_on_transition(tmp_path, temporal_digest):
    svc = make_service(tmp_path)
    svc.register(make_spec(temporal_digest=temporal_digest), registered_at=_at(2026, 1, 1))
    svc._registry.close()
    raw = RawRegistry(db_path=str(tmp_path / "experiments.duckdb"))
    try:
        with pytest.raises(ValueError, match="invalid experiment status"):
            raw.transition(
                experiment_id="EXP-00001", from_status="registered", to_status="deleted",
                transitioned_at=_at(2026, 1, 2).isoformat(),
            )
    finally:
        raw.close()


# --------------------------------------------------------------------- R6


def test_db_refuses_garbage_transition_rows(tmp_path, temporal_digest):
    svc = make_service(tmp_path)
    svc.register(make_spec(temporal_digest=temporal_digest), registered_at=_at(2026, 1, 1))
    svc._registry.close()
    con = duckdb.connect(str(tmp_path / "experiments.duckdb"))
    with pytest.raises(duckdb.ConstraintException):
        con.execute(
            "INSERT INTO transitions VALUES "
            "('EXP-00001', 'garbage', 'deleted', '2026-01-02T00:00:00Z', NULL, "
            "'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')"
        )
    con.close()


def test_db_refuses_bad_lineage_join_ids(tmp_path, temporal_digest):
    svc = make_service(tmp_path)
    svc.register(make_spec(temporal_digest=temporal_digest), registered_at=_at(2026, 1, 1))
    svc._registry.close()
    con = duckdb.connect(str(tmp_path / "experiments.duckdb"))
    with pytest.raises(duckdb.ConstraintException):
        con.execute(
            "INSERT INTO experiment_datasets VALUES ('EXP-00001', 'latest')"
        )
    with pytest.raises(duckdb.ConstraintException):
        con.execute(
            "INSERT INTO experiment_features VALUES ('EXP-00001', 'momentum', 'v1')"
        )
    with pytest.raises(duckdb.ConstraintException):
        con.execute(
            "INSERT INTO experiments VALUES "
            "('EXP-1', 'H-1', 'forged', NULL, '{}', '0'*64, 42, NULL, NULL, "
            "NULL, NULL, NULL, 1, 0, 'attacker', "
            "'2026-01-02T00:00:00Z', '2026-01-02T00:00:00Z')"
        )
    con.close()


def test_db_refuses_experiment_with_label_id_but_no_version(tmp_path, temporal_digest):
    svc = make_service(tmp_path)
    svc.register(make_spec(temporal_digest=temporal_digest), registered_at=_at(2026, 1, 1))
    svc._registry.close()
    con = duckdb.connect(str(tmp_path / "experiments.duckdb"))
    with pytest.raises(duckdb.ConstraintException):
        con.execute(
            "INSERT INTO experiments VALUES "
            "('EXP-00002', 'H-001', 'forged', NULL, '{}', '0'*64, 42, "
            "'LAB-001', NULL, NULL, NULL, NULL, 1, 0, 'attacker', "
            "'2026-01-02T00:00:00Z', '2026-01-02T00:00:00Z')"
        )
    con.close()


# --------------------------------------------------------------------- R7


def test_trial_number_collision_is_detected_by_invariants(tmp_path, temporal_digest):
    svc = make_service(tmp_path)
    svc.register(make_spec("EXP-00001", temporal_digest=temporal_digest), registered_at=_at(2026, 1, 1))
    svc.register(make_spec("EXP-00002", temporal_digest=temporal_digest), registered_at=_at(2026, 1, 2))
    svc._registry.close()

    # an attacker resets the per-family counter: the next registration gets
    # a duplicate trial ordinal (the search depth is falsified)
    con = duckdb.connect(str(tmp_path / "experiments.duckdb"))
    con.execute("UPDATE trial_counters SET value = 0 WHERE family = 'H-001'")
    con.close()

    svc2 = make_service(tmp_path)
    svc2.register(make_spec("EXP-00003", temporal_digest=temporal_digest), registered_at=_at(2026, 1, 3))
    assert [e.trial_number for e in svc2.list()] == [1, 2, 1]
    report = svc2.validate_invariants()
    assert not report["ok"]
    assert any("trial-number integrity" in v for v in report["violations"])


def test_trial_numbers_are_contiguous_in_an_untampered_ledger(service, temporal_digest):
    service.register(make_spec("EXP-00001", temporal_digest=temporal_digest), registered_at=_at(2026, 1, 1))
    service.fail("EXP-00001")
    service.register(make_spec("EXP-00002", temporal_digest=temporal_digest), registered_at=_at(2026, 1, 2))
    service.retire("EXP-00002")
    service.register(make_spec("EXP-00003", temporal_digest=temporal_digest), registered_at=_at(2026, 1, 3))
    assert service.validate_invariants()["ok"]


# --------------------------------------------------------------------- R8


def test_list_filters_by_label_version(service, temporal_digest):
    from orbit.labels.contract import AnchorMode, LabelContract, ReturnConvention

    service._labels.register(
        LabelContract(
            label_id="LAB-001", version="v2", target_type="excess_return",
            horizon=10, anchor_mode=AnchorMode.DECISION_INSTANT,
            return_convention=ReturnConvention.SIMPLE_PRICE_RETURN,
            benchmark="SPY", formula="a later, different definition",
        )
    )
    service.register(
        make_spec("EXP-00001", temporal_digest=temporal_digest, label_version="v1"),
        registered_at=_at(2026, 1, 1),
    )
    service.register(
        make_spec("EXP-00002", temporal_digest=temporal_digest, label_version="v2"),
        registered_at=_at(2026, 1, 2),
    )
    assert [e.experiment_id for e in service.list(label_version="v1")] == ["EXP-00001"]
    assert [e.experiment_id for e in service.list(label_version="v2")] == ["EXP-00002"]
    assert [e.experiment_id for e in service.list(label_id="LAB-001", label_version="v2")] == [
        "EXP-00002"
    ]


# --------------------------------------------------------------------- R10


def test_registry_level_decision_requires_recorded_result(tmp_path, temporal_digest):
    """The ledger itself enforces decisions-cite-evidence: a direct
    ExperimentRegistry user cannot record a decision on a completed
    experiment that has no result (the service check is not the only one)."""
    svc = make_service(tmp_path)
    svc.register(make_spec(temporal_digest=temporal_digest), registered_at=_at(2026, 1, 1))
    _run(svc, "EXP-00001")
    svc._registry.close()

    raw = RawRegistry(db_path=str(tmp_path / "experiments.duckdb"))
    try:
        with pytest.raises(ValueError, match="recorded result"):
            raw.record_decision(
                experiment_id="EXP-00001", decision="rejected",
                reason="OOS IC below the Gate-C threshold.", policy_version="Gate-C-v1",
                decision_maker="attacker", decided_at=_at(2026, 1, 2).isoformat(),
            )
        assert raw.get("EXP-00001")["status"] == "completed"
        assert raw.decisions("EXP-00001") == []
    finally:
        raw.close()


def test_registry_level_decision_works_with_result(tmp_path, temporal_digest):
    svc = make_service(tmp_path)
    svc.register(make_spec(temporal_digest=temporal_digest), registered_at=_at(2026, 1, 1))
    _run(svc, "EXP-00001")
    svc.record_result("EXP-00001", kind=ResultKind.NULL, summary="no evidence")
    svc._registry.close()

    raw = RawRegistry(db_path=str(tmp_path / "experiments.duckdb"))
    try:
        raw.record_decision(
            experiment_id="EXP-00001", decision="rejected",
            reason="OOS IC below the Gate-C threshold.", policy_version="Gate-C-v1",
            decision_maker="researcher", decided_at=_at(2026, 1, 2).isoformat(),
        )
        assert raw.get("EXP-00001")["status"] == "rejected"
    finally:
        raw.close()


# --------------------------------------------------------------------- R11


def test_decision_vs_retire_race_has_exactly_one_winner_and_no_crash(
    tmp_path, hypotheses, labels, temporal, datasets, temporal_digest
):
    """A decision racing a retire must produce exactly one winner with a
    clean, catchable ValueError for the loser - never a leaked
    duckdb.TransactionException and never two winners."""
    import threading

    svc = make_service(
        tmp_path, hypotheses=hypotheses, labels=labels, temporal=temporal, datasets=datasets
    )
    svc.register(make_spec(temporal_digest=temporal_digest), registered_at=_at(2026, 1, 1))
    _run(svc, "EXP-00001")
    svc.record_result("EXP-00001", kind=ResultKind.NULL, summary="no evidence")
    svc._registry.close()

    barrier = threading.Barrier(2)
    outcomes: list[str] = []
    unexpected: list[BaseException] = []

    def worker(decide: bool):
        s = make_service(
            tmp_path, hypotheses=hypotheses, labels=labels, temporal=temporal, datasets=datasets
        )
        barrier.wait()
        try:
            if decide:
                s.record_decision(
                    "EXP-00001", decision=Decision.REJECTED,
                    reason="OOS IC below the Gate-C threshold.",
                    policy_version="Gate-C-v1", decision_maker="r1",
                )
            else:
                s.retire("EXP-00001", note="archival")
            outcomes.append("ok")
        except ValueError:
            outcomes.append("failed")
        except Exception as exc:  # noqa: BLE001 - must not leak raw DuckDB errors
            unexpected.append(exc)
            outcomes.append("crashed")
        finally:
            s._registry.close()

    threads = [
        threading.Thread(target=worker, args=(True,)),
        threading.Thread(target=worker, args=(False,)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # the loser fails with a clean, catchable ValueError - never a leaked
    # duckdb.TransactionException from the retry path
    assert unexpected == [], [repr(e) for e in unexpected]
    assert sorted(outcomes) == ["failed", "ok"], outcomes
    check = RawRegistry(db_path=str(tmp_path / "experiments.duckdb"))
    try:
        final = check.get("EXP-00001")
        # either the retire won (retired, 0 decisions) or the decision won
        # (rejected, 1 decision); both are valid, both are exactly one write
        if final["status"] == "retired":
            assert check.decisions("EXP-00001") == []
        else:
            assert final["status"] == "rejected"
            assert len(check.decisions("EXP-00001")) == 1
        assert check.orphan_counts() == {
            "experiment_state": 0, "artifacts": 0, "results": 0,
            "decisions": 0, "transitions": 0,
            "experiment_datasets": 0, "experiment_features": 0,
        }
    finally:
        check.close()


# --------------------------------------------------------------------- R9


def test_budget_check_is_atomic_under_concurrency(
    tmp_path, hypotheses, labels, temporal, datasets, temporal_digest
):
    import threading

    from hypotheses.seeds import register_seeds

    hypotheses = register_seeds()
    hypotheses.hypotheses[0] = HypothesisSpec.model_validate(
        {**hypotheses.hypotheses[0].model_dump(), "research_budget": {"max_trials": 1}}
    )
    n = 4
    barrier = threading.Barrier(n)
    outcomes: list[str] = []

    def worker(i: int):
        svc = ExperimentService(
            registry=ExperimentRegistry(db_path=str(tmp_path / "experiments.duckdb")),
            hypothesis_registry=hypotheses, label_registry=labels, temporal_contract=temporal,
            dataset_registry=datasets,
        )
        barrier.wait()
        try:
            svc.register(
                make_spec(f"EXP-{i:05d}", temporal_digest=temporal_digest),
                registered_at=_at(2026, 1, 1),
            )
            outcomes.append("ok")
        except ValueError:
            outcomes.append("budget")
        finally:
            svc._registry.close()

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert outcomes.count("ok") == 1, outcomes
    assert outcomes.count("budget") == n - 1, outcomes
    con = ExperimentRegistry(db_path=str(tmp_path / "experiments.duckdb"))
    assert con.count() == 1
    con.close()


# --------------------------------------------------------------------- R12


def _audit_violations(tmp_path):
    svc = make_service(tmp_path)
    try:
        return svc.validate_invariants()["violations"]
    finally:
        svc._registry.close()


def test_state_tamper_without_audit_trail_is_detected(tmp_path, temporal_digest):
    """`experiment_state` is not FK-referenced, so its status column can be
    rewritten by raw SQL. Every such change must be detected: a status with
    no matching transition record (or a transition chain that cannot explain
    it) is an audit violation."""
    for tampered in ("retired", "completed", "failed"):
        dirname = tmp_path / tampered
        dirname.mkdir(parents=True)
        svc = make_service(dirname)
        svc.register(make_spec(temporal_digest=temporal_digest), registered_at=_at(2026, 1, 1))
        svc._registry.close()
        con = duckdb.connect(str(dirname / "experiments.duckdb"))
        con.execute(
            "UPDATE experiment_state SET status = ? WHERE experiment_id = 'EXP-00001'",
            [tampered],
        )
        con.close()
        violations = _audit_violations(dirname)
        assert any("no transition record" in v for v in violations), (
            tampered, violations
        )


def test_register_refuses_birth_into_later_lifecycle_states(tmp_path, temporal_digest):
    """register() is the only entry point of the ledger; an experiment is
    born, never teleported into a later lifecycle state with no history."""
    import json

    tmp_path.mkdir(parents=True, exist_ok=True)
    svc = make_service(tmp_path)
    spec = make_spec(temporal_digest=temporal_digest)
    svc._registry.close()
    raw = RawRegistry(db_path=str(tmp_path / "experiments.duckdb"))
    try:
        for forbidden in ("running", "completed", "failed", "rejected", "promoted", "retired"):
            with pytest.raises(ValueError, match="birth status"):
                raw.register(
                    experiment_id="EXP-00001", hypothesis_id="H-001", title="x",
                    parent_id=None, status=forbidden,
                    spec_json=json.dumps(spec.model_dump(mode="json"), sort_keys=True),
                    content_hash=spec.content_hash(), code_hash="c" * 64,
                    config_hash="g" * 64, seed=1, label_id=None, label_version=None,
                    hypothesis_family=None, research_epoch=None, selection_stage=None,
                    trial_family="H-001", declared_trial_number=None, declared_prior=None,
                    researcher="attacker", created_at=_at(2026, 1, 1).isoformat(),
                    registered_at=_at(2026, 1, 1).isoformat(),
                    dataset_snapshot_ids=["DS-000001"],
                    feature_rows=[("FEAT-001", "v1")],
                )
        ids = {"draft": "EXP-00003", "registered": "EXP-00002"}
        for allowed in ("draft", "registered"):
            raw.register(
                experiment_id=ids[allowed],
                hypothesis_id="H-001", title="x", parent_id=None, status=allowed,
                spec_json=json.dumps(spec.model_dump(mode="json"), sort_keys=True),
                content_hash=spec.content_hash(), code_hash=None, config_hash=None,
                seed=1, label_id=None, label_version=None, hypothesis_family=None,
                research_epoch=None, selection_stage=None, trial_family="H-001",
                declared_trial_number=None, declared_prior=None, researcher="attacker",
                created_at=_at(2026, 1, 1).isoformat(),
                registered_at=_at(2026, 1, 1).isoformat(),
                dataset_snapshot_ids=["DS-000001"], feature_rows=[("FEAT-001", "v1")],
            )
        assert _audit_violations(tmp_path) == []
    finally:
        raw.close()


def test_decision_state_requires_exactly_one_matching_decision(
    tmp_path, temporal_digest
):
    """A decision state must have exactly one decision, and its value must
    match the status: a forged second decision row (or a state tamper that
    disagrees with the recorded decision) is an audit violation."""
    svc = make_service(tmp_path)
    svc.register(make_spec(temporal_digest=temporal_digest), registered_at=_at(2026, 1, 1))
    _run(svc, "EXP-00001")
    svc.record_result("EXP-00001", kind=ResultKind.NULL, summary="no evidence")
    svc.record_decision(
        "EXP-00001", decision=Decision.REJECTED,
        reason="OOS IC below the Gate-C threshold.",
        policy_version="Gate-C-v1", decision_maker="r1",
    )
    svc._registry.close()

    con = duckdb.connect(str(tmp_path / "experiments.duckdb"))
    con.execute(
        "INSERT INTO decisions VALUES "
        "('DEC-999999', 'EXP-00001', 'promoted', 'forged', 'Gate-C-v1', 'attacker', "
        "'2026-01-02T00:00:00Z', "
        "'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')"
    )
    con.close()
    violations = _audit_violations(tmp_path)
    assert any("exactly one is allowed" in v for v in violations), violations

    # now a state tamper that contradicts the recorded decision
    con = duckdb.connect(str(tmp_path / "experiments.duckdb"))
    con.execute("DELETE FROM decisions WHERE decision_id = 'DEC-999999'")
    con.execute(
        "UPDATE experiment_state SET status = 'promoted' WHERE experiment_id = 'EXP-00001'"
    )
    con.close()
    violations = _audit_violations(tmp_path)
    assert any("does not match status promoted" in v for v in violations), violations
    assert any("does not match the last recorded transition" in v for v in violations), violations


def test_forged_transition_chain_is_detected(tmp_path, temporal_digest):
    """A forged transition row must itself be consistent: the chain must
    start at a birth state, stitch hop-to-hop, and only take valid lifecycle
    steps - registered->completed (no running) is an audit violation."""
    svc = make_service(tmp_path)
    svc.register(make_spec(temporal_digest=temporal_digest), registered_at=_at(2026, 1, 1))
    svc._registry.close()
    con = duckdb.connect(str(tmp_path / "experiments.duckdb"))
    con.execute(
        "UPDATE experiment_state SET status = 'completed', "
        "code_hash = repeat('c', 64), config_hash = repeat('g', 64) "
        "WHERE experiment_id = 'EXP-00001'"
    )
    con.execute(
        "INSERT INTO transitions VALUES "
        "('EXP-00001', 'registered', 'completed', '2026-01-01T00:00:00Z', 'forged', "
        "'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')"
    )
    con.close()
    violations = _audit_violations(tmp_path)
    assert any("invalid experiment transition" in v for v in violations), violations


def test_legit_lifecycle_chains_pass_invariants(tmp_path, temporal_digest):
    """Control: every sanctioned path (full decision chain, retired chain)
    has a valid, self-consistent audit trail - no false positives."""
    svc = make_service(tmp_path)
    svc.register(make_spec(temporal_digest=temporal_digest), registered_at=_at(2026, 1, 1))
    _run(svc, "EXP-00001")
    svc.record_result("EXP-00001", kind=ResultKind.NULL, summary="no evidence")
    svc.record_decision(
        "EXP-00001", decision=Decision.REJECTED,
        reason="OOS IC below the Gate-C threshold.",
        policy_version="Gate-C-v1", decision_maker="r1",
    )
    svc.register(make_spec("EXP-00002", temporal_digest=temporal_digest), registered_at=_at(2026, 1, 2))
    svc.retire("EXP-00002", note="archival")
    svc.register(make_spec("EXP-00003", temporal_digest=temporal_digest), registered_at=_at(2026, 1, 3))
    svc.fail("EXP-00003", note="infra")
    violations = _audit_violations(tmp_path)
    assert violations == []


# --------------------------------------------------------------------- R13


def test_child_record_content_tampering_is_detected(tmp_path, temporal_digest):
    """results / decisions / artifacts / transitions are FK-referencing
    tables - no index protects their columns from raw-SQL UPDATE. Every
    record stores a content hash at write time; rewriting the summary, the
    decision reason (the cited evidence), the artifact checksum, the metrics
    or a transition note must be flagged by validate_invariants."""
    from orbit.experiments.registry import record_content_hash

    def _tamper(dir_name, tampered_table, sql):
        tmp = tmp_path / dir_name
        tmp.mkdir(parents=True)
        svc = make_service(tmp)
        svc.register(make_spec(temporal_digest=temporal_digest), registered_at=_at(2026, 1, 1))
        _run(svc, "EXP-00001")
        svc.record_result("EXP-00001", kind=ResultKind.NULL, summary="no evidence",
                          metrics={"oos_ic": 0.01})
        svc.attach_artifact("EXP-00001", kind="metrics", path="metrics.json",
                            checksum="abc123")
        svc.record_decision(
            "EXP-00001", decision=Decision.REJECTED,
            reason="OOS IC below the Gate-C threshold.",
            policy_version="Gate-C-v1", decision_maker="r1",
        )
        svc._registry.close()
        con = duckdb.connect(str(tmp / "experiments.duckdb"))
        con.execute(sql)
        con.close()
        violations = _audit_violations(tmp)
        assert any(
            f"{tampered_table} row content_hash mismatch" in v for v in violations
        ), (tampered_table, violations)

    _tamper("results", "results", "UPDATE results SET summary = 'forged: huge success' WHERE experiment_id = 'EXP-00001'")
    _tamper("decisions", "decisions", "UPDATE decisions SET reason = 'forged: promoted' WHERE experiment_id = 'EXP-00001'")
    _tamper("artifacts", "artifacts", "UPDATE artifacts SET checksum = 'tampered' WHERE experiment_id = 'EXP-00001'")
    _tamper("transitions", "transitions", "UPDATE transitions SET note = 'forged note' WHERE experiment_id = 'EXP-00001'")
    _tamper("metrics", "results", "UPDATE results SET metrics_json = '{\"oos_ic\": 0.99}' WHERE experiment_id = 'EXP-00001'")


def test_record_content_hash_is_deterministic_and_timestamp_canonical():
    """The content hash is deterministic, sensitive to every field, and
    canonicalizes timestamps: the ISO string the registry receives and the
    naive datetime DuckDB returns hash identically."""
    from datetime import datetime as _dt, timezone as _tz

    from orbit.experiments.registry import record_content_hash

    payload = {
        "decision_id": "DEC-000001",
        "experiment_id": "EXP-00001",
        "decision": "rejected",
        "reason": "OOS IC below the Gate-C threshold.",
        "policy_version": "Gate-C-v1",
        "decision_maker": "r1",
        "decided_at": "2026-01-02T00:00:00Z",
    }
    a = record_content_hash(payload)
    b = record_content_hash({**payload, "decided_at": "2026-01-02T00:00:00+00:00"})
    c = record_content_hash({**payload, "decided_at": _dt(2026, 1, 2, tzinfo=_tz.utc)})
    d = record_content_hash({**payload, "decided_at": _dt(2026, 1, 2)})
    assert a == b == c == d
    assert len(a) == 64
    altered = record_content_hash({**payload, "reason": "forged: promoted"})
    assert altered != a


# --------------------------------------------------------------------- R14


def test_lineage_join_rows_must_match_the_pinned_spec(tmp_path, temporal_digest):
    """experiment_datasets / experiment_features are FK-referencing tables:
    raw-SQL UPDATE or DELETE of a join row silently rewrites (or erases) the
    experiment's real lineage while the hash-protected spec still pins the
    original. validate_invariants must flag every divergence."""
    def _mutate(dir_name, sql):
        tmp = tmp_path / dir_name
        tmp.mkdir(parents=True)
        svc = make_service(tmp)
        svc.register(make_spec(temporal_digest=temporal_digest), registered_at=_at(2026, 1, 1))
        svc._registry.close()
        con = duckdb.connect(str(tmp / "experiments.duckdb"))
        con.execute(sql)
        con.close()
        return _audit_violations(tmp)

    violations = _mutate(
        "ds-swap",
        "UPDATE experiment_datasets SET dataset_snapshot_id = 'DS-000002' "
        "WHERE experiment_id = 'EXP-00001'",
    )
    assert any("dataset lineage join rows" in v for v in violations), violations

    violations = _mutate(
        "ds-delete",
        "DELETE FROM experiment_datasets WHERE experiment_id = 'EXP-00001'",
    )
    assert any("dataset lineage join rows" in v for v in violations), violations

    violations = _mutate(
        "feat-id",
        "UPDATE experiment_features SET feature_id = 'FEAT-999' "
        "WHERE experiment_id = 'EXP-00001'",
    )
    assert any("feature lineage join rows" in v for v in violations), violations

    violations = _mutate(
        "feat-ver",
        "UPDATE experiment_features SET feature_version = 'v9' "
        "WHERE experiment_id = 'EXP-00001'",
    )
    assert any("feature lineage join rows" in v for v in violations), violations


def test_untampered_lineage_join_rows_pass_invariants(tmp_path, temporal_digest):
    svc = make_service(tmp_path)
    spec = make_spec(temporal_digest=temporal_digest)
    svc.register(spec, registered_at=_at(2026, 1, 1))
    assert _audit_violations(tmp_path) == []
    joins = svc._registry.lineage_joins("EXP-00001")
    assert joins["datasets"] == sorted(set(spec.dataset_snapshot_ids))
    assert joins["features"] == sorted(
        {(f.feature_id, f.feature_version) for f in spec.features.feature_refs}
    )
    svc._registry.close()

