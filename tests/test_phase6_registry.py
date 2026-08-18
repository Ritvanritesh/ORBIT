"""Phase 6 tests: registration, lifecycle, genealogy, lineage, immutability,
search/filtering, artifacts, results and decisions.

All tests are hermetic (tmp_path registries + fake datasets). The registry
under test is `orbit.experiments` (DuckDB-backed research control plane).
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from conftest import DS_000001, make_spec

from orbit.experiments import Decision, ExperimentService, ResultKind
from orbit.schemas.common import ExperimentStatus
from orbit.schemas.experiment import TemporalConfigRef


def _at(y, m=1, d=1):
    return datetime(y, m, d, tzinfo=timezone.utc)


def _register(service: ExperimentService, spec, *, at=_at(2026, 1, 1), **kw):
    return service.register(spec, registered_at=at, **kw)


def _run(service: ExperimentService, experiment_id: str, code="c" * 64, config="g" * 64):
    service.mark_running(experiment_id, code_hash=code, config_hash=config)
    return service.complete(experiment_id)


# -------------------------------------------------------------- registration


def test_valid_registration_enters_registered_with_computed_trials(service, temporal_digest):
    spec = make_spec(temporal_digest=temporal_digest)
    exp = _register(service, spec, at=_at(2026, 1, 1))
    assert exp.status == ExperimentStatus.REGISTERED
    assert exp.trial_number == 1
    assert exp.number_of_prior_trials == 0
    assert exp.registered_at == _at(2026, 1, 1)
    assert exp.created_at == _at(2026, 1, 1)
    assert exp.content_hash() == spec.content_hash()
    assert service.get("EXP-00001").experiment_id == "EXP-00001"


def test_registration_forces_registered_status_even_for_draft_spec(service, temporal_digest):
    spec = make_spec(temporal_digest=temporal_digest, status=ExperimentStatus.DRAFT)
    exp = _register(service, spec)
    assert exp.status == ExperimentStatus.REGISTERED


def test_duplicate_registration_is_refused(service, temporal_digest):
    _register(service, make_spec(temporal_digest=temporal_digest))
    with pytest.raises(ValueError, match="duplicate experiment id"):
        _register(service, make_spec(temporal_digest=temporal_digest))


def test_raw_dict_registration_is_refused(service):
    with pytest.raises(TypeError, match="ExperimentSpec"):
        service.register({"experiment_id": "EXP-00001"})  # type: ignore[arg-type]


def test_registration_requires_exact_dataset_snapshots(service, temporal_digest):
    spec = make_spec(temporal_digest=temporal_digest, dataset_snapshot_ids=[])
    with pytest.raises(ValueError, match="dataset_snapshot_ids"):
        _register(service, spec)


def test_registration_rejects_unknown_dataset_snapshot(service, temporal_digest):
    spec = make_spec(temporal_digest=temporal_digest, dataset_snapshot_ids=["DS-999999"])
    with pytest.raises(ValueError, match="unknown dataset snapshot"):
        _register(service, spec)


def test_registration_rejects_invalid_snapshot_id_pattern(service, temporal_digest):
    # the DS-xxxxxx pattern is enforced at schema construction, so a malformed
    # snapshot id can never reach the registry
    with pytest.raises(ValidationError, match="DS-xxxxxx"):
        make_spec(temporal_digest=temporal_digest, dataset_snapshot_ids=["latest"])


def test_registration_requires_temporal_config(service):
    spec = make_spec(temporal_digest="x" * 64, temporal_config=None)
    with pytest.raises(ValueError, match="temporal_config"):
        _register(service, spec)


def test_registration_rejects_temporal_digest_mismatch(service, temporal_digest):
    spec = make_spec(
        temporal_digest=temporal_digest,
        temporal_config=TemporalConfigRef(engine_version="v1.0.0", config_digest="0" * 64),
    )
    with pytest.raises(ValueError, match="config_digest does not match"):
        _register(service, spec)


def test_registration_rejects_temporal_engine_mismatch(service, temporal_digest):
    spec = make_spec(
        temporal_digest=temporal_digest,
        temporal_config=TemporalConfigRef(engine_version="v99.0.0", config_digest=temporal_digest),
    )
    with pytest.raises(ValueError, match="engine_version"):
        _register(service, spec)


def test_registration_validates_hypothesis_exists(service, temporal_digest):
    spec = make_spec(temporal_digest=temporal_digest, hypothesis_id="H-999")
    with pytest.raises(ValueError, match="unknown hypothesis"):
        _register(service, spec)


def test_registration_refuses_draft_hypothesis(tmp_path, temporal_digest):
    from hypotheses.seeds import build_seed_registry
    from conftest import make_service

    svc = make_service(tmp_path, hypotheses=build_seed_registry())  # drafts, not registered
    with pytest.raises(ValueError, match="registered hypotheses"):
        _register(svc, make_spec(temporal_digest=temporal_digest))


def test_registration_validates_label_reference(service, temporal_digest):
    spec = make_spec(temporal_digest=temporal_digest, label_id="LAB-999")
    with pytest.raises(ValueError, match="unknown label reference"):
        _register(service, spec)


def test_registration_resolves_pinned_label_version(service, temporal_digest):
    spec = make_spec(temporal_digest=temporal_digest, label_id="LAB-003", label_version="v1")
    _register(service, spec)
    assert service.get("EXP-00001").label_id == "LAB-003"


def test_label_version_without_label_id_rejected_at_schema(service, temporal_digest):
    with pytest.raises(ValidationError, match="label_version requires label_id"):
        make_spec(temporal_digest=temporal_digest, label_id=None, label_version="v1")


def test_registration_enforces_research_budget(tmp_path, hypotheses, temporal_digest):
    from orbit.schemas.hypothesis import HypothesisSpec
    from conftest import make_service

    # model_copy(update=...) skips re-validation, so build a properly
    # validated HypothesisSpec with the reduced budget instead
    hypotheses.hypotheses[0] = HypothesisSpec.model_validate(
        {**hypotheses.hypotheses[0].model_dump(), "research_budget": {"max_trials": 1}}
    )
    svc = make_service(tmp_path, hypotheses=hypotheses)
    _register(svc, make_spec("EXP-00001", temporal_digest=temporal_digest))
    with pytest.raises(ValueError, match="research budget exhausted"):
        _register(svc, make_spec("EXP-00002", temporal_digest=temporal_digest))


def test_trial_counters_are_computed_not_declared(service, temporal_digest):
    _register(service, make_spec("EXP-00001", temporal_digest=temporal_digest))
    with pytest.raises(ValueError, match="trial_number is computed"):
        _register(
            service,
            make_spec("EXP-00002", temporal_digest=temporal_digest, trial_number=7),
        )
    _register(service, make_spec("EXP-00003", temporal_digest=temporal_digest))
    exp = service.get("EXP-00003")
    assert exp.trial_number == 2
    assert exp.number_of_prior_trials == 1


def test_prior_trials_counter_cannot_be_declared(service, temporal_digest):
    with pytest.raises(ValueError, match="number_of_prior_trials is computed"):
        _register(
            service,
            make_spec(temporal_digest=temporal_digest, number_of_prior_trials=3),
        )


def test_failed_and_rejected_experiments_count_as_prior_trials(service, temporal_digest):
    first = _register(service, make_spec("EXP-00001", temporal_digest=temporal_digest))
    service.fail("EXP-00001")
    assert first.number_of_prior_trials == 0
    second = _register(service, make_spec("EXP-00002", temporal_digest=temporal_digest))
    assert second.trial_number == 2
    assert second.number_of_prior_trials == 1
    service.retire("EXP-00002")
    # trial ordinals never renumber: the full search history is preserved,
    # even for archived experiments
    third = _register(service, make_spec("EXP-00003", temporal_digest=temporal_digest))
    assert third.trial_number == 3
    assert third.number_of_prior_trials == 2
    # the live budget, however, ignores retired experiments (Phase 1 semantics)
    assert service.count_trials("H-001") == 2


def test_family_trial_counting_uses_declared_family(service, temporal_digest):
    _register(service, make_spec("EXP-00001", temporal_digest=temporal_digest, hypothesis_family="momentum"))
    _register(
        service,
        make_spec("EXP-00002", hypothesis_id="H-003", temporal_digest=temporal_digest, hypothesis_family="momentum"),
    )
    exp = service.get("EXP-00002")
    assert exp.trial_number == 2  # same family, across hypotheses


def test_feature_count_must_match_pinned_refs(service, temporal_digest):
    with pytest.raises(ValueError, match="feature_count"):
        _register(
            service,
            make_spec(temporal_digest=temporal_digest, feature_count=5),
        )


def test_code_hash_can_be_pinned_at_registration(service, temporal_digest):
    spec = make_spec(temporal_digest=temporal_digest, code_hash="p" * 64, config_hash="q" * 64)
    exp = _register(service, spec)
    assert exp.code_hash == "p" * 64


# ----------------------------------------------------------------- lifecycle


def test_full_lifecycle_to_promoted(service, temporal_digest):
    _register(service, make_spec(temporal_digest=temporal_digest))
    running = service.mark_running("EXP-00001", code_hash="c" * 64, config_hash="g" * 64)
    assert running.status == ExperimentStatus.RUNNING
    completed = service.complete("EXP-00001")
    assert completed.status == ExperimentStatus.COMPLETED
    service.record_result(
        "EXP-00001", kind=ResultKind.SUPPORTED, summary="OOS IC 0.041, after-cost excess 4.2%"
    )
    promoted = service.record_decision(
        "EXP-00001",
        decision=Decision.PROMOTED,
        reason="OOS rank IC 0.041 exceeds Gate-C threshold 0.03.",
        decision_maker="orbit-research",
        policy_version="Gate-C-v1",
    )
    assert promoted.status == ExperimentStatus.PROMOTED
    assert service.decisions("EXP-00001")[-1]["decision"] == "promoted"
    retired = service.retire("EXP-00001")
    assert retired.status == ExperimentStatus.RETIRED


def test_mark_running_requires_code_identity(service, temporal_digest):
    _register(service, make_spec(temporal_digest=temporal_digest))
    with pytest.raises(ValueError, match="code_hash"):
        service.mark_running("EXP-00001", config_hash="g" * 64)
    with pytest.raises(ValueError, match="config_hash"):
        service.mark_running("EXP-00001", code_hash="c" * 64)


def test_code_hash_is_immutable_once_set(service, temporal_digest):
    # pin the identity at registration, then try to run with a different one
    _register(
        service,
        make_spec(temporal_digest=temporal_digest, code_hash="c" * 64, config_hash="g" * 64),
    )
    with pytest.raises(ValueError, match="cannot be changed"):
        service.mark_running("EXP-00001", code_hash="e" * 64, config_hash="g" * 64)
    with pytest.raises(ValueError, match="cannot be changed"):
        service.mark_running("EXP-00001", code_hash="c" * 64, config_hash="f" * 64)
    # matching hashes still enter RUNNING
    running = service.mark_running("EXP-00001", code_hash="c" * 64, config_hash="g" * 64)
    assert running.code_hash == "c" * 64
    assert running.config_hash == "g" * 64


def test_invalid_transition_registered_to_completed(service, temporal_digest):
    _register(service, make_spec(temporal_digest=temporal_digest))
    with pytest.raises(ValueError, match="invalid experiment transition"):
        service.complete("EXP-00001")


def test_invalid_transition_running_to_rejected(service, temporal_digest):
    _register(service, make_spec(temporal_digest=temporal_digest))
    service.mark_running("EXP-00001", code_hash="c" * 64, config_hash="g" * 64)
    with pytest.raises(ValueError, match="requires status COMPLETED"):
        service.record_decision(
            "EXP-00001", decision=Decision.REJECTED,
            reason="not a real decision path",
            decision_maker="orbit-research",
        )


def test_invalid_transition_completed_to_running(service, temporal_digest):
    _register(service, make_spec(temporal_digest=temporal_digest))
    _run(service, "EXP-00001")
    with pytest.raises(ValueError, match="invalid experiment transition"):
        service.mark_running("EXP-00001", code_hash="c" * 64, config_hash="g" * 64)


def test_rejected_cannot_be_reversed_or_promoted(service, temporal_digest):
    _register(service, make_spec(temporal_digest=temporal_digest))
    _run(service, "EXP-00001")
    service.record_result("EXP-00001", kind=ResultKind.NULL, summary="no economic evidence")
    service.record_decision(
        "EXP-00001", decision=Decision.REJECTED,
        reason="Failed after-cost OOS threshold.",
        decision_maker="orbit-research",
    )
    with pytest.raises(ValueError, match="decision state"):
        service.transition("EXP-00001", ExperimentStatus.PROMOTED)
    with pytest.raises(ValueError, match="invalid experiment transition"):
        service.transition("EXP-00001", ExperimentStatus.COMPLETED)


def test_retired_is_terminal(service, temporal_digest):
    _register(service, make_spec(temporal_digest=temporal_digest))
    service.retire("EXP-00001")
    with pytest.raises(ValueError, match="invalid experiment transition"):
        service.mark_running("EXP-00001", code_hash="c" * 64, config_hash="g" * 64)


def test_decision_states_require_record_decision(service, temporal_digest):
    _register(service, make_spec(temporal_digest=temporal_digest))
    _run(service, "EXP-00001")
    with pytest.raises(ValueError, match="record_decision"):
        service.transition("EXP-00001", ExperimentStatus.REJECTED)


def test_running_to_failed_and_failed_to_retired(service, temporal_digest):
    _register(service, make_spec(temporal_digest=temporal_digest))
    service.mark_running("EXP-00001", code_hash="c" * 64, config_hash="g" * 64)
    failed = service.fail("EXP-00001", note="provider outage")
    assert failed.status == ExperimentStatus.FAILED
    assert service.transitions("EXP-00001")[-1]["note"] == "provider outage"
    retired = service.retire("EXP-00001")
    assert retired.status == ExperimentStatus.RETIRED


def test_unknown_experiment_transition_fails_loudly(service):
    with pytest.raises(ValueError, match="unknown experiment"):
        service.complete("EXP-99999")


def test_transition_records_history(service, temporal_digest):
    _register(service, make_spec(temporal_digest=temporal_digest))
    _run(service, "EXP-00001")
    log = service.transitions("EXP-00001")
    assert [(t["from_status"], t["to_status"]) for t in log] == [
        ("registered", "running"),
        ("running", "completed"),
    ]


# ---------------------------------------------------------------- genealogy


def test_root_experiment_has_null_parent(service, temporal_digest):
    exp = _register(service, make_spec(temporal_digest=temporal_digest))
    assert exp.parent_id is None
    assert service.ancestry("EXP-00001") == []


def test_parent_child_and_multilevel_ancestry(service, temporal_digest):
    _register(service, make_spec("EXP-00001", temporal_digest=temporal_digest))
    _register(
        service,
        make_spec("EXP-00002", temporal_digest=temporal_digest, parent_id="EXP-00001"),
    )
    _register(
        service,
        make_spec("EXP-00003", temporal_digest=temporal_digest, parent_id="EXP-00002"),
    )
    assert [e.experiment_id for e in service.ancestry("EXP-00003")] == [
        "EXP-00001", "EXP-00002"
    ]
    assert [e.experiment_id for e in service.children("EXP-00001")] == ["EXP-00002"]
    assert [e.experiment_id for e in service.descendants("EXP-00001")] == [
        "EXP-00002", "EXP-00003"
    ]


def test_branching_genealogy_is_preserved(service, temporal_digest):
    _register(service, make_spec("EXP-00001", temporal_digest=temporal_digest))
    _register(
        service,
        make_spec("EXP-00002", temporal_digest=temporal_digest, parent_id="EXP-00001", title="branch A"),
    )
    _register(
        service,
        make_spec("EXP-00003", temporal_digest=temporal_digest, parent_id="EXP-00001", title="branch B"),
    )
    _register(
        service,
        make_spec("EXP-00004", temporal_digest=temporal_digest, parent_id="EXP-00002"),
    )
    descendants = {e.experiment_id for e in service.descendants("EXP-00001")}
    assert descendants == {"EXP-00002", "EXP-00003", "EXP-00004"}
    assert {e.experiment_id for e in service.children("EXP-00001")} == {
        "EXP-00002", "EXP-00003"
    }
    # the losing branch is NOT collapsed into the winning branch
    assert service.children("EXP-00003") == []


def test_nonexistent_parent_rejected(service, temporal_digest):
    with pytest.raises(ValueError, match="unknown parent experiment"):
        _register(
            service,
            make_spec(temporal_digest=temporal_digest, parent_id="EXP-99999"),
        )


def test_self_parenting_rejected(service, temporal_digest):
    with pytest.raises(ValidationError, match="own parent"):
        make_spec(temporal_digest=temporal_digest, parent_id="EXP-00001")


def test_retired_parent_cannot_take_children(service, temporal_digest):
    _register(service, make_spec("EXP-00001", temporal_digest=temporal_digest))
    service.retire("EXP-00001")
    with pytest.raises(ValueError, match="parent"):
        _register(
            service,
            make_spec("EXP-00002", temporal_digest=temporal_digest, parent_id="EXP-00001"),
        )


def test_genealogy_is_hypothesis_scoped(service, temporal_digest):
    _register(service, make_spec("EXP-00001", temporal_digest=temporal_digest))
    with pytest.raises(ValueError, match="genealogy is hypothesis-scoped"):
        _register(
            service,
            make_spec("EXP-00002", hypothesis_id="H-003", temporal_digest=temporal_digest, parent_id="EXP-00001"),
        )


def test_circular_ancestry_is_impossible_via_registration(service, temporal_digest):
    # The graph grows only downward: a parent must already exist, so a new
    # node can never close a cycle. Duplicate ids are refused, so a "cycle
    # builder" attempt is rejected at registration.
    _register(service, make_spec("EXP-00001", temporal_digest=temporal_digest))
    _register(
        service,
        make_spec("EXP-00002", temporal_digest=temporal_digest, parent_id="EXP-00001"),
    )
    with pytest.raises(ValueError, match="duplicate experiment id"):
        _register(
            service,
            make_spec("EXP-00001", temporal_digest=temporal_digest, parent_id="EXP-00002"),
        )


def test_genealogy_is_immutable_after_execution(service, temporal_digest):
    _register(service, make_spec("EXP-00001", temporal_digest=temporal_digest))
    _run(service, "EXP-00001")
    exp = service.get("EXP-00001")
    assert exp.parent_id is None
    # there is no API to rewrite lineage; the stored spec is frozen
    with pytest.raises(ValidationError):
        exp.parent_id = "EXP-00002"  # type: ignore[misc]


# ------------------------------------------------------------------- lineage


def test_dataset_lineage_preserved(service, temporal_digest):
    spec = make_spec(
        temporal_digest=temporal_digest,
        dataset_snapshot_ids=["DS-000001", "DS-000002"],
    )
    _register(service, spec)
    _run(service, "EXP-00001")
    exp = service.get("EXP-00001")
    assert sorted(exp.dataset_snapshot_ids) == ["DS-000001", "DS-000002"]


def test_exact_dataset_snapshots_are_not_names(service, temporal_digest):
    _register(service, make_spec(temporal_digest=temporal_digest))
    exp = service.get("EXP-00001")
    assert exp.datasets == ["market_daily_v1"]  # descriptive only
    assert exp.dataset_snapshot_ids == ["DS-000001"]  # exact identity


def test_feature_and_model_lineage_preserved(service, temporal_digest):
    spec = make_spec(
        temporal_digest=temporal_digest,
        features={
            "feature_names": ["ret_12m_1m"],
            "feature_version": "v1",
            "feature_refs": [
                {"feature_id": "FEAT-001", "feature_version": "v1", "transformation": "xform-v1"},
                {"feature_id": "FEAT-002", "feature_version": "v2", "transformation": "xform-v2"},
            ],
        },
        model={"family": "xgboost", "hyperparameters": {"eta": 0.05, "depth": 4}},
    )
    _register(service, spec)
    _run(service, "EXP-00001")
    exp = service.get("EXP-00001")
    assert [(f.feature_id, f.feature_version) for f in exp.features.feature_refs] == [
        ("FEAT-001", "v1"),
        ("FEAT-002", "v2"),
    ]
    assert exp.model.family == "xgboost"
    assert exp.model.hyperparameters == {"eta": 0.05, "depth": 4}


def test_windows_and_cost_model_preserved(service, temporal_digest):
    _register(service, make_spec(temporal_digest=temporal_digest))
    _run(service, "EXP-00001")
    exp = service.get("EXP-00001")
    assert str(exp.windows.train_start) == "2015-01-01"
    assert str(exp.windows.test_end) == "2022-01-01"
    assert exp.cost_model.total_bps() == 5.0
    assert exp.cost_model_id == "CM-001"


def test_two_experiments_with_different_costs_are_distinguishable(service, temporal_digest):
    _register(service, make_spec("EXP-00001", temporal_digest=temporal_digest, cost_model_id="CM-001"))
    _register(
        service,
        make_spec("EXP-00002", temporal_digest=temporal_digest, cost_model_id="CM-002"),
    )
    assert service.get("EXP-00001").content_hash() != service.get("EXP-00002").content_hash()


def test_two_xgboost_models_with_different_params_are_distinguishable(service, temporal_digest):
    _register(service, make_spec("EXP-00001", temporal_digest=temporal_digest, model={"family": "xgboost", "hyperparameters": {"eta": 0.05}}))
    _register(service, make_spec("EXP-00002", temporal_digest=temporal_digest, model={"family": "xgboost", "hyperparameters": {"eta": 0.5}}))
    assert service.get("EXP-00001").content_hash() != service.get("EXP-00002").content_hash()


def test_temporal_lineage_preserved(service, temporal_digest):
    _register(service, make_spec(temporal_digest=temporal_digest))
    _run(service, "EXP-00001")
    t = service.get("EXP-00001").temporal_config
    assert t.engine_version == "v1.0.0"
    assert t.config_digest == temporal_digest
    assert t.as_of_semantics == "strict_publication_less_than_asof"


# --------------------------------------------------------------- immutability


def test_scientific_identity_is_frozen(service, temporal_digest):
    _register(service, make_spec(temporal_digest=temporal_digest))
    _run(service, "EXP-00001")
    exp = service.get("EXP-00001")
    with pytest.raises(ValidationError):
        exp.model = exp.model.model_copy(update={"family": "ridge"})  # type: ignore[misc]


def test_content_hash_stable_across_lifecycle(service, temporal_digest):
    exp = _register(service, make_spec(temporal_digest=temporal_digest))
    before = exp.content_hash()
    _run(service, "EXP-00001")
    after = service.get("EXP-00001")
    assert after.content_hash() == before
    assert after.status == ExperimentStatus.COMPLETED


def test_identity_changes_require_a_new_experiment(service, temporal_digest):
    exp = _register(service, make_spec("EXP-00001", temporal_digest=temporal_digest))
    _run(service, "EXP-00001")
    child = _register(
        service,
        make_spec(
            "EXP-00002",
            temporal_digest=temporal_digest,
            parent_id="EXP-00001",
            model={"family": "ridge"},
        ),
    )
    assert exp.content_hash() != child.content_hash()
    assert child.parent_id == "EXP-00001"


# ------------------------------------------------------ search and filtering


def test_list_filters_by_status_and_hypothesis(service, temporal_digest):
    _register(service, make_spec("EXP-00001", temporal_digest=temporal_digest))
    _register(service, make_spec("EXP-00002", hypothesis_id="H-003", temporal_digest=temporal_digest))
    _run(service, "EXP-00001")
    assert {e.experiment_id for e in service.list(status="completed")} == {"EXP-00001"}
    assert {e.experiment_id for e in service.list(hypothesis_id="H-003")} == {"EXP-00002"}


def test_list_filters_by_family_label_dataset_feature_epoch(service, temporal_digest):
    _register(
        service,
        make_spec("EXP-00001", temporal_digest=temporal_digest, research_epoch="EPOCH-001", hypothesis_family="momentum"),
    )
    _register(
        service,
        make_spec(
            "EXP-00002",
            hypothesis_id="H-003",
            temporal_digest=temporal_digest,
            research_epoch="EPOCH-002",
            hypothesis_family="pead",
            dataset_snapshot_ids=["DS-000002"],
            features={
                "feature_names": ["sue"],
                "feature_version": "v1",
                "feature_refs": [{"feature_id": "FEAT-009", "feature_version": "v1"}],
            },
            label_id="LAB-003",
        ),
    )
    assert [e.experiment_id for e in service.list(hypothesis_family="momentum")] == ["EXP-00001"]
    assert [e.experiment_id for e in service.list(label_id="LAB-003")] == ["EXP-00002"]
    assert [e.experiment_id for e in service.list(dataset_snapshot_id="DS-000002")] == ["EXP-00002"]
    assert [e.experiment_id for e in service.list(feature_id="FEAT-009")] == ["EXP-00002"]
    assert [e.experiment_id for e in service.list(research_epoch="EPOCH-001")] == ["EXP-00001"]
    assert [e.experiment_id for e in service.list(parent_id="EXP-00001")] == []


def test_list_default_shows_failed_and_rejected_experiments(service, temporal_digest):
    _register(service, make_spec("EXP-00001", temporal_digest=temporal_digest))
    _register(service, make_spec("EXP-00002", temporal_digest=temporal_digest))
    service.fail("EXP-00001")
    _run(service, "EXP-00002")
    service.record_result("EXP-00002", kind=ResultKind.NULL, summary="no evidence")
    service.record_decision(
        "EXP-00002", decision=Decision.REJECTED,
        reason="Below Gate-C OOS threshold.", decision_maker="orbit-research",
    )
    all_ids = {e.experiment_id for e in service.list()}
    assert all_ids == {"EXP-00001", "EXP-00002"}


# ------------------------------------------------------------- null/failed


def test_null_result_experiment_remains_registered_and_queryable(service, temporal_digest):
    _register(service, make_spec(temporal_digest=temporal_digest))
    _run(service, "EXP-00001")
    service.record_result(
        "EXP-00001",
        kind=ResultKind.NULL,
        summary="NO SIGNIFICANT / NO ECONOMIC EVIDENCE",
        metrics={"oos_rank_ic": 0.004},
    )
    service.record_decision(
        "EXP-00001",
        decision=Decision.REJECTED,
        reason="Failed after-cost OOS threshold.",
        policy_version="Gate-C-v1",
        decision_maker="orbit-research",
    )
    result = service.result("EXP-00001")
    assert result["kind"] == "null"
    assert result["summary"] == "NO SIGNIFICANT / NO ECONOMIC EVIDENCE"
    exp = service.get("EXP-00001")
    assert exp.status == ExperimentStatus.REJECTED
    assert service.list(status="rejected") == [exp]


def test_failed_experiment_retains_full_history(service, temporal_digest):
    _register(service, make_spec(temporal_digest=temporal_digest))
    service.mark_running("EXP-00001", code_hash="c" * 64, config_hash="g" * 64)
    service.fail("EXP-00001", note="infrastructure failure")
    service.record_result(
        "EXP-00001",
        kind=ResultKind.INFRASTRUCTURE_FAILURE,
        summary="node lost mid-run; no outputs produced",
        recorded_by="orbit-ops",
    )
    assert service.get("EXP-00001").status == ExperimentStatus.FAILED
    assert service.result("EXP-00001")["kind"] == "infrastructure_failure"
    assert len(service.transitions("EXP-00001")) == 2


# ---------------------------------------------------------------- artifacts


def test_artifact_attached_and_associated(service, temporal_digest):
    _register(service, make_spec(temporal_digest=temporal_digest))
    _run(service, "EXP-00001")
    aid = service.attach_artifact(
        "EXP-00001", kind="metrics", path="artifacts/EXP-00001/metrics.json", checksum="e" * 64
    )
    assert aid.startswith("ART-")
    arts = service.artifacts("EXP-00001")
    assert len(arts) == 1
    assert arts[0]["path"] == "artifacts/EXP-00001/metrics.json"
    assert arts[0]["checksum"] == "e" * 64


def test_orphan_artifact_is_refused(service):
    with pytest.raises(ValueError, match="unknown experiment"):
        service.attach_artifact("EXP-99999", kind="metrics", path="x.json")


def test_duplicate_artifact_path_refused(service, temporal_digest):
    _register(service, make_spec(temporal_digest=temporal_digest))
    service.attach_artifact("EXP-00001", kind="metrics", path="m.json")
    with pytest.raises(ValueError, match="constraint violation"):
        service.attach_artifact("EXP-00001", kind="plots", path="m.json")


# ----------------------------------------------------------------- decisions


def test_decision_requires_completed_experiment(service, temporal_digest):
    _register(service, make_spec(temporal_digest=temporal_digest))
    with pytest.raises(ValueError, match="requires status COMPLETED"):
        service.record_decision(
            "EXP-00001", decision=Decision.REJECTED, reason="too early for a decision",
            decision_maker="orbit-research",
        )


def test_decision_reason_must_be_substantive(service, temporal_digest):
    _register(service, make_spec(temporal_digest=temporal_digest))
    _run(service, "EXP-00001")
    with pytest.raises(ValueError, match="substantive"):
        service.record_decision(
            "EXP-00001", decision=Decision.REJECTED, reason="nope",
            decision_maker="orbit-research",
        )
    with pytest.raises(ValueError, match="placeholder"):
        service.record_decision(
            "EXP-00001", decision=Decision.REJECTED, reason="We didn't like it",
            decision_maker="orbit-research",
        )
    with pytest.raises(ValueError, match="decision_maker"):
        service.record_decision(
            "EXP-00001", decision=Decision.REJECTED, reason="Failed after-cost OOS threshold.", decision_maker=""
        )


def test_decision_is_recorded_with_full_context(service, temporal_digest):
    _register(service, make_spec(temporal_digest=temporal_digest))
    _run(service, "EXP-00001")
    service.record_result("EXP-00001", kind=ResultKind.SUPPORTED, summary="strong OOS evidence")
    service.record_decision(
        "EXP-00001",
        decision=Decision.PROMOTED,
        reason="After-cost annual excess 4.1% exceeds Gate-C threshold 3%.",
        decision_maker="orbit-research",
        policy_version="Gate-C-v1",
        decided_at=_at(2026, 3, 1),
    )
    d = service.decisions("EXP-00001")[-1]
    assert d["decision"] == "promoted"
    assert d["policy_version"] == "Gate-C-v1"
    assert d["decision_maker"] == "orbit-research"
    assert d["decided_at"] == _at(2026, 3, 1).replace(tzinfo=None)  # DuckDB stores naive UTC
    assert service.get("EXP-00001").status == ExperimentStatus.PROMOTED


def test_second_decision_after_rejection_is_refused(service, temporal_digest):
    _register(service, make_spec(temporal_digest=temporal_digest))
    _run(service, "EXP-00001")
    service.record_result("EXP-00001", kind=ResultKind.NULL, summary="no evidence")
    service.record_decision(
        "EXP-00001", decision=Decision.REJECTED,
        reason="Failed after-cost OOS threshold.", decision_maker="orbit-research",
    )
    with pytest.raises(ValueError, match="requires status COMPLETED"):
        service.record_decision(
            "EXP-00001", decision=Decision.PROMOTED, reason="changed our minds",
            decision_maker="orbit-research",
        )


# ------------------------------------------------------------------ results


def test_result_requires_completed_or_failed(service, temporal_digest):
    _register(service, make_spec(temporal_digest=temporal_digest))
    with pytest.raises(ValueError, match="COMPLETED or FAILED"):
        service.record_result("EXP-00001", kind=ResultKind.NULL, summary="x")


def test_result_is_immutable_one_per_experiment(service, temporal_digest):
    _register(service, make_spec(temporal_digest=temporal_digest))
    _run(service, "EXP-00001")
    service.record_result("EXP-00001", kind=ResultKind.NULL, summary="no evidence")
    with pytest.raises(ValueError, match="constraint violation"):
        service.record_result(
            "EXP-00001", kind=ResultKind.SUPPORTED, summary="rewritten result"
        )
    assert service.result("EXP-00001")["summary"] == "no evidence"


def test_result_metrics_are_preserved(service, temporal_digest):
    _register(service, make_spec(temporal_digest=temporal_digest))
    _run(service, "EXP-00001")
    service.record_result(
        "EXP-00001",
        kind=ResultKind.NEGATIVE,
        summary="effect reverses OOS",
        metrics={"oos_rank_ic": -0.02, "walkforward_windows": 4},
    )
    result = service.result("EXP-00001")
    assert result["kind"] == "negative"
    assert "oos_rank_ic" in result["metrics_json"]


def test_result_kind_validation(service, temporal_digest):
    _register(service, make_spec(temporal_digest=temporal_digest))
    _run(service, "EXP-00001")
    with pytest.raises(ValueError, match="invalid result kind"):
        service.record_result("EXP-00001", kind="amazing", summary="x")