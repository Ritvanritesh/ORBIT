"""Phase 6 second independent audit (section 39).

This file deliberately does NOT reuse the Phase 6 authoring fixtures' happy
paths: it pokes the ledger with raw SQL and adversarial call sequences that a
fresh reviewer would try, and verifies the database itself refuses every
mutation that would break the immutability/lineage contract.
"""

from __future__ import annotations

from datetime import datetime, timezone

import duckdb
import pytest

from conftest import make_spec

from orbit.experiments import Decision, ExperimentService, ResultKind
from orbit.experiments.registry import ExperimentRegistry


def _at(y, m, d):
    return datetime(y, m, d, tzinfo=timezone.utc)


@pytest.fixture
def audited_service(tmp_path, hypotheses, labels, temporal, datasets, temporal_digest):
    svc = ExperimentService(
        registry=ExperimentRegistry(db_path=str(tmp_path / "experiments.duckdb")),
        hypothesis_registry=hypotheses, label_registry=labels, temporal_contract=temporal,
        dataset_registry=datasets,
    )
    svc.register(
        make_spec(
            temporal_digest=temporal_digest, research_epoch="EPOCH-001",
            selection_stage="candidate",
        ),
        registered_at=_at(2026, 1, 1),
    )
    svc.mark_running("EXP-00001", code_hash="a" * 64, config_hash="b" * 64)
    svc.complete("EXP-00001")
    svc.record_result("EXP-00001", kind=ResultKind.SUPPORTED, summary="IC 0.041, 4.2% excess")
    svc.record_decision(
        "EXP-00001", decision=Decision.PROMOTED,
        reason="After-cost annual excess 4.2% exceeds Gate-C threshold 3%.",
        policy_version="Gate-C-v1", decision_maker="orbit-research",
    )
    return svc


def test_ledger_has_no_delete_paths(audited_service, tmp_path):
    # the public API exposes no destructive operations at all
    for name in ("delete", "drop", "remove", "purge", "reset", "wipe"):
        assert not hasattr(ExperimentService, name), name
        assert not hasattr(ExperimentRegistry, name), name
    # and raw SQL cannot drop the table the whole ledger hangs off
    con = duckdb.connect(str(tmp_path / "experiments.duckdb"))
    with pytest.raises((duckdb.ConstraintException, duckdb.CatalogException)):
        con.execute("DROP TABLE experiments")
    con.close()


def test_content_hash_cannot_be_recomputed_away(audited_service, tmp_path):
    con = duckdb.connect(str(tmp_path / "experiments.duckdb"))
    with pytest.raises(duckdb.ConstraintException):
        con.execute("UPDATE experiments SET content_hash = ? WHERE experiment_id = 'EXP-00001'",
                    ["0" * 64])
    con.close()


def test_status_check_constraint_rejects_garbage(audited_service, tmp_path):
    con = duckdb.connect(str(tmp_path / "experiments.duckdb"))
    with pytest.raises(duckdb.ConstraintException):
        con.execute(
            "UPDATE experiment_state SET status = 'deleted' WHERE experiment_id = 'EXP-00001'"
        )
    con.close()


def test_self_parenting_is_impossible(audited_service, tmp_path):
    con = duckdb.connect(str(tmp_path / "experiments.duckdb"))
    with pytest.raises(duckdb.ConstraintException):
        con.execute(
            "UPDATE experiments SET parent_id = 'EXP-00001' WHERE experiment_id = 'EXP-00001'"
        )
    con.close()


def test_foreign_keys_are_enforced_even_under_raw_sql(audited_service, tmp_path):
    con = duckdb.connect(str(tmp_path / "experiments.duckdb"))
    with pytest.raises(duckdb.ConstraintException):
        con.execute(
            "INSERT INTO artifacts VALUES "
            "('ART-99999', 'EXP-NOPE', 'csv', 'x.csv', NULL, '2026-01-01T00:00:00Z')"
        )
    con.close()


def test_second_result_for_one_experiment_is_impossible(audited_service, tmp_path):
    con = duckdb.connect(str(tmp_path / "experiments.duckdb"))
    with pytest.raises(duckdb.ConstraintException):
        con.execute(
            "INSERT INTO results VALUES "
            "('R-2', 'EXP-00001', 'negative', 'forged', NULL, "
            "'2026-01-02T00:00:00Z', 'attacker')"
        )
    con.close()


def test_code_and_config_identity_is_immutable_once_set(audited_service, temporal_digest):
    svc = audited_service
    # pin the identity at registration, then try to run with a different one:
    # the hash check fires on the (valid) registered->running transition
    svc.register(
        make_spec("EXP-00002", temporal_digest=temporal_digest, code_hash="a" * 64, config_hash="b" * 64),
        registered_at=_at(2026, 1, 2),
    )
    with pytest.raises(ValueError, match="cannot be changed"):
        svc.mark_running("EXP-00002", code_hash="f" * 64, config_hash="b" * 64)
    with pytest.raises(ValueError, match="cannot be changed"):
        svc.mark_running("EXP-00002", code_hash="a" * 64, config_hash="f" * 64)
    assert svc.get("EXP-00002").code_hash == "a" * 64
    assert svc.get("EXP-00002").config_hash == "b" * 64


def test_no_experiment_can_be_hidden(audited_service, temporal_digest):
    # every registered experiment is visible through every list view,
    # including failures and nulls
    svc = audited_service
    svc.register(make_spec("EXP-00002", temporal_digest=temporal_digest), registered_at=_at(2026, 1, 2))
    svc.mark_running("EXP-00002", code_hash="a" * 64, config_hash="b" * 64)
    svc.fail("EXP-00002", note="OOM")
    assert [e.experiment_id for e in svc.list()] == ["EXP-00001", "EXP-00002"]
    assert svc.list(status="failed")[0].experiment_id == "EXP-00002"
    assert svc.count_trials("H-001") == 2


def test_trial_number_cannot_be_redeclared(audited_service, temporal_digest):
    svc = audited_service
    with pytest.raises(ValueError, match="trial_number is computed"):
        svc.register(
            make_spec("EXP-00002", temporal_digest=temporal_digest, trial_number=99),
            registered_at=_at(2026, 1, 2),
        )


def test_draft_hypotheses_cannot_enter_the_ledger(tmp_path, labels, temporal, datasets, temporal_digest):
    from hypotheses.seeds import build_seed_registry

    drafts = build_seed_registry()  # every seed starts as a draft
    svc = ExperimentService(
        registry=ExperimentRegistry(db_path=str(tmp_path / "experiments.duckdb")),
        hypothesis_registry=drafts, label_registry=labels, temporal_contract=temporal,
        dataset_registry=datasets,
    )
    with pytest.raises(ValueError, match="registered hypotheses"):
        svc.register(
            make_spec(temporal_digest=temporal_digest),
            registered_at=_at(2026, 1, 1),
        )


def test_unknown_parent_is_refused(audited_service, temporal_digest):
    svc = audited_service
    with pytest.raises(ValueError, match="unknown parent"):
        svc.register(
            make_spec("EXP-00002", temporal_digest=temporal_digest, parent_id="EXP-99999"),
            registered_at=_at(2026, 1, 2),
        )


def test_promoted_experiment_still_walks_its_ancestry(audited_service, temporal_digest):
    svc = audited_service
    svc.register(
        make_spec("EXP-00002", temporal_digest=temporal_digest, parent_id="EXP-00001"),
        registered_at=_at(2026, 1, 2),
    )
    svc.mark_running("EXP-00002", code_hash="a" * 64, config_hash="b" * 64)
    svc.complete("EXP-00002")
    svc.record_result("EXP-00002", kind=ResultKind.NEGATIVE, summary="effect reverses OOS")
    svc.record_decision(
        "EXP-00002", decision=Decision.REJECTED,
        reason="OOS rank IC 0.008 below the 0.03 Gate-C threshold.",
        policy_version="Gate-C-v1", decision_maker="orbit-research",
    )
    chain = svc.ancestry("EXP-00002")
    assert [e.experiment_id for e in chain] == ["EXP-00001"]  # ancestors, excluding self
    report = svc.validate_invariants()
    assert report["ok"], report["violations"]


def test_research_epoch_and_selection_stage_are_searchable(audited_service, temporal_digest):
    svc = audited_service
    svc.register(
        make_spec(
            "EXP-00002", temporal_digest=temporal_digest,
            research_epoch="EPOCH-002", selection_stage="archived",
        ),
        registered_at=_at(2026, 1, 2),
    )
    assert len(svc.list(research_epoch="EPOCH-002")) == 1
    assert len(svc.list(research_epoch="EPOCH-001")) == 1
    assert len(svc.list(selection_stage="candidate")) == 1
    assert len(svc.list(selection_stage="archived")) == 1