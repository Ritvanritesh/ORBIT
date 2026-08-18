"""Phase 6 concurrency tests (section 34).

Every thread opens its OWN DuckDB connection to the same file (the intended
production layout: one writer per researcher process). DuckDB serializes
writers, so these tests prove the registry's retry-on-lock-contention logic
and the optimistic WHERE-guarded transitions, atomic trial counters and
PK/FK-unique constraints turn races into exactly-one-winner outcomes.
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone

import pytest

from conftest import make_spec

from orbit.experiments import ExperimentRegistry, ExperimentService, ResultKind


def _at(y, m, d):
    return datetime(y, m, d, tzinfo=timezone.utc)


def _errors(outcomes):
    return [o for o in outcomes if o[1] is not None]


def test_concurrent_registrations_all_succeed_with_unique_trial_numbers(
    tmp_path, hypotheses, labels, temporal, datasets, temporal_digest
):
    n = 6
    barrier = threading.Barrier(n)
    results: list[tuple[str, Exception | None]] = []

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
            results.append((f"EXP-{i:05d}", None))
        except Exception as exc:  # noqa: BLE001 - concurrency probe
            results.append((f"EXP-{i:05d}", exc))
        finally:
            svc._registry.close()

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert _errors(results) == [], [str(e) for _, e in _errors(results)]

    # trial numbers are the atomically-assigned ordinals 1..n, no duplicates
    con = ExperimentRegistry(db_path=str(tmp_path / "experiments.duckdb"))
    numbers = sorted(int(r["trial_number"]) for r in con.dump())
    con.close()
    assert numbers == list(range(1, n + 1))
    assert len(set(numbers)) == n


def test_concurrent_duplicate_registration_has_exactly_one_winner(
    tmp_path, hypotheses, labels, temporal, datasets, temporal_digest
):
    barrier = threading.Barrier(2)
    outcomes = []

    def worker():
        svc = ExperimentService(
            registry=ExperimentRegistry(db_path=str(tmp_path / "experiments.duckdb")),
            hypothesis_registry=hypotheses, label_registry=labels, temporal_contract=temporal,
            dataset_registry=datasets,
        )
        barrier.wait()
        try:
            svc.register(make_spec(temporal_digest=temporal_digest), registered_at=_at(2026, 1, 1))
            outcomes.append("ok")
        except Exception:  # noqa: BLE001 - concurrency probe
            outcomes.append("failed")
        finally:
            svc._registry.close()

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(outcomes) == ["failed", "ok"]


def test_concurrent_mark_running_has_exactly_one_winner(
    tmp_path, hypotheses, labels, temporal, datasets, temporal_digest
):
    svc = ExperimentService(
        registry=ExperimentRegistry(db_path=str(tmp_path / "experiments.duckdb")),
        hypothesis_registry=hypotheses, label_registry=labels, temporal_contract=temporal,
        dataset_registry=datasets,
    )
    svc.register(make_spec(temporal_digest=temporal_digest), registered_at=_at(2026, 1, 1))
    svc._registry.close()

    barrier = threading.Barrier(2)
    outcomes = []

    def worker():
        s = ExperimentService(
            registry=ExperimentRegistry(db_path=str(tmp_path / "experiments.duckdb")),
            hypothesis_registry=hypotheses, label_registry=labels, temporal_contract=temporal,
            dataset_registry=datasets,
        )
        barrier.wait()
        try:
            s.mark_running("EXP-00001", code_hash="a" * 64, config_hash="b" * 64)
            outcomes.append("ok")
        except Exception:  # noqa: BLE001 - concurrency probe
            outcomes.append("failed")
        finally:
            s._registry.close()

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(outcomes) == ["failed", "ok"]
    # the winner's state is what the ledger records
    check = ExperimentRegistry(db_path=str(tmp_path / "experiments.duckdb"))
    assert check.get("EXP-00001")["status"] == "running"
    check.close()


def test_concurrent_result_recording_has_exactly_one_winner(
    tmp_path, hypotheses, labels, temporal, datasets, temporal_digest
):
    svc = ExperimentService(
        registry=ExperimentRegistry(db_path=str(tmp_path / "experiments.duckdb")),
        hypothesis_registry=hypotheses, label_registry=labels, temporal_contract=temporal,
        dataset_registry=datasets,
    )
    svc.register(make_spec(temporal_digest=temporal_digest), registered_at=_at(2026, 1, 1))
    svc.mark_running("EXP-00001", code_hash="a" * 64, config_hash="b" * 64)
    svc.complete("EXP-00001")
    svc._registry.close()

    barrier = threading.Barrier(2)
    outcomes = []

    def worker():
        s = ExperimentService(
            registry=ExperimentRegistry(db_path=str(tmp_path / "experiments.duckdb")),
            hypothesis_registry=hypotheses, label_registry=labels, temporal_contract=temporal,
            dataset_registry=datasets,
        )
        barrier.wait()
        try:
            s.record_result("EXP-00001", kind=ResultKind.SUPPORTED, summary="IC 0.03, 2.1% excess")
            outcomes.append("ok")
        except Exception:  # noqa: BLE001 - concurrency probe
            outcomes.append("failed")
        finally:
            s._registry.close()

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(outcomes) == ["failed", "ok"]


def test_concurrent_decisions_do_not_race_the_state_machine(
    tmp_path, hypotheses, labels, temporal, datasets, temporal_digest
):
    svc = ExperimentService(
        registry=ExperimentRegistry(db_path=str(tmp_path / "experiments.duckdb")),
        hypothesis_registry=hypotheses, label_registry=labels, temporal_contract=temporal,
        dataset_registry=datasets,
    )
    svc.register(make_spec(temporal_digest=temporal_digest), registered_at=_at(2026, 1, 1))
    svc.mark_running("EXP-00001", code_hash="a" * 64, config_hash="b" * 64)
    svc.complete("EXP-00001")
    svc.record_result("EXP-00001", kind=ResultKind.NULL, summary="NO SIGNIFICANT / NO ECONOMIC EVIDENCE")
    svc._registry.close()

    barrier = threading.Barrier(2)
    outcomes = []

    def worker():
        s = ExperimentService(
            registry=ExperimentRegistry(db_path=str(tmp_path / "experiments.duckdb")),
            hypothesis_registry=hypotheses, label_registry=labels, temporal_contract=temporal,
            dataset_registry=datasets,
        )
        barrier.wait()
        try:
            s.record_decision(
                "EXP-00001", decision="rejected",
                reason="OOS IC 0.005 below the 0.03 Gate-C threshold.",
                policy_version="Gate-C-v1", decision_maker="orbit-research",
            )
            outcomes.append("ok")
        except Exception:  # noqa: BLE001 - concurrency probe
            outcomes.append("failed")
        finally:
            s._registry.close()

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(outcomes) == ["failed", "ok"]