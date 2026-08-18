"""Experiment registry: the durable, constraint-enforced research ledger
(Phase 6).

A single DuckDB file records every experiment, its immutable scientific
identity (`spec_json`), its mutable operational state (status, code/config
hashes), and every operational record (transitions, artifacts, results,
decisions). The pattern follows the Phase 3 ingestion registry: one file, one
connection per process, per-write commits, atomic counters for id generation.

Two-table design (DuckDB FK/UPDATE limitation): DuckDB refuses to UPDATE a
column that carries a secondary index when the table is referenced by a
foreign key. The scientific identity table `experiments` is therefore
WRITE-ONCE (INSERT only - the identity never changes), while the mutable
operational state lives in `experiment_state`, which no foreign key
references, so its `status` column can be indexed and updated freely. The
optimistic `WHERE status = ?` guard on that UPDATE is the concurrency check:
a transition racing with another writer fails loudly instead of silently
overwriting it.

Database constraints enforce the research-record invariants that application
code must not be trusted to enforce alone:

  - `experiment_id` is a PRIMARY KEY: no duplicate experiments, even under
    concurrent registration;
  - `parent_id` is a FOREIGN KEY with CHECK (parent_id <> experiment_id):
    no orphan parents, no self-parenting;
  - `experiment_state.status` is CHECK-constrained to the lifecycle enum;
  - `results.experiment_id` is UNIQUE: one immutable result per experiment,
    re-recording is impossible;
  - artifacts / decisions / results / transitions / state / lineage join rows
    are FOREIGN KEY-bound to a real experiment: no orphan research records;
  - DELETE of an experiment with any child record is blocked by the foreign
    keys: retirement (RETIRED status) is the only way an experiment leaves
    active research, and its history stays intact;
  - counters are incremented atomically (`UPDATE ... RETURNING`): concurrent
    registrations cannot collide on ids.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb

from orbit.experiments.lifecycle import validate_transition
from orbit.schemas.common import ExperimentStatus

_SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    experiment_id            VARCHAR PRIMARY KEY,
    hypothesis_id            VARCHAR NOT NULL,
    title                    VARCHAR NOT NULL,
    parent_id                VARCHAR,
    spec_json                VARCHAR NOT NULL,
    content_hash             VARCHAR NOT NULL,
    seed                     BIGINT,
    label_id                 VARCHAR,
    label_version            VARCHAR,
    hypothesis_family        VARCHAR,
    research_epoch           VARCHAR,
    selection_stage          VARCHAR,
    trial_number             BIGINT,
    number_of_prior_trials   BIGINT,
    researcher               VARCHAR NOT NULL,
    created_at               TIMESTAMP NOT NULL,
    registered_at            TIMESTAMP NOT NULL,
    FOREIGN KEY (parent_id) REFERENCES experiments(experiment_id),
    CHECK (experiment_id ~ '^EXP-[0-9]{5}$'),
    CHECK (hypothesis_id ~ '^H-[0-9]{3}$'),
    CHECK (parent_id IS NULL OR parent_id ~ '^EXP-[0-9]{5}$'),
    CHECK (parent_id IS NULL OR parent_id <> experiment_id),
    CHECK (label_id IS NULL OR label_version IS NOT NULL)
);
CREATE INDEX IF NOT EXISTS idx_experiments_hypothesis  ON experiments(hypothesis_id);

-- A secondary index on a column of an FK-referenced table makes DuckDB refuse
-- every UPDATE to that column (ConstraintException), which is exactly the
-- immutability we want: the scientific identity (content_hash, parent_id)
-- of a write-once experiment row can never be rewritten, even via raw SQL.
CREATE INDEX IF NOT EXISTS idx_experiments_parent      ON experiments(parent_id);
CREATE INDEX IF NOT EXISTS idx_experiments_content_hash ON experiments(content_hash);

CREATE INDEX IF NOT EXISTS idx_experiments_label       ON experiments(label_id, label_version);
CREATE INDEX IF NOT EXISTS idx_experiments_epoch       ON experiments(research_epoch);
CREATE INDEX IF NOT EXISTS idx_experiments_trial       ON experiments(trial_number);

CREATE TABLE IF NOT EXISTS experiment_state (
    experiment_id   VARCHAR PRIMARY KEY REFERENCES experiments(experiment_id),
    status          VARCHAR NOT NULL CHECK (status IN
        ('draft','registered','running','completed','failed','rejected','promoted','retired')),
    code_hash       VARCHAR,
    config_hash     VARCHAR,
    updated_at      TIMESTAMP NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_state_status ON experiment_state(status);

CREATE TABLE IF NOT EXISTS experiment_datasets (
    experiment_id       VARCHAR NOT NULL REFERENCES experiments(experiment_id),
    dataset_snapshot_id VARCHAR NOT NULL,
    PRIMARY KEY (experiment_id, dataset_snapshot_id),
    CHECK (dataset_snapshot_id ~ '^DS-[0-9]{6}$')
);
CREATE INDEX IF NOT EXISTS idx_experiment_datasets_snapshot
    ON experiment_datasets(dataset_snapshot_id);

CREATE TABLE IF NOT EXISTS experiment_features (
    experiment_id    VARCHAR NOT NULL REFERENCES experiments(experiment_id),
    feature_id       VARCHAR NOT NULL,
    feature_version  VARCHAR NOT NULL,
    PRIMARY KEY (experiment_id, feature_id),
    CHECK (feature_id ~ '^FEAT-[0-9]{3,}$')
);
CREATE INDEX IF NOT EXISTS idx_experiment_features_id
    ON experiment_features(feature_id);

CREATE TABLE IF NOT EXISTS transitions (
    experiment_id    VARCHAR NOT NULL REFERENCES experiments(experiment_id),
    from_status      VARCHAR NOT NULL CHECK (from_status IN
        ('draft','registered','running','completed','failed','rejected','promoted','retired')),
    to_status        VARCHAR NOT NULL CHECK (to_status IN
        ('draft','registered','running','completed','failed','rejected','promoted','retired')),
    transitioned_at  TIMESTAMP NOT NULL,
    note             VARCHAR,
    content_hash     VARCHAR NOT NULL CHECK (content_hash ~ '^[0-9a-f]{64}$')
);
CREATE INDEX IF NOT EXISTS idx_transitions_experiment ON transitions(experiment_id);

CREATE TABLE IF NOT EXISTS artifacts (
    artifact_id    VARCHAR PRIMARY KEY,
    experiment_id  VARCHAR NOT NULL REFERENCES experiments(experiment_id),
    kind           VARCHAR NOT NULL,
    path           VARCHAR NOT NULL,
    checksum       VARCHAR,
    created_at     TIMESTAMP NOT NULL,
    content_hash   VARCHAR NOT NULL CHECK (content_hash ~ '^[0-9a-f]{64}$'),
    UNIQUE (experiment_id, path)
);
CREATE INDEX IF NOT EXISTS idx_artifacts_experiment ON artifacts(experiment_id);

CREATE TABLE IF NOT EXISTS results (
    result_id     VARCHAR PRIMARY KEY,
    experiment_id VARCHAR NOT NULL UNIQUE REFERENCES experiments(experiment_id),
    kind          VARCHAR NOT NULL CHECK (kind IN
        ('supported','null','negative','invalid','infrastructure_failure')),
    summary       VARCHAR NOT NULL,
    metrics_json  VARCHAR,
    recorded_at   TIMESTAMP NOT NULL,
    recorded_by   VARCHAR NOT NULL,
    content_hash  VARCHAR NOT NULL CHECK (content_hash ~ '^[0-9a-f]{64}$')
);

CREATE TABLE IF NOT EXISTS decisions (
    decision_id     VARCHAR PRIMARY KEY,
    experiment_id   VARCHAR NOT NULL REFERENCES experiments(experiment_id),
    decision        VARCHAR NOT NULL CHECK (decision IN ('rejected','promoted')),
    reason          VARCHAR NOT NULL,
    policy_version  VARCHAR,
    decision_maker  VARCHAR NOT NULL,
    decided_at      TIMESTAMP NOT NULL,
    content_hash    VARCHAR NOT NULL CHECK (content_hash ~ '^[0-9a-f]{64}$')
);
CREATE INDEX IF NOT EXISTS idx_decisions_experiment ON decisions(experiment_id);

CREATE TABLE IF NOT EXISTS counters (kind VARCHAR PRIMARY KEY, value INTEGER NOT NULL);
INSERT OR IGNORE INTO counters (kind, value) VALUES ('experiment', 0), ('artifact', 0), ('decision', 0);

CREATE TABLE IF NOT EXISTS trial_counters (family VARCHAR PRIMARY KEY, value INTEGER NOT NULL);
"""

_VALID_STATUSES = frozenset(
    {"draft", "registered", "running", "completed", "failed", "rejected", "promoted", "retired"}
)

# The ONLY states an experiment may be born into. register() is the only
# entry point of the ledger; a raw-registry user cannot create an experiment
# that already lives in a later lifecycle state with no history behind it
# (the audit trail must explain how it got there).
_BIRTH_STATUSES = frozenset({"draft", "registered"})


def _next_id(con: duckdb.DuckDBPyConnection, kind: str, prefix: str, width: int) -> str:
    con.execute("INSERT OR IGNORE INTO counters (kind, value) VALUES (?, 0)", [kind])
    row = con.execute(
        "UPDATE counters SET value = value + 1 WHERE kind = ? RETURNING value", [kind]
    ).fetchone()
    return f"{prefix}{row[0]:0{width}d}"


def _is_lock_error(exc: BaseException) -> bool:
    if isinstance(exc, duckdb.IOException) or "lock" in str(exc).lower():
        return True
    # concurrent schema creation: catalog write-write conflict
    if isinstance(exc, duckdb.TransactionException):
        msg = str(exc).lower()
        if "write-write conflict" in msg or "conflict on tuple deletion" in msg:
            return True
    return False


class _LockContention(Exception):
    """Raised internally when DuckDB reports a busy/locked database."""


def _retry(fn, attempts: int = 20, base_delay: float = 0.01) -> Any:
    """Retry a write on lock contention (concurrent writers to one file)."""
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - duckdb raises several types
            if not _is_lock_error(exc):
                raise
            if i == attempts - 1:
                raise _LockContention(str(exc)) from exc
            time.sleep(base_delay * (2 ** i))
    raise AssertionError("unreachable")


def _constraint_error(exc: Exception) -> ValueError:
    return ValueError(f"registry constraint violation: {exc}")


def record_content_hash(fields: dict[str, Any]) -> str:
    """sha256 over the canonical content of a child record (result, decision,
    artifact, transition).

    The hash is computed at write time by the registry and stored in the
    record's `content_hash` column; `validate_invariants()` recomputes it over
    the stored row and flags any mismatch. Raw-SQL UPDATEs of child-record
    content (summary, reason, checksum, note, metrics, ...) are therefore
    detected even though DuckDB's index trick does not protect these
    FK-referencing tables.

    Timestamps are canonicalized because the registry receives ISO strings
    while DuckDB returns naive datetimes: both sides must produce exactly the
    same hash payload.
    """
    canon: dict[str, Any] = {}
    for key, value in fields.items():
        if isinstance(value, datetime):
            value = value.replace(tzinfo=None).isoformat(sep=" ", timespec="seconds")
        elif isinstance(value, str):
            try:
                parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                pass
            else:
                if parsed.tzinfo is not None:
                    value = parsed.astimezone(timezone.utc).replace(tzinfo=None)
                    value = value.isoformat(sep=" ", timespec="seconds")
        canon[key] = value
    return hashlib.sha256(
        json.dumps(canon, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


# Join used by every read: identity + operational state.
_ROW_SQL = (
    "SELECT e.experiment_id, e.hypothesis_id, e.title, e.parent_id, "
    "s.status, e.spec_json, e.content_hash, s.code_hash, s.config_hash, "
    "e.seed, e.label_id, e.label_version, e.hypothesis_family, e.research_epoch, "
    "e.trial_number, e.number_of_prior_trials, e.researcher, "
    "e.created_at, e.registered_at, s.updated_at "
    "FROM experiments e JOIN experiment_state s ON s.experiment_id = e.experiment_id"
)

_ROW_COLUMNS = [
    "experiment_id", "hypothesis_id", "title", "parent_id", "status",
    "spec_json", "content_hash", "code_hash", "config_hash", "seed",
    "label_id", "label_version", "hypothesis_family", "research_epoch",
    "trial_number", "number_of_prior_trials", "researcher",
    "created_at", "registered_at", "updated_at",
]


_SCHEMA_LOCK = threading.Lock()


class ExperimentRegistry:
    """Owns the DuckDB experiment ledger. One instance per process.

    All writes commit per call; multi-statement operations (registration,
    status + transition row, decision + status) run inside a transaction so
    the ledger is never half-written.
    """

    def __init__(self, db_path: str | Path | None = None):
        self._path = str(db_path) if db_path else ":memory:"
        self._con = duckdb.connect(self._path)
        # DuckDB serializes catalog writes per file; concurrent connections
        # (one per writer thread/process) must not race CREATE TABLE.
        with _SCHEMA_LOCK:
            self._con.execute(_SCHEMA)

    def close(self) -> None:
        self._con.close()

    @property
    def path(self) -> str:
        return self._path

    # ------------------------------------------------------------ register

    def register(
        self,
        *,
        experiment_id: str,
        hypothesis_id: str,
        title: str,
        parent_id: str | None,
        status: str,
        spec_json: str,
        content_hash: str,
        code_hash: str | None = None,
        config_hash: str | None = None,
        seed: int,
        label_id: str | None,
        label_version: str | None,
        hypothesis_family: str | None,
        research_epoch: str | None,
        selection_stage: str | None,
        trial_family: str,
        declared_trial_number: int | None,
        declared_prior: int | None,
        researcher: str,
        created_at: str,
        registered_at: str,
        dataset_snapshot_ids: list[str],
        feature_rows: list[tuple[str, str]],
        max_trials: int | None = None,
    ) -> tuple[int, int]:
        """Insert one experiment and its lineage join rows, atomically.

        The trial number is assigned from the atomic per-family counter
        inside the same transaction as the INSERT: concurrent registrations
        in one family can never collide on trial numbers, and a researcher-
        declared trial_number that disagrees with the assigned value rolls
        the registration back (search depth cannot be declared). When
        `max_trials` is given (hypothesis research budget), the count of
        live experiments for that hypothesis is re-checked INSIDE the same
        transaction, so concurrent registrations cannot overshoot the budget
        (the application-level pre-check is only a fast fail, never the
        guarantee). Returns (trial_number, number_of_prior_trials)."""
        if status not in _BIRTH_STATUSES:
            raise ValueError(
                f"invalid birth status: {status!r} - an experiment may only "
                "be registered as draft or registered, never into a later "
                "lifecycle state"
            )

        def _do() -> tuple[int, int]:
            self._con.execute("BEGIN TRANSACTION")
            try:
                if max_trials is not None:
                    row = self._con.execute(
                        """
                        SELECT COUNT(*) FROM experiments e
                          JOIN experiment_state s ON s.experiment_id = e.experiment_id
                         WHERE e.hypothesis_id = ? AND s.status NOT IN ('draft', 'retired')
                        """,
                        [hypothesis_id],
                    ).fetchone()
                    if int(row[0]) >= max_trials:
                        raise ValueError(
                            f"research budget exhausted for {hypothesis_id}: "
                            f"{max_trials} trials maximum"
                        )
                # Bump the family's trial counter: UPDATE when the row exists
                # (the common path, conflict-free under the writer lock), plain
                # INSERT for the very first registration of a family (which
                # races with other first-registrations and is retried below).
                counter = self._con.execute(
                    "UPDATE trial_counters SET value = value + 1 WHERE family = ? RETURNING value",
                    [trial_family],
                ).fetchone()
                if counter is None:
                    self._con.execute(
                        "INSERT INTO trial_counters (family, value) VALUES (?, 1)",
                        [trial_family],
                    )
                    trial_number = 1
                else:
                    trial_number = int(counter[0])
                prior = trial_number - 1
                if (
                    declared_trial_number is not None
                    and declared_trial_number != trial_number
                ):
                    raise ValueError(
                        f"trial_number is computed by the registry: expected "
                        f"{trial_number} (prior trials in family {trial_family!r}: "
                        f"{prior}), got {declared_trial_number}"
                    )
                if declared_prior is not None and declared_prior != prior:
                    raise ValueError(
                        f"number_of_prior_trials is computed by the registry: "
                        f"expected {prior}, got {declared_prior}"
                    )
                self._con.execute(
                    """
                    INSERT INTO experiments
                    (experiment_id, hypothesis_id, title, parent_id, spec_json,
                     content_hash, seed, label_id, label_version,
                     hypothesis_family, research_epoch, selection_stage,
                     trial_number, number_of_prior_trials, researcher,
                     created_at, registered_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        experiment_id, hypothesis_id, title, parent_id, spec_json,
                        content_hash, seed, label_id, label_version,
                        hypothesis_family, research_epoch, selection_stage,
                        trial_number, prior, researcher, created_at, registered_at,
                    ],
                )
                self._con.execute(
                    """
                    INSERT INTO experiment_state
                    (experiment_id, status, code_hash, config_hash, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    [experiment_id, status, code_hash, config_hash, registered_at],
                )
                for ds in sorted(set(dataset_snapshot_ids)):
                    self._con.execute(
                        "INSERT INTO experiment_datasets VALUES (?, ?)",
                        [experiment_id, ds],
                    )
                for fid, fver in sorted(set(feature_rows)):
                    self._con.execute(
                        "INSERT INTO experiment_features VALUES (?, ?, ?)",
                        [experiment_id, fid, fver],
                    )
                self._con.execute("COMMIT")
                return trial_number, prior
            except Exception:
                try:
                    self._con.execute("ROLLBACK")
                except duckdb.TransactionException:
                    pass  # a COMMIT-time failure already rolled the txn back
                raise

        try:
            # The per-family trial counter row may not exist yet; two concurrent
            # first registrations can both INSERT it and one loses at COMMIT
            # (duplicate-key error, ConstraintException or TransactionException
            # depending on timing). That is a retryable race, not a business
            # error: on retry the winner's row exists and is bumped instead.
            for attempt in range(20):
                try:
                    return _retry(_do)
                except (duckdb.ConstraintException, duckdb.TransactionException) as exc:
                    msg = str(exc).lower()
                    if "duplicate key" not in msg and "conflict on tuple deletion" not in msg:
                        raise
                    taken = self._con.execute(
                        "SELECT 1 FROM experiments WHERE experiment_id = ?",
                        [experiment_id],
                    ).fetchone()
                    if taken:
                        raise ValueError(
                            f"duplicate experiment id: {experiment_id}"
                        ) from exc
                    if attempt == 19:
                        raise ValueError(
                            f"registration of {experiment_id} failed on a "
                            f"trial-counter conflict: {exc}"
                        ) from exc
                    time.sleep(0.02)
                    continue
            raise AssertionError("unreachable")
        except _LockContention as exc:
            raise ValueError(f"registration of {experiment_id} failed: {exc}") from exc
        except duckdb.ConstraintException as exc:
            raise _constraint_error(exc) from exc

    # ---------------------------------------------------------- transitions

    def transition(
        self,
        *,
        experiment_id: str,
        from_status: str,
        to_status: str,
        code_hash: str | None = None,
        config_hash: str | None = None,
        transitioned_at: str,
        note: str | None = None,
    ) -> None:
        """Transition status iff the state row is still in `from_status`.

        The WHERE guard is the optimistic concurrency check: a transition
        racing with another writer fails loudly instead of silently
        overwriting it. code/config hashes are set once (COALESCE) and can
        never be overwritten.

        The lifecycle state machine is enforced HERE, in the ledger itself,
        not only in the service: a direct `ExperimentRegistry` user (an AI
        agent, a script, a future service) cannot bypass it. REJECTED and
        PROMOTED are decision states and refuse a bare transition - they are
        only reachable through `record_decision()`."""
        if from_status not in _VALID_STATUSES:
            raise ValueError(f"invalid experiment status: {from_status!r}")
        if to_status not in _VALID_STATUSES:
            raise ValueError(f"invalid experiment status: {to_status!r}")
        if to_status in ("rejected", "promoted"):
            raise ValueError(
                f"{to_status} is a decision state: it may only be entered "
                "through record_decision() with a reason and decision-maker, "
                "never by a bare status update"
            )
        validate_transition(
            ExperimentStatus(from_status), ExperimentStatus(to_status)
        )

        def _do() -> None:
            self._con.execute("BEGIN TRANSACTION")
            try:
                updated = self._con.execute(
                    """
                    UPDATE experiment_state
                       SET status = ?, code_hash = COALESCE(code_hash, ?),
                           config_hash = COALESCE(config_hash, ?), updated_at = ?
                     WHERE experiment_id = ? AND status = ?
                    """,
                    [to_status, code_hash, config_hash, transitioned_at,
                     experiment_id, from_status],
                ).fetchone()
                if updated is None or updated[0] != 1:
                    raise ValueError(
                        f"transition {from_status} -> {to_status} for {experiment_id} "
                        "failed: status already changed (concurrent update) or "
                        "experiment missing"
                    )
                self._con.execute(
                    "INSERT INTO transitions VALUES (?, ?, ?, ?, ?, ?)",
                    [experiment_id, from_status, to_status, transitioned_at, note,
                     record_content_hash(
                         {
                             "experiment_id": experiment_id,
                             "from_status": from_status,
                             "to_status": to_status,
                             "transitioned_at": transitioned_at,
                             "note": note,
                         }
                     )],
                )
                self._con.execute("COMMIT")
            except Exception:
                try:
                    self._con.execute("ROLLBACK")
                except duckdb.TransactionException:
                    pass  # a COMMIT-time failure already rolled the txn back
                raise

        try:
            _retry(_do)
        except _LockContention as exc:
            raise ValueError(f"transition of {experiment_id} failed: {exc}") from exc

    # -------------------------------------------------------------- decision

    def record_decision(
        self,
        *,
        experiment_id: str,
        decision: str,
        reason: str,
        policy_version: str | None,
        decision_maker: str,
        decided_at: str,
    ) -> str:
        """Append a decision and move the experiment to REJECTED/PROMOTED,
        atomically. The experiment must be COMPLETED and must carry a
        recorded result: a selection decision cites evidence, and this
        holds at the ledger level, not only in the service (a direct
        `ExperimentRegistry` user cannot bypass it)."""
        if decision not in {"rejected", "promoted"}:
            raise ValueError(f"invalid decision: {decision!r}")

        def _do() -> str:
            self._con.execute("BEGIN TRANSACTION")
            try:
                row = self._con.execute(
                    "SELECT status FROM experiment_state WHERE experiment_id = ?",
                    [experiment_id],
                ).fetchone()
                if row is None:
                    raise ValueError(f"unknown experiment: {experiment_id}")
                if row[0] != "completed":
                    raise ValueError(
                        f"decision on {experiment_id} requires status COMPLETED, "
                        f"got {row[0]}"
                    )
                if self._con.execute(
                    "SELECT 1 FROM results WHERE experiment_id = ?",
                    [experiment_id],
                ).fetchone() is None:
                    raise ValueError(
                        f"decision on {experiment_id} requires a recorded result: "
                        "a selection decision must cite the evidence it was made on"
                    )
                decision_id = _next_id(self._con, "decision", "DEC-", 6)
                self._con.execute(
                    "INSERT INTO decisions VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    [decision_id, experiment_id, decision, reason,
                     policy_version, decision_maker, decided_at,
                     record_content_hash(
                         {
                             "decision_id": decision_id,
                             "experiment_id": experiment_id,
                             "decision": decision,
                             "reason": reason,
                             "policy_version": policy_version,
                             "decision_maker": decision_maker,
                             "decided_at": decided_at,
                         }
                     )],
                )
                self._con.execute(
                    """
                    UPDATE experiment_state SET status = ?, updated_at = ?
                     WHERE experiment_id = ?
                    """,
                    [decision, decided_at, experiment_id],
                )
                self._con.execute(
                    "INSERT INTO transitions VALUES (?, ?, ?, ?, ?, ?)",
                    [experiment_id, "completed", decision, decided_at,
                     f"decision {decision_id}",
                     record_content_hash(
                         {
                             "experiment_id": experiment_id,
                             "from_status": "completed",
                             "to_status": decision,
                             "transitioned_at": decided_at,
                             "note": f"decision {decision_id}",
                         }
                     )],
                )
                self._con.execute("COMMIT")
                return decision_id
            except Exception:
                try:
                    self._con.execute("ROLLBACK")
                except duckdb.TransactionException:
                    pass  # a COMMIT-time failure already rolled the txn back
                raise

        try:
            return _retry(_do)
        except _LockContention as exc:
            raise ValueError(f"decision on {experiment_id} failed: {exc}") from exc

    # --------------------------------------------------------------- result

    def record_result(
        self,
        *,
        experiment_id: str,
        kind: str,
        summary: str,
        metrics_json: str | None,
        recorded_by: str,
        recorded_at: str,
    ) -> str:
        """Record the single immutable result of an experiment.

        `results.experiment_id` is UNIQUE: a second recording is a constraint
        violation and fails loudly - results are never silently rewritten."""
        if kind not in {
            "supported", "null", "negative", "invalid", "infrastructure_failure",
        }:
            raise ValueError(f"invalid result kind: {kind!r}")

        def _do() -> str:
            self._con.execute("BEGIN TRANSACTION")
            try:
                row = self._con.execute(
                    "SELECT status FROM experiment_state WHERE experiment_id = ?",
                    [experiment_id],
                ).fetchone()
                if row is None:
                    raise ValueError(f"unknown experiment: {experiment_id}")
                if row[0] not in {"completed", "failed"}:
                    raise ValueError(
                        f"result on {experiment_id} requires status COMPLETED or "
                        f"FAILED, got {row[0]}"
                    )
                result_id = _next_id(self._con, "result", "RES-", 6)
                self._con.execute(
                    "INSERT INTO results VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    [result_id, experiment_id, kind, summary, metrics_json,
                     recorded_at, recorded_by,
                     record_content_hash(
                         {
                             "result_id": result_id,
                             "experiment_id": experiment_id,
                             "kind": kind,
                             "summary": summary,
                             "metrics_json": metrics_json,
                             "recorded_at": recorded_at,
                             "recorded_by": recorded_by,
                         }
                     )],
                )
                self._con.execute("COMMIT")
                return result_id
            except Exception:
                try:
                    self._con.execute("ROLLBACK")
                except duckdb.TransactionException:
                    pass  # a COMMIT-time failure already rolled the txn back
                raise

        try:
            return _retry(_do)
        except _LockContention as exc:
            raise ValueError(f"result for {experiment_id} failed: {exc}") from exc
        except duckdb.ConstraintException as exc:
            raise _constraint_error(exc) from exc

    # -------------------------------------------------------------- artifact

    def attach_artifact(
        self,
        *,
        experiment_id: str,
        kind: str,
        path: str,
        checksum: str | None,
        created_at: str,
    ) -> str:
        """Attach one artifact to an experiment (FOREIGN KEY enforced)."""
        if not kind:
            raise ValueError("artifact kind is required")
        if not path:
            raise ValueError("artifact path is required")

        def _do() -> str:
            artifact_id = _next_id(self._con, "artifact", "ART-", 6)
            self._con.execute(
                "INSERT INTO artifacts VALUES (?, ?, ?, ?, ?, ?, ?)",
                [artifact_id, experiment_id, kind, path, checksum, created_at,
                 record_content_hash(
                     {
                         "artifact_id": artifact_id,
                         "experiment_id": experiment_id,
                         "kind": kind,
                         "path": path,
                         "checksum": checksum,
                         "created_at": created_at,
                     }
                 )],
            )
            return artifact_id

        try:
            return _retry(_do)
        except _LockContention as exc:
            raise ValueError(f"artifact attach for {experiment_id} failed: {exc}") from exc
        except duckdb.ConstraintException as exc:
            raise _constraint_error(exc) from exc

    # ----------------------------------------------------------------- reads

    @staticmethod
    def _parse_row(row: tuple, cols: list[str]) -> dict[str, Any]:
        return dict(zip(cols, row))

    def get(self, experiment_id: str) -> dict[str, Any] | None:
        row = self._con.execute(
            _ROW_SQL + " WHERE e.experiment_id = ?", [experiment_id]
        ).fetchone()
        if row is None:
            return None
        return self._parse_row(row, _ROW_COLUMNS)

    def list(
        self,
        *,
        status: str | None = None,
        hypothesis_id: str | None = None,
        hypothesis_family: str | None = None,
        research_epoch: str | None = None,
        selection_stage: str | None = None,
        label_id: str | None = None,
        label_version: str | None = None,
        dataset_snapshot_id: str | None = None,
        feature_id: str | None = None,
        parent_id: str | None = None,
        limit: int | None = 10000,
    ) -> list[dict[str, Any]]:
        """Indexed queries over the ledger (section 31 search/filtering)."""
        sql = _ROW_SQL
        where: list[str] = []
        params: list[Any] = []
        if dataset_snapshot_id is not None:
            sql += " JOIN experiment_datasets d ON d.experiment_id = e.experiment_id"
            where.append("d.dataset_snapshot_id = ?")
            params.append(dataset_snapshot_id)
        if feature_id is not None:
            sql += " JOIN experiment_features f ON f.experiment_id = e.experiment_id"
            where.append("f.feature_id = ?")
            params.append(feature_id)
        if status is not None:
            where.append("s.status = ?")
            params.append(status)
        if hypothesis_id is not None:
            where.append("e.hypothesis_id = ?")
            params.append(hypothesis_id)
        if hypothesis_family is not None:
            where.append(
                "(e.hypothesis_family = ? OR "
                "(e.hypothesis_family IS NULL AND e.hypothesis_id = ?))"
            )
            params.extend([hypothesis_family, hypothesis_family])
        if research_epoch is not None:
            where.append("e.research_epoch = ?")
            params.append(research_epoch)
        if selection_stage is not None:
            where.append("e.selection_stage = ?")
            params.append(selection_stage)
        if label_id is not None:
            where.append("e.label_id = ?")
            params.append(label_id)
        if label_version is not None:
            where.append("e.label_version = ?")
            params.append(label_version)
        if parent_id is not None:
            where.append("e.parent_id = ?")
            params.append(parent_id)
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY e.registered_at, e.experiment_id"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(int(limit))
        rows = self._con.execute(sql, params).fetchall()
        # dedupe join rows (same experiment matched via multiple features)
        seen: set[str] = set()
        out: list[dict[str, Any]] = []
        for row in rows:
            rec = self._parse_row(row, _ROW_COLUMNS)
            if rec["experiment_id"] in seen:
                continue
            seen.add(rec["experiment_id"])
            out.append(rec)
        return out

    def children(self, experiment_id: str) -> list[dict[str, Any]]:
        rows = self._con.execute(
            _ROW_SQL + " WHERE e.parent_id = ? ORDER BY e.registered_at, e.experiment_id",
            [experiment_id],
        ).fetchall()
        return [self._parse_row(r, _ROW_COLUMNS) for r in rows]

    def trial_count(self, trial_family: str) -> int:
        """Number of active experiments in the trial family.

        `trial_family` is the resolved family key (declared hypothesis_family
        or hypothesis_id). Retired and draft experiments do not count as
        search history against the live budget (Phase 1 semantics)."""
        row = self._con.execute(
            """
            SELECT COUNT(*) FROM experiments e
              JOIN experiment_state s ON s.experiment_id = e.experiment_id
             WHERE (e.hypothesis_family = ? OR (e.hypothesis_family IS NULL AND e.hypothesis_id = ?))
               AND s.status NOT IN ('draft', 'retired')
            """,
            [trial_family, trial_family],
        ).fetchone()
        return int(row[0])

    def count_by_hypothesis(self, hypothesis_id: str) -> int:
        """Live experiment count for one hypothesis, REGARDLESS of the
        declared hypothesis_family.

        This is the research-budget number: a researcher must not be able to
        escape a hypothesis budget by declaring a new family label per
        attempt (search depth is recorded per hypothesis, not per chosen
        family name)."""
        row = self._con.execute(
            """
            SELECT COUNT(*) FROM experiments e
              JOIN experiment_state s ON s.experiment_id = e.experiment_id
             WHERE e.hypothesis_id = ? AND s.status NOT IN ('draft', 'retired')
            """,
            [hypothesis_id],
        ).fetchone()
        return int(row[0])

    def has_id(self, experiment_id: str) -> bool:
        row = self._con.execute(
            "SELECT 1 FROM experiments WHERE experiment_id = ?", [experiment_id]
        ).fetchone()
        return row is not None

    def artifacts(self, experiment_id: str) -> list[dict[str, Any]]:
        rows = self._con.execute(
            "SELECT artifact_id, kind, path, checksum, created_at FROM artifacts "
            "WHERE experiment_id = ? ORDER BY artifact_id",
            [experiment_id],
        ).fetchall()
        cols = ["artifact_id", "kind", "path", "checksum", "created_at"]
        return [dict(zip(cols, r)) for r in rows]

    def result(self, experiment_id: str) -> dict[str, Any] | None:
        row = self._con.execute(
            "SELECT result_id, kind, summary, metrics_json, recorded_at, recorded_by "
            "FROM results WHERE experiment_id = ?",
            [experiment_id],
        ).fetchone()
        if row is None:
            return None
        cols = ["result_id", "kind", "summary", "metrics_json", "recorded_at", "recorded_by"]
        return dict(zip(cols, row))

    def decisions(self, experiment_id: str) -> list[dict[str, Any]]:
        rows = self._con.execute(
            "SELECT decision_id, decision, reason, policy_version, decision_maker, decided_at "
            "FROM decisions WHERE experiment_id = ? ORDER BY decided_at",
            [experiment_id],
        ).fetchall()
        cols = ["decision_id", "decision", "reason", "policy_version", "decision_maker", "decided_at"]
        return [dict(zip(cols, r)) for r in rows]

    def transitions(self, experiment_id: str) -> list[dict[str, Any]]:
        rows = self._con.execute(
            "SELECT from_status, to_status, transitioned_at, note FROM transitions "
            "WHERE experiment_id = ? ORDER BY transitioned_at, rowid",
            [experiment_id],
        ).fetchall()
        cols = ["from_status", "to_status", "transitioned_at", "note"]
        return [dict(zip(cols, r)) for r in rows]

    def lineage_joins(self, experiment_id: str) -> dict[str, list[Any]]:
        """The lineage join rows (datasets, features) for audit verification."""
        datasets = [
            row[0]
            for row in self._con.execute(
                "SELECT dataset_snapshot_id FROM experiment_datasets "
                "WHERE experiment_id = ?",
                [experiment_id],
            ).fetchall()
        ]
        features = [
            tuple(row)
            for row in self._con.execute(
                "SELECT feature_id, feature_version FROM experiment_features "
                "WHERE experiment_id = ?",
                [experiment_id],
            ).fetchall()
        ]
        return {"datasets": sorted(datasets), "features": sorted(features)}

    def audit_records(self, experiment_id: str) -> dict[str, list[dict[str, Any]]]:
        """Every child record (artifacts, results, decisions, transitions)
        with its stored `content_hash`, for audit verification. The public
        readers deliberately exclude the hash; this is the audit view."""
        out: dict[str, list[dict[str, Any]]] = {}
        for table, cols in (
            ("artifacts", ["artifact_id", "kind", "path", "checksum", "created_at", "content_hash"]),
            ("results", ["result_id", "kind", "summary", "metrics_json", "recorded_at", "recorded_by", "content_hash"]),
            ("decisions", ["decision_id", "decision", "reason", "policy_version", "decision_maker", "decided_at", "content_hash"]),
            ("transitions", ["from_status", "to_status", "transitioned_at", "note", "content_hash"]),
        ):
            rows = self._con.execute(
                f"SELECT {', '.join(cols)} FROM {table} WHERE experiment_id = ?",
                [experiment_id],
            ).fetchall()
            out[table] = [dict(zip(cols, r)) for r in rows]
        return out

    def count(self) -> int:
        return int(self._con.execute("SELECT COUNT(*) FROM experiments").fetchone()[0])

    def status_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for row in self._con.execute(
            "SELECT status, COUNT(*) FROM experiment_state GROUP BY status"
        ).fetchall():
            counts[row[0]] = int(row[1])
        return counts

    def orphan_counts(self) -> dict[str, int]:
        """Counts of records whose experiment reference is missing.

        Every value must be 0: artifacts, results, decisions, transitions,
        state and lineage join rows are FK-bound to a real experiment."""
        out: dict[str, int] = {}
        for table in (
            "experiment_state",
            "artifacts",
            "results",
            "decisions",
            "transitions",
            "experiment_datasets",
            "experiment_features",
        ):
            row = self._con.execute(
                f"SELECT COUNT(*) FROM {table} t "
                "LEFT JOIN experiments e ON e.experiment_id = t.experiment_id "
                "WHERE e.experiment_id IS NULL"
            ).fetchone()
            out[table] = int(row[0])
        return out

    def dump(self) -> list[dict[str, Any]]:
        """Full ledger dump for auditing (identity + status + hashes)."""
        rows = self._con.execute(
            _ROW_SQL + " ORDER BY e.registered_at, e.experiment_id"
        ).fetchall()
        return [self._parse_row(r, _ROW_COLUMNS) for r in rows]


__all__ = ["ExperimentRegistry", "record_content_hash"]