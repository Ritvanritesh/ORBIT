"""ExperimentService: ORBIT's internal research-control plane (Phase 6).

The service is the ONLY sanctioned path into the experiment registry. It
enforces, in one place:

  - register-before-run: registration happens before any execution, and a
    registered experiment cannot have its scientific identity changed;
  - structured hypotheses (Phase 1 registry) - experiments cannot float free
    of a registered, falsifiable hypothesis;
  - hypothesis-scoped genealogy with acyclic, immutable ancestry;
  - exact dataset snapshot ids (Phase 3), pinned label versions (Phase 5),
    pinned temporal configuration (Phase 4), cost-model identity;
  - code/config hashes captured before execution starts;
  - registry-computed trial numbers (search depth cannot be declared by the
    researcher, only observed);
  - validated lifecycle transitions with a recorded decision log for
    REJECTED/PROMOTED;
  - immutable result records and FK-bound artifacts (no orphans);
  - reproduction specifications that resolve every lineage element;
  - invariant validation for audits (content-hash recomputation, acyclicity,
    orphan counts, decision/status consistency).

AI researchers get no special path: an agent submits the same structured
ExperimentSpec through the same register() and is recorded by the same
`researcher` identity field.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from orbit.experiments.lifecycle import (
    DECISION_STATES,
    PARENT_ELIGIBLE_STATES,
    validate_transition,
)
from orbit.experiments.registry import ExperimentRegistry
from orbit.experiments.reproduction import ReproductionSpec, build_reproduction_spec
from orbit.schemas.common import ExperimentStatus
from orbit.schemas.experiment import ExperimentSpec
from orbit.schemas.hypothesis import HypothesisRegistry


class Decision(str, Enum):
    """Selection decisions recorded in the decision log."""

    REJECTED = "rejected"
    PROMOTED = "promoted"


class ResultKind(str, Enum):
    """Kinds of recorded experiment results (null/failed history is kept)."""

    SUPPORTED = "supported"
    NULL = "null"
    NEGATIVE = "negative"
    INVALID = "invalid"
    INFRASTRUCTURE_FAILURE = "infrastructure_failure"


# Placeholder reasons are not a research record (section 25).
_PLACEHOLDER_REASONS = frozenset(
    {
        "we didn't like it",
        "we didn't like the results",
        "didn't like it",
        "did not like it",
        "no reason",
        "because",
        "n/a",
        "na",
        "idk",
        "not good",
        "it didn't work",
        "didn't work",
    }
)

_RESULT_REQUIRING_STATUSES = frozenset(
    {ExperimentStatus.COMPLETED, ExperimentStatus.FAILED}
)
_DECISION_REQUIRING_STATUSES = frozenset({ExperimentStatus.COMPLETED})
_DECISION_STATE_REQUIRES_RESULT = frozenset(
    {ExperimentStatus.REJECTED, ExperimentStatus.PROMOTED}
)


def temporal_config_digest(contract: Any) -> str:
    """sha256 of the loaded Phase 4 TemporalContract's canonical JSON.

    This is the digest an experiment pins in `TemporalConfigRef.config_digest`:
    a future rerun must resolve the same temporal policy, or the pin fails.
    """
    payload = contract.model_dump(mode="json")
    raw = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _aware(ts: datetime) -> datetime:
    """DuckDB returns naive TIMESTAMP values; the schema is tz-aware UTC."""
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts


class ExperimentService:
    """Research-control API over the persistent experiment registry."""

    def __init__(
        self,
        registry: ExperimentRegistry | None = None,
        *,
        db_path: str | Path | None = None,
        hypothesis_registry: HypothesisRegistry | None = None,
        label_registry: Any | None = None,
        dataset_registry: Any | None = None,
        temporal_contract: Any | None = None,
    ):
        self._registry = registry or ExperimentRegistry(db_path)
        self._hypotheses = hypothesis_registry
        self._labels = label_registry
        self._datasets = dataset_registry
        self._temporal = temporal_contract

    # ------------------------------------------------------------ helpers

    def _now(self) -> datetime:
        return _now()

    def _require(self, experiment_id: str) -> ExperimentSpec:
        exp = self.get(experiment_id)
        if exp is None:
            raise ValueError(f"unknown experiment: {experiment_id}")
        return exp

    def _reconstruct(self, row: dict[str, Any]) -> ExperimentSpec:
        """Reconstruct the frozen spec from stored identity + operational cols."""
        data = json.loads(row["spec_json"])
        data["status"] = row["status"]
        data["code_hash"] = row["code_hash"]
        data["config_hash"] = row["config_hash"]
        data["trial_number"] = row["trial_number"]
        data["number_of_prior_trials"] = row["number_of_prior_trials"]
        if row["created_at"] is not None:
            data["created_at"] = _aware(row["created_at"])
        if row["registered_at"] is not None:
            data["registered_at"] = _aware(row["registered_at"])
        return ExperimentSpec(**data)

    # ----------------------------------------------------------- register

    def register(
        self,
        spec: ExperimentSpec,
        *,
        registered_at: datetime | None = None,
        created_at: datetime | None = None,
    ) -> ExperimentSpec:
        """Register an experiment BEFORE its result is known.

        Validation performed here:
          - the spec is a canonical ExperimentSpec (no raw-dict bypass);
          - the experiment id is unique;
          - the parent exists, is not draft/retired, is hypothesis-scoped, and
            the ancestry is acyclic;
          - the hypothesis exists in the Phase 1 registry and its research
            budget is not exhausted;
          - every dataset_snapshot_id resolves in the Phase 3 registry;
          - the label (id, version) resolves in the Phase 5 registry;
          - the temporal configuration pins the loaded Phase 4 contract;
          - trial counters are computed by the registry, not declared.
        """
        if not isinstance(spec, ExperimentSpec):
            raise TypeError(
                "register() requires an ExperimentSpec instance (a raw dict "
                "cannot bypass schema validation)"
            )
        if self._registry.has_id(spec.experiment_id):
            raise ValueError(f"duplicate experiment id: {spec.experiment_id}")
        if spec.parent_id is not None:
            if spec.parent_id == spec.experiment_id:
                raise ValueError(f"self-parenting is forbidden: {spec.experiment_id}")
            parent = self.get(spec.parent_id)
            if parent is None:
                raise ValueError(
                    f"unknown parent experiment: {spec.parent_id}"
                )
            if parent.status not in PARENT_ELIGIBLE_STATES:
                raise ValueError(
                    f"parent {spec.parent_id} is {parent.status.value}; a parent "
                    "must be an active registered experiment (draft and retired "
                    "parents cannot take children)"
                )
            if parent.hypothesis_id != spec.hypothesis_id:
                raise ValueError(
                    f"parent {spec.parent_id} belongs to {parent.hypothesis_id}, "
                    f"not {spec.hypothesis_id}: genealogy is hypothesis-scoped"
                )
            self._assert_acyclic(spec.parent_id, spec.experiment_id)

        if self._hypotheses is not None:
            try:
                hyp = self._hypotheses.get(spec.hypothesis_id)
            except KeyError:
                raise ValueError(
                    f"experiment references unknown hypothesis: {spec.hypothesis_id}"
                ) from None
            if hyp.status.value not in ("registered", "active", "falsified", "abandoned", "promoted"):
                raise ValueError(
                    f"hypothesis {spec.hypothesis_id} is {hyp.status.value}; "
                    "experiments may only reference registered hypotheses"
                )
            if self.count_trials(spec.hypothesis_id) >= hyp.research_budget.max_trials:
                raise ValueError(
                    f"research budget exhausted for {spec.hypothesis_id}: "
                    f"{hyp.research_budget.max_trials} trials maximum"
                )

        if not spec.dataset_snapshot_ids:
            raise ValueError(
                "registration requires exact dataset_snapshot_ids (DS-xxxxxx); "
                "descriptive dataset names alone are not reproducible"
            )
        if self._datasets is not None:
            for ds in spec.dataset_snapshot_ids:
                if self._datasets.snapshot(ds) is None:
                    raise ValueError(f"unknown dataset snapshot: {ds}")

        if spec.label_id is not None and self._labels is not None:
            try:
                self._labels.get(spec.label_id, spec.label_version)
            except KeyError as exc:
                raise ValueError(f"unknown label reference: {exc}") from None

        if spec.temporal_config is None:
            raise ValueError(
                "registration requires temporal_config (Phase 4 temporal identity); "
                "an experiment must never silently run under a different temporal policy"
            )
        if self._temporal is not None:
            expected_digest = temporal_config_digest(self._temporal)
            if spec.temporal_config.config_digest != expected_digest:
                raise ValueError(
                    "temporal_config.config_digest does not match the loaded Phase 4 "
                    "contract; the temporal policy pin is wrong"
                )
            if spec.temporal_config.engine_version != self._temporal.engine_version:
                raise ValueError(
                    f"temporal_config.engine_version {spec.temporal_config.engine_version} "
                    f"does not match the loaded contract {self._temporal.engine_version}"
                )

        family = spec.hypothesis_family or spec.hypothesis_id
        if spec.features.feature_refs:
            if (
                spec.feature_count is not None
                and spec.feature_count != len(spec.features.feature_refs)
            ):
                raise ValueError(
                    f"feature_count {spec.feature_count} does not match the "
                    f"{len(spec.features.feature_refs)} pinned feature refs"
                )

        reg_at = registered_at or self._now()
        created = created_at or reg_at
        feature_rows = [
            (f.feature_id, f.feature_version) for f in spec.features.feature_refs
        ]
        trial_number, prior = self._registry.register(
            experiment_id=spec.experiment_id,
            hypothesis_id=spec.hypothesis_id,
            title=spec.title,
            parent_id=spec.parent_id,
            status="registered",
            spec_json=json.dumps(
                spec.model_dump(exclude={"status", "code_hash", "config_hash"}, mode="json"),
                sort_keys=True,
            ),
            content_hash=spec.content_hash(),
            code_hash=spec.code_hash,
            config_hash=spec.config_hash,
            seed=spec.seed,
            label_id=spec.label_id,
            label_version=spec.label_version,
            hypothesis_family=spec.hypothesis_family,
            research_epoch=spec.research_epoch,
            selection_stage=spec.selection_stage,
            trial_family=family,
            declared_trial_number=spec.trial_number,
            declared_prior=spec.number_of_prior_trials,
            researcher=spec.researcher,
            created_at=created.isoformat(),
            registered_at=reg_at.isoformat(),
            dataset_snapshot_ids=spec.dataset_snapshot_ids,
            feature_rows=feature_rows,
        )
        return spec.model_copy(
            update={
                "status": ExperimentStatus.REGISTERED,
                "registered_at": reg_at,
                "created_at": created,
                "trial_number": trial_number,
                "number_of_prior_trials": prior,
            }
        )

    # ------------------------------------------------------------- reads

    def get(self, experiment_id: str) -> ExperimentSpec | None:
        row = self._registry.get(experiment_id)
        if row is None:
            return None
        return self._reconstruct(row)

    def list(
        self,
        *,
        status: ExperimentStatus | str | None = None,
        hypothesis_id: str | None = None,
        hypothesis_family: str | None = None,
        research_epoch: str | None = None,
        selection_stage: str | None = None,
        label_id: str | None = None,
        dataset_snapshot_id: str | None = None,
        feature_id: str | None = None,
        parent_id: str | None = None,
        limit: int | None = 10000,
    ) -> list[ExperimentSpec]:
        status_value = status.value if isinstance(status, ExperimentStatus) else status
        rows = self._registry.list(
            status=status_value,
            hypothesis_id=hypothesis_id,
            hypothesis_family=hypothesis_family,
            research_epoch=research_epoch,
            selection_stage=selection_stage,
            label_id=label_id,
            dataset_snapshot_id=dataset_snapshot_id,
            feature_id=feature_id,
            parent_id=parent_id,
            limit=limit,
        )
        return [self._reconstruct(r) for r in rows]

    def children(self, experiment_id: str) -> list[ExperimentSpec]:
        self._require(experiment_id)
        return [self._reconstruct(r) for r in self._registry.children(experiment_id)]

    def descendants(self, experiment_id: str) -> list[ExperimentSpec]:
        """All experiments in the subtree rooted at experiment_id (BFS)."""
        self._require(experiment_id)
        out: list[ExperimentSpec] = []
        frontier = [experiment_id]
        seen = {experiment_id}
        while frontier:
            current = frontier.pop(0)
            kids = self.children(current)
            for kid in kids:
                if kid.experiment_id in seen:
                    raise ValueError(
                        f"genealogy cycle detected at {kid.experiment_id}"
                    )
                seen.add(kid.experiment_id)
                out.append(kid)
                frontier.append(kid.experiment_id)
        return out

    def ancestry(self, experiment_id: str) -> list[ExperimentSpec]:
        """The complete ancestry, root first (the path that produced EXP-x)."""
        exp = self._require(experiment_id)
        chain: list[ExperimentSpec] = []
        seen: set[str] = set()
        current = exp
        while current.parent_id is not None:
            if current.experiment_id in seen:
                raise ValueError(
                    f"genealogy cycle detected at {current.experiment_id}"
                )
            seen.add(current.experiment_id)
            parent = self._require(current.parent_id)
            chain.append(parent)
            current = parent
        chain.reverse()
        return chain

    def _assert_acyclic(self, ancestor_id: str, child_id: str) -> None:
        """Defense in depth: registering `child_id` under `ancestor_id` must
        never close a cycle. New nodes cannot close cycles (the ancestry was
        acyclic and is being extended), but we check anyway."""
        seen: set[str] = set()
        current = ancestor_id
        while current is not None:
            if current in seen:
                raise ValueError(f"genealogy cycle detected at {current}")
            seen.add(current)
            if current == child_id:
                raise ValueError(f"genealogy cycle: {child_id} is its own ancestor")
            row = self._registry.get(current)
            if row is None:
                raise ValueError(f"unknown parent experiment: {current}")
            current = row["parent_id"]

    def count_trials(self, hypothesis_id: str) -> int:
        """Active experiment count per hypothesis (Phase 1 budget semantics)."""
        return self._registry.trial_count(hypothesis_id)

    # -------------------------------------------------------- lifecycle

    def mark_running(
        self,
        experiment_id: str,
        *,
        code_hash: str | None = None,
        config_hash: str | None = None,
        note: str | None = None,
    ) -> ExperimentSpec:
        """Enter RUNNING with the executing code/config identity captured.

        Register-before-run is enforced by requiring the code identity before
        execution starts; the hashes are immutable once set."""
        exp = self._require(experiment_id)
        validate_transition(exp.status, ExperimentStatus.RUNNING)
        if exp.code_hash is None and code_hash is None:
            raise ValueError(
                f"mark_running({experiment_id}) requires code_hash: the executing "
                "code identity must be captured before execution"
            )
        if exp.config_hash is None and config_hash is None:
            raise ValueError(
                f"mark_running({experiment_id}) requires config_hash: the executing "
                "configuration identity must be captured before execution"
            )
        if exp.code_hash is not None and code_hash is not None and exp.code_hash != code_hash:
            raise ValueError(
                f"code_hash for {experiment_id} is already set ({exp.code_hash[:12]}...); "
                "it cannot be changed after execution begins"
            )
        if exp.config_hash is not None and config_hash is not None and exp.config_hash != config_hash:
            raise ValueError(
                f"config_hash for {experiment_id} is already set; it cannot be "
                "changed after execution begins"
            )
        self._registry.transition(
            experiment_id=experiment_id,
            from_status=exp.status.value,
            to_status="running",
            code_hash=code_hash,
            config_hash=config_hash,
            transitioned_at=self._now().isoformat(),
            note=note,
        )
        return self._require(experiment_id)

    def transition(
        self,
        experiment_id: str,
        to: ExperimentStatus | str,
        *,
        note: str | None = None,
    ) -> ExperimentSpec:
        """A validated lifecycle transition (not for REJECTED/PROMOTED - those
        require a recorded decision via record_decision())."""
        target = to if isinstance(to, ExperimentStatus) else ExperimentStatus(to)
        if target in DECISION_STATES:
            raise ValueError(
                f"{target.value} is a decision state: use record_decision() with a "
                "reason and decision-maker, never a bare status update"
            )
        exp = self._require(experiment_id)
        validate_transition(exp.status, target)
        self._registry.transition(
            experiment_id=experiment_id,
            from_status=exp.status.value,
            to_status=target.value,
            transitioned_at=self._now().isoformat(),
            note=note,
        )
        return self._require(experiment_id)

    def complete(self, experiment_id: str, *, note: str | None = None) -> ExperimentSpec:
        return self.transition(experiment_id, ExperimentStatus.COMPLETED, note=note)

    def fail(self, experiment_id: str, *, note: str | None = None) -> ExperimentSpec:
        return self.transition(experiment_id, ExperimentStatus.FAILED, note=note)

    def retire(self, experiment_id: str, *, note: str | None = None) -> ExperimentSpec:
        """Archival: the experiment leaves active research but its full history
        (identity, lineage, results, decisions, artifacts) remains intact."""
        return self.transition(experiment_id, ExperimentStatus.RETIRED, note=note)

    # ---------------------------------------------------------- artifacts

    def attach_artifact(
        self,
        experiment_id: str,
        *,
        kind: str,
        path: str,
        checksum: str | None = None,
        created_at: datetime | None = None,
    ) -> str:
        """Attach one output (metrics, predictions, logs, reports, plots,
        model artifact, ...) to an experiment. Every artifact is FK-bound:
        no orphan artifacts can exist."""
        self._require(experiment_id)
        return self._registry.attach_artifact(
            experiment_id=experiment_id,
            kind=kind,
            path=path,
            checksum=checksum,
            created_at=(created_at or self._now()).isoformat(),
        )

    def artifacts(self, experiment_id: str) -> list[dict[str, Any]]:
        self._require(experiment_id)
        return self._registry.artifacts(experiment_id)

    # ------------------------------------------------------------ results

    def record_result(
        self,
        experiment_id: str,
        *,
        kind: ResultKind | str,
        summary: str,
        metrics: dict[str, Any] | None = None,
        recorded_by: str = "orbit-research",
        recorded_at: datetime | None = None,
    ) -> str:
        """Record the single immutable result of an experiment.

        One result per experiment: a second recording is refused loudly.
        Null, negative and failed results are recorded exactly like successes -
        history is never hidden."""
        exp = self._require(experiment_id)
        if exp.status not in _RESULT_REQUIRING_STATUSES:
            raise ValueError(
                f"result for {experiment_id} requires status COMPLETED or FAILED, "
                f"got {exp.status.value}"
            )
        kind_value = kind.value if isinstance(kind, ResultKind) else str(kind)
        if not summary or not str(summary).strip():
            raise ValueError("result summary is required")
        metrics_json = (
            json.dumps(metrics, sort_keys=True, default=str) if metrics is not None else None
        )
        return self._registry.record_result(
            experiment_id=experiment_id,
            kind=kind_value,
            summary=str(summary),
            metrics_json=metrics_json,
            recorded_by=recorded_by,
            recorded_at=(recorded_at or self._now()).isoformat(),
        )

    def result(self, experiment_id: str) -> dict[str, Any] | None:
        return self._registry.result(experiment_id)

    # ---------------------------------------------------------- decisions

    def record_decision(
        self,
        experiment_id: str,
        *,
        decision: Decision | str,
        reason: str,
        decision_maker: str,
        policy_version: str | None = None,
        decided_at: datetime | None = None,
    ) -> ExperimentSpec:
        """Record a selection decision and move the experiment to REJECTED or
        PROMOTED, atomically. Decisions require a COMPLETED experiment and a
        substantive reason - 'we didn't like it' is not a research record."""
        exp = self._require(experiment_id)
        if exp.status not in _DECISION_REQUIRING_STATUSES:
            raise ValueError(
                f"decision on {experiment_id} requires status COMPLETED, got {exp.status.value}"
            )
        decision_value = decision.value if isinstance(decision, Decision) else str(decision)
        if decision_value not in {Decision.REJECTED.value, Decision.PROMOTED.value}:
            raise ValueError(f"invalid decision: {decision_value!r}")
        cleaned = str(reason).strip()
        if len(cleaned) < 10:
            raise ValueError(
                "decision reason must be substantive (>= 10 characters); a bare "
                "or empty reason is not a research record"
            )
        if cleaned.casefold() in _PLACEHOLDER_REASONS:
            raise ValueError(
                f"placeholder reason {cleaned!r} is not a research decision; "
                "cite the evidence and criteria"
            )
        if not decision_maker or not str(decision_maker).strip():
            raise ValueError("decision_maker is required (researcher or agent id)")
        self._registry.record_decision(
            experiment_id=experiment_id,
            decision=decision_value,
            reason=cleaned,
            policy_version=policy_version,
            decision_maker=decision_maker,
            decided_at=(decided_at or self._now()).isoformat(),
        )
        return self._require(experiment_id)

    def decisions(self, experiment_id: str) -> list[dict[str, Any]]:
        return self._registry.decisions(experiment_id)

    def transitions(self, experiment_id: str) -> list[dict[str, Any]]:
        return self._registry.transitions(experiment_id)

    # -------------------------------------------------- reproduction spec

    def reproduction_spec(self, experiment_id: str) -> ReproductionSpec:
        """Resolve everything needed to reproduce the experiment (section 22).

        Raises ValueError when a pinned lineage element cannot be resolved
        (a missing dataset snapshot or label contract is a lineage violation,
        not a cosmetic warning)."""
        exp = self._require(experiment_id)
        row = self._registry.get(experiment_id)

        datasets: list[dict[str, Any]] = []
        for ds in exp.dataset_snapshot_ids:
            if self._datasets is not None:
                record = self._datasets.snapshot(ds)
                if record is None:
                    raise ValueError(
                        f"lineage violation: dataset snapshot {ds} referenced by "
                        f"{experiment_id} no longer resolves in the Phase 3 registry"
                    )
                datasets.append(dict(record))
            else:
                datasets.append({"snapshot_id": ds, "resolved": False})

        label: dict[str, Any] | None = None
        if exp.label_id is not None:
            if self._labels is not None:
                try:
                    rec = self._labels.get(exp.label_id, exp.label_version)
                except KeyError as exc:
                    raise ValueError(
                        f"lineage violation: label {exp.label_id} v{exp.label_version} "
                        f"referenced by {experiment_id} no longer resolves: {exc}"
                    ) from None
                label = rec.contract.definition_summary()
            else:
                label = {
                    "label_id": exp.label_id,
                    "version": exp.label_version,
                    "resolved": False,
                }

        temporal: dict[str, Any] | None = None
        if self._temporal is not None:
            t = self._temporal
            temporal = {
                "engine_version": t.engine_version,
                "config_digest": temporal_config_digest(t),
                "boundary": t.boundary,
                "date_precision": t.date_precision,
                "exchange_tz": t.exchange_tz,
                "session_close": t.session_close,
                "market_bar_available": t.market_bar_available,
                "forward_dated_events": t.forward_dated_events,
                "default_series_policy": t.default_series_policy,
                "series_policies": dict(t.series_policies),
            }

        hypothesis: dict[str, Any] | None = None
        if self._hypotheses is not None:
            try:
                hypothesis = self._hypotheses.get(exp.hypothesis_id).model_dump(mode="json")
            except KeyError:
                raise ValueError(
                    f"lineage violation: hypothesis {exp.hypothesis_id} referenced by "
                    f"{experiment_id} no longer resolves"
                ) from None

        features = [
            {
                "feature_id": f.feature_id,
                "feature_version": f.feature_version,
                "transformation": f.transformation,
            }
            for f in exp.features.feature_refs
        ]

        return build_reproduction_spec(
            spec=exp,
            status=exp.status,
            code_hash=row["code_hash"],
            config_hash=row["config_hash"],
            registered_at=exp.registered_at,
            hypothesis=hypothesis,
            datasets=datasets,
            temporal=temporal,
            label=label,
            features=features,
            result=self._registry.result(experiment_id),
            decision=(
                self._registry.decisions(experiment_id)[-1]
                if self._registry.decisions(experiment_id)
                else None
            ),
            artifacts=self._registry.artifacts(experiment_id),
        )

    # ---------------------------------------------------------- invariants

    def validate_invariants(self) -> dict[str, Any]:
        """Audit every registered experiment against the Phase 6 invariants.

        Returns a report; `ok` is True only when no violation is found. This
        is the machine half of the second independent audit (section 39)."""
        violations: list[str] = []
        rows = self._registry.dump()

        for row in rows:
            exp_id = row["experiment_id"]
            try:
                exp = self._reconstruct(row)
            except Exception as exc:  # noqa: BLE001
                violations.append(f"{exp_id}: spec no longer reconstructs: {exc}")
                continue

            # identity integrity: stored hash must match recomputation
            if exp.content_hash() != row["content_hash"]:
                violations.append(
                    f"{exp_id}: content_hash mismatch - the stored scientific "
                    "identity has been altered"
                )

            # acyclicity of the ancestry chain
            try:
                self.ancestry(exp_id)
            except ValueError as exc:
                violations.append(f"{exp_id}: {exc}")

            # lineage completeness
            if exp.temporal_config is None:
                violations.append(f"{exp_id}: missing temporal_config")
            if not exp.dataset_snapshot_ids:
                violations.append(f"{exp_id}: missing dataset snapshot lineage")
            if exp.status.value in ("running", "completed", "rejected", "promoted"):
                if row["code_hash"] is None:
                    violations.append(f"{exp_id}: running/completed without code_hash")
                if row["config_hash"] is None:
                    violations.append(f"{exp_id}: running/completed without config_hash")

            # decision/result consistency
            has_decision = bool(self._registry.decisions(exp_id))
            has_result = self._registry.result(exp_id) is not None
            if exp.status.value in ("rejected", "promoted"):
                if not has_decision:
                    violations.append(f"{exp_id}: status {exp.status.value} without a recorded decision")
                if not has_result:
                    violations.append(f"{exp_id}: status {exp.status.value} without a recorded result")
            elif has_decision:
                violations.append(
                    f"{exp_id}: decision recorded but status is {exp.status.value}"
                )

        orphans = self._registry.orphan_counts()
        for table, count in orphans.items():
            if count:
                violations.append(f"orphan records in {table}: {count}")

        return {
            "ok": not violations,
            "violations": violations,
            "experiments": len(rows),
            "orphan_counts": orphans,
            "status_counts": self._registry.status_counts(),
        }


__all__ = [
    "Decision",
    "ExperimentService",
    "ResultKind",
    "temporal_config_digest",
]