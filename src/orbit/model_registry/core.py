"""Phase 14 — Model Registry & Evidence-Gated Promotion (core).

Central principle: "A model is replaceable. The evidence chain is the asset."

Design contracts enforced here:
  - Identity immutability: identity_digest covers every scientifically
    meaningful component (family, params, data, features, labels, windows,
    seed, preprocessing, code/config hashes). Any mutation changes the digest;
    a digest collision is a duplicate-registration error.
  - Evidence-gated lifecycle: status changes ONLY through an immutable
    PromotionDecisionRecord produced by gate evaluation. Missing evidence is
    BLOCKED/NOT_EVALUATED — never converted into PASS.
  - Determinism: all record timestamps use a fixed phase clock so two builds
    from identical inputs produce byte-equivalent registries.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

PHASE_CLOCK = "2026-08-23T00:00:00+00:00"  # fixed logical clock (determinism)


# =====================================================================
# ENUMS
# =====================================================================

class LifecycleStatus(str, Enum):
    RESEARCH = "RESEARCH"      # registered artifact; weak/incomplete evidence allowed
    VALIDATED = "VALIDATED"    # reproducibility + evidence gates passed; still not economic proof
    PAPER = "PAPER"            # paper-trading eligible (portfolio/cost/execution gates passed)
    RETIRED = "RETIRED"        # permanently ineligible; artifacts remain replayable


class PromotionAction(str, Enum):
    PROMOTE = "PROMOTE"
    RETAIN = "RETAIN"
    BLOCK = "BLOCK"
    DEMOTE = "DEMOTE"
    RETIRE = "RETIRE"


class GateResult(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    NOT_EVALUATED = "NOT_EVALUATED"
    BLOCKED = "BLOCKED"


class EvidenceType(str, Enum):
    REPRODUCIBILITY = "REPRODUCIBILITY"
    DATA_INTEGRITY = "DATA_INTEGRITY"
    PIT_INTEGRITY = "PIT_INTEGRITY"
    LABEL_VALIDITY = "LABEL_VALIDITY"
    STATISTICAL_INFERENCE = "STATISTICAL_INFERENCE"
    MULTIPLE_TESTING = "MULTIPLE_TESTING"
    TEMPORAL_STABILITY = "TEMPORAL_STABILITY"
    UNIVERSE_STABILITY = "UNIVERSE_STABILITY"
    MODEL_FAMILY_STABILITY = "MODEL_FAMILY_STABILITY"
    STRESS_TESTING = "STRESS_TESTING"
    ECONOMIC_MATERIALITY = "ECONOMIC_MATERIALITY"
    PORTFOLIO_VALIDATION = "PORTFOLIO_VALIDATION"
    EXECUTION_VALIDATION = "EXECUTION_VALIDATION"
    DEFECT = "DEFECT"
    AUDIT = "AUDIT"


class ReplayMode(str, Enum):
    IDENTITY_REPLAY = "IDENTITY_REPLAY"
    FULL_NUMERICAL_REPLAY = "FULL_NUMERICAL_REPLAY"


# =====================================================================
# ERRORS
# =====================================================================

class RegistryViolation(Exception):
    """Any attempt to violate registry integrity."""


# =====================================================================
# DIGEST HELPERS
# =====================================================================

def canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, default=str)


def digest_full(obj: Any) -> str:
    return hashlib.sha256(canonical(obj).encode("utf-8")).hexdigest()


def digest_short(obj: Any) -> str:
    return digest_full(obj)[:16]


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(Path(path).read_bytes())
    return h.hexdigest()


# =====================================================================
# IDENTITY CONTRACT
# =====================================================================

#: Fields that constitute scientific identity. ANY change alters identity.
IDENTITY_FIELDS = [
    "model_family", "hyperparameters", "preprocessing", "target_transform",
    "dataset_snapshot_ids", "universe_id", "instrument_identity_version",
    "feature_set_id", "feature_set_version", "feature_definitions_digest",
    "label_id", "label_version", "label_contract_digest",
    "benchmark_id", "cost_model_id",
    "train_start", "train_end", "val_start", "val_end",
    "test_start", "test_end", "purge_policy", "embargo_days", "window_protocol_digest",
    "seed", "code_hash", "config_hash",
]

REQUIRED_MODEL_KEYS = [
    "model_id", "model_version", *IDENTITY_FIELDS,
]


def identity_of(model: dict) -> dict:
    return {k: model[k] for k in IDENTITY_FIELDS}


def identity_digest(model: dict) -> str:
    return digest_short(identity_of(model))


def validate_model_payload(model: dict) -> None:
    missing = [k for k in REQUIRED_MODEL_KEYS if k not in model]
    if missing:
        raise RegistryViolation(f"incomplete model identity; missing {missing}")
    if not str(model["model_id"]).startswith("MODEL-"):
        raise RegistryViolation("model_id must start with MODEL-")
    art = model.get("artifacts") or {}
    uri = art.get("model_artifact_uri")
    cksum = art.get("model_artifact_checksum")
    if uri and not cksum:
        raise RegistryViolation(
            f"A9 violation: anonymous artifact for {model['model_id']} "
            "(artifact URI present without checksum)")
    if cksum and len(cksum) != 64:
        raise RegistryViolation("artifact checksum must be sha256 hex (64 chars)")


# =====================================================================
# PROMOTION POLICY v1 (thresholds locked BEFORE any decision)
# =====================================================================

POLICY_VERSION = "v1"

R2V_REQUIRED = [
    EvidenceType.REPRODUCIBILITY, EvidenceType.DATA_INTEGRITY,
    EvidenceType.PIT_INTEGRITY, EvidenceType.LABEL_VALIDITY,
    EvidenceType.STATISTICAL_INFERENCE, EvidenceType.MULTIPLE_TESTING,
    EvidenceType.UNIVERSE_STABILITY, EvidenceType.TEMPORAL_STABILITY,
    EvidenceType.MODEL_FAMILY_STABILITY,
]

V2P_REQUIRED = [
    EvidenceType.REPRODUCIBILITY, EvidenceType.TEMPORAL_STABILITY,
    EvidenceType.UNIVERSE_STABILITY, EvidenceType.ECONOMIC_MATERIALITY,
    EvidenceType.PORTFOLIO_VALIDATION, EvidenceType.EXECUTION_VALIDATION,
    EvidenceType.STRESS_TESTING,
]

PROMOTION_POLICY = {
    "policy_version": POLICY_VERSION,
    "locked_before_decisions": True,
    "principles": [
        "Missing evidence => BLOCKED / NOT_EVALUATED, never PASS",
        "Any evidence record with gate_result=FAIL for a required type => gate FAIL",
        "Unresolved DEFECT (severity >= HIGH, resolved=False) => immediate gate FAIL",
        "Status changes only via immutable PromotionDecisionRecord",
    ],
    "gates": {
        "RESEARCH_TO_VALIDATED": {
            "description": "Research graduates to validated (still not economic proof)",
            "required_evidence": [e.value for e in R2V_REQUIRED],
            "additional_rules": [
                "G1 complete model identity",
                "G2 dataset/feature provenance complete",
                "G3 label contract valid",
                "G4 PIT integrity present",
                "G5 reproducibility present",
                "G6 statistical inference completed",
                "G7 multiple-testing documented",
                "G8 no unresolved CRITICAL failure",
                "G9 required universe replication completed",
                "G10 temporal evidence not dependent on one unexplained window",
                "G11 no implementation defect invalidating canonical evidence",
            ],
        },
        "VALIDATED_TO_PAPER": {
            "description": "Paper-trading eligibility (NOT satisfied by any current model)",
            "required_evidence": [e.value for e in V2P_REQUIRED],
            "additional_rules": [
                "H1 robust cross-universe evidence",
                "H2 temporal/regime stability",
                "H3 economic materiality assessment",
                "H4 portfolio construction evaluation",
                "H5 realistic transaction-cost evaluation",
                "H6 risk-engine compatibility",
                "H7 execution simulation",
                "H8 complete replayability",
                "H9 no unresolved material implementation defects",
            ],
            "status_at_phase_end": "NOT_SATISFIED_BY_ANY_MODEL (per Phase 13C)",
        },
    },
}


# =====================================================================
# REGISTRY ENGINE
# =====================================================================

class ModelRegistry:
    def __init__(self) -> None:
        self.models: dict[str, dict] = {}
        self.identity_index: dict[str, str] = {}       # identity_digest -> model_id
        self.evidence: dict[str, dict] = {}
        self.decisions: list[dict] = []
        self._decision_ids: set[str] = set()

    # -- registration --------------------------------------------------

    def register(self, model: dict, *, known_experiment_ids: Optional[set] = None) -> dict:
        validate_model_payload(model)
        mid = model["model_id"]
        if mid in self.models:
            raise RegistryViolation(f"duplicate model_id {mid}")
        pid = model.get("parent_experiment_id")
        if pid is not None:
            if known_experiment_ids is None or pid not in known_experiment_ids:
                raise RegistryViolation(
                    f"A16 violation: {mid} references unknown parent experiment {pid}")
        idg = identity_digest(model)
        if idg in self.identity_index:
            raise RegistryViolation(
                f"A8 violation: {mid} collides with "
                f"{self.identity_index[idg]} (identical scientific identity)")
        rec = dict(model)
        rec["status"] = LifecycleStatus.RESEARCH.value
        rec["status_reason"] = model.get("initial_status_reason", "Initial registration")
        rec["status_history"] = [{
            "status": LifecycleStatus.RESEARCH.value,
            "timestamp": PHASE_CLOCK, "reason": rec["status_reason"],
        }]
        rec["identity_digest"] = idg
        rec["created_at"] = PHASE_CLOCK
        rec["registered_at"] = PHASE_CLOCK
        rec["retired_at"] = None
        rec["promotion_policy_version"] = POLICY_VERSION
        rec["evidence_ids"] = []
        self.models[mid] = rec
        self.identity_index[idg] = mid
        return rec

    # -- identity integrity ---------------------------------------------

    def get(self, model_id: str) -> dict:
        if model_id not in self.models:
            raise RegistryViolation(f"unknown model {model_id}")
        return self.models[model_id]

    def verify_identity(self, model_id: str) -> bool:
        """Recompute identity digest from stored fields."""
        m = self.get(model_id)
        return identity_digest(m) == m["identity_digest"]

    # -- evidence ---------------------------------------------------------

    def attach_evidence(self, ev: dict) -> dict:
        eid = ev.get("evidence_id")
        if not eid or not eid.startswith("EVID-"):
            raise RegistryViolation("evidence_id must start with EVID-")
        if eid in self.evidence:
            raise RegistryViolation(f"duplicate evidence {eid}")
        mid = ev.get("model_id")
        if mid not in self.models:
            raise RegistryViolation(f"evidence references unknown model {mid}")
        rec = dict(ev)
        rec["generated_at"] = PHASE_CLOCK
        rec["evidence_digest"] = digest_short(rec)
        self.evidence[eid] = rec
        self.models[mid]["evidence_ids"].append(eid)
        return rec

    def model_evidence(self, model_id: str) -> list[dict]:
        m = self.get(model_id)
        out = []
        for eid in m["evidence_ids"]:
            e = dict(self.evidence[eid])
            # tamper check: stored digest must match recomputed
            stored = e.pop("evidence_digest")
            if digest_short(e) != stored:
                raise RegistryViolation(f"evidence digest mismatch for {eid}")
            e["evidence_digest"] = stored
            out.append(e)
        return out

    # -- gate evaluation ---------------------------------------------------

    def _unresolved_critical_defects(self, model_id: str) -> list[dict]:
        return [e for e in self.model_evidence(model_id)
                if e["evidence_type"] == EvidenceType.DEFECT.value
                and e.get("gate_result") == GateResult.FAIL.value
                and not e.get("resolved", False)
                and str(e.get("severity", "LOW")).upper() in ("HIGH", "CRITICAL")]

    def evaluate_gates(self, model_id: str, target: LifecycleStatus) -> tuple[dict, dict]:
        m = self.get(model_id)
        cur = LifecycleStatus(m["status"])
        details: dict[str, Any] = {}
        results: dict[str, str] = {}

        if target == LifecycleStatus.RETIRED:
            results["RETIREMENT"] = GateResult.PASS.value
            details["RETIREMENT"] = {"note": "retirement always permitted with immutable record"}
            return results, details

        if not ((cur, target) in [(LifecycleStatus.RESEARCH, LifecycleStatus.VALIDATED),
                                  (LifecycleStatus.VALIDATED, LifecycleStatus.PAPER)]):
            raise RegistryViolation(f"illegal transition {cur.value} -> {target.value}")

        gate_key = f"{cur.value}_TO_{target.value}"
        required = R2V_REQUIRED if target == LifecycleStatus.VALIDATED else V2P_REQUIRED
        ev = self.model_evidence(model_id)
        by_type: dict[str, list[dict]] = {}
        for e in ev:
            by_type.setdefault(e["evidence_type"], []).append(e)

        # G8/G11-style rule: unresolved high-severity defect fails immediately
        defects = self._unresolved_critical_defects(model_id)
        if defects:
            results[gate_key] = GateResult.FAIL.value
            details[gate_key] = {
                "reason": "unresolved critical defect(s)",
                "defects": [d["evidence_id"] for d in defects],
            }
            return results, details

        gate_result = GateResult.PASS
        det = {"evidence_present": {}, "missing": [], "failing": [], "not_evaluated": []}
        for rt in required:
            recs = by_type.get(rt.value, [])
            det["evidence_present"][rt.value] = bool(recs)
            if not recs:
                det["missing"].append(rt.value)
                if gate_result == GateResult.PASS:
                    gate_result = GateResult.BLOCKED
                continue
            statuses = [r["gate_result"] for r in recs]
            if GateResult.FAIL.value in statuses:
                det["failing"].append(rt.value)
                gate_result = GateResult.FAIL
            elif any(s == GateResult.NOT_EVALUATED.value for s in statuses) and gate_result == GateResult.PASS:
                det["not_evaluated"].append(rt.value)
                gate_result = GateResult.BLOCKED
        det["policy_note"] = "missing evidence is BLOCKED, never PASS (A18)"
        results[gate_key] = gate_result.value
        details[gate_key] = det
        return results, details

    # -- status mutation guard -----------------------------------------------

    def set_status(self, model_id: str, new_status: str) -> None:
        """Public setter deliberately forbidden: status changes ONLY through
        decide(), so every transition carries a decision record (A14)."""
        raise RegistryViolation(
            "A14: direct status mutation forbidden; every transition must go "
            "through decide() to create an immutable PromotionDecisionRecord")

    def decide(self, model_id: str, *, target: LifecycleStatus, action: PromotionAction,
               decision_id: str, rationale: str) -> dict:
        if decision_id in self._decision_ids:
            raise RegistryViolation(f"duplicate decision id {decision_id}")
        m = self.get(model_id)
        cur = LifecycleStatus(m["status"])

        gate_results, gate_details = ({}, {})
        if action != PromotionAction.RETAIN:
            gate_results, gate_details = self.evaluate_gates(model_id, target)

        new_status = cur.value
        history_entry = None
        if action == PromotionAction.PROMOTE:
            ok = all(v == GateResult.PASS.value for v in gate_results.values())
            if not ok:
                raise RegistryViolation("PROMOTE attempted with non-PASS gates")
            new_status = target.value
            history_entry = {"from": cur.value, "to": new_status,
                             "timestamp": PHASE_CLOCK, "decision_id": decision_id}
        elif action == PromotionAction.RETIRE:
            new_status = LifecycleStatus.RETIRED.value
            history_entry = {"from": cur.value, "to": new_status,
                             "timestamp": PHASE_CLOCK, "decision_id": decision_id}
            m["retired_at"] = PHASE_CLOCK
        elif action == PromotionAction.DEMOTE:
            new_status = LifecycleStatus.RESEARCH.value
            history_entry = {"from": cur.value, "to": new_status,
                             "timestamp": PHASE_CLOCK, "decision_id": decision_id}

        blocking = []
        for gk, gv in gate_results.items():
            if gv != GateResult.PASS.value:
                blocking.append(f"{gk}={gv}")

        ev_refs = [e["evidence_id"] for e in self.model_evidence(model_id)]
        ev_digs = [self.evidence[eid]["evidence_digest"] for eid in ev_refs]

        dec = {
            "decision_id": decision_id,
            "model_id": model_id,
            "model_version": m["model_version"],
            "from_status": cur.value,
            "to_status": new_status,
            "action": action.value,
            "promotion_policy_version": POLICY_VERSION,
            "gates_evaluated": gate_results,
            "gate_details": gate_details,
            "blocking_conditions": blocking,
            "rationale": rationale,
            "evidence_references": ev_refs,
            "evidence_digests": ev_digs,
            "decision_timestamp": PHASE_CLOCK,
        }
        dec["decision_digest"] = digest_short(dec)
        self.decisions.append(dec)
        self._decision_ids.add(decision_id)

        if history_entry:
            m["status"] = new_status
            m["status_reason"] = rationale
            m["status_history"].append(history_entry)
        return dec

    def verify_decision_chain(self) -> list[str]:
        """Detect any mutation of recorded decisions (A15)."""
        problems = []
        for i, d in enumerate(self.decisions):
            stored = d["decision_digest"]
            probe = dict(d)
            probe.pop("decision_digest")
            if digest_short(probe) != stored:
                problems.append(f"decision[{i}] {d['decision_id']} digest mismatch")
        # monotone append-only check
        if len({d["decision_id"] for d in self.decisions}) != len(self.decisions):
            problems.append("duplicate decision ids in chain")
        return problems

    # -- retirement guard -------------------------------------------------------

    def ensure_not_promotable(self, model_id: str) -> bool:
        """RETIRED models can never be promoted (audit requirement)."""
        m = self.get(model_id)
        if LifecycleStatus(m["status"]) == LifecycleStatus.RETIRED:
            try:
                self.decide(model_id, target=LifecycleStatus.PAPER,
                            action=PromotionAction.PROMOTE,
                            decision_id=f"PROM-9{len(self.decisions):05d}",
                            rationale="illegal retire->paper attempt")
                return False
            except RegistryViolation:
                return True
        return True

    # -- replay -----------------------------------------------------------------

    def resolve_replay(self, model_id: str, mode: ReplayMode = ReplayMode.IDENTITY_REPLAY,
                       bench_dir: Optional[Path] = None) -> dict:
        m = self.get(model_id)
        spec = {
            "model_id": model_id,
            "model_version": m["model_version"],
            "replay_mode": mode.value,
            "resolved": identity_of(m),
            "artifacts": m.get("artifacts", {}),
            "referenced_evidence": [],
        }
        checksum_checks: dict[str, Any] = {}
        for eid in m.get("source_artifact_files", []):
            name, rel = eid["name"], eid["path"]
            p = (bench_dir / rel) if bench_dir else Path(rel)
            entry: dict[str, Any] = {"path": rel, "exists": p.exists()}
            if p.exists():
                actual = file_sha256(p)
                entry["checksum_match"] = (actual == eid["sha256"])
            else:
                entry["checksum_match"] = None
            checksum_checks[name] = entry
        spec["artifact_verification"] = checksum_checks
        spec["resolution_complete"] = all(
            c["exists"] and c["checksum_match"] for c in checksum_checks.values()
        ) if checksum_checks else True
        spec["missing_components"] = [k for k, v in checksum_checks.items()
                                      if not (v["exists"] and v["checksum_match"])]
        spec["numerical_replay_performed"] = False
        if mode == ReplayMode.FULL_NUMERICAL_REPLAY:
            raise RegistryViolation(
                "FULL_NUMERICAL_REPLAY not implemented for historical artifacts; "
                "only IDENTITY_REPLAY is claimed")
        return spec

    # -- predictions --------------------------------------------------------------

    def validate_prediction(self, pred: dict) -> tuple[bool, str]:
        """Prediction provenance validation (schema-level interface test)."""
        for k in ("prediction_id", "timestamp", "symbol", "model_id", "model_version",
                  "dataset_snapshot_id", "feature_snapshot_id", "feature_version_digest",
                  "prediction_value", "provenance_digest"):
            if k not in pred:
                return False, f"missing field {k}"
        if pred["model_id"] not in self.models:
            return False, "A10: prediction references unknown model"
        m = self.get(pred["model_id"])
        if LifecycleStatus(m["status"]) == LifecycleStatus.RETIRED:
            return False, "A11: prediction references RETIRED model"
        if m["model_version"] != pred["model_version"]:
            return False, "model version mismatch"
        if not pred.get("feature_snapshot_id") or not pred.get("feature_version_digest"):
            return False, "prediction lacks feature provenance"
        return True, "ok"

    # -- manifest --------------------------------------------------------------------

    def status_distribution(self) -> dict:
        dist: dict[str, int] = {}
        for m in self.models.values():
            dist[m["status"]] = dist.get(m["status"], 0) + 1
        return dist

    def manifest(self) -> dict:
        man = {
            "registry_version": "phase14-v1",
            "generated_at": PHASE_CLOCK,
            "model_count": len(self.models),
            "evidence_count": len(self.evidence),
            "promotion_decisions_count": len(self.decisions),
            "status_distribution": self.status_distribution(),
            "models": {mid: {
                "model_version": m["model_version"],
                "status": m["status"],
                "identity_digest": m["identity_digest"],
                "candidate_id": m.get("candidate_id"),
                "evidence_count": len(m["evidence_ids"]),
            } for mid, m in sorted(self.models.items())},
        }
        man["manifest_digest"] = digest_short(man)
        return man


__all__ = [
    "LifecycleStatus", "PromotionAction", "GateResult", "EvidenceType", "ReplayMode",
    "RegistryViolation", "PHASE_CLOCK", "canonical", "digest_full", "digest_short",
    "file_sha256", "IDENTITY_FIELDS", "identity_of", "identity_digest",
    "validate_model_payload", "PROMOTION_POLICY", "POLICY_VERSION",
    "ModelRegistry",
]
