"""Phase 11 independent audit layer.

Verifies that the inference analysis was conducted according to the
locked plan, all sources are intact, no cherry-picking occurred, and
the results are reproducible.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
AUDIT_JSON = _REPO_ROOT / "benchmarks" / "phase11_audit_results.json"


def _check(name: str, passed: bool, evidence: str) -> dict[str, Any]:
    return {"check": name, "status": "PASS" if passed else "FAIL", "evidence": evidence}


def _file_checksum(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def run_phase11_audit(
    *,
    plan: dict[str, Any] | None = None,
    analysis: dict[str, Any] | None = None,
    synthetic_validation_passed: bool = False,
) -> list[dict[str, Any]]:
    """Run the full Phase 11 independent audit.

    Returns a list of check dicts with {check, status, evidence}.
    """
    checks: list[dict[str, Any]] = []

    # 1. Inference plan exists and is digest-verified
    try:
        from orbit.ml.phase11_plan import load_plan, phase11_plan_digest
        if plan is None:
            plan = load_plan()
        # Verify digest by recomputing from the plan payload (excluding plan_digest field)
        stored_digest = plan.get("plan_digest")
        # Recompute: strip plan_digest and recompute
        import json as _json
        payload_for_digest = {k: v for k, v in plan.items() if k != "plan_digest"}
        raw = _json.dumps(payload_for_digest, sort_keys=True, default=str, separators=(",", ":"))
        import hashlib as _hl
        computed_digest = _hl.sha256(raw.encode("utf-8")).hexdigest()
        digest_ok = stored_digest is not None and stored_digest == computed_digest
        # Verify family membership
        family = plan.get("hypothesis_families", {}).get("phase10_grid", {})
        members = family.get("members", [])
        family_ok = len(members) == 52
        plan_ok = digest_ok and family_ok
        checks.append(_check(
            "inference_plan_digest",
            plan_ok,
            f"plan digest {stored_digest[:16] if stored_digest else 'N/A'}... "
            f"verified={digest_ok}; family_members={len(members)}",
        ))
    except Exception as exc:
        checks.append(_check("inference_plan_digest", False, str(exc)))
        return checks  # Cannot proceed without a valid plan

    # 2. Source experiment inventory matches the locked plan
    expected_family = plan.get("hypothesis_families", {}).get("phase10_grid", {})
    expected_members = set(expected_family.get("members", []))
    if analysis is not None:
        # Check that all 52 Phase 10 experiments are present
        phase10_ids = set()
        for r in analysis.get("inference_results", []):
            for eid in r.source_experiment_ids:
                if eid.startswith("EXP-10"):
                    phase10_ids.add(eid)
        checks.append(_check(
            "source_experiment_inventory",
            expected_members == phase10_ids,
            f"expected {len(expected_members)} experiments, found {len(phase10_ids)} "
            f"({len(expected_members - phase10_ids)} missing, "
            f"{len(phase10_ids - expected_members)} extra)",
        ))
    else:
        checks.append(_check(
            "source_experiment_inventory",
            False,
            "no analysis results provided",
        ))

    # 3. No hidden experiment exclusion
    if analysis is not None:
        all_exp_ids = set()
        for r in analysis.get("inference_results", []):
            for eid in r.source_experiment_ids:
                all_exp_ids.add(eid)
        missing = expected_members - all_exp_ids
        checks.append(_check(
            "no_hidden_exclusion",
            len(missing) == 0,
            f"all {len(expected_members)} family members present"
            if not missing
            else f"missing experiments: {sorted(missing)[:5]}",
        ))

    # 4. Source artifacts exist
    p9_parquet = _REPO_ROOT / "benchmarks" / "phase9_ml_benchmark.parquet"
    p10_parquet = _REPO_ROOT / "benchmarks" / "phase10_feature_research.parquet"
    p10_plan = _REPO_ROOT / "benchmarks" / "phase10_plan.json"
    checks.append(_check(
        "source_artifacts_exist",
        p9_parquet.exists() and p10_parquet.exists() and p10_plan.exists(),
        f"phase9={p9_parquet.exists()}, phase10={p10_parquet.exists()}, "
        f"plan={p10_plan.exists()}",
    ))

    # 5. Artifact checksums match
    if analysis is not None:
        p9_ck = analysis.get("phase9_checksum", "MISSING")
        p10_ck = analysis.get("phase10_checksum", "MISSING")
        actual_p9 = _file_checksum(p9_parquet) if p9_parquet.exists() else "MISSING"
        actual_p10 = _file_checksum(p10_parquet) if p10_parquet.exists() else "MISSING"
        checks.append(_check(
            "artifact_checksums_match",
            p9_ck == actual_p9 and p10_ck == actual_p10,
            f"phase9: stored={p9_ck[:16]}... actual={actual_p9[:16]}...; "
            f"phase10: stored={p10_ck[:16]}... actual={actual_p10[:16]}...",
        ))

    # 6. Seed lock holds
    if analysis is not None:
        seed = plan.get("seed", 42)
        all_seeds_ok = all(
            r.seed == seed
            for r in analysis.get("inference_results", [])
        )
        checks.append(_check(
            "seed_lock",
            all_seeds_ok,
            f"expected seed {seed}, all results use seed {seed}"
            if all_seeds_ok
            else f"seed mismatch detected",
        ))

    # 7. Confidence level matches plan
    if analysis is not None:
        expected_cl = plan.get("confidence_level", 0.95)
        all_cl_ok = all(
            r.ci.confidence_level == expected_cl
            for r in analysis.get("inference_results", [])
        )
        checks.append(_check(
            "confidence_level_lock",
            all_cl_ok,
            f"expected CL {expected_cl}, all results match"
            if all_cl_ok
            else f"confidence level mismatch",
        ))

    # 8. Multiple-comparison family is complete
    if analysis is not None:
        mt = analysis.get("multiple_testing")
        if mt is not None:
            family = mt.get("family", {})
            family_members = set(family.get("members", []))
            checks.append(_check(
                "multiple_comparison_family_complete",
                family_members == expected_members,
                f"family has {len(family_members)} members, expected {len(expected_members)}",
            ))
        else:
            checks.append(_check(
                "multiple_comparison_family_complete",
                False,
                "no multiple-testing analysis produced",
            ))

    # 9. No result is from unsupported metric/test combination
    if analysis is not None:
        supported_metrics = {
            "oos_ic", "rank_ic", "hit_rate", "mse", "brier",
            "ece", "after_cost_total_return", "turnover", "total_costs",
        }
        unsupported = [
            r.metric for r in analysis.get("inference_results", [])
            if r.metric not in supported_metrics
        ]
        checks.append(_check(
            "supported_metric_test_combination",
            len(unsupported) == 0,
            f"all {len(analysis.get('inference_results', []))} results use supported metrics"
            if not unsupported
            else f"unsupported metrics: {set(unsupported)}",
        ))

    # 10. Assumptions are present in every result
    if analysis is not None:
        missing_assumptions = [
            r.inference_result_id
            for r in analysis.get("inference_results", [])
            if not r.ci.assumptions
        ]
        checks.append(_check(
            "assumptions_present",
            len(missing_assumptions) == 0,
            f"all results have stated assumptions"
            if not missing_assumptions
            else f"{len(missing_assumptions)} results missing assumptions",
        ))

    # 11. Statistical and economic conclusions are stored separately
    # (structural check: InferenceResult has both ci and effect_size)
    if analysis is not None:
        has_both = all(
            r.ci is not None and r.effect_size is not None
            for r in analysis.get("inference_results", [])
        )
        checks.append(_check(
            "stat_econ_separation",
            has_both,
            "all results have both CI (statistical) and effect_size (economic)",
        ))

    # 12. Phase 9/10 artifacts remain unchanged
    # Compare checksums recorded at analysis time vs current
    if analysis is not None:
        p9_ck_stored = analysis.get("phase9_checksum")
        p10_ck_stored = analysis.get("phase10_checksum")
        if p9_ck_stored and p10_ck_stored:
            p9_now = _file_checksum(p9_parquet) if p9_parquet.exists() else "MISSING"
            p10_now = _file_checksum(p10_parquet) if p10_parquet.exists() else "MISSING"
            checks.append(_check(
                "artifacts_unchanged",
                p9_ck_stored == p9_now and p10_ck_stored == p10_now,
                "Phase 9/10 parquet checksums unchanged since analysis",
            ))

    # 13. Synthetic validation passed before real-data interpretation
    checks.append(_check(
        "synthetic_validation_passed",
        synthetic_validation_passed,
        "synthetic validation must pass before real-data interpretation",
    ))

    # 14. Plan digest matches
    plan_digest_stored = plan.get("plan_digest")
    if plan_digest_stored:
        import json as _json2
        import hashlib as _hl2
        payload_for_digest = {k: v for k, v in plan.items() if k != "plan_digest"}
        raw = _json2.dumps(payload_for_digest, sort_keys=True, default=str, separators=(",", ":"))
        computed_digest = _hl2.sha256(raw.encode("utf-8")).hexdigest()
        checks.append(_check(
            "plan_digest_match",
            plan_digest_stored == computed_digest,
            f"stored={plan_digest_stored[:16]}... computed={computed_digest[:16]}...",
        ))

    # 15. Inference results have line IDs
    if analysis is not None:
        all_have_ids = all(
            r.inference_result_id and r.inference_result_id.startswith("INF-")
            for r in analysis.get("inference_results", [])
        )
        checks.append(_check(
            "inference_result_ids_valid",
            all_have_ids,
            f"{len(analysis.get('inference_results', []))} results all have valid IDs"
            if all_have_ids
            else "some results missing valid inference IDs",
        ))

    return checks


def audit_summary(checks: list[dict[str, Any]]) -> dict[str, Any]:
    failed = [c for c in checks if c["status"] == "FAIL"]
    return {
        "checks": len(checks),
        "passed": len(checks) - len(failed),
        "failed": len(failed),
        "blocked": bool(failed),
        "failed_checks": [c["check"] for c in failed],
    }


def persist_audit(checks: list[dict[str, Any]], analysis: dict[str, Any] | None = None) -> Path:
    """Write permanent audit results."""
    AUDIT_JSON.parent.mkdir(parents=True, exist_ok=True)
    summary = audit_summary(checks)
    payload = {
        "phase": 11,
        "summary": summary,
        "checks": checks,
        "analysis_summary": {
            "plan_digest": analysis.get("plan_digest") if analysis else None,
            "n_inference_results": analysis.get("n_inference_results") if analysis else 0,
            "n_phase9_experiments": analysis.get("n_phase9_experiments") if analysis else 0,
            "n_phase10_experiments": analysis.get("n_phase10_experiments") if analysis else 0,
        },
    }
    AUDIT_JSON.write_text(
        json.dumps(payload, sort_keys=True, indent=2, default=str),
        encoding="utf-8",
    )
    return AUDIT_JSON


__all__ = [
    "AUDIT_JSON",
    "run_phase11_audit",
    "audit_summary",
    "persist_audit",
]
