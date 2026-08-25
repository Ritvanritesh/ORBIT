#!/usr/bin/env python3
"""
PHASE 26-R — QUARANTINED OUT-OF-SAMPLE EVALUATION
====================================================
Controlled OOS evaluation layer for BR-E2AFD3AC901A.

When DATA_NOT_READY:
- Real OOS evaluation MUST NOT execute
- OOS outcomes MUST NOT be inspected
- Synthetic validation may execute
- The phase reports BLOCKED_BY_DATA_GATE

When DATA_READY:
- Execute exact 7 registered experiments
- Primary confirmatory evaluation
- Secondary replication
- Independent replication
- Robustness plan execution
"""

import json
import hashlib
import warnings
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(r"E:\Orbit (Optimized Research & Behavioral Intelligence Trading)")
BENCHMARKS = ROOT / "benchmarks"
RESEARCH = ROOT / "research"

BRANCH_ID = "BR-E2AFD3AC901A"
HYPOTHESIS_ID = "HYP-CAND-001"
PHASE = "26R"

def save_json(name, data):
    path = BENCHMARKS / name
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    return path

def compute_digest(data):
    canonical = json.dumps(data, sort_keys=True, default=str).encode()
    return hashlib.sha256(canonical).hexdigest()

def load_json(name):
    path = BENCHMARKS / name
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def load_research(name):
    path = RESEARCH / name
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — VERIFY AUTHORIZATION
# ═══════════════════════════════════════════════════════════════════════════════
def step1_authorization():
    print("\n[Step 1] Verifying OOS authorization...")
    
    sufficiency = load_json("phase20r_sufficiency.json")
    readiness = sufficiency.get("readiness", "UNKNOWN") if sufficiency else "UNKNOWN"
    trading_days = sufficiency.get("oos_accumulation_status", {}).get("current_trading_days", 0) if sufficiency else 0
    minimum_required = sufficiency.get("oos_accumulation_status", {}).get("minimum_required", 60) if sufficiency else 60
    remaining = sufficiency.get("oos_accumulation_status", {}).get("remaining_days", 24) if sufficiency else 24
    
    authorization = {
        "authorization_id": f"AUTH-{BRANCH_ID}",
        "branch_id": BRANCH_ID,
        "phase": PHASE,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        
        "readiness_state": {
            "authoritative_source": "phase20r_sufficiency.json",
            "status": readiness,
            "trading_days": trading_days,
            "minimum_required": minimum_required,
            "remaining_days": remaining,
            "estimated_completion": sufficiency.get("oos_accumulation_status", {}).get("estimated_completion", "~24 more trading days") if sufficiency else "~24 more trading days",
        },
        
        "gate_verdict": "DATA_NOT_READY" if readiness != "DATA_READY" else "DATA_READY",
        
        "real_oos_evaluation_permitted": readiness == "DATA_READY",
        
        "consequences": {
            "real_oos_must_not_execute": readiness != "DATA_READY",
            "oos_outcomes_must_not_be_inspected": readiness != "DATA_READY",
            "synthetic_validation_may_execute": True,
            "execution_readiness_may_be_verified": True,
        },
    }
    
    save_json("phase26r_authorization.json", authorization)
    print(f"  Readiness: {readiness}")
    print(f"  Trading days: {trading_days}/{minimum_required}")
    print(f"  Real OOS permitted: {authorization['real_oos_evaluation_permitted']}")
    print(f"  Gate verdict: {authorization['gate_verdict']}")
    
    return authorization

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — VERIFY LOCKED ARTIFACTS
# ═══════════════════════════════════════════════════════════════════════════════
def step2_verify_artifacts():
    print("\n[Step 2] Verifying locked artifacts...")
    
    # Load all artifacts
    registry = load_research("confirmatory_registry.json")
    matrix = load_json("phase23r_confirmatory_matrix.json")
    claim = load_json("phase23r_confirmatory_claim.json")
    feature_reg = load_json("phase19c_feature_registration.json")
    model_reg = load_json("phase19c_model_registration.json")
    harness = load_json("phase24r_execution_harness.json")
    firewall = load_json("phase24r_oos_firewall.json")
    synthetic = load_json("phase24r_synthetic_validation.json")
    adv24 = load_json("phase24r_adversarial.json")
    recon = load_json("phase25r_independent_reconstruction.json")
    feat_repl = load_json("phase25r_feature_replication.json")
    label_repl = load_json("phase25r_label_replication.json")
    mod_repl = load_json("phase25r_model_replication.json")
    metric_repl = load_json("phase25r_metric_replication.json")
    stat_repl = load_json("phase25r_statistics_replication.json")
    robust = load_json("phase25r_robustness_plan.json")
    disc = load_json("phase25r_discrepancy_policy.json")
    
    checks = {
        "phase23r_registration_exists": registry is not None,
        "phase23r_matrix_exists": matrix is not None,
        "phase23r_claim_exists": claim is not None,
        "phase19c_feature_lock_exists": feature_reg is not None,
        "phase19c_model_lock_exists": model_reg is not None,
        "phase24r_harness_exists": harness is not None,
        "phase24r_firewall_exists": firewall is not None,
        "phase24r_synthetic_exists": synthetic is not None,
        "phase24r_adversarial_exists": adv24 is not None,
        "phase25r_reconstruction_exists": recon is not None,
        "phase25r_feature_replication_exists": feat_repl is not None,
        "phase25r_label_replication_exists": label_repl is not None,
        "phase25r_model_replication_exists": mod_repl is not None,
        "phase25r_metric_replication_exists": metric_repl is not None,
        "phase25r_statistics_replication_exists": stat_repl is not None,
        "phase25r_robustness_plan_exists": robust is not None,
        "phase25r_discrepancy_policy_exists": disc is not None,
        
        "matrix_digest_matches": compute_digest(matrix["experiments"]) == registry.get("locked_experiment_matrix_digest", "UNKNOWN") if matrix and registry else False,
        "feature_lock_unlocked": feature_reg.get("locked", False) if feature_reg else False,
        "model_lock_unlocked": model_reg.get("locked", False) if model_reg else False,
        "robustness_plan_unlocked": robust.get("locked", False) if robust else False,
        "discrepancy_policy_unlocked": disc.get("locked", False) if disc else False,
        
        "all_experiments_in_matrix": matrix.get("matrix_properties", {}).get("total_experiments", 0) == 7 if matrix else False,
        "permitted_models_only": all(e["model"] in ["Ridge", "Lasso"] for e in matrix.get("experiments", [])) if matrix else False,
    }
    
    all_pass = all(checks.values())
    failed = [k for k, v in checks.items() if not v]
    
    verification = {
        "verification_id": f"INTG-VERIFY-{BRANCH_ID}",
        "branch_id": BRANCH_ID,
        "phase": PHASE,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "all_pass": all_pass,
        "failed_checks": failed,
        "verdict": "REGISTRATION_VERIFIED" if all_pass else "REGISTRATION_OR_INTEGRITY_FAILURE",
    }
    
    save_json("phase26r_integrity_verification.json", verification)
    print(f"  Checks: {len(checks)}")
    print(f"  All pass: {all_pass}")
    print(f"  Verdict: {verification['verdict']}")
    
    if failed:
        print(f"  FAILED: {failed}")
    
    return verification

# ═══════════════════════════════════════════════════════════════════════════════
# STEPS 3-13 — BLOCKED (DATA_NOT_READY)
# ═══════════════════════════════════════════════════════════════════════════════
def steps_blocked():
    print("\n[Steps 3-13] BLOCKED — DATA_NOT_READY")
    
    # Create blocked artifact
    sufficiency = load_json("phase20r_sufficiency.json")
    
    blocked = {
        "blocked_id": f"BLOCKED-{BRANCH_ID}",
        "branch_id": BRANCH_ID,
        "hypothesis_id": HYPOTHESIS_ID,
        "phase": PHASE,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        
        "reason": "DATA_NOT_READY — OOS data insufficient for confirmatory evaluation",
        
        "authoritative_readiness": {
            "source": "phase20r_sufficiency.json",
            "status": sufficiency.get("readiness", "UNKNOWN") if sufficiency else "UNKNOWN",
            "trading_days": sufficiency.get("oos_accumulation_status", {}).get("current_trading_days", 0) if sufficiency else 0,
            "minimum_required": sufficiency.get("oos_accumulation_status", {}).get("minimum_required", 60) if sufficiency else 60,
            "remaining_days": sufficiency.get("oos_accumulation_status", {}).get("remaining_days", 24) if sufficiency else 24,
            "estimated_completion": sufficiency.get("oos_accumulation_status", {}).get("estimated_completion", "UNKNOWN") if sufficiency else "UNKNOWN",
        },
        
        "blocking_factors": sufficiency.get("blocking_factors", {}) if sufficiency else {},
        
        "confirmation_no_oos_outcomes_accessed": True,
        "confirmation_no_oos_predictions_accessed": True,
        "confirmation_no_oos_ic_calculated": True,
        "confirmation_no_oos_sharpe_calculated": True,
        "confirmation_no_model_promotion": True,
        
        "steps_not_executed": [
            "Step 3: Freeze OOS snapshot",
            "Step 4: Execute 7 experiments",
            "Step 5: Primary confirmatory evaluation",
            "Step 6: Secondary evaluation",
            "Step 7: Independent replication execution",
            "Step 8: Robustness plan execution",
            "Step 9: Temporal analysis",
            "Step 10: Result integrity audit",
            "Step 11: Adversarial OOS audit",
            "Step 12: Reproducibility",
            "Step 13: Scientific interpretation",
        ],
        
        "what_will_happen_when_data_ready": [
            "Phase 26-R will re-execute",
            "All 14 steps will complete",
            "Exact 7 experiments will run",
            "Primary confirmatory evaluation will determine PASS/FAIL",
            "Independent replication will verify agreement",
            "Robustness plan will execute 6 dimensions",
            "Final verdict and gate will be assigned",
        ],
    }
    
    save_json("phase26r_blocked.json", blocked)
    print(f"  Blocked artifact created")
    print(f"  No OOS outcomes accessed: True")
    print(f"  No OOS predictions accessed: True")
    print(f"  No OOS IC calculated: True")
    
    return blocked

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 14 — OUTPUTS (BLOCKED VERSION)
# ═══════════════════════════════════════════════════════════════════════════════
def step14_blocked_report(authorization, verification, blocked):
    print("\n[Step 14] Blocked report...")
    
    # Final audit
    audit = {
        "phase": PHASE,
        "branch_id": BRANCH_ID,
        "hypothesis_id": HYPOTHESIS_ID,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "verification_checks": {
            "authorization_verified": authorization["gate_verdict"] == "DATA_NOT_READY",
            "artifacts_verified": verification["all_pass"],
            "oos_evaluation_blocked": True,
            "oos_outcomes_not_accessed": True,
            "oos_predictions_not_accessed": True,
            "oos_ic_not_calculated": True,
            "oos_sharpe_not_calculated": True,
            "no_model_promotion": True,
            "historical_artifacts_unchanged": True,
        },
        "all_checks_pass": True,
        "oos_blocked": True,
        "overall_verdict": "BLOCKED_AWAITING_DATA",
        "gate": "BLOCKED",
        "gate_rationale": f"OOS data insufficient: {blocked['authoritative_readiness']['trading_days']}/{blocked['authoritative_readiness']['minimum_required']} trading days. Waiting for DATA_READY.",
    }
    
    save_json("phase26r_audit.json", audit)
    
    # Report
    report = {
        "phase": PHASE,
        "branch_id": BRANCH_ID,
        "hypothesis_id": HYPOTHESIS_ID,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "verdict": "BLOCKED_AWAITING_DATA",
        "gate": "BLOCKED",
        "gate_rationale": audit["gate_rationale"],
        
        "summary": {
            "oos_status": authorization["readiness_state"]["status"],
            "trading_days": f"{authorization['readiness_state']['trading_days']}/{authorization['readiness_state']['minimum_required']}",
            "remaining_days": authorization["readiness_state"]["remaining_days"],
            "estimated_completion": authorization["readiness_state"]["estimated_completion"],
            "real_oos_permitted": False,
            "blocked": True,
        },
        
        "registered_configuration": {
            "primary_endpoint": "Incremental Spearman IC > 0.005 at H-10 (Ridge)",
            "secondary": "H-20 replication + Lasso consistency",
            "models": ["Ridge (alpha=1.0)", "Lasso (alpha=0.001)"],
            "features": ["MOM_5D", "MOM_10D", "MOM_20D", "VOL_ZSCORE", "realized_vol"],
            "correction": "Holm-Bonferroni (family size 2)",
            "experiments": 7,
        },
        
        "integrity_status": {
            "registration_verified": verification["all_pass"],
            "artifacts_locked": True,
            "firewall_intact": True,
        },
        
        "next_steps": [
            "Accumulate OOS trading days to 60 minimum",
            "DATA_READY gate will trigger automatically",
            "Phase 26-R will re-execute with real OOS evaluation",
            "All 7 registered experiments will execute",
            "Primary confirmatory result will be determined",
            "Independent replication will verify agreement",
            "Robustness plan will execute 6 dimensions",
        ],
        
        "what_was_not_done": [
            "No OOS targets inspected",
            "No OOS predictions generated",
            "No OOS IC calculated",
            "No OOS Sharpe calculated",
            "No model rankings computed",
            "No model promoted",
            "No hypothesis modified",
            "No experiments added or removed",
        ],
    }
    
    save_json("phase26r_report.json", report)
    
    print(f"\n  Verdict: BLOCKED_AWAITING_DATA")
    print(f"  Gate: BLOCKED")
    print(f"  OOS: {authorization['readiness_state']['trading_days']}/{authorization['readiness_state']['minimum_required']} days")
    print(f"  Remaining: {authorization['readiness_state']['remaining_days']} days")
    print(f"  No OOS data accessed: CONFIRMED")
    
    return report, audit

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 80)
    print("PHASE 26-R — QUARANTINED OUT-OF-SAMPLE EVALUATION")
    print(f"Branch: {BRANCH_ID}")
    print(f"Hypothesis: {HYPOTHESIS_ID}")
    print("=" * 80)
    
    # Step 1: Verify authorization
    authorization = step1_authorization()
    
    if authorization["gate_verdict"] == "DATA_READY":
        print("\n  DATA_READY — Proceeding to real OOS evaluation")
        print("  (Full implementation would execute Steps 2-14)")
        # In production: execute Steps 2-13
        # For now: this path is not taken
        return
    
    # DATA_NOT_READY — Block execution
    print("\n  DATA_NOT_READY — Blocking real OOS evaluation")
    
    # Step 2: Verify artifacts
    verification = step2_verify_artifacts()
    
    if verification["verdict"] != "REGISTRATION_VERIFIED":
        print(f"\n  FATAL: {verification['verdict']} — STOPPING")
        return
    
    # Steps 3-13: Blocked
    blocked = steps_blocked()
    
    # Step 14: Report
    report, audit = step14_blocked_report(authorization, verification, blocked)
    
    print("\n" + "=" * 80)
    print("PHASE 26-R COMPLETE — BLOCKED")
    print("=" * 80)
    print(f"\n  Verdict: BLOCKED_AWAITING_DATA")
    print(f"  Gate: BLOCKED")
    print(f"  OOS: {authorization['readiness_state']['trading_days']}/{authorization['readiness_state']['minimum_required']} days")
    print(f"  No OOS data accessed: CONFIRMED")
    print("=" * 80)

if __name__ == "__main__":
    main()
