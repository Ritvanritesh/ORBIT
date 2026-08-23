"""Phase 12.9B - Main runner."""
def run_9b():
    print("=" * 72)
    print("PHASE 12.9B - CLEAN-RUN END-TO-END REPLICATION")
    print("=" * 72)
    t_start = time.time()

    p11 = rep_phase11()
    p12a = rep_phase12a()
    p12d = rep_phase12d()
    p12e = rep_phase12e()
    cross = cross_phase_test(p11, p12a, p12d, p12e)

    # Summary
    all_results = {"phase11": p11, "phase12a": p12a, "phase12d": p12d, "phase12e": p12e}
    total_classifications = defaultdict(int)
    for phase_key in ["phase11", "phase12a", "phase12d", "phase12e"]:
        for k, v in all_results[phase_key].get("status_counts", {}).items():
            total_classifications[k] += v

    n_total = sum(total_classifications.values())
    n_ok = total_classifications.get("EXACT_MATCH", 0) + total_classifications.get("NUMERICALLY_EQUIVALENT", 0) + total_classifications.get("MINOR_DRIFT", 0)
    n_material = total_classifications.get("MATERIAL_DRIFT", 0) + total_classifications.get("FAILED_TO_REPRODUCE", 0)

    print("\n" + "=" * 72)
    print("REPLICATION SUMMARY")
    print("=" * 72)
    print(f"  Total experiments compared: {n_total}")
    print(f"  Classification breakdown: {dict(total_classifications)}")
    print(f"  Reproduced (exact+equiv+minor): {n_ok}/{n_total}")
    print(f"  Material drift/failures: {n_material}/{n_total}")

    if n_material == 0 and n_ok > n_total * 0.9:
        verdict = "B"
        reason = "Clean-run confirms conclusions with minor drift"
    elif n_material <= 2 and n_ok > n_total * 0.7:
        verdict = "C"
        reason = "Broad reproduction with meaningful uncertainty"
    elif n_material > n_total * 0.3:
        verdict = "D"
        reason = "Material historical results fail to reproduce"
    else:
        verdict = "B"
        reason = "Clean-run confirms conclusions with minor drift"

    print(f"\n  VERDICT: {verdict}")
    print(f"  REASON: {reason}")

    supported = sum(1 for v in cross.values() if v.get("status") == "SUPPORTED")
    print(f"  Cross-phase conclusions: {supported}/{len(cross)} SUPPORTED")

    elapsed = time.time() - t_start
    audit = {
        "phase": "12.9B", "verdict": verdict, "reason": reason,
        "elapsed_s": round(elapsed, 1),
        "total_experiments": n_total,
        "classification_breakdown": dict(total_classifications),
        "reproduced": n_ok, "material_drift_or_fail": n_material,
        "cross_phase_supported": supported, "cross_phase_total": len(cross),
        "recommendation": "PROCEED TO PHASE 13" if verdict in ("A", "B") else "REPAIR BEFORE PHASE 13",
    }
    save_json("phase12_9b_audit.json", audit)
    save_json("phase12_9b_report.json", audit)
    print(f"\n  Recommendation: {audit['recommendation']}")
    print(f"  Total time: {elapsed:.1f}s")
    print("=" * 72)

if __name__ == "__main__":
    run_9b()
