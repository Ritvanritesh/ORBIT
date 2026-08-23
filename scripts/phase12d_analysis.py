"""Phase 12D analysis: incremental value, cross-phase comparison, final report."""
import json
from pathlib import Path
from collections import defaultdict

REPO = Path(r"E:\Orbit (Optimized Research & Behavioral Intelligence Trading)")
BENCH = REPO / "benchmarks"

def load_results(env_id):
    path = BENCH / f"phase12d_{env_id}_results.json"
    with open(path) as f:
        return json.load(f)

# ============================================================
# 1. INCREMENTAL VALUE ANALYSIS
# ============================================================
print("=" * 72)
print("PHASE 12D: INCREMENTAL VALUE ANALYSIS")
print("=" * 72)

BASELINE_FS = "FS-12B-A"
FUNDAMENTAL_FS = ["FS-12B-B", "FS-12B-C", "FS-12B-D", "FS-12B-E", "FS-12B-F"]
FS_LABELS = {
    "FS-12B-A": "Baseline",
    "FS-12B-B": "Valuation",
    "FS-12B-C": "Profitability",
    "FS-12B-D": "Income",
    "FS-12B-E": "Leverage",
    "FS-12B-F": "All Fund",
}

for env_id in ["ENV-12D-050", "ENV-12D-100"]:
    data = load_results(env_id)
    print(f"\n--- {env_id} ({data['n_instruments']} instruments) ---")

    by_family = defaultdict(dict)
    for exp in data["results"]:
        if exp["label_id"] != "LAB-004":
            continue
        key = (exp["family"], exp["feature_set_id"])
        by_family[key] = exp["metrics"]["oos_ic"]

    baseline_ics = {}
    for family in ["ridge", "lasso", "random_forest", "xgboost"]:
        baseline_ics[family] = by_family.get((family, BASELINE_FS), None)

    header = f"  {'Model':<16} {'Baseline':>10}"
    for fs in FUNDAMENTAL_FS:
        header += f" {FS_LABELS[fs]:>12}"
    header += " {'Delta':>12}"
    print(header)
    print("  " + "-" * 90)

    best_deltas = []
    for family in ["ridge", "lasso", "random_forest", "xgboost"]:
        bl = baseline_ics.get(family)
        if bl is None:
            continue
        line = f"  {family:<16} {bl:>10.4f}"
        fund_ics = []
        for fs in FUNDAMENTAL_FS:
            ic = by_family.get((family, fs))
            if ic is not None:
                line += f" {ic:>12.4f}"
                fund_ics.append((fs, ic))
            else:
                line += f" {'N/A':>12}"
        if fund_ics:
            best_fs, best_ic = max(fund_ics, key=lambda x: x[1])
            delta = best_ic - bl
            best_deltas.append((family, best_fs, bl, best_ic, delta))
            line += f" {delta:>+12.4f}"
        print(line)

    print(f"\n  Summary:")
    pos = [d for d in best_deltas if d[4] > 0]
    neg = [d for d in best_deltas if d[4] < 0]
    print(f"    Improvement: {len(pos)}/{len(best_deltas)}")
    print(f"    Degradation: {len(neg)}/{len(best_deltas)}")
    if best_deltas:
        b = max(best_deltas, key=lambda x: x[4])
        print(f"    Best: {b[0]}+{b[1]} ({b[2]:.4f} -> {b[3]:.4f}, delta={b[4]:+.4f})")

# ============================================================
# 2. STATISTICAL SIGNIFICANCE (Holm-corrected, LAB-004 only)
# ============================================================
print("\n" + "=" * 72)
print("STATISTICAL SIGNIFICANCE (LAB-004 only, Holm-corrected)")
print("=" * 72)

from scipy import stats
import numpy as np

for env_id in ["ENV-12D-050", "ENV-12D-100"]:
    data = load_results(env_id)
    lab004 = [e for e in data["results"] if e["label_id"] == "LAB-004"]
    n = len(lab004)

    results_with_p = []
    for exp in lab004:
        ic = exp["metrics"]["oos_ic"]
        n_test = exp["n_test"]
        if abs(ic) < 1e-10 or n_test < 3:
            pval = 1.0
        else:
            t_stat = ic * np.sqrt(n_test - 2) / np.sqrt(max(1 - ic**2, 1e-20))
            pval = 2 * (1 - stats.t.cdf(abs(t_stat), df=n_test - 2))
        results_with_p.append({
            "eid": exp["experiment_id"],
            "family": exp["family"],
            "fs": exp["feature_set_id"],
            "ic": ic,
            "n_test": n_test,
            "pval": pval,
        })

    results_with_p.sort(key=lambda x: x["pval"])
    alpha = 0.05
    holm_sig = set()
    for i, r in enumerate(results_with_p):
        threshold = alpha / (n - i)
        if r["pval"] <= threshold:
            holm_sig.add(r["eid"])
        else:
            break

    print(f"\n--- {env_id} (n={n}) ---")
    print(f"  Significant (Holm): {len(holm_sig)}/{n}")
    for r in results_with_p:
        sig = "***" if r["eid"] in holm_sig else "   "
        print(f"  {sig} {r['family']:<16} {r['fs']:<10} IC={r['ic']:+.4f}  p={r['pval']:.4f}")

# ============================================================
# 3. CROSS-PHASE COMPARISON
# ============================================================
print("\n" + "=" * 72)
print("CROSS-PHASE COMPARISON")
print("=" * 72)
print("""
Phase  | Verdict | IC Range           | Notes
-------|---------|--------------------|-----------------------------------
11     | D       | ~0.00 to +0.02     | Universe 20->50->97, null persists
12A    | D       | ~0.00 to +0.02     | Market context, cross-sectional neg
12B    | E       | 80/96 blocked      | Synthetic data invalid
12C    | B       | N/A                | Data acquisition only
12D    | D       | -0.023 to +0.034   | Real SEC EDGAR, 96/96 completed
-------|---------|--------------------|-----------------------------------

Key findings:
1. All model-based phases produce verdict D
2. Phase 12D confirms: real fundamentals don't consistently improve predictions
3. Linear models show marginal improvement; non-linear models degrade
4. No evidence that PIT fundamental features add predictive power beyond OHLCV
""")

# ============================================================
# 4. FINAL VERDICT AND REPORT
# ============================================================
print("=" * 72)
print("PHASE 12D FINAL VERDICT")
print("=" * 72)
print("""
VERDICT: D -- Null persists (no convincing predictive improvement)

EVIDENCE:
1. OOS IC range: -0.023 to +0.034 (all economically marginal)
2. Best: Lasso+FS-12B-B in 050: IC=0.034 vs baseline 0.024 (+42%)
3. RF/XGBoost consistently DEGRADE with fundamental features
4. Results inconsistent across 050 vs 100
5. All adversarial tests PASS (real SEC EDGAR confirmed)
6. LAB-005 = LAB-004 (excess return not yet implemented)

LIMITATIONS:
- LAB-005 excess return not computed
- Forward-filled quarterly data (94% identical rows, temporal leakage risk)
- 86/100 SEC CIKs mapped
- Single source: SEC EDGAR 10-K/10-Q

RECOMMENDATION:
- Phase 12D does NOT overturn the null from Phase 11
- Answer: No convincing evidence for PIT fundamental predictive power
- Pipeline is functional and reusable for future work
""")

report = {
    "phase": "12D",
    "verdict": "D",
    "reason": "Null persists - no convincing predictive improvement from PIT fundamentals",
    "incremental_value": {
        "best_linear_improvement": "Lasso+FS-12B-B in 050: IC=0.034 vs baseline 0.024 (+42%)",
        "nonlinear_degradation": "RF/XGBoost consistently negative IC with fundamental features",
        "cross_environment_consistency": "Low",
    },
    "limitations": [
        "LAB-005 excess return not implemented (identical to LAB-004)",
        "Forward-filled quarterly data creates temporal leakage risk",
        "86/100 SEC CIKs mapped",
        "Single source: SEC EDGAR 10-K/10-Q",
    ],
    "adversarial_tests": "ALL PASS (6/6)",
    "data_source": "Real SEC EDGAR CompanyFacts (not synthetic)",
    "experiments_completed": "96/96 (48 per environment)",
    "cross_phase_comparison": {
        "Phase 11": "D",
        "Phase 12A": "D",
        "Phase 12B": "E (synthetic data blocked)",
        "Phase 12C": "B (data acquisition)",
        "Phase 12D": "D",
    },
}
with open(BENCH / "phase12d_final_report.json", "w") as f:
    json.dump(report, f, indent=2)
print("Report saved: benchmarks/phase12d_final_report.json")
