"""Phase 12A final comparison and report."""
import json, sys
from pathlib import Path
import numpy as np

REPO_ROOT = Path.cwd()

def load_results(env_id):
    p = REPO_ROOT / "benchmarks" / f"phase12a_{env_id}_results.json"
    return json.loads(p.read_text(encoding="utf-8"))

def main():
    print("=" * 78)
    print("PHASE 12A FINAL REPORT")
    print("=" * 78)

    envs = {}
    for eid in ["ENV-12A-050", "ENV-12A-100"]:
        try:
            envs[eid] = load_results(eid)
        except FileNotFoundError:
            print(f"  WARNING: {eid} not found")

    try:
        p11_3 = json.loads((REPO_ROOT / "benchmarks" / "phase11_2_ENV-3_results.json").read_text(encoding="utf-8"))
        p11_4 = json.loads((REPO_ROOT / "benchmarks" / "phase11_2_ENV-4_results.json").read_text(encoding="utf-8"))
    except FileNotFoundError:
        p11_3 = p11_4 = None

    print("\n1. ENVIRONMENT SUMMARY")
    print("-" * 78)
    print(f"{'Env':<16} {'Sym':<6} {'FS':<10} {'IC Mean':<10} {'IC Median':<10} {'OK/Tot':<8}")
    print("-" * 78)

    fsets = ["FS-001", "FS-101", "FS-102", "FS-103", "FS-104"]
    for eid, data in envs.items():
        for fs_id in fsets:
            fic = [r["metrics"]["oos_ic"] for r in data["results"]
                   if r.get("feature_set_id") == fs_id and "error" not in r
                   and r["metrics"].get("oos_ic") is not None]
            n_ok = len([r for r in data["results"] if r.get("feature_set_id") == fs_id and "error" not in r])
            n_tot = len([r for r in data["results"] if r.get("feature_set_id") == fs_id])
            if fic:
                print(f"{eid:<16} {data['n_instruments']:<6} {fs_id:<10} {np.mean(fic):<10.4f} {np.median(fic):<10.4f} {n_ok}/{n_tot}")
            else:
                print(f"{eid:<16} {data['n_instruments']:<6} {fs_id:<10} {'N/A':<10} {'N/A':<10} {n_ok}/{n_tot}")

    if p11_3 and p11_4:
        print("\n2. PHASE 11.2 COMPARISON (baseline only)")
        print("-" * 78)
        for p11, label in [(p11_3, "Phase 11.2 ENV-3 (50 sym)"), (p11_4, "Phase 11.2 ENV-4 (97 sym)")]:
            ics = [r["metrics"]["oos_ic"] for r in p11["results"]
                   if "error" not in r and r["metrics"].get("oos_ic") is not None]
            if ics:
                print(f"  {label}: mean={np.mean(ics):.4f}, median={np.median(ics):.4f}")

    print("\n3. INFORMATION FAMILY EFFECT")
    print("-" * 78)
    all_baseline, all_market, all_xs = [], [], []
    for eid, data in envs.items():
        for r in data["results"]:
            if "error" in r: continue
            ic = r["metrics"].get("oos_ic")
            if ic is None: continue
            if r["feature_set_id"] == "FS-001": all_baseline.append(ic)
            elif r["feature_set_id"] == "FS-101": all_market.append(ic)
            elif r["feature_set_id"] == "FS-103": all_xs.append(ic)

    print(f"  Baseline (FS-001):       mean={np.mean(all_baseline):.4f}, n={len(all_baseline)}")
    print(f"  +Market (FS-101):        mean={np.mean(all_market):.4f}, n={len(all_market)}")
    print(f"  +Cross-section (FS-103): mean={np.mean(all_xs):.4f}, n={len(all_xs)}")
    print(f"  +Sector (FS-102):        FAILED (sparse data)")
    print(f"  +All context (FS-104):   FAILED (sparse data)")

    market_diff = np.mean(all_market) - np.mean(all_baseline)
    xs_diff = np.mean(all_xs) - np.mean(all_baseline)

    print(f"\n  Market context effect:     {market_diff:+.4f} IC")
    print(f"  Cross-sectional effect:    {xs_diff:+.4f} IC")

    print("\n4. VERDICT")
    print("-" * 78)
    best = max(np.mean(all_baseline), np.mean(all_market), np.mean(all_xs))
    if abs(market_diff) < 0.01 and abs(xs_diff) < 0.01:
        verdict = "D"
        desc = "No convincing predictive improvement from new information domains."
    elif market_diff > 0.005 or xs_diff > 0.005:
        verdict = "B"
        desc = "Some evidence of improvement, but uncertainty remains substantial."
    elif best > 0.02:
        verdict = "C"
        desc = "Results change materially but remain mixed."
    else:
        verdict = "D"
        desc = "No convincing predictive improvement."

    print(f"  Best OOS IC: {best:.4f}")
    print(f"  Market effect: {market_diff:+.4f}")
    print(f"  XS effect: {xs_diff:+.4f}")
    print(f"  VERDICT: {verdict} - {desc}")

    print("\n5. HONEST LIMITATIONS")
    print("-" * 78)
    print("  - Sector features (FS-102/FS-104) failed due to sparse sector data")
    print("  - Sector membership treated as time-invariant (documented limitation)")
    print("  - Market context uses SPY as sole proxy")
    print("  - Cross-sectional features limited to ret_20 and vol_10")
    print("  - Label computation takes ~250s (dominant bottleneck)")

    report = {
        "phase": "12A",
        "environments": {},
        "verdict": verdict,
        "verdict_description": desc,
        "market_effect": float(market_diff),
        "xs_effect": float(xs_diff),
        "baseline_mean_ic": float(np.mean(all_baseline)),
        "market_mean_ic": float(np.mean(all_market)),
        "xs_mean_ic": float(np.mean(all_xs)),
    }
    for eid, data in envs.items():
        report["environments"][eid] = {
            "n_instruments": data["n_instruments"],
            "n_successful": data["n_successful"],
            "n_failed": data["n_failed"],
        }
    out = REPO_ROOT / "benchmarks" / "phase12a_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nReport saved: {out.name}")

if __name__ == "__main__":
    main()
