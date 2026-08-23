"""Phase 11.2 cross-generation comparison and inference."""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_env_results(env_id: str) -> dict:
    path = REPO_ROOT / "benchmarks" / f"phase11_2_{env_id}_results.json"
    with open(path) as f:
        return json.load(f)


def extract_ics(results: dict) -> dict[str, list[float]]:
    """Extract IC values grouped by (family, feature_set)."""
    ics_by_combo = {}
    for r in results["results"]:
        if "error" in r:
            continue
        key = f"{r['family']}+{r['feature_set_id']}"
        ic = r.get("metrics", {}).get("oos_ic")
        if ic is not None:
            ics_by_combo.setdefault(key, []).append(ic)
    return ics_by_combo


def main():
    envs = {}
    for env_id in ["ENV-1", "ENV-2", "ENV-3", "ENV-4"]:
        try:
            envs[env_id] = load_env_results(env_id)
        except FileNotFoundError:
            print(f"WARNING: {env_id} results not found")

    print("=" * 78)
    print("PHASE 11.2 CROSS-GENERATION COMPARISON")
    print("=" * 78)

    # Per-environment summary
    print("\n--- Per-Environment Summary ---")
    print(f"{'Env':<8} {'Symbols':<10} {'Sessions':<10} {'IC Mean':<10} {'IC Median':<12} {'IC Min':<10} {'IC Max':<10}")
    print("-" * 78)

    env_ics = {}
    for env_id, data in envs.items():
        ics = [r["metrics"]["oos_ic"] for r in data["results"]
               if "error" not in r and r.get("metrics", {}).get("oos_ic") is not None]
        env_ics[env_id] = ics
        n_sym = data.get("n_instruments", "?")
        n_sess = data.get("n_sessions", "?")
        if ics:
            print(f"{env_id:<8} {n_sym:<10} {n_sess:<10} {np.mean(ics):<10.4f} {np.median(ics):<12.4f} {np.min(ics):<10.4f} {np.max(ics):<10.4f}")
        else:
            print(f"{env_id:<8} {n_sym:<10} {n_sess:<10} {'N/A':<10}")

    # Per-combo breakdown
    print("\n--- Per-Model-FeatureSet IC Values ---")
    print(f"{'Combo':<30} {'ENV-1':<10} {'ENV-2':<10} {'ENV-3':<10} {'ENV-4':<10}")
    print("-" * 78)

    all_combos = set()
    combo_data = {}
    for env_id, data in envs.items():
        ics = extract_ics(data)
        for combo, vals in ics.items():
            all_combos.add(combo)
            combo_data.setdefault(combo, {})[env_id] = vals[0] if vals else None

    for combo in sorted(all_combos):
        vals = [combo_data.get(combo, {}).get(e) for e in ["ENV-1", "ENV-2", "ENV-3", "ENV-4"]]
        row = f"{combo:<30}"
        for v in vals:
            if v is not None:
                row += f" {v:<10.4f}"
            else:
                row += f" {'N/A':<10}"
        print(row)

    # Statistical comparison
    print("\n--- Universe Expansion Effect ---")
    print("(ENV-3 vs ENV-1, ENV-4 vs ENV-1)")

    for env_id in ["ENV-3", "ENV-4"]:
        if env_id not in env_ics or "ENV-1" not in env_ics:
            continue
        ics_small = env_ics["ENV-1"]
        ics_large = env_ics[env_id]
        diff = np.mean(ics_large) - np.mean(ics_small)
        pooled_se = np.sqrt(np.var(ics_small, ddof=1)/len(ics_small) + np.var(ics_large, ddof=1)/len(ics_large))
        t_stat = diff / pooled_se if pooled_se > 0 else 0
        from scipy.stats import t as t_dist
        df = len(ics_small) + len(ics_large) - 2
        p_val = 2 * (1 - t_dist.cdf(abs(t_stat), df)) if df > 0 else 1.0
        print(f"  {env_id} vs ENV-1: mean_diff={diff:+.4f}, t={t_stat:.3f}, p={p_val:.4f} {'(sig)' if p_val < 0.05 else '(not sig)'}")

    # Verdict
    print("\n--- Verdict ---")
    all_ics_flat = []
    for ics in env_ics.values():
        all_ics_flat.extend(ics)
    max_ic = max(all_ics_flat) if all_ics_flat else 0
    mean_ic = np.mean(all_ics_flat) if all_ics_flat else 0

    if max_ic > 0.05:
        verdict = "A — Stronger evidence for signal"
    elif mean_ic > 0.02:
        verdict = "B — Improved but uncertain"
    elif any(d < -0.005 for d in [np.mean(env_ics.get(e, [0])) - np.mean(env_ics.get("ENV-1", [0])) for e in ["ENV-3", "ENV-4"]]):
        verdict = "D — Null persists"
    else:
        verdict = "D — Null persists"

    print(f"  Best OOS IC across all environments: {max_ic:.4f}")
    print(f"  Mean OOS IC across all environments: {mean_ic:.4f}")
    print(f"  VERDICT: {verdict}")

    # Save comparison
    comparison = {
        "phase": "11.2",
        "environments": {},
        "cross_generation": {},
        "verdict": verdict,
    }
    for env_id, data in envs.items():
        ics = env_ics.get(env_id, [])
        comparison["environments"][env_id] = {
            "n_instruments": data.get("n_instruments"),
            "n_sessions": data.get("n_sessions"),
            "n_experiments": len(data.get("results", [])),
            "oos_ic_mean": float(np.mean(ics)) if ics else None,
            "oos_ic_median": float(np.median(ics)) if ics else None,
            "oos_ic_min": float(np.min(ics)) if ics else None,
            "oos_ic_max": float(np.max(ics)) if ics else None,
        }

    with open("benchmarks/phase11_2_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2, default=str)
    print("\nComparison saved to benchmarks/phase11_2_comparison.json")


if __name__ == "__main__":
    main()
