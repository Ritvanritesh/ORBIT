"""Phase 11.2 main execution: run benchmark suite on expanded universes.

Runs the locked benchmark suite on 50-symbol and 100-symbol datasets.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orbit.ml.phase11_2_benchmark import (
    run_benchmark_suite,
    persist_results,
    load_dataset,
    load_benchmark_bars,
)


def main():
    print("=" * 72)
    print("PHASE 11.2 - BENCHMARK EXECUTION ON EXPANDED UNIVERSES")
    print("=" * 72)

    results = {}

    # Run on 50-symbol dataset
    print("\n" + "#" * 72)
    print("# ENV-3: 50-SYMBOL UNIVERSE")
    print("#" * 72)
    try:
        result_50 = run_benchmark_suite(
            snapshot_id="DS-EXP-050",
            env_id="ENV-3",
            label_type="both",
            progress=True,
        )
        path_50 = persist_results(result_50)
        print(f"\nResults persisted to: {path_50}")
        results["ENV-3"] = result_50
    except Exception as e:
        print(f"\nERROR running ENV-3: {e}")
        results["ENV-3"] = {"error": str(e)}

    # Run on 100-symbol dataset
    print("\n" + "#" * 72)
    print("# ENV-4: 100-SYMBOL UNIVERSE")
    print("#" * 72)
    try:
        result_100 = run_benchmark_suite(
            snapshot_id="DS-EXP-100",
            env_id="ENV-4",
            label_type="both",
            progress=True,
        )
        path_100 = persist_results(result_100)
        print(f"\nResults persisted to: {path_100}")
        results["ENV-4"] = result_100
    except Exception as e:
        print(f"\nERROR running ENV-4: {e}")
        results["ENV-4"] = {"error": str(e)}

    # Summary
    print("\n" + "=" * 72)
    print("EXECUTION COMPLETE")
    print("=" * 72)

    for env_id, res in results.items():
        print(f"\n{env_id}:")
        if "error" in res and isinstance(res["error"], str):
            print(f"  ERROR: {res['error']}")
        else:
            print(f"  Instruments: {res.get('n_instruments', '?')}")
            print(f"  Sessions: {res.get('n_sessions', '?')}")
            print(f"  Successful: {res.get('n_successful', '?')}")
            print(f"  Failed: {res.get('n_failed', '?')}")

            # Show IC summary by label
            for lab_id in ["LAB-004", "LAB-005"]:
                lab_results = [r for r in res.get("results", []) if r.get("label_id") == lab_id and "error" not in r]
                ics = [r["metrics"]["oos_ic"] for r in lab_results if r["metrics"].get("oos_ic") is not None]
                if ics:
                    import numpy as np
                    print(f"  {lab_id} OOS IC: mean={np.mean(ics):.4f}, median={np.median(ics):.4f}")

    return results


if __name__ == "__main__":
    results = main()
    sys.exit(0)
