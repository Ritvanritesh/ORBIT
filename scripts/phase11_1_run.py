"""Phase 11.1 entry point: run the full controlled expansion analysis."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orbit.ml.phase11_1_audit import persist_audit, run_full_audit
from orbit.ml.phase11_1_runner import run_phase11_1_analysis


def main() -> None:
    # Run the analysis
    results = run_phase11_1_analysis(progress=True)

    # Run the audit
    print("\n" + "=" * 72)
    print("RUNNING INDEPENDENT AUDIT")
    print("=" * 72)

    checks = run_full_audit(
        stage_a_results=results.get("stage_a"),
        stage_b_results=results.get("stage_b"),
    )

    from orbit.ml.phase11_1_audit import audit_summary
    summary = audit_summary(checks)
    persist_audit(checks, results.get("stage_a"), results.get("stage_b"))

    print(f"\nAudit: {summary['passed']}/{summary['checks']} checks passed")
    if summary["blocked"]:
        print(f"FAILED CHECKS: {summary['failed_checks']}")
        sys.exit(1)
    else:
        print("All checks passed.")
        sys.exit(0)


if __name__ == "__main__":
    main()
