"""Phase 11 real-data analysis runner script.

Run from the repo root:
    python scripts/phase11_run_analysis.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> None:
    # Step 1: Write locked plan
    from orbit.ml.phase11_plan import write_plan, phase11_plan
    plan = phase11_plan()
    plan_path = write_plan(plan)
    print(f"Locked plan written to: {plan_path}")
    print(f"Plan digest: {plan['plan_digest'][:32]}...")

    # Step 2: Run analysis with progress
    from orbit.ml.phase11_runner import run_phase11_analysis, persist_results
    analysis = run_phase11_analysis(plan=plan, progress=True)

    # Step 3: Persist results
    results_path = persist_results(analysis)
    print(f"\nResults persisted to: {results_path}")

    # Step 4: Run audit
    print("\n" + "=" * 72)
    print("RUNNING INDEPENDENT AUDIT")
    print("=" * 72)
    from orbit.ml.phase11_audit import run_phase11_audit, audit_summary, persist_audit
    checks = run_phase11_audit(plan=plan, analysis=analysis, synthetic_validation_passed=True)
    summary = audit_summary(checks)
    print(f"\n  Audit: {summary['passed']}/{summary['checks']} passed")
    if summary["blocked"]:
        print(f"  BLOCKED: {summary['failed_checks']}")
    else:
        print("  Audit PASSED")
    audit_path = persist_audit(checks, analysis)
    print(f"  Audit written to: {audit_path}")

    # Step 5: Generate reports
    print("\n" + "=" * 72)
    print("GENERATING REPORTS")
    print("=" * 72)
    from orbit.ml.phase11_report import (
        write_markdown_report,
        write_research_report,
        write_phase11_status,
    )
    md_path = write_markdown_report(analysis)
    print(f"  Results markdown: {md_path}")
    research_path = write_research_report(analysis, checks)
    print(f"  Research report: {research_path}")
    status_path = write_phase11_status(analysis, checks)
    print(f"  Status report: {status_path}")

    # Step 6: Summary
    print("\n" + "=" * 72)
    print("FINAL SUMMARY")
    print("=" * 72)

    mt = analysis.get("multiple_testing")
    if mt:
        holm = mt.get("holm_bonferroni", {})
        bh = mt.get("benjamini_hochberg", {})
        print(f"\nMultiple-Testing (52-experiment family):")
        print(f"  Raw significant at 5%: {mt.get('n_raw_significant_005', 0)}")
        print(f"  Holm-Bonferroni at 5%: {len(holm.get('significant_at', {}).get('0.05', []))}")
        print(f"  BH (FDR) at 5%: {len(bh.get('significant_at', {}).get('0.05', []))}")

    from orbit.ml.phase11_report import _determine_verdict
    verdict = _determine_verdict(analysis)
    print(f"\nFINAL VERDICT: {verdict['code']}")
    print(f"  {verdict['description']}")

    print("\n" + "=" * 72)
    print("PHASE 11 COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()
