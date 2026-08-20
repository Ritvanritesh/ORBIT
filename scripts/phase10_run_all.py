"""Phase 10 - Feature Engineering + Ablation: full run.

Run:  python scripts/phase10_run_all.py

Pipeline (register-before-run throughout; Phase 9 artifacts are read-only):
  1. load DS-000004 + instrument master + events; data-expansion guard
  2. build/load (digest-verified) the frozen FS-001 base snapshot, the
     FS-002..FS-013 feature-set snapshots, and the LAB-004 label snapshot
  3. feature quality + redundancy diagnostics (training split only)
  4. run all 52 pre-registered ablation experiments (13 feature sets x 4
     model points): train on train, calibrate on validation, evaluate OOS
     IC / rank IC / ECE / Brier / MSE / hit rate on the locked test window,
     convert top-3 predictions to canonical signals, backtest through the
     canonical Phase 7 engine with CM-001 costs (identical to Phase 9), and
     record every experiment through the Phase 6 lifecycle
  5. write the permanent reports (plan, diagnostics, parquet, markdown,
     research report)
  6. run the Phase 10 independent audit (strong temporal boundary included)

Every experiment id is deterministic (EXP-10001..EXP-10052). All runs,
including null and failed ones, are recorded in the report. Nothing in
Phase 10 mutates Phase 9 artifacts.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "src"))

from orbit.ml.phase10_runner import run_phase10_all  # noqa: E402


def main() -> None:
    result = run_phase10_all()
    audit = result["audit"]
    if audit["blocked"]:
        raise SystemExit(
            f"PHASE 10 AUDIT BLOCKED: {audit['failed']} failed checks: "
            f"{audit['failed_checks']}"
        )
    print(
        f"PHASE 10 STATUS: benchmark report at {result['report_path']}; "
        f"audit {audit['passed']}/{audit['checks']} PASS; verdict must follow "
        "the checklist in docs/PHASE_10_STATUS.md"
    )


if __name__ == "__main__":
    main()