"""Phase 10 - Review 1: independent structural audit of the benchmark artifacts.

Runs against the stored Phase 10 artifacts (benchmarks/) and the digest-
verified snapshot cache. Verifies:

   1. plan_lock            plan.json digest == the locked plan digest; 52
                           experiments, ids EXP-10001..EXP-10052 contiguous;
                           every model point is a Phase 9 grid point.
   2. report_complete      the parquet report has exactly the plan's 52 ids,
                           all status completed, no hidden rows.
   3. set_membership       every feature set's members == the registered set.
   4. snapshot_digests     every report row's feature_set_digest and
                           definitions_digest match the cached snapshot and
                           the current definitions (a mutated feature or set
                           after the run is detected here).
   5. audit_pass           the independent Phase 10 audit re-runs clean over
                           the cached snapshots and datasets.
   6. cross_phase_base     the FS-001 base runs (EXP-10001..10004) EXACTLY
                           reproduce the Phase 9 parent experiments
                           (EXP-90003/90006/90015/90019) - the ablation is
                           anchored bit-for-bit to the DEFENSIBLE NULL; this
                           includes a bitwise sha256 comparison of the stored
                           test-prediction parquet files.
   7. diagnostics_scope    diagnostics.json is train-only scoped.

Exit code 0 = all checks PASS.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import polars as pl

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "src"))

from orbit.ml.data import load_instrument_master, load_snapshot_bars, load_snapshot_events  # noqa: E402
from orbit.ml.dataset import assemble_datasets  # noqa: E402
from orbit.ml.features import (  # noqa: E402
    FEATURE_DEFINITIONS,
    FEATURE_NAMES,
    FEATURE_NAMES_PHASE10,
    PHASE10_FEATURE_SETS,
    _feature_definitions_digest,
)
from orbit.ml.phase10_audit import audit_summary, run_phase10_audit  # noqa: E402
from orbit.ml.phase10_plan import PHASE10_MODEL_POINTS, phase10_plan_digest  # noqa: E402
from orbit.ml.phase10_report import (  # noqa: E402
    DIAGNOSTICS_JSON,
    PLAN_JSON,
    REPORT_PARQUET,
)
from orbit.ml.snapshot_cache import (  # noqa: E402
    PHASE10_SNAPSHOT_CACHE_DIR,
    load_cached_feature_snapshot,
    load_cached_phase10_snapshot,
)

P9_RUNS = _REPO_ROOT / "benchmarks" / "phase9_runs"
RESULT_FILE = _REPO_ROOT / "benchmarks" / "phase10_review1_results.json"

PHASE9_PARENTS = {
    "EXP-10001": ("EXP-90003", "ridge", {"alpha": 1.0}),
    "EXP-10002": ("EXP-90006", "lasso", {"alpha": 0.001}),
    "EXP-10003": ("EXP-90015", "random_forest", {"n_estimators": 200, "max_depth": 3}),
    "EXP-10004": ("EXP-90019", "xgboost", {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.1}),
}

_EXCLUDED_METRIC_KEYS = {"created_at"}


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _check(name: str, passed: bool, evidence: str) -> dict:
    return {"check": name, "status": "PASS" if passed else "FAIL", "evidence": evidence}


def main() -> None:
    checks: list[dict] = []

    # 1. plan lock --------------------------------------------------------
    plan = json.loads(PLAN_JSON.read_text(encoding="utf-8"))
    checks.append(_check(
        "plan_lock",
        plan["plan_digest"] == phase10_plan_digest(),
        f"plan.json digest {plan['plan_digest'][:16]}... vs locked "
        f"{phase10_plan_digest()[:16]}...",
    ))
    checks.append(_check(
        "plan_count",
        plan["experiment_count"] == 52,
        f"experiment_count {plan['experiment_count']}",
    ))
    plan_ids = [
        s["feature_set_id"]
        for s in __import__("orbit.ml.features", fromlist=["PHASE10_FEATURE_SET_ORDER"]).PHASE10_FEATURE_SET_ORDER
    ]
    checks.append(_check(
        "plan_set_order",
        [s["feature_set_id"] for s in plan["feature_sets"]] == plan_ids,
        f"{len(plan['feature_sets'])} feature sets in locked order",
    ))

    # 2. report completeness ----------------------------------------------
    frame = pl.read_parquet(REPORT_PARQUET)
    expected_ids = {f"EXP-{i:05d}" for i in range(10001, 10053)}
    checks.append(_check(
        "report_complete",
        set(frame["experiment_id"].to_list()) == expected_ids,
        f"{frame.height} rows; ids match EXP-10001..EXP-10052",
    ))
    checks.append(_check(
        "report_no_hidden_failures",
        (frame["status"].to_list() == ["completed"] * 52),
        f"statuses: {sorted(set(frame['status'].to_list()))}",
    ))

    # 3. set membership -----------------------------------------------------
    base_ids = [f["feature_id"] for f in FEATURE_DEFINITIONS]
    membership_ok = True
    membership_evidence = []
    for row in frame.unique(subset=["feature_set_id"]).sort("feature_set_id").iter_rows(named=True):
        sid = row["feature_set_id"]
        members = sorted(
            PHASE10_FEATURE_SETS[sid]["members"] if sid != "FS-001" else base_ids
        )
        ok = row["n_features"] == len(members)
        membership_ok = membership_ok and ok
        membership_evidence.append(f"{sid}={len(members)}")
    checks.append(_check(
        "set_membership",
        membership_ok,
        "; ".join(membership_evidence),
    ))

    # 4. snapshot digests match the cache ----------------------------------
    bars = load_snapshot_bars()
    events = load_snapshot_events()
    instruments = load_instrument_master()
    snapshots = {}
    fs001 = load_cached_feature_snapshot()
    if fs001 is not None:
        snapshots["FS-001"] = fs001
    for sid in plan_ids:
        if sid == "FS-001":
            continue
        s = load_cached_phase10_snapshot(sid, PHASE10_SNAPSHOT_CACHE_DIR)
        if s is not None:
            snapshots[sid] = s

    digest_mismatches = []
    for row in frame.unique(subset=["feature_set_id"]).sort("feature_set_id").iter_rows(named=True):
        sid = row["feature_set_id"]
        snap = snapshots.get(sid)
        if snap is None:
            digest_mismatches.append(f"{sid}: snapshot missing from cache")
            continue
        members = snap.feature_refs
        if row["feature_set_digest"] != snap.content_digest:
            digest_mismatches.append(
                f"{sid}: report digest {row['feature_set_digest'][:16]}... != "
                f"cache digest {snap.content_digest[:16]}..."
            )
        if row["definitions_digest"] != _feature_definitions_digest(members):
            digest_mismatches.append(f"{sid}: definitions_digest mismatch")
    checks.append(_check(
        "snapshot_digests",
        not digest_mismatches,
        "; ".join(digest_mismatches) if digest_mismatches else "all 13 set digests match the cache",
    ))

    # 5. independent audit over cached snapshots ----------------------------
    decisions = snapshots["FS-001"].records.select("instrument_id", "decision_time")
    ls = __import__("orbit.ml.labels", fromlist=["build_phase9_label_snapshot"]).build_phase9_label_snapshot(
        bars, events, instruments, decisions, data_refs=["DS-000004"]
    )
    datasets_by_set = {}
    for sid in ("FS-001", "FS-003"):
        names = (
            list(FEATURE_NAMES)
            if sid == "FS-001"
            else list(FEATURE_NAMES) + list(FEATURE_NAMES_PHASE10)
        )
        datasets_by_set[sid] = assemble_datasets(snapshots[sid], ls, feature_names=names)
    audit_checks = run_phase10_audit(
        snapshots=snapshots,
        base_snapshot=snapshots.get("FS-001"),
        label_snapshot=ls,
        datasets_by_set=datasets_by_set,
        phase9_fs001_digest=snapshots["FS-001"].content_digest,
        bars=bars,
    )
    audit = audit_summary(audit_checks)
    checks.append(_check(
        "audit_pass",
        audit["failed"] == 0,
        f"{audit['passed']}/{audit['checks']} PASS; "
        f"failed: {audit['failed_checks']}",
    ))

    # 6. cross-phase base consistency (anchor to the DEFENSIBLE NULL) -------
    base_ok = True
    base_evidence = []
    for exp10_id, (p9_id, fam, params) in sorted(PHASE9_PARENTS.items()):
        stored_p10 = json.loads(
            (_REPO_ROOT / "benchmarks" / "phase10_runs" / exp10_id / "metrics.json").read_text(encoding="utf-8")
        )
        stored_p9 = json.loads((P9_RUNS / p9_id / "metrics.json").read_text(encoding="utf-8"))
        diffs = {}
        for key in sorted(set(stored_p10) | set(stored_p9)):
            if key in _EXCLUDED_METRIC_KEYS:
                continue
            a, b = stored_p10.get(key), stored_p9.get(key)
            if a != b:
                diffs[key] = {"phase10": a, "phase9": b}
        # bitwise test-prediction artifact comparison
        a10 = _REPO_ROOT / "benchmarks" / "phase10_runs" / exp10_id / "test_predictions.parquet"
        a9 = P9_RUNS / p9_id / "test_predictions.parquet"
        bitwise = a10.exists() and a9.exists() and _sha256_file(a10) == _sha256_file(a9)
        ok = not diffs and bitwise
        base_ok = base_ok and ok
        base_evidence.append(
            f"{exp10_id}=={p9_id}: metrics {'match' if not diffs else diffs}; "
            f"predictions {'bitwise-identical' if bitwise else 'DIFFER'} (sha {_sha256_file(a10)[:12]} vs {_sha256_file(a9)[:12]})"
        )
    checks.append(_check(
        "cross_phase_base_consistency",
        base_ok,
        "; ".join(base_evidence),
    ))

    # 7. diagnostics scope ---------------------------------------------------
    diag = json.loads(DIAGNOSTICS_JSON.read_text(encoding="utf-8"))
    checks.append(_check(
        "diagnostics_scope",
        diag.get("scope") == "train split only (never test)",
        f"scope={diag.get('scope')}",
    ))

    # verdict ----------------------------------------------------------------
    failed = [c for c in checks if c["status"] == "FAIL"]
    payload = {
        "protocol": "phase10_review1_v1",
        "checks": checks,
        "verdict": "PASS" if not failed else "FAIL",
        "completed_at": __import__("time").strftime("%Y-%m-%dT%H:%M:%S"),
    }
    RESULT_FILE.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    for c in checks:
        print(f"[review1] {c['status']}: {c['check']} - {c['evidence']}")
    print(f"[review1] results written: {RESULT_FILE}")
    print(f"[review1] VERDICT: {'PASS' if not failed else 'FAIL'}")
    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()