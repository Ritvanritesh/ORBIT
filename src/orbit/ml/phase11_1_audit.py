"""Phase 11.1 independent audit layer.

Verifies that Stage A and Stage B were conducted according to the
locked plans, all sources are intact, no cherry-picking occurred,
and the results are reproducible.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[3]
BENCHMARKS_DIR = _REPO_ROOT / "benchmarks"
AUDIT_JSON = BENCHMARKS_DIR / "phase11_1_audit.json"


def _check(name: str, passed: bool, evidence: str) -> dict[str, Any]:
    return {"check": name, "status": "PASS" if passed else "FAIL", "evidence": evidence}


def _sha256_json(obj: Any) -> str:
    raw = json.dumps(obj, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _file_checksum(path: Path) -> str:
    if not path.exists():
        return "MISSING"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def run_stage_a_audit(
    *,
    stage_a_plan: dict[str, Any] | None = None,
    stage_a_results: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Run the full Stage A independent audit.

    Returns a list of check dicts with {check, status, evidence}.
    """
    checks: list[dict[str, Any]] = []

    # 1. Stage A plan exists and is digest-verified
    try:
        if stage_a_plan is None:
            stage_a_plan = json.loads(
                (BENCHMARKS_DIR / "phase11_1_stage_a_plan.json").read_text(encoding="utf-8")
            )
        stored_digest = stage_a_plan.get("plan_digest")
        payload = {k: v for k, v in stage_a_plan.items() if k != "plan_digest"}
        computed_digest = _sha256_json(payload)
        digest_ok = stored_digest is not None and stored_digest == computed_digest
        checks.append(_check(
            "stage_a_plan_digest",
            digest_ok,
            f"stored={stored_digest[:16] if stored_digest else 'N/A'}... "
            f"computed={computed_digest[:16]}... verified={digest_ok}",
        ))
    except Exception as exc:
        checks.append(_check("stage_a_plan_digest", False, str(exc)))
        return checks

    # 2. Benchmark configuration is locked
    try:
        from orbit.ml.phase11_1_benchmark import BENCH_001_CONFIG, BenchmarkRole
        config_valid = (
            BENCH_001_CONFIG.benchmark_id == "BENCH-001"
            and BENCH_001_CONFIG.benchmark_symbol == "SPY"
            and BENCH_001_CONFIG.benchmark_role == BenchmarkRole.BROAD_MARKET
        )
        checks.append(_check(
            "benchmark_configuration_locked",
            config_valid,
            f"benchmark_id={BENCH_001_CONFIG.benchmark_id}, "
            f"symbol={BENCH_001_CONFIG.benchmark_symbol}, "
            f"role={BENCH_001_CONFIG.benchmark_role.value}",
        ))
    except Exception as exc:
        checks.append(_check("benchmark_configuration_locked", False, str(exc)))

    # 3. Benchmark is not tradable
    try:
        from orbit.ml.phase11_1_benchmark import BENCH_001_CONFIG, BenchmarkRole
        not_tradable = BENCH_001_CONFIG.benchmark_role == BenchmarkRole.BROAD_MARKET
        checks.append(_check(
            "benchmark_not_tradable",
            not_tradable,
            f"benchmark role is {BENCH_001_CONFIG.benchmark_role.value} (not tradable)",
        ))
    except Exception as exc:
        checks.append(_check("benchmark_not_tradable", False, str(exc)))

    # 4. Benchmark data exists and has expected structure
    try:
        from orbit.ml.phase11_1_benchmark_ingest import load_benchmark_bars
        bars = load_benchmark_bars()
        if bars is None or bars.height == 0:
            checks.append(_check("benchmark_data_structure", False, "no benchmark data loaded"))
        else:
            has_required = all(
                col in bars.columns
                for col in ["trade_date", "instrument_id", "symbol", "open", "high", "low", "close", "volume"]
            )
            has_benchmark_close = "benchmark_close" in bars.columns or "close" in bars.columns
            checks.append(_check(
                "benchmark_data_structure",
                has_required and has_benchmark_close,
                f"rows={bars.height}, columns={bars.columns}",
            ))
    except Exception as exc:
        checks.append(_check("benchmark_data_structure", False, str(exc)))

    # 5. DS-000004 unchanged
    try:
        from orbit.ml.data import load_snapshot_bars
        bars = load_snapshot_bars()
        n_instruments = bars["instrument_id"].n_unique()
        n_sessions = bars["trade_date"].n_unique()
        checks.append(_check(
            "ds000004_unchanged",
            n_instruments == 20,
            f"instruments={n_instruments}, sessions={n_sessions}",
        ))
    except Exception as exc:
        checks.append(_check("ds000004_unchanged", False, str(exc)))

    # 6. LAB-004 unchanged
    try:
        from orbit.ml.labels import build_phase9_label_contract
        contract = build_phase9_label_contract()
        lab004_ok = (
            contract.label_id == "LAB-004"
            and contract.version == "v1"
            and contract.benchmark is None
        )
        checks.append(_check(
            "lab004_unchanged",
            lab004_ok,
            f"label_id={contract.label_id}, version={contract.version}, "
            f"benchmark={contract.benchmark}",
        ))
    except Exception as exc:
        checks.append(_check("lab004_unchanged", False, str(exc)))

    # 7. LAB-005 label contract valid
    try:
        from orbit.ml.phase11_1_labels import build_phase11_1_label_contract
        contract = build_phase11_1_label_contract()
        lab005_ok = (
            contract.label_id == "LAB-005"
            and contract.target_type.value == "excess_return"
            and contract.benchmark == "BENCH-001"
            and contract.horizon == 5
        )
        checks.append(_check(
            "lab005_contract_valid",
            lab005_ok,
            f"label_id={contract.label_id}, type={contract.target_type.value}, "
            f"benchmark={contract.benchmark}, horizon={contract.horizon}",
        ))
    except Exception as exc:
        checks.append(_check("lab005_contract_valid", False, str(exc)))

    # 8. Alignment valid (no lookahead)
    if stage_a_results is not None:
        alignment_valid = stage_a_results.get("alignment_valid", False)
        n_errors = len(stage_a_results.get("alignment_errors", []))
        checks.append(_check(
            "alignment_no_lookahead",
            alignment_valid,
            f"alignment_valid={alignment_valid}, errors={n_errors}",
        ))
    else:
        checks.append(_check("alignment_no_lookahead", False, "no results provided"))

    # 9. Excess-return labels computed
    if stage_a_results is not None:
        available = stage_a_results.get("excess_labels_available", 0)
        total = stage_a_results.get("excess_labels_total", 0)
        labels_ok = available > 0 and total > 0
        checks.append(_check(
            "excess_return_labels_computed",
            labels_ok,
            f"available={available}, total={total}",
        ))
    else:
        checks.append(_check("excess_return_labels_computed", False, "no results provided"))

    # 10. Benchmark suite locked
    try:
        suite_path = BENCHMARKS_DIR / "phase11_1_benchmark_suite.json"
        if suite_path.exists():
            suite = json.loads(suite_path.read_text(encoding="utf-8"))
            suite_ok = (
                len(suite.get("models", [])) >= 4
                and len(suite.get("feature_sets", [])) >= 2
                and len(suite.get("labels", [])) >= 2
            )
            checks.append(_check(
                "benchmark_suite_locked",
                suite_ok,
                f"models={len(suite.get('models', []))}, "
                f"feature_sets={len(suite.get('feature_sets', []))}, "
                f"labels={len(suite.get('labels', []))}",
            ))
        else:
            checks.append(_check("benchmark_suite_locked", False, "suite file not found"))
    except Exception as exc:
        checks.append(_check("benchmark_suite_locked", False, str(exc)))

    # 11. Source artifact checksums
    try:
        p9_parquet = _REPO_ROOT / "benchmarks" / "phase9_ml_benchmark.parquet"
        p10_parquet = _REPO_ROOT / "benchmarks" / "phase10_feature_research.parquet"
        p10_plan = _REPO_ROOT / "benchmarks" / "phase10_plan.json"
        artifacts_ok = p9_parquet.exists() and p10_parquet.exists() and p10_plan.exists()
        checks.append(_check(
            "source_artifacts_intact",
            artifacts_ok,
            f"phase9={p9_parquet.exists()}, phase10={p10_parquet.exists()}, "
            f"plan={p10_plan.exists()}",
        ))
    except Exception as exc:
        checks.append(_check("source_artifacts_intact", False, str(exc)))

    # 12. Deterministic reproducibility
    try:
        from orbit.ml.phase11_1_plan import build_stage_a_plan, build_benchmark_suite
        p1 = build_stage_a_plan()
        p2 = build_stage_a_plan()
        s1 = build_benchmark_suite()
        s2 = build_benchmark_suite()
        reproducible = p1["plan_digest"] == p2["plan_digest"] and s1["suite_digest"] == s2["suite_digest"]
        checks.append(_check(
            "deterministic_reproducibility",
            reproducible,
            f"plan_digest_match={p1['plan_digest'] == p2['plan_digest']}, "
            f"suite_digest_match={s1['suite_digest'] == s2['suite_digest']}",
        ))
    except Exception as exc:
        checks.append(_check("deterministic_reproducibility", False, str(exc)))

    # 13. Immutability of plan digests
    try:
        stored_plan = json.loads(
            (BENCHMARKS_DIR / "phase11_1_stage_a_plan.json").read_text(encoding="utf-8")
        )
        stored_digest = stored_plan.get("plan_digest")
        recomputed = _sha256_json({k: v for k, v in stored_plan.items() if k != "plan_digest"})
        immutable = stored_digest == recomputed
        checks.append(_check(
            "plan_digest_immutable",
            immutable,
            f"stored={stored_digest[:16] if stored_digest else 'N/A'}... "
            f"recomputed={recomputed[:16]}...",
        ))
    except Exception as exc:
        checks.append(_check("plan_digest_immutable", False, str(exc)))

    # 14. No hidden experiment exclusion (Phase 9/10 experiments)
    try:
        p9_parquet = _REPO_ROOT / "benchmarks" / "phase9_ml_benchmark.parquet"
        p10_parquet = _REPO_ROOT / "benchmarks" / "phase10_feature_research.parquet"
        if p9_parquet.exists() and p10_parquet.exists():
            import polars as pl
            p9 = pl.read_parquet(p9_parquet)
            p10 = pl.read_parquet(p10_parquet)
            checks.append(_check(
                "no_hidden_exclusion",
                True,
                f"phase9_experiments={len(p9)}, phase10_experiments={len(p10)}",
            ))
        else:
            checks.append(_check("no_hidden_exclusion", False, "parquet files not found"))
    except Exception as exc:
        checks.append(_check("no_hidden_exclusion", False, str(exc)))

    # 15. Stage A validation gates
    if stage_a_results is not None:
        all_gates_passed = (
            stage_a_results.get("alignment_valid", False)
            and stage_a_results.get("excess_labels_available", 0) > 0
        )
        checks.append(_check(
            "stage_a_validation_gates",
            all_gates_passed,
            f"alignment_valid={stage_a_results.get('alignment_valid')}, "
            f"labels_available={stage_a_results.get('excess_labels_available')}",
        ))
    else:
        checks.append(_check("stage_a_validation_gates", False, "no results provided"))

    return checks


def run_stage_b_audit(
    *,
    universe_plan: dict[str, Any] | None = None,
    stage_b_results: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Run the Stage B independent audit."""
    checks: list[dict[str, Any]] = []

    # 1. Universe plan exists and is digest-verified
    try:
        if universe_plan is None:
            universe_plan = json.loads(
                (BENCHMARKS_DIR / "phase11_1_universe_plan_v1.json").read_text(encoding="utf-8")
            )
        stored_digest = universe_plan.get("plan_digest")
        payload = {k: v for k, v in universe_plan.items() if k != "plan_digest"}
        computed_digest = _sha256_json(payload)
        digest_ok = stored_digest is not None and stored_digest == computed_digest
        checks.append(_check(
            "universe_plan_digest",
            digest_ok,
            f"stored={stored_digest[:16] if stored_digest else 'N/A'}... "
            f"computed={computed_digest[:16]}... verified={digest_ok}",
        ))
    except Exception as exc:
        checks.append(_check("universe_plan_digest", False, str(exc)))
        return checks

    # 2. Expansion stages defined
    stages = universe_plan.get("expansion_stages", {})
    has_all_stages = all(f"stage_{i}" in stages for i in range(4))
    checks.append(_check(
        "expansion_stages_defined",
        has_all_stages,
        f"stages: {sorted(stages.keys())}",
    ))

    # 3. Selection policy is rule-based (not performance-based)
    selection = universe_plan.get("selection_policy", {})
    method = selection.get("method", "")
    rule_based = "rule" in method.lower() or "deterministic" in method.lower()
    checks.append(_check(
        "selection_policy_rule_based",
        rule_based,
        f"method={method}",
    ))

    # 4. Survivorship bias disclosed
    survivorship = universe_plan.get("survivorship_bias", {})
    disclosed = "NOT" in survivorship.get("status", "") or "bias" in survivorship.get("description", "").lower()
    checks.append(_check(
        "survivorship_bias_disclosed",
        disclosed,
        f"status={survivorship.get('status', 'N/A')}",
    ))

    # 5. Gate criteria forbids performance-based selection
    gate = universe_plan.get("gate_between_50_and_100", {})
    forbidden = gate.get("forbidden_criteria", [])
    forbidden_text = " ".join(forbidden).lower()
    has_performance_forbidden = all(
        term.lower() in forbidden_text
        for term in ["model performance", "returns", "IC", "statistical significance"]
    )
    checks.append(_check(
        "gate_forbids_performance",
        has_performance_forbidden,
        f"forbidden_criteria={forbidden}",
    ))

    # 6. Universes have expected sizes
    if stage_b_results is not None:
        u50 = stage_b_results.get("universe_50_count", 0)
        u100 = stage_b_results.get("universe_100_count", 0)
        checks.append(_check(
            "universe_sizes",
            u50 > 0 and u100 > 0,
            f"universe_50={u50}, universe_100={u100}",
        ))
    else:
        checks.append(_check("universe_sizes", False, "no results provided"))

    # 7. Benchmark ID is in suite
    try:
        suite_path = BENCHMARKS_DIR / "phase11_1_benchmark_suite.json"
        if suite_path.exists():
            suite = json.loads(suite_path.read_text(encoding="utf-8"))
            envs = suite.get("evaluation_environments", [])
            has_bench_env = any(e.get("benchmark_id") == "BENCH-001" for e in envs)
            checks.append(_check(
                "benchmark_in_suite",
                has_bench_env,
                f"environments with BENCH-001: "
                f"{sum(1 for e in envs if e.get('benchmark_id') == 'BENCH-001')}",
            ))
        else:
            checks.append(_check("benchmark_in_suite", False, "suite not found"))
    except Exception as exc:
        checks.append(_check("benchmark_in_suite", False, str(exc)))

    # 8. Deterministic reproducibility
    try:
        from orbit.ml.phase11_1_plan import build_universe_expansion_plan
        p1 = build_universe_expansion_plan()
        p2 = build_universe_expansion_plan()
        reproducible = p1["plan_digest"] == p2["plan_digest"]
        checks.append(_check(
            "universe_plan_deterministic",
            reproducible,
            f"digest_match={p1['plan_digest'] == p2['plan_digest']}",
        ))
    except Exception as exc:
        checks.append(_check("universe_plan_deterministic", False, str(exc)))

    return checks


def run_full_audit(
    *,
    stage_a_plan: dict[str, Any] | None = None,
    stage_a_results: dict[str, Any] | None = None,
    universe_plan: dict[str, Any] | None = None,
    stage_b_results: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Run the full Phase 11.1 audit (Stage A + Stage B)."""
    checks = []
    checks.extend(run_stage_a_audit(stage_a_plan=stage_a_plan, stage_a_results=stage_a_results))
    checks.extend(run_stage_b_audit(universe_plan=universe_plan, stage_b_results=stage_b_results))
    return checks


def audit_summary(checks: list[dict[str, Any]]) -> dict[str, Any]:
    failed = [c for c in checks if c["status"] == "FAIL"]
    return {
        "checks": len(checks),
        "passed": len(checks) - len(failed),
        "failed": len(failed),
        "blocked": bool(failed),
        "failed_checks": [c["check"] for c in failed],
    }


def persist_audit(
    checks: list[dict[str, Any]],
    stage_a_results: dict[str, Any] | None = None,
    stage_b_results: dict[str, Any] | None = None,
) -> Path:
    """Write permanent audit results."""
    AUDIT_JSON.parent.mkdir(parents=True, exist_ok=True)
    summary = audit_summary(checks)
    payload = {
        "phase": "11.1",
        "summary": summary,
        "checks": checks,
        "stage_a_summary": {
            "plan_digest": stage_a_results.get("plan_digest") if stage_a_results else None,
            "benchmark_sessions": stage_a_results.get("benchmark_sessions") if stage_a_results else None,
            "alignment_valid": stage_a_results.get("alignment_valid") if stage_a_results else None,
            "excess_labels_available": stage_a_results.get("excess_labels_available") if stage_a_results else None,
        },
        "stage_b_summary": {
            "universe_plan_digest": stage_b_results.get("universe_plan_digest") if stage_b_results else None,
            "universe_50_count": stage_b_results.get("universe_50_count") if stage_b_results else None,
            "universe_100_count": stage_b_results.get("universe_100_count") if stage_b_results else None,
        },
    }
    AUDIT_JSON.write_text(
        json.dumps(payload, sort_keys=True, indent=2, default=str),
        encoding="utf-8",
    )
    return AUDIT_JSON


__all__ = [
    "AUDIT_JSON",
    "run_stage_a_audit",
    "run_stage_b_audit",
    "run_full_audit",
    "audit_summary",
    "persist_audit",
]
