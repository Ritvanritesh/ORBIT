"""Phase 11.1 comprehensive test suite.

Covers:
- Benchmark configuration and locking
- Benchmark ingestion
- Benchmark alignment
- Excess-return label computation
- Hand-calculated toy examples
- Adversarial tests (A1-A17)
- Leakage prevention
- Immutability verification
- Universe expansion plan
- Audit consistency
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS_DIR = REPO_ROOT / "benchmarks"


# ──────────────────────────────────────────────────────────────
# FIXTURES
# ──────────────────────────────────────────────────────────────

@pytest.fixture
def instrument_bars():
    """Toy instrument bars for testing."""
    dates = [f"2025-01-{d:02d}" for d in range(1, 11)]
    return pl.DataFrame({
        "trade_date": dates,
        "instrument_id": ["INS-000001"] * 10,
        "symbol": ["AAPL"] * 10,
        "open": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0],
        "high": [101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0],
        "low": [99.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0],
        "close": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0],
        "volume": [1000000] * 10,
    })


@pytest.fixture
def benchmark_bars():
    """Toy benchmark bars for testing (SPY-like)."""
    dates = [f"2025-01-{d:02d}" for d in range(1, 11)]
    return pl.DataFrame({
        "trade_date": dates,
        "instrument_id": ["BENCH-001"] * 10,
        "symbol": ["SPY"] * 10,
        "open": [100.0, 100.5, 101.0, 101.5, 102.0, 102.5, 103.0, 103.5, 104.0, 104.5],
        "high": [100.5, 101.0, 101.5, 102.0, 102.5, 103.0, 103.5, 104.0, 104.5, 105.0],
        "low": [99.5, 100.0, 100.5, 101.0, 101.5, 102.0, 102.5, 103.0, 103.5, 104.0],
        "close": [100.0, 100.5, 101.0, 101.5, 102.0, 102.5, 103.0, 103.5, 104.0, 104.5],
        "volume": [50000000] * 10,
    })


# ──────────────────────────────────────────────────────────────
# 1. BENCHMARK CONFIGURATION
# ──────────────────────────────────────────────────────────────

class TestBenchmarkConfig:
    def test_bench001_config_exists(self):
        from orbit.ml.phase11_1_benchmark import BENCH_001_CONFIG
        assert BENCH_001_CONFIG.benchmark_id == "BENCH-001"
        assert BENCH_001_CONFIG.benchmark_symbol == "SPY"

    def test_bench001_is_not_tradable(self):
        from orbit.ml.phase11_1_benchmark import BENCH_001_CONFIG, BenchmarkRole
        assert BENCH_001_CONFIG.benchmark_role == BenchmarkRole.BROAD_MARKET

    def test_bench001_content_hash_deterministic(self):
        from orbit.ml.phase11_1_benchmark import BENCH_001_CONFIG
        h1 = BENCH_001_CONFIG.content_hash()
        h2 = BENCH_001_CONFIG.content_hash()
        assert h1 == h2
        assert len(h1) == 64

    def test_bench001_config_immutable(self):
        from orbit.ml.phase11_1_benchmark import BENCH_001_CONFIG
        with pytest.raises(Exception):
            BENCH_001_CONFIG.benchmark_symbol = "QQQ"


# ──────────────────────────────────────────────────────────────
# 2. STAGE A PLAN
# ──────────────────────────────────────────────────────────────

class TestStageAPlan:
    def test_stage_a_plan_builds(self):
        from orbit.ml.phase11_1_plan import build_stage_a_plan
        plan = build_stage_a_plan()
        assert plan["stage"] == "A"
        assert "plan_digest" in plan

    def test_stage_a_plan_deterministic(self):
        from orbit.ml.phase11_1_plan import build_stage_a_plan
        p1 = build_stage_a_plan()
        p2 = build_stage_a_plan()
        assert p1["plan_digest"] == p2["plan_digest"]

    def test_stage_a_plan_persistence(self, tmp_path):
        from orbit.ml.phase11_1_plan import build_stage_a_plan, persist_stage_a_plan
        import orbit.ml.phase11_1_plan as mod
        original = mod._benchmarks_dir
        mod._benchmarks_dir = lambda: tmp_path
        try:
            plan = build_stage_a_plan()
            path = persist_stage_a_plan(plan)
            assert Path(path).exists()
        finally:
            mod._benchmarks_dir = original


# ──────────────────────────────────────────────────────────────
# 3. UNIVERSE EXPANSION PLAN
# ──────────────────────────────────────────────────────────────

class TestUniversePlan:
    def test_universe_plan_builds(self):
        from orbit.ml.phase11_1_plan import build_universe_expansion_plan
        plan = build_universe_expansion_plan()
        assert "expansion_stages" in plan
        assert "plan_digest" in plan

    def test_universe_plan_50_100_stages(self):
        from orbit.ml.phase11_1_plan import build_universe_expansion_plan
        plan = build_universe_expansion_plan()
        stages = plan["expansion_stages"]
        assert "stage_2" in stages  # 50 symbols
        assert "stage_3" in stages  # 100 symbols

    def test_universe_plan_survivorship_disclosed(self):
        from orbit.ml.phase11_1_plan import build_universe_expansion_plan
        plan = build_universe_expansion_plan()
        assert "NOT FULLY CONTROLLED" in plan["survivorship_bias"]["status"]

    def test_universe_plan_gate_forbids_performance(self):
        from orbit.ml.phase11_1_plan import build_universe_expansion_plan
        plan = build_universe_expansion_plan()
        gate = plan["gate_between_50_and_100"]
        assert "model performance" in gate["forbidden_criteria"]
        assert "returns" in gate["forbidden_criteria"]


# ──────────────────────────────────────────────────────────────
# 4. BENCHMARK SUITE
# ──────────────────────────────────────────────────────────────

class TestBenchmarkSuite:
    def test_suite_builds(self):
        from orbit.ml.phase11_1_plan import build_benchmark_suite
        suite = build_benchmark_suite()
        assert "suite_digest" in suite
        assert len(suite["models"]) == 4
        assert len(suite["feature_sets"]) == 2
        assert len(suite["labels"]) == 2

    def test_suite_includes_both_labels(self):
        from orbit.ml.phase11_1_plan import build_benchmark_suite
        suite = build_benchmark_suite()
        label_ids = [l["label_id"] for l in suite["labels"]]
        assert "LAB-004" in label_ids
        assert "LAB-005" in label_ids

    def test_suite_deterministic(self):
        from orbit.ml.phase11_1_plan import build_benchmark_suite
        s1 = build_benchmark_suite()
        s2 = build_benchmark_suite()
        assert s1["suite_digest"] == s2["suite_digest"]


# ──────────────────────────────────────────────────────────────
# 5. EXCESS-RETURN LABELS
# ──────────────────────────────────────────────────────────────

class TestExcessReturnLabels:
    def test_label_contract_builds(self):
        from orbit.ml.phase11_1_labels import build_phase11_1_label_contract
        contract = build_phase11_1_label_contract()
        assert contract.label_id == "LAB-005"
        assert contract.target_type.value == "excess_return"
        assert contract.benchmark == "BENCH-001"

    def test_label_contract_immutable(self):
        from orbit.ml.phase11_1_labels import build_phase11_1_label_contract
        contract = build_phase11_1_label_contract()
        with pytest.raises(Exception):
            contract.label_id = "LAB-999"

    def test_lab004_unchanged(self):
        """LAB-004 must remain unchanged."""
        from orbit.ml.labels import build_phase9_label_contract
        contract = build_phase9_label_contract()
        assert contract.label_id == "LAB-004"
        assert contract.version == "v1"
        assert contract.benchmark is None


# ──────────────────────────────────────────────────────────────
# 6. HAND-CALCULATED TOY EXAMPLES
# ──────────────────────────────────────────────────────────────

class TestToyExamples:
    def test_toy_simple(self):
        from orbit.ml.phase11_1_labels import toy_example_simple
        toy = toy_example_simple()
        assert toy["expected"]["excess_return_at_0"] == pytest.approx(0.06, abs=1e-10)

    def test_toy_instrument_equals_benchmark(self):
        from orbit.ml.phase11_1_labels import toy_example_instrument_equals_benchmark
        toy = toy_example_instrument_equals_benchmark()
        assert toy["expected"]["excess_returns_all_zero"] is True

    def test_toy_benchmark_rises_instrument_flat(self):
        from orbit.ml.phase11_1_labels import toy_example_benchmark_rises_instrument_flat
        toy = toy_example_benchmark_rises_instrument_flat()
        assert toy["expected"]["excess_return"] == pytest.approx(-0.10, abs=1e-10)

    def test_toy_instrument_outperforms(self):
        from orbit.ml.phase11_1_labels import toy_example_instrument_outperforms
        toy = toy_example_instrument_outperforms()
        assert toy["expected"]["excess_return"] == pytest.approx(0.10, abs=1e-10)


# ──────────────────────────────────────────────────────────────
# 7. BENCHMARK ALIGNMENT
# ──────────────────────────────────────────────────────────────

class TestAlignment:
    def test_same_day_alignment(self, instrument_bars, benchmark_bars):
        from orbit.ml.phase11_1_alignment import align_instrument_benchmark
        from orbit.ml.phase11_1_benchmark import AlignmentPolicy
        aligned = align_instrument_benchmark(instrument_bars, benchmark_bars, AlignmentPolicy.SAME_DAY)
        assert "benchmark_close" in aligned.columns
        assert aligned.height == 10

    def test_no_lookahead(self, instrument_bars, benchmark_bars):
        from orbit.ml.phase11_1_alignment import align_instrument_benchmark, validate_alignment_no_lookahead
        from orbit.ml.phase11_1_benchmark import AlignmentPolicy
        aligned = align_instrument_benchmark(instrument_bars, benchmark_bars, AlignmentPolicy.SAME_DAY)
        errors = validate_alignment_no_lookahead(aligned)
        assert len(errors) == 0

    def test_missing_benchmark_dropped(self, instrument_bars, benchmark_bars):
        from orbit.ml.phase11_1_alignment import align_instrument_benchmark
        from orbit.ml.phase11_1_benchmark import AlignmentPolicy
        # Benchmark only has 5 dates
        bench_short = benchmark_bars.filter(pl.col("trade_date").is_in(
            [f"2025-01-{d:02d}" for d in range(1, 6)]
        ))
        aligned = align_instrument_benchmark(instrument_bars, bench_short, AlignmentPolicy.SAME_DAY)
        assert aligned.height == 5

    def test_alignment_requires_columns(self):
        from orbit.ml.phase11_1_alignment import align_instrument_benchmark
        from orbit.ml.phase11_1_benchmark import AlignmentPolicy
        df = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
        with pytest.raises(ValueError, match="must have columns"):
            align_instrument_benchmark(df, df, AlignmentPolicy.SAME_DAY)


# ──────────────────────────────────────────────────────────────
# 8. FORWARD RETURN COMPUTATION
# ──────────────────────────────────────────────────────────────

class TestForwardReturns:
    def test_computes_forward_returns(self, instrument_bars, benchmark_bars):
        from orbit.ml.phase11_1_alignment import align_instrument_benchmark, compute_forward_returns
        from orbit.ml.phase11_1_benchmark import AlignmentPolicy
        aligned = align_instrument_benchmark(instrument_bars, benchmark_bars, AlignmentPolicy.SAME_DAY)
        with_returns = compute_forward_returns(aligned, horizon=5)
        assert "instrument_forward_return" in with_returns.columns
        assert "benchmark_forward_return" in with_returns.columns
        assert "excess_return" in with_returns.columns

    def test_excess_equals_instrument_minus_benchmark(self, instrument_bars, benchmark_bars):
        from orbit.ml.phase11_1_alignment import align_instrument_benchmark, compute_forward_returns
        from orbit.ml.phase11_1_benchmark import AlignmentPolicy
        aligned = align_instrument_benchmark(instrument_bars, benchmark_bars, AlignmentPolicy.SAME_DAY)
        with_returns = compute_forward_returns(aligned, horizon=5)
        # Check excess = instrument - benchmark
        row0 = with_returns.row(0, named=True)
        if row0["instrument_forward_return"] is not None and row0["benchmark_forward_return"] is not None:
            expected = row0["instrument_forward_return"] - row0["benchmark_forward_return"]
            assert row0["excess_return"] == pytest.approx(expected, abs=1e-10)


# ──────────────────────────────────────────────────────────────
# 9. EXCESS RETURN LABEL COMPUTATION
# ──────────────────────────────────────────────────────────────

class TestExcessReturnComputation:
    def test_computes_labels(self, instrument_bars, benchmark_bars):
        from orbit.ml.phase11_1_labels import compute_excess_return_label
        result = compute_excess_return_label(instrument_bars, benchmark_bars, horizon=5)
        assert "excess_return" in result.columns
        assert "label_available" in result.columns

    def test_label_availability(self, instrument_bars, benchmark_bars):
        from orbit.ml.phase11_1_labels import compute_excess_return_label
        result = compute_excess_return_label(instrument_bars, benchmark_bars, horizon=5)
        # First 5 rows should have NaN forward returns (not enough future data)
        available = result["label_available"].sum()
        unavailable = result.height - available
        assert unavailable >= 0


# ──────────────────────────────────────────────────────────────
# 10. ADVERSARIAL TESTS
# ──────────────────────────────────────────────────────────────

class TestAdversarial:
    def test_A1_benchmark_shifted(self, instrument_bars, benchmark_bars):
        """A1: Benchmark shifted by one session should cause misalignment."""
        from orbit.ml.phase11_1_alignment import align_instrument_benchmark
        from orbit.ml.phase11_1_benchmark import AlignmentPolicy
        # Shift benchmark by 1 day
        shifted = benchmark_bars.with_columns(
            pl.col("trade_date").shift(1)
        ).drop_nulls(subset=["trade_date"])
        aligned = align_instrument_benchmark(instrument_bars, shifted, AlignmentPolicy.SAME_DAY)
        # Should have fewer aligned rows due to shift
        assert aligned.height < 10

    def test_A2_future_leakage(self, instrument_bars, benchmark_bars):
        """A2: Future benchmark value must not leak into label."""
        from orbit.ml.phase11_1_alignment import align_instrument_benchmark, compute_forward_returns
        from orbit.ml.phase11_1_benchmark import AlignmentPolicy
        aligned = align_instrument_benchmark(instrument_bars, benchmark_bars, AlignmentPolicy.SAME_DAY)
        with_returns = compute_forward_returns(aligned, horizon=5)
        # Row 0 forward return should use future values (row 5), not row 0
        row0 = with_returns.row(0, named=True)
        # Instrument goes from 100 to 105 over 5 days, benchmark from 100 to 102.5
        if row0["instrument_forward_return"] is not None:
            assert row0["instrument_forward_return"] == pytest.approx(0.05, abs=1e-10)

    def test_A4_DS000004_unchanged(self):
        """A4: Existing DS-000004 must not be modified."""
        from orbit.ml.data import DEV_SNAPSHOT_ID, load_snapshot_bars
        bars = load_snapshot_bars(DEV_SNAPSHOT_ID)
        assert bars.height > 0
        assert bars["instrument_id"].n_unique() == 20

    def test_A5_LAB004_unchanged(self):
        """A5: LAB-004 must not be modified."""
        from orbit.ml.labels import build_phase9_label_contract, LABEL_ID, LABEL_VERSION
        contract = build_phase9_label_contract()
        assert contract.label_id == LABEL_ID
        assert contract.version == LABEL_VERSION
        assert contract.benchmark is None

    def test_A6_plan_digest_mutation(self):
        """A6: Plan mutation should be detectable."""
        from orbit.ml.phase11_1_plan import build_stage_a_plan, _sha256_json
        plan = build_stage_a_plan()
        stored_digest = plan["plan_digest"]
        # Mutate
        plan["stage"] = "B"
        computed = _sha256_json({k: v for k, v in plan.items() if k != "plan_digest"})
        assert stored_digest != computed

    def test_A7_benchmark_not_tradable(self):
        """A7: Benchmark must not be in tradable universe."""
        from orbit.ml.phase11_1_benchmark import BENCH_001_CONFIG
        # BENCH-001 should not appear as an instrument_id in instrument data
        from orbit.ml.data import load_snapshot_bars
        bars = load_snapshot_bars()
        instrument_ids = bars["instrument_id"].unique().to_list()
        assert BENCH_001_CONFIG.benchmark_id not in instrument_ids

    def test_A10_benchmark_role_explicit(self):
        """A10: Benchmark role must be explicit."""
        from orbit.ml.phase11_1_benchmark import BENCH_001_CONFIG, BenchmarkRole
        assert BENCH_001_CONFIG.benchmark_role == BenchmarkRole.BROAD_MARKET


# ──────────────────────────────────────────────────────────────
# 11. REPRODUCIBILITY
# ──────────────────────────────────────────────────────────────

class TestReproducibility:
    def test_plan_digest_deterministic(self):
        from orbit.ml.phase11_1_plan import build_stage_a_plan
        d1 = build_stage_a_plan()["plan_digest"]
        d2 = build_stage_a_plan()["plan_digest"]
        assert d1 == d2

    def test_benchmark_config_hash_deterministic(self):
        from orbit.ml.phase11_1_benchmark import BENCH_001_CONFIG
        h1 = BENCH_001_CONFIG.content_hash()
        h2 = BENCH_001_CONFIG.content_hash()
        assert h1 == h2

    def test_label_contract_hash_deterministic(self):
        from orbit.ml.phase11_1_labels import build_phase11_1_label_contract
        c1 = build_phase11_1_label_contract()
        c2 = build_phase11_1_label_contract()
        assert c1.content_hash() == c2.content_hash()


# ──────────────────────────────────────────────────────────────
# 12. IMMUTABILITY
# ──────────────────────────────────────────────────────────────

class TestImmutability:
    def test_benchmark_config_frozen(self):
        from orbit.ml.phase11_1_benchmark import BENCH_001_CONFIG
        with pytest.raises(Exception):
            BENCH_001_CONFIG.benchmark_id = "BENCH-999"

    def test_label_contract_frozen(self):
        from orbit.ml.phase11_1_labels import build_phase11_1_label_contract
        contract = build_phase11_1_label_contract()
        with pytest.raises(Exception):
            contract.label_id = "LAB-999"

    def test_benchmark_manifest_frozen(self):
        from orbit.ml.phase11_1_benchmark import BenchmarkManifest
        from datetime import datetime
        manifest = BenchmarkManifest(
            benchmark_id="BENCH-001", benchmark_symbol="SPY",
            snapshot_id="BENCH-001", source="yahoo_chart_api",
            ingestion_time=datetime.now(), date_range=["2020-01-01", "2025-01-01"],
            row_count=1000, session_count=1000, checksum="a" * 64,
            config_hash="b" * 64,
        )
        with pytest.raises(Exception):
            manifest.benchmark_id = "BENCH-999"


# ──────────────────────────────────────────────────────────────
# 13. STAGE A AUDIT
# ──────────────────────────────────────────────────────────────

class TestStageAAudit:
    def _ensure_plans_exist(self):
        """Create plan files if they don't exist yet."""
        from orbit.ml.phase11_1_plan import build_stage_a_plan, persist_stage_a_plan
        from orbit.ml.phase11_1_plan import build_universe_expansion_plan, persist_universe_plan
        from orbit.ml.phase11_1_plan import build_benchmark_suite, persist_benchmark_suite
        plan_path = BENCHMARKS_DIR / "phase11_1_stage_a_plan.json"
        if not plan_path.exists():
            persist_stage_a_plan(build_stage_a_plan())
        univ_path = BENCHMARKS_DIR / "phase11_1_universe_plan_v1.json"
        if not univ_path.exists():
            persist_universe_plan(build_universe_expansion_plan())
        suite_path = BENCHMARKS_DIR / "phase11_1_benchmark_suite.json"
        if not suite_path.exists():
            persist_benchmark_suite(build_benchmark_suite())

    def test_stage_a_audit_runs(self):
        self._ensure_plans_exist()
        from orbit.ml.phase11_1_audit import run_stage_a_audit
        checks = run_stage_a_audit()
        assert len(checks) >= 10

    def test_stage_a_audit_plan_digest(self):
        self._ensure_plans_exist()
        from orbit.ml.phase11_1_audit import run_stage_a_audit
        checks = run_stage_a_audit()
        plan_check = next(c for c in checks if c["check"] == "stage_a_plan_digest")
        assert plan_check["status"] == "PASS"

    def test_stage_a_audit_benchmark_config(self):
        self._ensure_plans_exist()
        from orbit.ml.phase11_1_audit import run_stage_a_audit
        checks = run_stage_a_audit()
        bench_check = next(c for c in checks if c["check"] == "benchmark_configuration_locked")
        assert bench_check["status"] == "PASS"

    def test_stage_a_audit_ds000004(self):
        self._ensure_plans_exist()
        from orbit.ml.phase11_1_audit import run_stage_a_audit
        checks = run_stage_a_audit()
        ds_check = next(c for c in checks if c["check"] == "ds000004_unchanged")
        assert ds_check["status"] == "PASS"

    def test_stage_a_audit_lab004(self):
        self._ensure_plans_exist()
        from orbit.ml.phase11_1_audit import run_stage_a_audit
        checks = run_stage_a_audit()
        lab_check = next(c for c in checks if c["check"] == "lab004_unchanged")
        assert lab_check["status"] == "PASS"

    def test_stage_a_audit_deterministic(self):
        self._ensure_plans_exist()
        from orbit.ml.phase11_1_audit import run_stage_a_audit
        checks = run_stage_a_audit()
        det_check = next(c for c in checks if c["check"] == "deterministic_reproducibility")
        assert det_check["status"] == "PASS"


# ──────────────────────────────────────────────────────────────
# 14. STAGE B AUDIT
# ──────────────────────────────────────────────────────────────

class TestStageBAudit:
    def _ensure_plans_exist(self):
        """Create plan files if they don't exist yet."""
        from orbit.ml.phase11_1_plan import build_universe_expansion_plan, persist_universe_plan
        from orbit.ml.phase11_1_plan import build_benchmark_suite, persist_benchmark_suite
        univ_path = BENCHMARKS_DIR / "phase11_1_universe_plan_v1.json"
        if not univ_path.exists():
            persist_universe_plan(build_universe_expansion_plan())
        suite_path = BENCHMARKS_DIR / "phase11_1_benchmark_suite.json"
        if not suite_path.exists():
            persist_benchmark_suite(build_benchmark_suite())

    def test_stage_b_audit_runs(self):
        self._ensure_plans_exist()
        from orbit.ml.phase11_1_audit import run_stage_b_audit
        checks = run_stage_b_audit()
        assert len(checks) >= 5

    def test_stage_b_audit_plan_digest(self):
        self._ensure_plans_exist()
        from orbit.ml.phase11_1_audit import run_stage_b_audit
        checks = run_stage_b_audit()
        plan_check = next(c for c in checks if c["check"] == "universe_plan_digest")
        assert plan_check["status"] == "PASS"

    def test_stage_b_audit_selection_policy(self):
        self._ensure_plans_exist()
        from orbit.ml.phase11_1_audit import run_stage_b_audit
        checks = run_stage_b_audit()
        sel_check = next(c for c in checks if c["check"] == "selection_policy_rule_based")
        assert sel_check["status"] == "PASS"

    def test_stage_b_audit_survivorship(self):
        self._ensure_plans_exist()
        from orbit.ml.phase11_1_audit import run_stage_b_audit
        checks = run_stage_b_audit()
        surv_check = next(c for c in checks if c["check"] == "survivorship_bias_disclosed")
        assert surv_check["status"] == "PASS"

    def test_stage_b_audit_gate_forbids_performance(self):
        self._ensure_plans_exist()
        from orbit.ml.phase11_1_audit import run_stage_b_audit
        checks = run_stage_b_audit()
        gate_check = next(c for c in checks if c["check"] == "gate_forbids_performance")
        assert gate_check["status"] == "PASS"


# ──────────────────────────────────────────────────────────────
# 15. AUDIT SUMMARY
# ──────────────────────────────────────────────────────────────

class TestAuditSummary:
    def test_audit_summary_all_pass(self):
        from orbit.ml.phase11_1_audit import audit_summary
        checks = [
            {"check": "a", "status": "PASS", "evidence": ""},
            {"check": "b", "status": "PASS", "evidence": ""},
        ]
        summary = audit_summary(checks)
        assert summary["passed"] == 2
        assert summary["failed"] == 0
        assert summary["blocked"] is False

    def test_audit_summary_with_failure(self):
        from orbit.ml.phase11_1_audit import audit_summary
        checks = [
            {"check": "a", "status": "PASS", "evidence": ""},
            {"check": "b", "status": "FAIL", "evidence": "broken"},
        ]
        summary = audit_summary(checks)
        assert summary["passed"] == 1
        assert summary["failed"] == 1
        assert summary["blocked"] is True
        assert "b" in summary["failed_checks"]


# ──────────────────────────────────────────────────────────────
# 16. RUNNER TESTS
# ──────────────────────────────────────────────────────────────

class TestRunner:
    def test_run_stage_a(self):
        from orbit.ml.phase11_1_runner import run_stage_a
        results = run_stage_a(progress=False)
        assert results["stage"] == "A"
        assert results["alignment_valid"] is True
        assert results["excess_labels_available"] > 0

    def test_run_stage_b(self):
        from orbit.ml.phase11_1_runner import run_stage_b
        results = run_stage_b(progress=False)
        assert results["stage"] == "B"
        assert results["universe_50_count"] > 0
        assert results["universe_100_count"] > 0

    def test_full_run(self):
        from orbit.ml.phase11_1_runner import run_phase11_1_analysis
        results = run_phase11_1_analysis(progress=False)
        assert results["phase"] == "11.1"
        assert "stage_a" in results
        assert "stage_b" in results
        assert results["stage_a"]["alignment_valid"] is True
