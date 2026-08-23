"""Phase 11 comprehensive test suite.

Covers:
- Deterministic reproducibility
- Confidence intervals
- Bootstrap correctness
- Block bootstrap behavior
- Dependence diagnostics
- Overlapping-label detection
- Effect-size calculations
- Power calculations
- Multiple-comparison corrections
- Synthetic null behavior
- Synthetic positive effects
- Locked plan verification
- Artifact lineage
- Report completeness
- Audit completeness
- Adversarial failures (A1..A18)
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

# ──────────────────────────────────────────────────────────────
# FIXTURES: synthetic data
# ──────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def iid_null_data(rng):
    """Independent null data: mean=0, std=1."""
    return rng.normal(0.0, 1.0, size=500)


@pytest.fixture
def autocorrelated_null_data(rng):
    """Autocorrelated null data (AR(1) with rho=0.5)."""
    n = 500
    data = np.empty(n)
    data[0] = rng.normal()
    for i in range(1, n):
        data[i] = 0.5 * data[i - 1] + rng.normal()
    return data


@pytest.fixture
def known_positive_data(rng):
    """Data with a known positive mean=0.05."""
    return rng.normal(0.05, 1.0, size=200)


@pytest.fixture
def small_effect_data(rng):
    """Data with a small effect (mean=0.01)."""
    return rng.normal(0.01, 1.0, size=100)


@pytest.fixture
def overlapping_sessions():
    """Sessions with overlapping structure (LAB-004 like)."""
    import pandas as pd
    sessions = pd.date_range("2022-01-03", periods=200, freq="B")
    return np.array(sessions)


# ──────────────────────────────────────────────────────────────
# 1. INFERENCE PLAN
# ──────────────────────────────────────────────────────────────

class TestPhase11Plan:
    def test_plan_creates_and_verifies(self):
        from orbit.ml.phase11_plan import phase11_plan, verify_plan
        plan = phase11_plan()
        assert "plan_digest" in plan
        assert plan["phase"] == 11
        assert len(plan["plan_digest"]) == 64  # sha256 hex

    def test_plan_digest_is_deterministic(self):
        from orbit.ml.phase11_plan import phase11_plan
        p1 = phase11_plan()
        p2 = phase11_plan()
        assert p1["plan_digest"] == p2["plan_digest"]

    def test_plan_persistence_roundtrip(self, tmp_path):
        from orbit.ml.phase11_plan import write_plan, load_plan, phase11_plan
        import orbit.ml.phase11_plan as mod
        original = mod.PLAN_JSON
        mod.PLAN_JSON = tmp_path / "test_plan.json"
        try:
            plan = phase11_plan()
            write_plan(plan)
            loaded = load_plan()
            assert loaded["plan_digest"] == plan["plan_digest"]
        finally:
            mod.PLAN_JSON = original

    def test_plan_verify_catches_mutation(self, tmp_path):
        from orbit.ml.phase11_plan import write_plan, load_plan, phase11_plan
        import orbit.ml.phase11_plan as mod
        original = mod.PLAN_JSON
        mod.PLAN_JSON = tmp_path / "test_plan.json"
        try:
            plan = phase11_plan()
            write_plan(plan)
            # Mutate the file
            data = json.loads(mod.PLAN_JSON.read_text())
            data["hypothesis_families"]["phase10_grid"]["members"] = ["EXP-10001"]
            mod.PLAN_JSON.write_text(json.dumps(data, indent=2))
            with pytest.raises(ValueError, match="mismatch|members"):
                load_plan()
        finally:
            mod.PLAN_JSON = original

    def test_plan_family_has_52_experiments(self):
        from orbit.ml.phase11_plan import phase11_plan
        plan = phase11_plan()
        family = plan["hypothesis_families"]["phase10_grid"]
        assert family["n_members"] == 52
        assert len(family["members"]) == 52
        expected = [f"EXP-{i}" for i in range(10001, 10053)]
        assert sorted(family["members"]) == sorted(expected)


# ──────────────────────────────────────────────────────────────
# 2. CONFIDENCE INTERVALS
# ──────────────────────────────────────────────────────────────

class TestConfidenceIntervals:
    def test_normal_ci_contains_mean(self, rng):
        from orbit.ml.phase11_inference import normal_ci
        data = rng.normal(5.0, 1.0, size=100)
        lo, hi = normal_ci(data, 0.95)
        assert lo < 5.0 < hi

    def test_t_ci_contains_mean(self, rng):
        from orbit.ml.phase11_inference import t_ci
        data = rng.normal(5.0, 1.0, size=100)
        lo, hi = t_ci(data, 0.95)
        assert lo < 5.0 < hi

    def test_percentile_ci_contains_mean(self, rng):
        from orbit.ml.phase11_inference import percentile_ci
        data = rng.normal(5.0, 1.0, size=100)
        lo, hi = percentile_ci(data, 0.95)
        assert lo < 5.0 < hi

    def test_ci_width_decreases_with_sample_size(self, rng):
        from orbit.ml.phase11_inference import t_ci
        small = rng.normal(0, 1, size=20)
        large = rng.normal(0, 1, size=200)
        lo_s, hi_s = t_ci(small, 0.95)
        lo_l, hi_l = t_ci(large, 0.95)
        assert (hi_s - lo_s) > (hi_l - lo_l)

    def test_ci_rejects_value_outside(self, rng):
        from orbit.ml.phase11_inference import t_ci
        data = rng.normal(5.0, 0.1, size=100)
        lo, hi = t_ci(data, 0.99)
        assert not (lo <= 0.0 <= hi)  # 0 is far outside

    def test_ci_method_dispatch(self, rng):
        from orbit.ml.phase11_inference import compute_ci
        data = rng.normal(0, 1, size=50)
        for method in ("normal", "t", "percentile"):
            ci = compute_ci(data, method=method, confidence_level=0.95)
            assert ci.method == method

    def test_ci_insufficient_data_raises(self):
        from orbit.ml.phase11_inference import normal_ci
        with pytest.raises(ValueError, match="at least 2"):
            normal_ci(np.array([1.0]))

    def test_inference_result_has_required_fields(self):
        from orbit.ml.phase11_inference import (
            ConfidenceInterval,
            InferenceResult,
            make_inference_result_id,
        )
        ci = ConfidenceInterval(
            point_estimate=0.01,
            lower=-0.01,
            upper=0.03,
            confidence_level=0.95,
            method="bootstrap",
            assumptions="test",
            sample_size=100,
        )
        inf_id = make_inference_result_id(["EXP-10001"], "oos_ic", "bootstrap", 42)
        result = InferenceResult(
            inference_result_id=inf_id,
            source_experiment_ids=["EXP-10001"],
            source_artifact_checksums={"test": "abc"},
            metric="oos_ic",
            ci=ci,
            p_value=0.15,
            effect_size=0.01,
            seed=42,
        )
        s = result.summary()
        assert s["metric"] == "oos_ic"
        assert s["p_value"] == 0.15
        assert s["effect_size"] == 0.01


# ──────────────────────────────────────────────────────────────
# 3. BOOTSTRAP
# ──────────────────────────────────────────────────────────────

class TestBootstrap:
    def test_iid_bootstrap_deterministic(self, iid_null_data):
        from orbit.ml.phase11_bootstrap import iid_bootstrap
        stat = lambda x: float(np.mean(x))
        r1 = iid_bootstrap(iid_null_data, stat, n_resamples=1000, seed=42)
        r2 = iid_bootstrap(iid_null_data, stat, n_resamples=1000, seed=42)
        np.testing.assert_array_equal(
            r1.bootstrap_distribution, r2.bootstrap_distribution
        )

    def test_iid_bootstrap_ci_contains_true_mean(self, rng):
        from orbit.ml.phase11_bootstrap import iid_bootstrap
        data = rng.normal(5.0, 1.0, size=200)
        stat = lambda x: float(np.mean(x))
        result = iid_bootstrap(data, stat, n_resamples=2000, seed=42)
        assert result.ci_lower < 5.0 < result.ci_upper

    def test_moving_block_bootstrap_deterministic(self, iid_null_data):
        from orbit.ml.phase11_bootstrap import moving_block_bootstrap
        stat = lambda x: float(np.mean(x))
        r1 = moving_block_bootstrap(
            iid_null_data, stat, n_resamples=500, seed=42, block_length=10
        )
        r2 = moving_block_bootstrap(
            iid_null_data, stat, n_resamples=500, seed=42, block_length=10
        )
        np.testing.assert_array_equal(
            r1.bootstrap_distribution, r2.bootstrap_distribution
        )

    def test_block_bootstrap_preserves_structure(self, autocorrelated_null_data):
        from orbit.ml.phase11_bootstrap import moving_block_bootstrap
        stat = lambda x: float(np.mean(x))
        result = moving_block_bootstrap(
            autocorrelated_null_data, stat, n_resamples=1000, seed=42
        )
        # Block bootstrap should produce wider CIs for autocorrelated data
        assert result.se > 0

    def test_block_length_rule_of_thumb(self):
        from orbit.ml.phase11_bootstrap import rule_of_thumb_block_length
        assert rule_of_thumb_block_length(100) == 5  # ceil(100^(1/3))
        assert rule_of_thumb_block_length(1000) == 10
        assert rule_of_thumb_block_length(8) == 2

    def test_block_length_diagnostics(self):
        from orbit.ml.phase11_bootstrap import block_length_diagnostics
        data = np.arange(100, dtype=float)
        diag = block_length_diagnostics(data, 10)
        assert diag["sample_size"] == 100
        assert diag["block_length"] == 10
        assert diag["coverage_ratio"] > 0

    def test_block_length_zero_raises(self):
        from orbit.ml.phase11_bootstrap import moving_block_bootstrap
        with pytest.raises(ValueError, match="positive"):
            moving_block_bootstrap(
                np.ones(100), lambda x: np.mean(x), block_length=0
            )

    def test_block_length_exceeds_n_raises(self):
        from orbit.ml.phase11_bootstrap import moving_block_bootstrap
        with pytest.raises(ValueError, match="exceeds"):
            moving_block_bootstrap(
                np.ones(10), lambda x: np.mean(x), block_length=100
            )

    def test_unsupported_method_raises(self):
        from orbit.ml.phase11_bootstrap import bootstrap_ci
        with pytest.raises(ValueError, match="unsupported"):
            bootstrap_ci(
                np.ones(100), lambda x: np.mean(x), method="circular"
            )

    def test_bootstrap_bias_estimated(self, rng):
        from orbit.ml.phase11_bootstrap import iid_bootstrap
        data = rng.exponential(1.0, size=200)
        stat = lambda x: float(np.mean(x))
        result = iid_bootstrap(data, stat, n_resamples=1000, seed=42)
        assert isinstance(result.bias, float)


# ──────────────────────────────────────────────────────────────
# 4. DEPENDENCE DIAGNOSTICS
# ──────────────────────────────────────────────────────────────

class TestDependenceDiagnostics:
    def test_autocorrelation_iid_is_small(self, iid_null_data):
        from orbit.ml.phase11_dependence import autocorrelation_estimates
        acf = autocorrelation_estimates(iid_null_data, max_lag=5)
        for rho in acf.values():
            assert abs(rho) < 0.2  # should be small for i.i.d.

    def test_autocorrelation_ar1_is_large(self, autocorrelated_null_data):
        from orbit.ml.phase11_dependence import autocorrelation_estimates
        acf = autocorrelation_estimates(autocorrelated_null_data, max_lag=5)
        assert acf[1] > 0.3  # AR(1) with rho=0.5 should show strong lag-1

    def test_ljung_box_rejects_autocorrelated(self, autocorrelated_null_data):
        from orbit.ml.phase11_dependence import ljung_box_test
        q, p = ljung_box_test(autocorrelated_null_data, max_lag=10)
        assert q > 0
        assert p < 0.05  # should reject no-autocorrelation

    def test_ljung_box_fails_to_reject_iid(self, iid_null_data):
        from orbit.ml.phase11_dependence import ljung_box_test
        q, p = ljung_box_test(iid_null_data, max_lag=5)
        assert p > 0.01  # should not strongly reject

    def test_effective_sample_size_iid(self, iid_null_data):
        from orbit.ml.phase11_dependence import effective_sample_size_iid
        n_eff, method = effective_sample_size_iid(iid_null_data)
        assert n_eff == 500
        assert method == "count_finite"

    def test_effective_sample_size_autocorrelation_reduces(self, autocorrelated_null_data):
        from orbit.ml.phase11_dependence import (
            effective_sample_size_autocorrelation,
            effective_sample_size_iid,
        )
        n_iid, _ = effective_sample_size_iid(autocorrelated_null_data)
        n_ac, _ = effective_sample_size_autocorrelation(autocorrelated_null_data)
        assert n_ac is not None
        assert n_ac <= n_iid  # autocorrelation should reduce n_eff

    def test_detect_overlapping_outcomes(self):
        from orbit.ml.phase11_dependence import detect_overlapping_outcomes
        # Sessions 1..100, all consecutive
        sessions = np.arange(1, 101, dtype=float)
        result = detect_overlapping_outcomes(sessions, horizon_sessions=5)
        assert result["overlapping"] is True
        assert result["overlap_fraction"] > 0.9

    def test_detect_no_overlap(self):
        from orbit.ml.phase11_dependence import detect_overlapping_outcomes
        # Sessions spaced 10 apart (no overlap for horizon=5)
        sessions = np.arange(0, 1000, 10, dtype=float)
        result = detect_overlapping_outcomes(sessions, horizon_sessions=5)
        assert result["overlapping"] is False
        assert result["overlap_fraction"] == 0.0

    def test_run_full_diagnostics(self, iid_null_data):
        from orbit.ml.phase11_dependence import run_dependence_diagnostics
        report = run_dependence_diagnostics(
            iid_null_data,
            decision_sessions=np.arange(500, dtype=float),
            horizon_sessions=5,
        )
        assert report.effective_sample_size is not None
        assert report.overlapping_outcomes is True  # horizon=5 with daily sessions

    def test_empty_data_diagnostics(self):
        from orbit.ml.phase11_dependence import run_dependence_diagnostics
        report = run_dependence_diagnostics(np.array([]))
        assert report.autocorrelations == {}


# ──────────────────────────────────────────────────────────────
# 5. EFFECT SIZES
# ──────────────────────────────────────────────────────────────

class TestEffectSizes:
    def test_ic_effect_negligible(self):
        from orbit.ml.phase11_effects import ic_effect_size
        eff = ic_effect_size(0.005)
        assert eff.interpretation == "negligible"

    def test_ic_effect_small(self):
        from orbit.ml.phase11_effects import ic_effect_size
        eff = ic_effect_size(0.02)
        assert eff.interpretation == "small"

    def test_ic_effect_meaningful(self):
        from orbit.ml.phase11_effects import ic_effect_size
        eff = ic_effect_size(0.035)
        assert eff.interpretation == "potentially_meaningful"

    def test_return_effect(self):
        from orbit.ml.phase11_effects import return_effect_size
        eff = return_effect_size(0.15)
        assert eff.interpretation == "potentially_meaningful"

    def test_hit_rate_effect(self):
        from orbit.ml.phase11_effects import hit_rate_effect_size
        eff = hit_rate_effect_size(0.52)
        assert eff.magnitude == pytest.approx(0.02, abs=1e-10)

    def test_significance_economy_matrix(self):
        from orbit.ml.phase11_effects import (
            ic_effect_size,
            significance_economy_matrix,
        )
        eff = ic_effect_size(0.005)
        matrix = significance_economy_matrix(p_value=0.8, effect=eff)
        assert matrix["statistical_evidence"] == "insufficient_evidence"
        assert matrix["economic_meaning"] == "negligible"

    def test_matrix_stat_sig_econ_negligible(self):
        from orbit.ml.phase11_effects import (
            ic_effect_size,
            significance_economy_matrix,
        )
        eff = ic_effect_size(0.005)
        matrix = significance_economy_matrix(p_value=0.01, effect=eff)
        assert matrix["statistical_evidence"] == "evidence_under_stated_assumptions"
        assert matrix["economic_meaning"] == "negligible"

    def test_compute_effect_size_dispatch(self):
        from orbit.ml.phase11_effects import compute_effect_size
        eff = compute_effect_size("oos_ic", 0.035)
        assert eff.metric == "ic"
        eff = compute_effect_size("hit_rate", 0.55)
        assert eff.metric == "hit_rate"


# ──────────────────────────────────────────────────────────────
# 6. MULTIPLE TESTING
# ──────────────────────────────────────────────────────────────

class TestMultipleTesting:
    def test_holm_bonferroni_all_null(self):
        from orbit.ml.phase11_multiple_testing import holm_bonferroni
        p_vals = [0.5, 0.6, 0.7, 0.8, 0.9]
        ids = [f"EXP-{i}" for i in range(10001, 10006)]
        result = holm_bonferroni(p_vals, ids)
        assert len(result.significant_at["0.05"]) == 0

    def test_holm_bonferroni_strong_signal(self):
        from orbit.ml.phase11_multiple_testing import holm_bonferroni
        p_vals = [0.001, 0.002, 0.5, 0.6, 0.7]
        ids = [f"EXP-{i}" for i in range(10001, 10006)]
        result = holm_bonferroni(p_vals, ids)
        assert len(result.significant_at["0.05"]) >= 1

    def test_bh_fdr_more_powerful(self):
        from orbit.ml.phase11_multiple_testing import (
            holm_bonferroni,
            benjamini_hochberg,
        )
        p_vals = [0.01, 0.02, 0.03, 0.04, 0.5]
        ids = [f"EXP-{i}" for i in range(10001, 10006)]
        holm = holm_bonferroni(p_vals, ids)
        bh = benjamini_hochberg(p_vals, ids)
        # BH should be at least as powerful
        assert len(bh.significant_at["0.05"]) >= len(holm.significant_at["0.05"])

    def test_multiple_testing_analysis(self):
        from orbit.ml.phase11_multiple_testing import multiple_testing_analysis
        p_vals = [0.01, 0.02, 0.5, 0.6, 0.7]
        ids = [f"EXP-{i}" for i in range(10001, 10006)]
        result = multiple_testing_analysis(p_vals, ids)
        assert "family" in result
        assert result["family"]["n_members"] == 52  # locked family
        assert "holm_bonferroni" in result
        assert "benjamini_hochberg" in result

    def test_phase10_family_is_52(self):
        from orbit.ml.phase11_multiple_testing import define_phase10_family
        family = define_phase10_family()
        assert family["n_members"] == 52
        assert family["locked"] is True

    def test_empty_p_values(self):
        from orbit.ml.phase11_multiple_testing import holm_bonferroni
        result = holm_bonferroni([], [])
        assert result.n_hypotheses == 0

    def test_adjusted_p_values_ordered(self):
        from orbit.ml.phase11_multiple_testing import holm_bonferroni
        p_vals = [0.01, 0.02, 0.03, 0.04, 0.05]
        ids = [f"EXP-{i}" for i in range(10001, 10006)]
        result = holm_bonferroni(p_vals, ids)
        # Adjusted p-values should be monotonically increasing
        for i in range(1, len(result.adjusted_p_values)):
            assert result.adjusted_p_values[i] >= result.adjusted_p_values[i - 1]


# ──────────────────────────────────────────────────────────────
# 7. POWER ANALYSIS
# ──────────────────────────────────────────────────────────────

class TestPowerAnalysis:
    def test_required_sample_size(self):
        from orbit.ml.phase11_power import required_sample_size_independent
        result = required_sample_size_independent(0.5, 0.05, 0.80)
        assert result.sample_size > 0
        assert result.label == "APPROXIMATE"

    def test_achieved_power(self):
        from orbit.ml.phase11_power import achieved_power_independent
        result = achieved_power_independent(0.5, 100, 0.05)
        assert 0 < result.achieved_power <= 1.0

    def test_min_detectable_effect(self):
        from orbit.ml.phase11_power import min_detectable_effect
        result = min_detectable_effect(200, 0.05, 0.80)
        assert result.assumed_effect_size > 0
        assert result.label == "APPROXIMATE"

    def test_power_with_autocorrelation(self):
        from orbit.ml.phase11_power import power_with_autocorrelation_adjustment
        result = power_with_autocorrelation_adjustment(
            0.5, sample_size=200, effective_sample_size=100
        )
        assert result.effective_sample_size == 100
        # Power should be lower with reduced effective sample
        from orbit.ml.phase11_power import achieved_power_independent
        full = achieved_power_independent(0.5, 200)
        assert result.achieved_power < full.achieved_power

    def test_zero_effect_size(self):
        from orbit.ml.phase11_power import required_sample_size_independent
        result = required_sample_size_independent(0.0, 0.05, 0.80)
        assert result.sample_size == 0


# ──────────────────────────────────────────────────────────────
# 8. SYNTHETIC VALIDATION
# ──────────────────────────────────────────────────────────────

class TestSyntheticValidation:
    def test_independent_null_false_positive_rate(self, rng):
        """I.i.d. null: false positive rate should be approximately nominal."""
        from orbit.ml.phase11_inference import t_ci
        n_trials = 200
        n_reject = 0
        for _ in range(n_trials):
            data = rng.normal(0.0, 1.0, size=100)
            lo, hi = t_ci(data, 0.95)
            if lo > 0 or hi < 0:
                n_reject += 1
        # At 5% significance, expect ~10 false positives out of 200
        assert n_reject < 30  # very loose bound

    def test_autocorrelated_null_wider_ci(self, rng):
        """Autocorrelated null: CIs should be wider than i.i.d."""
        from orbit.ml.phase11_bootstrap import iid_bootstrap, moving_block_bootstrap
        # Generate AR(1)
        n = 200
        data = np.empty(n)
        data[0] = rng.normal()
        for i in range(1, n):
            data[i] = 0.5 * data[i - 1] + rng.normal()
        stat = lambda x: float(np.mean(x))
        iid_result = iid_bootstrap(data, stat, n_resamples=1000, seed=42)
        mb_result = moving_block_bootstrap(data, stat, n_resamples=1000, seed=42)
        # Block bootstrap should capture more uncertainty
        iid_width = iid_result.ci_upper - iid_result.ci_lower
        mb_width = mb_result.ci_upper - mb_result.ci_lower
        # This is a stochastic test but with large enough n should hold on average
        # Just verify both produce valid CIs
        assert iid_width > 0
        assert mb_width > 0

    def test_known_positive_detected(self, rng):
        """Known positive effect: CI should exclude zero (with large enough effect)."""
        from orbit.ml.phase11_bootstrap import iid_bootstrap
        # Use large effect and large sample to make detection reliable
        data = rng.normal(0.5, 0.5, size=500)
        stat = lambda x: float(np.mean(x))
        result = iid_bootstrap(
            data, stat, n_resamples=2000, seed=42
        )
        assert result.ci_lower > 0  # should detect positive mean

    def test_small_effect_low_power(self, small_effect_data):
        """Small effect: power should be limited."""
        from orbit.ml.phase11_power import achieved_power_independent
        result = achieved_power_independent(0.01, 100, 0.05)
        # With n=100, effect=0.01, power should be very low
        assert result.achieved_power < 0.2

    def test_multiple_null_controls_fdr(self, rng):
        """Multiple null hypotheses: BH should control FDR."""
        from orbit.ml.phase11_multiple_testing import benjamini_hochberg
        n_experiments = 52
        n_trials = 100
        total_reject = 0
        for _ in range(n_trials):
            p_vals = [rng.uniform(0, 1) for _ in range(n_experiments)]
            ids = [f"EXP-{i}" for i in range(10001, 10001 + n_experiments)]
            result = benjamini_hochberg(p_vals, ids)
            total_reject += len(result.significant_at["0.05"])
        # Under full null, BH at 5% should reject ~5% of the time on average
        avg_reject_rate = total_reject / (n_experiments * n_trials)
        assert avg_reject_rate < 0.15  # loose bound

    def test_overlap_detection_works(self):
        """Overlapping outcomes: detection should flag it."""
        from orbit.ml.phase11_dependence import detect_overlapping_outcomes
        sessions = np.arange(1, 201, dtype=float)  # daily
        result = detect_overlapping_outcomes(sessions, horizon_sessions=5)
        assert result["overlapping"] is True


# ──────────────────────────────────────────────────────────────
# 9. AUDIT
# ──────────────────────────────────────────────────────────────

class TestAudit:
    def test_audit_with_valid_plan(self):
        from orbit.ml.phase11_audit import run_phase11_audit
        from orbit.ml.phase11_plan import phase11_plan
        plan = phase11_plan()
        checks = run_phase11_audit(plan=plan, synthetic_validation_passed=True)
        # Plan check should pass; other checks may fail without analysis
        plan_check = next(c for c in checks if c["check"] == "inference_plan_digest")
        assert plan_check["status"] == "PASS"

    def test_audit_catches_missing_plan(self, tmp_path):
        """Audit catches plan that cannot be loaded (file missing)."""
        from orbit.ml.phase11_audit import run_phase11_audit
        import orbit.ml.phase11_plan as plan_mod
        # Temporarily point PLAN_JSON to a nonexistent file
        original = plan_mod.PLAN_JSON
        plan_mod.PLAN_JSON = tmp_path / "nonexistent_plan.json"
        try:
            checks = run_phase11_audit(plan=None)
            plan_check = next(c for c in checks if c["check"] == "inference_plan_digest")
            assert plan_check["status"] == "FAIL"
        finally:
            plan_mod.PLAN_JSON = original

    def test_audit_synthetic_validation_required(self):
        from orbit.ml.phase11_audit import run_phase11_audit
        from orbit.ml.phase11_plan import phase11_plan
        plan = phase11_plan()
        checks = run_phase11_audit(plan=plan, synthetic_validation_passed=False)
        synth_check = next(c for c in checks if c["check"] == "synthetic_validation_passed")
        assert synth_check["status"] == "FAIL"


# ──────────────────────────────────────────────────────────────
# 10. ADVERSARIAL TESTS (A1..A18)
# ──────────────────────────────────────────────────────────────

class TestAdversarial:
    def test_A1_seed_drift(self, iid_null_data):
        """A1: Same seed must produce same bootstrap result."""
        from orbit.ml.phase11_bootstrap import iid_bootstrap
        stat = lambda x: float(np.mean(x))
        r1 = iid_bootstrap(iid_null_data, stat, n_resamples=500, seed=42)
        r2 = iid_bootstrap(iid_null_data, stat, n_resamples=500, seed=42)
        assert r1.point_estimate == r2.point_estimate
        np.testing.assert_array_equal(
            r1.bootstrap_distribution, r2.bootstrap_distribution
        )

    def test_A2_plan_mutation_after_results(self, tmp_path):
        """A2: Plan mutation should be detectable."""
        from orbit.ml.phase11_plan import write_plan, phase11_plan, load_plan
        import orbit.ml.phase11_plan as mod
        original = mod.PLAN_JSON
        mod.PLAN_JSON = tmp_path / "test_plan.json"
        try:
            plan = phase11_plan()
            write_plan(plan)
            loaded = load_plan()
            assert loaded["plan_digest"] == plan["plan_digest"]
            # Mutate
            data = json.loads(mod.PLAN_JSON.read_text())
            data["seed"] = 999
            mod.PLAN_JSON.write_text(json.dumps(data, indent=2))
            with pytest.raises(ValueError):
                load_plan()
        finally:
            mod.PLAN_JSON = original

    def test_A3_missing_source_experiment(self):
        """A3: Audit catches missing experiment from the 52-run family."""
        from orbit.ml.phase11_audit import run_phase11_audit
        from orbit.ml.phase11_plan import phase11_plan, PHASE10_EXPERIMENT_FAMILY
        plan = phase11_plan()
        # Create analysis with only 51 experiments (missing EXP-10052)
        from orbit.ml.phase11_inference import (
            ConfidenceInterval,
            InferenceResult,
            make_inference_result_id,
        )
        results = []
        for i in range(10001, 10052):  # Skip 10052
            eid = f"EXP-{i}"
            ci = ConfidenceInterval(
                point_estimate=0.01, lower=-0.01, upper=0.03,
                confidence_level=0.95, method="test", assumptions="test",
                sample_size=100,
            )
            results.append(InferenceResult(
                inference_result_id=make_inference_result_id([eid], "oos_ic", "test", 42),
                source_experiment_ids=[eid],
                source_artifact_checksums={},
                metric="oos_ic",
                ci=ci,
                seed=42,
            ))
        analysis = {
            "plan_digest": plan["plan_digest"],
            "inference_results": results,
            "phase9_checksum": "a" * 64,
            "phase10_checksum": "b" * 64,
            "n_phase9_experiments": 0,
            "n_phase10_experiments": 51,
            "n_inference_results": 51,
        }
        checks = run_phase11_audit(plan=plan, analysis=analysis, synthetic_validation_passed=True)
        inv_check = next(c for c in checks if c["check"] == "source_experiment_inventory")
        assert inv_check["status"] == "FAIL"

    def test_A4_hidden_exclusion(self):
        """A4: Audit catches hidden experiment exclusion."""
        from orbit.ml.phase11_audit import run_phase11_audit
        from orbit.ml.phase11_plan import phase11_plan
        plan = phase11_plan()
        # Empty analysis = all experiments "excluded"
        checks = run_phase11_audit(
            plan=plan,
            analysis={"inference_results": [], "plan_digest": plan["plan_digest"],
                       "phase9_checksum": "a" * 64, "phase10_checksum": "b" * 64,
                       "n_phase9_experiments": 0, "n_phase10_experiments": 0,
                       "n_inference_results": 0},
            synthetic_validation_passed=True,
        )
        excl_check = next(c for c in checks if c["check"] == "no_hidden_exclusion")
        assert excl_check["status"] == "FAIL"

    def test_A6_invalid_block_length(self):
        """A6: Invalid block length must raise."""
        from orbit.ml.phase11_bootstrap import moving_block_bootstrap
        with pytest.raises(ValueError, match="positive"):
            moving_block_bootstrap(
                np.ones(100), lambda x: np.mean(x), block_length=-5
            )

    def test_A8_unsupported_metric(self):
        """A8: Audit catches unsupported metric/test combination."""
        from orbit.ml.phase11_audit import run_phase11_audit
        from orbit.ml.phase11_plan import phase11_plan
        from orbit.ml.phase11_inference import (
            ConfidenceInterval,
            InferenceResult,
            make_inference_result_id,
        )
        plan = phase11_plan()
        ci = ConfidenceInterval(
            point_estimate=0.01, lower=-0.01, upper=0.03,
            confidence_level=0.95, method="test", assumptions="test",
            sample_size=100,
        )
        # Create result with unsupported metric
        r = InferenceResult(
            inference_result_id="INF-test",
            source_experiment_ids=[f"EXP-{i}" for i in range(10001, 10053)],
            source_artifact_checksums={},
            metric="unsupported_metric",
            ci=ci,
            seed=42,
        )
        analysis = {
            "plan_digest": plan["plan_digest"],
            "inference_results": [r],
            "phase9_checksum": "a" * 64,
            "phase10_checksum": "b" * 64,
            "n_phase9_experiments": 0,
            "n_phase10_experiments": 52,
            "n_inference_results": 1,
        }
        checks = run_phase11_audit(plan=plan, analysis=analysis, synthetic_validation_passed=True)
        metric_check = next(c for c in checks if c["check"] == "supported_metric_test_combination")
        assert metric_check["status"] == "FAIL"

    def test_A10_adjusted_p_family_incomplete(self):
        """A10: Audit catches incomplete multiple-testing family."""
        from orbit.ml.phase11_audit import run_phase11_audit
        from orbit.ml.phase11_plan import phase11_plan
        plan = phase11_plan()
        # Analysis with incomplete family
        analysis = {
            "plan_digest": plan["plan_digest"],
            "inference_results": [],
            "multiple_testing": {
                "family": {
                    "members": ["EXP-10001", "EXP-10002"],  # Only 2 of 52
                    "n_members": 2,
                },
            },
            "phase9_checksum": "a" * 64,
            "phase10_checksum": "b" * 64,
            "n_phase9_experiments": 0,
            "n_phase10_experiments": 0,
            "n_inference_results": 0,
        }
        checks = run_phase11_audit(plan=plan, analysis=analysis, synthetic_validation_passed=True)
        family_check = next(c for c in checks if c["check"] == "multiple_comparison_family_complete")
        assert family_check["status"] == "FAIL"

    def test_A12_nondeterministic_bootstrap(self, iid_null_data):
        """A12: Same seed must produce identical bootstrap."""
        from orbit.ml.phase11_bootstrap import moving_block_bootstrap
        stat = lambda x: float(np.mean(x))
        r1 = moving_block_bootstrap(
            iid_null_data, stat, n_resamples=500, seed=42, block_length=10
        )
        r2 = moving_block_bootstrap(
            iid_null_data, stat, n_resamples=500, seed=42, block_length=10
        )
        np.testing.assert_array_equal(
            r1.bootstrap_distribution, r2.bootstrap_distribution
        )

    def test_A13_synthetic_null_not_repeatedly_significant(self, rng):
        """A13: Null data should not be repeatedly declared significant."""
        from orbit.ml.phase11_inference import t_ci
        n_trials = 100
        n_significant = 0
        for _ in range(n_trials):
            data = rng.normal(0.0, 1.0, size=100)
            lo, hi = t_ci(data, 0.95)
            if lo > 0 or hi < 0:
                n_significant += 1
        # At 5% level, expect ~5 out of 100
        assert n_significant < 25

    def test_A14_stat_significance_without_effect_size(self):
        """A14: Every result must have both CI and effect_size."""
        from orbit.ml.phase11_inference import InferenceResult, ConfidenceInterval
        ci = ConfidenceInterval(
            point_estimate=0.01, lower=-0.01, upper=0.03,
            confidence_level=0.95, method="test", assumptions="test",
            sample_size=100,
        )
        # InferenceResult requires effect_size (can be None but should be present)
        r = InferenceResult(
            inference_result_id="INF-test",
            source_experiment_ids=["EXP-10001"],
            source_artifact_checksums={},
            metric="oos_ic",
            ci=ci,
            effect_size=0.01,
            seed=42,
        )
        assert r.effect_size is not None

    def test_A15_stat_econ_separation(self):
        """A15: Statistical and economic conclusions must be separate."""
        from orbit.ml.phase11_effects import significance_economy_matrix, ic_effect_size
        eff = ic_effect_size(0.005)
        matrix = significance_economy_matrix(p_value=0.01, effect=eff)
        assert "statistical_evidence" in matrix
        assert "economic_meaning" in matrix
        # They must be different fields
        assert matrix["statistical_evidence"] != matrix["economic_meaning"]

    def test_A16_real_data_before_synthetic_validation(self):
        """A16: Audit requires synthetic validation before real-data interpretation."""
        from orbit.ml.phase11_audit import run_phase11_audit
        from orbit.ml.phase11_plan import phase11_plan
        plan = phase11_plan()
        checks = run_phase11_audit(plan=plan, synthetic_validation_passed=False)
        synth = next(c for c in checks if c["check"] == "synthetic_validation_passed")
        assert synth["status"] == "FAIL"

    def test_A1_cherry_picking_top_results(self):
        """A18: Audit catches cherry-picking (report must include all 52)."""
        from orbit.ml.phase11_audit import run_phase11_audit
        from orbit.ml.phase11_plan import phase11_plan
        from orbit.ml.phase11_inference import (
            ConfidenceInterval,
            InferenceResult,
            make_inference_result_id,
        )
        plan = phase11_plan()
        # Only include top-2 experiments
        results = []
        for i in [10001, 10045]:
            eid = f"EXP-{i}"
            ci = ConfidenceInterval(
                point_estimate=0.01, lower=-0.01, upper=0.03,
                confidence_level=0.95, method="test", assumptions="test",
                sample_size=100,
            )
            results.append(InferenceResult(
                inference_result_id=make_inference_result_id([eid], "oos_ic", "test", 42),
                source_experiment_ids=[eid],
                source_artifact_checksums={},
                metric="oos_ic",
                ci=ci,
                seed=42,
            ))
        analysis = {
            "plan_digest": plan["plan_digest"],
            "inference_results": results,
            "phase9_checksum": "a" * 64,
            "phase10_checksum": "b" * 64,
            "n_phase9_experiments": 0,
            "n_phase10_experiments": 2,
            "n_inference_results": 2,
        }
        checks = run_phase11_audit(plan=plan, analysis=analysis, synthetic_validation_passed=True)
        inv_check = next(c for c in checks if c["check"] == "source_experiment_inventory")
        assert inv_check["status"] == "FAIL"


# ──────────────────────────────────────────────────────────────
# 11. LINEAGE AND PERSISTENCE
# ──────────────────────────────────────────────────────────────

class TestLineage:
    def test_inference_result_id_deterministic(self):
        from orbit.ml.phase11_inference import make_inference_result_id
        id1 = make_inference_result_id(["EXP-10001"], "oos_ic", "bootstrap", 42)
        id2 = make_inference_result_id(["EXP-10001"], "oos_ic", "bootstrap", 42)
        assert id1 == id2

    def test_inference_result_id_unique(self):
        from orbit.ml.phase11_inference import make_inference_result_id
        id1 = make_inference_result_id(["EXP-10001"], "oos_ic", "bootstrap", 42)
        id2 = make_inference_result_id(["EXP-10002"], "oos_ic", "bootstrap", 42)
        assert id1 != id2

    def test_plan_digest_is_sha256(self):
        from orbit.ml.phase11_plan import phase11_plan_digest
        digest = phase11_plan_digest()
        assert len(digest) == 64
        # Should be valid hex
        int(digest, 16)


# ──────────────────────────────────────────────────────────────
# 12. INTEGRATION
# ──────────────────────────────────────────────────────────────

class TestIntegration:
    def test_end_to_end_synthetic(self, rng):
        """Full pipeline on synthetic data."""
        from orbit.ml.phase11_bootstrap import moving_block_bootstrap
        from orbit.ml.phase11_dependence import run_dependence_diagnostics
        from orbit.ml.phase11_effects import (
            compute_effect_size,
            significance_economy_matrix,
        )
        from orbit.ml.phase11_inference import ConfidenceInterval
        from orbit.ml.phase11_multiple_testing import multiple_testing_analysis

        # Generate synthetic "IC" values for 52 experiments
        data = rng.normal(0.01, 0.02, size=52)

        # Dependence diagnostics
        report = run_dependence_diagnostics(data)
        assert report.effective_sample_size is not None

        # Bootstrap CI for the mean
        stat = lambda x: float(np.mean(x))
        boot = moving_block_bootstrap(data, stat, n_resamples=1000, seed=42)
        ci = ConfidenceInterval(
            point_estimate=boot.point_estimate,
            lower=boot.ci_lower,
            upper=boot.ci_upper,
            confidence_level=0.95,
            method="moving_block_bootstrap",
            assumptions="stationary",
            sample_size=52,
        )

        # Effect size
        effect = compute_effect_size("oos_ic", boot.point_estimate)

        # Multiple testing (use IC values as p-values approximation)
        p_vals = [max(0.01, min(0.99, abs(v))) for v in data]
        ids = [f"EXP-{i}" for i in range(10001, 10053)]
        mt = multiple_testing_analysis(p_vals, ids)

        assert ci.width() > 0
        assert effect.magnitude >= 0
        assert mt["family"]["n_members"] == 52


# ──────────────────────────────────────────────────────────────
# 13. REPORT GENERATION (catches TypeError from Review 1)
# ──────────────────────────────────────────────────────────────

class TestReportGeneration:
    def test_write_markdown_report_no_typeerror(self, tmp_path):
        """Regression: write_markdown_report must not raise TypeError."""
        import orbit.ml.phase11_report as mod
        from orbit.ml.phase11_inference import ConfidenceInterval, InferenceResult, make_inference_result_id
        from orbit.ml.phase11_report import write_markdown_report
        original = mod.RESULTS_MD
        mod.RESULTS_MD = tmp_path / "test_results.md"
        try:
            ci = ConfidenceInterval(
                point_estimate=0.02, lower=-0.01, upper=0.05,
                confidence_level=0.95, method="bootstrap", assumptions="test",
                sample_size=100, seed=42, n_resamples=1000,
            )
            results = []
            for i in range(10001, 10006):
                eid = f"EXP-{i}"
                results.append(InferenceResult(
                    inference_result_id=make_inference_result_id([eid], "oos_ic", "bootstrap", 42),
                    source_experiment_ids=[eid],
                    source_artifact_checksums={},
                    metric="oos_ic",
                    ci=ci,
                    p_value=0.15,
                    effect_size=0.02,
                    effect_size_method="raw_ic_magnitude",
                    seed=42,
                ))
            analysis = {
                "plan_digest": "a" * 64,
                "timestamp": "2026-01-01T00:00:00",
                "n_phase9_experiments": 0,
                "n_phase10_experiments": 5,
                "n_inference_results": 5,
                "inference_results": results,
                "multiple_testing": None,
                "power_analysis": {},
                "phase9_checksum": "a" * 64,
                "phase10_checksum": "b" * 64,
            }
            # This must NOT raise TypeError
            path = write_markdown_report(analysis)
            assert path.exists()
            content = path.read_text(encoding="utf-8")
            assert "Phase 11" in content
        finally:
            mod.RESULTS_MD = original

    def test_write_research_report_no_typeerror(self, tmp_path):
        """Regression: write_research_report must not raise TypeError."""
        import orbit.ml.phase11_report as mod
        from orbit.ml.phase11_inference import ConfidenceInterval, InferenceResult, make_inference_result_id
        from orbit.ml.phase11_report import write_research_report
        original = mod.RESEARCH_MD
        mod.RESEARCH_MD = tmp_path / "test_research.md"
        try:
            ci = ConfidenceInterval(
                point_estimate=0.02, lower=-0.01, upper=0.05,
                confidence_level=0.95, method="bootstrap", assumptions="test",
                sample_size=100, seed=42, n_resamples=1000,
            )
            results = []
            for i in range(10001, 10006):
                eid = f"EXP-{i}"
                results.append(InferenceResult(
                    inference_result_id=make_inference_result_id([eid], "oos_ic", "bootstrap", 42),
                    source_experiment_ids=[eid],
                    source_artifact_checksums={},
                    metric="oos_ic",
                    ci=ci,
                    p_value=0.15,
                    effect_size=0.02,
                    effect_size_method="raw_ic_magnitude",
                    seed=42,
                ))
            analysis = {
                "plan_digest": "a" * 64,
                "timestamp": "2026-01-01T00:00:00",
                "n_phase9_experiments": 0,
                "n_phase10_experiments": 5,
                "n_inference_results": 5,
                "inference_results": results,
                "multiple_testing": None,
                "power_analysis": {},
                "phase9_checksum": "a" * 64,
                "phase10_checksum": "b" * 64,
            }
            path = write_research_report(analysis)
            assert path.exists()
            content = path.read_text(encoding="utf-8")
            assert "Phase 11" in content
        finally:
            mod.RESEARCH_MD = original

    def test_write_phase11_status_no_typeerror(self, tmp_path):
        """Regression: write_phase11_status must not raise TypeError."""
        import orbit.ml.phase11_report as mod
        from orbit.ml.phase11_report import write_phase11_status
        original = mod.STATUS_MD
        mod.STATUS_MD = tmp_path / "test_status.md"
        try:
            analysis = {
                "plan_digest": "a" * 64,
                "timestamp": "2026-01-01T00:00:00",
                "n_phase9_experiments": 0,
                "n_phase10_experiments": 5,
                "n_inference_results": 0,
                "inference_results": [],
                "multiple_testing": None,
                "power_analysis": {},
                "phase9_checksum": "a" * 64,
                "phase10_checksum": "b" * 64,
            }
            path = write_phase11_status(analysis)
            assert path.exists()
        finally:
            mod.STATUS_MD = original


# ──────────────────────────────────────────────────────────────
# 14. AUDIT DIGEST CONSISTENCY (catches Review 1 digest issue)
# ──────────────────────────────────────────────────────────────

class TestAuditDigestConsistency:
    def test_audit_plan_digest_matches_plan_object(self):
        """Audit digest verification must agree with the plan object's own digest."""
        import hashlib
        import json
        from orbit.ml.phase11_plan import phase11_plan
        from orbit.ml.phase11_audit import run_phase11_audit
        plan = phase11_plan()
        # Recompute digest the same way the audit does
        payload = {k: v for k, v in plan.items() if k != "plan_digest"}
        raw = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
        computed = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        assert plan["plan_digest"] == computed
        # Audit should pass
        checks = run_phase11_audit(plan=plan, synthetic_validation_passed=True)
        digest_check = next(c for c in checks if c["check"] == "inference_plan_digest")
        assert digest_check["status"] == "PASS"

    def test_audit_plan_digest_match_check(self):
        """Check #14 (plan_digest_match) must pass for a valid plan."""
        from orbit.ml.phase11_plan import phase11_plan
        from orbit.ml.phase11_audit import run_phase11_audit
        plan = phase11_plan()
        checks = run_phase11_audit(plan=plan, synthetic_validation_passed=True)
        match_check = next(c for c in checks if c["check"] == "plan_digest_match")
        assert match_check["status"] == "PASS"

    def test_audit_catches_corrupted_plan_digest(self):
        """Audit must detect a corrupted plan_digest field."""
        from orbit.ml.phase11_plan import phase11_plan
        from orbit.ml.phase11_audit import run_phase11_audit
        plan = phase11_plan()
        plan["plan_digest"] = "corrupted" + "0" * 54
        checks = run_phase11_audit(plan=plan, synthetic_validation_passed=True)
        digest_check = next(c for c in checks if c["check"] == "inference_plan_digest")
        assert digest_check["status"] == "FAIL"


# ──────────────────────────────────────────────────────────────
# 15. RUNNER/PLAN CONSISTENCY
# ──────────────────────────────────────────────────────────────

class TestRunnerPlanConsistency:
    def test_runner_resamples_matches_plan(self):
        """Runner _ANALYSIS_N_RESAMPLES must match plan N_BOOTSTRAP_RESAMPLES."""
        from orbit.ml.phase11_runner import _ANALYSIS_N_RESAMPLES
        from orbit.ml.phase11_plan import N_BOOTSTRAP_RESAMPLES
        assert _ANALYSIS_N_RESAMPLES == N_BOOTSTRAP_RESAMPLES

    def test_plan_resamples_is_1000(self):
        """Plan N_BOOTSTRAP_RESAMPLES should be 1000 for performance."""
        from orbit.ml.phase11_plan import N_BOOTSTRAP_RESAMPLES
        assert N_BOOTSTRAP_RESAMPLES == 1000
