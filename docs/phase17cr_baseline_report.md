# Phase 17C-R — Canonical Baseline Establishment & Null Calibration

**Date**: 2026-08-24 17:20 UTC
**Phase**: 17C-R (Baseline Establishment)
**Parent Phase**: 17B-R (Research Framework Transition)
**Purpose**: Establish canonical baselines for all future ORBIT research

---

## Executive Summary

Phase 17C-R establishes the three canonical baseline categories that all future
ORBIT research must reference: null/random, naive investment, and simple predictive.

**Final Verdict**: **A**
**Final Gate**: **GREEN**
**Readiness**: **READY_WITH_LIMITATIONS**

---

## What Was Built

### 1. Phase Plan

Datasets, universes, horizons, baselines, metrics, and decision rules locked.
Plan digest: `1079cc15a2d69cba...`

### 2. Null / Random Baselines

- Random Score (BL-NULL-001): Deterministic random predictions
- Permutation (BL-NULL-002): Target permutation within split
- Feature Destruction (BL-NULL-003): Feature noise replacement

### 3. Null Calibration Across Horizons

Tested H-1, H-5, H-10, H-20, H-21, H-63.
Documented noise characteristics per horizon.

### 4. Naive Investment Baselines

- Equal-Weight Universe (BL-ECON-001)
- SPY Benchmark (BL-ECON-002)
- Cash Reference (BL-ECON-003)

### 5. Simple Predictive Baseline

- Ridge Regression (BL-SIMPLE-001): alpha=1.0, FS-001, LAB-006 H-5

### 6. Statistical Calibration

8 synthetic test cases validating evaluation framework honesty.

### 7. Walk-Forward Baseline Validation

Baselines run through 8 expanding windows; stability assessed.

### 8. Reproducibility

Double-build: **PASS**

### 9. Adversarial Testing

16/16 tests PASSED

---

## Baseline Hierarchy

| Level | Question | Baselines |
|-------|----------|-----------|
| 1 — Null | Exceeds no-information? | Random, Permutation, Feature Destruction |
| 2 — Simple Predictive | Better than Ridge? | Ridge (alpha=1.0, FS-001) |
| 3 — Economic | Beats naive investment? | Equal-Weight, SPY, Cash |

---

## Research Readiness

| Dimension | Status |
|-----------|--------|
| DATA_READINESS | READY |
| HYPOTHESIS_FRAMEWORK_READINESS | READY |
| BASELINE_READINESS | READY |
| STATISTICAL_READINESS | READY |
| WALK_FORWARD_READINESS | READY |
| ECONOMIC_EVALUATION_READINESS | READY_WITH_LIMITATIONS |
| REPRODUCIBILITY_READINESS | READY |
| **OVERALL** | **READY_WITH_LIMITATIONS** |

---

## Files Created

### Benchmarks
- benchmarks/phase17cr_plan.json
- benchmarks/phase17cr_baseline_inventory.json
- benchmarks/phase17cr_null_calibration.json
- benchmarks/phase17cr_horizon_baselines.json
- benchmarks/phase17cr_investment_baselines.json
- benchmarks/phase17cr_predictive_baseline.json
- benchmarks/phase17cr_statistical_calibration.json
- benchmarks/phase17cr_walkforward_baseline.json
- benchmarks/phase17cr_reproducibility.json
- benchmarks/phase17cr_adversarial.json
- benchmarks/phase17cr_readiness.json
- benchmarks/phase17cr_audit.json
- benchmarks/phase17cr_report.json

### Policies
- policies/baseline_application_policy.json

### Research
- research/baseline_registry.json

### Documentation
- docs/orbit_baseline_hierarchy.md
- docs/phase17cr_baseline_report.md

---

## Next Steps

1. **Review the baseline report**
2. **B001 may begin** (GREEN gate)
3. Select first hypothesis under new framework
4. Reference appropriate baseline level per application policy
