# ORBIT Baseline Hierarchy

**Version**: 1.0
**Date**: 2026-08-24 17:20 UTC

---

## Overview

Every future ORBIT experiment must reference the appropriate baseline level.
The baseline hierarchy ensures that predictive and economic claims are properly calibrated.

---

## Level 1 — Null / Random

**Question**: Does predictive performance exceed no-information behavior?

**Baselines**:
- Random Score: Deterministic random predictions (fixed seed)
- Permutation: Targets permuted within valid grouping
- Feature Destruction: Predictive features replaced with noise

**Required for**: ALL hypotheses

**Pass criterion**: IC significantly different from null distribution

**Interpretation**: If a model cannot beat random noise, it has no predictive value.

---

## Level 2 — Simple Predictive

**Question**: Does the proposed hypothesis improve over a simple existing model?

**Baselines**:
- Ridge Regression (alpha=1.0, FS-001 features, LAB-006 H-5)

**Required for**: All hypotheses claiming improvement over existing methods

**Pass criterion**: IC or Sharpe materially better than Ridge baseline

**Interpretation**: Complex models must justify their complexity over a simple linear model.

---

## Level 3 — Economic Value

**Question**: Does the resulting strategy add value over naive investment?

**Baselines**:
- Equal-Weight Universe Portfolio
- SPY Buy-and-Hold Benchmark
- Cash (Zero-Exposure) Reference

**Required for**: All hypotheses with economic or deployment claims

**Pass criterion**: Sharpe > benchmark, material excess return after costs

**Interpretation**: Statistical significance alone is insufficient; economic materiality is required.

---

## Application Rules

1. Every exploratory experiment must at least reference Level 1
2. Economic claims require Level 3 comparison
3. No baseline substitution after seeing hypothesis results
4. Baseline versions are frozen and versioned
5. Regression to a lower baseline level is always allowed

---

## Baseline IDs

| Level | Baseline | ID |
|-------|----------|-----|
| 1 | Random Score | BL-NULL-001 |
| 1 | Permutation | BL-NULL-002 |
| 1 | Feature Destruction | BL-NULL-003 |
| 2 | Ridge Predictive | BL-SIMPLE-001 |
| 3 | Equal-Weight | BL-ECON-001 |
| 3 | SPY Benchmark | BL-ECON-002 |
| 3 | Cash Reference | BL-ECON-003 |
