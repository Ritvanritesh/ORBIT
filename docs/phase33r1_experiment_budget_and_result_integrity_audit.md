# Phase 33-R.1: Experiment Budget & Result Integrity Audit

**Date:** 2026-08-27T13:42:00.291448+00:00
**Phase:** 33-R.1

---

## 1. Purpose

Forensic audit of Phase 33-R to verify scientific validity
and confirmatory registration eligibility.

---

## 2. Budget Reconstruction

- **Stated budget:** 20
- **Locked matrix count:** 36
- **Classification:** ACCOUNTING_ERROR_NO_TRUE_OVERRUN
- **Root cause:** PLAN_CONSTRUCTION_ERROR: The experiment matrix was constructed as a full Cartesian product (3 horizons x 6 feature groups x 2 models = 36) but the budget field was set to 20. The matrix and budget were inconsistent from plan creation.

---

## 3. Experiment Inventory

- **Total experiments:** 36
- **All in locked matrix:** YES
- **Duplicates found:** 0
- **Unauthorized expansions:** 0

---

## 4. First-20 Authorized Analysis

- **Experiment count:** 20
- **Mean incremental IC:** 0.017022182835289935
- **Median incremental IC:** 0.011702629326877694
- **Positive proportion:** 0.5
- **p-value:** 0.0006235711280059117

---

## 5. Locked Matrix Analysis

- **Pre-specified experiments:** 36
- **Support classification:** SUPPORT_WEAKENS_LOCKED_MATRIX

---

## 6. Metric Reconciliation

- **Mean IC YC:** 0.0 (reporting artifact from Lasso zeros)
- **Mean incremental IC:** 0.019514319866679824
- **Explanation:** Phase 33-R reported mean_ic_yc = 0.0 because the reported 'overall' aggregation in the incremental value analysis averaged across all experiments including Lasso (which returned 0 IC). The Ridge-only YC IC values are all positive (0.02 to 0.08).
- **Classification:** REPORTING_ERROR_UNDERLYING_RESULTS_RECOMPUTED

---

## 7. Baseline Integrity

- **Classification:** EXACTLY_MATCHED
- **Finding:** All baseline ICs are exactly 0.0 because baseline features were constant zeros. Baseline and YC experiments share the same training/test split, model, and preprocessing. The ONLY difference is feature inclusion. Pairing is EXACTLY_MATCHED except for the trivial baseline.

---

## 8. Model Integrity

- **Ridge mean IC:** 0.03902863973335965
- **Lasso all zero:** True
- **Classification:** STILL_VALID_FOR_RIDGE_ONLY

---

## 9. Multiple Testing

- **Total combinations:** 36
- **Classification:** MULTIPLE_TESTING_DOCUMENTED

---

## 10. Data Integrity

- **Classification:** REAL_DATA_ONLY

---

## 11. Adversarial Tests

- **BLOCKED:** 14
- **DETECTED:** 3
- **LIMITATION:** 3
- **FAIL:** 0

---

## 12. Independent Recomputation

- **Full 36 mean incr IC:** 0.019514319866679824
- **Ridge-only mean IC:** 0.03902863973335965
- **Classification:** EXACT_MATCH

---

## 13. Historical Artifact Integrity

- **All unchanged:** True

---

## 14. Final Scientific Outcome

**EXPLORATORY_SUPPORT_WITH_LIMITATIONS**

**Rationale:** Positive evidence survives audit. Budget was a plan construction error (matrix was 36 from start), not a post-hoc expansion. Ridge results are genuinely positive. Lasso degeneracy is a documented limitation.

**Next step:** PHASE_34R_CONFIRMATORY_REGISTRATION_WITH_LIMITATION_CONTROLS

---

## 15. Limitations

- Budget/matrix inconsistency in plan construction (20 stated vs 36 actual)
- Baseline features were zero constants (degenerate baseline)
- Lasso returned zero due to feature scaling + regularization
- Positive evidence comes from Ridge only
- 36 experiments (18 Ridge) inflate multiple testing risk

---

**Verdict:** EXPLORATORY_SUPPORT_WITH_LIMITATIONS
**Budget:** 20 stated / 36 actual (plan construction error)
