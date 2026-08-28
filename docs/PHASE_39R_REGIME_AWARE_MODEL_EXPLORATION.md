# Phase 39-R: Regime-Aware Model Exploration

**Date:** 2026-08-28T14:01:07.092886+00:00
**Phase:** 39-R

---

## 1. Primary Hypothesis

Regime-aware model specifications produce incremental predictive performance relative to equivalent regime-agnostic specifications.

---

## 2. Experiments

24 / 24 completed

---

## 3. MODEL COMPARISON

| Architecture | Mean IC | Mean Incr IC | Positive | N |
|---|---:|---|---|---|
| A_BASELINE                     | 0.063928 | 0.000000 | 0% | 4 |
| B_REGIME_CONDITIONED           | 0.065191 | 0.001262 | 62% | 8 |
| C_INTERACTION                  | 0.072898 | 0.001230 | 67% | 6 |
| D_SEPARATE_REGIME              | 0.072038 | 0.000370 | 75% | 4 |
| E_TREE_CONDITIONAL             | 0.070275 | -0.001392 | 50% | 2 |

---

## 4. SAMPLE FRAGMENTATION

LOW_RISK — Regime splits are approximately balanced (median-based). Minimum regime samples > 30.

---

## 5. REGIME INTEGRITY

PASS

---

## 6. EVIDENCE OUTCOME

**PARTIAL_SUPPORT**

---

## 7. SELECTED ARCHITECTURE

PHASE_40R_REGIME_MODEL_REFINEMENT

---

## 8. FIREWALL

- OOS targets accessed: NO
- Confirmatory tests executed: NO
- Locked registrations modified: NO

---

## 9. ADVERSARIAL

21/22 PASS

---

## 10. REPRODUCIBILITY

PASS

---

## 11. Verdict

**B (YELLOW)**
