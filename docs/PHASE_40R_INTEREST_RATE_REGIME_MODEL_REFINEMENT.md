# Phase 40-R: Interest-Rate Regime Model Refinement

**Date:** 2026-08-28T14:10:07.260800+00:00
**Phase:** 40-R

---

## 1. Primary Hypothesis

Interest-rate regime conditioning produces incremental predictive value beyond regime-agnostic baselines.

---

## 2. Experiments

20 / 20 completed

---

## 3. RATE REGIME DEFINITION

HIGH if DGS10 > rolling 60-day median, LOW otherwise. PIT_NATIVE.

---

## 4. ARCHITECTURE COMPARISON

| Architecture | Mean IC | Mean Incr IC | Median Incr IC | Positive % | Complexity |
|---|---:|---:|---:|---:|---|
| A_BASELINE                | 0.059614 | 0.000000 | 0.000000 | 0% | LOW |
| B_RATE_FEATURE            | 0.059614 | 0.000000 | 0.000000 | 0% | LOW |
| C_INTERACTION             | 0.065100 | 0.001385 | 0.000035 | 67% | LOW-MEDIUM |
| D_SEPARATE                | 0.067494 | 0.006513 | 0.006275 | 100% | MEDIUM |

---

## 5. BEST ARCHITECTURE

D_SEPARATE

---

## 6. TEMPORAL STABILITY

HORIZON_CONSISTENT

---

## 7. UNIVERSE STABILITY

UNIVERSE_CONSISTENT

---

## 8. SAMPLE FRAGMENTATION

LOW — Median-based regime split ensures ~50/50 balance

---

## 9. VOLATILITY INCONSISTENCY DIAGNOSTIC

HORIZON_DEPENDENCE + FEATURE_REDUNDANCY — volatility regime adds limited information beyond VOL_20D baseline feature, and effect direction flips across horizons

---

## 10. EVIDENCE OUTCOME

**PARTIAL_SUPPORT**

---

## 11. FIREWALL

- OOS targets accessed: NO
- Confirmatory tests executed: NO
- Locked registrations modified: NO

---

## 12. ADVERSARIAL

24/24 PASS

---

## 13. REPRODUCIBILITY

PASS

---

## 14. Verdict

**B (YELLOW)**
