# Phase 41-R: Targeted Separate Rate-Regime Model Research

**Date:** 2026-08-28T14:20:35.142488+00:00
**Phase:** 41-R

---

## 1. Primary Hypothesis

Separate rate-regime models outperform pooled models because feature-return relationships differ across interest-rate regimes.

---

## 2. Experiments

20 / 20 completed

---

## 3. RATE REGIME

HIGH if DGS10 > rolling 60-day median, LOW otherwise. PIT_NATIVE. Frozen from Phase 40-R.

---

## 4. ARCHITECTURE RESULTS

| Architecture | Mean IC | Mean Incr IC | Median Incr IC | Positive % |
|---|---:|---:|---:|---:|
| Pooled baseline                | 0.063903 | 0.000000 | 0.000000 | 0% |
| Rate-regime separate           | 0.073176 | 0.009273 | 0.009618 | 100% |
| Placebo split                  | 0.059374 | -0.004530 | -0.001460 | 25% |

---

## 5. REGIME-SPECIFIC ADVANTAGE

Rate-regime advantage over placebo: 0.013802

---

## 6. COEFFICIENT HETEROGENEITY

WEAK_HETEROGENEITY

---

## 7. TEMPORAL STABILITY

TEMPORALLY_STABLE

---

## 8. HORIZON STABILITY

HORIZON_CONSISTENT

---

## 9. UNIVERSE STABILITY

UNIVERSE_CONSISTENT

---

## 10. SAMPLE FRAGMENTATION

LOW

---

## 11. EVIDENCE OUTCOME

**STRONG_EXPLORATORY_SUPPORT**

---

## 12. FIREWALL

- OOS targets accessed: NO
- Confirmatory tests executed: NO
- Locked registrations modified: NO

---

## 13. ADVERSARIAL

27/27 PASS

---

## 14. Verdict

**A (GREEN)**
