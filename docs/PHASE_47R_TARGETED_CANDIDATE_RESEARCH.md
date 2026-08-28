# Phase 47-R: Targeted Candidate Research

**Date:** 2026-08-28T17:38:52.451930+00:00

---

## Summary

| Item | Value |
|---|---|
| **Candidate** | CAND-RIDGE-FS001-001 |
| **Experiments** | 20/20 |
| **Budget Integrity** | PASS |
| **Evidence** | WEAK_OR_UNCERTAIN |

---

## Temporal Performance

| Config | Early | Middle | Late | Mean | Worst | Stability |
|---|---|---|---|---|---|---|
| a1.0_wNone_nglobal | +0.0610 | +0.0502 | +0.0199 | +0.0437 | late | N/A |
| a0.1_wNone_nglobal | +0.0576 | +0.0410 | +0.0114 | +0.0367 | late | N/A |
| a10.0_wNone_nglobal | +0.0576 | +0.0410 | +0.0114 | +0.0367 | late | N/A |
| a1.0_w500_nglobal | +0.0576 | +0.0410 | +0.0114 | +0.0367 | late | N/A |
| a1.0_wNone_nrolling | +0.0576 | +0.0410 | +0.0114 | +0.0367 | late | N/A |

---

## Feature Relationship Stability

| Feature | Mean Stability | Corr Stability | Classification |
|---|---|---|---|
| YC_LEVEL | 1.0566 | 0.0103 | UNSTABLE |
| YC_SLOPE | 0.2686 | 0.0479 | STABLE |
| YC_CURVATURE | 0.1464 | 0.0399 | STABLE |
| YC_CHG_10D | 0.0197 | 0.0665 | PARTIAL |

---

## Model Explainability
- YC_LEVEL: coef=-0.0041, stability=STABLE
- YC_SLOPE: coef=+0.0010, stability=STABLE
- YC_CURVATURE: coef=-0.0004, stability=STABLE
- YC_CHG_10D: coef=+0.0009, stability=STABLE

---

## Signal vs Randomness
- Real IC: 0.0611
- Feature permutation: -0.0027
- Label permutation: -0.0067
- Classification: SIGNAL_DISTINGUISHABLE_FROM_RANDOMNESS

## FIREWALL
- OOS targets accessed: NO
- Confirmatory tests executed: NO
- Locked registrations modified: NO

## ADVERSARIAL
- 31/31 PASS

## REPRODUCIBILITY
PASS
