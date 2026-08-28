# Phase 46-R: Model Selection, System Freeze & Validity Audit

**Date:** 2026-08-28T17:14:46.312964+00:00

---

## Summary

| Item | Value |
|---|---|
| **Candidate** | CS-LGBM-FS001-001 |
| **Experiments** | 24/24 |
| **Budget Integrity** | PASS |
| **Configuration Freeze** | PASS |

---

## Model Comparison

| System | Mean IC | Incr vs Baseline | Temporal | Universe |
|---|---|---|---|---|
| A_LGBM_FS001 | -0.0381 | -0.0856 | STABLE | CONSISTENT |
| B_LGBM_FULL | 0.0064 | +0.0000 | STABLE | CONSISTENT |
| C_HGB_FS001 | -0.0331 | +0.0000 | STABLE | CONSISTENT |
| D_RIDGE_FS001 | 0.0475 | +0.0000 | STABLE | CONSISTENT |

---

## Temporal Stability: PARTIAL
## Universe Stability: PARTIAL
## Performance Concentration: LOW
## Nonlinearity: NONLINEARITY_NOT_REQUIRED
## Placebo: PLACEBO_PASS

## Feature Importance
- YC_LEVEL: 33.2%
- YC_SLOPE: 22.9%
- YC_CURVATURE: 23.1%
- YC_CHG_10D: 20.9%

---

## FIREWALL
- OOS targets accessed: NO
- Confirmatory tests executed: NO
- Locked registrations modified: NO

## ADVERSARIAL
- 33/33 PASS

## REPRODUCIBILITY
PASS

---

## NEXT ALLOWED STEP
Wait for user approval.
