# Phase 44-R: Feature Selection and System Freeze

**Date:** 2026-08-28T16:24:00.077536+00:00

---

## Summary

| Item | Value |
|---|---|
| **Experiments** | 24/24 |
| **Budget Integrity** | PASS |
| **Features Evaluated** | 19 |
| **Input Families** | 6 |
| **Selected Families** | 1 |
| **Selected Features** | 4 |
| **FS Version** | FS-001 |

---

## Family Contribution

| Family | Avg Incremental IC | Positive | Decision |
|---|---|---|---|
| A_MOMENTUM | +0.0006 | 2/4 | MARGINAL |
| B_TREND | +0.0009 | 2/4 | MARGINAL |
| C_VOLATILITY | +0.0071 | 2/4 | MARGINAL |
| D_RELATIVE | +0.0027 | 2/4 | MARGINAL |
| F_YIELD | +0.0153 | 2/4 | MARGINAL |
| G_REGIME_COND | +0.0025 | 2/4 | MARGINAL |

---

## A_MOMENTUM vs D_RELATIVE

Cross-family correlation: 0.539
Decision: KEEP_D

---

## System Comparison

| System | Features | Mean IC | Complexity |
|---|---|---|---|
| SYSTEM_FULL | 19 | 0.2830 | 22.0 |
| SYSTEM_COMPACT | 16 | 0.2501 | 18.5 |
| SYSTEM_MINIMAL | 10 | 0.2378 | 11.5 |
| SYSTEM_YIELD_MOMENTUM | 7 | 0.2253 | 8.0 |
| SYSTEM_YIELD_ONLY | 4 | 0.2830 | 4.5 |
| SYSTEM_VERY_COMPACT | 10 | 0.2476 | 11.5 |

---

## Selected System

**FS-001**: ['F_YIELD'] (4 features)

## FIREWALL
- OOS targets accessed: NO
- Confirmatory tests executed: NO
- Locked registrations modified: NO

## ADVERSARIAL
- 30/30 PASS

## REPRODUCIBILITY
PASS

---

## NEXT ALLOWED STEP
Wait for user approval.
