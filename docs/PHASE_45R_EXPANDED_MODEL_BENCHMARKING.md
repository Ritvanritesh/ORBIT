# Phase 45-R: Expanded Model Benchmarking

**Date:** 2026-08-28T16:50:15.956897+00:00

---

## Summary

| Item | Value |
|---|---|
| **Experiments** | 24/24 |
| **Budget Integrity** | PASS |
| **Feature System** | FS-001 (4 features) |
| **Feature Freeze** | PASS |

---

## Model Comparison

| Model | Feature System | Mean IC | Incremental IC |
|---|---|---|---|
| Ridge | FS-001 | 0.1423 | -0.0571 |
| Ridge | SYSTEM_FULL | 0.2786 | +0.0000 |
| Ridge | BASELINE | 0.1995 | +0.0000 |
| ElasticNet | FS-001 | 0.0955 | -0.1004 |
| ElasticNet | SYSTEM_FULL | 0.2229 | +0.0000 |
| ElasticNet | BASELINE | 0.1959 | +0.0000 |
| HistGradientBoosting | FS-001 | 0.4839 | +0.2711 |
| HistGradientBoosting | SYSTEM_FULL | 0.4871 | +0.0000 |
| HistGradientBoosting | BASELINE | 0.2128 | +0.0000 |
| LightGBM | FS-001 | 0.4864 | +0.2783 |
| LightGBM | SYSTEM_FULL | 0.4904 | +0.0000 |
| LightGBM | BASELINE | 0.2081 | +0.0000 |

---

## Feature System Comparison

| Feature System | Features | Mean IC | Incr vs Baseline |
|---|---|---|---|
| FS-001 | 4 | 0.3020 | +0.0980 |
| SYSTEM_FULL | 0 | 0.3698 | +0.1657 |
| BASELINE | 0 | 0.2040 | +0.0000 |

---

## Regime Comparison

| Model | Pooled IC | Separate IC | Incremental |
|---|---|---|---|
| Ridge | 0.1423 | 0.0830 | -0.0593 |
| ElasticNet | 0.0955 | 0.0768 | -0.0187 |
| HistGradientBoosting | 0.4839 | 0.4188 | -0.0651 |
| LightGBM | 0.4864 | 0.4265 | -0.0599 |

---

## Best Configuration: LightGBM_FS-001

## FIREWALL
- OOS targets accessed: NO
- Confirmatory tests executed: NO
- Locked registrations modified: NO

## ADVERSARIAL
- 34/34 PASS

## REPRODUCIBILITY
PASS

---

## NEXT ALLOWED STEP
Wait for user approval.
