# Phase 37-R: Regime-Conditional Confirmatory Registration

**Date:** 2026-08-28T13:40:34.876704+00:00
**Phase:** 37-R

---

## 1. Branch

- **Branch ID:** BR-C3D4E5F6A1B2
- **Registration Status:** REGISTERED_WAITING_FOR_DATA

---

## 2. Primary Scientific Question

Does the predictive relationship between the locked baseline feature set and future equity returns differ materially between high and low interest-rate regimes at H-10?

---

## 3. Primary Hypothesis

Under the locked interest-rate regime definition, the predictive performance of the locked baseline model differs materially across pre-defined interest-rate regimes at the H-10 forecast horizon.

---

## 4. Primary Regime

- **Family:** B_INTEREST_RATE
- **Definition:** HIGH if DGS10 > 60-day rolling median; LOW otherwise
- **PIT Classification:** PIT_NATIVE

---

## 5. Primary Horizon

- **Horizon:** H-10 (10-day forward returns)
- **Justification:** Consistent regime differentials across both binary and continuous representations in Phase 36-R

---

## 6. Primary Model

- **Model:** Ridge (alpha=1.0)
- **Preprocessing:** Z-score standardization on training data
- **Features:** RET_5D, RET_10D, RET_20D, VOL_20D, MKT_RET_20D

---

## 7. Primary Metric

REGIME_DIFFERENTIAL = |IC(rate_regime_HIGH) - IC(rate_regime_LOW)|

---

## 8. Minimum Meaningful Effect

- **Threshold:** 0.010
- **Rationale:** ~47% shrinkage from Phase 36-R exploratory estimate (0.021479), accounting for winner's curse

---

## 9. Baseline

Non-regime-conditioned Ridge on the same 5 locked features

---

## 10. Experiment Matrix

5 experiments (1 primary + 4 secondary)

Budget = Matrix Size = 5

---

## 11. Secondary Tests

- CONF-002: H-20 Robustness
- CONF-003: DS-EXP-100 Robustness
- CONF-004: Continuous Regime Representation
- CONF-005: Incremental IC Test

---

## 12. Multiple Testing

- Family size: 5
- Correction: Holm-Bonferroni

---

## 13. OOS Status

DATA_NOT_READY (36/60 trading days)

No confirmatory execution occurred.

---

## 14. Firewall

- OOS targets accessed: NO
- OOS IC calculated: NO
- OOS portfolio metrics calculated: NO

---

## 15. Adversarial Review

25/25 attacks passed or appropriately classified.

---

## 16. Reproducibility

EXACT_MATCH

---

## 17. Final Registration Decision

**REGISTERED_WAITING_FOR_DATA**

---

## 18. Next Allowed Step

Wait for DATA_READY, then execute the locked confirmatory evaluation.

Do NOT automatically execute it. Wait for user approval.
