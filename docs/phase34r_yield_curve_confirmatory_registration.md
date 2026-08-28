# Phase 34-R: Yield Curve Confirmatory Registration

**Date:** 2026-08-28T13:01:23.125659+00:00
**Phase:** 34-R

---

## 1. Registration Status

**REGISTERED_WAITING_FOR_DATA**

---

## 2. Branch

- **Branch ID:** BR-A1B2C3D4E5F6
- **Research Direction:** Yield Curve / Term Structure -> equity return prediction

---

## 3. Primary Confirmatory Hypothesis

Adding the locked 12-feature real yield curve set to the locked baseline under Ridge regression produces incremental Spearman rank IC greater than 0.005 at the primary horizon on the untouched confirmatory OOS dataset.

---

## 4. Primary Model

- **Model:** Ridge regression
- **Alpha:** 1.0
- **Preprocessing:** StandardScaler (fit on training only)
- **Random seed:** 42

---

## 5. Lasso Policy

**EXCLUDED** — Lasso returned zero IC in all 18 exploratory experiments due to degenerate baseline features. No validated Lasso configuration exists.

---

## 6. Primary Horizon

- **H-20** (PRIMARY)
- **H-5**, **H-10** (SECONDARY ROBUSTNESS)

---

## 7. Feature Set

- **12 yield curve features** (LOCKED)
- **Manifest digest:** e41307333151f4e4...
- **Source:** FRED Treasury yields (REAL)
- **All PIT_NATIVE:** YES

---

## 8. Baseline

- **Features:** 5 price-derived features (returns, volatility, volume)
- **Integrity:** PASS
- **Non-degenerate:** YES

---

## 9. Experiment Budget

- **Declared budget:** 6
- **Matrix size:** 6
- **Match:** True

---

## 10. Primary Success Criterion

- **Metric:** Incremental Spearman IC
- **Minimum:** 0.005
- **Significance:** p < 0.05 (corrected)

---

## 11. Falsification Criteria

6 hard failure conditions locked. Cannot be weakened.

---

## 12. Multiple Testing

- **Primary family:** 2 tests (H-20, 2 universes)
- **Correction:** Holm-Bonferroni

---

## 13. OOS Status

- **Current days:** 36/60
- **Status:** REGISTERED_WAITING_FOR_DATA

---

## 14. Firewall

**ACTIVE** — No OOS data accessed during registration.

---

## 15. Adversarial Tests

**20/20 PASS**

---

## 16. Reproducibility

**EXACT_REPRODUCTION**

---

**Verdict:** REGISTERED_WAITING_FOR_DATA
