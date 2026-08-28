# Phase 42-R: Regime-Aware Confirmatory Registration

**Date:** 2026-08-28T14:43:35.435621+00:00
**Phase:** 42-R

---

## 1. Confirmatory Hypothesis

Using a pre-defined, PIT-safe interest-rate regime to route observations to separately trained Ridge models produces incremental predictive value relative to an otherwise equivalent pooled Ridge model.

---

## 2. Registration Status

**CONFIRMATORY_REGISTERED** — waiting for DATA_READY

---

## 3. RATE REGIME

HIGH if DGS10 > rolling 60-day median, LOW otherwise. PIT_NATIVE.

---

## 4. REGIME DEFINITION DIGEST

fa67a9590f1373fd2da6579b08d9d25680cb985ab0a297b39acff3f2f7841600

---

## 5. PRIMARY MODEL

Ridge (alpha=1.0), StandardScaler, separate models per RATE_REGIME

---

## 6. MATCHED BASELINE

Ridge (alpha=1.0), StandardScaler, single pooled model

---

## 7. FEATURE SET

5 features: RET_5D, RET_10D, RET_20D, VOL_20D, MKT_RET_20D

---

## 8. PRIMARY HORIZON

H-10

---

## 9. SECONDARY HORIZON

H-20

---

## 10. UNIVERSES

DS-EXP-050, DS-EXP-100

---

## 11. PRIMARY SUCCESS CRITERION

Incremental IC > +0.005 AND p < 0.05 (Holm-Bonferroni) in BOTH universes

---

## 12. MULTIPLE TESTING

Family size: 2 (primary), Holm-Bonferroni correction

---

## 13. EXPERIMENTS

6 (budget = matrix = 6)

---

## 14. FALSIFICATION CONDITIONS

10 locked failure conditions

---

## 15. CONFIGURATION FREEZE

PASS — all digests locked

---

## 16. REGISTRATION DIGEST

ef26fae0b7c8619ec0fd0b2c7d46f25ca86d508ccb490653ed422966898026d3

---

## 17. FIREWALL

- OOS targets accessed: NO
- OOS IC calculated: NO
- Confirmatory tests executed: NO
- Existing registrations modified: NO

---

## 18. ADVERSARIAL

26/26 PASS

---

## 19. REPRODUCIBILITY

PASS

---

## 20. FINAL REGISTRATION DECISION

**CONFIRMATORY_REGISTERED**

---

## 21. NEXT ALLOWED STEP

Wait for DATA_READY, then execute the locked confirmatory evaluation. Do NOT automatically execute. Wait for user approval.
