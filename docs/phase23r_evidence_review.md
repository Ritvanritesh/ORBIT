# PHASE 23-R — Evidence Review & Confirmatory Registration

**Branch:** BR-E2AFD3AC901A  
**Hypothesis:** HYP-CAND-001  
**Timestamp:** 2026-08-25T15:39:36.968167+00:00  
**Verdict:** B — ELIGIBLE_WITH_DOCUMENTED_LIMITATIONS  
**Gate:** YELLOW

---

## Executive Summary

Phase 23-R reviews the complete evidence for BR-E2AFD3AC901A and determines what deserves formal confirmatory registration.

**Key Finding:** HYP-CAND-001 is eligible for confirmation with documented limitations.

**Evidence Quality:** Strong — 16/16 vol experiments show positive IC, 100% sign consistency, incremental IC positive.

**Models Justified:** Ridge and Lasso only. ElasticNet, HistGradientBoosting, and LightGBM excluded (no exploratory evidence for this hypothesis).

**Horizons:** H-10 (primary), H-20 (secondary). Both pre-registered in Phase 19-C.

**Data Gate:** DATA_NOT_READY (36/60 OOS trading days). Confirmation cannot execute until data sufficiency reached.

---

## 1. Evidence Inventory

All 20 artifacts frozen with SHA-256 digests. No artifacts excluded.

## 2. Hypothesis Reconstruction

**Research Question:** Does volatility regime information improve equity return prediction at intermediate horizons?

**Mechanism:** Volatility regimes persist and influence investor risk appetite, affecting expected returns over multi-week periods.

**Prediction:** Adding VOL_ZSCORE produces positive incremental Spearman IC at H-10 and H-20.

## 3. Exploratory Evidence Review

| Metric | Value |
|--------|-------|
| Vol experiments | 16 |
| Mean IC | 0.144799 |
| Median IC | 0.148691 |
| Sign consistency | 1.0 |
| Incremental IC | 0.017668 |

**All experiments show positive IC.** Both horizons and universes consistent.

## 4. Model Justification

| Model | Classification | Rationale |
|-------|---------------|-----------|
| Ridge | CONFIRMATORY_JUSTIFIED | Primary model, strongest exploratory IC |
| Lasso | CONFIRMATORY_JUSTIFIED | Secondary model, confirms Ridge findings |
| ElasticNet | EXPLORATORY_ONLY | No exploratory evidence for this hypothesis |
| HistGradientBoosting | EXPLORATORY_ONLY | No exploratory evidence for this hypothesis |
| LightGBM | EXPLORATORY_ONLY | No exploratory evidence for this hypothesis |

**Phase 22-R toolbox expansion does NOT automatically justify adding models to every hypothesis.**

## 5. Horizon Decision

**OPTION C: Both H-10 and H-20** as a pre-registered hypothesis family.

- H-10: Primary (mechanism supports intermediate horizon)
- H-20: Secondary (replication at longer horizon)
- Holm-Bonferroni correction applied to horizon family

## 6. Confirmatory Claim

> Under US equity universes (ENV-050, ENV-100) at horizon H-10, adding a VOL_ZSCORE volatility regime feature to a momentum-only baseline produces incremental Spearman IC > 0 against 5-day forward excess returns, and this incremental IC is replicated at horizon H-20.

**Minimum effect size:** Incremental IC > 0.005

## 7. Experiment Matrix

7 experiments locked:
- 1 primary hypothesis test (Ridge, H-10, ENV-050)
- 3 secondary tests (Ridge H-20, Ridge ENV-100, Lasso H-10)
- 3 baselines (momentum-only, vol binary)

**Matrix is FINITE, LOCKED, and FULLY ENUMERATED.**

## 8. Statistics Plan

- Family-wise error rate: 0.05
- Correction: Holm-Bonferroni
- Total tests: 4 (2 horizon family + 2 model consistency)
- Lasso treated as consistency check, not formal hypothesis

## 9. Firewall Verification

All 7 adversarial firewall attacks PASS. No OOS data accessed.

## 10. Limitations

| Limitation | Status | Impact |
|-----------|--------|--------|
| Temporal stability | PARTIAL | Documented; confirmation will test OOS persistence |
| Economic relevance | INSUFFICIENT_DATA | Deferred to Phase 24 after predictive confirmation |
| OOS data | 36/60 days | Confirmation blocked until DATA_READY |
| Regime analysis | INSUFFICIENT_DATA | Cannot resolve in current phase |

## 11. Before Confirmation Can Execute

1. OOS data must reach 60 trading days minimum
2. DATA_READY gate must be triggered
3. Phase 20-B must execute OOS validation
4. Only then can confirmatory evaluation begin

---

**Final Verdict:** B — ELIGIBLE_WITH_DOCUMENTED_LIMITATIONS  
**Gate:** YELLOW  
**Registration Status:** REGISTERED_WAITING_FOR_DATA
