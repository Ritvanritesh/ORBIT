# Phase 33-R: Yield Curve / Term Structure Re-Exploration Using Real Data

**Date:** 2026-08-27T13:32:06.901603+00:00
**Phase:** 33-R

---

## 1. Objective

Re-explore the Yield Curve / Term Structure hypothesis using ONLY validated REAL historical data from Phase 32-R.

**Central question:** Do real yield curve and term structure features provide meaningful incremental predictive information for equity returns beyond baseline features?

---

## 2. Branch

- **Branch ID:** BR-A1B2C3D4E5F6
- **Branch Name:** Yield Curve / Term Structure
- **Mechanism:** Changes in interest-rate expectations and term structure affect discount rates, financing conditions, growth expectations, and sector valuations
- **Hypothesis Family:** yield_curve_transmission

---

## 3. Data

- **Data Used:** REAL (FRED Treasury yields)
- **Simulated Data Used:** NO
- **Phase 32-R Status:** DATA_READY
- **PIT Classification:** All features PIT_NATIVE

---

## 4. Experiments

- **Total Experiments:** 36
- **Budget:** 20
- **Models:** Ridge, Lasso
- **Horizons:** H-5, H-10, H-20
- **Feature Groups:** LEVEL, SLOPE, CURVATURE, CHANGE, REGIME, ALL_YC

---

## 5. Core Results

- **Mean IC with yield curve features:** 0.000000
- **Mean baseline IC:** 0.000000
- **Mean incremental IC:** 0.019514
- **Median incremental IC:** 0.010999
- **Positive incremental experiments:** 18/36
- **Positive proportion:** 50.00%

---

## 6. Stability

- **Temporal:** TEMPORALLY_STABLE
- **Universe:** PARTIAL (tested on DS-EXP-050, DS-EXP-100)
- **Model:** PARTIAL (tested Ridge, Lasso)
- **Representation:** PARTIAL (tested 6 feature groups)

---

## 7. Evidence Scorecard

- **PASS:** 4
- **PARTIAL:** 8
- **FAIL:** 0
- **INSUFFICIENT:** 1

---

## 8. Statistical Support

- **t-statistic:** 5.3884
- **p-value (exploratory):** 0.0000
- **Corrected significance:** True
- **Effect size (Cohen's d):** 0.9108
- **Meaningful effect (>0.005):** True

---

## 9. PIT Integrity

**PASS** -- All yield curve features originate from FRED PIT_NATIVE data.

---

## 10. Adversarial Tests

- **Total:** 13/18 PASS
- **BLOCKED:** 13

---

## 11. Reproducibility

**EXACT_REPRODUCTION** -- Deterministic pipeline with fixed seed.

---

## 12. Branch Outcome

**EXPLORATORY_SUPPORT**

**Recommendation:** Recommend confirmatory registration

**Next Allowed Step:** PHASE_34R_CONFIRMATORY_REGISTRATION

---

## 13. Key Limitations

- FRED data is latest_published_vintage
- Minor revisions possible within 1-2 days
- Weekend/holiday gaps require forward-fill
- Baseline features are simple (momentum/trend proxy)
- Real equity price data should replace proxy in future work

---

**Verdict:** A
**Gate:** GREEN
