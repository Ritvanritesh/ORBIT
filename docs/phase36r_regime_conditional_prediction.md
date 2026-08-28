# Phase 36-R: Regime-Conditional Prediction Exploratory Research

**Date:** 2026-08-28T13:28:50.304341+00:00
**Phase:** 36-R

---

## 1. Branch

- **Branch ID:** BR-C3D4E5F6A1B2
- **Research Question:** Does predictive strength materially differ across objectively defined market regimes?

---

## 2. Regime Families Tested

- **A: Volatility Regime** — Binary (high/low) and Continuous (percentile)
- **B: Interest Rate Regime** — Binary (high/low rate level) and Continuous (yield curve slope)
- **C: Market Trend Regime** — Binary (positive/negative trend) and Continuous (momentum z-score)

---

## 3. Experiments

- **Completed:** 20 / 20
- **Budget:** 20 (MATCHED)

---

## 4. Core Results

### Overall

- **Mean Incremental IC:** 0.000450
- **Mean Regime Differential:** 0.014180
- **Positive Regime Differentials:** 18/20

### By Regime Family

- **A_VOLATILITY:** mean incr IC = -0.002199, mean regime diff = 0.010482
- **B_INTEREST_RATE:** mean incr IC = 0.003025, mean regime diff = 0.022729
- **C_MARKET_TREND:** mean incr IC = 0.000598, mean regime diff = 0.004479

---

## 5. Stability

- **Temporal:** PARTIALLY_STABLE
- **Universe:** UNIVERSE_CONSISTENT
- **Model:** Not varied (Ridge only)
- **Representation:** PARTIAL

---

## 6. Scorecard

- **PASS:** 6
- **PARTIAL:** 3
- **FAIL:** 0

---

## 7. Statistical Support

- **Regime Differential t-stat:** 5.9142
- **Regime Differential p-value:** 0.0000
- **Incr IC t-stat:** 0.4908
- **Incr IC p-value:** 0.6292

---

## 8. PIT Integrity

PASS

---

## 9. Firewall

- **OOS targets accessed:** NO
- **OOS IC calculated:** NO
- **OOS portfolio metrics calculated:** NO

---

## 10. Adversarial Review

20/22 attacks passed or appropriately classified.

---

## 11. Reproducibility

PASS

---

## 12. Economic Interpretation

ECONOMICALLY_PLAUSIBLE

---

## 13. Branch Outcome

**STRONG_EXPLORATORY_SUPPORT**

---

**Verdict:** A
