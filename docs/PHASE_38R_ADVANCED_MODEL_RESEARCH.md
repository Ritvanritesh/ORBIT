# Phase 38-R: Advanced Model Research Design

**Date:** 2026-08-28T13:50:06.235339+00:00
**Phase:** 38-R

---

## 1. Primary Capability Gap

The primary missing capability is **regime-aware modelling** — the ability to condition predictive relationships on objectively defined market states.

---

## 2. Primary Recommendation

**PHASE_39R_REGIME_AWARE_MODEL_EXPLORATION**

Regime-conditioned linear model exploration, testing whether interest-rate regime conditioning improves predictive IC.

---

## 3. Why

1. **Strongest evidence chain**: Phase 36-R (STRONG_EXPLORATORY_SUPPORT) -> Phase 37-R (REGISTERED) -> Phase 39-R
2. **Simplest architecture**: Explicit regime interaction features extend existing linear framework
3. **Data ready**: Uses existing PIT data with no additional requirements
4. **Low overfitting risk**: Linear model with regime interactions
5. **High interpretability**: Regime-dependent coefficients are economically meaningful

---

## 4. Model Readiness

| Capability | Classification | Evidence |
|---|---|---|
| Regime-aware | PRIORITY_1 | Phase 36-R STRONG_EXPLORATORY_SUPPORT |
| Nonlinear | PRIORITY_2 | Yield curve branch Ridge > Lasso |
| Feature interactions | PRIORITY_2 | Sector x macro small effect |
| Ensemble | NOT_JUSTIFIED | No complementarity evidence |
| LSTM | NOT_JUSTIFIED | Data insufficient |
| TCN | NOT_JUSTIFIED | Data insufficient |
| Transformer | NOT_JUSTIFIED | Far beyond data scale |

---

## 5. Deep Learning

**NOT_JUSTIFIED**

- Insufficient temporal depth (~2000 periods vs 10K+ needed)
- Feature dimensionality too low (5-15 features)
- No evidence of sequential structure superiority
- Simpler alternatives (lag features) already exist
- Very high overfitting risk

---

## 6. Data Sufficiency

- Linear models: DATA_READY
- Tree models: DATA_READY
- Regime-conditioned linear: DATA_READY
- Deep learning: CURRENTLY_UNJUSTIFIED

---

## 7. Firewall

- OOS targets accessed: NO
- Confirmatory tests executed: NO
- Locked registrations modified: NO

---

## 8. Adversarial

19/20 PASS

---

## 9. Reproducibility

EXACT_MATCH

---

## 10. Next Allowed Step

PHASE_39R_REGIME_AWARE_MODEL_EXPLORATION

Wait for user approval.
