# Phase 29-R: Multi-Branch Research Expansion & Prioritization

**Date:** 2026-08-26T14:32:12.503269+00:00
**Phase:** 29-R
**Branch:** BR-E2AFD3AC901A (frozen for confirmatory evaluation)

---

## 1. Why ORBIT is Expanding Research Branches

ORBIT has completed the confirmatory registration for its first research branch (BR-E2AFD3AC901A — volatility regime hypothesis). While this branch awaits OOS data accumulation (36/60 trading days), ORBIT development must continue.

Phase 29-R identifies new independent research directions that are scientifically justified enough to receive future exploratory research budgets. This ensures ORBIT maintains a pipeline of research candidates without compromising the frozen volatility branch.

## 2. Prior Evidence Motivating New Directions

### Active Branch
- **BR-E2AFD3AC901A**: Volatility regime hypothesis — CONFIRMATORY_REGISTERED
  - Mean IC: 0.143282, Incremental IC: 0.007583
  - Models: Ridge, Lasso approved
  - OOS: 36/60 days (DATA_NOT_READY)

### Legacy Hypotheses (PARTIALLY_CONFIRMED)
- **HYP-MAC**: Macro conditions (IC 0.0197) — data availability limited
- **HYP-MOM**: Momentum (IC 0.0142) — price-based features available
- **HYP-VOL**: Volatility (IC 0.0118) — superseded by BR-E2AFD3AC901A
- **HYP-XSEC**: Cross-sectional (IC 0.0264) — technical features available

### Model Toolbox
- Approved: Ridge, Lasso, ElasticNet, HistGradientBoosting, LightGBM
- Rejected: Deep learning
- Regime model: RESEARCH_JUSTIFIED

## 3. Candidate Branches

### 7 Candidates Evaluated

| ID | Name | Mechanism | Feasibility | Budget |
|----|------|-----------|-------------|--------|
| CAND-A | Yield Curve / Term Structure | STRONG | LIKELY_AVAILABLE | 20 |
| CAND-B | Credit Stress / Credit Spreads | MODERATE | LIKELY_AVAILABLE | 20 |
| CAND-C | Sector x Macro Interaction | STRONG | UNCERTAIN | 20 |
| CAND-D | Horizon-Aware Momentum | WEAK | AVAILABLE | 20 |
| CAND-E | Regime-Conditional Prediction | MODERATE | AVAILABLE | 20 |
| CAND-F | Nonlinear Feature Interactions | WEAK | AVAILABLE | 20 |
| CAND-G | Momentum Decay & Reversal | MODERATE | AVAILABLE | 20 |

## 4. Selection Decisions

### Selected (3)
1. **BR-A1B2C3D4E5F6** — Yield Curve / Term Structure (Priority 1)
   - Strongest mechanism, highest independence, feasible data acquisition
2. **BR-B2C3D4E5F6A1** — Sector x Macro Interaction (Priority 2)
   - Strong mechanism, high information value, sector classification needed
3. **BR-C3D4E5F6A1B2** — Regime-Conditional Prediction (Priority 3)
   - Moderate mechanism, builds on volatility branch, incremental value must be tested

### Deferred (2)
- CAND-B (Credit Stress) — may overlap with volatility branch
- CAND-G (Momentum Decay) — overlaps with CAND-D

### Rejected (2)
- CAND-D (Horizon-Aware Momentum) — weak mechanism, needs reframing
- CAND-F (Nonlinear Interactions) — weak mechanism, needs reframing

## 5. Economic Mechanisms

### Yield Curve (STRONG)
- Discount rate transmission is well-established in finance
- Yield curve changes affect equity valuations through discount rates, financing conditions, growth expectations
- Specific falsification: if yield curve changes have zero correlation with equity returns

### Sector x Macro Interaction (STRONG)
- Different sectors have different exposures to macroeconomic factors
- Sector heterogeneity is well-documented and economically intuitive
- Specific falsification: if sector-specific models do not outperform pooled models

### Regime-Conditional Prediction (MODERATE)
- Market relationships are not stationary
- Predictive effects may be regime-dependent
- Specific falsification: if regime-conditional models do not outperform unconditional models

## 6. Data Requirements

### Yield Curve
- US Treasury yields (2Y, 5Y, 10Y, 30Y) — LIKELY_AVAILABLE
- Term spreads (10Y-2Y, 30Y-100) — derived from yield data
- PIT risk: LOW — daily data, no revision

### Sector x Macro
- Sector classification (GICS) — UNCERTAIN
- Sector-level feature aggregations — derived from existing data
- PIT risk: MODERATE — sector classification may change over time

### Regime-Conditional
- Regime classification (VOL_ZSCORE) — AVAILABLE
- Regime-conditional feature interactions — derived from existing data
- PIT risk: LOW — VOL_ZSCORE is PIT-compatible

## 7. PIT Risks

| Branch | PIT Risk | Mitigation |
|--------|----------|------------|
| Yield Curve | LOW | Daily data, no revision, source verification |
| Sector x Macro | MODERATE | Historical sector labels needed, GICS reclassification handling |
| Regime-Conditional | LOW | VOL_ZSCORE is PIT-compatible |

## 8. Research Budgets

All branches receive 20-experiment exploratory budgets with:
- Checkpoints at experiments 5, 10, 15, 20
- Stopping rules: STOP_NO_SIGNAL, STOP_PIT_FAILURE, STOP_MECHANISM_FAILURE, STOP_REDUNDANT
- Budget expansion requires new registered decision

## 9. Prioritization Methodology

### Weights
| Criterion | Weight |
|-----------|--------|
| Economic mechanism strength | 0.20 |
| Independence from existing | 0.15 |
| Research gap importance | 0.15 |
| Data feasibility | 0.15 |
| PIT integrity feasibility | 0.10 |
| Falsifiability | 0.10 |
| Expected information value | 0.05 |
| Computational feasibility | 0.05 |
| Estimated research cost | 0.03 |
| Relevance to long-term objective | 0.02 |

### No predictive performance metrics used in scoring.

## 10. Sensitivity Analysis

Varying economic mechanism weight by +/- 20%:
- CAND-A (Yield Curve) remains top across all scenarios
- CAND-C (Sector x Macro) remains second
- CAND-E (Regime-Conditional) and CAND-B (Credit) may swap positions

Ranking is moderately sensitive to mechanism weight but top 2 are stable.

## 11. Hostile Review Findings

| Branch | Attacks | PASS | FAIL | LIMITATION |
|--------|---------|------|------|------------|
| CAND-A | 15 | 15 | 0 | 0 |
| CAND-C | 15 | 11 | 0 | 4 |
| CAND-E | 15 | 12 | 0 | 3 |

All top branches PASS hostile review. Documented limitations must be addressed in exploratory phase.

## 12. Selected Branches

See Section 4 for detailed selection decisions.

## 13. Recommended Execution Sequence

1. **Priority 1 (Yield Curve)**: Start data acquisition immediately
2. **Priority 2 (Sector x Macro)**: Start data acquisition in parallel
3. **Priority 1 & 2 Exploratory**: Run in parallel after data acquisition
4. **Priority 3 (Regime-Conditional)**: Wait for BR-E2AFD3AC901A OOS results, then proceed

## 14. Known Limitations

1. Yield curve data must be acquired from external source
2. Sector classification must be PIT-compatible and historically available
3. Regime-conditional prediction depends on volatility branch OOS results
4. Budget expansion requires new registered decision
5. No predictive experiments were run — this is a planning phase only

## 15. Explicit Statement

**No predictive experiments were run in Phase 29-R.**

No IC was calculated. No Sharpe was calculated. No models were trained. No OOS data was accessed. No features were generated. This phase only determined which research directions are scientifically justified for future exploration.

---

**Verdict:** A
**Gate:** GREEN
**Next Step:** Phase 30-R data acquisition for Priority 1 and Priority 2 branches (after approval)
