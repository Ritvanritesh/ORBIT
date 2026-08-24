# Phase 16.5 — Research Reset & Next-Hypothesis Selection

**Date**: 2026-08-24 13:57 UTC  
**Parent Phase**: 16 (Verdict D, Gate RED)  
**Phase 16.5 Verdict**: Research Reset Complete  
**Phase 16.5 Gate**: **YELLOW**  

---

## Executive Summary

Phase 16.5 performed a comprehensive research reset of ORBIT's predictive modeling pipeline. After systematically reconstructing the research history (Phases 9-16), decomposing failure modes, analyzing information gaps, auditing target/horizon choices, temporal coverage, and data limitations, this phase identified **three scientifically justified next research branches**.

**Key Finding**: ORBIT's research space is **NOT exhausted**. The failure of H-3 macro regime at the portfolio level (Phase 16, Sharpe +0.016 vs baseline) does NOT mean the information is worthless — it means the current formulation (5-day horizon, single train/val/test split, uniform sector treatment) may be suboptimal.

**Selected Next Branches**:
1. **B01** — Horizon Extension: Test H-3 macro features at H-10 and H-20 horizons
2. **B03** — Sector-Macro Interaction: Test differential macro sensitivity across sectors
3. **B07** — Walk-Forward Validation: Resolve temporal instability with expanding-window validation

**Adversarial Tests**: 10/10 PASS  
**Final Gate**: YELLOW — Continue to Phase 17 after user approval  

---

## Research History Summary

### Information Domains Tested (Phases 9-16)

| Domain | Phases | Classification | Key Finding |
|--------|--------|---------------|-------------|
| OHLCV/Technical | 9-13C | EXHAUSTED | No robust signal |
| Market Context | 9-12 | EXHAUSTED | No robust signal |
| Sector Context | 9-11 | EXHAUSTED | No robust signal |
| Cross-Sectional | 10-12 | EXHAUSTED | Marginal IC, no robustness |
| Fundamentals | 12-13 | PARTIALLY_EXPLORED | Inconsistent, horizon-mismatched |
| Path Structure | 14.5 | INCONCLUSIVE | FRAGILE (sign-inconsistent) |
| Return Asymmetry | 14.5 | INCONCLUSIVE | FRAGILE |
| Volatility Dynamics | 14.5 | INCONCLUSIVE | FRAGILE |
| **Macro Regime (H-3)** | 14.5-16 | **PROMISING_BUT_UNVALIDATED** | **Temporal instability, portfolio translation failure** |
| Portfolio Construction | 16 | PARTIALLY_EXPLORED | No robust alpha vs baseline |

### Failure Mode Summary

| Failure ID | Description | Primary Mode | Confidence |
|------------|-------------|--------------|------------|
| F01 | OHLCV features: small IC, no generalization | WEAK_INFORMATION | 0.90 |
| F02 | Market/sector context: OOS failure | FEATURE_FRAGILITY | 0.85 |
| F03 | Fundamentals: LAB-005 defect, inconsistent | TARGET_MISMATCH | 0.80 |
| F04 | H-1/H-2/H-4: FRAGILE classification | UNIVERSE_DEPENDENCE | 0.85 |
| **F05** | **H-3 macro: temporal instability** | **TEMPORAL_INSTABILITY** | **0.95** |
| **F06** | **H-3 macro: CLIFF sensitivity, collinearity** | **FEATURE_FRAGILITY** | **0.90** |
| F07 | Portfolio: +0.016 Sharpe vs baseline | PORTFOLIO_TRANSLATION_FAILURE | 0.90 |
| F08 | No val period predictions in Phase 16 | IMPLEMENTATION_LIMITATION | 1.00 |
| F09 | Lasso degeneracy at alpha=0.001 | IMPLEMENTATION_LIMITATION | 0.95 |
| F10 | Nonlinear models: sign reversal | MODEL_DEPENDENCE | 0.85 |

---

## Information Gap Analysis

**Genuinely Untested Domains** (require new data):
- Earnings surprises/revisions
- Options implied volatility
- Liquidity dynamics / order flow
- Corporate events
- Cross-asset relationships
- Factor exposures

**Actionable Now** (existing data):
- Alternative prediction horizons (H-1, H-2, H-10, H-20, H-60)
- Event-conditioned returns
- Sector-relative macro sensitivity
- Macro momentum (rate-of-change features)

---

## Target & Horizon Audit

**Current**: LAB-006, 5-day excess return  
**Horizon Mismatch Hypothesis**: Monthly macro data may be better suited to H-20 horizons than H-5  

**Untested Horizons**: H-1, H-2, H-10, H-20, H-60  
**Key Insight**: Macro regime features (monthly frequency) tested only at H-5 — frequency mismatch likely  

---

## Temporal Coverage Audit

**Train Period** (2010-2018): Post-GFC recovery, low rates, low volatility  
**Val Period** (2019-2021): COVID crash and recovery — extreme, non-representative  
**Test Period** (2022-2026): Inflation, rate hikes, geopolitical shocks  

**Critical Gap**: Train+val do NOT contain inflation or rising-rate regimes. Test period contains macro conditions unseen in training data.

**H-3 Temporal Instability**: Validation IC negative (COVID regime), test IC positive (inflation regime) — effect is regime-dependent, not persistent.

---

## Data Limitations

**Critical** (3): No options data, no earnings data, no order flow data  
**Material** (1): Macro release timing alignment not audited  
**Moderate** (4): Sector diversity, fundamental frequency, multi-frequency alignment, sample complexity  
**Minor** (5): Survivorship, historical coverage, delisting, corporate actions, benchmark construction  

---

## Research Branch Generation

**8 branches generated**, all survive redundancy filter, all feasible for pre-registration.

| Rank | Branch | Hypothesis | Composite Score |
|------|--------|------------|-----------------|
| 1 | B01 | Horizon Extension (H-10, H-20) | 0.88 |
| 2 | B03 | Sector-Macro Interaction | 0.86 |
| 3 | B07 | Walk-Forward Validation | 0.86 |
| 4 | B02 | Macro Momentum (rate-of-change) | 0.84 |
| 5 | B08 | Multi-Horizon Ensemble | 0.80 |
| 6 | B06 | Fundamental Horizon Match (H-60) | 0.78 |
| 7 | B04 | Event-Conditioned Macro | 0.76 |
| 8 | B05 | Path Structure Revisited | 0.72 |

---

## Selected Next Branches

### B01 — Horizon Extension
**Hypothesis**: Macro regime information predicts medium-horizon (H-10, H-20) returns better than short-horizon (H-5)  
**Rationale**: Monthly macro data has low update frequency. Predictive power should be stronger at horizons matching its update cadence (~20 business days).  
**Data**: Existing DS-EXP-050/100 + DS-000003  
**Falsification**: IC at H-10/H-20 must be significantly higher than IC at H-5  

### B03 — Sector-Macro Interaction
**Hypothesis**: Sectors respond heterogeneously to macro regime changes, and this differential response is predictable  
**Rationale**: Interest-rate-sensitive sectors (REITs, utilities, financials) respond differently to Fed Funds Rate changes than cyclical sectors.  
**Data**: Existing DS-EXP-050/100 (sector labels) + DS-000003  
**Falsification**: Sector-macro interaction IC must exceed macro-only IC  

### B07 — Walk-Forward Validation
**Hypothesis**: Temporal instability of H-3 can be resolved with expanding-window validation  
**Rationale**: Phase 16 used single train/val/test split. H-3 temporal instability may reflect a single unlucky validation window (COVID).  
**Data**: Existing DS-EXP-050/100 + DS-000003  
**Falsification**: Walk-forward IC must be positive and stable across all windows  

---

## Adversarial Test Results

| Test | Result | Detail |
|------|--------|--------|
| A1: Historical artifact modification | PASS | No prior artifacts modified |
| A2: Branch duplicates prior work | PASS | All 8 branches tested; none REDUNDANT |
| A3: Branch cannot be PIT implemented | PASS | All branches use existing PIT-correct data |
| A4: Branch lacks sufficient data coverage | PASS | All branches use existing datasets |
| A5: Target/horizon chosen opportunistically | PASS | Horizon choices motivated by frequency matching |
| A6: Hypothesis family expanded after ranking | PASS | 8→3 branches; family controlled |
| A7: Scoring formula modified after results | PASS | Formula defined before computing scores |
| A8: Recommended branch lacks falsification | PASS | All 3 selected branches have explicit criteria |
| A9: Branch selected for profitability | PASS | No profitability criterion in scoring |
| A10: Research history omitted for novelty | PASS | Research map documents all prior work |

---

## Final Gate

**GATE: YELLOW**

**Rationale**: Multiple scientifically motivated next branches exist with clear economic rationale, feasible data, and distinct hypotheses from prior work. However:
- The current H-3 macro regime hypothesis is NOT validated
- Walk-forward validation is required before any promotion
- New feature construction (macro-momentum, sector-macro interactions) needed
- Research space is NOT exhausted

**Next Action**: Lock plan for Phase 17 (selected branch execution) after user approval. Do NOT start Phase 17 without explicit approval.

---

## Files Generated

```
benchmarks/phase16_5_plan.json
benchmarks/phase16_5_research_map.json
benchmarks/phase16_5_failure_modes.json
benchmarks/phase16_5_information_gaps.json
benchmarks/phase16_5_target_horizon_audit.json
benchmarks/phase16_5_temporal_audit.json
benchmarks/phase16_5_data_limitations.json
benchmarks/phase16_5_candidate_branches.json
benchmarks/phase16_5_redundancy_review.json
benchmarks/phase16_5_preregistration_review.json
benchmarks/phase16_5_prioritization.json
benchmarks/phase16_5_hostile_review.json
benchmarks/phase16_5_recommendation.json
benchmarks/phase16_5_audit.json
docs/phase16_5_research_reset.md
```

**Total artifacts modified**: 0  
**Total artifacts created**: 15  
**Total scripts created**: 4 (`_phase16_5_part1.py` through `_phase16_5_part4.py`)
