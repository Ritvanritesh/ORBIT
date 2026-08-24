# ORBIT — Full Strategic Research Review

**Date**: 2026-08-24  
**Status**: Strategic Review Complete  
**Decision Authority**: Project Owner  

---

## INTERVIEW ROUNDS — COMPLETE ANSWERS

### ROUND 1 — ORBIT'S TRUE PURPOSE

| Question | Answer |
|----------|--------|
| Q1: What problem should ORBIT solve? | **C** — Both research framework AND trading system, strict separation |
| Q2: Intended end state? | **A+B** — Robust research framework + paper trading system |
| Q3: Intended time horizon? | **F** — Multiple horizons, prioritize 5-20 days |
| Q4: Risk tolerance? | **E** — Not relevant now; define later when something survives research |
| Q5: Computational resources? | **A** — Laptop only, optional GPU later |
| Q6: Additional data access? | **B first, possibly C later** — Expand macro first (yield curves, credit, volatility, commodities, liquidity) |
| Q7: Intended deployment universe? | **A** — US large-cap equities for now |
| Q8: Intended trading frequency? | **D** — Signal-driven |
| Q9: What does "success" mean? | **A+B+E** — Robust strategy + validated framework + understanding failures |
| Q10: Continue blueprint or pivot? | **B** — Modify based on findings, don't blindly continue |

---

### ROUND 2 — TRADING / DECISION DESIGN

| Question | Answer |
|----------|--------|
| Q11: Long-only or long/short? | **D** — Decide later based on research findings |
| Q12: Portfolio concentration? | **E** — Let signal determine, with guardrails later |
| Q13: Turnover tolerance? | **E** — Let signal decide, but measure and penalize excessive turnover |
| Q14: Multiple horizons simultaneously? | **D** — Let research question determine horizon selection |
| Q15: Macro expansion strategy? | **D** — Hypothesis-driven, with strict PIT requirements |
| Q16: Revisit failed domains? | **D** — Only if specific hypothesis justifies it |
| Q17: Add new model classes? | **E** — Let research question determine model choice |
| Q18: Redesign walk-forward? | **E** — Chronological primary, regime-stratified secondary diagnostic |
| Q19: Formal null baseline? | **C** — Both random-guess AND equal-weight baselines |
| Q20: Research budget? | **D** — Per research branch, separate exploratory and confirmatory |

---

### ROUND 3 — DATA & RESOURCES

| Question | Answer |
|----------|--------|
| Q21: Which macro series first? | **G** — Hypothesis determines first acquisition |
| Q22: PIT handling for imperfect series? | **D** — Vintage where available, documented limitations where not |
| Q23: What is a research branch? | **B** — Family of related hypotheses |
| Q24: Exploratory vs confirmatory? | **D** — Exploratory: no pre-registration. Confirmatory: pre-registered. |
| Q25: What earns research budget? | **D** — Testable prediction + falsification criteria + expected effect size |
| Q26: Conflicting results? | **D** — Let pre-registered decision criteria decide |
| Q27: Who defines regimes? | **D** — Hypothesis-specific regimes, pre-registered for confirmation |
| Q28: FRED vintage policy? | **B** — ALFRED where available, current FRED where not |
| Q29: Research governance? | **C** — Formal review at 5, 10, and branch completion, with stopping rules |
| Q30: What happens to Phase 17B? | **C** — Replace with new phase reflecting updated strategy |

---

### ROUND 4 — SCIENTIFIC DESIGN DECISIONS

| Question | Answer |
|----------|--------|
| Q31: Existing H-3 evidence? | **E** — Treat as warm start, not validated, not dead |
| Q32: Original seed hypotheses? | **C** — Re-evaluate under new hypothesis standards |
| Q33: Existing codebase? | **E** — Keep what works, replace what doesn't, case by case |
| Q34: First hypothesis under new framework? | **D** — Null hypothesis, establish baselines first |
| Q35: What should new Phase 17 be? | **A** — Framework establishment phase |
| Q36: How should exploration work? | **B+D** — Document rationale + follow standard checklist |
| Q37: Confirmatory minimum preregistration? | **A** — Everything locked before execution |
| Q38: Expected effect size? | **D** — Combination of prior evidence, economic rationale, statistical power |
| Q39: Transition from old blueprint? | **D** — Gradually migrate |
| Q40: Single most important next action? | **D** — Build the new hypothesis-driven research framework |

---

## SYNTHESIS — ORBIT PRODUCT DEFINITION

### What ORBIT Is

A **hypothesis-driven quantitative research framework** for discovering, testing, falsifying, and validating trading hypotheses on US equities.

ORBIT is NOT a trading bot. ORBIT is a scientific instrument.

### What ORBIT Produces

1. **Validated research evidence** — reproducible, auditable, falsifiable
2. **Paper-trading strategies** — fed by validated research
3. **Understanding of failures** — why hypotheses fail is as valuable as finding ones that work

### What ORBIT Does NOT Do (Yet)

- Trade real capital
- Run autonomously without human oversight
- Guarantee profitability
- Predict the future with certainty

---

## CURRENT RESEARCH DIAGNOSIS

### What Is Working

| Component | Status | Evidence |
|-----------|--------|----------|
| Data infrastructure | STRONG | OHLCV, FRED macro, universe engine all functional |
| PIT logic | PARTIAL | Some series genuinely PIT, some use revised values |
| Label generation | STRONG | LAB-006 corrected, deterministic |
| Feature infrastructure | STRONG | Computes correctly, extensible |
| Model training | STRONG | Ridge, Lasso functional |
| Walk-forward evaluation | STRONG | 8 windows executed, regime-stratified |
| Portfolio simulation | FUNCTIONAL | But economic evidence weak |
| Experiment tracking | STRONG | Immutable artifacts, reproducible |
| Model registry | FUNCTIONAL | All candidates tracked |
| Adversarial testing | STRONG | 10-12 tests per phase, consistently PASS |

### What Is NOT Working

| Component | Status | Evidence |
|-----------|--------|----------|
| Predictive signal | WEAK | Only H-3 showed any IC; regime-dependent, temporally fragile |
| Portfolio translation | FAILED | Sharpe +0.016 vs baseline — economically negligible |
| Feature domains | NARROW | Only 4 macro variables tested; most information domains untested |
| Horizon alignment | WRONG | H-5 tested against monthly macro — frequency mismatch |
| Model diversity | INSUFFICIENT | Only linear models tested; trees failed but may have been implemented poorly |
| Research governance | LAX | Blueprint-driven, not hypothesis-driven |
| Baseline establishment | MISSING | No formal null/equal-weight baselines established |

### What Was Never Actually Tested

| Domain | Status | Reason |
|--------|--------|--------|
| Yield curve | UNTESTED | No data acquired |
| Credit spreads | UNTESTED | No data acquired |
| Volatility surface | UNTESTED | No data acquired |
| Commodities | UNTESTED | No data acquired |
| Liquidity dynamics | UNTESTED | No data acquired |
| Earnings surprises | UNTESTED | No data acquired |
| Order flow | UNTESTED | No data acquired |
| Multiple horizons | UNTESTED | Only H-5 tested |
| Regime-aware models | UNTESTED | Only linear models tested |
| Sector-macro interactions | UNTESTED | Hypothesis exists, not tested |

### What Assumptions Were Wrong

| Assumption | Reality |
|------------|---------|
| OHLCV features contain robust predictive signal | Failed — EXHAUSTED after 4+ phases |
| Linear models would capture macro effects | Partially true — Ridge/Lasso worked better than trees, but still fragile |
| H-5 is the right horizon for macro features | Wrong — monthly macro operates on slower timescales |
| Portfolio construction would translate IC to alpha | Failed — +0.016 Sharpe vs baseline |
| More features = better performance | Failed — feature engineering without new information is redundant |
| Phase sequence would converge to validated models | Failed — blueprint-driven approach lacks scientific rigor |

---

## BLUEPRINT REVIEW

### Classification of Major Blueprint Components

| Component | Classification | Rationale |
|-----------|---------------|-----------|
| Phase sequence (9-16) | **POSTPONE** | Historical; useful as evidence but no longer controls future |
| OHLCV feature research | **REMOVE** | Exhausted; no robust signal found |
| Market/sector context | **REMOVE** | Exhausted; no robust signal found |
| Cross-sectional features | **REMOVE** | Exhausted; marginal IC, no robustness |
| Fundamental features | **POSTPONE** | Horizon mismatch; revisit only if hypothesis justifies |
| Path structure (H-1) | **REMOVE** | FRAGILE classification |
| Return asymmetry (H-2) | **REMOVE** | FRAGILE classification |
| Volatility dynamics (H-4) | **REMOVE** | FRAGILE classification |
| Macro regime (H-3) | **POSTPONE** | Warm start; needs re-evaluation under new framework |
| Portfolio construction | **KEEP** | Infrastructure works; redesign evaluation criteria |
| Walk-forward evaluation | **KEEP** | Infrastructure works; add regime-stratified diagnostics |
| Model registry | **KEEP** | Works; extend to new hypothesis framework |
| Adversarial testing | **KEEP** | Works; extend to new hypothesis framework |
| Pre-registration | **KEEP** | Works; strengthen requirements |
| Research budgets | **MODIFY** | Per branch, not per family; separate exploratory/confirmatory |
| Baseline establishment | **ADD** | Three baselines: random, equal-weight, existing model |
| PIT classification | **ADD** | Formal classification system for data quality |
| Hypothesis specification | **ADD** | Mechanism + prediction + measurement + effect size + falsification |
| Exploratory protocol | **ADD** | No pre-registration, but logged, budgeted, no deletion |
| Confirmatory protocol | **ADD** | Full pre-registration, locked before execution |
| Stopping rules | **ADD** | Formal review at 5, 10, branch completion |

---

## RESEARCH GAP MAP

### Critical Gaps (Must Address)

1. **No baselines established** — Cannot evaluate any hypothesis without baselines
2. **No hypothesis-driven architecture** — Current blueprint is feature-driven
3. **Horizon mismatch unresolved** — Macro tested at wrong horizon
4. **PIT classification missing** — Data quality determines conclusion strength
5. **No formal stopping rules** — Research can continue indefinitely

### Important Gaps (Should Address)

1. **Macro data expansion** — Yield curve, credit, volatility, commodities, liquidity
2. **Regime definitions** — Hypothesis-specific, pre-registered
3. **Multiple testing correction** — Current approach may be too aggressive or lenient
4. **Economic plausibility** — IC must translate to something economically meaningful
5. **Exploratory vs confirmatory separation** — Currently conflated

### Optional Improvements (Could Address)

1. **New model classes** — If hypothesis justifies
2. **New data domains** — Earnings, options, order flow (future phases)
3. **Multi-asset expansion** — After US large-cap is proven
4. **Intraday data** — After daily frequency is proven
5. **Alternative data** — After traditional data is proven

---

## MODEL STRATEGY

### Current Models

| Model | Status | Evidence |
|-------|--------|----------|
| Ridge | FUNCTIONAL | Works for linear relationships; interpretable |
| Lasso | FUNCTIONAL | Works for sparse linear relationships; some degeneracy issues |
| Random Forest | FAILED | Sign reversal on test; overfit |
| XGBoost | FAILED | Sign reversal on test; overfit |

### Recommended Model Strategy

**Short-term (next phase):** Keep Ridge and Lasso. They are interpretable, functional, and sufficient for hypothesis testing.

**Medium-term:** Add models only when hypothesis justifies. For example:
- If regime dependence is confirmed → add regime-aware models (HMM, state-space)
- If nonlinear effects are hypothesized → add tree-based models with proper cross-validation
- If temporal dynamics are hypothesized → add sequence models (if data permits)

**Long-term:** Model complexity should follow evidence, not fashion.

**Do NOT add:** Transformers, LSTMs, or other complex models until simpler models have been exhausted and the bottleneck is clearly model capacity, not information.

---

## DATA STRATEGY

### Current Data

| Dataset | Content | PIT Status | Limitation |
|---------|---------|------------|------------|
| DS-EXP-050 | OHLCV, 50 instruments, 1996-2026 | N/A (price data) | Survivorship bias possible |
| DS-EXP-100 | OHLCV, 97 instruments, 1996-2026 | N/A (price data) | Survivorship bias possible |
| DS-000003 | FRED macro (DFF, UNRATE, CPI) | PARTIAL | UNRATE and CPI use revised values |
| BENCH-001 | SPY benchmark | N/A | Single benchmark |

### Recommended Data Strategy

**Immediate (next phase):** No new data acquisition. Establish baselines with existing data.

**Short-term (Phase 18-19):** Acquire expanded macro data, hypothesis-driven:
1. Treasury yield curve (2Y, 5Y, 10Y, 30Y, term spreads)
2. Credit spreads (High Yield OAS, Investment Grade OAS)
3. Volatility (VIX, implied-realized spread)
4. Commodities (Gold, Oil, Copper, Dollar Index)
5. Liquidity (Ted spread, bid-ask, Amihud illiquidity)

**PIT Classification:**
| Data Type | Allowed Research Status |
|-----------|------------------------|
| Genuine vintage/PIT | Confirmatory eligible |
| Current revised, low revision sensitivity | Exploratory or sensitivity-qualified |
| Current revised, revision-sensitive | Not eligible for strong deployment conclusions |

**Medium-term (Phase 20+):** Acquire earnings data if hypothesis justifies.

**NOT acquiring (for now):** Options, order flow, alternative data, intraday data.

---

## NEW RESEARCH ROADMAP

### Architecture

```
Research Question
       ↓
Economic Mechanism
       ↓
Formal Hypothesis
       ↓
Data + PIT Classification
       ↓
Exploratory Branch
  ├── Fixed research budget
  ├── Full experiment ledger
  └── Stability diagnostics
       ↓
Evidence Review
       ↓
CONFIRMATORY REGISTRATION
       ↓
Locked Test
       ↓
PASS / PARTIAL / FAIL
       ↓
Registry + Evidence Record
```

### Phase Sequence (Revised)

| Phase | Name | Purpose | Success Criteria | Failure Criteria |
|-------|------|---------|------------------|------------------|
| **17B** | Research Framework Transition | Establish new hypothesis-driven architecture | Framework operational, baselines established | Framework incomplete |
| **18** | Baseline Establishment | Establish three baselines (random, equal-weight, existing model) | Baselines computed, documented, frozen | Baselines not reproducible |
| **19** | First Hypothesis Design | Design and register first confirmatory hypothesis under new standards | Hypothesis registered with mechanism, prediction, measurement, effect size, falsification | Hypothesis incomplete |
| **20** | First Exploratory Branch | Execute exploratory analysis for first hypothesis | Evidence collected, logged, no deletion | Data integrity failure |
| **21** | First Confirmatory Test | Execute pre-registered confirmatory test | Test executed, results recorded, decision made | Test not reproducible |
| **22** | Second Hypothesis Design | Design second hypothesis (macro expansion) | Hypothesis registered | Hypothesis incomplete |
| **23** | Second Exploratory Branch | Execute exploratory analysis for second hypothesis | Evidence collected | Data integrity failure |
| **24** | Second Confirmatory Test | Execute pre-registered confirmatory test | Test executed, results recorded | Test not reproducible |
| **25** | Evidence Synthesis | Combine evidence from all branches | Comprehensive evidence record | Evidence incomplete |
| **26** | Framework Validation | Validate the new framework itself | Framework produces reliable, reproducible results | Framework unreliable |

### Research Branches (Planned)

**Branch 1: Baseline Establishment**
- Hypothesis: "ORBIT's baselines are well-defined and reproducible"
- Data: Existing DS-EXP-050/100, DS-000003
- Models: None (baselines only)
- Horizons: H-5, H-10, H-20
- Success: Baselines computed, documented, frozen

**Branch 2: Macro Regime Re-evaluation**
- Hypothesis: "H-3 macro regime effects are horizon-dependent"
- Data: Existing DS-000003 (DFF, UNRATE, CPI)
- Models: Ridge, Lasso
- Horizons: H-5, H-10, H-20
- Success: IC > 0.03 in at least 2 horizons, temporally stable

**Branch 3: Yield Curve and Equity Predictability**
- Hypothesis: "Yield curve shape predicts cross-sectional equity returns at H-20"
- Data: Treasury yield curve (to be acquired)
- Models: Ridge, Lasso
- Horizons: H-20
- Success: IC > 0.02, positive in both universes, temporally stable

---

## PRIORITY MATRIX

| Action | Information Gain | Scientific Importance | Implementation Difficulty | Data Requirements | Computational Cost | Snooping Risk | **Priority Score** |
|--------|-----------------|----------------------|--------------------------|-------------------|-------------------|---------------|-------------------|
| Build new framework | HIGH | HIGH | MEDIUM | LOW | LOW | LOW | **1** |
| Establish baselines | HIGH | HIGH | LOW | LOW | LOW | LOW | **2** |
| Acquire yield curve data | MEDIUM | MEDIUM | LOW | MEDIUM | LOW | LOW | **3** |
| Re-evaluate H-3 at new horizons | MEDIUM | MEDIUM | MEDIUM | LOW | LOW | MEDIUM | **4** |
| Design first confirmatory hypothesis | MEDIUM | HIGH | LOW | LOW | LOW | LOW | **5** |
| Execute first exploratory branch | MEDIUM | MEDIUM | MEDIUM | LOW | LOW | MEDIUM | **6** |
| Acquire credit spread data | MEDIUM | MEDIUM | LOW | MEDIUM | LOW | LOW | **7** |
| Execute first confirmatory test | MEDIUM | HIGH | MEDIUM | LOW | LOW | LOW | **8** |
| Design second hypothesis | MEDIUM | MEDIUM | LOW | MEDIUM | LOW | LOW | **9** |
| Acquire volatility data | LOW | LOW | LOW | MEDIUM | LOW | LOW | **10** |

---

## THE SINGLE BEST NEXT ACTION

### Build the New Hypothesis-Driven Research Framework

**Why this is #1:**

Without it:
- New macro data becomes feature collection
- New horizons become multiple testing
- New models become complexity accumulation
- New hypotheses become endless experimentation

With it, ORBIT finally gets a proper research operating system.

**What this means concretely:**

1. Define the new hypothesis specification format
2. Define branch structure (family of related hypotheses)
3. Define research budgets (per branch)
4. Define exploratory protocol (logged, budgeted, no deletion)
5. Define confirmatory protocol (fully pre-registered)
6. Define the three baselines (random, equal-weight, existing model)
7. Define evidence tiers (exploratory, confirmatory, validated)
8. Define stopping rules (review at 5, 10, branch completion)
9. Define transition rules for existing hypotheses
10. Document everything

**This is NOT a phase that discovers alpha.**  
**This is a phase that builds the scientific machinery that decides what deserves to be tested next.**

---

## CONCLUSION

ORBIT has spent 17+ phases executing a blueprint designed before data was examined. The data has spoken: most hypotheses failed. The strongest finding (H-3 macro regime) is temporally fragile and fails economically.

The correct response is NOT to continue the blueprint.  
The correct response is NOT to abandon the project.  
The correct response is to **redesign the research architecture** based on what has been learned.

ORBIT should become:
- **Hypothesis-driven** (not feature-driven)
- **Horizon-aware** (not H-5 only)
- **Evidence-budgeted** (not phase-number-driven)
- **PIT-classified** (not assuming all data is equally reliable)
- **Exploratory/confirmatory separated** (not conflated)

The single best next action is to **build this new framework** before doing anything else.

---

*Document generated by ORBIT Strategic Review Process*  
*Interview Rounds 1-4 completed*  
*All questions answered by project owner*
