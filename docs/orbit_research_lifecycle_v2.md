# ORBIT Research Lifecycle v2

**Version**: 1.0  
**Date**: 2026-08-24 16:47 UTC  

---

## Overview

ORBIT is a hypothesis-driven quantitative research framework. This document defines the research lifecycle that future ORBIT research will follow.

---

## Lifecycle Stages

### Stage 1: Research Question
- Define the economic question
- Identify the domain
- Document the motivation

### Stage 2: Mechanism
- Describe the proposed mechanism
- Document the economic rationale
- Identify assumptions and failure modes
- Define falsification conditions

### Stage 3: Hypothesis
- Formulate the hypothesis with:
  - Statement
  - Directional prediction
  - Target variable
  - Universe
  - Horizon
  - Expected effect size
  - Economic materiality threshold
  - Falsification criteria

### Stage 4: Data + PIT Classification
- Identify required datasets
- Classify PIT status
- Document revision risk
- Document survivorship risk

### Stage 5: Exploratory Branch
- Create branch with defined scope
- Set experiment budget (default: 20)
- Execute exploratory experiments
- Log all experiments (no deletion)
- Collect evidence

### Stage 6: Evidence Review Gate
- Evaluate mechanism plausibility
- Check effect consistency
- Check universe consistency
- Check temporal stability
- Check economic materiality
- Check data/PIT integrity
- Decision: CONTINUE / REPLICATE / ADVANCE / REJECT / RETIRE

### Stage 7: Confirmatory Registration
- Lock all experiment parameters:
  - Hypothesis
  - Mechanism
  - Expected effect size
  - Economic materiality threshold
  - Datasets
  - PIT classification
  - Universe
  - Horizon
  - Features
  - Label
  - Model family
  - Parameter policy
  - Split method
  - Walk-forward design
  - Statistical tests
  - Multiple-testing family
  - Baselines
  - Robustness tests
  - Promotion criteria
  - rejection criteria
- Generate registration digest

### Stage 8: Locked Validation
- Execute pre-registered experiment
- Record all results
- No modification allowed

### Stage 9: Portfolio and Economic Validation
- Test economic materiality
- Evaluate portfolio implications
- Assess transaction costs

### Stage 10: Promotion or Retirement
- Apply promotion ladder:
  - PROPOSED → EXPLORATORY → EVIDENCE_REVIEW → CONFIRMATORY_REGISTERED → CONFIRMATORY_TESTED → RESEARCH_SUPPORTED → PAPER_ELIGIBLE → VALIDATED
- Or: REJECT / RETIRE

---

## Three Baselines Required

### Baseline A: Null / Random Prediction
- Purpose: Evaluate whether predictive metrics exceed chance
- Implementation: Random ranking or permutation test

### Baseline B: Naive Investment
- Purpose: Evaluate economic value
- Examples: equal-weight portfolio, universe benchmark

### Baseline C: Simple Existing Model
- Purpose: Measure incremental complexity value
- Examples: momentum baseline, previous validated model

---

## Research Modes

### Exploratory
- No pre-registration required
- Full experiment logging required
- Failed experiments may not be deleted
- Budget defined at branch creation
- May change direction within scope

### Confirmatory
- Full pre-registration required
- Locked before execution
- No modification allowed
- Registration digest required
- Must pass evidence review gate first

---

## PIT Classifications

| Classification | Allowed Conclusion Strength |
|---------------|---------------------------|
| STRICT_PIT | Full evidential status |
| PIT_WITH_KNOWN_LAG | Full with lag documented |
| REVISED_HISTORY | Exploratory or sensitivity-qualified only |
| NON_PIT_RESEARCH_ONLY | Research only; no deployment claims |
| UNKNOWN | No strong conclusions allowed |

---

## Conflict Resolution

| Conflict Type | Classification |
|--------------|----------------|
| Works in both universes | ROBUST (if other criteria met) |
| Works in one universe only | UNIVERSE_DEPENDENT |
| Works at one horizon only | HORIZON_DEPENDENT |
| One model succeeds, others fail | MODEL_DEPENDENT |
| One regime supports, others reject | CONTEXT_DEPENDENT |
| Sign inconsistent across windows | FRAGILE |
| No positive evidence | NOT_SUPPORTED |

---

## Promotion Requirements for PAPER_ELIGIBLE

1. Successful confirmatory test
2. Predefined statistical support
3. Predefined economic materiality
4. Completed required universe tests
5. Temporal validation
6. Reproducibility
7. PIT/data integrity
8. Portfolio economics where relevant

---

## Key Rules

1. Hypothesis must have mechanism, prediction, and falsification criteria
2. Exploratory and confirmatory modes are distinct
3. Failed experiments cannot be deleted
4. Branch budgets are enforced
5. Confirmatory experiments require locked registration
6. Data PIT status is explicit
7. Revised data cannot masquerade as strict PIT
8. Conflicting evidence is preserved
9. Negative evidence is preserved
10. Promotion ladder cannot be skipped
