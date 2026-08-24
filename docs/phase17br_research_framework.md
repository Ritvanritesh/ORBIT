# Phase 17B-R — Research Framework Transition & Hypothesis Engine

**Date**: 2026-08-24 16:47 UTC  
**Phase**: 17B-R (Research Framework Transition)  
**Parent Phase**: 17A (Walk-Forward Validation)  
**Purpose**: Build the new hypothesis-driven research framework  

---

## Executive Summary

Phase 17B-R builds the new hypothesis-driven research framework that future ORBIT research will use. This phase does NOT search for alpha, add features, tune models, or modify historical results. The output is research infrastructure and governance.

**Final Verdict**: **A**  
**Final Gate**: **GREEN**  

---

## What Was Built

### 1. Legacy State Freeze

18 components classified:
- KEEP: Components that work and will be reused
- KEEP_WITH_LIMITATIONS: Functional but with documented issues
- REPAIR_BEFORE_REUSE: Needs updates for new framework
- DEPRECATE_FOR_FUTURE_RESEARCH: Historical use only
- HISTORICAL_ONLY: Evidence preserved, not reused

### 2. Research Object Model

7 formal schemas created:
- Research Question
- Mechanism
- Hypothesis (with mechanism, prediction, measurement, effect size, falsification)
- Data Specification (with PIT classification)
- Experiment Specification (exploratory/confirmatory modes)
- Evidence Record
- Decision Record
- Confirmatory Registration

### 3. Governance Policies

10 policies created:
- Exploratory Research Policy
- Evidence Review Gate Policy
- Confirmatory Preregistration Policy
- Baseline Framework Policy
- Horizon-Aware Research Policy
- Data Governance Policy
- Regime Governance Policy
- Model Governance Policy
- Conflict Resolution Policy
- Promotion Ladder Policy v2

### 4. Research Branch Registry

Registry created with prevention rules:
- No experiment deletion
- No hidden budget expansion
- No duplicate hypothesis IDs
- No confirmatory execution without registration
- No result replacement without provenance

### 5. Legacy Evidence Migration

All prior research mapped to new framework:
- H-3 classified as EXPLORATORY_FINDING (warm start, not validated)
- Phase 13B defect documented
- Phase 16 economic failure registered
- Phase 17A temporal fragility registered

### 6. Adversarial Testing

16/16 tests PASSED

### 7. Reproducibility

Deterministic double-build: PASS

---

## The New Architecture

```
Research Question → Mechanism → Hypothesis → Data + PIT Classification
    → Exploratory Branch → Evidence Review → Confirmatory Registration
    → Locked Test → PASS/FAIL → Registry
```

---

## Files Created

### Benchmarks
- benchmarks/phase17br_legacy_inventory.json
- benchmarks/phase17br_transition_map.json
- benchmarks/phase17br_legacy_evidence_migration.json
- benchmarks/phase17br_adversarial.json
- benchmarks/phase17br_reproducibility.json
- benchmarks/phase17br_audit.json

### Schemas
- schemas/research_question_schema.json
- schemas/mechanism_schema.json
- schemas/hypothesis_schema.json
- schemas/data_spec_schema.json
- schemas/experiment_spec_schema.json
- schemas/evidence_record_schema.json
- schemas/decision_record_schema.json
- schemas/confirmatory_registration_schema.json

### Policies
- policies/exploratory_policy.json
- policies/evidence_review_policy.json
- policies/confirmatory_policy.json
- policies/baseline_policy.json
- policies/horizon_policy.json
- policies/data_governance_policy.json
- policies/regime_policy.json
- policies/model_governance_policy.json
- policies/conflict_resolution_policy.json
- policies/promotion_policy_v2.json

### Research
- research/branch_registry.json

### Documentation
- docs/phase17br_research_framework.md
- docs/phase17br_transition_report.md
- docs/orbit_research_lifecycle_v2.md

---

## Next Steps

1. **Do NOT start the next research branch**
2. Review the framework documentation
3. Approve the framework before any new research begins
4. The next action is to select the first hypothesis under the new framework
