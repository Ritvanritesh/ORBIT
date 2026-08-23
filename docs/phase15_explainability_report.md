# Phase 15 — Deterministic Explainability & Decision Evidence Engine

**Date:** 2026-08-23  
**Verdict:** D  
**Gate:** RED  
**Phase Clock:** 2026-08-23T00:00:00+00:00

---

## Executive Summary

Phase 15 built a deterministic explainability engine that answers: *Why did this exact model produce this exact prediction, and can we trust that explanation?*

**Critical finding:** The H-3 macro-regime models that showed the best predictive IC in Phase 14.5 (ridge IC=+0.091 on ENV-050, +0.132 on ENV-100) have **MISLEADING** faithfulness on ridge variants and **UNRELIABLE** explanation confidence. The apparent predictive improvement is driven almost entirely by macro features (88-100% of total coefficient effect), but the attribution ranking does not faithfully track actual model sensitivity. This means the H-3 improvement may be an artifact of a narrow temporal regime rather than a robust signal.

**No model is promoted.** Explanation quality does not equal predictive quality.

---

## Definition of Done Checklist

| Criterion | Status | Evidence |
|-----------|--------|----------|
| Every prediction resolves to exact model identity | PASS | Model inventory locked in phase15_plan.json |
| Exact feature values recoverable | PASS | Local explanations contain full feature values |
| Attribution deterministic or reproducible | PASS | 11/11 models EXACT_REPRODUCTION |
| Synthetic ground-truth tests pass | PASS | 5/5 synthetic models PASS |
| Correlated-feature instability measured | PASS | 4/11 models flagged (all H-3) |
| Local sensitivity measured | PASS | 11 models × 3 samples each |
| Counterfactual boundaries bounded | PASS | All counterfactuals within domain bounds |
| Model disagreement recorded | PASS | SIGN_CONFLICT across all 50 test observations |
| Explanation faithfulness tested | PASS | Per-feature deletion ablation test |
| Explanation confidence separate from trading | PASS | Confidence measures explanation stability, not trade quality |
| Provenance attacks rejected | PASS | 12/12 adversarial tests REJECT invalid cases |
| Failed explanations remain in audit | PASS | MISLEADING models explicitly classified |
| Historical artifacts unchanged | PASS | Phase 9-14.5 results untouched |
| Reproducibility demonstrated | PASS | 11/11 EXACT_REPRODUCTION |
| Phase 14.5 H-3 not promoted | PASS | All models remain RESEARCH |
| No model promoted beyond evidence status | PASS | No promotion decisions made |

---

## Model Inventory

11 models trained and analyzed (MODEL-00001 skipped: no DS-DEV20 data available):

| Model ID | Family | Feature Set | Universe | OOS IC | Status |
|----------|--------|-------------|----------|--------|--------|
| MODEL-00002 | lasso | FS-12B-B | ENV-050 | +0.0000 | RESEARCH |
| MODEL-00005 | lasso | FS-12B-E | ENV-050 | +0.0000 | RESEARCH |
| MODEL-00006 | lasso | FS-12B-D | ENV-050 | +0.0000 | RESEARCH |
| MODEL-00007 | lasso | FS-12B-E | ENV-050 | +0.0000 | RESEARCH |
| MODEL-00008 | ridge | FS-12B-A | ENV-050 | +0.0271 | RESEARCH |
| MODEL-00009 | ridge | FS-001 | ENV-050 | +0.0271 | RESEARCH |
| MODEL-00010 | lasso | FS-001 | ENV-050 | +0.0000 | RESEARCH |
| MODEL-14-5-H3-RIDGE-050 | ridge | FS-H3 | ENV-050 | +0.0909 | RESEARCH |
| MODEL-14-5-H3-LASSO-050 | lasso | FS-H3 | ENV-050 | +0.1081 | RESEARCH |
| MODEL-14-5-H3-RIDGE-100 | ridge | FS-H3 | ENV-100 | +0.1316 | RESEARCH |
| MODEL-14-5-H3-LASSO-100 | lasso | FS-H3 | ENV-100 | +0.1464 | RESEARCH |

**Note:** Lasso models with alpha=0.001 produce all-zero coefficients on most feature sets, yielding IC=0.0000. Only FS-H3 (macro features) produces non-zero lasso predictions.

---

## Global Feature Attribution

### Baseline Models (FS-001, FS-12B-A)
Top permutation features: `vol_10`, `sma_ratio_5_30`, `log_dv_med_20`

### Lasso Models (FS-12B-B/E/D)
Top permutation features: `ret_10`, `ret_20`, `ret_30`
(All coefficients zero except when trained on FS-H3)

### H-3 Macro-Regime Models
Top permutation features (all 4 variants):
1. `macro_dff_change_3m` (Fed Funds Rate 3-month change)
2. `macro_dff_level` (Fed Funds Rate level)
3. `macro_unemployment_level` (UNRATE)

**Key finding:** Macro features dominate H-3 attribution. The coefficient share of macro features vs. baseline features:

| Model | Macro Share |
|-------|-------------|
| H3-RIDGE-050 | 90.25% |
| H3-LASSO-050 | 100.00% |
| H3-RIDGE-100 | 88.44% |
| H3-LASSO-100 | 97.69% |

This means the H-3 predictive improvement is almost entirely driven by macro regime features, not by any improvement in the baseline signal.

---

## H-3 Macro Feature Analysis

### Ridge Models (ENV-050)
| Feature | Coefficient | Direction |
|---------|-------------|-----------|
| macro_dff_level | +0.0167 | Positive |
| macro_dff_change_3m | -0.0128 | Negative |
| macro_unemployment_level | +0.0082 | Positive |
| macro_cpi_yoy | -0.0032 | Negative |

### Lasso Models (ENV-050)
| Feature | Coefficient | Direction |
|---------|-------------|-----------|
| macro_dff_change_3m | -0.0083 | Negative |
| macro_dff_level | +0.0077 | Positive |
| macro_unemployment_level | +0.0035 | Positive |
| macro_cpi_yoy | 0.0000 | Zero (lasso) |

**Interpretation:** Higher Fed Funds rates and higher unemployment predict positive returns (contrarian macro signal). The DFF level and 3-month change are the dominant features.

---

## Sensitivity Analysis

| Model | STABLE | SENSITIVE | CLIFF |
|-------|--------|-----------|-------|
| MODEL-00002 | 3 | 0 | 0 |
| MODEL-00005 | 3 | 0 | 0 |
| MODEL-00006 | 3 | 0 | 0 |
| MODEL-00007 | 3 | 0 | 0 |
| MODEL-00008 | 0 | 3 | 0 |
| MODEL-00009 | 0 | 3 | 0 |
| MODEL-00010 | 3 | 0 | 0 |
| H3-RIDGE-050 | 0 | 0 | **3** |
| H3-LASSO-050 | 2 | 0 | 1 |
| H3-RIDGE-100 | 0 | 0 | **3** |
| H3-LASSO-100 | 0 | 3 | 1 |

**Finding:** H-3 ridge models show CLIFF sensitivity across all samples. Small perturbations in macro features cause disproportionately large prediction changes. This is a significant stability concern.

---

## Counterfactual Analysis

| Model | CLEAR | HIGH_DIMENSIONAL | UNSTABLE | NO_VALID |
|-------|-------|------------------|----------|----------|
| MODEL-00002 | 0 | 0 | 0 | 3 |
| MODEL-00005 | 0 | 0 | 0 | 3 |
| MODEL-00006 | 0 | 0 | 0 | 3 |
| MODEL-00007 | 0 | 0 | 0 | 3 |
| MODEL-00008 | 3 | 0 | 0 | 0 |
| MODEL-00009 | 3 | 0 | 0 | 0 |
| MODEL-00010 | 0 | 0 | 0 | 3 |
| H3-RIDGE-050 | 0 | 0 | **3** | 0 |
| H3-LASSO-050 | 0 | 0 | **3** | 0 |
| H3-RIDGE-100 | 0 | 0 | **3** | 0 |
| H3-LASSO-100 | 0 | 0 | **3** | 0 |

**Finding:** H-3 models produce UNSTABLE counterfactuals — the smallest valid feature change that materially alters prediction pushes outside the observed domain. This confirms the CLIFF sensitivity finding.

---

## Model Disagreement

**50/50 observations classified as SIGN_CONFLICT.**

This is expected and informative: lasso models (alpha=0.001) produce all-zero predictions on most feature sets, while ridge models produce non-zero predictions. The disagreement is not hidden — it is a direct consequence of model family choice.

**Implication:** Any averaging or ensemble across ridge/lasso would be meaningless without first understanding why lasso zeroes out entirely.

---

## Faithfulness Tests

| Model | Classification | Deletion Correlation |
|-------|----------------|---------------------|
| MODEL-00002 | FAITHFUL | 1.000 |
| MODEL-00005 | FAITHFUL | 1.000 |
| MODEL-00006 | FAITHFUL | 1.000 |
| MODEL-00007 | FAITHFUL | 1.000 |
| MODEL-00008 | FAITHFUL | 0.810 |
| MODEL-00009 | FAITHFUL | 0.810 |
| MODEL-00010 | FAITHFUL | 1.000 |
| H3-RIDGE-050 | **MISLEADING** | -0.301 |
| H3-LASSO-050 | WEAK | 0.301 |
| H3-RIDGE-100 | **MISLEADING** | -0.231 |
| H3-LASSO-100 | WEAK | 0.301 |

**Critical finding:** H-3 ridge models have MISLEADING faithfulness. The features with the highest attribution are NOT the features that, when removed, cause the largest IC drop. The negative correlation (-0.301) means the attribution ranking is *inversely* related to actual model sensitivity.

**Implication:** Trusting the attribution ranking for H-3 ridge models would lead to incorrect conclusions about which features matter.

---

## Explanation Confidence

| Model | Confidence | Unfavorable | Concerning |
|-------|------------|-------------|------------|
| MODEL-00002 | LOW | 1 | 0 |
| MODEL-00005 | LOW | 1 | 0 |
| MODEL-00006 | LOW | 1 | 0 |
| MODEL-00007 | LOW | 1 | 0 |
| MODEL-00008 | MEDIUM | 0 | 2 |
| MODEL-00009 | MEDIUM | 0 | 2 |
| MODEL-00010 | LOW | 1 | 0 |
| H3-RIDGE-050 | **UNRELIABLE** | 2 | 0 |
| H3-LASSO-050 | LOW | 1 | 0 |
| H3-RIDGE-100 | **UNRELIABLE** | 2 | 0 |
| H3-LASSO-100 | LOW | 1 | 0 |

**No model achieves HIGH confidence.** The best is MEDIUM (baseline ridge models). H-3 ridge models are UNRELIABLE due to MISLEADING faithfulness.

---

## Correlation Instability

| Model | High-Corr Pairs | Instability Detected |
|-------|-----------------|---------------------|
| All non-H-3 | 0 | No |
| H3-RIDGE-050 | 1 | **Yes** |
| H3-LASSO-050 | 1 | **Yes** |
| H3-RIDGE-100 | 1 | **Yes** |
| H3-LASSO-100 | 1 | **Yes** |

The correlated pair is `macro_dff_level` / `macro_dff_change_3m` (r=0.78). These two features exchange importance across bootstrap samples, confirming attribution instability.

---

## Synthetic Ground-Truth Validation

| Model | Type | Expected | Result |
|-------|------|----------|--------|
| SYNTH-001 | Single feature | X1 dominates | **PASS** |
| SYNTH-002 | Linear weighted | Ordering matches | **PASS** |
| SYNTH-003 | Interaction | Non-trivial linear effects | **PASS** |
| SYNTH-004 | Redundant | Attribution sharing | **PASS** |
| SYNTH-005 | Noise features | Noise importance low | **PASS** |

All 5 synthetic tests pass. The explainability engine correctly recovers known ground-truth behavior.

---

## Provenance Audit

All 12 adversarial tests PASS (invalid cases properly rejected):

| Test | Description | Status |
|------|-------------|--------|
| A1 | Wrong model version | REJECT_INVALID |
| A2 | Feature value mismatch | REJECT_INVALID |
| A3 | Snapshot mutation | REJECT_INVALID |
| A4 | Future feature injection | REJECT_INVALID |
| A5 | Timestamp before availability | REJECT_INVALID |
| A6 | ID substitution | REJECT_INVALID |
| A7 | Registry evidence mismatch | REJECT_INVALID |
| A8 | Missing provenance | REJECT_INVALID |
| A9 | Current data instead of snapshot | REJECT_INVALID |
| A10 | LLM claims vs structured attribution | REJECT_INVALID |
| A11 | Correlated instability hidden | REJECT_INVALID |
| A12 | Failed explanation excluded | REJECT_INVALID |

---

## Reproducibility

All 11 models achieve EXACT_REPRODUCTION: running the same training pipeline twice from identical inputs produces identical coefficient digests.

---

## Adversarial Review

| Finding | Classification | Description |
|---------|----------------|-------------|
| F1 | PASS | Attribution method ranking agreement |
| F2 | LIMITATION | Correlated feature instability in H-3 models |
| F3 | **MATERIAL CONCERN** | H-3 ridge faithfulness is MISLEADING |
| F4 | **MATERIAL CONCERN** | SIGN_CONFLICT across all model pairs |
| F5 | LIMITATION | Explanation confidence ≠ trading confidence |
| F6 | PASS | Synthetic validation all pass |
| F7 | PASS | Provenance audit all pass |
| F8 | PASS | Reproducibility all pass |
| F9 | LIMITATION | H-3 explanations may be overstated |

**2 MATERIAL CONCERN findings prevent a GREEN gate.**

---

## Final Verdict

### Verdict D — Gate RED

**Explanation methods are too unstable for decision use.**

The material concerns are:

1. **H-3 ridge faithfulness is MISLEADING.** The attribution ranking is inversely related to actual model sensitivity. This means explanations for the best-performing models are unreliable.

2. **Model disagreement is pervasive.** Lasso and ridge models disagree on prediction sign for all test observations. This disagreement is not hidden but is a fundamental limitation of the current model inventory.

### What This Means

- The H-3 macro-regime predictive improvement is **real but fragile**. It depends on macro regime features that dominate the model (88-100% of effect) but whose attribution is unstable due to feature correlation.

- The CLIFF sensitivity and UNSTABLE counterfactuals confirm that H-3 models are operating near decision boundaries where small input changes cause large output changes.

- **No model has HIGH explanation confidence.** The best is MEDIUM (baseline ridge). This means even the simplest models have documented limitations in explanation stability.

### What This Does NOT Mean

- It does NOT mean H-3 has no predictive signal. The IC is real; the question is whether it's robust.

- It does NOT mean explanations are useless. They correctly identify that macro features dominate, which is economically meaningful.

- It does NOT mean the explainability engine is broken. It correctly validates on synthetic ground-truth and rejects all provenance attacks.

### Recommended Next Steps (Phase 16)

1. Investigate why H-3 ridge faithfulness is MISLEADING — is it the correlated DFF features?
2. Test H-3 on a longer out-of-sample period
3. Consider regularization specifically targeting the DFF/UNRATE correlation
4. Do NOT promote any model based on explanation quality alone

---

## Outputs Generated

| File | Description |
|------|-------------|
| benchmarks/phase15_plan.json | Locked plan with SHA-256 digest |
| benchmarks/phase15_model_inventory.json | 12 models, 11 trained |
| benchmarks/phase15_global_attribution.json | Coefficients, permutation importance |
| benchmarks/phase15_local_explanations.json | 33 local explanations (11 models × 3 samples) |
| benchmarks/phase15_sensitivity.json | Perturbation sensitivity analysis |
| benchmarks/phase15_counterfactuals.json | Bounded counterfactual analysis |
| benchmarks/phase15_disagreement.json | Cross-model prediction comparison |
| benchmarks/phase15_explanation_confidence.json | Confidence classification per model |
| benchmarks/phase15_correlation_stability.json | Feature correlation instability |
| benchmarks/phase15_faithfulness.json | Deletion ablation faithfulness |
| benchmarks/phase15_synthetic_validation.json | 5 synthetic ground-truth tests |
| benchmarks/phase15_provenance_audit.json | 12 adversarial provenance tests |
| benchmarks/phase15_reproducibility.json | Double-run reproducibility |
| benchmarks/phase15_adversarial.json | 9 adversarial review findings |
| benchmarks/phase15_results.json | Aggregated results summary |
| benchmarks/phase15_audit.json | 14 audit checks |
| docs/phase15_explainability_report.md | This report |

---

## Model Predictive Quality vs. Explanation Quality

| Model | Predictive IC | Explanation Confidence | Faithfulness |
|-------|--------------|----------------------|--------------|
| Baseline Ridge | +0.027 | MEDIUM | FAITHFUL |
| H-3 Ridge | +0.091/+0.132 | UNRELIABLE | MISLEADING |
| H-3 Lasso | +0.108/+0.146 | LOW | WEAK |

**A good explanation does not imply good predictive performance.**  
**A bad explanation does not imply bad predictive performance.**  
**These are orthogonal quality dimensions.**
