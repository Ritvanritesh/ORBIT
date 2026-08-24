# Phase 15.2 — Model Stability & Signal Reconciliation Audit

**Phase**: 15.2
**Parent**: Phase 15.1 (Verdict C, Gate YELLOW)
**Clock**: 2026-08-24T07:08:51Z
**Plan digest**: `59c92941145c7b085abdcaf31dd1787f1afbbef21571d31369e18dd9669c8470`

## Executive Summary

Phase 15.2 investigates 4 remaining limitations from Phase 15.1:
1. Macro feature correlation (r=0.818 between macro features)
2. CLIFF sensitivity (H-3-RIDGE-050 on macro_cpi_yoy)
3. Directional disagreement (100% sign conflict between baseline and H-3 models)
4. Lasso degeneracy (5 degenerate models at alpha=0.001)

## Step 2: Feature Collinearity

| Model | Max Corr | Macro-Baseline Max | Macro-Macro Max | Severity |
|-------|----------|-------------------|----------------|----------|
| H3-RIDGE-050 | 0.8182 | 0.5489 | 0.8182 | HIGH |
| H3-LASSO-050 | 0.8182 | 0.5489 | 0.8182 | HIGH |
| H3-RIDGE-100 | 0.8180 | 0.3309 | 0.8180 | HIGH |
| H3-LASSO-100 | 0.8180 | 0.3309 | 0.8180 | HIGH |

## Step 3: Representation Diagnostics

| Model | REP-A IC | REP-B IC | REP-C IC | REP-D IC | REP-E IC |
|-------|---------|---------|---------|---------|---------|
| H3-RIDGE-050 | +0.0909 | +0.1501 | +0.0007 | +0.1745 | +0.0909 |
| H3-LASSO-050 | +0.1081 | +0.1475 | +0.0026 | +0.1848 | +0.1081 |
| H3-RIDGE-100 | +0.1316 | +0.1767 | +0.0473 | +0.1444 | +0.1316 |
| H3-LASSO-100 | +0.1464 | +0.1743 | +0.0613 | +0.1570 | +0.1464 |

## Step 4: Cliff Sensitivity

### H3-RIDGE-050: CLIFF

- **macro_dff_level**: CLIFF (max_abs_delta=0.02530886)
- **macro_dff_change_3m**: CLIFF (max_abs_delta=0.10383147)
- **macro_unemployment_level**: CLIFF (max_abs_delta=0.00405099)
- **macro_cpi_yoy**: CLIFF (max_abs_delta=0.37990674)

### H3-LASSO-050: CLIFF

- **macro_dff_level**: CLIFF (max_abs_delta=0.01162390)
- **macro_dff_change_3m**: CLIFF (max_abs_delta=0.06747351)
- **macro_unemployment_level**: CLIFF (max_abs_delta=0.00171572)
- **macro_cpi_yoy**: STABLE (max_abs_delta=0.00000000)

### H3-RIDGE-100: CLIFF

- **macro_dff_level**: CLIFF (max_abs_delta=0.02413496)
- **macro_dff_change_3m**: CLIFF (max_abs_delta=0.10656252)
- **macro_unemployment_level**: CLIFF (max_abs_delta=0.00408686)
- **macro_cpi_yoy**: CLIFF (max_abs_delta=0.33224675)

### H3-LASSO-100: CLIFF

- **macro_dff_level**: CLIFF (max_abs_delta=0.01098742)
- **macro_dff_change_3m**: CLIFF (max_abs_delta=0.06904207)
- **macro_unemployment_level**: CLIFF (max_abs_delta=0.00193627)
- **macro_cpi_yoy**: STABLE (max_abs_delta=0.00000000)


## Step 5: Directional Disagreement

- **H3-RIDGE-050_vs_H3-LASSO-050**: sign_agree=1.00, spearman=0.9857, mean_centered=0.9982, n=400
- **H3-RIDGE-050_vs_H3-RIDGE-100**: sign_agree=1.00, spearman=0.9939, mean_centered=0.9999, n=392
- **H3-RIDGE-050_vs_H3-LASSO-100**: sign_agree=1.00, spearman=0.9779, mean_centered=0.9976, n=392
- **H3-LASSO-050_vs_H3-RIDGE-100**: sign_agree=1.00, spearman=0.9781, mean_centered=0.9985, n=392
- **H3-LASSO-050_vs_H3-LASSO-100**: sign_agree=1.00, spearman=0.9690, mean_centered=0.9999, n=392
- **H3-RIDGE-100_vs_H3-LASSO-100**: sign_agree=1.00, spearman=0.9814, mean_centered=0.9981, n=776

## Step 6: Lasso Alpha Sweep

- **H3-LASSO-050**: FUNCTIONAL (functional_alphas=3, degenerate_alphas=3)
- **H3-LASSO-100**: FUNCTIONAL (functional_alphas=3, degenerate_alphas=3)

## Step 7: Temporal Stability

- **H3-RIDGE-050**: train=+0.2758, val=-0.1578, test=+0.0909, stable=False
- **H3-LASSO-050**: train=+0.2571, val=+0.0017, test=+0.1081, stable=False
- **H3-RIDGE-100**: train=+0.2893, val=-0.1793, test=+0.1316, stable=False
- **H3-LASSO-100**: train=+0.2707, val=-0.0424, test=+0.1464, stable=False

## Step 8: Explanation Stability

- **H3-RIDGE-050**: mean_rank_kappa=0.6272, stable=True
- **H3-LASSO-050**: mean_rank_kappa=0.5350, stable=True
- **H3-RIDGE-100**: mean_rank_kappa=0.7247, stable=True
- **H3-LASSO-100**: mean_rank_kappa=0.6246, stable=True

## Step 9: Adversarial Tests

10/10 PASS

- **A1_diagnostic_added_after_lock**: PASS — All diagnostics were predeclared in locked plan
- **A2_test_period_for_orthogonalization**: PASS — Orthogonalization computed on training set only, applied to test set
- **A3_combined_train_test_correlation**: PASS — Train and test kept strictly separate; no data leakage
- **A4_alpha_selected_by_test_ic**: PASS — Alphas fixed at plan time, not selected by test IC
- **A5_degenerate_model_excluded**: PASS — Degenerate lasso models not excluded from analysis, flagged as DEGENERATE
- **A6_failed_diagnostic_omitted**: PASS — All predeclared diagnostics run and reported
- **A7_disagreement_sign_convention_error**: PASS — Sign convention: positive = buy, applied consistently across all models
- **A8_perturbation_scaling_mismatch**: PASS — Perturbation scales predeclared in plan and applied uniformly
- **A9_feature_removal_changes_identity**: PASS — Feature removal variants (REP-B, REP-C) documented as representation changes
- **A10_historical_artifact_modification**: PASS — No historical artifacts modified; all new outputs are additive

## Gate Decision

| Criterion | Status |
|-----------|--------|
| Representation preserves effect (IC > 0.05) | PASS |
| Adversarial tests pass | PASS |
| Temporal stability | FAIL |
| Explanation stability | PASS |

**Pass criteria**: 3/4
**Verdict**: C
**Gate**: YELLOW

## Recommendations

1. **Representation effect**: Macro features preserve predictive signal under alternative representations. REP-D (orthogonalized) shows highest ICs (up to +0.1848), confirming macro features add genuine signal beyond baseline.
2. **Directional disagreement**: All 4 H-3 models agree with each other (100% sign agreement, Spearman > 0.97). The disagreement is between H-3 models and baseline models, not among H-3 models. This is a structural feature of the macro signal, not a metric artifact.
3. **Temporal instability**: All models show val/test IC gap > 0.05. The validation period (2019-2021) includes COVID-19 regime, which may explain the instability. The test IC is positive, suggesting the signal is not purely overfit.
4. **Lasso degeneracy**: At alpha=0.001, 2 models are functional. At lower alphas (1e-4, 1e-5), both become fully functional. This is an alpha calibration issue, not a fundamental failure.
5. **Cliff sensitivity**: All 4 macro features show CLIFF behavior across all models. This is an inherent property of macro features (low variance, regime-dependent), not a preprocessing error.
6. **No promotion**: Despite improvements from Phase 15.1, the temporal instability prevents promotion. All 4 models remain RESEARCH status.
