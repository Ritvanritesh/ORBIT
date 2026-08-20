# Phase 10 - Feature Engineering + Ablation (benchmark table)

Generated from `phase10_feature_research.parquet` (deterministic rows).

## Protocol

- Dataset: DS-000004 (20-symbol development universe)
- Label: LAB-004 v1 (5-session forward total return)
- Windows: train 2010-01-04..2018-12-31, val 2019-01-02..2021-12-31, test 2022-01-03..2026-06-30 (locked)
- Cost model: CM-001 (spread 2 bps, fees 1 bps, slippage 2 bps)
- Signal construction: top-3 long, equal weight 1/3 (Phase 9 path)

## Results

### EXP-10001 - FS-001 (base ) - ridge

- status: `completed` | params: `{"alpha": 1.0}`
- features: 8 (digest 1137e3fda1fa8656...)
- OOS IC: -0.0003 | rank IC: 0.0003
- after-cost total return: 49.3164% | turnover: 545.2741 | costs: 54527.38
- notes: top-3 long, weight 1/3; calibration fit on validation; phase9_parent=EXP-90003

### EXP-10002 - FS-001 (base ) - lasso

- status: `completed` | params: `{"alpha": 0.001}`
- features: 8 (digest 1137e3fda1fa8656...)
- OOS IC: 0.0226 | rank IC: 0.0122
- after-cost total return: 21.1397% | turnover: 313.1856 | costs: 31318.53
- notes: top-3 long, weight 1/3; calibration fit on validation; phase9_parent=EXP-90006

### EXP-10003 - FS-001 (base ) - random_forest

- status: `completed` | params: `{"max_depth": 3, "n_estimators": 200}`
- features: 8 (digest 1137e3fda1fa8656...)
- OOS IC: 0.0119 | rank IC: 0.0168
- after-cost total return: 127.1341% | turnover: 968.6418 | costs: 96864.16
- notes: top-3 long, weight 1/3; calibration fit on validation; phase9_parent=EXP-90015

### EXP-10004 - FS-001 (base ) - xgboost

- status: `completed` | params: `{"learning_rate": 0.1, "max_depth": 3, "n_estimators": 200}`
- features: 8 (digest 1137e3fda1fa8656...)
- OOS IC: 0.0113 | rank IC: 0.0095
- after-cost total return: 23.6567% | turnover: 689.7041 | costs: 68970.40
- notes: top-3 long, weight 1/3; calibration fit on validation; phase9_parent=EXP-90019

### EXP-10005 - FS-002 (new ) - ridge

- status: `completed` | params: `{"alpha": 1.0}`
- features: 15 (digest 03d6f45483a6b7f3...)
- OOS IC: 0.0148 | rank IC: 0.0123
- after-cost total return: 24.0076% | turnover: 529.4375 | costs: 52943.73
- notes: top-3 long, weight 1/3; calibration fit on validation; phase9_parent=EXP-90003

### EXP-10006 - FS-002 (new ) - lasso

- status: `completed` | params: `{"alpha": 0.001}`
- features: 15 (digest 03d6f45483a6b7f3...)
- OOS IC: 0.0256 | rank IC: 0.0270
- after-cost total return: 201.8618% | turnover: 679.5963 | costs: 67959.61
- notes: top-3 long, weight 1/3; calibration fit on validation; phase9_parent=EXP-90006

### EXP-10007 - FS-002 (new ) - random_forest

- status: `completed` | params: `{"max_depth": 3, "n_estimators": 200}`
- features: 15 (digest 03d6f45483a6b7f3...)
- OOS IC: 0.0013 | rank IC: 0.0138
- after-cost total return: 146.5808% | turnover: 729.7811 | costs: 72978.12
- notes: top-3 long, weight 1/3; calibration fit on validation; phase9_parent=EXP-90015

### EXP-10008 - FS-002 (new ) - xgboost

- status: `completed` | params: `{"learning_rate": 0.1, "max_depth": 3, "n_estimators": 200}`
- features: 15 (digest 03d6f45483a6b7f3...)
- OOS IC: -0.0021 | rank IC: 0.0018
- after-cost total return: 57.0781% | turnover: 803.1786 | costs: 80317.85
- notes: top-3 long, weight 1/3; calibration fit on validation; phase9_parent=EXP-90019

### EXP-10009 - FS-003 (all ) - ridge

- status: `completed` | params: `{"alpha": 1.0}`
- features: 23 (digest 966db59a346d1805...)
- OOS IC: 0.0202 | rank IC: 0.0148
- after-cost total return: 7.5463% | turnover: 507.7576 | costs: 50775.74
- notes: top-3 long, weight 1/3; calibration fit on validation; phase9_parent=EXP-90003

### EXP-10010 - FS-003 (all ) - lasso

- status: `completed` | params: `{"alpha": 0.001}`
- features: 23 (digest 966db59a346d1805...)
- OOS IC: 0.0254 | rank IC: 0.0267
- after-cost total return: 200.4125% | turnover: 686.5670 | costs: 68656.68
- notes: top-3 long, weight 1/3; calibration fit on validation; phase9_parent=EXP-90006

### EXP-10011 - FS-003 (all ) - random_forest

- status: `completed` | params: `{"max_depth": 3, "n_estimators": 200}`
- features: 23 (digest 966db59a346d1805...)
- OOS IC: -0.0056 | rank IC: 0.0094
- after-cost total return: 94.9169% | turnover: 750.5413 | costs: 75054.13
- notes: top-3 long, weight 1/3; calibration fit on validation; phase9_parent=EXP-90015

### EXP-10012 - FS-003 (all ) - xgboost

- status: `completed` | params: `{"learning_rate": 0.1, "max_depth": 3, "n_estimators": 200}`
- features: 23 (digest 966db59a346d1805...)
- OOS IC: 0.0100 | rank IC: 0.0125
- after-cost total return: 116.5684% | turnover: 866.2086 | costs: 86620.84
- notes: top-3 long, weight 1/3; calibration fit on validation; phase9_parent=EXP-90019

### EXP-10013 - FS-004 (base_plus_family momentum) - ridge

- status: `completed` | params: `{"alpha": 1.0}`
- features: 11 (digest 83a9d5803d57e26a...)
- OOS IC: 0.0082 | rank IC: 0.0067
- after-cost total return: 66.0230% | turnover: 669.2814 | costs: 66928.14
- notes: top-3 long, weight 1/3; calibration fit on validation; phase9_parent=EXP-90003

### EXP-10014 - FS-004 (base_plus_family momentum) - lasso

- status: `completed` | params: `{"alpha": 0.001}`
- features: 11 (digest 83a9d5803d57e26a...)
- OOS IC: 0.0073 | rank IC: 0.0065
- after-cost total return: 59.4617% | turnover: 742.7142 | costs: 74271.40
- notes: top-3 long, weight 1/3; calibration fit on validation; phase9_parent=EXP-90006

### EXP-10015 - FS-004 (base_plus_family momentum) - random_forest

- status: `completed` | params: `{"max_depth": 3, "n_estimators": 200}`
- features: 11 (digest 83a9d5803d57e26a...)
- OOS IC: 0.0055 | rank IC: 0.0134
- after-cost total return: 103.2839% | turnover: 891.0547 | costs: 89105.47
- notes: top-3 long, weight 1/3; calibration fit on validation; phase9_parent=EXP-90015

### EXP-10016 - FS-004 (base_plus_family momentum) - xgboost

- status: `completed` | params: `{"learning_rate": 0.1, "max_depth": 3, "n_estimators": 200}`
- features: 11 (digest 83a9d5803d57e26a...)
- OOS IC: 0.0131 | rank IC: 0.0157
- after-cost total return: 63.3620% | turnover: 817.7522 | costs: 81775.20
- notes: top-3 long, weight 1/3; calibration fit on validation; phase9_parent=EXP-90019

### EXP-10017 - FS-005 (base_plus_family trend) - ridge

- status: `completed` | params: `{"alpha": 1.0}`
- features: 11 (digest 0c923e2950cd60e7...)
- OOS IC: 0.0009 | rank IC: 0.0010
- after-cost total return: 60.0751% | turnover: 520.1107 | costs: 52011.06
- notes: top-3 long, weight 1/3; calibration fit on validation; phase9_parent=EXP-90003

### EXP-10018 - FS-005 (base_plus_family trend) - lasso

- status: `completed` | params: `{"alpha": 0.001}`
- features: 11 (digest 0c923e2950cd60e7...)
- OOS IC: 0.0238 | rank IC: 0.0213
- after-cost total return: 24.6917% | turnover: 322.9311 | costs: 32293.08
- notes: top-3 long, weight 1/3; calibration fit on validation; phase9_parent=EXP-90006

### EXP-10019 - FS-005 (base_plus_family trend) - random_forest

- status: `completed` | params: `{"max_depth": 3, "n_estimators": 200}`
- features: 11 (digest 0c923e2950cd60e7...)
- OOS IC: -0.0081 | rank IC: 0.0053
- after-cost total return: 61.1783% | turnover: 721.3102 | costs: 72131.00
- notes: top-3 long, weight 1/3; calibration fit on validation; phase9_parent=EXP-90015

### EXP-10020 - FS-005 (base_plus_family trend) - xgboost

- status: `completed` | params: `{"learning_rate": 0.1, "max_depth": 3, "n_estimators": 200}`
- features: 11 (digest 0c923e2950cd60e7...)
- OOS IC: 0.0042 | rank IC: 0.0152
- after-cost total return: 27.8649% | turnover: 674.3499 | costs: 67434.98
- notes: top-3 long, weight 1/3; calibration fit on validation; phase9_parent=EXP-90019

### EXP-10021 - FS-006 (base_plus_family volatility) - ridge

- status: `completed` | params: `{"alpha": 1.0}`
- features: 11 (digest 40b4590d5d420aae...)
- OOS IC: 0.0121 | rank IC: 0.0153
- after-cost total return: 28.8810% | turnover: 469.4662 | costs: 46946.60
- notes: top-3 long, weight 1/3; calibration fit on validation; phase9_parent=EXP-90003

### EXP-10022 - FS-006 (base_plus_family volatility) - lasso

- status: `completed` | params: `{"alpha": 0.001}`
- features: 11 (digest 40b4590d5d420aae...)
- OOS IC: 0.0358 | rank IC: 0.0412
- after-cost total return: 93.1407% | turnover: 305.5082 | costs: 30550.79
- notes: top-3 long, weight 1/3; calibration fit on validation; phase9_parent=EXP-90006

### EXP-10023 - FS-006 (base_plus_family volatility) - random_forest

- status: `completed` | params: `{"max_depth": 3, "n_estimators": 200}`
- features: 11 (digest 40b4590d5d420aae...)
- OOS IC: 0.0058 | rank IC: 0.0275
- after-cost total return: 83.9461% | turnover: 699.5049 | costs: 69950.47
- notes: top-3 long, weight 1/3; calibration fit on validation; phase9_parent=EXP-90015

### EXP-10024 - FS-006 (base_plus_family volatility) - xgboost

- status: `completed` | params: `{"learning_rate": 0.1, "max_depth": 3, "n_estimators": 200}`
- features: 11 (digest 40b4590d5d420aae...)
- OOS IC: 0.0092 | rank IC: 0.0121
- after-cost total return: 30.3770% | turnover: 656.1183 | costs: 65611.82
- notes: top-3 long, weight 1/3; calibration fit on validation; phase9_parent=EXP-90019

### EXP-10025 - FS-007 (base_plus_family volume) - ridge

- status: `completed` | params: `{"alpha": 1.0}`
- features: 11 (digest 9eab7bf46089ea7a...)
- OOS IC: 0.0079 | rank IC: 0.0111
- after-cost total return: 9.7276% | turnover: 428.8724 | costs: 42887.21
- notes: top-3 long, weight 1/3; calibration fit on validation; phase9_parent=EXP-90003

### EXP-10026 - FS-007 (base_plus_family volume) - lasso

- status: `completed` | params: `{"alpha": 0.001}`
- features: 11 (digest 9eab7bf46089ea7a...)
- OOS IC: 0.0238 | rank IC: 0.0213
- after-cost total return: 24.6917% | turnover: 322.9311 | costs: 32293.08
- notes: top-3 long, weight 1/3; calibration fit on validation; phase9_parent=EXP-90006

### EXP-10027 - FS-007 (base_plus_family volume) - random_forest

- status: `completed` | params: `{"max_depth": 3, "n_estimators": 200}`
- features: 11 (digest 9eab7bf46089ea7a...)
- OOS IC: 0.0027 | rank IC: 0.0159
- after-cost total return: 93.7232% | turnover: 913.6659 | costs: 91366.57
- notes: top-3 long, weight 1/3; calibration fit on validation; phase9_parent=EXP-90015

### EXP-10028 - FS-007 (base_plus_family volume) - xgboost

- status: `completed` | params: `{"learning_rate": 0.1, "max_depth": 3, "n_estimators": 200}`
- features: 11 (digest 9eab7bf46089ea7a...)
- OOS IC: 0.0180 | rank IC: 0.0104
- after-cost total return: 43.2588% | turnover: 710.6545 | costs: 71065.42
- notes: top-3 long, weight 1/3; calibration fit on validation; phase9_parent=EXP-90019

### EXP-10029 - FS-008 (base_plus_family range) - ridge

- status: `completed` | params: `{"alpha": 1.0}`
- features: 11 (digest 45968259497fa0b0...)
- OOS IC: 0.0023 | rank IC: 0.0008
- after-cost total return: 49.8809% | turnover: 500.2010 | costs: 50020.07
- notes: top-3 long, weight 1/3; calibration fit on validation; phase9_parent=EXP-90003

### EXP-10030 - FS-008 (base_plus_family range) - lasso

- status: `completed` | params: `{"alpha": 0.001}`
- features: 11 (digest 45968259497fa0b0...)
- OOS IC: 0.0220 | rank IC: 0.0178
- after-cost total return: 153.7786% | turnover: 640.6037 | costs: 64060.34
- notes: top-3 long, weight 1/3; calibration fit on validation; phase9_parent=EXP-90006

### EXP-10031 - FS-008 (base_plus_family range) - random_forest

- status: `completed` | params: `{"max_depth": 3, "n_estimators": 200}`
- features: 11 (digest 45968259497fa0b0...)
- OOS IC: 0.0006 | rank IC: 0.0046
- after-cost total return: 38.3854% | turnover: 614.9472 | costs: 61494.70
- notes: top-3 long, weight 1/3; calibration fit on validation; phase9_parent=EXP-90015

### EXP-10032 - FS-008 (base_plus_family range) - xgboost

- status: `completed` | params: `{"learning_rate": 0.1, "max_depth": 3, "n_estimators": 200}`
- features: 11 (digest 45968259497fa0b0...)
- OOS IC: 0.0172 | rank IC: 0.0172
- after-cost total return: 56.0058% | turnover: 699.6822 | costs: 69968.19
- notes: top-3 long, weight 1/3; calibration fit on validation; phase9_parent=EXP-90019

### EXP-10033 - FS-009 (all_minus_family momentum) - ridge

- status: `completed` | params: `{"alpha": 1.0}`
- features: 20 (digest 9fe80a99f543528e...)
- OOS IC: 0.0167 | rank IC: 0.0165
- after-cost total return: 18.7865% | turnover: 415.8127 | costs: 41581.24
- notes: top-3 long, weight 1/3; calibration fit on validation; phase9_parent=EXP-90003

### EXP-10034 - FS-009 (all_minus_family momentum) - lasso

- status: `completed` | params: `{"alpha": 0.001}`
- features: 20 (digest 9fe80a99f543528e...)
- OOS IC: 0.0312 | rank IC: 0.0270
- after-cost total return: 221.2059% | turnover: 508.5221 | costs: 50852.19
- notes: top-3 long, weight 1/3; calibration fit on validation; phase9_parent=EXP-90006

### EXP-10035 - FS-009 (all_minus_family momentum) - random_forest

- status: `completed` | params: `{"max_depth": 3, "n_estimators": 200}`
- features: 20 (digest 9fe80a99f543528e...)
- OOS IC: -0.0048 | rank IC: 0.0256
- after-cost total return: 141.0314% | turnover: 777.9940 | costs: 77799.38
- notes: top-3 long, weight 1/3; calibration fit on validation; phase9_parent=EXP-90015

### EXP-10036 - FS-009 (all_minus_family momentum) - xgboost

- status: `completed` | params: `{"learning_rate": 0.1, "max_depth": 3, "n_estimators": 200}`
- features: 20 (digest 9fe80a99f543528e...)
- OOS IC: 0.0062 | rank IC: 0.0118
- after-cost total return: 9.2770% | turnover: 609.9191 | costs: 60991.88
- notes: top-3 long, weight 1/3; calibration fit on validation; phase9_parent=EXP-90019

### EXP-10037 - FS-010 (all_minus_family trend) - ridge

- status: `completed` | params: `{"alpha": 1.0}`
- features: 20 (digest 101c4fff5978ae62...)
- OOS IC: 0.0196 | rank IC: 0.0197
- after-cost total return: 66.2217% | turnover: 636.2488 | costs: 63624.87
- notes: top-3 long, weight 1/3; calibration fit on validation; phase9_parent=EXP-90003

### EXP-10038 - FS-010 (all_minus_family trend) - lasso

- status: `completed` | params: `{"alpha": 0.001}`
- features: 20 (digest 101c4fff5978ae62...)
- OOS IC: 0.0254 | rank IC: 0.0267
- after-cost total return: 200.4125% | turnover: 686.5670 | costs: 68656.68
- notes: top-3 long, weight 1/3; calibration fit on validation; phase9_parent=EXP-90006

### EXP-10039 - FS-010 (all_minus_family trend) - random_forest

- status: `completed` | params: `{"max_depth": 3, "n_estimators": 200}`
- features: 20 (digest 101c4fff5978ae62...)
- OOS IC: 0.0022 | rank IC: 0.0182
- after-cost total return: 63.8878% | turnover: 653.7234 | costs: 65372.34
- notes: top-3 long, weight 1/3; calibration fit on validation; phase9_parent=EXP-90015

### EXP-10040 - FS-010 (all_minus_family trend) - xgboost

- status: `completed` | params: `{"learning_rate": 0.1, "max_depth": 3, "n_estimators": 200}`
- features: 20 (digest 101c4fff5978ae62...)
- OOS IC: 0.0252 | rank IC: 0.0171
- after-cost total return: 100.4752% | turnover: 874.6112 | costs: 87461.10
- notes: top-3 long, weight 1/3; calibration fit on validation; phase9_parent=EXP-90019

### EXP-10041 - FS-011 (all_minus_family volatility) - ridge

- status: `completed` | params: `{"alpha": 1.0}`
- features: 20 (digest 314d45e83c09fc03...)
- OOS IC: 0.0163 | rank IC: 0.0099
- after-cost total return: 57.7588% | turnover: 606.0535 | costs: 60605.35
- notes: top-3 long, weight 1/3; calibration fit on validation; phase9_parent=EXP-90003

### EXP-10042 - FS-011 (all_minus_family volatility) - lasso

- status: `completed` | params: `{"alpha": 0.001}`
- features: 20 (digest 314d45e83c09fc03...)
- OOS IC: 0.0169 | rank IC: 0.0140
- after-cost total return: 97.4792% | turnover: 708.2475 | costs: 70824.72
- notes: top-3 long, weight 1/3; calibration fit on validation; phase9_parent=EXP-90006

### EXP-10043 - FS-011 (all_minus_family volatility) - random_forest

- status: `completed` | params: `{"max_depth": 3, "n_estimators": 200}`
- features: 20 (digest 314d45e83c09fc03...)
- OOS IC: -0.0128 | rank IC: 0.0009
- after-cost total return: 55.7370% | turnover: 725.8018 | costs: 72580.17
- notes: top-3 long, weight 1/3; calibration fit on validation; phase9_parent=EXP-90015

### EXP-10044 - FS-011 (all_minus_family volatility) - xgboost

- status: `completed` | params: `{"learning_rate": 0.1, "max_depth": 3, "n_estimators": 200}`
- features: 20 (digest 314d45e83c09fc03...)
- OOS IC: 0.0168 | rank IC: 0.0038
- after-cost total return: 163.6155% | turnover: 1146.8323 | costs: 114683.21
- notes: top-3 long, weight 1/3; calibration fit on validation; phase9_parent=EXP-90019

### EXP-10045 - FS-012 (all_minus_family volume) - ridge

- status: `completed` | params: `{"alpha": 1.0}`
- features: 20 (digest a4afbcb4d9f879e3...)
- OOS IC: 0.0157 | rank IC: 0.0085
- after-cost total return: 17.5693% | turnover: 458.0279 | costs: 45802.76
- notes: top-3 long, weight 1/3; calibration fit on validation; phase9_parent=EXP-90003

### EXP-10046 - FS-012 (all_minus_family volume) - lasso

- status: `completed` | params: `{"alpha": 0.001}`
- features: 20 (digest a4afbcb4d9f879e3...)
- OOS IC: 0.0254 | rank IC: 0.0267
- after-cost total return: 200.4125% | turnover: 686.5670 | costs: 68656.68
- notes: top-3 long, weight 1/3; calibration fit on validation; phase9_parent=EXP-90006

### EXP-10047 - FS-012 (all_minus_family volume) - random_forest

- status: `completed` | params: `{"max_depth": 3, "n_estimators": 200}`
- features: 20 (digest a4afbcb4d9f879e3...)
- OOS IC: -0.0064 | rank IC: 0.0060
- after-cost total return: 116.6455% | turnover: 667.1556 | costs: 66715.56
- notes: top-3 long, weight 1/3; calibration fit on validation; phase9_parent=EXP-90015

### EXP-10048 - FS-012 (all_minus_family volume) - xgboost

- status: `completed` | params: `{"learning_rate": 0.1, "max_depth": 3, "n_estimators": 200}`
- features: 20 (digest a4afbcb4d9f879e3...)
- OOS IC: 0.0142 | rank IC: 0.0110
- after-cost total return: 51.1059% | turnover: 612.2973 | costs: 61229.70
- notes: top-3 long, weight 1/3; calibration fit on validation; phase9_parent=EXP-90019

### EXP-10049 - FS-013 (all_minus_family range) - ridge

- status: `completed` | params: `{"alpha": 1.0}`
- features: 20 (digest b30143c99de3b4cc...)
- OOS IC: 0.0190 | rank IC: 0.0173
- after-cost total return: 39.2977% | turnover: 616.6784 | costs: 61667.83
- notes: top-3 long, weight 1/3; calibration fit on validation; phase9_parent=EXP-90003

### EXP-10050 - FS-013 (all_minus_family range) - lasso

- status: `completed` | params: `{"alpha": 0.001}`
- features: 20 (digest b30143c99de3b4cc...)
- OOS IC: 0.0236 | rank IC: 0.0257
- after-cost total return: 160.9663% | turnover: 712.3999 | costs: 71239.97
- notes: top-3 long, weight 1/3; calibration fit on validation; phase9_parent=EXP-90006

### EXP-10051 - FS-013 (all_minus_family range) - random_forest

- status: `completed` | params: `{"max_depth": 3, "n_estimators": 200}`
- features: 20 (digest b30143c99de3b4cc...)
- OOS IC: -0.0029 | rank IC: 0.0098
- after-cost total return: 75.2780% | turnover: 670.3313 | costs: 67033.12
- notes: top-3 long, weight 1/3; calibration fit on validation; phase9_parent=EXP-90015

### EXP-10052 - FS-013 (all_minus_family range) - xgboost

- status: `completed` | params: `{"learning_rate": 0.1, "max_depth": 3, "n_estimators": 200}`
- features: 20 (digest b30143c99de3b4cc...)
- OOS IC: 0.0070 | rank IC: 0.0144
- after-cost total return: 241.1603% | turnover: 1092.5423 | costs: 109254.21
- notes: top-3 long, weight 1/3; calibration fit on validation; phase9_parent=EXP-90019
