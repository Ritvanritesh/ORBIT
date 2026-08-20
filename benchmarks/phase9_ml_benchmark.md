# Phase 9 - Baseline ML Benchmark (permanent report)

Generated from `phase9_ml_benchmark.parquet` (deterministic rows; a rerun appends or refreshes rows by experiment_id).

## Protocol

- Dataset: DS-000004 (20-symbol development universe)
- Features: FS-001 v1 (8 point-in-time numerics, strict boundary)
- Label: LAB-004 v1 (5-session forward total return)
- Windows: train 2010-01-04..2018-12-31, val 2019-01-02..2021-12-31, test 2022-01-03..2026-06-30 (locked)
- Evaluation: OOS IC / rank IC (per-session mean), ECE (10 bins), Brier, MSE, hit rate
- Backtest: canonical Phase 7, WEIGHT sizing, CM-001 costs (2 bps spread, 1 bps fees, 2 bps slippage), open/delay=1

## Results

### control - buy_and_hold (EXP-90021)

- status: `completed`
- params: `{}`
- after-cost total return: 58.0008% | turnover: 0.9937 | costs: 99.33
- notes: Phase 8 documented rules on the real dataset; strict point-in-time boundary

### control - equal_weight (EXP-90022)

- status: `completed`
- params: `{}`
- after-cost total return: 61.2266% | turnover: 17.7454 | costs: 1774.50
- notes: Phase 8 documented rules on the real dataset; strict point-in-time boundary

### control - mean_reversion (EXP-90026)

- status: `completed`
- params: `{"lookback": 10}`
- after-cost total return: -42.8993% | turnover: 362.1128 | costs: 36211.27
- notes: Phase 8 documented rules on the real dataset; strict point-in-time boundary

### control - mean_reversion (EXP-90027)

- status: `completed`
- params: `{"lookback": 20}`
- after-cost total return: 6.9205% | turnover: 360.4864 | costs: 36048.65
- notes: Phase 8 documented rules on the real dataset; strict point-in-time boundary

### control - mean_reversion (EXP-90028)

- status: `completed`
- params: `{"lookback": 30}`
- after-cost total return: -23.9474% | turnover: 243.4580 | costs: 24345.79
- notes: Phase 8 documented rules on the real dataset; strict point-in-time boundary

### control - momentum (EXP-90023)

- status: `completed`
- params: `{"lookback": 10}`
- after-cost total return: 26.6464% | turnover: 856.4915 | costs: 85649.18
- notes: Phase 8 documented rules on the real dataset; strict point-in-time boundary

### control - momentum (EXP-90024)

- status: `completed`
- params: `{"lookback": 20}`
- after-cost total return: 58.8672% | turnover: 628.7164 | costs: 62871.68
- notes: Phase 8 documented rules on the real dataset; strict point-in-time boundary

### control - momentum (EXP-90025)

- status: `completed`
- params: `{"lookback": 30}`
- after-cost total return: 161.3140% | turnover: 688.9114 | costs: 68891.12
- notes: Phase 8 documented rules on the real dataset; strict point-in-time boundary

### control - moving_average (EXP-90029)

- status: `completed`
- params: `{"long_window": 30, "short_window": 5}`
- after-cost total return: 205.5420% | turnover: 584.2375 | costs: 58423.72
- notes: Phase 8 documented rules on the real dataset; strict point-in-time boundary

### control - moving_average (EXP-90030)

- status: `completed`
- params: `{"long_window": 30, "short_window": 10}`
- after-cost total return: 136.9950% | turnover: 323.9789 | costs: 32397.86
- notes: Phase 8 documented rules on the real dataset; strict point-in-time boundary

### control - moving_average (EXP-90031)

- status: `completed`
- params: `{"long_window": 40, "short_window": 15}`
- after-cost total return: 124.3309% | turnover: 286.6747 | costs: 28667.43
- notes: Phase 8 documented rules on the real dataset; strict point-in-time boundary

### control - null_flat (EXP-90036)

- status: `completed`
- params: `{}`
- after-cost total return: 0.0000% | turnover: 0.0000 | costs: 0.00
- notes: Phase 8 documented rules on the real dataset; strict point-in-time boundary

### control - random_null (EXP-90035)

- status: `completed`
- params: `{}`
- after-cost total return: -15.4854% | turnover: 787.2690 | costs: 78726.88
- notes: Phase 8 documented rules on the real dataset; strict point-in-time boundary

### control - volatility_targeted (EXP-90032)

- status: `completed`
- params: `{"estimation_window": 10, "target_volatility": 0.1}`
- after-cost total return: 61.2266% | turnover: 17.7454 | costs: 1774.50
- notes: Phase 8 documented rules on the real dataset; strict point-in-time boundary

### control - volatility_targeted (EXP-90033)

- status: `completed`
- params: `{"estimation_window": 30, "target_volatility": 0.15}`
- after-cost total return: 61.2266% | turnover: 17.7454 | costs: 1774.50
- notes: Phase 8 documented rules on the real dataset; strict point-in-time boundary

### control - volatility_targeted (EXP-90034)

- status: `completed`
- params: `{"estimation_window": 60, "target_volatility": 0.2}`
- after-cost total return: 61.2266% | turnover: 17.7454 | costs: 1774.50
- notes: Phase 8 documented rules on the real dataset; strict point-in-time boundary

### ml - lasso (EXP-90005)

- status: `completed`
- params: `{"alpha": 0.0001}`
- OOS IC: 0.0052 | rank IC: 0.0028
- ECE: 0.0411 | Brier: 0.2503
- after-cost total return: 32.2400% | turnover: 458.5761 | costs: 45857.60
- notes: top-3 long, weight 1/3; calibration fit on validation

### ml - lasso (EXP-90006)

- status: `completed`
- params: `{"alpha": 0.001}`
- OOS IC: 0.0226 | rank IC: 0.0122
- ECE: 0.0411 | Brier: 0.2503
- after-cost total return: 21.1397% | turnover: 313.1856 | costs: 31318.53
- notes: top-3 long, weight 1/3; calibration fit on validation

### ml - lasso (EXP-90007)

- status: `completed`
- params: `{"alpha": 0.01}`
- OOS IC: nan | rank IC: nan
- ECE: 0.0411 | Brier: 0.2503
- after-cost total return: 45.4927% | turnover: 12.2039 | costs: 1220.35
- notes: top-3 long, weight 1/3; calibration fit on validation

### ml - lasso (EXP-90008)

- status: `completed`
- params: `{"alpha": 0.1}`
- OOS IC: nan | rank IC: nan
- ECE: 0.0411 | Brier: 0.2503
- after-cost total return: 45.4927% | turnover: 12.2039 | costs: 1220.35
- notes: top-3 long, weight 1/3; calibration fit on validation

### ml - logistic (EXP-90009)

- status: `completed`
- params: `{"C": 0.01}`
- OOS IC: -0.0202 | rank IC: -0.0126
- ECE: 0.0416 | Brier: 0.2504
- after-cost total return: 5.9417% | turnover: 488.5876 | costs: 48858.74
- notes: top-3 long, weight 1/3; calibration fit on validation

### ml - logistic (EXP-90010)

- status: `completed`
- params: `{"C": 0.1}`
- OOS IC: -0.0205 | rank IC: -0.0131
- ECE: 0.0416 | Brier: 0.2504
- after-cost total return: -5.4981% | turnover: 453.3746 | costs: 45337.45
- notes: top-3 long, weight 1/3; calibration fit on validation

### ml - logistic (EXP-90011)

- status: `completed`
- params: `{"C": 1.0}`
- OOS IC: -0.0205 | rank IC: -0.0130
- ECE: 0.0416 | Brier: 0.2504
- after-cost total return: -5.9043% | turnover: 454.7238 | costs: 45472.36
- notes: top-3 long, weight 1/3; calibration fit on validation

### ml - logistic (EXP-90012)

- status: `completed`
- params: `{"C": 10.0}`
- OOS IC: -0.0205 | rank IC: -0.0132
- ECE: 0.0416 | Brier: 0.2504
- after-cost total return: -7.3151% | turnover: 450.7528 | costs: 45075.26
- notes: top-3 long, weight 1/3; calibration fit on validation

### ml - random_forest (EXP-90013)

- status: `completed`
- params: `{"max_depth": 3, "n_estimators": 50}`
- OOS IC: 0.0100 | rank IC: 0.0189
- ECE: 0.0411 | Brier: 0.2503
- after-cost total return: 35.7925% | turnover: 687.6657 | costs: 68766.55
- notes: top-3 long, weight 1/3; calibration fit on validation

### ml - random_forest (EXP-90014)

- status: `completed`
- params: `{"max_depth": 6, "n_estimators": 50}`
- OOS IC: -0.0064 | rank IC: 0.0120
- ECE: 0.0411 | Brier: 0.2503
- after-cost total return: 34.9220% | turnover: 805.2502 | costs: 80525.01
- notes: top-3 long, weight 1/3; calibration fit on validation

### ml - random_forest (EXP-90015)

- status: `completed`
- params: `{"max_depth": 3, "n_estimators": 200}`
- OOS IC: 0.0119 | rank IC: 0.0168
- ECE: 0.0411 | Brier: 0.2503
- after-cost total return: 127.1341% | turnover: 968.6418 | costs: 96864.16
- notes: top-3 long, weight 1/3; calibration fit on validation

### ml - random_forest (EXP-90016)

- status: `completed`
- params: `{"max_depth": 6, "n_estimators": 200}`
- OOS IC: -0.0041 | rank IC: 0.0109
- ECE: 0.0411 | Brier: 0.2503
- after-cost total return: 47.3085% | turnover: 744.8502 | costs: 74485.01
- notes: top-3 long, weight 1/3; calibration fit on validation

### ml - ridge (EXP-90001)

- status: `completed`
- params: `{"alpha": 0.01}`
- OOS IC: -0.0003 | rank IC: 0.0003
- ECE: 0.0411 | Brier: 0.2503
- after-cost total return: 49.3164% | turnover: 545.2741 | costs: 54527.38
- notes: top-3 long, weight 1/3; calibration fit on validation

### ml - ridge (EXP-90002)

- status: `completed`
- params: `{"alpha": 0.1}`
- OOS IC: -0.0003 | rank IC: 0.0003
- ECE: 0.0411 | Brier: 0.2503
- after-cost total return: 49.3164% | turnover: 545.2741 | costs: 54527.38
- notes: top-3 long, weight 1/3; calibration fit on validation

### ml - ridge (EXP-90003)

- status: `completed`
- params: `{"alpha": 1.0}`
- OOS IC: -0.0003 | rank IC: 0.0003
- ECE: 0.0411 | Brier: 0.2503
- after-cost total return: 49.3164% | turnover: 545.2741 | costs: 54527.38
- notes: top-3 long, weight 1/3; calibration fit on validation

### ml - ridge (EXP-90004)

- status: `completed`
- params: `{"alpha": 10.0}`
- OOS IC: -0.0003 | rank IC: 0.0003
- ECE: 0.0411 | Brier: 0.2503
- after-cost total return: 49.2515% | turnover: 548.9717 | costs: 54897.15
- notes: top-3 long, weight 1/3; calibration fit on validation

### ml - xgboost (EXP-90017)

- status: `completed`
- params: `{"learning_rate": 0.1, "max_depth": 3, "n_estimators": 50}`
- OOS IC: 0.0021 | rank IC: 0.0044
- ECE: 0.0411 | Brier: 0.2503
- after-cost total return: 45.9966% | turnover: 703.2441 | costs: 70324.40
- notes: top-3 long, weight 1/3; calibration fit on validation

### ml - xgboost (EXP-90018)

- status: `completed`
- params: `{"learning_rate": 0.1, "max_depth": 6, "n_estimators": 50}`
- OOS IC: 0.0176 | rank IC: 0.0156
- ECE: 0.0411 | Brier: 0.2503
- after-cost total return: 34.5274% | turnover: 734.5994 | costs: 73459.93
- notes: top-3 long, weight 1/3; calibration fit on validation

### ml - xgboost (EXP-90019)

- status: `completed`
- params: `{"learning_rate": 0.1, "max_depth": 3, "n_estimators": 200}`
- OOS IC: 0.0113 | rank IC: 0.0095
- ECE: 0.0411 | Brier: 0.2503
- after-cost total return: 23.6567% | turnover: 689.7041 | costs: 68970.40
- notes: top-3 long, weight 1/3; calibration fit on validation

### ml - xgboost (EXP-90020)

- status: `completed`
- params: `{"learning_rate": 0.1, "max_depth": 6, "n_estimators": 200}`
- OOS IC: 0.0101 | rank IC: 0.0036
- ECE: 0.0411 | Brier: 0.2503
- after-cost total return: 25.0080% | turnover: 823.9668 | costs: 82396.67
- notes: top-3 long, weight 1/3; calibration fit on validation
