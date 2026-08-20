# ORBIT Phase 10: Feature Engineering + Ablation

Version 1.0 - permanent research report

## 1. Purpose

Phase 10 investigates whether the Phase 9 null result was caused by an insufficient feature representation. The central question: do additional, scientifically justified, point-in-time-valid feature families contain incremental information beyond FS-001 v1? The design isolates FEATURE REPRESENTATION, not model complexity: the Phase 9 model families are reused unchanged with pre-registered grid points, the dataset (DS-000004), label (LAB-004 v1), split, cost model and Phase 7 backtester are identical to Phase 9, and every experiment is registered before execution.

## 2. Feature families

| Family | Feature IDs | Kind |
|--------|-------------|------|
| momentum | FEAT-101, FEAT-102, FEAT-103 | momentum_return |
| trend | FEAT-104, FEAT-105, FEAT-106 | moving_average_ratio, trend_distance |
| volatility | FEAT-107, FEAT-108, FEAT-109 | realized_volatility, volatility_ratio |
| volume | FEAT-110, FEAT-111, FEAT-112 | dollar_volume_median, volume_zscore |
| range | FEAT-113, FEAT-114, FEAT-115 | high_low_position, normalized_range |

Only families computable from the existing DS-000004 OHLCV bars are implemented. No fundamentals, macro, news, text, options, or alternative data were invented (they belong to later data-expansion phases).

## Feature inventory

Every feature: ID, name, family, definition, lookback, temporal boundary, missingness policy, snapshot version. All Phase 10 features are point-in-time-valid at the strict boundary (window_end_session = D-1 < decision session D).

| ID | Name | Family | Definition | Lookback | Raw inputs | Missing policy |
|----|------|--------|------------|----------|------------|----------------|
| FEAT-001 | ret_10 | phase9_base | momentum_return | 10 |  |  |
| FEAT-002 | ret_20 | phase9_base | momentum_return | 20 |  |  |
| FEAT-003 | ret_30 | phase9_base | momentum_return | 30 |  |  |
| FEAT-004 | sma_ratio_5_30 | phase9_base | moving_average_ratio |  |  |  |
| FEAT-005 | sma_ratio_15_40 | phase9_base | moving_average_ratio |  |  |  |
| FEAT-006 | vol_10 | phase9_base | realized_volatility | 10 |  |  |
| FEAT-007 | vol_30 | phase9_base | realized_volatility | 30 |  |  |
| FEAT-008 | log_dv_med_20 | phase9_base | liquidity | 20 |  |  |
| FEAT-101 | ret_5 | momentum | close(D-1)/close(D-5) - 1 | 5 | close | null until 5 completed sessions before D |
| FEAT-102 | ret_60 | momentum | close(D-1)/close(D-60) - 1 | 60 | close | null until 60 completed sessions before D |
| FEAT-103 | ret_120 | momentum | close(D-1)/close(D-120) - 1 | 120 | close | null until 120 completed sessions before D |
| FEAT-104 | sma_ratio_10_30 | trend | SMA10(D-1)/SMA30(D-1) (ratio, same form as FEAT-004/005) |  | close | null until 30 completed sessions before D |
| FEAT-105 | sma_ratio_20_50 | trend | SMA20(D-1)/SMA50(D-1) (ratio, same form as FEAT-004/005) |  | close | null until 50 completed sessions before D |
| FEAT-106 | price_distance_200ma | trend | (close(D-1) - SMA200(D-1)) / SMA200(D-1) | 200 | close | null until 200 completed sessions before D |
| FEAT-107 | vol_60 | volatility | sample std of daily close-to-close returns over the 60 sessions ending at D-1 (same rolling_std semantics as FEAT-006/007) | 60 | close | null until 60 completed sessions before D |
| FEAT-108 | vol_90 | volatility | sample std of daily close-to-close returns over the 90 sessions ending at D-1 | 90 | close | null until 90 completed sessions before D |
| FEAT-109 | vol_ratio_10_30 | volatility | vol_10(D-1) / vol_30(D-1) (short/long realized-volatility ratio) |  | close | null until 30 completed sessions before D; null when vol_30 is 0 |
| FEAT-110 | dv_med_10 | volume | log1p(median(close*volume) over the 10 sessions ending at D-1) (same form as FEAT-008) | 10 | close,volume | null until 10 completed sessions before D |
| FEAT-111 | dv_med_30 | volume | log1p(median(close*volume) over the 30 sessions ending at D-1) | 30 | close,volume | null until 30 completed sessions before D |
| FEAT-112 | vol_zscore_20 | volume | (dv(D-1) - mean(dv over 20 sessions ending at D-1)) / std(dv over 20 sessions ending at D-1); dv = close*volume | 20 | close,volume | null until 20 completed sessions before D; null when the 20-session std is 0 |
| FEAT-113 | high_low_10_pos | range | (close(D-1) - min(low over 10 sessions ending at D-1)) / (max(high over 10 sessions ending at D-1) - min(low over 10 sessions ending at D-1)) | 10 | close,high,low | null until 10 completed sessions before D; null when the 10-session range is 0 |
| FEAT-114 | high_low_30_pos | range | (close(D-1) - min(low over 30 sessions ending at D-1)) / (max(high over 30 sessions ending at D-1) - min(low over 30 sessions ending at D-1)) | 30 | close,high,low | null until 30 completed sessions before D; null when the 30-session range is 0 |
| FEAT-115 | normalized_range_20 | range | (max(high over 20 sessions ending at D-1) - min(low over 20 sessions ending at D-1)) / close(D-1) | 20 | close,high,low | null until 20 completed sessions before D |

## 3. Feature snapshots

FS-001 v1 is frozen (digest 1137e3fda1fa8656...). New immutable snapshots:

| Snapshot | Role | Members | Digest |
|----------|------|---------|--------|
| FS-001 v1 | base (frozen) | 8 | 1137e3fda1fa8656... |
| FS-002 v1 | new | 15 | 03d6f45483a6b7f3... |
| FS-003 v1 | all | 23 | 966db59a346d1805... |
| FS-004 v1 | base_plus_family | 11 | 83a9d5803d57e26a... |
| FS-005 v1 | base_plus_family | 11 | 0c923e2950cd60e7... |
| FS-006 v1 | base_plus_family | 11 | 40b4590d5d420aae... |
| FS-007 v1 | base_plus_family | 11 | 9eab7bf46089ea7a... |
| FS-008 v1 | base_plus_family | 11 | 45968259497fa0b0... |
| FS-009 v1 | all_minus_family | 20 | 9fe80a99f543528e... |
| FS-010 v1 | all_minus_family | 20 | 101c4fff5978ae62... |
| FS-011 v1 | all_minus_family | 20 | 314d45e83c09fc03... |
| FS-012 v1 | all_minus_family | 20 | a4afbcb4d9f879e3... |
| FS-013 v1 | all_minus_family | 20 | b30143c99de3b4cc... |

## 4. Diagnostics

```json
{
  "feature_sets": {
    "FS-001": {
      "quality": {
        "features": [
          {
            "feature": "ret_10",
            "frac_most_common_value": 0.002027072681817158,
            "is_constant": false,
            "is_near_constant": false,
            "missing_frac": 0.0,
            "n_null": 0,
            "n_unique": 44200,
            "percentiles": {
              "0.01": -0.12002125583554885,
              "0.05": -0.06666786668966719,
              "0.5": 0.005829544357810246,
              "0.95": 0.07669393806753894,
              "0.99": 0.14144574747122307
            },
            "rows": 44399
          },
          {
            "feature": "ret_20",
            "frac_most_common_value": 0.001328858758080137,
            "is_constant": false,
            "is_near_constant": false,
            "missing_frac": 0.0,
            "n_null": 0,
            "n_unique": 44246,
            "percentiles": {
              "0.01": -0.15571680591252318,
              "0.05": -0.08988102878919126,
              "0.5": 0.012346614645121878,
              "0.95": 0.11658716896246331,
              "0.99": 0.20945454277272824
            },
            "rows": 44399
          },
          {
            "feature": "ret_30",
            "frac_most_common_value": 0.0011036284601004527,
            "is_constant": false,
            "is_near_constant": false,
            "missing_frac": 0.0,
            "n_null": 0,
            "n_unique": 44272,
            "percentiles": {
              "0.01": -0.1873942521093042,
              "0.05": -0.10607281850600835,
              "0.5": 0.01764118776566903,
              "0.95": 0.15118289085674144,
              "0.99": 0.2809306796284045
            },
            "rows": 44399
          },
          {
            "feature": "sma_ratio_5_30",
            "frac_most_common_value": 2.2523029797968422e-05,
            "is_constant": false,
            "is_near_constant": false,
            "missing_frac": 0.0,
            "n_null": 0,
            "n_unique": 44399,
            "percentiles": {
              "0.01": 0.8995394480587233,
              "0.05": 0.9443231074538143,
              "0.5": 1.0080115137921053,
              "0.95": 1.067809151861312,
              "0.99": 1.1202904946296355
            },
            "rows": 44399
          },
          {
            "feature": "sma_ratio_15_40",
            "frac_most_common_value": 2.2523029797968422e-05,
            "is_constant": false,
            "is_near_constant": false,
            "missing_frac": 0.0,
            "n_null": 0,
            "n_unique": 44399,
            "percentiles": {
              "0.01": 0.9158199655910249,
              "0.05": 0.9538706958759304,
              "0.5": 1.0079065728077585,
              "0.95": 1.05916131972537,
              "0.99": 1.1052262827387558
            },
            "rows": 44399
          },
          {
            "feature": "vol_10",
            "frac_most_common_value": 4.5046059595936844e-05,
            "is_constant": false,
            "is_near_constant": false,
            "missing_frac": 0.0,
            "n_null": 0,
            "n_unique": 44396,
            "percentiles": {
              "0.01": 0.0035010024425757512,
              "0.05": 0.0049786334410091655,
              "0.5": 0.011599517195158846,
              "0.95": 0.03092689423311777,
              "0.99": 0.047978882147934145
            },
            "rows": 44399
          },
          {
            "feature": "vol_30",
            "frac_most_common_value": 4.5046059595936844e-05,
            "is_constant": false,
            "is_near_constant": false,
            "missing_frac": 0.0,
            "n_null": 0,
            "n_unique": 44396,
            "percentiles": {
              "0.01": 0.004918518382428328,
              "0.05": 0.006315676515547055,
              "0.5": 0.012368180104352953,
              "0.95": 0.030030993218364423,
              "0.99": 0.04371824008984934
            },
            "rows": 44399
          },
          {
            "feature": "log_dv_med_20",
            "frac_most_common_value": 0.0003153224171715579,
            "is_constant": false,
            "is_near_constant": false,
            "missing_frac": 0.0,
            "n_null": 0,
            "n_unique": 20706,
            "percentiles": {
              "0.01": 17.301735618996762,
              "0.05": 19.39076859907243,
              "0.5": 20.488964217980037,
              "0.95": 22.092245317071086,
              "0.99": 22.774902583831697
            },
            "rows": 44399
          }
        ],
        "n_rows": 44399,
        "split_stats": [
          {
            "feature": "ret_10",
            "max": 0.6911089393387799,
            "mean": 0.006174018786929825,
            "min": -0.3489999771118164,
            "n_null": 0,
            "rows": 44399,
            "split": "train",
            "std": 0.04796607885011429
          },
          {
            "feature": "ret_20",
            "max": 1.0435266071777156,
            "mean": 0.013301111222223503,
            "min": -0.3684210146662068,
            "n_null": 0,
            "rows": 44399,
            "split": "train",
            "std": 0.06990387438425325
          },
          {
            "feature": "ret_30",
            "max": 1.4200482793736717,
            "mean": 0.020643955933139124,
            "min": -0.45507268956117053,
            "n_null": 0,
            "rows": 44399,
            "split": "train",
            "std": 0.0885116906187921
          },
          {
            "feature": "sma_ratio_5_30",
            "max": 1.5309583140922272,
            "mean": 1.007557800914384,
            "min": 0.7339216712314885,
            "n_null": 0,
            "rows": 44399,
            "split": "train",
            "std": 0.04117768246091516
          },
          {
            "feature": "sma_ratio_15_40",
            "max": 1.4055219739205849,
            "mean": 1.0074913086974033,
            "min": 0.7856628664280966,
            "n_null": 0,
            "rows": 44399,
            "split": "train",
            "std": 0.03536571434292026
          },
          {
            "feature": "vol_10",
            "max": 0.10663484464732022,
            "mean": 0.013893742264023966,
            "min": 0.0015616788656635896,
            "n_null": 0,
            "rows": 44399,
            "split": "train",
            "std": 0.009058792570832584
          },
          {
            "feature": "vol_30",
            "max": 0.07478897964239332,
            "mean": 0.01446821518481949,
            "min": 0.002968462793863276,
            "n_null": 0,
            "rows": 44399,
            "split": "train",
            "std": 0.007944775645858005
          },
          {
            "feature": "log_dv_med_20",
            "max": 23.616243563140582,
            "mean": 20.56692328271212,
            "min": 15.604335482532274,
            "n_null": 0,
            "rows": 44399,
            "split": "train",
            "std": 0.8816685238585874
          }
        ]
      },
      "redundancy": {
        "duplicates": [],
        "high_correlation_pairs": [],
        "pearson": {
          "features": [
            "ret_10",
            "ret_20",
            "ret_30",
            "sma_ratio_5_30",
            "sma_ratio_15_40",
            "vol_10",
            "vol_30",
            "log_dv_med_20"
          ],
          "matrix": "[[ 1.          0.67952783  0.56041555  0.69425249  0.28102717  0.03668575\n   0.06414068 -0.03389358]\n [ 0.67952783  1.          0.80887059  0.91622045  0.73328518 -0.02551727\n   0.06496323 -0.02971144]\n [ 0.56041555  0.80887059  1.          0.87353104  0.89648746 -0.02764583\n   0.05545343 -0.0204993 ]\n [ 0.69425249  0.91622045  0.87353104  1.          0.80379899 -0.0722564\n   0.01796346 -0.03092222]\n [ 0.28102717  0.73328518  0.89648746  0.80379899  1.         -0.10025704\n  -0.02611943 -0.01279342]\n [ 0.03668575 -0.02551727 -0.02764583 -0.0722564  -0.10025704  1.\n   0.81795895  0.05531198]\n [ 0.06414068  0.06496323  0.05545343  0.01796346 -0.02611943  0.81795895\n   1.          0.05953264]\n [-0.03389358 -0.02971144 -0.0204993  -0.03092222 -0.01279342  0.05531198\n   0.05953264  1.        ]]",
          "method": "pearson",
          "pairs": [
            {
              "correlation": 0.91622,
              "feature_a": "ret_20",
              "feature_b": "sma_ratio_5_30"
            },
            {
              "correlation": 0.896487,
              "feature_a": "ret_30",
              "feature_b": "sma_ratio_15_40"
            },
            {
              "correlation": 0.873531,
              "feature_a": "ret_30",
              "feature_b": "sma_ratio_5_30"
            },
            {
              "correlation": 0.817959,
              "feature_a": "vol_10",
              "feature_b": "vol_30"
            },
            {
              "correlation": 0.808871,
              "feature_a": "ret_20",
              "feature_b": "ret_30"
            },
            {
              "correlation": 0.803799,
              "feature_a": "sma_ratio_5_30",
              "feature_b": "sma_ratio_15_40"
            },
            {
              "correlation": 0.733285,
              "feature_a": "ret_20",
              "feature_b": "sma_ratio_15_40"
            },
            {
              "correlation": 0.694252,
              "feature_a": "ret_10",
              "feature_b": "sma_ratio_5_30"
            },
            {
              "correlation": 0.679528,
              "feature_a": "ret_10",
              "feature_b": "ret_20"
            },
            {
              "correlation": 0.560416,
              "feature_a": "ret_10",
              "feature_b": "ret_30"
            },
            {
              "correlation": 0.281027,
              "feature_a": "ret_10",
              "feature_b": "sma_ratio_15_40"
            },
            {
              "correlation": -0.100257,
              "feature_a": "sma_ratio_15_40",
              "feature_b": "vol_10"
            },
            {
              "correlation": -0.072256,
              "feature_a": "sma_ratio_5_30",
              "feature_b": "vol_10"
            },
            {
              "correlation": 0.064963,
              "feature_a": "ret_20",
              "feature_b": "vol_30"
            },
            {
              "correlation": 0.064141,
              "feature_a": "ret_10",
              "feature_b": "vol_30"
            },
            {
              "correlation": 0.059533,
              "feature_a": "vol_30",
              "feature_b": "log_dv_med_20"
            },
            {
              "correlation": 0.055453,
              "feature_a": "ret_30",
              "feature_b": "vol_30"
            },
            {
              "correlation": 0.055312,
              "feature_a": "vol_10",
              "feature_b": "log_dv_med_20"
            },
            {
              "correlation": 0.036686,
              "feature_a": "ret_10",
              "feature_b": "vol_10"
            },
            {
              "correlation": -0.033894,
              "feature_a": "ret_10",
              "feature_b": "log_dv_med_20"
            },
            {
              "correlation": -0.030922,
              "feature_a": "sma_ratio_5_30",
              "feature_b": "log_dv_med_20"
            },
            {
              "correlation": -0.029711,
              "feature_a": "ret_20",
              "feature_b": "log_dv_med_20"
            },
            {
              "correlation": -0.027646,
              "feature_a": "ret_30",
              "feature_b": "vol_10"
            },
            {
              "correlation": -0.026119,
              "feature_a": "sma_ratio_15_40",
              "feature_b": "vol_30"
            },
            {
              "correlation": -0.025517,
              "feature_a": "ret_20",
              "feature_b": "vol_10"
            },
            {
              "correlation": -0.020499,
              "feature_a": "ret_30",
              "feature_b": "log_dv_med_20"
            },
            {
              "correlation": 0.017963,
              "feature_a": "sma_ratio_5_30",
              "feature_b": "vol_30"
            },
            {
              "correlation": -0.012793,
              "feature_a": "sma_ratio_15_40",
              "feature_b": "log_dv_med_20"
            }
          ],
          "rows_used": 44399
        },
        "spearman": {
          "features": [
            "ret_10",
            "ret_20",
            "ret_30",
            "sma_ratio_5_30",
            "sma_ratio_15_40",
            "vol_10",
            "vol_30",
            "log_dv_med_20"
          ],
          "matrix": "[[ 1.          0.63521498  0.5110463   0.64496689  0.23658017 -0.045716\n   0.01357252 -0.01057548]\n [ 0.63521498  1.          0.77177138  0.90037106  0.68739201 -0.12407963\n  -0.02415748 -0.01867395]\n [ 0.5110463   0.77177138  1.          0.85674059  0.8805052  -0.12490699\n  -0.06214016 -0.02338221]\n [ 0.64496689  0.90037106  0.85674059  1.          0.77402402 -0.13942763\n  -0.0370588  -0.02117347]\n [ 0.23658017  0.68739201  0.8805052   0.77402402  1.         -0.13184529\n  -0.08111114 -0.01718044]\n [-0.045716   -0.12407963 -0.12490699 -0.13942763 -0.13184529  1.\n   0.80899606  0.23219926]\n [ 0.01357252 -0.02415748 -0.06214016 -0.0370588  -0.08111114  0.80899606\n   1.          0.27431349]\n [-0.01057548 -0.01867395 -0.02338221 -0.02117347 -0.01718044  0.23219926\n   0.27431349  1.        ]]",
          "method": "spearman",
          "pairs": [
            {
              "correlation": 0.900371,
              "feature_a": "ret_20",
              "feature_b": "sma_ratio_5_30"
            },
            {
              "correlation": 0.880505,
              "feature_a": "ret_30",
              "feature_b": "sma_ratio_15_40"
            },
            {
              "correlation": 0.856741,
              "feature_a": "ret_30",
              "feature_b": "sma_ratio_5_30"
            },
            {
              "correlation": 0.808996,
              "feature_a": "vol_10",
              "feature_b": "vol_30"
            },
            {
              "correlation": 0.774024,
              "feature_a": "sma_ratio_5_30",
              "feature_b": "sma_ratio_15_40"
            },
            {
              "correlation": 0.771771,
              "feature_a": "ret_20",
              "feature_b": "ret_30"
            },
            {
              "correlation": 0.687392,
              "feature_a": "ret_20",
              "feature_b": "sma_ratio_15_40"
            },
            {
              "correlation": 0.644967,
              "feature_a": "ret_10",
              "feature_b": "sma_ratio_5_30"
            },
            {
              "correlation": 0.635215,
              "feature_a": "ret_10",
              "feature_b": "ret_20"
            },
            {
              "correlation": 0.511046,
              "feature_a": "ret_10",
              "feature_b": "ret_30"
            },
            {
              "correlation": 0.274313,
              "feature_a": "vol_30",
              "feature_b": "log_dv_med_20"
            },
            {
              "correlation": 0.23658,
              "feature_a": "ret_10",
              "feature_b": "sma_ratio_15_40"
            },
            {
              "correlation": 0.232199,
              "feature_a": "vol_10",
              "feature_b": "log_dv_med_20"
            },
            {
              "correlation": -0.139428,
              "feature_a": "sma_ratio_5_30",
              "feature_b": "vol_10"
            },
            {
              "correlation": -0.131845,
              "feature_a": "sma_ratio_15_40",
              "feature_b": "vol_10"
            },
            {
              "correlation": -0.124907,
              "feature_a": "ret_30",
              "feature_b": "vol_10"
            },
            {
              "correlation": -0.12408,
              "feature_a": "ret_20",
              "feature_b": "vol_10"
            },
            {
              "correlation": -0.081111,
              "feature_a": "sma_ratio_15_40",
              "feature_b": "vol_30"
            },
            {
              "correlation": -0.06214,
              "feature_a": "ret_30",
              "feature_b": "vol_30"
            },
            {
              "correlation": -0.045716,
              "feature_a": "ret_10",
              "feature_b": "vol_10"
            },
            {
              "correlation": -0.037059,
              "feature_a": "sma_ratio_5_30",
              "feature_b": "vol_30"
            },
            {
              "correlation": -0.024157,
              "feature_a": "ret_20",
              "feature_b": "vol_30"
            },
            {
              "correlation": -0.023382,
              "feature_a": "ret_30",
              "feature_b": "log_dv_med_20"
            },
            {
              "correlation": -0.021173,
              "feature_a": "sma_ratio_5_30",
              "feature_b": "log_dv_med_20"
            },
            {
              "correlation": -0.018674,
              "feature_a": "ret_20",
              "feature_b": "log_dv_med_20"
            },
            {
              "correlation": -0.01718,
              "feature_a": "sma_ratio_15_40",
              "feature_b": "log_dv_med_20"
            },
            {
              "correlation": 0.013573,
              "feature_a": "ret_10",
              "feature_b": "vol_30"
            },
            {
              "correlation": -0.010575,
              "feature_a": "ret_10",
              "feature_b": "log_dv_med_20"
            }
          ],
          "rows_used": 44399
        },
        "train_rows": 44399
      }
    },
    "FS-002": {
      "quality": {
        "features": [
          {
            "feature": "ret_5",
            "frac_most_common_value": 0.0034029810113659566,
            "is_constant": false,
            "is_near_constant": false,
            "missing_frac": 0.0,
            "n_null": 0,
            "n_unique": 43742,
            "percentiles": {
              "0.01": -0.08519447854909686,
              "0.05": -0.04657101473959926,
              "0.5": 0.0026413562600937635,
              "0.95": 0.050431776131422947,
              "0.99": 0.09011672879234522
            },
            "rows": 44079
          },
          {
            "feature": "ret_60",
            "frac_most_common_value": 0.0007486558225005105,
            "is_constant": false,
            "is_near_constant": false,
            "missing_frac": 0.0,
            "n_null": 0,
            "n_unique": 43998,
            "percentiles": {
              "0.01": -0.22603053979847226,
              "0.05": -0.13145417109363425,
              "0.5": 0.03670873487124049,
              "0.95": 0.2259978661085785,
              "0.99": 0.4485449516843603
            },
            "rows": 44079
          },
          {
            "feature": "ret_120",
            "frac_most_common_value": 0.0003629846412123687,
            "is_constant": false,
            "is_near_constant": false,
            "missing_frac": 0.0,
            "n_null": 0,
            "n_unique": 44018,
            "percentiles": {
              "0.01": -0.28326662573546213,
              "0.05": -0.14591622400797652,
              "0.5": 0.07388087086835426,
              "0.95": 0.36839598537377294,
              "0.99": 0.7658738384982484
            },
            "rows": 44079
          },
          {
            "feature": "sma_ratio_10_30",
            "frac_most_common_value": 2.2686540075773043e-05,
            "is_constant": false,
            "is_near_constant": false,
            "missing_frac": 0.0,
            "n_null": 0,
            "n_unique": 44079,
            "percentiles": {
              "0.01": 0.9204806940984668,
              "0.05": 0.9559253758344521,
              "0.5": 1.0064500154936735,
              "0.95": 1.0535450276290457,
              "0.99": 1.092460644823427
            },
            "rows": 44079
          },
          {
            "feature": "sma_ratio_20_50",
            "frac_most_common_value": 2.2686540075773043e-05,
            "is_constant": false,
            "is_near_constant": false,
            "missing_frac": 0.0,
            "n_null": 0,
            "n_unique": 44079,
            "percentiles": {
              "0.01": 0.9143528387323512,
              "0.05": 0.9520096265817739,
              "0.5": 1.0093712362647946,
              "0.95": 1.0635588305045758,
              "0.99": 1.1128943541289564
            },
            "rows": 44079
          },
          {
            "feature": "price_distance_200ma",
            "frac_most_common_value": 4.5373080151546086e-05,
            "is_constant": false,
            "is_near_constant": false,
            "missing_frac": 0.0,
            "n_null": 0,
            "n_unique": 44078,
            "percentiles": {
              "0.01": -0.2181947725364505,
              "0.05": -0.10517973950299972,
              "0.5": 0.05976649479212117,
              "0.95": 0.25095854781451005,
              "0.99": 0.5067726706958768
            },
            "rows": 44079
          },
          {
            "feature": "vol_60",
            "frac_most_common_value": 4.5373080151546086e-05,
            "is_constant": false,
            "is_near_constant": false,
            "missing_frac": 0.0,
            "n_null": 0,
            "n_unique": 44074,
            "percentiles": {
              "0.01": 0.0055466523308712445,
              "0.05": 0.007007644893428239,
              "0.5": 0.012685875153042179,
              "0.95": 0.028405130251284213,
              "0.99": 0.04244269190019712
            },
            "rows": 44079
          },
          {
            "feature": "vol_90",
            "frac_most_common_value": 4.5373080151546086e-05,
            "is_constant": false,
            "is_near_constant": false,
            "missing_frac": 0.0,
            "n_null": 0,
            "n_unique": 44077,
            "percentiles": {
              "0.01": 0.0060177553547351445,
              "0.05": 0.007316666310178612,
              "0.5": 0.01281876677603952,
              "0.95": 0.028384252536876122,
              "0.99": 0.039462237730868086
            },
            "rows": 44079
          },
          {
            "feature": "vol_ratio_10_30",
            "frac_most_common_value": 2.2686540075773043e-05,
            "is_constant": false,
            "is_near_constant": false,
            "missing_frac": 0.0,
            "n_null": 0,
            "n_unique": 44079,
            "percentiles": {
              "0.01": 0.37031201529048763,
              "0.05": 0.5145880370186117,
              "0.5": 0.9595451145612078,
              "0.95": 1.4414981556396955,
              "0.99": 1.5968279829146168
            },
            "rows": 44079
          },
          {
            "feature": "dv_med_10",
            "frac_most_common_value": 0.00020417886068195738,
            "is_constant": false,
            "is_near_constant": false,
            "missing_frac": 0.0,
            "n_null": 0,
            "n_unique": 25697,
            "percentiles": {
              "0.01": 17.688198206010366,
              "0.05": 19.407266344368413,
              "0.5": 20.503008362880127,
              "0.95": 22.123256355080706,
              "0.99": 22.826972743750584
            },
            "rows": 44079
          },
          {
            "feature": "dv_med_30",
            "frac_most_common_value": 0.0004537308015154609,
            "is_constant": false,
            "is_near_constant": false,
            "missing_frac": 0.0,
            "n_null": 0,
            "n_unique": 18548,
            "percentiles": {
              "0.01": 17.574291661484615,
              "0.05": 19.445723920957043,
              "0.5": 20.48174114793814,
              "0.95": 22.081243555658737,
              "0.99": 22.771109763438513
            },
            "rows": 44079
          },
          {
            "feature": "vol_zscore_20",
            "frac_most_common_value": 2.2686540075773043e-05,
            "is_constant": false,
            "is_near_constant": false,
            "missing_frac": 0.0,
            "n_null": 0,
            "n_unique": 44079,
            "percentiles": {
              "0.01": -1.6819276623681836,
              "0.05": -1.2507613715349037,
              "0.5": -0.2240219197741499,
              "0.95": 2.2098992999615907,
              "0.99": 3.476521717969192
            },
            "rows": 44079
          },
          {
            "feature": "high_low_10_pos",
            "frac_most_common_value": 0.006079992740307176,
            "is_constant": false,
            "is_near_constant": false,
            "missing_frac": 0.0,
            "n_null": 0,
            "n_unique": 41958,
            "percentiles": {
              "0.01": 0.009889234285109048,
              "0.05": 0.06090791660548324,
              "0.5": 0.5974011733099402,
              "0.95": 0.9704271578324056,
              "0.99": 0.9962381581197126
            },
            "rows": 44079
          },
          {
            "feature": "high_low_30_pos",
            "frac_most_common_value": 0.003992831053336056,
            "is_constant": false,
            "is_near_constant": false,
            "missing_frac": 0.0,
            "n_null": 0,
            "n_unique": 42804,
            "percentiles": {
              "0.01": 0.014286236024797828,
              "0.05": 0.07264451505584091,
              "0.5": 0.6460256987912134,
              "0.95": 0.9746168953794586,
              "0.99": 0.9961080112723782
            },
            "rows": 44079
          },
          {
            "feature": "normalized_range_20",
            "frac_most_common_value": 6.805962022731914e-05,
            "is_constant": false,
            "is_near_constant": false,
            "missing_frac": 0.0,
            "n_null": 0,
            "n_unique": 43594,
            "percentiles": {
              "0.01": 0.02880692916098055,
              "0.05": 0.03839202706116745,
              "0.5": 0.08281248097440994,
              "0.95": 0.21174442842723853,
              "0.99": 0.31201389776078714
            },
            "rows": 44079
          }
        ],
        "n_rows": 44079,
        "split_stats": [
          {
            "feature": "ret_5",
            "max": 0.5816967988474475,
            "mean": 0.0026756634202760956,
            "min": -0.31401471170796547,
            "n_null": 0,
            "rows": 44079,
            "split": "train",
            "std": 0.0320526204984578
          },
          {
            "feature": "ret_60",
            "max": 2.100899205931436,
            "mean": 0.04375938749919386,
            "min": -0.51544503030466,
            "n_null": 0,
            "rows": 44079,
            "split": "train",
            "std": 0.13254476040000207
          },
          {
            "feature": "ret_120",
            "max": 3.834796729281427,
            "mean": 0.09467406994914238,
            "min": -0.5832705336062745,
            "n_null": 0,
            "rows": 44079,
            "split": "train",
            "std": 0.21711777027542925
          },
          {
            "feature": "sma_ratio_10_30",
            "max": 1.3706788100338605,
            "mean": 1.0059632211598166,
            "min": 0.8072433430117467,
            "n_null": 0,
            "rows": 44079,
            "split": "train",
            "std": 0.031928543325962525
          },
          {
            "feature": "sma_ratio_20_50",
            "max": 1.4078583578644,
            "mean": 1.0090617050359507,
            "min": 0.7873475414436673,
            "n_null": 0,
            "rows": 44079,
            "split": "train",
            "std": 0.03727462740472448
          },
          {
            "feature": "price_distance_200ma",
            "max": 1.8574830846214505,
            "mean": 0.06834975291071914,
            "min": -0.5171254377381036,
            "n_null": 0,
            "rows": 44079,
            "split": "train",
            "std": 0.13568246576909126
          },
          {
            "feature": "vol_60",
            "max": 0.06342152258097594,
            "mean": 0.014528036849176076,
            "min": 0.004065848213442764,
            "n_null": 0,
            "rows": 44079,
            "split": "train",
            "std": 0.007156646653350946
          },
          {
            "feature": "vol_90",
            "max": 0.05567015172956919,
            "mean": 0.014638731138695849,
            "min": 0.004685476193175406,
            "n_null": 0,
            "rows": 44079,
            "split": "train",
            "std": 0.006910788960585458
          },
          {
            "feature": "vol_ratio_10_30",
            "max": 1.7348770309840975,
            "mean": 0.9652698767578071,
            "min": 0.13793209538108092,
            "n_null": 0,
            "rows": 44079,
            "split": "train",
            "std": 0.27916079825580403
          },
          {
            "feature": "dv_med_10",
            "max": 23.730753953205873,
            "mean": 20.58659496200901,
            "min": 16.614860951920644,
            "n_null": 0,
            "rows": 44079,
            "split": "train",
            "std": 0.8645630093721696
          },
          {
            "feature": "dv_med_30",
            "max": 23.57904635268519,
            "mean": 20.575551948811444,
            "min": 16.771171081296785,
            "n_null": 0,
            "rows": 44079,
            "split": "train",
            "std": 0.8486108140790976
          },
          {
            "feature": "vol_zscore_20",
            "max": 4.227336983944763,
            "mean": 0.019491480212611566,
            "min": -2.891870736831146,
            "n_null": 0,
            "rows": 44079,
            "split": "train",
            "std": 1.046009731909504
          },
          {
            "feature": "high_low_10_pos",
            "max": 1.0,
            "mean": 0.560743766955331,
            "min": 0.0,
            "n_null": 0,
            "rows": 44079,
            "split": "train",
            "std": 0.3013550587748326
          },
          {
            "feature": "high_low_30_pos",
            "max": 1.0,
            "mean": 0.5920283160021937,
            "min": 0.0,
            "n_null": 0,
            "rows": 44079,
            "split": "train",
            "std": 0.297191847601921
          },
          {
            "feature": "normalized_range_20",
            "max": 0.6456419808048206,
            "mean": 0.09857059558187131,
            "min": 0.01695126173364293,
            "n_null": 0,
            "rows": 44079,
            "split": "train",
            "std": 0.05930858718109879
          }
        ]
      },
      "redundancy": {
        "duplicates": [],
        "high_correlation_pairs": [
          {
            "feature_a": "dv_med_10",
            "feature_b": "dv_med_30",
            "pearson": 0.979203,
            "spearman": 0.967756
          },
          {
            "feature_a": "vol_60",
            "feature_b": "vol_90",
            "pearson": 0.960375,
            "spearman": 0.956033
          }
        ],
        "pearson": {
          "features": [
            "ret_5",
            "ret_60",
            "ret_120",
            "sma_ratio_10_30",
            "sma_ratio_20_50",
            "price_distance_200ma",
            "vol_60",
            "vol_90",
            "vol_ratio_10_30",
            "dv_med_10",
            "dv_med_30",
            "vol_zscore_20",
            "high_low_10_pos",
            "high_low_30_pos",
            "normalized_range_20"
          ],
          "matrix": "[[ 1.00000000e+00  2.52096604e-01  1.74664628e-01  1.49156524e-01\n   7.63588536e-02  2.49274396e-01  5.54447522e-02  5.26792618e-02\n   1.04099102e-04 -1.74689855e-02 -2.14060112e-02 -2.58365850e-03\n   6.83288079e-01  5.13988556e-01 -1.27682601e-02]\n [ 2.52096604e-01  1.00000000e+00  6.92495125e-01  6.10015824e-01\n   7.94686895e-01  8.22080640e-01  1.27644712e-01  1.60211854e-01\n  -3.76523496e-02  4.00247220e-02  1.42253235e-02  3.19620567e-02\n   2.64860623e-01  4.59532723e-01  3.31923627e-02]\n [ 1.74664628e-01  6.92495125e-01  1.00000000e+00  4.11703185e-01\n   5.30706925e-01  9.17163566e-01  1.85548235e-01  2.12389578e-01\n  -2.58358587e-02  9.91475377e-02  8.25909362e-02  1.82659131e-02\n   1.78674059e-01  2.91441270e-01  1.00785489e-01]\n [ 1.49156524e-01  6.10015824e-01  4.11703185e-01  1.00000000e+00\n   7.32405726e-01  5.53055575e-01  5.11186659e-02  6.62814054e-02\n  -1.70030478e-01 -1.14865870e-02 -3.48661508e-02  3.90219673e-02\n   2.57580252e-01  6.56200027e-01 -6.18088964e-02]\n [ 7.63588536e-02  7.94686895e-01  5.30706925e-01  7.32405726e-01\n   1.00000000e+00  6.75351089e-01  3.41271758e-02  6.99678103e-02\n  -4.99431671e-02  1.59210476e-02 -1.38677571e-02  4.55640679e-02\n   1.12494774e-01  4.63981972e-01 -6.37104650e-02]\n [ 2.49274396e-01  8.22080640e-01  9.17163566e-01  5.53055575e-01\n   6.75351089e-01  1.00000000e+00  1.41677880e-01  1.67246470e-01\n  -3.14672004e-02  1.02487195e-01  7.94358242e-02  3.40869181e-02\n   2.61699209e-01  4.22750520e-01  6.23316404e-02]\n [ 5.54447522e-02  1.27644712e-01  1.85548235e-01  5.11186659e-02\n   3.41271758e-02  1.41677880e-01  1.00000000e+00  9.60375168e-01\n  -3.63073298e-02  8.93075540e-02  8.62306635e-02 -2.42133202e-02\n  -2.68589324e-03 -4.16326390e-02  7.52052237e-01]\n [ 5.26792618e-02  1.60211854e-01  2.12389578e-01  6.62814054e-02\n   6.99678103e-02  1.67246470e-01  9.60375168e-01  1.00000000e+00\n  -3.69674712e-02  7.55136463e-02  7.40179676e-02 -2.28947188e-02\n  -3.31338224e-03 -3.32814823e-02  7.19228741e-01]\n [ 1.04099102e-04 -3.76523496e-02 -2.58358587e-02 -1.70030478e-01\n  -4.99431671e-02 -3.14672004e-02 -3.63073298e-02 -3.69674712e-02\n   1.00000000e+00  9.28456089e-02 -7.19848122e-03  1.83181968e-01\n  -6.13275049e-02 -1.66067798e-01  1.09654996e-01]\n [-1.74689855e-02  4.00247220e-02  9.91475377e-02 -1.14865870e-02\n   1.59210476e-02  1.02487195e-01  8.93075540e-02  7.55136463e-02\n   9.28456089e-02  1.00000000e+00  9.79203494e-01  1.68181777e-02\n  -1.23633926e-03 -2.19695908e-02  1.62633710e-01]\n [-2.14060112e-02  1.42253235e-02  8.25909362e-02 -3.48661508e-02\n  -1.38677571e-02  7.94358242e-02  8.62306635e-02  7.40179676e-02\n  -7.19848122e-03  9.79203494e-01  1.00000000e+00 -1.94091926e-02\n   2.20824736e-03 -1.87617572e-02  1.13454683e-01]\n [-2.58365850e-03  3.19620567e-02  1.82659131e-02  3.90219673e-02\n   4.55640679e-02  3.40869181e-02 -2.42133202e-02 -2.28947188e-02\n   1.83181968e-01  1.68181777e-02 -1.94091926e-02  1.00000000e+00\n  -3.40305542e-02 -9.74360541e-03  1.17822536e-02]\n [ 6.83288079e-01  2.64860623e-01  1.78674059e-01  2.57580252e-01\n   1.12494774e-01  2.61699209e-01 -2.68589324e-03 -3.31338224e-03\n  -6.13275049e-02 -1.23633926e-03  2.20824736e-03 -3.40305542e-02\n   1.00000000e+00  7.50005376e-01 -4.42363167e-02]\n [ 5.13988556e-01  4.59532723e-01  2.91441270e-01  6.56200027e-01\n   4.63981972e-01  4.22750520e-01 -4.16326390e-02 -3.32814823e-02\n  -1.66067798e-01 -2.19695908e-02 -1.87617572e-02 -9.74360541e-03\n   7.50005376e-01  1.00000000e+00 -1.29210341e-01]\n [-1.27682601e-02  3.31923627e-02  1.00785489e-01 -6.18088964e-02\n  -6.37104650e-02  6.23316404e-02  7.52052237e-01  7.19228741e-01\n   1.09654996e-01  1.62633710e-01  1.13454683e-01  1.17822536e-02\n  -4.42363167e-02 -1.29210341e-01  1.00000000e+00]]",
          "method": "pearson",
          "pairs": [
            {
              "correlation": 0.979203,
              "feature_a": "dv_med_10",
              "feature_b": "dv_med_30"
            },
            {
              "correlation": 0.960375,
              "feature_a": "vol_60",
              "feature_b": "vol_90"
            },
            {
              "correlation": 0.917164,
              "feature_a": "ret_120",
              "feature_b": "price_distance_200ma"
            },
            {
              "correlation": 0.822081,
              "feature_a": "ret_60",
              "feature_b": "price_distance_200ma"
            },
            {
              "correlation": 0.794687,
              "feature_a": "ret_60",
              "feature_b": "sma_ratio_20_50"
            },
            {
              "correlation": 0.752052,
              "feature_a": "vol_60",
              "feature_b": "normalized_range_20"
            },
            {
              "correlation": 0.750005,
              "feature_a": "high_low_10_pos",
              "feature_b": "high_low_30_pos"
            },
            {
              "correlation": 0.732406,
              "feature_a": "sma_ratio_10_30",
              "feature_b": "sma_ratio_20_50"
            },
            {
              "correlation": 0.719229,
              "feature_a": "vol_90",
              "feature_b": "normalized_range_20"
            },
            {
              "correlation": 0.692495,
              "feature_a": "ret_60",
              "feature_b": "ret_120"
            },
            {
              "correlation": 0.683288,
              "feature_a": "ret_5",
              "feature_b": "high_low_10_pos"
            },
            {
              "correlation": 0.675351,
              "feature_a": "sma_ratio_20_50",
              "feature_b": "price_distance_200ma"
            },
            {
              "correlation": 0.6562,
              "feature_a": "sma_ratio_10_30",
              "feature_b": "high_low_30_pos"
            },
            {
              "correlation": 0.610016,
              "feature_a": "ret_60",
              "feature_b": "sma_ratio_10_30"
            },
            {
              "correlation": 0.553056,
              "feature_a": "sma_ratio_10_30",
              "feature_b": "price_distance_200ma"
            },
            {
              "correlation": 0.530707,
              "feature_a": "ret_120",
              "feature_b": "sma_ratio_20_50"
            },
            {
              "correlation": 0.513989,
              "feature_a": "ret_5",
              "feature_b": "high_low_30_pos"
            },
            {
              "correlation": 0.463982,
              "feature_a": "sma_ratio_20_50",
              "feature_b": "high_low_30_pos"
            },
            {
              "correlation": 0.459533,
              "feature_a": "ret_60",
              "feature_b": "high_low_30_pos"
            },
            {
              "correlation": 0.422751,
              "feature_a": "price_distance_200ma",
              "feature_b": "high_low_30_pos"
            },
            {
              "correlation": 0.411703,
              "feature_a": "ret_120",
              "feature_b": "sma_ratio_10_30"
            },
            {
              "correlation": 0.291441,
              "feature_a": "ret_120",
              "feature_b": "high_low_30_pos"
            },
            {
              "correlation": 0.264861,
              "feature_a": "ret_60",
              "feature_b": "high_low_10_pos"
            },
            {
              "correlation": 0.261699,
              "feature_a": "price_distance_200ma",
              "feature_b": "high_low_10_pos"
            },
            {
              "correlation": 0.25758,
              "feature_a": "sma_ratio_10_30",
              "feature_b": "high_low_10_pos"
            },
            {
              "correlation": 0.252097,
              "feature_a": "ret_5",
              "feature_b": "ret_60"
            },
            {
              "correlation": 0.249274,
              "feature_a": "ret_5",
              "feature_b": "price_distance_200ma"
            },
            {
              "correlation": 0.21239,
              "feature_a": "ret_120",
              "feature_b": "vol_90"
            },
            {
              "correlation": 0.185548,
              "feature_a": "ret_120",
              "feature_b": "vol_60"
            },
            {
              "correlation": 0.183182,
              "feature_a": "vol_ratio_10_30",
              "feature_b": "vol_zscore_20"
            },
            {
              "correlation": 0.178674,
              "feature_a": "ret_120",
              "feature_b": "high_low_10_pos"
            },
            {
              "correlation": 0.174665,
              "feature_a": "ret_5",
              "feature_b": "ret_120"
            },
            {
              "correlation": -0.17003,
              "feature_a": "sma_ratio_10_30",
              "feature_b": "vol_ratio_10_30"
            },
            {
              "correlation": 0.167246,
              "feature_a": "price_distance_200ma",
              "feature_b": "vol_90"
            },
            {
              "correlation": -0.166068,
              "feature_a": "vol_ratio_10_30",
              "feature_b": "high_low_30_pos"
            },
            {
              "correlation": 0.162634,
              "feature_a": "dv_med_10",
              "feature_b": "normalized_range_20"
            },
            {
              "correlation": 0.160212,
              "feature_a": "ret_60",
              "feature_b": "vol_90"
            },
            {
              "correlation": 0.149157,
              "feature_a": "ret_5",
              "feature_b": "sma_ratio_10_30"
            },
            {
              "correlation": 0.141678,
              "feature_a": "price_distance_200ma",
              "feature_b": "vol_60"
            },
            {
              "correlation": -0.12921,
              "feature_a": "high_low_30_pos",
              "feature_b": "normalized_range_20"
            },
            {
              "correlation": 0.127645,
              "feature_a": "ret_60",
              "feature_b": "vol_60"
            },
            {
              "correlation": 0.113455,
              "feature_a": "dv_med_30",
              "feature_b": "normalized_range_20"
            },
            {
              "correlation": 0.112495,
              "feature_a": "sma_ratio_20_50",
              "feature_b": "high_low_10_pos"
            },
            {
              "correlation": 0.109655,
              "feature_a": "vol_ratio_10_30",
              "feature_b": "normalized_range_20"
            },
            {
              "correlation": 0.102487,
              "feature_a": "price_distance_200ma",
              "feature_b": "dv_med_10"
            },
            {
              "correlation": 0.100785,
              "feature_a": "ret_120",
              "feature_b": "normalized_range_20"
            },
            {
              "correlation": 0.099148,
              "feature_a": "ret_120",
              "feature_b": "dv_med_10"
            },
            {
              "correlation": 0.092846,
              "feature_a": "vol_ratio_10_30",
              "feature_b": "dv_med_10"
            },
            {
              "correlation": 0.089308,
              "feature_a": "vol_60",
              "feature_b": "dv_med_10"
            },
            {
              "correlation": 0.086231,
              "feature_a": "vol_60",
              "feature_b": "dv_med_30"
            },
            {
              "correlation": 0.082591,
              "feature_a": "ret_120",
              "feature_b": "dv_med_30"
            },
            {
              "correlation": 0.079436,
              "feature_a": "price_distance_200ma",
              "feature_b": "dv_med_30"
            },
            {
              "correlation": 0.076359,
              "feature_a": "ret_5",
              "feature_b": "sma_ratio_20_50"
            },
            {
              "correlation": 0.075514,
              "feature_a": "vol_90",
              "feature_b": "dv_med_10"
            },
            {
              "correlation": 0.074018,
              "feature_a": "vol_90",
              "feature_b": "dv_med_30"
            },
            {
              "correlation": 0.069968,
              "feature_a": "sma_ratio_20_50",
              "feature_b": "vol_90"
            },
            {
              "correlation": 0.066281,
              "feature_a": "sma_ratio_10_30",
              "feature_b": "vol_90"
            },
            {
              "correlation": -0.06371,
              "feature_a": "sma_ratio_20_50",
              "feature_b": "normalized_range_20"
            },
            {
              "correlation": 0.062332,
              "feature_a": "price_distance_200ma",
              "feature_b": "normalized_range_20"
            },
            {
              "correlation": -0.061809,
              "feature_a": "sma_ratio_10_30",
              "feature_b": "normalized_range_20"
            },
            {
              "correlation": -0.061328,
              "feature_a": "vol_ratio_10_30",
              "feature_b": "high_low_10_pos"
            },
            {
              "correlation": 0.055445,
              "feature_a": "ret_5",
              "feature_b": "vol_60"
            },
            {
              "correlation": 0.052679,
              "feature_a": "ret_5",
              "feature_b": "vol_90"
            },
            {
              "correlation": 0.051119,
              "feature_a": "sma_ratio_10_30",
              "feature_b": "vol_60"
            },
            {
              "correlation": -0.049943,
              "feature_a": "sma_ratio_20_50",
              "feature_b": "vol_ratio_10_30"
            },
            {
              "correlation": 0.045564,
              "feature_a": "sma_ratio_20_50",
              "feature_b": "vol_zscore_20"
            },
            {
              "correlation": -0.044236,
              "feature_a": "high_low_10_pos",
              "feature_b": "normalized_range_20"
            },
            {
              "correlation": -0.041633,
              "feature_a": "vol_60",
              "feature_b": "high_low_30_pos"
            },
            {
              "correlation": 0.040025,
              "feature_a": "ret_60",
              "feature_b": "dv_med_10"
            },
            {
              "correlation": 0.039022,
              "feature_a": "sma_ratio_10_30",
              "feature_b": "vol_zscore_20"
            },
            {
              "correlation": -0.037652,
              "feature_a": "ret_60",
              "feature_b": "vol_ratio_10_30"
            },
            {
              "correlation": -0.036967,
              "feature_a": "vol_90",
              "feature_b": "vol_ratio_10_30"
            },
            {
              "correlation": -0.036307,
              "feature_a": "vol_60",
              "feature_b": "vol_ratio_10_30"
            },
            {
              "correlation": -0.034866,
              "feature_a": "sma_ratio_10_30",
              "feature_b": "dv_med_30"
            },
            {
              "correlation": 0.034127,
              "feature_a": "sma_ratio_20_50",
              "feature_b": "vol_60"
            },
            {
              "correlation": 0.034087,
              "feature_a": "price_distance_200ma",
              "feature_b": "vol_zscore_20"
            },
            {
              "correlation": -0.034031,
              "feature_a": "vol_zscore_20",
              "feature_b": "high_low_10_pos"
            },
            {
              "correlation": -0.033281,
              "feature_a": "vol_90",
              "feature_b": "high_low_30_pos"
            },
            {
              "correlation": 0.033192,
              "feature_a": "ret_60",
              "feature_b": "normalized_range_20"
            },
            {
              "correlation": 0.031962,
              "feature_a": "ret_60",
              "feature_b": "vol_zscore_20"
            },
            {
              "correlation": -0.031467,
              "feature_a": "price_distance_200ma",
              "feature_b": "vol_ratio_10_30"
            },
            {
              "correlation": -0.025836,
              "feature_a": "ret_120",
              "feature_b": "vol_ratio_10_30"
            },
            {
              "correlation": -0.024213,
              "feature_a": "vol_60",
              "feature_b": "vol_zscore_20"
            },
            {
              "correlation": -0.022895,
              "feature_a": "vol_90",
              "feature_b": "vol_zscore_20"
            },
            {
              "correlation": -0.02197,
              "feature_a": "dv_med_10",
              "feature_b": "high_low_30_pos"
            },
            {
              "correlation": -0.021406,
              "feature_a": "ret_5",
              "feature_b": "dv_med_30"
            },
            {
              "correlation": -0.019409,
              "feature_a": "dv_med_30",
              "feature_b": "vol_zscore_20"
            },
            {
              "correlation": -0.018762,
              "feature_a": "dv_med_30",
              "feature_b": "high_low_30_pos"
            },
            {
              "correlation": 0.018266,
              "feature_a": "ret_120",
              "feature_b": "vol_zscore_20"
            },
            {
              "correlation": -0.017469,
              "feature_a": "ret_5",
              "feature_b": "dv_med_10"
            },
            {
              "correlation": 0.016818,
              "feature_a": "dv_med_10",
              "feature_b": "vol_zscore_20"
            },
            {
              "correlation": 0.015921,
              "feature_a": "sma_ratio_20_50",
              "feature_b": "dv_med_10"
            },
            {
              "correlation": 0.014225,
              "feature_a": "ret_60",
              "feature_b": "dv_med_30"
            },
            {
              "correlation": -0.013868,
              "feature_a": "sma_ratio_20_50",
              "feature_b": "dv_med_30"
            },
            {
              "correlation": -0.012768,
              "feature_a": "ret_5",
              "feature_b": "normalized_range_20"
            },
            {
              "correlation": 0.011782,
              "feature_a": "vol_zscore_20",
              "feature_b": "normalized_range_20"
            },
            {
              "correlation": -0.011487,
              "feature_a": "sma_ratio_10_30",
              "feature_b": "dv_med_10"
            },
            {
              "correlation": -0.009744,
              "feature_a": "vol_zscore_20",
              "feature_b": "high_low_30_pos"
            },
            {
              "correlation": -0.007198,
              "feature_a": "vol_ratio_10_30",
              "feature_b": "dv_med_30"
            },
            {
              "correlation": -0.003313,
              "feature_a": "vol_90",
              "feature_b": "high_low_10_pos"
            },
            {
              "correlation": -0.002686,
              "feature_a": "vol_60",
              "feature_b": "high_low_10_pos"
            },
            {
              "correlation": -0.002584,
              "feature_a": "ret_5",
              "feature_b": "vol_zscore_20"
            },
            {
              "correlation": 0.002208,
              "feature_a": "dv_med_30",
              "feature_b": "high_low_10_pos"
            },
            {
              "correlation": -0.001236,
              "feature_a": "dv_med_10",
              "feature_b": "high_low_10_pos"
            },
            {
              "correlation": 0.000104,
              "feature_a": "ret_5",
              "feature_b": "vol_ratio_10_30"
            }
          ],
          "rows_used": 44079
        },
        "spearman": {
          "features": [
            "ret_5",
            "ret_60",
            "ret_120",
            "sma_ratio_10_30",
            "sma_ratio_20_50",
            "price_distance_200ma",
            "vol_60",
            "vol_90",
            "vol_ratio_10_30",
            "dv_med_10",
            "dv_med_30",
            "vol_zscore_20",
            "high_low_10_pos",
            "high_low_30_pos",
            "normalized_range_20"
          ],
          "matrix": "[[ 1.00000000e+00  2.24684091e-01  1.67326869e-01  1.21400778e-01\n   4.58516721e-02  2.24650056e-01  3.85685725e-02  4.03005067e-02\n  -8.59644812e-03 -2.54843742e-03 -3.35931443e-03 -1.96283997e-02\n   7.82661758e-01  5.67382457e-01  1.87508934e-02]\n [ 2.24684091e-01  1.00000000e+00  6.35542824e-01  5.82990152e-01\n   7.75713387e-01  7.63891936e-01 -1.96023494e-02  3.41439863e-02\n  -3.84946846e-02  1.02952958e-02 -2.21483844e-03  3.30611889e-02\n   3.06635525e-01  5.48194626e-01 -5.13460289e-02]\n [ 1.67326869e-01  6.35542824e-01  1.00000000e+00  3.94886705e-01\n   5.07029100e-01  9.14063510e-01  3.10418975e-02  4.36572506e-02\n  -2.56690480e-02  7.62227717e-02  6.46189928e-02  1.68135018e-02\n   2.25269925e-01  3.73622030e-01  1.65454238e-02]\n [ 1.21400778e-01  5.82990152e-01  3.94886705e-01  1.00000000e+00\n   6.93717555e-01  5.18114643e-01  1.68703650e-02  4.03753493e-02\n  -1.89357452e-01 -1.87466683e-02 -2.16744862e-02  4.07763287e-02\n   2.79216121e-01  7.41816852e-01  4.00605723e-03]\n [ 4.58516721e-02  7.75713387e-01  5.07029100e-01  6.93717555e-01\n   1.00000000e+00  6.32081325e-01 -1.07006479e-02  3.13499354e-02\n  -3.09597004e-02 -4.00829087e-03 -2.10518956e-02  5.09819782e-02\n   1.17322780e-01  5.04862528e-01 -6.34322392e-02]\n [ 2.24650056e-01  7.63891936e-01  9.14063510e-01  5.18114643e-01\n   6.32081325e-01  1.00000000e+00  2.91802200e-02  5.51069770e-02\n  -3.46860501e-02  7.51653658e-02  6.23808786e-02  3.03867697e-02\n   3.01634242e-01  4.96990571e-01  6.59710703e-03]\n [ 3.85685725e-02 -1.96023494e-02  3.10418975e-02  1.68703650e-02\n  -1.07006479e-02  2.91802200e-02  1.00000000e+00  9.56032551e-01\n  -4.68215836e-02  2.69676780e-01  2.81556659e-01 -4.14163955e-02\n   4.67350737e-04 -4.17036022e-02  7.35860200e-01]\n [ 4.03005067e-02  3.41439863e-02  4.36572506e-02  4.03753493e-02\n   3.13499354e-02  5.51069770e-02  9.56032551e-01  1.00000000e+00\n  -4.29482381e-02  2.63828925e-01  2.76182210e-01 -3.91445207e-02\n   3.48191097e-03 -2.68349038e-02  7.03066646e-01]\n [-8.59644812e-03 -3.84946846e-02 -2.56690480e-02 -1.89357452e-01\n  -3.09597004e-02 -3.46860501e-02 -4.68215836e-02 -4.29482381e-02\n   1.00000000e+00  9.83879895e-02 -1.38006393e-02  1.84097387e-01\n  -6.13585453e-02 -1.66850481e-01  8.35570411e-02]\n [-2.54843742e-03  1.02952958e-02  7.62227717e-02 -1.87466683e-02\n  -4.00829087e-03  7.51653658e-02  2.69676780e-01  2.63828925e-01\n   9.83879895e-02  1.00000000e+00  9.67755839e-01  2.21394465e-02\n  -4.62522758e-03 -3.57062981e-02  2.98436933e-01]\n [-3.35931443e-03 -2.21483844e-03  6.46189928e-02 -2.16744862e-02\n  -2.10518956e-02  6.23808786e-02  2.81556659e-01  2.76182210e-01\n  -1.38006393e-02  9.67755839e-01  1.00000000e+00 -3.02970371e-02\n  -2.00319032e-03 -3.13272789e-02  2.58547362e-01]\n [-1.96283997e-02  3.30611889e-02  1.68135018e-02  4.07763287e-02\n   5.09819782e-02  3.03867697e-02 -4.14163955e-02 -3.91445207e-02\n   1.84097387e-01  2.21394465e-02 -3.02970371e-02  1.00000000e+00\n  -2.59247175e-02  1.04069392e-02 -3.27933530e-03]\n [ 7.82661758e-01  3.06635525e-01  2.25269925e-01  2.79216121e-01\n   1.17322780e-01  3.01634242e-01  4.67350737e-04  3.48191097e-03\n  -6.13585453e-02 -4.62522758e-03 -2.00319032e-03 -2.59247175e-02\n   1.00000000e+00  7.63182782e-01 -9.10199105e-03]\n [ 5.67382457e-01  5.48194626e-01  3.73622030e-01  7.41816852e-01\n   5.04862528e-01  4.96990571e-01 -4.17036022e-02 -2.68349038e-02\n  -1.66850481e-01 -3.57062981e-02 -3.13272789e-02  1.04069392e-02\n   7.63182782e-01  1.00000000e+00 -7.82780499e-02]\n [ 1.87508934e-02 -5.13460289e-02  1.65454238e-02  4.00605723e-03\n  -6.34322392e-02  6.59710703e-03  7.35860200e-01  7.03066646e-01\n   8.35570411e-02  2.98436933e-01  2.58547362e-01 -3.27933530e-03\n  -9.10199105e-03 -7.82780499e-02  1.00000000e+00]]",
          "method": "spearman",
          "pairs": [
            {
              "correlation": 0.967756,
              "feature_a": "dv_med_10",
              "feature_b": "dv_med_30"
            },
            {
              "correlation": 0.956033,
              "feature_a": "vol_60",
              "feature_b": "vol_90"
            },
            {
              "correlation": 0.914064,
              "feature_a": "ret_120",
              "feature_b": "price_distance_200ma"
            },
            {
              "correlation": 0.782662,
              "feature_a": "ret_5",
              "feature_b": "high_low_10_pos"
            },
            {
              "correlation": 0.775713,
              "feature_a": "ret_60",
              "feature_b": "sma_ratio_20_50"
            },
            {
              "correlation": 0.763892,
              "feature_a": "ret_60",
              "feature_b": "price_distance_200ma"
            },
            {
              "correlation": 0.763183,
              "feature_a": "high_low_10_pos",
              "feature_b": "high_low_30_pos"
            },
            {
              "correlation": 0.741817,
              "feature_a": "sma_ratio_10_30",
              "feature_b": "high_low_30_pos"
            },
            {
              "correlation": 0.73586,
              "feature_a": "vol_60",
              "feature_b": "normalized_range_20"
            },
            {
              "correlation": 0.703067,
              "feature_a": "vol_90",
              "feature_b": "normalized_range_20"
            },
            {
              "correlation": 0.693718,
              "feature_a": "sma_ratio_10_30",
              "feature_b": "sma_ratio_20_50"
            },
            {
              "correlation": 0.635543,
              "feature_a": "ret_60",
              "feature_b": "ret_120"
            },
            {
              "correlation": 0.632081,
              "feature_a": "sma_ratio_20_50",
              "feature_b": "price_distance_200ma"
            },
            {
              "correlation": 0.58299,
              "feature_a": "ret_60",
              "feature_b": "sma_ratio_10_30"
            },
            {
              "correlation": 0.567382,
              "feature_a": "ret_5",
              "feature_b": "high_low_30_pos"
            },
            {
              "correlation": 0.548195,
              "feature_a": "ret_60",
              "feature_b": "high_low_30_pos"
            },
            {
              "correlation": 0.518115,
              "feature_a": "sma_ratio_10_30",
              "feature_b": "price_distance_200ma"
            },
            {
              "correlation": 0.507029,
              "feature_a": "ret_120",
              "feature_b": "sma_ratio_20_50"
            },
            {
              "correlation": 0.504863,
              "feature_a": "sma_ratio_20_50",
              "feature_b": "high_low_30_pos"
            },
            {
              "correlation": 0.496991,
              "feature_a": "price_distance_200ma",
              "feature_b": "high_low_30_pos"
            },
            {
              "correlation": 0.394887,
              "feature_a": "ret_120",
              "feature_b": "sma_ratio_10_30"
            },
            {
              "correlation": 0.373622,
              "feature_a": "ret_120",
              "feature_b": "high_low_30_pos"
            },
            {
              "correlation": 0.306636,
              "feature_a": "ret_60",
              "feature_b": "high_low_10_pos"
            },
            {
              "correlation": 0.301634,
              "feature_a": "price_distance_200ma",
              "feature_b": "high_low_10_pos"
            },
            {
              "correlation": 0.298437,
              "feature_a": "dv_med_10",
              "feature_b": "normalized_range_20"
            },
            {
              "correlation": 0.281557,
              "feature_a": "vol_60",
              "feature_b": "dv_med_30"
            },
            {
              "correlation": 0.279216,
              "feature_a": "sma_ratio_10_30",
              "feature_b": "high_low_10_pos"
            },
            {
              "correlation": 0.276182,
              "feature_a": "vol_90",
              "feature_b": "dv_med_30"
            },
            {
              "correlation": 0.269677,
              "feature_a": "vol_60",
              "feature_b": "dv_med_10"
            },
            {
              "correlation": 0.263829,
              "feature_a": "vol_90",
              "feature_b": "dv_med_10"
            },
            {
              "correlation": 0.258547,
              "feature_a": "dv_med_30",
              "feature_b": "normalized_range_20"
            },
            {
              "correlation": 0.22527,
              "feature_a": "ret_120",
              "feature_b": "high_low_10_pos"
            },
            {
              "correlation": 0.224684,
              "feature_a": "ret_5",
              "feature_b": "ret_60"
            },
            {
              "correlation": 0.22465,
              "feature_a": "ret_5",
              "feature_b": "price_distance_200ma"
            },
            {
              "correlation": -0.189357,
              "feature_a": "sma_ratio_10_30",
              "feature_b": "vol_ratio_10_30"
            },
            {
              "correlation": 0.184097,
              "feature_a": "vol_ratio_10_30",
              "feature_b": "vol_zscore_20"
            },
            {
              "correlation": 0.167327,
              "feature_a": "ret_5",
              "feature_b": "ret_120"
            },
            {
              "correlation": -0.16685,
              "feature_a": "vol_ratio_10_30",
              "feature_b": "high_low_30_pos"
            },
            {
              "correlation": 0.121401,
              "feature_a": "ret_5",
              "feature_b": "sma_ratio_10_30"
            },
            {
              "correlation": 0.117323,
              "feature_a": "sma_ratio_20_50",
              "feature_b": "high_low_10_pos"
            },
            {
              "correlation": 0.098388,
              "feature_a": "vol_ratio_10_30",
              "feature_b": "dv_med_10"
            },
            {
              "correlation": 0.083557,
              "feature_a": "vol_ratio_10_30",
              "feature_b": "normalized_range_20"
            },
            {
              "correlation": -0.078278,
              "feature_a": "high_low_30_pos",
              "feature_b": "normalized_range_20"
            },
            {
              "correlation": 0.076223,
              "feature_a": "ret_120",
              "feature_b": "dv_med_10"
            },
            {
              "correlation": 0.075165,
              "feature_a": "price_distance_200ma",
              "feature_b": "dv_med_10"
            },
            {
              "correlation": 0.064619,
              "feature_a": "ret_120",
              "feature_b": "dv_med_30"
            },
            {
              "correlation": -0.063432,
              "feature_a": "sma_ratio_20_50",
              "feature_b": "normalized_range_20"
            },
            {
              "correlation": 0.062381,
              "feature_a": "price_distance_200ma",
              "feature_b": "dv_med_30"
            },
            {
              "correlation": -0.061359,
              "feature_a": "vol_ratio_10_30",
              "feature_b": "high_low_10_pos"
            },
            {
              "correlation": 0.055107,
              "feature_a": "price_distance_200ma",
              "feature_b": "vol_90"
            },
            {
              "correlation": -0.051346,
              "feature_a": "ret_60",
              "feature_b": "normalized_range_20"
            },
            {
              "correlation": 0.050982,
              "feature_a": "sma_ratio_20_50",
              "feature_b": "vol_zscore_20"
            },
            {
              "correlation": -0.046822,
              "feature_a": "vol_60",
              "feature_b": "vol_ratio_10_30"
            },
            {
              "correlation": 0.045852,
              "feature_a": "ret_5",
              "feature_b": "sma_ratio_20_50"
            },
            {
              "correlation": 0.043657,
              "feature_a": "ret_120",
              "feature_b": "vol_90"
            },
            {
              "correlation": -0.042948,
              "feature_a": "vol_90",
              "feature_b": "vol_ratio_10_30"
            },
            {
              "correlation": -0.041704,
              "feature_a": "vol_60",
              "feature_b": "high_low_30_pos"
            },
            {
              "correlation": -0.041416,
              "feature_a": "vol_60",
              "feature_b": "vol_zscore_20"
            },
            {
              "correlation": 0.040776,
              "feature_a": "sma_ratio_10_30",
              "feature_b": "vol_zscore_20"
            },
            {
              "correlation": 0.040375,
              "feature_a": "sma_ratio_10_30",
              "feature_b": "vol_90"
            },
            {
              "correlation": 0.040301,
              "feature_a": "ret_5",
              "feature_b": "vol_90"
            },
            {
              "correlation": -0.039145,
              "feature_a": "vol_90",
              "feature_b": "vol_zscore_20"
            },
            {
              "correlation": 0.038569,
              "feature_a": "ret_5",
              "feature_b": "vol_60"
            },
            {
              "correlation": -0.038495,
              "feature_a": "ret_60",
              "feature_b": "vol_ratio_10_30"
            },
            {
              "correlation": -0.035706,
              "feature_a": "dv_med_10",
              "feature_b": "high_low_30_pos"
            },
            {
              "correlation": -0.034686,
              "feature_a": "price_distance_200ma",
              "feature_b": "vol_ratio_10_30"
            },
            {
              "correlation": 0.034144,
              "feature_a": "ret_60",
              "feature_b": "vol_90"
            },
            {
              "correlation": 0.033061,
              "feature_a": "ret_60",
              "feature_b": "vol_zscore_20"
            },
            {
              "correlation": 0.03135,
              "feature_a": "sma_ratio_20_50",
              "feature_b": "vol_90"
            },
            {
              "correlation": -0.031327,
              "feature_a": "dv_med_30",
              "feature_b": "high_low_30_pos"
            },
            {
              "correlation": 0.031042,
              "feature_a": "ret_120",
              "feature_b": "vol_60"
            },
            {
              "correlation": -0.03096,
              "feature_a": "sma_ratio_20_50",
              "feature_b": "vol_ratio_10_30"
            },
            {
              "correlation": 0.030387,
              "feature_a": "price_distance_200ma",
              "feature_b": "vol_zscore_20"
            },
            {
              "correlation": -0.030297,
              "feature_a": "dv_med_30",
              "feature_b": "vol_zscore_20"
            },
            {
              "correlation": 0.02918,
              "feature_a": "price_distance_200ma",
              "feature_b": "vol_60"
            },
            {
              "correlation": -0.026835,
              "feature_a": "vol_90",
              "feature_b": "high_low_30_pos"
            },
            {
              "correlation": -0.025925,
              "feature_a": "vol_zscore_20",
              "feature_b": "high_low_10_pos"
            },
            {
              "correlation": -0.025669,
              "feature_a": "ret_120",
              "feature_b": "vol_ratio_10_30"
            },
            {
              "correlation": 0.022139,
              "feature_a": "dv_med_10",
              "feature_b": "vol_zscore_20"
            },
            {
              "correlation": -0.021674,
              "feature_a": "sma_ratio_10_30",
              "feature_b": "dv_med_30"
            },
            {
              "correlation": -0.021052,
              "feature_a": "sma_ratio_20_50",
              "feature_b": "dv_med_30"
            },
            {
              "correlation": -0.019628,
              "feature_a": "ret_5",
              "feature_b": "vol_zscore_20"
            },
            {
              "correlation": -0.019602,
              "feature_a": "ret_60",
              "feature_b": "vol_60"
            },
            {
              "correlation": 0.018751,
              "feature_a": "ret_5",
              "feature_b": "normalized_range_20"
            },
            {
              "correlation": -0.018747,
              "feature_a": "sma_ratio_10_30",
              "feature_b": "dv_med_10"
            },
            {
              "correlation": 0.01687,
              "feature_a": "sma_ratio_10_30",
              "feature_b": "vol_60"
            },
            {
              "correlation": 0.016814,
              "feature_a": "ret_120",
              "feature_b": "vol_zscore_20"
            },
            {
              "correlation": 0.016545,
              "feature_a": "ret_120",
              "feature_b": "normalized_range_20"
            },
            {
              "correlation": -0.013801,
              "feature_a": "vol_ratio_10_30",
              "feature_b": "dv_med_30"
            },
            {
              "correlation": -0.010701,
              "feature_a": "sma_ratio_20_50",
              "feature_b": "vol_60"
            },
            {
              "correlation": 0.010407,
              "feature_a": "vol_zscore_20",
              "feature_b": "high_low_30_pos"
            },
            {
              "correlation": 0.010295,
              "feature_a": "ret_60",
              "feature_b": "dv_med_10"
            },
            {
              "correlation": -0.009102,
              "feature_a": "high_low_10_pos",
              "feature_b": "normalized_range_20"
            },
            {
              "correlation": -0.008596,
              "feature_a": "ret_5",
              "feature_b": "vol_ratio_10_30"
            },
            {
              "correlation": 0.006597,
              "feature_a": "price_distance_200ma",
              "feature_b": "normalized_range_20"
            },
            {
              "correlation": -0.004625,
              "feature_a": "dv_med_10",
              "feature_b": "high_low_10_pos"
            },
            {
              "correlation": -0.004008,
              "feature_a": "sma_ratio_20_50",
              "feature_b": "dv_med_10"
            },
            {
              "correlation": 0.004006,
              "feature_a": "sma_ratio_10_30",
              "feature_b": "normalized_range_20"
            },
            {
              "correlation": 0.003482,
              "feature_a": "vol_90",
              "feature_b": "high_low_10_pos"
            },
            {
              "correlation": -0.003359,
              "feature_a": "ret_5",
              "feature_b": "dv_med_30"
            },
            {
              "correlation": -0.003279,
              "feature_a": "vol_zscore_20",
              "feature_b": "normalized_range_20"
            },
            {
              "correlation": -0.002548,
              "feature_a": "ret_5",
              "feature_b": "dv_med_10"
            },
            {
              "correlation": -0.002215,
              "feature_a": "ret_60",
              "feature_b": "dv_med_30"
            },
            {
              "correlation": -0.002003,
              "feature_a": "dv_med_30",
              "feature_b": "high_low_10_pos"
            },
            {
              "correlation": 0.000467,
              "feature_a": "vol_60",
              "feature_b": "high_low_10_pos"
            }
          ],
          "rows_used": 44079
        },
        "train_rows": 44079
      }
    },
    "FS-003": {
      "quality": {
        "features": [
          {
            "feature": "ret_10",
            "frac_most_common_value": 0.002019102066743801,
            "is_constant": false,
            "is_near_constant": false,
            "missing_frac": 0.0,
            "n_null": 0,
            "n_unique": 43881,
            "percentiles": {
              "0.01": -0.11835545835866267,
              "0.05": -0.06594252989284874,
              "0.5": 0.00585072980242507,
              "0.95": 0.07601534118802611,
              "0.99": 0.1375298576254729
            },
            "rows": 44079
          },
          {
            "feature": "ret_20",
            "frac_most_common_value": 0.0013158193243948365,
            "is_constant": false,
            "is_near_constant": false,
            "missing_frac": 0.0,
            "n_null": 0,
            "n_unique": 43928,
            "percentiles": {
              "0.01": -0.15240317392292338,
              "0.05": -0.08873120274062085,
              "0.5": 0.012401198632101096,
              "0.95": 0.11528901539468393,
              "0.99": 0.2046896292729398
            },
            "rows": 44079
          },
          {
            "feature": "ret_30",
            "frac_most_common_value": 0.0010662673835613332,
            "is_constant": false,
            "is_near_constant": false,
            "missing_frac": 0.0,
            "n_null": 0,
            "n_unique": 43955,
            "percentiles": {
              "0.01": -0.18092240277786598,
              "0.05": -0.10457788799683007,
              "0.5": 0.017625267513192044,
              "0.95": 0.1498133462127771,
              "0.99": 0.2714901924913345
            },
            "rows": 44079
          },
          {
            "feature": "sma_ratio_5_30",
            "frac_most_common_value": 2.2686540075773043e-05,
            "is_constant": false,
            "is_near_constant": false,
            "missing_frac": 0.0,
            "n_null": 0,
            "n_unique": 44079,
            "percentiles": {
              "0.01": 0.9015911279132929,
              "0.05": 0.9449325800945056,
              "0.5": 1.0080364008675762,
              "0.95": 1.0670321684615374,
              "0.99": 1.1176318130416567
            },
            "rows": 44079
          },
          {
            "feature": "sma_ratio_15_40",
            "frac_most_common_value": 2.2686540075773043e-05,
            "is_constant": false,
            "is_near_constant": false,
            "missing_frac": 0.0,
            "n_null": 0,
            "n_unique": 44079,
            "percentiles": {
              "0.01": 0.9185877332116237,
              "0.05": 0.9545146971313394,
              "0.5": 1.0079069512696,
              "0.95": 1.058719926017665,
              "0.99": 1.102921823301901
            },
            "rows": 44079
          },
          {
            "feature": "vol_10",
            "frac_most_common_value": 4.5373080151546086e-05,
            "is_constant": false,
            "is_near_constant": false,
            "missing_frac": 0.0,
            "n_null": 0,
            "n_unique": 44076,
            "percentiles": {
              "0.01": 0.003489109281207255,
              "0.05": 0.004967580135171786,
              "0.5": 0.011539906614313221,
              "0.95": 0.03022793857415316,
              "0.99": 0.046713784832393156
            },
            "rows": 44079
          },
          {
            "feature": "vol_30",
            "frac_most_common_value": 4.5373080151546086e-05,
            "is_constant": false,
            "is_near_constant": false,
            "missing_frac": 0.0,
            "n_null": 0,
            "n_unique": 44076,
            "percentiles": {
              "0.01": 0.004913464706145423,
              "0.05": 0.0063065057311843175,
              "0.5": 0.012319361690490842,
              "0.95": 0.029145919792133497,
              "0.99": 0.04170914356389495
            },
            "rows": 44079
          },
          {
            "feature": "log_dv_med_20",
            "frac_most_common_value": 0.0003176115610608226,
            "is_constant": false,
            "is_near_constant": false,
            "missing_frac": 0.0,
            "n_null": 0,
            "n_unique": 20533,
            "percentiles": {
              "0.01": 17.622937173627776,
              "0.05": 19.439202629261363,
              "0.5": 20.490901758434354,
              "0.95": 22.100841994169294,
              "0.99": 22.777388322487468
            },
            "rows": 44079
          },
          {
            "feature": "ret_5",
            "frac_most_common_value": 0.0034029810113659566,
            "is_constant": false,
            "is_near_constant": false,
            "missing_frac": 0.0,
            "n_null": 0,
            "n_unique": 43742,
            "percentiles": {
              "0.01": -0.08519447854909686,
              "0.05": -0.04657101473959926,
              "0.5": 0.0026413562600937635,
              "0.95": 0.050431776131422947,
              "0.99": 0.09011672879234522
            },
            "rows": 44079
          },
          {
            "feature": "ret_60",
            "frac_most_common_value": 0.0007486558225005105,
            "is_constant": false,
            "is_near_constant": false,
            "missing_frac": 0.0,
            "n_null": 0,
            "n_unique": 43998,
            "percentiles": {
              "0.01": -0.22603053979847226,
              "0.05": -0.13145417109363425,
              "0.5": 0.03670873487124049,
              "0.95": 0.2259978661085785,
              "0.99": 0.4485449516843603
            },
            "rows": 44079
          },
          {
            "feature": "ret_120",
            "frac_most_common_value": 0.0003629846412123687,
            "is_constant": false,
            "is_near_constant": false,
            "missing_frac": 0.0,
            "n_null": 0,
            "n_unique": 44018,
            "percentiles": {
              "0.01": -0.28326662573546213,
              "0.05": -0.14591622400797652,
              "0.5": 0.07388087086835426,
              "0.95": 0.36839598537377294,
              "0.99": 0.7658738384982484
            },
            "rows": 44079
          },
          {
            "feature": "sma_ratio_10_30",
            "frac_most_common_value": 2.2686540075773043e-05,
            "is_constant": false,
            "is_near_constant": false,
            "missing_frac": 0.0,
            "n_null": 0,
            "n_unique": 44079,
            "percentiles": {
              "0.01": 0.9204806940984668,
              "0.05": 0.9559253758344521,
              "0.5": 1.0064500154936735,
              "0.95": 1.0535450276290457,
              "0.99": 1.092460644823427
            },
            "rows": 44079
          },
          {
            "feature": "sma_ratio_20_50",
            "frac_most_common_value": 2.2686540075773043e-05,
            "is_constant": false,
            "is_near_constant": false,
            "missing_frac": 0.0,
            "n_null": 0,
            "n_unique": 44079,
            "percentiles": {
              "0.01": 0.9143528387323512,
              "0.05": 0.9520096265817739,
              "0.5": 1.0093712362647946,
              "0.95": 1.0635588305045758,
              "0.99": 1.1128943541289564
            },
            "rows": 44079
          },
          {
            "feature": "price_distance_200ma",
            "frac_most_common_value": 4.5373080151546086e-05,
            "is_constant": false,
            "is_near_constant": false,
            "missing_frac": 0.0,
            "n_null": 0,
            "n_unique": 44078,
            "percentiles": {
              "0.01": -0.2181947725364505,
              "0.05": -0.10517973950299972,
              "0.5": 0.05976649479212117,
              "0.95": 0.25095854781451005,
              "0.99": 0.5067726706958768
            },
            "rows": 44079
          },
          {
            "feature": "vol_60",
            "frac_most_common_value": 4.5373080151546086e-05,
            "is_constant": false,
            "is_near_constant": false,
            "missing_frac": 0.0,
            "n_null": 0,
            "n_unique": 44074,
            "percentiles": {
              "0.01": 0.0055466523308712445,
              "0.05": 0.007007644893428239,
              "0.5": 0.012685875153042179,
              "0.95": 0.028405130251284213,
              "0.99": 0.04244269190019712
            },
            "rows": 44079
          },
          {
            "feature": "vol_90",
            "frac_most_common_value": 4.5373080151546086e-05,
            "is_constant": false,
            "is_near_constant": false,
            "missing_frac": 0.0,
            "n_null": 0,
            "n_unique": 44077,
            "percentiles": {
              "0.01": 0.0060177553547351445,
              "0.05": 0.007316666310178612,
              "0.5": 0.01281876677603952,
              "0.95": 0.028384252536876122,
              "0.99": 0.039462237730868086
            },
            "rows": 44079
          },
          {
            "feature": "vol_ratio_10_30",
            "frac_most_common_value": 2.2686540075773043e-05,
            "is_constant": false,
            "is_near_constant": false,
            "missing_frac": 0.0,
            "n_null": 0,
            "n_unique": 44079,
            "percentiles": {
              "0.01": 0.37031201529048763,
              "0.05": 0.5145880370186117,
              "0.5": 0.9595451145612078,
              "0.95": 1.4414981556396955,
              "0.99": 1.5968279829146168
            },
            "rows": 44079
          },
          {
            "feature": "dv_med_10",
            "frac_most_common_value": 0.00020417886068195738,
            "is_constant": false,
            "is_near_constant": false,
            "missing_frac": 0.0,
            "n_null": 0,
            "n_unique": 25697,
            "percentiles": {
              "0.01": 17.688198206010366,
              "0.05": 19.407266344368413,
              "0.5": 20.503008362880127,
              "0.95": 22.123256355080706,
              "0.99": 22.826972743750584
            },
            "rows": 44079
          },
          {
            "feature": "dv_med_30",
            "frac_most_common_value": 0.0004537308015154609,
            "is_constant": false,
            "is_near_constant": false,
            "missing_frac": 0.0,
            "n_null": 0,
            "n_unique": 18548,
            "percentiles": {
              "0.01": 17.574291661484615,
              "0.05": 19.445723920957043,
              "0.5": 20.48174114793814,
              "0.95": 22.081243555658737,
              "0.99": 22.771109763438513
            },
            "rows": 44079
          },
          {
            "feature": "vol_zscore_20",
            "frac_most_common_value": 2.2686540075773043e-05,
            "is_constant": false,
            "is_near_constant": false,
            "missing_frac": 0.0,
            "n_null": 0,
            "n_unique": 44079,
            "percentiles": {
              "0.01": -1.6819276623681836,
              "0.05": -1.2507613715349037,
              "0.5": -0.2240219197741499,
              "0.95": 2.2098992999615907,
              "0.99": 3.476521717969192
            },
            "rows": 44079
          },
          {
            "feature": "high_low_10_pos",
            "frac_most_common_value": 0.006079992740307176,
            "is_constant": false,
            "is_near_constant": false,
            "missing_frac": 0.0,
            "n_null": 0,
            "n_unique": 41958,
            "percentiles": {
              "0.01": 0.009889234285109048,
              "0.05": 0.06090791660548324,
              "0.5": 0.5974011733099402,
              "0.95": 0.9704271578324056,
              "0.99": 0.9962381581197126
            },
            "rows": 44079
          },
          {
            "feature": "high_low_30_pos",
            "frac_most_common_value": 0.003992831053336056,
            "is_constant": false,
            "is_near_constant": false,
            "missing_frac": 0.0,
            "n_null": 0,
            "n_unique": 42804,
            "percentiles": {
              "0.01": 0.014286236024797828,
              "0.05": 0.07264451505584091,
              "0.5": 0.6460256987912134,
              "0.95": 0.9746168953794586,
              "0.99": 0.9961080112723782
            },
            "rows": 44079
          },
          {
            "feature": "normalized_range_20",
            "frac_most_common_value": 6.805962022731914e-05,
            "is_constant": false,
            "is_near_constant": false,
            "missing_frac": 0.0,
            "n_null": 0,
            "n_unique": 43594,
            "percentiles": {
              "0.01": 0.02880692916098055,
              "0.05": 0.03839202706116745,
              "0.5": 0.08281248097440994,
              "0.95": 0.21174442842723853,
              "0.99": 0.31201389776078714
            },
            "rows": 44079
          }
        ],
        "n_rows": 44079,
        "split_stats": [
          {
            "feature": "ret_10",
            "max": 0.6911089393387799,
            "mean": 0.006129503034399674,
            "min": -0.3489999771118164,
            "n_null": 0,
            "rows": 44079,
            "split": "train",
            "std": 0.04710224951306396
          },
          {
            "feature": "ret_20",
            "max": 1.0435266071777156,
            "mean": 0.013244717310251937,
            "min": -0.36238979387277803,
            "n_null": 0,
            "rows": 44079,
            "split": "train",
            "std": 0.06838334079547113
          },
          {
            "feature": "ret_30",
            "max": 1.4200482793736717,
            "mean": 0.020519004786403786,
            "min": -0.45507268956117053,
            "n_null": 0,
            "rows": 44079,
            "split": "train",
            "std": 0.08671388881009878
          },
          {
            "feature": "sma_ratio_5_30",
            "max": 1.5309583140922272,
            "mean": 1.0075614457696023,
            "min": 0.7339216712314885,
            "n_null": 0,
            "rows": 44079,
            "split": "train",
            "std": 0.040316040741245396
          },
          {
            "feature": "sma_ratio_15_40",
            "max": 1.4055219739205849,
            "mean": 1.0075032514921376,
            "min": 0.7958153991309543,
            "n_null": 0,
            "rows": 44079,
            "split": "train",
            "std": 0.034611749497286515
          },
          {
            "feature": "vol_10",
            "max": 0.10663484464732022,
            "mean": 0.01374016918101806,
            "min": 0.0015616788656635896,
            "n_null": 0,
            "rows": 44079,
            "split": "train",
            "std": 0.008820871158673662
          },
          {
            "feature": "vol_30",
            "max": 0.07478897964239332,
            "mean": 0.014306333639561769,
            "min": 0.002968462793863276,
            "n_null": 0,
            "rows": 44079,
            "split": "train",
            "std": 0.007709441302863942
          },
          {
            "feature": "log_dv_med_20",
            "max": 23.616243563140582,
            "mean": 20.57987335071231,
            "min": 16.684990905005332,
            "n_null": 0,
            "rows": 44079,
            "split": "train",
            "std": 0.853530296332396
          },
          {
            "feature": "ret_5",
            "max": 0.5816967988474475,
            "mean": 0.0026756634202760956,
            "min": -0.31401471170796547,
            "n_null": 0,
            "rows": 44079,
            "split": "train",
            "std": 0.0320526204984578
          },
          {
            "feature": "ret_60",
            "max": 2.100899205931436,
            "mean": 0.04375938749919386,
            "min": -0.51544503030466,
            "n_null": 0,
            "rows": 44079,
            "split": "train",
            "std": 0.13254476040000207
          },
          {
            "feature": "ret_120",
            "max": 3.834796729281427,
            "mean": 0.09467406994914238,
            "min": -0.5832705336062745,
            "n_null": 0,
            "rows": 44079,
            "split": "train",
            "std": 0.21711777027542925
          },
          {
            "feature": "sma_ratio_10_30",
            "max": 1.3706788100338605,
            "mean": 1.0059632211598166,
            "min": 0.8072433430117467,
            "n_null": 0,
            "rows": 44079,
            "split": "train",
            "std": 0.031928543325962525
          },
          {
            "feature": "sma_ratio_20_50",
            "max": 1.4078583578644,
            "mean": 1.0090617050359507,
            "min": 0.7873475414436673,
            "n_null": 0,
            "rows": 44079,
            "split": "train",
            "std": 0.03727462740472448
          },
          {
            "feature": "price_distance_200ma",
            "max": 1.8574830846214505,
            "mean": 0.06834975291071914,
            "min": -0.5171254377381036,
            "n_null": 0,
            "rows": 44079,
            "split": "train",
            "std": 0.13568246576909126
          },
          {
            "feature": "vol_60",
            "max": 0.06342152258097594,
            "mean": 0.014528036849176076,
            "min": 0.004065848213442764,
            "n_null": 0,
            "rows": 44079,
            "split": "train",
            "std": 0.007156646653350946
          },
          {
            "feature": "vol_90",
            "max": 0.05567015172956919,
            "mean": 0.014638731138695849,
            "min": 0.004685476193175406,
            "n_null": 0,
            "rows": 44079,
            "split": "train",
            "std": 0.006910788960585458
          },
          {
            "feature": "vol_ratio_10_30",
            "max": 1.7348770309840975,
            "mean": 0.9652698767578071,
            "min": 0.13793209538108092,
            "n_null": 0,
            "rows": 44079,
            "split": "train",
            "std": 0.27916079825580403
          },
          {
            "feature": "dv_med_10",
            "max": 23.730753953205873,
            "mean": 20.58659496200901,
            "min": 16.614860951920644,
            "n_null": 0,
            "rows": 44079,
            "split": "train",
            "std": 0.8645630093721696
          },
          {
            "feature": "dv_med_30",
            "max": 23.57904635268519,
            "mean": 20.575551948811444,
            "min": 16.771171081296785,
            "n_null": 0,
            "rows": 44079,
            "split": "train",
            "std": 0.8486108140790976
          },
          {
            "feature": "vol_zscore_20",
            "max": 4.227336983944763,
            "mean": 0.019491480212611566,
            "min": -2.891870736831146,
            "n_null": 0,
            "rows": 44079,
            "split": "train",
            "std": 1.046009731909504
          },
          {
            "feature": "high_low_10_pos",
            "max": 1.0,
            "mean": 0.560743766955331,
            "min": 0.0,
            "n_null": 0,
            "rows": 44079,
            "split": "train",
            "std": 0.3013550587748326
          },
          {
            "feature": "high_low_30_pos",
            "max": 1.0,
            "mean": 0.5920283160021937,
            "min": 0.0,
            "n_null": 0,
            "rows": 44079,
            "split": "train",
            "std": 0.297191847601921
          },
          {
            "feature": "normalized_range_20",
            "max": 0.6456419808048206,
            "mean": 0.09857059558187131,
            "min": 0.01695126173364293,
            "n_null": 0,
            "rows": 44079,
            "split": "train",
            "std": 0.05930858718109879
          }
        ]
      },
      "redundancy": {
        "duplicates": [],
        "high_correlation_pairs": [
          {
            "feature_a": "dv_med_10",
            "feature_b": "dv_med_30",
            "pearson": 0.979203,
            "spearman": 0.967756
          },
          {
            "feature_a": "dv_med_10",
            "feature_b": "log_dv_med_20",
            "pearson": 0.986317,
            "spearman": 0.978755
          },
          {
            "feature_a": "dv_med_30",
            "feature_b": "log_dv_med_20",
            "pearson": 0.994678,
            "spearman": 0.991327
          },
          {
            "feature_a": "vol_60",
            "feature_b": "vol_90",
            "pearson": 0.960375,
            "spearman": 0.956033
          }
        ],
        "pearson": {
          "features": [
            "ret_10",
            "ret_20",
            "ret_30",
            "sma_ratio_5_30",
            "sma_ratio_15_40",
            "vol_10",
            "vol_30",
            "log_dv_med_20",
            "ret_5",
            "ret_60",
            "ret_120",
            "sma_ratio_10_30",
            "sma_ratio_20_50",
            "price_distance_200ma",
            "vol_60",
            "vol_90",
            "vol_ratio_10_30",
            "dv_med_10",
            "dv_med_30",
            "vol_zscore_20",
            "high_low_10_pos",
            "high_low_30_pos",
            "normalized_range_20"
          ],
          "matrix": "[[ 1.00000000e+00  6.77091535e-01  5.60602900e-01  6.92977356e-01\n   2.79285158e-01  2.77428593e-02  6.17434450e-02 -2.88359698e-02\n   6.46348520e-01  3.79121856e-01  2.64841278e-01  4.67471302e-01\n   2.04362751e-01  3.70704543e-01  7.62611137e-02  7.52078110e-02\n  -6.86352099e-02 -1.87639505e-02 -2.99533805e-02  7.91291374e-03\n   7.17666068e-01  6.49824756e-01 -1.40031888e-02]\n [ 6.77091535e-01  1.00000000e+00  8.07410262e-01  9.15517194e-01\n   7.33494777e-01 -3.44745681e-02  6.15595743e-02 -2.64502425e-02\n   4.52477747e-01  5.68911153e-01  3.89576722e-01  8.78385193e-01\n   5.52206713e-01  5.27740935e-01  9.60378359e-02  1.00969992e-01\n  -1.50443345e-01 -1.12747138e-02 -3.47333775e-02  2.57087680e-02\n   4.86656596e-01  7.21907830e-01  2.21373009e-03]\n [ 5.60602900e-01  8.07410262e-01  1.00000000e+00  8.72729272e-01\n   8.95470342e-01 -3.59946031e-02  4.94257348e-02 -1.71784697e-02\n   3.69187340e-01  7.07702002e-01  4.81066308e-01  8.69917695e-01\n   8.21847445e-01  6.35124253e-01  1.02028164e-01  1.18885467e-01\n  -1.12396598e-01 -1.21526680e-03 -2.85794396e-02  4.01527427e-02\n   3.96398276e-01  6.90543053e-01 -1.40333943e-02]\n [ 6.92977356e-01  9.15517194e-01  8.72729272e-01  1.00000000e+00\n   8.03315438e-01 -8.09659820e-02  1.50626449e-02 -2.97971464e-02\n   3.04362182e-01  6.07172746e-01  4.12645353e-01  9.45510739e-01\n   6.48758672e-01  5.59306666e-01  6.26447076e-02  7.44703506e-02\n  -1.58791822e-01 -1.51883186e-02 -3.76006570e-02  3.26959526e-02\n   4.28697438e-01  7.27319774e-01 -5.53830883e-02]\n [ 2.79285158e-01  7.33494777e-01  8.95470342e-01  8.03315438e-01\n   1.00000000e+00 -1.05858627e-01 -3.03674124e-02 -1.37086588e-02\n   9.59627884e-02  7.10153444e-01  4.74187568e-01  8.96685012e-01\n   9.34415379e-01  6.20717854e-01  4.21622081e-02  6.83791229e-02\n  -1.09341980e-01  2.59138473e-03 -2.62302105e-02  4.69317807e-02\n   1.51204402e-01  5.64048741e-01 -6.63634606e-02]\n [ 2.77428593e-02 -3.44745681e-02 -3.59946031e-02 -8.09659820e-02\n  -1.05858627e-01  1.00000000e+00  8.13416081e-01  9.70014945e-02\n   5.43344456e-02  2.03726015e-02  9.37275877e-02 -1.08670149e-01\n  -8.68451669e-02  5.65123074e-02  7.41060809e-01  7.04218325e-01\n   4.78427511e-01  1.44303221e-01  7.93138640e-02  6.81315921e-02\n  -4.20329644e-02 -1.56553034e-01  7.57146937e-01]\n [ 6.17434450e-02  6.15595743e-02  4.94257348e-02  1.50626449e-02\n  -3.03674124e-02  8.13416081e-01  1.00000000e+00  1.07897317e-01\n   5.28589674e-02  6.38480383e-02  1.43908240e-01 -2.45637764e-03\n  -4.74754958e-02  1.00184003e-01  9.11519438e-01  8.69175224e-01\n  -3.22024647e-02  1.13645491e-01  1.00971437e-01 -2.80598080e-02\n  -1.46278185e-02 -8.54055866e-02  8.10172875e-01]\n [-2.88359698e-02 -2.64502425e-02 -1.71784697e-02 -2.97971464e-02\n  -1.37086588e-02  9.70014945e-02  1.07897317e-01  1.00000000e+00\n  -2.26243197e-02  2.64864785e-02  9.06136024e-02 -2.54036425e-02\n   1.97837989e-04  9.02218080e-02  8.74835488e-02  7.48233261e-02\n   1.69675353e-02  9.86316742e-01  9.94678026e-01 -2.12383198e-02\n   7.45860373e-04 -2.08073101e-02  1.38311500e-01]\n [ 6.46348520e-01  4.52477747e-01  3.69187340e-01  3.04362182e-01\n   9.59627884e-02  5.43344456e-02  5.28589674e-02 -2.26243197e-02\n   1.00000000e+00  2.52096604e-01  1.74664628e-01  1.49156524e-01\n   7.63588536e-02  2.49274396e-01  5.54447522e-02  5.26792618e-02\n   1.04099102e-04 -1.74689855e-02 -2.14060112e-02 -2.58365850e-03\n   6.83288079e-01  5.13988556e-01 -1.27682601e-02]\n [ 3.79121856e-01  5.68911153e-01  7.07702002e-01  6.07172746e-01\n   7.10153444e-01  2.03726015e-02  6.38480383e-02  2.64864785e-02\n   2.52096604e-01  1.00000000e+00  6.92495125e-01  6.10015824e-01\n   7.94686895e-01  8.22080640e-01  1.27644712e-01  1.60211854e-01\n  -3.76523496e-02  4.00247220e-02  1.42253235e-02  3.19620567e-02\n   2.64860623e-01  4.59532723e-01  3.31923627e-02]\n [ 2.64841278e-01  3.89576722e-01  4.81066308e-01  4.12645353e-01\n   4.74187568e-01  9.37275877e-02  1.43908240e-01  9.06136024e-02\n   1.74664628e-01  6.92495125e-01  1.00000000e+00  4.11703185e-01\n   5.30706925e-01  9.17163566e-01  1.85548235e-01  2.12389578e-01\n  -2.58358587e-02  9.91475377e-02  8.25909362e-02  1.82659131e-02\n   1.78674059e-01  2.91441270e-01  1.00785489e-01]\n [ 4.67471302e-01  8.78385193e-01  8.69917695e-01  9.45510739e-01\n   8.96685012e-01 -1.08670149e-01 -2.45637764e-03 -2.54036425e-02\n   1.49156524e-01  6.10015824e-01  4.11703185e-01  1.00000000e+00\n   7.32405726e-01  5.53055575e-01  5.11186659e-02  6.62814054e-02\n  -1.70030478e-01 -1.14865870e-02 -3.48661508e-02  3.90219673e-02\n   2.57580252e-01  6.56200027e-01 -6.18088964e-02]\n [ 2.04362751e-01  5.52206713e-01  8.21847445e-01  6.48758672e-01\n   9.34415379e-01 -8.68451669e-02 -4.74754958e-02  1.97837989e-04\n   7.63588536e-02  7.94686895e-01  5.30706925e-01  7.32405726e-01\n   1.00000000e+00  6.75351089e-01  3.41271758e-02  6.99678103e-02\n  -4.99431671e-02  1.59210476e-02 -1.38677571e-02  4.55640679e-02\n   1.12494774e-01  4.63981972e-01 -6.37104650e-02]\n [ 3.70704543e-01  5.27740935e-01  6.35124253e-01  5.59306666e-01\n   6.20717854e-01  5.65123074e-02  1.00184003e-01  9.02218080e-02\n   2.49274396e-01  8.22080640e-01  9.17163566e-01  5.53055575e-01\n   6.75351089e-01  1.00000000e+00  1.41677880e-01  1.67246470e-01\n  -3.14672004e-02  1.02487195e-01  7.94358242e-02  3.40869181e-02\n   2.61699209e-01  4.22750520e-01  6.23316404e-02]\n [ 7.62611137e-02  9.60378359e-02  1.02028164e-01  6.26447076e-02\n   4.21622081e-02  7.41060809e-01  9.11519438e-01  8.74835488e-02\n   5.54447522e-02  1.27644712e-01  1.85548235e-01  5.11186659e-02\n   3.41271758e-02  1.41677880e-01  1.00000000e+00  9.60375168e-01\n  -3.63073298e-02  8.93075540e-02  8.62306635e-02 -2.42133202e-02\n  -2.68589324e-03 -4.16326390e-02  7.52052237e-01]\n [ 7.52078110e-02  1.00969992e-01  1.18885467e-01  7.44703506e-02\n   6.83791229e-02  7.04218325e-01  8.69175224e-01  7.48233261e-02\n   5.26792618e-02  1.60211854e-01  2.12389578e-01  6.62814054e-02\n   6.99678103e-02  1.67246470e-01  9.60375168e-01  1.00000000e+00\n  -3.69674712e-02  7.55136463e-02  7.40179676e-02 -2.28947188e-02\n  -3.31338224e-03 -3.32814823e-02  7.19228741e-01]\n [-6.86352099e-02 -1.50443345e-01 -1.12396598e-01 -1.58791822e-01\n  -1.09341980e-01  4.78427511e-01 -3.22024647e-02  1.69675353e-02\n   1.04099102e-04 -3.76523496e-02 -2.58358587e-02 -1.70030478e-01\n  -4.99431671e-02 -3.14672004e-02 -3.63073298e-02 -3.69674712e-02\n   1.00000000e+00  9.28456089e-02 -7.19848122e-03  1.83181968e-01\n  -6.13275049e-02 -1.66067798e-01  1.09654996e-01]\n [-1.87639505e-02 -1.12747138e-02 -1.21526680e-03 -1.51883186e-02\n   2.59138473e-03  1.44303221e-01  1.13645491e-01  9.86316742e-01\n  -1.74689855e-02  4.00247220e-02  9.91475377e-02 -1.14865870e-02\n   1.59210476e-02  1.02487195e-01  8.93075540e-02  7.55136463e-02\n   9.28456089e-02  1.00000000e+00  9.79203494e-01  1.68181777e-02\n  -1.23633926e-03 -2.19695908e-02  1.62633710e-01]\n [-2.99533805e-02 -3.47333775e-02 -2.85794396e-02 -3.76006570e-02\n  -2.62302105e-02  7.93138640e-02  1.00971437e-01  9.94678026e-01\n  -2.14060112e-02  1.42253235e-02  8.25909362e-02 -3.48661508e-02\n  -1.38677571e-02  7.94358242e-02  8.62306635e-02  7.40179676e-02\n  -7.19848122e-03  9.79203494e-01  1.00000000e+00 -1.94091926e-02\n   2.20824736e-03 -1.87617572e-02  1.13454683e-01]\n [ 7.91291374e-03  2.57087680e-02  4.01527427e-02  3.26959526e-02\n   4.69317807e-02  6.81315921e-02 -2.80598080e-02 -2.12383198e-02\n  -2.58365850e-03  3.19620567e-02  1.82659131e-02  3.90219673e-02\n   4.55640679e-02  3.40869181e-02 -2.42133202e-02 -2.28947188e-02\n   1.83181968e-01  1.68181777e-02 -1.94091926e-02  1.00000000e+00\n  -3.40305542e-02 -9.74360541e-03  1.17822536e-02]\n [ 7.17666068e-01  4.86656596e-01  3.96398276e-01  4.28697438e-01\n   1.51204402e-01 -4.20329644e-02 -1.46278185e-02  7.45860373e-04\n   6.83288079e-01  2.64860623e-01  1.78674059e-01  2.57580252e-01\n   1.12494774e-01  2.61699209e-01 -2.68589324e-03 -3.31338224e-03\n  -6.13275049e-02 -1.23633926e-03  2.20824736e-03 -3.40305542e-02\n   1.00000000e+00  7.50005376e-01 -4.42363167e-02]\n [ 6.49824756e-01  7.21907830e-01  6.90543053e-01  7.27319774e-01\n   5.64048741e-01 -1.56553034e-01 -8.54055866e-02 -2.08073101e-02\n   5.13988556e-01  4.59532723e-01  2.91441270e-01  6.56200027e-01\n   4.63981972e-01  4.22750520e-01 -4.16326390e-02 -3.32814823e-02\n  -1.66067798e-01 -2.19695908e-02 -1.87617572e-02 -9.74360541e-03\n   7.50005376e-01  1.00000000e+00 -1.29210341e-01]\n [-1.40031888e-02  2.21373009e-03 -1.40333943e-02 -5.53830883e-02\n  -6.63634606e-02  7.57146937e-01  8.10172875e-01  1.38311500e-01\n  -1.27682601e-02  3.31923627e-02  1.00785489e-01 -6.18088964e-02\n  -6.37104650e-02  6.23316404e-02  7.52052237e-01  7.19228741e-01\n   1.09654996e-01  1.62633710e-01  1.13454683e-01  1.17822536e-02\n  -4.42363167e-02 -1.29210341e-01  1.00000000e+00]]",
          "method": "pearson",
          "pairs": [
            {
              "correlation": 0.994678,
              "feature_a": "log_dv_med_20",
              "feature_b": "dv_med_30"
            },
            {
              "correlation": 0.986317,
              "feature_a": "log_dv_med_20",
              "feature_b": "dv_med_10"
            },
            {
              "correlation": 0.979203,
              "feature_a": "dv_med_10",
              "feature_b": "dv_med_30"
            },
            {
              "correlation": 0.960375,
              "feature_a": "vol_60",
              "feature_b": "vol_90"
            },
            {
              "correlation": 0.945511,
              "feature_a": "sma_ratio_5_30",
              "feature_b": "sma_ratio_10_30"
            },
            {
              "correlation": 0.934415,
              "feature_a": "sma_ratio_15_40",
              "feature_b": "sma_ratio_20_50"
            },
            {
              "correlation": 0.917164,
              "feature_a": "ret_120",
              "feature_b": "price_distance_200ma"
            },
            {
              "correlation": 0.915517,
              "feature_a": "ret_20",
              "feature_b": "sma_ratio_5_30"
            },
            {
              "correlation": 0.911519,
              "feature_a": "vol_30",
              "feature_b": "vol_60"
            },
            {
              "correlation": 0.896685,
              "feature_a": "sma_ratio_15_40",
              "feature_b": "sma_ratio_10_30"
            },
            {
              "correlation": 0.89547,
              "feature_a": "ret_30",
              "feature_b": "sma_ratio_15_40"
            },
            {
              "correlation": 0.878385,
              "feature_a": "ret_20",
              "feature_b": "sma_ratio_10_30"
            },
            {
              "correlation": 0.872729,
              "feature_a": "ret_30",
              "feature_b": "sma_ratio_5_30"
            },
            {
              "correlation": 0.869918,
              "feature_a": "ret_30",
              "feature_b": "sma_ratio_10_30"
            },
            {
              "correlation": 0.869175,
              "feature_a": "vol_30",
              "feature_b": "vol_90"
            },
            {
              "correlation": 0.822081,
              "feature_a": "ret_60",
              "feature_b": "price_distance_200ma"
            },
            {
              "correlation": 0.821847,
              "feature_a": "ret_30",
              "feature_b": "sma_ratio_20_50"
            },
            {
              "correlation": 0.813416,
              "feature_a": "vol_10",
              "feature_b": "vol_30"
            },
            {
              "correlation": 0.810173,
              "feature_a": "vol_30",
              "feature_b": "normalized_range_20"
            },
            {
              "correlation": 0.80741,
              "feature_a": "ret_20",
              "feature_b": "ret_30"
            },
            {
              "correlation": 0.803315,
              "feature_a": "sma_ratio_5_30",
              "feature_b": "sma_ratio_15_40"
            },
            {
              "correlation": 0.794687,
              "feature_a": "ret_60",
              "feature_b": "sma_ratio_20_50"
            },
            {
              "correlation": 0.757147,
              "feature_a": "vol_10",
              "feature_b": "normalized_range_20"
            },
            {
              "correlation": 0.752052,
              "feature_a": "vol_60",
              "feature_b": "normalized_range_20"
            },
            {
              "correlation": 0.750005,
              "feature_a": "high_low_10_pos",
              "feature_b": "high_low_30_pos"
            },
            {
              "correlation": 0.741061,
              "feature_a": "vol_10",
              "feature_b": "vol_60"
            },
            {
              "correlation": 0.733495,
              "feature_a": "ret_20",
              "feature_b": "sma_ratio_15_40"
            },
            {
              "correlation": 0.732406,
              "feature_a": "sma_ratio_10_30",
              "feature_b": "sma_ratio_20_50"
            },
            {
              "correlation": 0.72732,
              "feature_a": "sma_ratio_5_30",
              "feature_b": "high_low_30_pos"
            },
            {
              "correlation": 0.721908,
              "feature_a": "ret_20",
              "feature_b": "high_low_30_pos"
            },
            {
              "correlation": 0.719229,
              "feature_a": "vol_90",
              "feature_b": "normalized_range_20"
            },
            {
              "correlation": 0.717666,
              "feature_a": "ret_10",
              "feature_b": "high_low_10_pos"
            },
            {
              "correlation": 0.710153,
              "feature_a": "sma_ratio_15_40",
              "feature_b": "ret_60"
            },
            {
              "correlation": 0.707702,
              "feature_a": "ret_30",
              "feature_b": "ret_60"
            },
            {
              "correlation": 0.704218,
              "feature_a": "vol_10",
              "feature_b": "vol_90"
            },
            {
              "correlation": 0.692977,
              "feature_a": "ret_10",
              "feature_b": "sma_ratio_5_30"
            },
            {
              "correlation": 0.692495,
              "feature_a": "ret_60",
              "feature_b": "ret_120"
            },
            {
              "correlation": 0.690543,
              "feature_a": "ret_30",
              "feature_b": "high_low_30_pos"
            },
            {
              "correlation": 0.683288,
              "feature_a": "ret_5",
              "feature_b": "high_low_10_pos"
            },
            {
              "correlation": 0.677092,
              "feature_a": "ret_10",
              "feature_b": "ret_20"
            },
            {
              "correlation": 0.675351,
              "feature_a": "sma_ratio_20_50",
              "feature_b": "price_distance_200ma"
            },
            {
              "correlation": 0.6562,
              "feature_a": "sma_ratio_10_30",
              "feature_b": "high_low_30_pos"
            },
            {
              "correlation": 0.649825,
              "feature_a": "ret_10",
              "feature_b": "high_low_30_pos"
            },
            {
              "correlation": 0.648759,
              "feature_a": "sma_ratio_5_30",
              "feature_b": "sma_ratio_20_50"
            },
            {
              "correlation": 0.646349,
              "feature_a": "ret_10",
              "feature_b": "ret_5"
            },
            {
              "correlation": 0.635124,
              "feature_a": "ret_30",
              "feature_b": "price_distance_200ma"
            },
            {
              "correlation": 0.620718,
              "feature_a": "sma_ratio_15_40",
              "feature_b": "price_distance_200ma"
            },
            {
              "correlation": 0.610016,
              "feature_a": "ret_60",
              "feature_b": "sma_ratio_10_30"
            },
            {
              "correlation": 0.607173,
              "feature_a": "sma_ratio_5_30",
              "feature_b": "ret_60"
            },
            {
              "correlation": 0.568911,
              "feature_a": "ret_20",
              "feature_b": "ret_60"
            },
            {
              "correlation": 0.564049,
              "feature_a": "sma_ratio_15_40",
              "feature_b": "high_low_30_pos"
            },
            {
              "correlation": 0.560603,
              "feature_a": "ret_10",
              "feature_b": "ret_30"
            },
            {
              "correlation": 0.559307,
              "feature_a": "sma_ratio_5_30",
              "feature_b": "price_distance_200ma"
            },
            {
              "correlation": 0.553056,
              "feature_a": "sma_ratio_10_30",
              "feature_b": "price_distance_200ma"
            },
            {
              "correlation": 0.552207,
              "feature_a": "ret_20",
              "feature_b": "sma_ratio_20_50"
            },
            {
              "correlation": 0.530707,
              "feature_a": "ret_120",
              "feature_b": "sma_ratio_20_50"
            },
            {
              "correlation": 0.527741,
              "feature_a": "ret_20",
              "feature_b": "price_distance_200ma"
            },
            {
              "correlation": 0.513989,
              "feature_a": "ret_5",
              "feature_b": "high_low_30_pos"
            },
            {
              "correlation": 0.486657,
              "feature_a": "ret_20",
              "feature_b": "high_low_10_pos"
            },
            {
              "correlation": 0.481066,
              "feature_a": "ret_30",
              "feature_b": "ret_120"
            },
            {
              "correlation": 0.478428,
              "feature_a": "vol_10",
              "feature_b": "vol_ratio_10_30"
            },
            {
              "correlation": 0.474188,
              "feature_a": "sma_ratio_15_40",
              "feature_b": "ret_120"
            },
            {
              "correlation": 0.467471,
              "feature_a": "ret_10",
              "feature_b": "sma_ratio_10_30"
            },
            {
              "correlation": 0.463982,
              "feature_a": "sma_ratio_20_50",
              "feature_b": "high_low_30_pos"
            },
            {
              "correlation": 0.459533,
              "feature_a": "ret_60",
              "feature_b": "high_low_30_pos"
            },
            {
              "correlation": 0.452478,
              "feature_a": "ret_20",
              "feature_b": "ret_5"
            },
            {
              "correlation": 0.428697,
              "feature_a": "sma_ratio_5_30",
              "feature_b": "high_low_10_pos"
            },
            {
              "correlation": 0.422751,
              "feature_a": "price_distance_200ma",
              "feature_b": "high_low_30_pos"
            },
            {
              "correlation": 0.412645,
              "feature_a": "sma_ratio_5_30",
              "feature_b": "ret_120"
            },
            {
              "correlation": 0.411703,
              "feature_a": "ret_120",
              "feature_b": "sma_ratio_10_30"
            },
            {
              "correlation": 0.396398,
              "feature_a": "ret_30",
              "feature_b": "high_low_10_pos"
            },
            {
              "correlation": 0.389577,
              "feature_a": "ret_20",
              "feature_b": "ret_120"
            },
            {
              "correlation": 0.379122,
              "feature_a": "ret_10",
              "feature_b": "ret_60"
            },
            {
              "correlation": 0.370705,
              "feature_a": "ret_10",
              "feature_b": "price_distance_200ma"
            },
            {
              "correlation": 0.369187,
              "feature_a": "ret_30",
              "feature_b": "ret_5"
            },
            {
              "correlation": 0.304362,
              "feature_a": "sma_ratio_5_30",
              "feature_b": "ret_5"
            },
            {
              "correlation": 0.291441,
              "feature_a": "ret_120",
              "feature_b": "high_low_30_pos"
            },
            {
              "correlation": 0.279285,
              "feature_a": "ret_10",
              "feature_b": "sma_ratio_15_40"
            },
            {
              "correlation": 0.264861,
              "feature_a": "ret_60",
              "feature_b": "high_low_10_pos"
            },
            {
              "correlation": 0.264841,
              "feature_a": "ret_10",
              "feature_b": "ret_120"
            },
            {
              "correlation": 0.261699,
              "feature_a": "price_distance_200ma",
              "feature_b": "high_low_10_pos"
            },
            {
              "correlation": 0.25758,
              "feature_a": "sma_ratio_10_30",
              "feature_b": "high_low_10_pos"
            },
            {
              "correlation": 0.252097,
              "feature_a": "ret_5",
              "feature_b": "ret_60"
            },
            {
              "correlation": 0.249274,
              "feature_a": "ret_5",
              "feature_b": "price_distance_200ma"
            },
            {
              "correlation": 0.21239,
              "feature_a": "ret_120",
              "feature_b": "vol_90"
            },
            {
              "correlation": 0.204363,
              "feature_a": "ret_10",
              "feature_b": "sma_ratio_20_50"
            },
            {
              "correlation": 0.185548,
              "feature_a": "ret_120",
              "feature_b": "vol_60"
            },
            {
              "correlation": 0.183182,
              "feature_a": "vol_ratio_10_30",
              "feature_b": "vol_zscore_20"
            },
            {
              "correlation": 0.178674,
              "feature_a": "ret_120",
              "feature_b": "high_low_10_pos"
            },
            {
              "correlation": 0.174665,
              "feature_a": "ret_5",
              "feature_b": "ret_120"
            },
            {
              "correlation": -0.17003,
              "feature_a": "sma_ratio_10_30",
              "feature_b": "vol_ratio_10_30"
            },
            {
              "correlation": 0.167246,
              "feature_a": "price_distance_200ma",
              "feature_b": "vol_90"
            },
            {
              "correlation": -0.166068,
              "feature_a": "vol_ratio_10_30",
              "feature_b": "high_low_30_pos"
            },
            {
              "correlation": 0.162634,
              "feature_a": "dv_med_10",
              "feature_b": "normalized_range_20"
            },
            {
              "correlation": 0.160212,
              "feature_a": "ret_60",
              "feature_b": "vol_90"
            },
            {
              "correlation": -0.158792,
              "feature_a": "sma_ratio_5_30",
              "feature_b": "vol_ratio_10_30"
            },
            {
              "correlation": -0.156553,
              "feature_a": "vol_10",
              "feature_b": "high_low_30_pos"
            },
            {
              "correlation": 0.151204,
              "feature_a": "sma_ratio_15_40",
              "feature_b": "high_low_10_pos"
            },
            {
              "correlation": -0.150443,
              "feature_a": "ret_20",
              "feature_b": "vol_ratio_10_30"
            },
            {
              "correlation": 0.149157,
              "feature_a": "ret_5",
              "feature_b": "sma_ratio_10_30"
            },
            {
              "correlation": 0.144303,
              "feature_a": "vol_10",
              "feature_b": "dv_med_10"
            },
            {
              "correlation": 0.143908,
              "feature_a": "vol_30",
              "feature_b": "ret_120"
            },
            {
              "correlation": 0.141678,
              "feature_a": "price_distance_200ma",
              "feature_b": "vol_60"
            },
            {
              "correlation": 0.138311,
              "feature_a": "log_dv_med_20",
              "feature_b": "normalized_range_20"
            },
            {
              "correlation": -0.12921,
              "feature_a": "high_low_30_pos",
              "feature_b": "normalized_range_20"
            },
            {
              "correlation": 0.127645,
              "feature_a": "ret_60",
              "feature_b": "vol_60"
            },
            {
              "correlation": 0.118885,
              "feature_a": "ret_30",
              "feature_b": "vol_90"
            },
            {
              "correlation": 0.113645,
              "feature_a": "vol_30",
              "feature_b": "dv_med_10"
            },
            {
              "correlation": 0.113455,
              "feature_a": "dv_med_30",
              "feature_b": "normalized_range_20"
            },
            {
              "correlation": 0.112495,
              "feature_a": "sma_ratio_20_50",
              "feature_b": "high_low_10_pos"
            },
            {
              "correlation": -0.112397,
              "feature_a": "ret_30",
              "feature_b": "vol_ratio_10_30"
            },
            {
              "correlation": 0.109655,
              "feature_a": "vol_ratio_10_30",
              "feature_b": "normalized_range_20"
            },
            {
              "correlation": -0.109342,
              "feature_a": "sma_ratio_15_40",
              "feature_b": "vol_ratio_10_30"
            },
            {
              "correlation": -0.10867,
              "feature_a": "vol_10",
              "feature_b": "sma_ratio_10_30"
            },
            {
              "correlation": 0.107897,
              "feature_a": "vol_30",
              "feature_b": "log_dv_med_20"
            },
            {
              "correlation": -0.105859,
              "feature_a": "sma_ratio_15_40",
              "feature_b": "vol_10"
            },
            {
              "correlation": 0.102487,
              "feature_a": "price_distance_200ma",
              "feature_b": "dv_med_10"
            },
            {
              "correlation": 0.102028,
              "feature_a": "ret_30",
              "feature_b": "vol_60"
            },
            {
              "correlation": 0.100971,
              "feature_a": "vol_30",
              "feature_b": "dv_med_30"
            },
            {
              "correlation": 0.10097,
              "feature_a": "ret_20",
              "feature_b": "vol_90"
            },
            {
              "correlation": 0.100785,
              "feature_a": "ret_120",
              "feature_b": "normalized_range_20"
            },
            {
              "correlation": 0.100184,
              "feature_a": "vol_30",
              "feature_b": "price_distance_200ma"
            },
            {
              "correlation": 0.099148,
              "feature_a": "ret_120",
              "feature_b": "dv_med_10"
            },
            {
              "correlation": 0.097001,
              "feature_a": "vol_10",
              "feature_b": "log_dv_med_20"
            },
            {
              "correlation": 0.096038,
              "feature_a": "ret_20",
              "feature_b": "vol_60"
            },
            {
              "correlation": 0.095963,
              "feature_a": "sma_ratio_15_40",
              "feature_b": "ret_5"
            },
            {
              "correlation": 0.093728,
              "feature_a": "vol_10",
              "feature_b": "ret_120"
            },
            {
              "correlation": 0.092846,
              "feature_a": "vol_ratio_10_30",
              "feature_b": "dv_med_10"
            },
            {
              "correlation": 0.090614,
              "feature_a": "log_dv_med_20",
              "feature_b": "ret_120"
            },
            {
              "correlation": 0.090222,
              "feature_a": "log_dv_med_20",
              "feature_b": "price_distance_200ma"
            },
            {
              "correlation": 0.089308,
              "feature_a": "vol_60",
              "feature_b": "dv_med_10"
            },
            {
              "correlation": 0.087484,
              "feature_a": "log_dv_med_20",
              "feature_b": "vol_60"
            },
            {
              "correlation": -0.086845,
              "feature_a": "vol_10",
              "feature_b": "sma_ratio_20_50"
            },
            {
              "correlation": 0.086231,
              "feature_a": "vol_60",
              "feature_b": "dv_med_30"
            },
            {
              "correlation": -0.085406,
              "feature_a": "vol_30",
              "feature_b": "high_low_30_pos"
            },
            {
              "correlation": 0.082591,
              "feature_a": "ret_120",
              "feature_b": "dv_med_30"
            },
            {
              "correlation": -0.080966,
              "feature_a": "sma_ratio_5_30",
              "feature_b": "vol_10"
            },
            {
              "correlation": 0.079436,
              "feature_a": "price_distance_200ma",
              "feature_b": "dv_med_30"
            },
            {
              "correlation": 0.079314,
              "feature_a": "vol_10",
              "feature_b": "dv_med_30"
            },
            {
              "correlation": 0.076359,
              "feature_a": "ret_5",
              "feature_b": "sma_ratio_20_50"
            },
            {
              "correlation": 0.076261,
              "feature_a": "ret_10",
              "feature_b": "vol_60"
            },
            {
              "correlation": 0.075514,
              "feature_a": "vol_90",
              "feature_b": "dv_med_10"
            },
            {
              "correlation": 0.075208,
              "feature_a": "ret_10",
              "feature_b": "vol_90"
            },
            {
              "correlation": 0.074823,
              "feature_a": "log_dv_med_20",
              "feature_b": "vol_90"
            },
            {
              "correlation": 0.07447,
              "feature_a": "sma_ratio_5_30",
              "feature_b": "vol_90"
            },
            {
              "correlation": 0.074018,
              "feature_a": "vol_90",
              "feature_b": "dv_med_30"
            },
            {
              "correlation": 0.069968,
              "feature_a": "sma_ratio_20_50",
              "feature_b": "vol_90"
            },
            {
              "correlation": -0.068635,
              "feature_a": "ret_10",
              "feature_b": "vol_ratio_10_30"
            },
            {
              "correlation": 0.068379,
              "feature_a": "sma_ratio_15_40",
              "feature_b": "vol_90"
            },
            {
              "correlation": 0.068132,
              "feature_a": "vol_10",
              "feature_b": "vol_zscore_20"
            },
            {
              "correlation": -0.066363,
              "feature_a": "sma_ratio_15_40",
              "feature_b": "normalized_range_20"
            },
            {
              "correlation": 0.066281,
              "feature_a": "sma_ratio_10_30",
              "feature_b": "vol_90"
            },
            {
              "correlation": 0.063848,
              "feature_a": "vol_30",
              "feature_b": "ret_60"
            },
            {
              "correlation": -0.06371,
              "feature_a": "sma_ratio_20_50",
              "feature_b": "normalized_range_20"
            },
            {
              "correlation": 0.062645,
              "feature_a": "sma_ratio_5_30",
              "feature_b": "vol_60"
            },
            {
              "correlation": 0.062332,
              "feature_a": "price_distance_200ma",
              "feature_b": "normalized_range_20"
            },
            {
              "correlation": -0.061809,
              "feature_a": "sma_ratio_10_30",
              "feature_b": "normalized_range_20"
            },
            {
              "correlation": 0.061743,
              "feature_a": "ret_10",
              "feature_b": "vol_30"
            },
            {
              "correlation": 0.06156,
              "feature_a": "ret_20",
              "feature_b": "vol_30"
            },
            {
              "correlation": -0.061328,
              "feature_a": "vol_ratio_10_30",
              "feature_b": "high_low_10_pos"
            },
            {
              "correlation": 0.056512,
              "feature_a": "vol_10",
              "feature_b": "price_distance_200ma"
            },
            {
              "correlation": 0.055445,
              "feature_a": "ret_5",
              "feature_b": "vol_60"
            },
            {
              "correlation": -0.055383,
              "feature_a": "sma_ratio_5_30",
              "feature_b": "normalized_range_20"
            },
            {
              "correlation": 0.054334,
              "feature_a": "vol_10",
              "feature_b": "ret_5"
            },
            {
              "correlation": 0.052859,
              "feature_a": "vol_30",
              "feature_b": "ret_5"
            },
            {
              "correlation": 0.052679,
              "feature_a": "ret_5",
              "feature_b": "vol_90"
            },
            {
              "correlation": 0.051119,
              "feature_a": "sma_ratio_10_30",
              "feature_b": "vol_60"
            },
            {
              "correlation": -0.049943,
              "feature_a": "sma_ratio_20_50",
              "feature_b": "vol_ratio_10_30"
            },
            {
              "correlation": 0.049426,
              "feature_a": "ret_30",
              "feature_b": "vol_30"
            },
            {
              "correlation": -0.047475,
              "feature_a": "vol_30",
              "feature_b": "sma_ratio_20_50"
            },
            {
              "correlation": 0.046932,
              "feature_a": "sma_ratio_15_40",
              "feature_b": "vol_zscore_20"
            },
            {
              "correlation": 0.045564,
              "feature_a": "sma_ratio_20_50",
              "feature_b": "vol_zscore_20"
            },
            {
              "correlation": -0.044236,
              "feature_a": "high_low_10_pos",
              "feature_b": "normalized_range_20"
            },
            {
              "correlation": 0.042162,
              "feature_a": "sma_ratio_15_40",
              "feature_b": "vol_60"
            },
            {
              "correlation": -0.042033,
              "feature_a": "vol_10",
              "feature_b": "high_low_10_pos"
            },
            {
              "correlation": -0.041633,
              "feature_a": "vol_60",
              "feature_b": "high_low_30_pos"
            },
            {
              "correlation": 0.040153,
              "feature_a": "ret_30",
              "feature_b": "vol_zscore_20"
            },
            {
              "correlation": 0.040025,
              "feature_a": "ret_60",
              "feature_b": "dv_med_10"
            },
            {
              "correlation": 0.039022,
              "feature_a": "sma_ratio_10_30",
              "feature_b": "vol_zscore_20"
            },
            {
              "correlation": -0.037652,
              "feature_a": "ret_60",
              "feature_b": "vol_ratio_10_30"
            },
            {
              "correlation": -0.037601,
              "feature_a": "sma_ratio_5_30",
              "feature_b": "dv_med_30"
            },
            {
              "correlation": -0.036967,
              "feature_a": "vol_90",
              "feature_b": "vol_ratio_10_30"
            },
            {
              "correlation": -0.036307,
              "feature_a": "vol_60",
              "feature_b": "vol_ratio_10_30"
            },
            {
              "correlation": -0.035995,
              "feature_a": "ret_30",
              "feature_b": "vol_10"
            },
            {
              "correlation": -0.034866,
              "feature_a": "sma_ratio_10_30",
              "feature_b": "dv_med_30"
            },
            {
              "correlation": -0.034733,
              "feature_a": "ret_20",
              "feature_b": "dv_med_30"
            },
            {
              "correlation": -0.034475,
              "feature_a": "ret_20",
              "feature_b": "vol_10"
            },
            {
              "correlation": 0.034127,
              "feature_a": "sma_ratio_20_50",
              "feature_b": "vol_60"
            },
            {
              "correlation": 0.034087,
              "feature_a": "price_distance_200ma",
              "feature_b": "vol_zscore_20"
            },
            {
              "correlation": -0.034031,
              "feature_a": "vol_zscore_20",
              "feature_b": "high_low_10_pos"
            },
            {
              "correlation": -0.033281,
              "feature_a": "vol_90",
              "feature_b": "high_low_30_pos"
            },
            {
              "correlation": 0.033192,
              "feature_a": "ret_60",
              "feature_b": "normalized_range_20"
            },
            {
              "correlation": 0.032696,
              "feature_a": "sma_ratio_5_30",
              "feature_b": "vol_zscore_20"
            },
            {
              "correlation": -0.032202,
              "feature_a": "vol_30",
              "feature_b": "vol_ratio_10_30"
            },
            {
              "correlation": 0.031962,
              "feature_a": "ret_60",
              "feature_b": "vol_zscore_20"
            },
            {
              "correlation": -0.031467,
              "feature_a": "price_distance_200ma",
              "feature_b": "vol_ratio_10_30"
            },
            {
              "correlation": -0.030367,
              "feature_a": "sma_ratio_15_40",
              "feature_b": "vol_30"
            },
            {
              "correlation": -0.029953,
              "feature_a": "ret_10",
              "feature_b": "dv_med_30"
            },
            {
              "correlation": -0.029797,
              "feature_a": "sma_ratio_5_30",
              "feature_b": "log_dv_med_20"
            },
            {
              "correlation": -0.028836,
              "feature_a": "ret_10",
              "feature_b": "log_dv_med_20"
            },
            {
              "correlation": -0.028579,
              "feature_a": "ret_30",
              "feature_b": "dv_med_30"
            },
            {
              "correlation": -0.02806,
              "feature_a": "vol_30",
              "feature_b": "vol_zscore_20"
            },
            {
              "correlation": 0.027743,
              "feature_a": "ret_10",
              "feature_b": "vol_10"
            },
            {
              "correlation": 0.026486,
              "feature_a": "log_dv_med_20",
              "feature_b": "ret_60"
            },
            {
              "correlation": -0.02645,
              "feature_a": "ret_20",
              "feature_b": "log_dv_med_20"
            },
            {
              "correlation": -0.02623,
              "feature_a": "sma_ratio_15_40",
              "feature_b": "dv_med_30"
            },
            {
              "correlation": -0.025836,
              "feature_a": "ret_120",
              "feature_b": "vol_ratio_10_30"
            },
            {
              "correlation": 0.025709,
              "feature_a": "ret_20",
              "feature_b": "vol_zscore_20"
            },
            {
              "correlation": -0.025404,
              "feature_a": "log_dv_med_20",
              "feature_b": "sma_ratio_10_30"
            },
            {
              "correlation": -0.024213,
              "feature_a": "vol_60",
              "feature_b": "vol_zscore_20"
            },
            {
              "correlation": -0.022895,
              "feature_a": "vol_90",
              "feature_b": "vol_zscore_20"
            },
            {
              "correlation": -0.022624,
              "feature_a": "log_dv_med_20",
              "feature_b": "ret_5"
            },
            {
              "correlation": -0.02197,
              "feature_a": "dv_med_10",
              "feature_b": "high_low_30_pos"
            },
            {
              "correlation": -0.021406,
              "feature_a": "ret_5",
              "feature_b": "dv_med_30"
            },
            {
              "correlation": -0.021238,
              "feature_a": "log_dv_med_20",
              "feature_b": "vol_zscore_20"
            },
            {
              "correlation": -0.020807,
              "feature_a": "log_dv_med_20",
              "feature_b": "high_low_30_pos"
            },
            {
              "correlation": 0.020373,
              "feature_a": "vol_10",
              "feature_b": "ret_60"
            },
            {
              "correlation": -0.019409,
              "feature_a": "dv_med_30",
              "feature_b": "vol_zscore_20"
            },
            {
              "correlation": -0.018764,
              "feature_a": "ret_10",
              "feature_b": "dv_med_10"
            },
            {
              "correlation": -0.018762,
              "feature_a": "dv_med_30",
              "feature_b": "high_low_30_pos"
            },
            {
              "correlation": 0.018266,
              "feature_a": "ret_120",
              "feature_b": "vol_zscore_20"
            },
            {
              "correlation": -0.017469,
              "feature_a": "ret_5",
              "feature_b": "dv_med_10"
            },
            {
              "correlation": -0.017178,
              "feature_a": "ret_30",
              "feature_b": "log_dv_med_20"
            },
            {
              "correlation": 0.016968,
              "feature_a": "log_dv_med_20",
              "feature_b": "vol_ratio_10_30"
            },
            {
              "correlation": 0.016818,
              "feature_a": "dv_med_10",
              "feature_b": "vol_zscore_20"
            },
            {
              "correlation": 0.015921,
              "feature_a": "sma_ratio_20_50",
              "feature_b": "dv_med_10"
            },
            {
              "correlation": -0.015188,
              "feature_a": "sma_ratio_5_30",
              "feature_b": "dv_med_10"
            },
            {
              "correlation": 0.015063,
              "feature_a": "sma_ratio_5_30",
              "feature_b": "vol_30"
            },
            {
              "correlation": -0.014628,
              "feature_a": "vol_30",
              "feature_b": "high_low_10_pos"
            },
            {
              "correlation": 0.014225,
              "feature_a": "ret_60",
              "feature_b": "dv_med_30"
            },
            {
              "correlation": -0.014033,
              "feature_a": "ret_30",
              "feature_b": "normalized_range_20"
            },
            {
              "correlation": -0.014003,
              "feature_a": "ret_10",
              "feature_b": "normalized_range_20"
            },
            {
              "correlation": -0.013868,
              "feature_a": "sma_ratio_20_50",
              "feature_b": "dv_med_30"
            },
            {
              "correlation": -0.013709,
              "feature_a": "sma_ratio_15_40",
              "feature_b": "log_dv_med_20"
            },
            {
              "correlation": -0.012768,
              "feature_a": "ret_5",
              "feature_b": "normalized_range_20"
            },
            {
              "correlation": 0.011782,
              "feature_a": "vol_zscore_20",
              "feature_b": "normalized_range_20"
            },
            {
              "correlation": -0.011487,
              "feature_a": "sma_ratio_10_30",
              "feature_b": "dv_med_10"
            },
            {
              "correlation": -0.011275,
              "feature_a": "ret_20",
              "feature_b": "dv_med_10"
            },
            {
              "correlation": -0.009744,
              "feature_a": "vol_zscore_20",
              "feature_b": "high_low_30_pos"
            },
            {
              "correlation": 0.007913,
              "feature_a": "ret_10",
              "feature_b": "vol_zscore_20"
            },
            {
              "correlation": -0.007198,
              "feature_a": "vol_ratio_10_30",
              "feature_b": "dv_med_30"
            },
            {
              "correlation": -0.003313,
              "feature_a": "vol_90",
              "feature_b": "high_low_10_pos"
            },
            {
              "correlation": -0.002686,
              "feature_a": "vol_60",
              "feature_b": "high_low_10_pos"
            },
            {
              "correlation": 0.002591,
              "feature_a": "sma_ratio_15_40",
              "feature_b": "dv_med_10"
            },
            {
              "correlation": -0.002584,
              "feature_a": "ret_5",
              "feature_b": "vol_zscore_20"
            },
            {
              "correlation": -0.002456,
              "feature_a": "vol_30",
              "feature_b": "sma_ratio_10_30"
            },
            {
              "correlation": 0.002214,
              "feature_a": "ret_20",
              "feature_b": "normalized_range_20"
            },
            {
              "correlation": 0.002208,
              "feature_a": "dv_med_30",
              "feature_b": "high_low_10_pos"
            },
            {
              "correlation": -0.001236,
              "feature_a": "dv_med_10",
              "feature_b": "high_low_10_pos"
            },
            {
              "correlation": -0.001215,
              "feature_a": "ret_30",
              "feature_b": "dv_med_10"
            },
            {
              "correlation": 0.000746,
              "feature_a": "log_dv_med_20",
              "feature_b": "high_low_10_pos"
            },
            {
              "correlation": 0.000198,
              "feature_a": "log_dv_med_20",
              "feature_b": "sma_ratio_20_50"
            },
            {
              "correlation": 0.000104,
              "feature_a": "ret_5",
              "feature_b": "vol_ratio_10_30"
            }
          ],
          "rows_used": 44079
        },
        "spearman": {
          "features": [
            "ret_10",
            "ret_20",
            "ret_30",
            "sma_ratio_5_30",
            "sma_ratio_15_40",
            "vol_10",
            "vol_30",
            "log_dv_med_20",
            "ret_5",
            "ret_60",
            "ret_120",
            "sma_ratio_10_30",
            "sma_ratio_20_50",
            "price_distance_200ma",
            "vol_60",
            "vol_90",
            "vol_ratio_10_30",
            "dv_med_10",
            "dv_med_30",
            "vol_zscore_20",
            "high_low_10_pos",
            "high_low_30_pos",
            "normalized_range_20"
          ],
          "matrix": "[[ 1.00000000e+00  6.34630951e-01  5.12260341e-01  6.45209896e-01\n   2.36851132e-01 -4.56422390e-02  1.49512470e-02 -1.12847031e-02\n   6.06984278e-01  3.39873605e-01  2.46562178e-01  4.23843811e-01\n   1.60040749e-01  3.32645058e-01  4.07668591e-02  4.64142073e-02\n  -1.02941228e-01 -1.24596313e-02 -9.16278927e-03 -6.76876552e-03\n   8.26225394e-01  7.26877462e-01  2.95387392e-02]\n [ 6.34630951e-01  1.00000000e+00  7.72686156e-01  9.00550113e-01\n   6.88647335e-01 -1.25037900e-01 -2.33503524e-02 -2.13362717e-02\n   4.06246905e-01  5.27873486e-01  3.64039661e-01  8.55854926e-01\n   4.99745453e-01  4.83563833e-01  2.75098772e-02  4.48929583e-02\n  -1.83048522e-01 -1.90903433e-02 -1.96538471e-02  1.86472569e-02\n   5.45481157e-01  8.30856918e-01  2.63717762e-02]\n [ 5.12260341e-01  7.72686156e-01  1.00000000e+00  8.57978820e-01\n   8.80562386e-01 -1.27799819e-01 -6.39310772e-02 -2.65044679e-02\n   3.25753506e-01  6.65631558e-01  4.48522758e-01  8.56076155e-01\n   7.87783279e-01  5.84396220e-01  1.29101753e-02  4.18478479e-02\n  -1.23236387e-01 -2.07973946e-02 -2.70654562e-02  3.40814353e-02\n   4.45228936e-01  8.07978892e-01 -3.81642341e-02]\n [ 6.45209896e-01  9.00550113e-01  8.57978820e-01  1.00000000e+00\n   7.75051358e-01 -1.41118453e-01 -3.68142553e-02 -2.42608226e-02\n   2.67831926e-01  5.78308402e-01  3.97881394e-01  9.37745938e-01\n   6.08024328e-01  5.25049993e-01  2.31465162e-02  4.44041108e-02\n  -1.88191148e-01 -2.18180123e-02 -2.30147027e-02  3.00161818e-02\n   4.69460892e-01  8.32173327e-01  1.14275445e-02]\n [ 2.36851132e-01  6.88647335e-01  8.80562386e-01  7.75051358e-01\n   1.00000000e+00 -1.33909957e-01 -8.23706902e-02 -2.09065440e-02\n   6.56065437e-02  6.85897979e-01  4.53953843e-01  8.75785900e-01\n   9.19924997e-01  5.81528935e-01  2.41787417e-03  3.59117798e-02\n  -1.03504464e-01 -1.20075516e-02 -2.37830664e-02  5.21901296e-02\n   1.62850249e-01  6.26010436e-01 -3.85206434e-02]\n [-4.56422390e-02 -1.25037900e-01 -1.27799819e-01 -1.41118453e-01\n  -1.33909957e-01  1.00000000e+00  8.05617969e-01  2.42471673e-01\n   1.73429980e-02 -9.16162371e-02 -1.56390305e-02 -1.51308046e-01\n  -1.09331397e-01 -3.66689364e-02  7.27890186e-01  6.95138981e-01\n   5.11838670e-01  2.82961616e-01  2.23684912e-01  6.25474385e-02\n  -5.29799878e-02 -1.83623532e-01  7.16411460e-01]\n [ 1.49512470e-02 -2.33503524e-02 -6.39310772e-02 -3.68142553e-02\n  -8.23706902e-02  8.05617969e-01  1.00000000e+00  2.86752142e-01\n   2.85890729e-02 -7.10995264e-02  1.44655165e-02 -4.77315825e-02\n  -9.98102760e-02 -4.24820331e-03  9.01679433e-01  8.56637138e-01\n  -3.81014285e-02  2.81325984e-01  2.82542620e-01 -4.85663589e-02\n  -1.64828613e-02 -9.60084754e-02  7.99142640e-01]\n [-1.12847031e-02 -2.13362717e-02 -2.65044679e-02 -2.42608226e-02\n  -2.09065440e-02  2.42471673e-01  2.86752142e-01  1.00000000e+00\n  -4.82344240e-03  2.62224591e-03  6.88906918e-02 -2.20915342e-02\n  -1.53678029e-02  6.72031880e-02  2.78222041e-01  2.72627243e-01\n   1.50256937e-02  9.78754631e-01  9.91326871e-01 -2.98733673e-02\n  -3.69868670e-03 -3.51223310e-02  2.83994158e-01]\n [ 6.06984278e-01  4.06246905e-01  3.25753506e-01  2.67831926e-01\n   6.56065437e-02  1.73429980e-02  2.85890729e-02 -4.82344240e-03\n   1.00000000e+00  2.24684091e-01  1.67326869e-01  1.21400778e-01\n   4.58516721e-02  2.24650056e-01  3.85685725e-02  4.03005067e-02\n  -8.59644812e-03 -2.54843742e-03 -3.35931443e-03 -1.96283997e-02\n   7.82661758e-01  5.67382457e-01  1.87508934e-02]\n [ 3.39873605e-01  5.27873486e-01  6.65631558e-01  5.78308402e-01\n   6.85897979e-01 -9.16162371e-02 -7.10995264e-02  2.62224591e-03\n   2.24684091e-01  1.00000000e+00  6.35542824e-01  5.82990152e-01\n   7.75713387e-01  7.63891936e-01 -1.96023494e-02  3.41439863e-02\n  -3.84946846e-02  1.02952958e-02 -2.21483844e-03  3.30611889e-02\n   3.06635525e-01  5.48194626e-01 -5.13460289e-02]\n [ 2.46562178e-01  3.64039661e-01  4.48522758e-01  3.97881394e-01\n   4.53953843e-01 -1.56390305e-02  1.44655165e-02  6.88906918e-02\n   1.67326869e-01  6.35542824e-01  1.00000000e+00  3.94886705e-01\n   5.07029100e-01  9.14063510e-01  3.10418975e-02  4.36572506e-02\n  -2.56690480e-02  7.62227717e-02  6.46189928e-02  1.68135018e-02\n   2.25269925e-01  3.73622030e-01  1.65454238e-02]\n [ 4.23843811e-01  8.55854926e-01  8.56076155e-01  9.37745938e-01\n   8.75785900e-01 -1.51308046e-01 -4.77315825e-02 -2.20915342e-02\n   1.21400778e-01  5.82990152e-01  3.94886705e-01  1.00000000e+00\n   6.93717555e-01  5.18114643e-01  1.68703650e-02  4.03753493e-02\n  -1.89357452e-01 -1.87466683e-02 -2.16744862e-02  4.07763287e-02\n   2.79216121e-01  7.41816852e-01  4.00605723e-03]\n [ 1.60040749e-01  4.99745453e-01  7.87783279e-01  6.08024328e-01\n   9.19924997e-01 -1.09331397e-01 -9.98102760e-02 -1.53678029e-02\n   4.58516721e-02  7.75713387e-01  5.07029100e-01  6.93717555e-01\n   1.00000000e+00  6.32081325e-01 -1.07006479e-02  3.13499354e-02\n  -3.09597004e-02 -4.00829087e-03 -2.10518956e-02  5.09819782e-02\n   1.17322780e-01  5.04862528e-01 -6.34322392e-02]\n [ 3.32645058e-01  4.83563833e-01  5.84396220e-01  5.25049993e-01\n   5.81528935e-01 -3.66689364e-02 -4.24820331e-03  6.72031880e-02\n   2.24650056e-01  7.63891936e-01  9.14063510e-01  5.18114643e-01\n   6.32081325e-01  1.00000000e+00  2.91802200e-02  5.51069770e-02\n  -3.46860501e-02  7.51653658e-02  6.23808786e-02  3.03867697e-02\n   3.01634242e-01  4.96990571e-01  6.59710703e-03]\n [ 4.07668591e-02  2.75098772e-02  1.29101753e-02  2.31465162e-02\n   2.41787417e-03  7.27890186e-01  9.01679433e-01  2.78222041e-01\n   3.85685725e-02 -1.96023494e-02  3.10418975e-02  1.68703650e-02\n  -1.07006479e-02  2.91802200e-02  1.00000000e+00  9.56032551e-01\n  -4.68215836e-02  2.69676780e-01  2.81556659e-01 -4.14163955e-02\n   4.67350737e-04 -4.17036022e-02  7.35860200e-01]\n [ 4.64142073e-02  4.48929583e-02  4.18478479e-02  4.44041108e-02\n   3.59117798e-02  6.95138981e-01  8.56637138e-01  2.72627243e-01\n   4.03005067e-02  3.41439863e-02  4.36572506e-02  4.03753493e-02\n   3.13499354e-02  5.51069770e-02  9.56032551e-01  1.00000000e+00\n  -4.29482381e-02  2.63828925e-01  2.76182210e-01 -3.91445207e-02\n   3.48191097e-03 -2.68349038e-02  7.03066646e-01]\n [-1.02941228e-01 -1.83048522e-01 -1.23236387e-01 -1.88191148e-01\n  -1.03504464e-01  5.11838670e-01 -3.81014285e-02  1.50256937e-02\n  -8.59644812e-03 -3.84946846e-02 -2.56690480e-02 -1.89357452e-01\n  -3.09597004e-02 -3.46860501e-02 -4.68215836e-02 -4.29482381e-02\n   1.00000000e+00  9.83879895e-02 -1.38006393e-02  1.84097387e-01\n  -6.13585453e-02 -1.66850481e-01  8.35570411e-02]\n [-1.24596313e-02 -1.90903433e-02 -2.07973946e-02 -2.18180123e-02\n  -1.20075516e-02  2.82961616e-01  2.81325984e-01  9.78754631e-01\n  -2.54843742e-03  1.02952958e-02  7.62227717e-02 -1.87466683e-02\n  -4.00829087e-03  7.51653658e-02  2.69676780e-01  2.63828925e-01\n   9.83879895e-02  1.00000000e+00  9.67755839e-01  2.21394465e-02\n  -4.62522758e-03 -3.57062981e-02  2.98436933e-01]\n [-9.16278927e-03 -1.96538471e-02 -2.70654562e-02 -2.30147027e-02\n  -2.37830664e-02  2.23684912e-01  2.82542620e-01  9.91326871e-01\n  -3.35931443e-03 -2.21483844e-03  6.46189928e-02 -2.16744862e-02\n  -2.10518956e-02  6.23808786e-02  2.81556659e-01  2.76182210e-01\n  -1.38006393e-02  9.67755839e-01  1.00000000e+00 -3.02970371e-02\n  -2.00319032e-03 -3.13272789e-02  2.58547362e-01]\n [-6.76876552e-03  1.86472569e-02  3.40814353e-02  3.00161818e-02\n   5.21901296e-02  6.25474385e-02 -4.85663589e-02 -2.98733673e-02\n  -1.96283997e-02  3.30611889e-02  1.68135018e-02  4.07763287e-02\n   5.09819782e-02  3.03867697e-02 -4.14163955e-02 -3.91445207e-02\n   1.84097387e-01  2.21394465e-02 -3.02970371e-02  1.00000000e+00\n  -2.59247175e-02  1.04069392e-02 -3.27933530e-03]\n [ 8.26225394e-01  5.45481157e-01  4.45228936e-01  4.69460892e-01\n   1.62850249e-01 -5.29799878e-02 -1.64828613e-02 -3.69868670e-03\n   7.82661758e-01  3.06635525e-01  2.25269925e-01  2.79216121e-01\n   1.17322780e-01  3.01634242e-01  4.67350737e-04  3.48191097e-03\n  -6.13585453e-02 -4.62522758e-03 -2.00319032e-03 -2.59247175e-02\n   1.00000000e+00  7.63182782e-01 -9.10199105e-03]\n [ 7.26877462e-01  8.30856918e-01  8.07978892e-01  8.32173327e-01\n   6.26010436e-01 -1.83623532e-01 -9.60084754e-02 -3.51223310e-02\n   5.67382457e-01  5.48194626e-01  3.73622030e-01  7.41816852e-01\n   5.04862528e-01  4.96990571e-01 -4.17036022e-02 -2.68349038e-02\n  -1.66850481e-01 -3.57062981e-02 -3.13272789e-02  1.04069392e-02\n   7.63182782e-01  1.00000000e+00 -7.82780499e-02]\n [ 2.95387392e-02  2.63717762e-02 -3.81642341e-02  1.14275445e-02\n  -3.85206434e-02  7.16411460e-01  7.99142640e-01  2.83994158e-01\n   1.87508934e-02 -5.13460289e-02  1.65454238e-02  4.00605723e-03\n  -6.34322392e-02  6.59710703e-03  7.35860200e-01  7.03066646e-01\n   8.35570411e-02  2.98436933e-01  2.58547362e-01 -3.27933530e-03\n  -9.10199105e-03 -7.82780499e-02  1.00000000e+00]]",
          "method": "spearman",
          "pairs": [
            {
              "correlation": 0.991327,
              "feature_a": "log_dv_med_20",
              "feature_b": "dv_med_30"
            },
            {
              "correlation": 0.978755,
              "feature_a": "log_dv_med_20",
              "feature_b": "dv_med_10"
            },
            {
              "correlation": 0.967756,
              "feature_a": "dv_med_10",
              "feature_b": "dv_med_30"
            },
            {
              "correlation": 0.956033,
              "feature_a": "vol_60",
              "feature_b": "vol_90"
            },
            {
              "correlation": 0.937746,
              "feature_a": "sma_ratio_5_30",
              "feature_b": "sma_ratio_10_30"
            },
            {
              "correlation": 0.919925,
              "feature_a": "sma_ratio_15_40",
              "feature_b": "sma_ratio_20_50"
            },
            {
              "correlation": 0.914064,
              "feature_a": "ret_120",
              "feature_b": "price_distance_200ma"
            },
            {
              "correlation": 0.901679,
              "feature_a": "vol_30",
              "feature_b": "vol_60"
            },
            {
              "correlation": 0.90055,
              "feature_a": "ret_20",
              "feature_b": "sma_ratio_5_30"
            },
            {
              "correlation": 0.880562,
              "feature_a": "ret_30",
              "feature_b": "sma_ratio_15_40"
            },
            {
              "correlation": 0.875786,
              "feature_a": "sma_ratio_15_40",
              "feature_b": "sma_ratio_10_30"
            },
            {
              "correlation": 0.857979,
              "feature_a": "ret_30",
              "feature_b": "sma_ratio_5_30"
            },
            {
              "correlation": 0.856637,
              "feature_a": "vol_30",
              "feature_b": "vol_90"
            },
            {
              "correlation": 0.856076,
              "feature_a": "ret_30",
              "feature_b": "sma_ratio_10_30"
            },
            {
              "correlation": 0.855855,
              "feature_a": "ret_20",
              "feature_b": "sma_ratio_10_30"
            },
            {
              "correlation": 0.832173,
              "feature_a": "sma_ratio_5_30",
              "feature_b": "high_low_30_pos"
            },
            {
              "correlation": 0.830857,
              "feature_a": "ret_20",
              "feature_b": "high_low_30_pos"
            },
            {
              "correlation": 0.826225,
              "feature_a": "ret_10",
              "feature_b": "high_low_10_pos"
            },
            {
              "correlation": 0.807979,
              "feature_a": "ret_30",
              "feature_b": "high_low_30_pos"
            },
            {
              "correlation": 0.805618,
              "feature_a": "vol_10",
              "feature_b": "vol_30"
            },
            {
              "correlation": 0.799143,
              "feature_a": "vol_30",
              "feature_b": "normalized_range_20"
            },
            {
              "correlation": 0.787783,
              "feature_a": "ret_30",
              "feature_b": "sma_ratio_20_50"
            },
            {
              "correlation": 0.782662,
              "feature_a": "ret_5",
              "feature_b": "high_low_10_pos"
            },
            {
              "correlation": 0.775713,
              "feature_a": "ret_60",
              "feature_b": "sma_ratio_20_50"
            },
            {
              "correlation": 0.775051,
              "feature_a": "sma_ratio_5_30",
              "feature_b": "sma_ratio_15_40"
            },
            {
              "correlation": 0.772686,
              "feature_a": "ret_20",
              "feature_b": "ret_30"
            },
            {
              "correlation": 0.763892,
              "feature_a": "ret_60",
              "feature_b": "price_distance_200ma"
            },
            {
              "correlation": 0.763183,
              "feature_a": "high_low_10_pos",
              "feature_b": "high_low_30_pos"
            },
            {
              "correlation": 0.741817,
              "feature_a": "sma_ratio_10_30",
              "feature_b": "high_low_30_pos"
            },
            {
              "correlation": 0.73586,
              "feature_a": "vol_60",
              "feature_b": "normalized_range_20"
            },
            {
              "correlation": 0.72789,
              "feature_a": "vol_10",
              "feature_b": "vol_60"
            },
            {
              "correlation": 0.726877,
              "feature_a": "ret_10",
              "feature_b": "high_low_30_pos"
            },
            {
              "correlation": 0.716411,
              "feature_a": "vol_10",
              "feature_b": "normalized_range_20"
            },
            {
              "correlation": 0.703067,
              "feature_a": "vol_90",
              "feature_b": "normalized_range_20"
            },
            {
              "correlation": 0.695139,
              "feature_a": "vol_10",
              "feature_b": "vol_90"
            },
            {
              "correlation": 0.693718,
              "feature_a": "sma_ratio_10_30",
              "feature_b": "sma_ratio_20_50"
            },
            {
              "correlation": 0.688647,
              "feature_a": "ret_20",
              "feature_b": "sma_ratio_15_40"
            },
            {
              "correlation": 0.685898,
              "feature_a": "sma_ratio_15_40",
              "feature_b": "ret_60"
            },
            {
              "correlation": 0.665632,
              "feature_a": "ret_30",
              "feature_b": "ret_60"
            },
            {
              "correlation": 0.64521,
              "feature_a": "ret_10",
              "feature_b": "sma_ratio_5_30"
            },
            {
              "correlation": 0.635543,
              "feature_a": "ret_60",
              "feature_b": "ret_120"
            },
            {
              "correlation": 0.634631,
              "feature_a": "ret_10",
              "feature_b": "ret_20"
            },
            {
              "correlation": 0.632081,
              "feature_a": "sma_ratio_20_50",
              "feature_b": "price_distance_200ma"
            },
            {
              "correlation": 0.62601,
              "feature_a": "sma_ratio_15_40",
              "feature_b": "high_low_30_pos"
            },
            {
              "correlation": 0.608024,
              "feature_a": "sma_ratio_5_30",
              "feature_b": "sma_ratio_20_50"
            },
            {
              "correlation": 0.606984,
              "feature_a": "ret_10",
              "feature_b": "ret_5"
            },
            {
              "correlation": 0.584396,
              "feature_a": "ret_30",
              "feature_b": "price_distance_200ma"
            },
            {
              "correlation": 0.58299,
              "feature_a": "ret_60",
              "feature_b": "sma_ratio_10_30"
            },
            {
              "correlation": 0.581529,
              "feature_a": "sma_ratio_15_40",
              "feature_b": "price_distance_200ma"
            },
            {
              "correlation": 0.578308,
              "feature_a": "sma_ratio_5_30",
              "feature_b": "ret_60"
            },
            {
              "correlation": 0.567382,
              "feature_a": "ret_5",
              "feature_b": "high_low_30_pos"
            },
            {
              "correlation": 0.548195,
              "feature_a": "ret_60",
              "feature_b": "high_low_30_pos"
            },
            {
              "correlation": 0.545481,
              "feature_a": "ret_20",
              "feature_b": "high_low_10_pos"
            },
            {
              "correlation": 0.527873,
              "feature_a": "ret_20",
              "feature_b": "ret_60"
            },
            {
              "correlation": 0.52505,
              "feature_a": "sma_ratio_5_30",
              "feature_b": "price_distance_200ma"
            },
            {
              "correlation": 0.518115,
              "feature_a": "sma_ratio_10_30",
              "feature_b": "price_distance_200ma"
            },
            {
              "correlation": 0.51226,
              "feature_a": "ret_10",
              "feature_b": "ret_30"
            },
            {
              "correlation": 0.511839,
              "feature_a": "vol_10",
              "feature_b": "vol_ratio_10_30"
            },
            {
              "correlation": 0.507029,
              "feature_a": "ret_120",
              "feature_b": "sma_ratio_20_50"
            },
            {
              "correlation": 0.504863,
              "feature_a": "sma_ratio_20_50",
              "feature_b": "high_low_30_pos"
            },
            {
              "correlation": 0.499745,
              "feature_a": "ret_20",
              "feature_b": "sma_ratio_20_50"
            },
            {
              "correlation": 0.496991,
              "feature_a": "price_distance_200ma",
              "feature_b": "high_low_30_pos"
            },
            {
              "correlation": 0.483564,
              "feature_a": "ret_20",
              "feature_b": "price_distance_200ma"
            },
            {
              "correlation": 0.469461,
              "feature_a": "sma_ratio_5_30",
              "feature_b": "high_low_10_pos"
            },
            {
              "correlation": 0.453954,
              "feature_a": "sma_ratio_15_40",
              "feature_b": "ret_120"
            },
            {
              "correlation": 0.448523,
              "feature_a": "ret_30",
              "feature_b": "ret_120"
            },
            {
              "correlation": 0.445229,
              "feature_a": "ret_30",
              "feature_b": "high_low_10_pos"
            },
            {
              "correlation": 0.423844,
              "feature_a": "ret_10",
              "feature_b": "sma_ratio_10_30"
            },
            {
              "correlation": 0.406247,
              "feature_a": "ret_20",
              "feature_b": "ret_5"
            },
            {
              "correlation": 0.397881,
              "feature_a": "sma_ratio_5_30",
              "feature_b": "ret_120"
            },
            {
              "correlation": 0.394887,
              "feature_a": "ret_120",
              "feature_b": "sma_ratio_10_30"
            },
            {
              "correlation": 0.373622,
              "feature_a": "ret_120",
              "feature_b": "high_low_30_pos"
            },
            {
              "correlation": 0.36404,
              "feature_a": "ret_20",
              "feature_b": "ret_120"
            },
            {
              "correlation": 0.339874,
              "feature_a": "ret_10",
              "feature_b": "ret_60"
            },
            {
              "correlation": 0.332645,
              "feature_a": "ret_10",
              "feature_b": "price_distance_200ma"
            },
            {
              "correlation": 0.325754,
              "feature_a": "ret_30",
              "feature_b": "ret_5"
            },
            {
              "correlation": 0.306636,
              "feature_a": "ret_60",
              "feature_b": "high_low_10_pos"
            },
            {
              "correlation": 0.301634,
              "feature_a": "price_distance_200ma",
              "feature_b": "high_low_10_pos"
            },
            {
              "correlation": 0.298437,
              "feature_a": "dv_med_10",
              "feature_b": "normalized_range_20"
            },
            {
              "correlation": 0.286752,
              "feature_a": "vol_30",
              "feature_b": "log_dv_med_20"
            },
            {
              "correlation": 0.283994,
              "feature_a": "log_dv_med_20",
              "feature_b": "normalized_range_20"
            },
            {
              "correlation": 0.282962,
              "feature_a": "vol_10",
              "feature_b": "dv_med_10"
            },
            {
              "correlation": 0.282543,
              "feature_a": "vol_30",
              "feature_b": "dv_med_30"
            },
            {
              "correlation": 0.281557,
              "feature_a": "vol_60",
              "feature_b": "dv_med_30"
            },
            {
              "correlation": 0.281326,
              "feature_a": "vol_30",
              "feature_b": "dv_med_10"
            },
            {
              "correlation": 0.279216,
              "feature_a": "sma_ratio_10_30",
              "feature_b": "high_low_10_pos"
            },
            {
              "correlation": 0.278222,
              "feature_a": "log_dv_med_20",
              "feature_b": "vol_60"
            },
            {
              "correlation": 0.276182,
              "feature_a": "vol_90",
              "feature_b": "dv_med_30"
            },
            {
              "correlation": 0.272627,
              "feature_a": "log_dv_med_20",
              "feature_b": "vol_90"
            },
            {
              "correlation": 0.269677,
              "feature_a": "vol_60",
              "feature_b": "dv_med_10"
            },
            {
              "correlation": 0.267832,
              "feature_a": "sma_ratio_5_30",
              "feature_b": "ret_5"
            },
            {
              "correlation": 0.263829,
              "feature_a": "vol_90",
              "feature_b": "dv_med_10"
            },
            {
              "correlation": 0.258547,
              "feature_a": "dv_med_30",
              "feature_b": "normalized_range_20"
            },
            {
              "correlation": 0.246562,
              "feature_a": "ret_10",
              "feature_b": "ret_120"
            },
            {
              "correlation": 0.242472,
              "feature_a": "vol_10",
              "feature_b": "log_dv_med_20"
            },
            {
              "correlation": 0.236851,
              "feature_a": "ret_10",
              "feature_b": "sma_ratio_15_40"
            },
            {
              "correlation": 0.22527,
              "feature_a": "ret_120",
              "feature_b": "high_low_10_pos"
            },
            {
              "correlation": 0.224684,
              "feature_a": "ret_5",
              "feature_b": "ret_60"
            },
            {
              "correlation": 0.22465,
              "feature_a": "ret_5",
              "feature_b": "price_distance_200ma"
            },
            {
              "correlation": 0.223685,
              "feature_a": "vol_10",
              "feature_b": "dv_med_30"
            },
            {
              "correlation": -0.189357,
              "feature_a": "sma_ratio_10_30",
              "feature_b": "vol_ratio_10_30"
            },
            {
              "correlation": -0.188191,
              "feature_a": "sma_ratio_5_30",
              "feature_b": "vol_ratio_10_30"
            },
            {
              "correlation": 0.184097,
              "feature_a": "vol_ratio_10_30",
              "feature_b": "vol_zscore_20"
            },
            {
              "correlation": -0.183624,
              "feature_a": "vol_10",
              "feature_b": "high_low_30_pos"
            },
            {
              "correlation": -0.183049,
              "feature_a": "ret_20",
              "feature_b": "vol_ratio_10_30"
            },
            {
              "correlation": 0.167327,
              "feature_a": "ret_5",
              "feature_b": "ret_120"
            },
            {
              "correlation": -0.16685,
              "feature_a": "vol_ratio_10_30",
              "feature_b": "high_low_30_pos"
            },
            {
              "correlation": 0.16285,
              "feature_a": "sma_ratio_15_40",
              "feature_b": "high_low_10_pos"
            },
            {
              "correlation": 0.160041,
              "feature_a": "ret_10",
              "feature_b": "sma_ratio_20_50"
            },
            {
              "correlation": -0.151308,
              "feature_a": "vol_10",
              "feature_b": "sma_ratio_10_30"
            },
            {
              "correlation": -0.141118,
              "feature_a": "sma_ratio_5_30",
              "feature_b": "vol_10"
            },
            {
              "correlation": -0.13391,
              "feature_a": "sma_ratio_15_40",
              "feature_b": "vol_10"
            },
            {
              "correlation": -0.1278,
              "feature_a": "ret_30",
              "feature_b": "vol_10"
            },
            {
              "correlation": -0.125038,
              "feature_a": "ret_20",
              "feature_b": "vol_10"
            },
            {
              "correlation": -0.123236,
              "feature_a": "ret_30",
              "feature_b": "vol_ratio_10_30"
            },
            {
              "correlation": 0.121401,
              "feature_a": "ret_5",
              "feature_b": "sma_ratio_10_30"
            },
            {
              "correlation": 0.117323,
              "feature_a": "sma_ratio_20_50",
              "feature_b": "high_low_10_pos"
            },
            {
              "correlation": -0.109331,
              "feature_a": "vol_10",
              "feature_b": "sma_ratio_20_50"
            },
            {
              "correlation": -0.103504,
              "feature_a": "sma_ratio_15_40",
              "feature_b": "vol_ratio_10_30"
            },
            {
              "correlation": -0.102941,
              "feature_a": "ret_10",
              "feature_b": "vol_ratio_10_30"
            },
            {
              "correlation": -0.09981,
              "feature_a": "vol_30",
              "feature_b": "sma_ratio_20_50"
            },
            {
              "correlation": 0.098388,
              "feature_a": "vol_ratio_10_30",
              "feature_b": "dv_med_10"
            },
            {
              "correlation": -0.096008,
              "feature_a": "vol_30",
              "feature_b": "high_low_30_pos"
            },
            {
              "correlation": -0.091616,
              "feature_a": "vol_10",
              "feature_b": "ret_60"
            },
            {
              "correlation": 0.083557,
              "feature_a": "vol_ratio_10_30",
              "feature_b": "normalized_range_20"
            },
            {
              "correlation": -0.082371,
              "feature_a": "sma_ratio_15_40",
              "feature_b": "vol_30"
            },
            {
              "correlation": -0.078278,
              "feature_a": "high_low_30_pos",
              "feature_b": "normalized_range_20"
            },
            {
              "correlation": 0.076223,
              "feature_a": "ret_120",
              "feature_b": "dv_med_10"
            },
            {
              "correlation": 0.075165,
              "feature_a": "price_distance_200ma",
              "feature_b": "dv_med_10"
            },
            {
              "correlation": -0.0711,
              "feature_a": "vol_30",
              "feature_b": "ret_60"
            },
            {
              "correlation": 0.068891,
              "feature_a": "log_dv_med_20",
              "feature_b": "ret_120"
            },
            {
              "correlation": 0.067203,
              "feature_a": "log_dv_med_20",
              "feature_b": "price_distance_200ma"
            },
            {
              "correlation": 0.065607,
              "feature_a": "sma_ratio_15_40",
              "feature_b": "ret_5"
            },
            {
              "correlation": 0.064619,
              "feature_a": "ret_120",
              "feature_b": "dv_med_30"
            },
            {
              "correlation": -0.063931,
              "feature_a": "ret_30",
              "feature_b": "vol_30"
            },
            {
              "correlation": -0.063432,
              "feature_a": "sma_ratio_20_50",
              "feature_b": "normalized_range_20"
            },
            {
              "correlation": 0.062547,
              "feature_a": "vol_10",
              "feature_b": "vol_zscore_20"
            },
            {
              "correlation": 0.062381,
              "feature_a": "price_distance_200ma",
              "feature_b": "dv_med_30"
            },
            {
              "correlation": -0.061359,
              "feature_a": "vol_ratio_10_30",
              "feature_b": "high_low_10_pos"
            },
            {
              "correlation": 0.055107,
              "feature_a": "price_distance_200ma",
              "feature_b": "vol_90"
            },
            {
              "correlation": -0.05298,
              "feature_a": "vol_10",
              "feature_b": "high_low_10_pos"
            },
            {
              "correlation": 0.05219,
              "feature_a": "sma_ratio_15_40",
              "feature_b": "vol_zscore_20"
            },
            {
              "correlation": -0.051346,
              "feature_a": "ret_60",
              "feature_b": "normalized_range_20"
            },
            {
              "correlation": 0.050982,
              "feature_a": "sma_ratio_20_50",
              "feature_b": "vol_zscore_20"
            },
            {
              "correlation": -0.048566,
              "feature_a": "vol_30",
              "feature_b": "vol_zscore_20"
            },
            {
              "correlation": -0.047732,
              "feature_a": "vol_30",
              "feature_b": "sma_ratio_10_30"
            },
            {
              "correlation": -0.046822,
              "feature_a": "vol_60",
              "feature_b": "vol_ratio_10_30"
            },
            {
              "correlation": 0.046414,
              "feature_a": "ret_10",
              "feature_b": "vol_90"
            },
            {
              "correlation": 0.045852,
              "feature_a": "ret_5",
              "feature_b": "sma_ratio_20_50"
            },
            {
              "correlation": -0.045642,
              "feature_a": "ret_10",
              "feature_b": "vol_10"
            },
            {
              "correlation": 0.044893,
              "feature_a": "ret_20",
              "feature_b": "vol_90"
            },
            {
              "correlation": 0.044404,
              "feature_a": "sma_ratio_5_30",
              "feature_b": "vol_90"
            },
            {
              "correlation": 0.043657,
              "feature_a": "ret_120",
              "feature_b": "vol_90"
            },
            {
              "correlation": -0.042948,
              "feature_a": "vol_90",
              "feature_b": "vol_ratio_10_30"
            },
            {
              "correlation": 0.041848,
              "feature_a": "ret_30",
              "feature_b": "vol_90"
            },
            {
              "correlation": -0.041704,
              "feature_a": "vol_60",
              "feature_b": "high_low_30_pos"
            },
            {
              "correlation": -0.041416,
              "feature_a": "vol_60",
              "feature_b": "vol_zscore_20"
            },
            {
              "correlation": 0.040776,
              "feature_a": "sma_ratio_10_30",
              "feature_b": "vol_zscore_20"
            },
            {
              "correlation": 0.040767,
              "feature_a": "ret_10",
              "feature_b": "vol_60"
            },
            {
              "correlation": 0.040375,
              "feature_a": "sma_ratio_10_30",
              "feature_b": "vol_90"
            },
            {
              "correlation": 0.040301,
              "feature_a": "ret_5",
              "feature_b": "vol_90"
            },
            {
              "correlation": -0.039145,
              "feature_a": "vol_90",
              "feature_b": "vol_zscore_20"
            },
            {
              "correlation": 0.038569,
              "feature_a": "ret_5",
              "feature_b": "vol_60"
            },
            {
              "correlation": -0.038521,
              "feature_a": "sma_ratio_15_40",
              "feature_b": "normalized_range_20"
            },
            {
              "correlation": -0.038495,
              "feature_a": "ret_60",
              "feature_b": "vol_ratio_10_30"
            },
            {
              "correlation": -0.038164,
              "feature_a": "ret_30",
              "feature_b": "normalized_range_20"
            },
            {
              "correlation": -0.038101,
              "feature_a": "vol_30",
              "feature_b": "vol_ratio_10_30"
            },
            {
              "correlation": -0.036814,
              "feature_a": "sma_ratio_5_30",
              "feature_b": "vol_30"
            },
            {
              "correlation": -0.036669,
              "feature_a": "vol_10",
              "feature_b": "price_distance_200ma"
            },
            {
              "correlation": 0.035912,
              "feature_a": "sma_ratio_15_40",
              "feature_b": "vol_90"
            },
            {
              "correlation": -0.035706,
              "feature_a": "dv_med_10",
              "feature_b": "high_low_30_pos"
            },
            {
              "correlation": -0.035122,
              "feature_a": "log_dv_med_20",
              "feature_b": "high_low_30_pos"
            },
            {
              "correlation": -0.034686,
              "feature_a": "price_distance_200ma",
              "feature_b": "vol_ratio_10_30"
            },
            {
              "correlation": 0.034144,
              "feature_a": "ret_60",
              "feature_b": "vol_90"
            },
            {
              "correlation": 0.034081,
              "feature_a": "ret_30",
              "feature_b": "vol_zscore_20"
            },
            {
              "correlation": 0.033061,
              "feature_a": "ret_60",
              "feature_b": "vol_zscore_20"
            },
            {
              "correlation": 0.03135,
              "feature_a": "sma_ratio_20_50",
              "feature_b": "vol_90"
            },
            {
              "correlation": -0.031327,
              "feature_a": "dv_med_30",
              "feature_b": "high_low_30_pos"
            },
            {
              "correlation": 0.031042,
              "feature_a": "ret_120",
              "feature_b": "vol_60"
            },
            {
              "correlation": -0.03096,
              "feature_a": "sma_ratio_20_50",
              "feature_b": "vol_ratio_10_30"
            },
            {
              "correlation": 0.030387,
              "feature_a": "price_distance_200ma",
              "feature_b": "vol_zscore_20"
            },
            {
              "correlation": -0.030297,
              "feature_a": "dv_med_30",
              "feature_b": "vol_zscore_20"
            },
            {
              "correlation": 0.030016,
              "feature_a": "sma_ratio_5_30",
              "feature_b": "vol_zscore_20"
            },
            {
              "correlation": -0.029873,
              "feature_a": "log_dv_med_20",
              "feature_b": "vol_zscore_20"
            },
            {
              "correlation": 0.029539,
              "feature_a": "ret_10",
              "feature_b": "normalized_range_20"
            },
            {
              "correlation": 0.02918,
              "feature_a": "price_distance_200ma",
              "feature_b": "vol_60"
            },
            {
              "correlation": 0.028589,
              "feature_a": "vol_30",
              "feature_b": "ret_5"
            },
            {
              "correlation": 0.02751,
              "feature_a": "ret_20",
              "feature_b": "vol_60"
            },
            {
              "correlation": -0.027065,
              "feature_a": "ret_30",
              "feature_b": "dv_med_30"
            },
            {
              "correlation": -0.026835,
              "feature_a": "vol_90",
              "feature_b": "high_low_30_pos"
            },
            {
              "correlation": -0.026504,
              "feature_a": "ret_30",
              "feature_b": "log_dv_med_20"
            },
            {
              "correlation": 0.026372,
              "feature_a": "ret_20",
              "feature_b": "normalized_range_20"
            },
            {
              "correlation": -0.025925,
              "feature_a": "vol_zscore_20",
              "feature_b": "high_low_10_pos"
            },
            {
              "correlation": -0.025669,
              "feature_a": "ret_120",
              "feature_b": "vol_ratio_10_30"
            },
            {
              "correlation": -0.024261,
              "feature_a": "sma_ratio_5_30",
              "feature_b": "log_dv_med_20"
            },
            {
              "correlation": -0.023783,
              "feature_a": "sma_ratio_15_40",
              "feature_b": "dv_med_30"
            },
            {
              "correlation": -0.02335,
              "feature_a": "ret_20",
              "feature_b": "vol_30"
            },
            {
              "correlation": 0.023147,
              "feature_a": "sma_ratio_5_30",
              "feature_b": "vol_60"
            },
            {
              "correlation": -0.023015,
              "feature_a": "sma_ratio_5_30",
              "feature_b": "dv_med_30"
            },
            {
              "correlation": 0.022139,
              "feature_a": "dv_med_10",
              "feature_b": "vol_zscore_20"
            },
            {
              "correlation": -0.022092,
              "feature_a": "log_dv_med_20",
              "feature_b": "sma_ratio_10_30"
            },
            {
              "correlation": -0.021818,
              "feature_a": "sma_ratio_5_30",
              "feature_b": "dv_med_10"
            },
            {
              "correlation": -0.021674,
              "feature_a": "sma_ratio_10_30",
              "feature_b": "dv_med_30"
            },
            {
              "correlation": -0.021336,
              "feature_a": "ret_20",
              "feature_b": "log_dv_med_20"
            },
            {
              "correlation": -0.021052,
              "feature_a": "sma_ratio_20_50",
              "feature_b": "dv_med_30"
            },
            {
              "correlation": -0.020907,
              "feature_a": "sma_ratio_15_40",
              "feature_b": "log_dv_med_20"
            },
            {
              "correlation": -0.020797,
              "feature_a": "ret_30",
              "feature_b": "dv_med_10"
            },
            {
              "correlation": -0.019654,
              "feature_a": "ret_20",
              "feature_b": "dv_med_30"
            },
            {
              "correlation": -0.019628,
              "feature_a": "ret_5",
              "feature_b": "vol_zscore_20"
            },
            {
              "correlation": -0.019602,
              "feature_a": "ret_60",
              "feature_b": "vol_60"
            },
            {
              "correlation": -0.01909,
              "feature_a": "ret_20",
              "feature_b": "dv_med_10"
            },
            {
              "correlation": 0.018751,
              "feature_a": "ret_5",
              "feature_b": "normalized_range_20"
            },
            {
              "correlation": -0.018747,
              "feature_a": "sma_ratio_10_30",
              "feature_b": "dv_med_10"
            },
            {
              "correlation": 0.018647,
              "feature_a": "ret_20",
              "feature_b": "vol_zscore_20"
            },
            {
              "correlation": 0.017343,
              "feature_a": "vol_10",
              "feature_b": "ret_5"
            },
            {
              "correlation": 0.01687,
              "feature_a": "sma_ratio_10_30",
              "feature_b": "vol_60"
            },
            {
              "correlation": 0.016814,
              "feature_a": "ret_120",
              "feature_b": "vol_zscore_20"
            },
            {
              "correlation": 0.016545,
              "feature_a": "ret_120",
              "feature_b": "normalized_range_20"
            },
            {
              "correlation": -0.016483,
              "feature_a": "vol_30",
              "feature_b": "high_low_10_pos"
            },
            {
              "correlation": -0.015639,
              "feature_a": "vol_10",
              "feature_b": "ret_120"
            },
            {
              "correlation": -0.015368,
              "feature_a": "log_dv_med_20",
              "feature_b": "sma_ratio_20_50"
            },
            {
              "correlation": 0.015026,
              "feature_a": "log_dv_med_20",
              "feature_b": "vol_ratio_10_30"
            },
            {
              "correlation": 0.014951,
              "feature_a": "ret_10",
              "feature_b": "vol_30"
            },
            {
              "correlation": 0.014466,
              "feature_a": "vol_30",
              "feature_b": "ret_120"
            },
            {
              "correlation": -0.013801,
              "feature_a": "vol_ratio_10_30",
              "feature_b": "dv_med_30"
            },
            {
              "correlation": 0.01291,
              "feature_a": "ret_30",
              "feature_b": "vol_60"
            },
            {
              "correlation": -0.01246,
              "feature_a": "ret_10",
              "feature_b": "dv_med_10"
            },
            {
              "correlation": -0.012008,
              "feature_a": "sma_ratio_15_40",
              "feature_b": "dv_med_10"
            },
            {
              "correlation": 0.011428,
              "feature_a": "sma_ratio_5_30",
              "feature_b": "normalized_range_20"
            },
            {
              "correlation": -0.011285,
              "feature_a": "ret_10",
              "feature_b": "log_dv_med_20"
            },
            {
              "correlation": -0.010701,
              "feature_a": "sma_ratio_20_50",
              "feature_b": "vol_60"
            },
            {
              "correlation": 0.010407,
              "feature_a": "vol_zscore_20",
              "feature_b": "high_low_30_pos"
            },
            {
              "correlation": 0.010295,
              "feature_a": "ret_60",
              "feature_b": "dv_med_10"
            },
            {
              "correlation": -0.009163,
              "feature_a": "ret_10",
              "feature_b": "dv_med_30"
            },
            {
              "correlation": -0.009102,
              "feature_a": "high_low_10_pos",
              "feature_b": "normalized_range_20"
            },
            {
              "correlation": -0.008596,
              "feature_a": "ret_5",
              "feature_b": "vol_ratio_10_30"
            },
            {
              "correlation": -0.006769,
              "feature_a": "ret_10",
              "feature_b": "vol_zscore_20"
            },
            {
              "correlation": 0.006597,
              "feature_a": "price_distance_200ma",
              "feature_b": "normalized_range_20"
            },
            {
              "correlation": -0.004823,
              "feature_a": "log_dv_med_20",
              "feature_b": "ret_5"
            },
            {
              "correlation": -0.004625,
              "feature_a": "dv_med_10",
              "feature_b": "high_low_10_pos"
            },
            {
              "correlation": -0.004248,
              "feature_a": "vol_30",
              "feature_b": "price_distance_200ma"
            },
            {
              "correlation": -0.004008,
              "feature_a": "sma_ratio_20_50",
              "feature_b": "dv_med_10"
            },
            {
              "correlation": 0.004006,
              "feature_a": "sma_ratio_10_30",
              "feature_b": "normalized_range_20"
            },
            {
              "correlation": -0.003699,
              "feature_a": "log_dv_med_20",
              "feature_b": "high_low_10_pos"
            },
            {
              "correlation": 0.003482,
              "feature_a": "vol_90",
              "feature_b": "high_low_10_pos"
            },
            {
              "correlation": -0.003359,
              "feature_a": "ret_5",
              "feature_b": "dv_med_30"
            },
            {
              "correlation": -0.003279,
              "feature_a": "vol_zscore_20",
              "feature_b": "normalized_range_20"
            },
            {
              "correlation": 0.002622,
              "feature_a": "log_dv_med_20",
              "feature_b": "ret_60"
            },
            {
              "correlation": -0.002548,
              "feature_a": "ret_5",
              "feature_b": "dv_med_10"
            },
            {
              "correlation": 0.002418,
              "feature_a": "sma_ratio_15_40",
              "feature_b": "vol_60"
            },
            {
              "correlation": -0.002215,
              "feature_a": "ret_60",
              "feature_b": "dv_med_30"
            },
            {
              "correlation": -0.002003,
              "feature_a": "dv_med_30",
              "feature_b": "high_low_10_pos"
            },
            {
              "correlation": 0.000467,
              "feature_a": "vol_60",
              "feature_b": "high_low_10_pos"
            }
          ],
          "rows_used": 44079
        },
        "train_rows": 44079
      }
    }
  },
  "redundancy": {
    "FS-001": {
      "duplicates": [],
      "high_correlation_pairs": []
    },
    "FS-002": {
      "duplicates": [],
      "high_correlation_pairs": [
        {
          "feature_a": "dv_med_10",
          "feature_b": "dv_med_30",
          "pearson": 0.979203,
          "spearman": 0.967756
        },
        {
          "feature_a": "vol_60",
          "feature_b": "vol_90",
          "pearson": 0.960375,
          "spearman": 0.956033
        }
      ]
    },
    "FS-003": {
      "duplicates": [],
      "high_correlation_pairs": [
        {
          "feature_a": "dv_med_10",
          "feature_b": "dv_med_30",
          "pearson": 0.979203,
          "spearman": 0.967756
        },
        {
          "feature_a": "dv_med_10",
          "feature_b": "log_dv_med_20",
          "pearson": 0.986317,
          "spearman": 0.978755
        },
        {
          "feature_a": "dv_med_30",
          "feature_b": "log_dv_med_20",
          "pearson": 0.994678,
          "spearman": 0.991327
        },
        {
          "feature_a": "vol_60",
          "feature_b": "vol_90",
          "pearson": 0.960375,
          "spearman": 0.956033
        }
      ]
    }
  },
  "scope": "train split only (never test)"
}
```

## 5. Ablation results

One row per experiment (null/failed runs are never hidden).

- **EXP-10001** FS-001 (base ) ridge: status completed, OOS IC -0.000317924774225456, rank IC 0.00032585905260486796, after-cost return 0.4931642376486711, turnover 545.2741115069538
- **EXP-10002** FS-001 (base ) lasso: status completed, OOS IC 0.022560249298736096, rank IC 0.01224241776733129, after-cost return 0.21139672978059254, turnover 313.1856083612911
- **EXP-10003** FS-001 (base ) random_forest: status completed, OOS IC 0.011902891890009919, rank IC 0.016793121835003033, after-cost return 1.2713406209384774, turnover 968.6418009253501
- **EXP-10004** FS-001 (base ) xgboost: status completed, OOS IC 0.011344954300343671, rank IC 0.009512398527050063, after-cost return 0.23656657984663987, turnover 689.7041367906257
- **EXP-10005** FS-002 (new ) ridge: status completed, OOS IC 0.014777404277975676, rank IC 0.012261114598218457, after-cost return 0.24007647593541503, turnover 529.4374539590915
- **EXP-10006** FS-002 (new ) lasso: status completed, OOS IC 0.025559370169658358, rank IC 0.027003565752747764, after-cost return 2.0186184524177304, turnover 679.5962792970425
- **EXP-10007** FS-002 (new ) random_forest: status completed, OOS IC 0.0013213168956116915, rank IC 0.013771773162629682, after-cost return 1.4658079421190946, turnover 729.7811076046074
- **EXP-10008** FS-002 (new ) xgboost: status completed, OOS IC -0.0020974629587495373, rank IC 0.001750231518274598, after-cost return 0.5707809148244876, turnover 803.1785743844632
- **EXP-10009** FS-003 (all ) ridge: status completed, OOS IC 0.0201763200533262, rank IC 0.014754470545813914, after-cost return 0.07546265191640522, turnover 507.757581572939
- **EXP-10010** FS-003 (all ) lasso: status completed, OOS IC 0.025422594749713836, rank IC 0.026692397067268527, after-cost return 2.004125242856636, turnover 686.5670303991968
- **EXP-10011** FS-003 (all ) random_forest: status completed, OOS IC -0.00558237822083588, rank IC 0.009440380625299371, after-cost return 0.9491687902317518, turnover 750.5412919283436
- **EXP-10012** FS-003 (all ) xgboost: status completed, OOS IC 0.009960494891522785, rank IC 0.012478812190268893, after-cost return 1.1656841294013245, turnover 866.2085966500129
- **EXP-10013** FS-004 (base_plus_family momentum) ridge: status completed, OOS IC 0.008166089777130937, rank IC 0.006660078259592141, after-cost return 0.6602297783282343, turnover 669.2814032826151
- **EXP-10014** FS-004 (base_plus_family momentum) lasso: status completed, OOS IC 0.007328748238197995, rank IC 0.006541219834666596, after-cost return 0.5946174711336203, turnover 742.7141911214937
- **EXP-10015** FS-004 (base_plus_family momentum) random_forest: status completed, OOS IC 0.005525459126815919, rank IC 0.013404692303425796, after-cost return 1.032839469233544, turnover 891.0546600941824
- **EXP-10016** FS-004 (base_plus_family momentum) xgboost: status completed, OOS IC 0.013096534046500491, rank IC 0.015696189463651387, after-cost return 0.6336201171892213, turnover 817.7522238383899
- **EXP-10017** FS-005 (base_plus_family trend) ridge: status completed, OOS IC 0.0009302663413488234, rank IC 0.0010323321625555883, after-cost return 0.6007505852040649, turnover 520.1107069150971
- **EXP-10018** FS-005 (base_plus_family trend) lasso: status completed, OOS IC 0.023784120348947568, rank IC 0.02130370330800358, after-cost return 0.24691720767765712, turnover 322.93108971208176
- **EXP-10019** FS-005 (base_plus_family trend) random_forest: status completed, OOS IC -0.008110452736997158, rank IC 0.005332922017330818, after-cost return 0.6117830087760083, turnover 721.310211164077
- **EXP-10020** FS-005 (base_plus_family trend) xgboost: status completed, OOS IC 0.004173663442419529, rank IC 0.015172420495095349, after-cost return 0.27864851037717786, turnover 674.3499012255594
- **EXP-10021** FS-006 (base_plus_family volatility) ridge: status completed, OOS IC 0.012084560542571607, rank IC 0.015307362544905778, after-cost return 0.28880963739298093, turnover 469.4661543195445
- **EXP-10022** FS-006 (base_plus_family volatility) lasso: status completed, OOS IC 0.035758651568641026, rank IC 0.041206479787390324, after-cost return 0.9314074483572565, turnover 305.508216603876
- **EXP-10023** FS-006 (base_plus_family volatility) random_forest: status completed, OOS IC 0.00575621096792073, rank IC 0.027546624799795482, after-cost return 0.8394608414325964, turnover 699.5049447851641
- **EXP-10024** FS-006 (base_plus_family volatility) xgboost: status completed, OOS IC 0.009158726476298262, rank IC 0.012097811326176697, after-cost return 0.30377037550048525, turnover 656.1183374569682
- **EXP-10025** FS-007 (base_plus_family volume) ridge: status completed, OOS IC 0.007940629873649354, rank IC 0.011071194861042483, after-cost return 0.09727614584533106, turnover 428.8724072382642
- **EXP-10026** FS-007 (base_plus_family volume) lasso: status completed, OOS IC 0.023784120348947568, rank IC 0.02130370330800358, after-cost return 0.24691720767765712, turnover 322.93108971208176
- **EXP-10027** FS-007 (base_plus_family volume) random_forest: status completed, OOS IC 0.0026988101597204177, rank IC 0.015923720071459164, after-cost return 0.9372316246497265, turnover 913.6659041955423
- **EXP-10028** FS-007 (base_plus_family volume) xgboost: status completed, OOS IC 0.018025191029460468, rank IC 0.01035441661693123, after-cost return 0.4325883614966397, turnover 710.6544897856108
- **EXP-10029** FS-008 (base_plus_family range) ridge: status completed, OOS IC 0.0022547949878993882, rank IC 0.0008066347039891016, after-cost return 0.49880863862715885, turnover 500.2009779890835
- **EXP-10030** FS-008 (base_plus_family range) lasso: status completed, OOS IC 0.021962574891466503, rank IC 0.01777133775824998, after-cost return 1.5377855768553044, turnover 640.6036903973127
- **EXP-10031** FS-008 (base_plus_family range) random_forest: status completed, OOS IC 0.0006220037402438563, rank IC 0.004587500906054648, after-cost return 0.3838541930131778, turnover 614.947245390307
- **EXP-10032** FS-008 (base_plus_family range) xgboost: status completed, OOS IC 0.01715185911653009, rank IC 0.01724500694863711, after-cost return 0.560058155131173, turnover 699.6821542512296
- **EXP-10033** FS-009 (all_minus_family momentum) ridge: status completed, OOS IC 0.01671705969594199, rank IC 0.01645321118070487, after-cost return 0.18786508826981851, turnover 415.81272186221463
- **EXP-10034** FS-009 (all_minus_family momentum) lasso: status completed, OOS IC 0.03121509953906618, rank IC 0.02704630136620414, after-cost return 2.2120590363083537, turnover 508.52213215732223
- **EXP-10035** FS-009 (all_minus_family momentum) random_forest: status completed, OOS IC -0.0047825995368475415, rank IC 0.02555221903186185, after-cost return 1.4103140830874001, turnover 777.9939986799974
- **EXP-10036** FS-009 (all_minus_family momentum) xgboost: status completed, OOS IC 0.006207995440991667, rank IC 0.011797806284896304, after-cost return 0.09277022453881578, turnover 609.9191111538272
- **EXP-10037** FS-010 (all_minus_family trend) ridge: status completed, OOS IC 0.019640904786904098, rank IC 0.019653040238251044, after-cost return 0.6622171571926079, turnover 636.2487945161439
- **EXP-10038** FS-010 (all_minus_family trend) lasso: status completed, OOS IC 0.02542259474971383, rank IC 0.026692397067268527, after-cost return 2.004125242856636, turnover 686.5670303991968
- **EXP-10039** FS-010 (all_minus_family trend) random_forest: status completed, OOS IC 0.0021501612337400545, rank IC 0.018165560928009073, after-cost return 0.6388782799517316, turnover 653.723414515517
- **EXP-10040** FS-010 (all_minus_family trend) xgboost: status completed, OOS IC 0.025202031761366813, rank IC 0.017147960279068192, after-cost return 1.0047517193289246, turnover 874.6112459602309
- **EXP-10041** FS-011 (all_minus_family volatility) ridge: status completed, OOS IC 0.016326332415433145, rank IC 0.00993870110444851, after-cost return 0.5775877537304173, turnover 606.0535489783049
- **EXP-10042** FS-011 (all_minus_family volatility) lasso: status completed, OOS IC 0.016852571649961023, rank IC 0.013966532672711975, after-cost return 0.9747916527612979, turnover 708.2474732529828
- **EXP-10043** FS-011 (all_minus_family volatility) random_forest: status completed, OOS IC -0.012789562943303375, rank IC 0.0009031478611191903, after-cost return 0.5573699311108093, turnover 725.801780132017
- **EXP-10044** FS-011 (all_minus_family volatility) xgboost: status completed, OOS IC 0.01684471135887419, rank IC 0.003800899091061232, after-cost return 1.6361548644593227, turnover 1146.8322550825003
- **EXP-10045** FS-012 (all_minus_family volume) ridge: status completed, OOS IC 0.01570852256367986, rank IC 0.008531096836229114, after-cost return 0.17569255566973796, turnover 458.027859715799
- **EXP-10046** FS-012 (all_minus_family volume) lasso: status completed, OOS IC 0.025422594748887608, rank IC 0.026692397067268527, after-cost return 2.004125242856636, turnover 686.5670303991968
- **EXP-10047** FS-012 (all_minus_family volume) random_forest: status completed, OOS IC -0.00643641593903718, rank IC 0.005983515658873073, after-cost return 1.1664554607021644, turnover 667.1555629371629
- **EXP-10048** FS-012 (all_minus_family volume) xgboost: status completed, OOS IC 0.01423267670372257, rank IC 0.011007711210734697, after-cost return 0.5110588562548248, turnover 612.2972879987344
- **EXP-10049** FS-013 (all_minus_family range) ridge: status completed, OOS IC 0.018974945067122134, rank IC 0.017259845884693974, after-cost return 0.3929769825077343, turnover 616.678420513594
- **EXP-10050** FS-013 (all_minus_family range) lasso: status completed, OOS IC 0.02358281699291082, rank IC 0.02572951027657954, after-cost return 1.6096628078748902, turnover 712.39993243968
- **EXP-10051** FS-013 (all_minus_family range) random_forest: status completed, OOS IC -0.0029312727159835515, rank IC 0.009800995150183643, after-cost return 0.7527800256869019, turnover 670.3312854918614
- **EXP-10052** FS-013 (all_minus_family range) xgboost: status completed, OOS IC 0.007028651772787969, rank IC 0.014428469831517523, after-cost return 2.4116031590467366, turnover 1092.5422780759318

## 6. Scientific conclusion

See the permanent status report (PHASE_10_STATUS.md) for the final verdict and the independent reviews.
