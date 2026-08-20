# ORBIT Phase 9: Baseline ML Benchmark

Version 1.0 - 20 August 2026 - Phase 9

## 1. Purpose

Phase 9 is the first falsifiable machine-learning benchmark of the ORBIT
research operating system. It answers a single question under the Phase 4-7
gates:

> Do simple ML models, trained on point-in-time features, add value after
> documented trading costs relative to the Phase 8 documented controls,
> evaluated on a locked out-of-sample window?

The benchmark is deliberately small and strict: 20 pre-registered ML grid
points, 18 pre-registered controls, one locked train/validation/test
chronological split, register-before-run experiments (Phase 6), one cost
model (CM-001), the canonical Phase 7 backtester, and a permanent parquet +
markdown report. Nothing about the design allows post-hoc tuning: an
unregistered hyperparameter set cannot even be validated, let alone run.

## 2. Data and Universe

- Dataset snapshot: **DS-000004** (20-symbol development universe,
  `data/normalized/market/yahoo_chart_api/DS-000004/`).
- Bars: 1996-01-02..2026-08-14, 7,705 sessions, 139,961 daily bars,
  price basis `split_continuous_stored`.
- No SPY series is present in DS-000004, so the SPY-benchmarked labels
  (LAB-001/LAB-003) cannot be resolved. The benchmark therefore uses
  **LAB-004 v1** (registered in Phase 9): 5-session forward TOTAL return,
  DECISION_INSTANT anchor, SIMPLE_TOTAL_RETURN computation - the same
  horizon and anchor as LAB-001 minus the market benchmark. This is the
  documented data-sufficiency limitation of the Phase 9 comparison
  (see Section 9).

## 3. Features (FS-001 v1)

Eight point-in-time numerics, all strict-boundary: a feature row at decision
session D uses bars with session <= D-1 only (the bar of D is unavailable at
close(D) by the Phase 4 convention, `session_close_utc`).

| FEAT | Name | Definition (bars through D-1) |
|------|------|-------------------------------|
| FEAT-001 | ret_10 | 10-session forward-looking-absent simple return, close(D-1)/close(D-11) - 1 |
| FEAT-002 | ret_20 | close(D-1)/close(D-21) - 1 |
| FEAT-003 | ret_30 | close(D-1)/close(D-31) - 1 |
| FEAT-004 | sma_ratio_5_30 | SMA5/SMA30 - 1 |
| FEAT-005 | sma_ratio_15_40 | SMA15/SMA40 - 1 |
| FEAT-006 | vol_10 | 10-session return volatility |
| FEAT-007 | vol_30 | 30-session return volatility |
| FEAT-008 | log_dv_med_20 | log of the 20-session median dollar volume |

The feature snapshot (139,161 rows on the full data) carries a sha256 digest;
`assert_features_point_in_time` verifies every row's window end is strictly
before its decision session. A digest-verified cache lives at
`data/cache/phase9_snapshots/`.

## 4. Label (LAB-004 v1)

- Contract digest pinned at registration (via `contract.content_hash()`).
- Entry session = D-1, outcome = total return over sessions D..D+4.
- 139,081 of 139,161 rows available; 80 unavailable with reason
  `insufficient_future_data` (end of data) - reported, never imputed.
- Logistic regression target: `outcome_value > 0`.

## 5. Splits and Leakage Policy

Locked protocol `fixed_chronological_v1` (windows are part of the experiment
identity):

| Split | Range |
|-------|-------|
| train | 2010-01-04 .. 2018-12-31 |
| validation | 2019-01-02 .. 2021-12-31 |
| test | 2022-01-03 .. 2026-06-30 |

- `embargo_days = 0`, `purge_days = 0`; the exact outcome-window purge at
  the boundaries guarantees no train label overlaps the validation window
  and no validation label overlaps the test window. The test split is never
  purged.
- Dataset sizes on the full data: train 44,399 / validation 15,060 / test
  22,520 rows; the test window is exactly 2022-01-03..2026-06-30.
- No random split path exists. Rows outside all windows are dropped.
- Calibration (Platt) is fitted on validation scores ONLY;
  `assert_no_test_fit` refuses anything else and the audit flags it.

## 6. Models and Grids (pre-registered, locked)

Seed 42 for every run (a different seed is a different experiment). 20 ML
grid points total:

| Family | Grid |
|--------|------|
| ridge | alpha in {0.01, 0.1, 1.0, 10.0} |
| lasso | alpha in {0.0001, 0.001, 0.01, 0.1} |
| logistic | C in {0.01, 0.1, 1.0, 10.0} |
| random_forest | (n_estimators, max_depth) in {50, 200} x {3, 6} |
| xgboost | (n_estimators, max_depth, lr=0.1) in {50, 200} x {3, 6} |

- Linear families are standardized with a StandardScaler fit on the TRAIN
  split only; trees are trained on raw values.
- Logistic predictions are probabilities (`predict_proba[:, 1]`).
- Deterministic experiment ids: ML EXP-90001..EXP-90020, controls
  EXP-90021+ (canonical family/grid order). `register_ml_experiment`
  validates the grid point BEFORE any training is allowed.

## 7. Controls (Phase 8 documented rules on real bars)

Every Phase 8 strategy is executed on the real DS-000004 bars through the
identical Phase 7 path and cost model. Two documented adaptations, both
strictly conservative for the comparison:

1. Strict point-in-time boundary identical to FS-001 (controls never see
   information the models do not see).
2. WEIGHT sizing (top-1 -> 1.0, equal weight -> 1/n) instead of the Phase 8
   nominal-share quantities, so costs are comparable.
3. buy_and_hold enters with weight 0.99 (documented): at the canonical
   no-implicit-loan rule a weight-1.0 market-on-open buy can be rejected
   when the open gaps up beyond the CM-001 cost buffer, leaving the control
   degenerate (zero fills); the 1% residual absorbs entry costs and stays
   in cash for the full window.

16 control runs: buy_and_hold, equal_weight, momentum (10/20/30),
mean_reversion (10/20/30), moving_average (5/30, 10/30, 15/40),
volatility_targeted (0.10/10, 0.15/30, 0.20/60), random_null (seed 42),
null_flat. Experiment ids EXP-90021..EXP-90036.

## 8. Evaluation and Backtest

Per-run metrics (on the test split only):

- OOS IC / rank IC: per-session Pearson/Spearman mean, cross-sections with
  < 3 valid rows or zero variance skipped and counted.
- Calibration: ECE (10 uniform bins) and Brier on calibrated probabilities.
- MSE and hit rate on the uncalibrated predictions.

Backtest: canonical Phase 7 event-driven engine, WEIGHT sizing, long-only,
execution at open with delay 1, partial fills, order expiry 5 sessions,
cost model **CM-001** (spread 2 bps, fees 1 bps, slippage 2 bps) - identical
for ML and controls. Signals are complete per session (long 1/k for top-k,
flat 0.0 otherwise) so rebalance sells are priced. `BacktestManifest`
pins the ml code hash and the full lineage.

## 9. Verdict Rules and Limitation

The verdict is determined by the permanent report rows, not by prose:

- **incremental value**: an ML run with after-cost return and risk-adjusted
  metrics above the best control, reproduced in the audit double-run.
- **defensible null**: ML results indistinguishable from or below controls
  and the null_flat baseline, with the ICs near zero as expected under no
  signal.
- **inconclusive**: infrastructure failure, registration failure, or any
  audit check failing.

Data-sufficiency limitation (documented): without a market benchmark series
in DS-000004, LAB-001/LAB-003 excess-return labels cannot be resolved, so
the Phase 9 comparison is absolute (after-cost total return) rather than
benchmark-relative. The results therefore speak to "does the ML signal add
value" - not "does it add value over holding SPY".

## 10. Audit and Reproducibility

`run_phase9_audit` performs the independent checks: feature point-in-time
validity, feature-scope guard (no Phase 10 feature zoo), label contract
identity and availability, split integrity (exact purge assertion), test
lock, calibration fitted on validation only, grid lock, model-scope guard
(no Phase 22+ families), ranking determinism, backtest accounting
invariants, registry lineage, and data-expansion guard (no silent data
acquisition). The audit runs in the benchmark runner (snapshot + dataset
inputs) and as a full deep-input audit over real artifacts per model family
(`scripts/phase9_audit_deep.py`, results in
`benchmarks/phase9_audit_results.json`); its findings are part of the
status report. Reproducibility is tested directly - identical code + data
must produce bitwise-identical predictions, signals, and backtest results -
and independently re-verified by `scripts/phase9_review2_reproducibility.py`,
which recomputes representative ML and control experiments from scratch and
compares artifacts byte-for-byte against the stored benchmark
(`benchmarks/phase9_review2_results.json`).
`ml_code_hash`/`phase9_config_hash` pin the code and config at run start.

## 11. References

- Benchmark runner: `scripts/phase9_run_all.py`
- Deep-input independent audit (Review 1): `scripts/phase9_audit_deep.py`
- Reproducibility double-run (Review 2): `scripts/phase9_review2_reproducibility.py`
- Implementation: `src/orbit/ml/` (features, labels, splits, grids, models,
  dataset, ranking, calibration, metrics, signals, registry, baselines,
  report, audit, snapshot_cache)
- Tests: `tests/test_phase9_*.py` (130 tests incl. 24 adversarial scenarios)
- Permanent results: `benchmarks/phase9_ml_benchmark.parquet` and
  `benchmarks/phase9_ml_benchmark.md`
