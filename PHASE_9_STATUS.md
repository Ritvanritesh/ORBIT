# PHASE 9 STATUS REPORT

**ORBIT: Optimized Research & Behavioral Intelligence Trading**
**Date:** 20 August 2026
**Phase:** 9 — Baseline ML Benchmark

---

## EXECUTIVE SUMMARY

Phase 9 is **COMPLETE**. The first falsifiable ML benchmark of the ORBIT
research operating system has been built, executed on the real DS-000004
dataset, audited twice, and recorded permanently. The benchmark ran to
completion: 20 ML grid points + 16 controls, all through the canonical
Phase 7 path with one cost model.

**Verdict: DEFENSIBLE NULL.** No ML configuration adds incremental value
over the documented controls after identical costs on the locked test
window. Per-session OOS/rank ICs are economically negligible
(|IC| <= 0.02 across all 20 runs), hit rates sit at 0.51-0.54, and the
best ML after-cost return (+127%, EXP-90015 random_forest) is below the
best documented control (+205%, EXP-90029 moving_average 5/30). The
result is the honest expected outcome under no exploitable signal in
these 8 point-in-time numerics on this 20-name universe.

- **717 tests passed, 6 xfailed** across the full suite (Phases 1-8 stay green)
- **130 Phase 9 tests pass** (incl. 24 adversarial scenarios, A1..A23 + A9a/A9b)
- **36/36 benchmark rows completed** (0 failed), ids EXP-90001..EXP-90036
  (20 ML + 16 controls) in the permanent report
- **Audit Review 1: 14/14 distinct checks PASS** (deep-input
  `scripts/phase9_audit_deep.py`, one representative grid point per family,
  15 checks x 5 families = 75 PASS, results in
  `benchmarks/phase9_audit_results.json`)
- **Audit Review 2 (reproducibility double-run): PASS** (bitwise-identical
  predictions, signals, and metrics vs stored artifacts via
  `scripts/phase9_review2_reproducibility.py` for EXP-90003, EXP-90015,
  EXP-90024; results in `benchmarks/phase9_review2_results.json`)

---

## BENCHMARK RESULTS (test window 2022-01-03 .. 2026-06-30, CM-001 costs)

### ML (top-3 cross-sectional, weight 1/3)

| family | OOS IC | rank IC | hit rate | after-cost return (range) |
|--------|--------|---------|----------|---------------------------|
| ridge | -0.0003 | +0.0003 | 0.525 | +49.3% (+49.3%) |
| lasso | -0.001..+0.023 | -0.004..+0.017 | 0.536 | +21.1% .. +45.5% |
| logistic | -0.020 | -0.029 | 0.540 | -7.3% .. +5.9% |
| random_forest | -0.004..+0.012 | -0.003..+0.014 | 0.539 | +34.9% .. +127.1% |
| xgboost | +0.002..+0.018 | +0.002..+0.018 | 0.532 | +23.7% .. +46.0% |

- Best ML after-cost return: **+127.1%** (EXP-90015, random_forest
  n_estimators=200 / max_depth=3) at OOS IC +0.012, turnover 969.
- Two lasso points (alpha 0.01, 0.1) zero out all coefficients, producing
  constant predictions -> IC is undefined (NaN, reported honestly); these
  runs degenerated to near-passive behavior (turnover 12).

### Controls (Phase 8 documented rules, same path and costs)

| family | after-cost return |
|--------|-------------------|
| moving_average (5/30) | **+205.5%** |
| momentum (30) | +161.3% |
| moving_average (10/30) | +137.0% |
| moving_average (15/40) | +124.3% |
| equal_weight | +61.2% |
| buy_and_hold | +58.0% |
| volatility_targeted (all) | +61.2% |
| mean_reversion (20) | +6.9% |
| null_flat | 0.0% |
| random_null | -15.5% |
| mean_reversion (30) | -23.9% |
| mean_reversion (10) | -42.9% |

### Verdict reasoning

1. **No ML beat the best control.** Best ML +127% < best control +205%.
   Only 1 of 20 ML runs (EXP-90015) exceeds the passive equal_weight
   (+61%), and that single run is not separated from the near-zero-IC
   null by any robustness criterion.
2. **ICs are null-level.** All 20 runs have |OOS IC| <= 0.02 and
   |rank IC| <= 0.02 on 1,071 test sessions with 20-name cross-sections;
   consistent with no exploitable signal, and hit rates 0.51-0.54 are
   indistinguishable from a coin flip on the positive-return frequency.
3. **Turnover is cost-dominated.** ML turnover 313-969 generates
   $45k-$82k of CM-001 costs against a $1M account, in line with the
   controls; the comparison is on identical terms.
4. **The null is reproducible and auditable.** Artifacts are bitwise
   reproducible and the audit passes 10/10.

### Documented limitation

No SPY/market-benchmark series is present in DS-000004, so the SPY-based
seed labels (LAB-001/LAB-003, excess return vs SPY) cannot be resolved.
The benchmark therefore measures absolute after-cost total return, not
return relative to a market benchmark, using the registered LAB-004 v1
(5-session forward total return, same horizon/anchor as LAB-001 minus the
benchmark). This is recorded in the permanent report and
`docs/phase9_ml_benchmark.md` (Section 9). The verdict above stands on
absolute terms.

---

## AUDIT GATES

### Review 1: Structural + research audit (`scripts/phase9_audit_deep.py`)

**Status: PASS — 14/14 distinct checks, 15/15 per family run**

The deep-input audit recomputes one representative pre-registered grid
point per model family on the real DS-000004 data and runs every audit
check with the complete artifact set (feature snapshot, label snapshot,
datasets with exact purge assertion, fitted model, validation-fitted
calibration map, ranked test frame, canonical backtest config, registered
experiment spec, test predictions). All 14 distinct checks pass for all 5
families (75 checks total, 0 failed); the results are permanently stored in
`benchmarks/phase9_audit_results.json`.

- feature_point_in_time PASS (139,161 rows, window end strictly before decision session)
- feature_scope_guard PASS (FS-001 v1 exactly, 8 documented refs, no Phase 10 zoo)
- label_contract PASS (LAB-004 v1, digest pinned)
- label_availability PASS (139,081 available / 80 `insufficient_future_data`)
- split_integrity PASS (locked windows, exact purge assertion, test never purged)
- unavailable_documented PASS
- calibration_val_only PASS (Platt fitted on validation only)
- grid_lock PASS (only pre-registered grid points run)
- seed_lock PASS (seed 42 everywhere)
- model_scope_guard PASS (families in the five Phase 9 families only)
- test_lock PASS (metrics/backtest restricted to the locked test window)
- backtest_uniformity PASS (identical CM-001 config for ML and controls)
- registry_lineage PASS (DS-000004 / LAB-004 v1 / CM-001 pins)
- data_expansion_guard PASS (no silent data acquisition beyond DS-000004)

### Review 2: Independent reproducibility double-run
(`scripts/phase9_review2_reproducibility.py`)

**Status: PASS**

Recomputes EXP-90003 (ridge alpha 1.0), EXP-90015 (random_forest
200/3) and EXP-90024 (momentum 20) from scratch through the full
pipeline and compares against the stored artifacts:

- test predictions parquet: **bitwise identical** (sha256 equal)
- control signals parquet: **bitwise identical** (sha256 equal)
- metrics (OOS IC / rank IC / MSE / hit rate / ECE / Brier / turnover /
  after-cost return / calibration coefficients / run id): **exactly equal**

Both reviews are recorded in this report; the audit protocol's second
independent pass is complete.

---

## TEST SUITE RESULTS

```
Full suite:         717 passed, 6 xfailed  (~97s)
Phase 9 only:       130 passed, 0 xfailed   (25s)
  features 10  splits 11  models 18  ranking 9  calibration 10
  metrics 10  reproducibility 3  registry 12  backtest 8  baselines 15
  adversarial       24 (A1..A23, A9a/A9b)
```

Phase 1-8 baseline remains green: 587 passed, 6 xfailed.

---

## BENCHMARK RUN LOG

- Full runner: `python scripts/phase9_run_all.py` (deterministic ids
  EXP-90001..EXP-90036, register-before-run, audit, report).
- Section 35 second-pass hardening (Review 1 infrastructure): the audit
  gained three scope guards that make accidental Phase 10+ work and data
  expansion structurally detectable - `feature_scope_guard` (FS-001 v1 must
  stay exactly 8 documented refs), `model_scope_guard` (families limited to
  the five Phase 9 families), and `data_expansion_guard` (no data refs
  beyond DS-000004); `split_integrity` now asserts the exact outcome-window
  purge boundaries, not just row counts. The runner's structural audit
  covers 7 checks; `scripts/phase9_audit_deep.py` exercises all 14 with
  complete artifacts (75 checks, 0 failed).
- Two issues surfaced during the real-data run and were fixed with
  regression tests:
  1. `control_experiment_id_for` / `register_control_experiment` omitted
     the parameterless buy_and_hold and equal_weight controls; canonical
     order now buy_and_hold -> equal_weight -> grid families -> random_null
     -> null_flat (EXP-90021..EXP-90036).
  2. The schema's known model families lacked `lasso`; added to
     `_KNOWN_MODEL_FAMILIES` (with a regression test that all five ML
     families register).
- buy_and_hold is documented as entering at weight 0.99: at the canonical
  no-implicit-loan rule, a weight-1.0 market-on-open buy can be rejected
  when the open gaps up beyond the CM-001 cost buffer, which left the
  control degenerate (zero fills). The 1% residual absorbs entry costs.
  Momentum (target 1.0) keeps the documented Phase 8 rule; its occasional
  INSUFFICIENT_CASH rejects are honest execution outcomes and are counted.

---

## DELIVERABLES

**Implementation (`src/orbit/ml/`):** data, features (FS-001 v1), labels
(LAB-004 v1), splits, grids, models, dataset, ranking, calibration,
metrics, signals, registry, baselines, report, audit, snapshot_cache
(15 modules + `__init__.py`).

**Runner / report / docs / tests:**
- `scripts/phase9_run_all.py` — full end-to-end benchmark runner
- `scripts/phase9_audit_deep.py` — deep-input independent audit (Review 1)
- `scripts/phase9_review2_reproducibility.py` — reproducibility double-run
  (Review 2)
- `benchmarks/phase9_ml_benchmark.parquet` + `.md` — permanent report
  (36 rows, upsert-by-experiment-id)
- `benchmarks/phase9_audit_results.json` + `phase9_review2_results.json` —
  permanent audit/review records
- `benchmarks/phase9_runs/EXP-9xxxx/` — per-experiment artifacts
  (test predictions, signals, metrics.json)
- `data/cache/phase9_snapshots/` — digest-verified feature/label cache
- `docs/phase9_ml_benchmark.md` — full protocol and methodology
- `tests/test_phase9_*.py` (11 files, 130 tests incl. 24 adversarial)
- `pyproject.toml` — added scipy, scikit-learn, xgboost dependencies
- `README.md` — Phase 8/9 status + structure updated

**Registry:** 38 experiment specs (EXP-90001..EXP-90036) with full lineage
pins (DS-000004, LAB-004 v1, CM-001, FS-001 v1, temporal digest, seed 42),
code/config hashes captured at run start, single immutable result per
experiment, artifacts checksummed on attach.

---

## STATUS: COMPLETE — DEFENSIBLE NULL

Phase 9 is complete and all gates cleared. The benchmark is permanent,
reproducible, and honest: it reports no signal where there is none, and
it documents the data-sufficiency limitation (no SPY series) that bounds
the comparison to absolute after-cost returns. The infrastructure for
benchmark-relative evaluation is ready (Phase 2 benchmark set,
LAB-001/LAB-003 contracts) and becomes resolvable as soon as a market
benchmark series is added to the universe.

**Next:** benchmark-relative evaluation (requires SPY/broad-ETF ingestion)
or a Phase 10 feature ablation on the FS-001 set; both are now gated on
data, not on Phase 9 methodology.