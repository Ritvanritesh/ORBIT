# PHASE 10 STATUS REPORT

**ORBIT: Optimized Research & Behavioral Intelligence Trading**
**Date:** 20 August 2026
**Phase:** 10 — Feature Engineering + Ablation

---

## EXECUTIVE SUMMARY

Phase 10 answers the question the Phase 9 DEFENSIBLE NULL raises: was the
null caused by an insufficient feature representation? It isolates FEATURE
REPRESENTATION from model complexity by holding the Phase 9 protocol
exactly constant (dataset DS-000004, label LAB-004 v1, locked split,
CM-001 costs, Phase 7 backtester, top-3 WEIGHT signals, seed 42,
validation-only calibration) and ablating a pre-registered set of five
point-in-time-valid feature families.

**Phase 10 is COMPLETE. Verdict: C — LIMITED, NON-ROBUST FEATURE
SENSITIVITY; THE PHASE 9 NULL STANDS IN SUBSTANCE.**

The benchmark ran to completion: 52/52 experiments (EXP-10001..EXP-10052,
13 feature sets x 4 models) through the canonical Phase 7 path, audited
three times (runner audit 48/48, Review 1 structural audit 12/12 with a
53/53 audit pass, Review 2 reproducibility double-run 3/3). The
cross-phase anchor is verified bitwise: the FS-001 base runs reproduce
the Phase 9 parents EXP-90003/90006/90015/90019 exactly (identical
predictions and metrics).

The feature representation is NOT irrelevant: ridge and lasso OOS ICs
improve with the new features (ridge -0.0003 -> +0.0202, lasso +0.0226 ->
+0.0254..+0.0312), and two of 52 runs (lasso FS-009 +221%, xgboost FS-013
+241%) exceed the best documented Phase 8 control (+205%, moving_average
5/30). But the effect is small, inconsistent across model families
(random forest DECREASES, IC -0.018), and the best returns come from
LEAVE-ONE-FAMILY-OUT sets (reduced feature zoos), not from any family's
incremental signal. Every OOS IC across all 52 runs sits at |IC| <= 0.032
with hit rates 0.517-0.537 (coin-flip), i.e. economically negligible on
1,071 test sessions. No robust, family-specific, economically meaningful
signal is established; the effect is consistent with noise under 52
comparisons.

- **807 tests passed, 6 xfailed** across the full suite (Phases 1-9 stay green)
- **90 Phase 10 tests pass** (incl. 18 adversarial scenarios, A1..A20)
- **52/52 benchmark rows completed** (0 failed) in the permanent report
- **Runner audit: 48/48 PASS** (incl. strong temporal boundary: 45,326
  sampled FS-003 rows recomputed from truncated bars, 0 mismatches; the
  test-lock and both row-identity checks now exercise real artifacts)
- **Audit Review 1: 12/12 PASS** (`scripts/phase10_review1.py`; audit pass
  53/53 incl. the deep checks — grid/model/seed locks, preprocessing
  train-only, registry lineage, test lock, row identity; results in
  `benchmarks/phase10_review1_results.json`)
- **Audit Review 2: 3/3 PASS** (reproducibility double-run; bitwise-
  identical predictions + exactly equal metrics vs stored artifacts for
  EXP-10001, EXP-10009, EXP-10052; results in
  `benchmarks/phase10_review2_results.json`)

---

## BENCHMARK RESULTS (test window 2022-01-03 .. 2026-06-30, CM-001 costs)

### Per family: FS-001 (base) vs FS-002 (new-only) vs FS-003 (ALL) vs best set

| family | FS-001 IC / ret | FS-002 IC / ret | FS-003 IC / ret | best set (IC / ret) |
|--------|-----------------|-----------------|-----------------|---------------------|
| ridge | -0.0003 / +49.3% | +0.0148 / +24.0% | +0.0202 / +7.6% | FS-010 (+0.0196 / +66.2%) |
| lasso | +0.0226 / +21.1% | +0.0256 / +201.9% | +0.0254 / +200.4% | FS-009 (+0.0312 / +221.2%) |
| random_forest | +0.0119 / +127.1% | +0.0013 / +146.6% | -0.0056 / +94.9% | FS-002 (+0.0013 / +146.6%) |
| xgboost | +0.0113 / +23.7% | -0.0021 / +57.1% | +0.0100 / +116.6% | FS-013 (+0.0070 / +241.2%) |

### Headline numbers

- **All 52 runs: |OOS IC| <= 0.0312, |rank IC| <= 0.0270, hit rate
  0.517-0.537** — economically negligible on 1,071 test sessions with
  20-name cross-sections.
- Best after-cost returns: **+241.2%** (EXP-10052, xgboost FS-013) and
  **+221.2%** (EXP-10034, lasso FS-009) — above the best Phase 9 ML
  (+127.1%, EXP-90015) and the best Phase 8 control (+205.5%,
  moving_average 5/30).
- Both best runs use **ALL-minus-family** sets (FS-009 = ALL - momentum,
  FS-013 = ALL - range): removing features helped more than any family
  added, a signature of regularization/noise effects rather than a
  family-specific predictive signal.
- Feature-set sensitivity is real but directionally inconsistent:
  ridge IC +0.020, lasso IC +0.003..+0.009, but random_forest IC -0.018
  and its return falls; ridge's IC gain does not translate to return
  (return -42 points).
- FS-001 base rows reproduce the Phase 9 parents **bitwise**: EXP-10001
  == EXP-90003, EXP-10002 == EXP-90006, EXP-10003 == EXP-90015,
  EXP-10004 == EXP-90019 (identical test predictions sha256 + identical
  metrics; the only stored-field differences are non-metric metadata:
  run_id content hash, coefs persistence, feature_set_id lineage field).

### Verdict reasoning

1. **The strict null (representation is irrelevant) is weakened but not
   refuted.** Ridge/lasso ICs improve with the extended representation and
   two runs beat the best documented control on after-cost return; yet
   every IC remains null-adjacent (<= 0.032), every hit rate remains a
   coin flip, and the improvement is family-dependent (RF worsens).
2. **No feature family carries identifiable incremental signal.**
   BASE+family sets (FS-004..008) never dominate; the two best returns
   come from ALL-minus-family sets, i.e. from REMOVING families. This is
   the pattern expected under noise/overfitting, not under a real
   family-specific effect.
3. **52 comparisons with no pre-registered multiple-comparison guard:**
   selecting the 2 best of 52 outcomes (+241%, +221%) without correction
   is not evidence of a deployable edge. The pre-registered plan requires
   robustness criteria that these runs do not meet (no consistency across
   families, no separation in IC space).
4. **The benchmark is permanent and reproducible.** Bitwise reproducibility
   (Review 2), structural audit (Review 1), the strong temporal-boundary
   recomputation (45,326 sampled rows, 0 mismatches), and the cross-phase
   anchor (FS-001 == Phase 9 parents, bitwise) all PASS.

**Conclusion:** Phase 10 does NOT overturn the Phase 9 DEFENSIBLE NULL.
The new point-in-time features change results at the margin and in
family-dependent directions, but no economically meaningful, robust,
family-specific signal is established on DS-000004. Feature
representation alone does not rescue the benchmark on this universe; the
bottleneck is more plausibly information (20-name universe, OHLCV-only
inputs, no market-benchmark series) than representation.

### Documented limitation

Same as Phase 9: no SPY/market-benchmark series in DS-000004, so
excess-return labels (LAB-001/LAB-003) remain unresolved and all
comparisons are absolute after-cost total return, not benchmark-relative.

Phase 10 adds one disclosed comparison limitation: 320 train rows (0.7%,
2 mid-window-listed instruments) exist only in FS-001 because the Phase 10
warm-up policy drops their first 200 sessions. Val/test and the remaining
99.3% of train are row-identical across sets, and the row-identity audit
gates make any further set drift structurally impossible to miss (see
FEATURE SETS).

---

## PRE-REGISTERED PROTOCOL (locked before any execution)

Plan digest (sha256 over the full pre-registered payload):
`16d62bff387704746fe2ac23742045dcf27314109957752473ed4b0edff64910`

- **Dataset:** DS-000004 (20-symbol development universe, 139,961 bar rows,
  1996-01-02 .. 2026-08-14) — Phase 9 frozen, read-only input.
- **Label:** LAB-004 v1 (5-session forward total return) — Phase 9 frozen.
- **Windows:** train 2010-01-04..2018-12-31, val 2019-01-02..2021-12-31,
  test 2022-01-03..2026-06-30 (locked; exact outcome-window purge).
- **Cost model:** CM-001 (spread 2 bps, fees 1 bps, slippage 2 bps).
- **Signals:** top-3 long, equal weight 1/3 (Phase 9 path).
- **Seed:** 42 everywhere. **Calibration:** Platt on validation only.
- **Registry:** register-before-run; config hash pins the feature-set id +
  version + exact feature refs + definitions digest, so post-hoc feature
  mutation is structurally detectable.

## FEATURE FAMILIES (FEAT-101..FEAT-115)

All computed from existing DS-000004 OHLCV bars only (no invented
fundamentals/macro/news/options/alternative data). Every feature is
point-in-time at the strict boundary: `window_end_session = D-1 < D`.

| Family | Feature IDs | Kind |
|--------|-------------|------|
| momentum | FEAT-101 ret_5, FEAT-102 ret_60, FEAT-103 ret_120 | close(D-1)/close(D-N)-1 (same convention as FS-001) |
| trend | FEAT-104 sma_ratio_10_30, FEAT-105 sma_ratio_20_50, FEAT-106 price_distance_200ma | moving-average structure |
| volatility | FEAT-107 vol_60, FEAT-108 vol_90, FEAT-109 vol_ratio_10_30 | realized-vol horizons |
| volume / liquidity | FEAT-110 dv_med_10, FEAT-111 dv_med_30, FEAT-112 vol_zscore_20 | dollar-volume medians + trailing z-score |
| range / price structure | FEAT-113 high_low_10_pos, FEAT-114 high_low_30_pos, FEAT-115 normalized_range_20 | position within rolling range |

## FEATURE SETS (13 snapshots)

| Snapshot | Role | Members |
|----------|------|---------|
| FS-001 v1 | BASE (frozen Phase 9 artifact, digest preserved) | 8 (FEAT-001..008) |
| FS-002 v1 | NEW (Phase 10 families only) | 15 |
| FS-003 v1 | ALL (FS-001 + FS-002) | 23 |
| FS-004..FS-008 v1 | BASE + one family (momentum..range) | 11 each |
| FS-009..FS-013 v1 | ALL - one family (momentum..range) | 20 each |

All sets resolve IDENTICAL rows in the VAL and TEST splits and on 44,079
of 44,399 train rows; the remaining 320 train rows (0.7%) exist only in
FS-001 (2 instruments, 160 rows each, listed mid-train-window). Root cause
is the Phase 10 warm-up policy (price_distance_200ma needs 200 completed
sessions, MAX_FEATURE_WINDOW_PHASE10=200): instruments listed after
2009-05 — INS-000008 (bars start 2012-05-18) and INS-000010 (bars start
2010-06-29) — drop their first 200 sessions from the Phase 10 sets,
i.e. 160 in-train sessions per instrument that FS-001 keeps. Across the
full history all 3,200 FS-001-only rows sit inside the first 200 sessions
of their instrument (verified 3,200 within warm-up, 0 beyond). The
ablation is therefore feature-only on val/test and on 99.3% of train; the
0.7% train difference is quantified, disclosed, and structurally checked
by the `row_identity_phase10_sets` / `row_identity_fs001_warmup` audit
gates (they FAIL if any Phase 10 set drifts from its siblings or if any
FS-001 row falls outside the explainable warm-up zone).
Snapshots are immutable: the `transformation` field binds set id + version +
definitions digest, and each snapshot is cached digest-verified under
`data/cache/phase10_snapshots/`.

## MODEL POINTS (4, one per Phase 9 family, chosen BEFORE Phase 10 results)

| family | params | Phase 9 parent |
|--------|--------|----------------|
| ridge | alpha=1.0 | EXP-90003 |
| lasso | alpha=0.001 | EXP-90006 |
| random_forest | n_estimators=200, max_depth=3 | EXP-90015 |
| xgboost | n_estimators=200, max_depth=3, learning_rate=0.1 | EXP-90019 |

Logistic is intentionally excluded from the ablation subset: its score is a
monotone transform of the ridge sign for the binary target and adds no
independent ranking information for the feature-representation question
(the infrastructure supports it for a later registered experiment).

13 x 4 = **52 experiments**, EXP-10001..EXP-10052 (deterministic ordering:
feature sets in the order above, then models). Every experiment is pinned to
its Phase 9 parent in the registry notes.

---

## AUDIT GATES

### Runner audit (inside `scripts/phase10_run_all.py`)

**Status: PASS — 48/48 checks**

Includes: plan lock, every snapshot point-in-time valid + exact membership
(13 sets), feature-scope guard, frozen FS-001 digest, label contract,
split integrity + locked windows, dataset unchanged vs manifest, model
scope guard, seed lock, test lock (exercised against the run's own test
predictions), backtest uniformity, registry lineage, data expansion guard,
the **row-identity gates** (all Phase 10 sets resolve identical rows; every
FS-001-only row is inside the Phase 10 warm-up zone), and the **strong
temporal boundary**: 45,326 sampled FS-003 rows recomputed from truncated
bar history (bars <= D) and compared to the snapshot — 0 mismatches.

### Review 1: Independent structural audit (`scripts/phase10_review1.py`)

**Status: PASS — 12/12 checks**

plan_lock (digest match), plan_count (52), plan_set_order (13 sets in
locked order), report_complete (EXP-10001..EXP-10052, 52 rows),
report_no_hidden_failures (all completed), set_membership (8/15/23/11x5/
20x5), snapshot_digests (all 13 match the cache), **audit_pass (53/53)**
— the audit is fed real artifacts (test predictions, fitted-model stand-in
from EXP-10001's metrics, the registered EXP-10001 spec, bars), so the
deep checks that a bare run can miss are exercised: test_lock, grid_lock,
model_scope_guard, seed_lock, preprocessing_train_only, registry_lineage,
row_identity_phase10_sets, row_identity_fs001_warmup —
audit_exercises_deep_checks (no silent checks on the real run),
**cross_phase_base_consistency** (EXP-10001..10004 == EXP-90003/90006/
90015/90019: bitwise-identical predictions + exactly equal substantive
metrics), diagnostics_scope (train split only), report_pins (all 52 rows:
seed 42, CM-001, LAB-004 v1, DS-000004, locked test window).

### Review 2: Independent reproducibility double-run
(`scripts/phase10_review2.py`)

**Status: PASS — 3/3 runs**

Recomputes EXP-10001 (FS-001/ridge), EXP-10009 (FS-003/ridge), EXP-10052
(FS-013/xgboost) from scratch through the full pipeline from digest-verified
cached snapshots:

- test predictions parquet: **bitwise identical** (sha256 equal) for all 3
- metrics (OOS IC / rank IC / MSE / hit rate / ECE / Brier / turnover /
  after-cost return / calibration coefficients / backtest counts): exactly
  equal; the only stored-field differences are non-metric metadata
  (run_id content hash from the registry spec encoding, coefs persistence,
  feature_set_id lineage field)

---

## TEST SUITE RESULTS

```
Full suite:         807 passed, 6 xfailed
Phase 10 only:        90 passed, 0 xfailed
  features 16  plan 11  registry 7  dataset 5  diagnostics 7
  adversarial 18 (A1..A20)  audit 11  report 8  pipeline 7 (52-exp run)
```

The full-pipeline integration test runs the ENTIRE locked plan (52
experiments) on hermetic synthetic data, verifies register-before-run for
every experiment, bit-identical reproducibility across two full runs, an
audit with 0 failures, and the permanent report/plan/diagnostics artifacts.

Phase 1-9 baseline remains green: 717 passed, 6 xfailed.

---

## REVIEW-HARDENING PASS (post-benchmark, before sign-off)

A full independent review of the Phase 10 implementation (all modules,
scripts, tests, and the permanent artifacts) surfaced five defects plus a
disclosed limitation; every defect is fixed, regression-tested, and
re-verified on the real artifacts:

1. **Permanent inventory rendered family as `?`** for Phase 10 features
   (`_feature_inventory_lines` looked families up in the Phase 9 map).
   Fixed via `FEATURE_FAMILY_BY_ID_PHASE10`; the research doc was
   regenerated (families present, digests unchanged).
2. **Snapshot table rendered `FS-002 vv1`** (extra `v` prepended to the
   stored `v1`). Fixed in the report and in registry title/notes.
3. **Deep audit gates were structurally dead on the real run** — the
   runner and Review 1 never passed test predictions / fitted-model /
   experiment spec, so test_lock, grid_lock, model_scope_guard, seed_lock,
   preprocessing_train_only and registry_lineage were never exercised
   (45/45 was misleading). The runner now feeds test predictions; Review 1
   feeds the full artifact set and asserts `audit_exercises_deep_checks`
   (audit now 53/53 on the real run).
4. **Redundancy diagnostics computed the correlation matrices 4x** per
   call (pearson + spearman recomputed inside the pair filter). Now
   computed once and passed through; a consistency regression test pins
   the pair list to the published matrices.
5. **Row identity between sets was never verified.** New
   `verify_row_identity` gate + two audit checks: FS-002..FS-013 must
   resolve exactly the same rows, and every FS-001-only row must be inside
   the Phase 10 warm-up zone. On the real data: 3,200 FS-001-only rows,
   3,200 within warm-up, 0 beyond (see the disclosed 0.7% train
   difference above).
6. **`FEAT-115` was misnamed `close_in_range_10` in the status doc** (the
   feature is `normalized_range_20`). Corrected here and in the research
   doc.

Re-verification: Review 1 PASS 12/12 (audit 53/53), Review 2 PASS 3/3
(bitwise), full Phase 10 suite 90/90, full repo suite 807 passed.

---

## ADVERSARIAL SCENARIOS (A1..A20, all refused or detected loudly)

A1 future-feature leakage detected · A2 same-session feature reference
detected · A3 non-finite features rejected · A4 unregistered model point
refused · A5 unknown feature set refused · A6 membership drift raises ·
A7 FS-001 frozen-digest mutation detected · A8 definition tampering
invalidates the digest · A9 corrupt cache refused on load · A10 wrong
feature-set version rejected · A11 duplicate registration refused ·
A12 ids outside the locked range impossible · A13 exact membership
assertion · A14 test-lock violation flagged · A15 backtest-config drift
flagged · A16 label-contract drift flagged · A17 diagnostics train-only
scope enforced · A18 data expansion detected · A19 feature-zoo scope
expansion detected · A20 split-integrity violation flagged.

---

## BENCHMARK RUN LOG

- Full runner: `python scripts/phase10_run_all.py` (deterministic ids
  EXP-10001..EXP-10052, register-before-run, diagnostics, reports, audit,
  live progress lines). Wall time on the real run: 1,600 s (26.7 min),
  of which the strong temporal-boundary audit ~6 min.
- Two review-script fixes surfaced during execution, both regression-tested:
  1. `phase10_review1.py` read `PHASE10_FEATURE_SET_ORDER` as dicts (it is
     a list of ids); fixed, plus live step markers (7 steps) and a
     progress line in the audit's temporal-boundary loop.
  2. Both reviews initially compared ALL stored metric keys strictly,
     which flagged non-metric metadata (run_id content hash, coefs
     persistence, feature_set_id lineage field) as diffs. The substantive
     anchor — bitwise predictions + every shared metric exactly equal —
     held for all pairs; the checks now exclude the documented metadata
     keys and PASS 10/10 and 3/3.
- A partial run reached EXP-10010 before being stopped for live-progress
  instrumentation; the final run upserted every experiment id with fresh
  results, so nothing is polluted.
- Feature diagnostics (train-only) recorded: FS-002 has 2 high-correlation
  pairs (dv_med_10/dv_med_30 r=0.98; vol_60/vol_90 r=0.96) and FS-001 has
  none; no exact duplicates in any set. Redundancy is documented, never
  auto-removed.

---

## DELIVERABLES

**Implementation (`src/orbit/ml/`):**
- `features.py` — 15 new documented features, families, sets FS-002..FS-013,
  snapshot builders + digests, PIT + finiteness assertions (FS-001 v1
  untouched)
- `dataset.py` — parameterized `assemble_datasets(..., feature_names=...)`
  (Phase 9 default behavior unchanged)
- `phase10_plan.py` — locked 52-experiment plan, digest, id mapping
- `phase10_diagnostics.py` — feature quality + redundancy (train-only)
- `phase10_registry.py` — register-before-run + config hash (feature
  mutation detectable)
- `phase10_audit.py` — independent audit incl. strong temporal boundary
  (progress instrumentation)
- `phase10_report.py` — permanent report writers
- `phase10_runner.py` — full pipeline runner with live progress lines
- `snapshot_cache.py` — Phase 10 snapshot cache (digest-verified)

**Runner / reviews / docs / tests:**
- `scripts/phase10_run_all.py` — full end-to-end benchmark runner
- `scripts/phase10_review1.py` — independent structural audit (12/12 PASS,
  incl. cross-phase base anchor: FS-001 runs exactly reproduce the Phase 9
  parents, and the deep audit gates exercised on real artifacts)
- `scripts/phase10_review2.py` — reproducibility double-run (3/3 PASS,
  bitwise)
- `benchmarks/phase10_plan.json` / `phase10_diagnostics.json` — locked plan
  + train-only diagnostics
- `benchmarks/phase10_feature_research.parquet` + `.md` — permanent report
  (52 rows, upsert-by-experiment-id, all completed)
- `benchmarks/phase10_review1_results.json` + `phase10_review2_results.json`
  — permanent review records
- `benchmarks/phase10_runs/EXP-1xxxx/` — per-experiment artifacts
  (test predictions, signals, metrics.json)
- `data/cache/phase10_snapshots/` — digest-verified snapshot cache (13 sets)
- `docs/phase10_feature_research.md` — full protocol, feature inventory,
  diagnostics, and methodology
- `tests/test_phase10_*.py` (9 files, 90 tests incl. 18 adversarial)
- `PHASE_10_STATUS.md` + `README.md` — status and structure updated

**Registry:** 52 experiment specs (EXP-10001..EXP-10052) with full lineage
pins (DS-000004, LAB-004 v1, CM-001, feature-set id + version + definitions
digest, temporal digest, seed 42, Phase 9 parent in notes), code/config
hashes captured at run start, single immutable result per experiment,
artifacts checksummed on attach.

---

## STATUS: COMPLETE — C (LIMITED, NON-ROBUST FEATURE SENSITIVITY)

Phase 10 is complete and all gates cleared. The feature-representation
ablation is permanent, reproducible, and honest: it reports that the new
point-in-time features move results at the margin and in family-dependent
directions, that no family carries identifiable incremental signal, that
every IC remains null-adjacent, and that the Phase 9 DEFENSIBLE NULL
therefore stands in substance. The infrastructure for the next
information-adding steps (market-benchmark series for excess-return labels,
universe expansion with delisted names for survivorship, and genuinely new
data domains) is ready and gated on data acquisition, not on methodology.

**Next:** benchmark-relative evaluation (requires SPY/broad-ETF ingestion)
or universe expansion (50 -> 100 names with delisted history) — both are
now gated on data, not on Phase 10 methodology.