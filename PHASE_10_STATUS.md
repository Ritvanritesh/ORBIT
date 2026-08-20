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

**Status: CODE COMPLETE — FULL BENCHMARK RUN PENDING.**

- The locked ablation plan (13 feature sets x 4 model points = 52
  experiments, EXP-10001..EXP-10052) is implemented and digest-pinned.
- All 15 new point-in-time features (FEAT-101..FEAT-115) are implemented,
  documented, and verified leak-free (strong boundary recomputation).
- The full end-to-end pipeline (runner, registry, diagnostics, reports,
  audit, reviews) is implemented and green on hermetic synthetic data.
- **792 tests passed, 6 xfailed** across the full suite (Phases 1-9 stay
  green; ~75 new Phase 10 tests incl. 18 adversarial scenarios A1..A20).
- The permanent benchmark artifacts (52 real-data experiments), Review 1
  (structural audit) and Review 2 (reproducibility double-run), and the
  scientific verdict are PENDING: run `python scripts/phase10_run_all.py`
  (~2-3 h), then `scripts/phase10_review1.py` and `scripts/phase10_review2.py`.
  This section will be replaced by the verdict once those complete.

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
| range / price structure | FEAT-113 high_low_10_pos, FEAT-114 high_low_30_pos, FEAT-115 close_in_range_10 | position within rolling range |

## FEATURE SETS (13 snapshots)

| Snapshot | Role | Members |
|----------|------|---------|
| FS-001 v1 | BASE (frozen Phase 9 artifact, digest preserved) | 8 (FEAT-001..008) |
| FS-002 v1 | NEW (Phase 10 families only) | 15 |
| FS-003 v1 | ALL (FS-001 + FS-002) | 23 |
| FS-004..FS-008 v1 | BASE + one family (momentum..range) | 11 each |
| FS-009..FS-013 v1 | ALL - one family (momentum..range) | 20 each |

All sets resolve IDENTICAL rows inside the split windows (warm-up is
complete before 2010), so every comparison is a clean feature-only ablation.
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

## TEST SUITE RESULTS

```
Full suite:         792 passed, 6 xfailed
Phase 10 only:        76 passed, 0 xfailed
  features 12  plan 9  registry 6  dataset 5  diagnostics 6
  adversarial 18 (A1..A20)  audit 6  report 7  pipeline 7 (52-exp run)
```

The full-pipeline integration test runs the ENTIRE locked plan (52
experiments) on hermetic synthetic data, verifies register-before-run for
every experiment, bit-identical reproducibility across two full runs, an
audit with 0 failures, and the permanent report/plan/diagnostics artifacts.

Phase 1-9 baseline remains green: 717 passed, 6 xfailed.

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
- `phase10_report.py` — permanent report writers
- `phase10_runner.py` — full pipeline runner
- `snapshot_cache.py` — Phase 10 snapshot cache (digest-verified)

**Runner / reviews / docs / tests:**
- `scripts/phase10_run_all.py` — full end-to-end benchmark runner
- `scripts/phase10_review1.py` — independent structural audit (incl.
  cross-phase anchor: FS-001 base runs must EXACTLY reproduce the Phase 9
  parents EXP-90003/90006/90015/90019)
- `scripts/phase10_review2.py` — reproducibility double-run (bitwise
  artifact comparison for EXP-10001, EXP-10009, EXP-10052)
- `benchmarks/phase10_plan.json` / `phase10_diagnostics.json` — locked plan
  + train-only diagnostics (plan/diagnostics written on first run)
- `benchmarks/phase10_feature_research.parquet` + `.md` — permanent report
  (52 rows, upsert-by-experiment-id; PENDING the full run)
- `benchmarks/phase10_runs/EXP-1xxxx/` — per-experiment artifacts
  (PENDING the full run)
- `docs/phase10_feature_research.md` — permanent research report (generated
  by the runner; PENDING the full run)
- `tests/test_phase10_*.py` (9 files, 76 tests incl. 18 adversarial)

---

## PENDING (run at the end)

1. `python scripts/phase10_run_all.py` — full 52-experiment benchmark on
   DS-000004 (est. 2-3 h; snapshots already cached, so it starts fast).
2. `python scripts/phase10_review1.py` — independent structural audit.
3. `python scripts/phase10_review2.py` — independent reproducibility
   double-run.
4. Fill in the BENCHMARK RESULTS + VERDICT sections above and commit.

A benign early note: a partial run reached EXP-10010 before being stopped;
those per-experiment artifacts are legitimate executions and the final run
upserts every experiment id with fresh results, so nothing is polluted.