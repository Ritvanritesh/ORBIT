# ORBIT

**Optimized Research & Behavioral Intelligence Trading**

A research operating system for discovering, testing, falsifying, explaining and
replaying trading hypotheses. Paper-trading first, evidence-gated.

Core thesis: the model is replaceable; the evidence chain is the product.

## Status

- Phase 1 (Research Charter and Falsifiable Success Criteria) - complete:
  charter, HypothesisSpec/ExperimentSpec schemas, 3 registered seed
  hypotheses, promotion gates (21 tests).
- Phase 2 (Universe and Data Architecture) - complete: instrument master,
  symbol history, corporate actions, benchmark set, versioned membership
  rules, delisting-aware + lagged-liquidity reconstruction engine (26 tests).
- Phase 3 (Raw Historical Data Ingestion) - complete: free-source ingestion
  (Yahoo chart API, SEC EDGAR companyfacts, FRED), immutable raw storage with
  checksums, validation gates, corporate-action reconciliation + records,
  normalized parquet artifacts, DuckDB provenance registry, reproducibility
  check (99 tests). Run: `python scripts/phase3_run_all.py`.
- Phase 4 (Point-in-Time & Temporal Truth Engine) - complete: temporal
  classification of every record (publication/event/ingestion/effective),
  strict `publication < as_of` availability (ties rejected), date-precision
  next-day convention, market bars available at session close (16:00 ET,
  never the session-open `ts_utc`), vintage resolution with superseded
  versions audited not dropped, point-in-time snapshots with content digests
  and provenance, temporal as-of joins, feature-time guards, 6 permanent
  synthetic leak fixtures, macro schema v1.1.0 with `vintage_date` (ALFRED
  ready), as-published market payloads (split-adjusted OHLCV reconstructed
  from the events artifact, `price_basis` in the payload, provider-aware
  volume basis), effective-time gate, release-calendar sharpening, contract
  convention validation, full test suite 216 tests. See
  `docs/phase4_temporal_truth.md`.
- Phase 5 (Labeling & Outcome Engine) - complete: one immutable, versioned
  `LabelContract` per prediction target (semantics required, nothing
  inferred), `LabelEngine` computing exactly one reproducible future
  outcome per decision row (trading-session horizons that never count
  calendar gaps, strict session-close boundaries, POST_EVENT anchors,
  split-continuous return basis with as-published audit closes, ex-date
  dividends on the stored share basis, delisting vs data-shortfall
  classification, explicit unavailable reasons, benchmark windows that
  share the exact anchor/horizon), `LabelVersionRegistry` immutability
  (no re-registration, no version inflation), `LabelSnapshot` with a
  deterministic content digest, overlap metadata for later purging,
  golden hand-calculation tests, Phase 4 integration tests (entry-bar and
  as-published agreement, no label leakage, artifact separation), full
  test suite 311 tests. Seed contracts registered: LAB-001 (H-001
  momentum) and LAB-003 (H-003 PEAD); LAB-002 (H-002 risk-adjusted) is
  deferred. See `docs/phase5_labels.md`.
- Phase 6 (Experiment Registry Foundation) - complete: register-before-run\- Phase 7 (Event-Driven Backtesting Engine) - complete: deterministic replayable daily/EOD execution+accounting simulator (Python 3.10, pydantic 2.13.4, polars 1.43.2, DuckDB), integrated with Phases 4/5/6, 587 passed 6 xfailed across full suite, long-only enforcement (selling refused loudly), order IDs ORD-000000+, execution price OPEN/CLOSE, execution delay >= 1 for OPEN, default execution open delay=1, expiry: eligible + order_expiry_sessions, cash identity equity = cash + realized + unrealized - fees_total, Phase 4 temporal truth validation, Phase 5 label engine bridge (DECISION_INSTANT anchor), Phase 6 experiment lifecycle (run_backtest_experiment), manifest/run identity (BT-<config8>-<content12>), content_hash excludes created_at/run_id, 107 Phase 7-only tests PASS, Review 1/2/3 audits all PASS with second independent pass, PHASE 7 STATUS report PASS.


  with an immutable canonical `ExperimentSpec` (content-hash identity,
  DB-enforced against raw SQL), validated lifecycle state machine
  (registered/running/completed/failed/rejected/promoted/retired) with a
  recorded decision log, acyclic hypothesis-scoped genealogy, full lineage
  pins (dataset snapshots, temporal contract, label versions, feature
  refs, model/cost identity), code/config hashes captured before
  execution, registry-computed trial numbers and search depth, failed and
  null experiments retained (never hidden), FK-bound artifacts/results/
  decisions with one immutable result per experiment, reproduction
  specifications that resolve every lineage element (missing lineage is a
  loud violation) with replay tests, invariant validation for audits
  (content-hash recomputation, acyclicity, orphan counts), and concurrency
  tests proving exactly-one-winner semantics under racing writers. Full
  test suite 427 tests. See `docs/phase6_experiment_registry.md`; run the
  end-to-end demo with `python scripts/phase6_demo.py`.
- Follow-ups before modeling: universe expansion (5 -> 20 -> 50 -> 100,
  config-only) MUST include delisted names for survivorship control;
  benchmark instruments (SPY + broad/sector ETFs) for excess-return labels
  (the seed contracts require the SPY series); ALFRED vintage access
  (schema-ready; revised macro series are point-in-time blocked - reported
  as snapshot limitations - until then).
- Phase 8 (Documented Baseline Strategies) - complete: five parameterized
  documented strategies (momentum, mean reversion, moving average,
  volatility targeting, buy-and-hold) plus null/random baselines,
  cross-validated parameter grids, and a benchmark report module executed
  on synthetic bars. Phase 9 re-executes these same rules on the real
  dataset as controls.
- Phase 9 (Baseline ML Benchmark) - complete: first falsifiable ML
  benchmark - 5 model families x 20 pre-registered grid points (ridge,
  lasso, logistic, random forest, xgboost, seed 42) on FS-001 v1
  point-in-time features (8 numerics, strict boundary) against LAB-004 v1
  (5-session forward total return), locked chronological split
  (train 2010-2018 / val 2019-2021 / test 2022-2026, exact outcome-window
  purge, test never purged), register-before-run experiments
  (EXP-90001..EXP-90036) with full lineage pins and code/config hashes,
  validation-only Platt calibration with `assert_no_test_fit`, per-session
  OOS/rank IC, ECE/Brier/MSE/hit rate, canonical Phase 7 backtest with
  CM-001 costs and WEIGHT sizing identical to the 16 Phase 8 controls on
  real bars, permanent parquet+markdown report
  (`benchmarks/phase9_ml_benchmark.*`), digest-verified snapshot cache,
  full audit, 130 Phase 9 tests including 24 adversarial scenarios, and
  two independent review passes (deep-input audit:
  `python scripts/phase9_audit_deep.py`; reproducibility double-run:
  `python scripts/phase9_review2_reproducibility.py`). Documented
  limitation: no SPY series in DS-000004, so excess-return labels
  (LAB-001/LAB-003) are unresolved and the comparison is absolute
  (after-cost total return), not benchmark-relative. See
`docs/phase9_ml_benchmark.md`; run everything with
   `python scripts/phase9_run_all.py`.
- Phase 10 (Feature Engineering + Ablation) - complete: 15 new
  point-in-time features (FEAT-101..FEAT-115) in 5 documented families
  (momentum, trend, volatility, volume/liquidity, range/price structure)
  computed from DS-000004 OHLCV bars only, all at the strict boundary
  (window end strictly before decision session); a locked 13-set x 4-model
  ablation plan (FS-002..FS-013 vs the frozen Phase 9 FS-001 v1, 52
  experiments EXP-10001..EXP-10052, digest-pinned, one model point per
  Phase 9 family with parent linkage EXP-90003/90006/90015/90019,
  deterministic id mapping); immutable digest-bound feature snapshots
  cached digest-verified; strong temporal-boundary audit (45,326 sampled
  FS-003 rows recomputed from truncated bars, 0 mismatches); train-only
  feature diagnostics (quality + redundancy); register-before-run with a
  feature-mutation-detecting config hash; benchmark 52/52 completed with
  runner audit 48/48 PASS (incl. row-identity gates: all Phase 10 sets
  resolve identical rows, every FS-001-only row explained by the Phase 10
  warm-up policy), Review 1 (structural audit incl. cross-phase
  base-anchor: FS-001 runs exactly reproduce the Phase 9 parents bitwise,
  and the deep checks exercised against real artifacts) 12/12 PASS with a
  53/53 audit pass, Review 2 (reproducibility double-run) 3/3 PASS; 90
  Phase 10 tests incl. 18 adversarial scenarios (A1..A20); full suite 807
  passed, 6 xfailed. Verdict: C - limited, non-robust feature sensitivity;
  no economically meaningful family-specific signal; the Phase 9 DEFENSIBLE
  NULL stands in substance. Run/review:
  `python scripts/phase10_run_all.py`, `scripts/phase10_review1.py`,
  `scripts/phase10_review2.py`. See `PHASE_10_STATUS.md`.

## Structure

```
docs/                  Charter, seed hypotheses, gates, universe architecture,
                       phase4_temporal_truth.md (temporal engine conventions),
                       phase5_labels.md (label contract and outcome conventions),
                       phase6_experiment_registry.md (registry design),
                       phase9_ml_benchmark.md (Phase 9 protocol)
src/orbit/schemas/     HypothesisSpec / ExperimentSpec / Instrument / data contracts
src/orbit/universe/    Membership rules + delisting-aware reconstruction engine
src/orbit/temporal/    Phase 4: times, rules, adapters, engine, snapshot,
                       features, fixtures, contracts
src/orbit/labels/      Phase 5: contract, outcomes, engine, snapshot, registry, seeds
src/orbit/experiments/ Phase 6: lifecycle, registry (DuckDB), reproduction, service
src/orbit/ml/          Phase 9/10: features, labels, splits, grids, models, ranking,
                       calibration, metrics, signals, registry, baselines, report,
                       audit + phase10_* modules (plan, diagnostics, registry, audit,
                       report, runner), snapshot_cache
hypotheses/            Registered seed hypotheses (validated instances)
src/orbit/             Ingestion (Phase 3): providers, parsing, registry, storage, validators
scripts/               phase3_run_all.py - end-to-end ingest + verify + provenance
                       phase6_demo.py - experiment registry end-to-end example
                       phase9_run_all.py - Phase 9 ML benchmark end-to-end
                       phase9_audit_deep.py - Phase 9 deep-input independent audit
                       phase9_review2_reproducibility.py - Phase 9 reproducibility double-run
                       phase10_run_all.py - Phase 10 feature-ablation benchmark end-to-end
                       phase10_review1.py - Phase 10 independent structural audit
                       phase10_review2.py - Phase 10 reproducibility double-run
tests/                 Validation tests
```

## Rules

- Paper trading only. No customer money, ever (Phase 1 policy).
- Pre-register hypotheses before feature exploration.
- The final holdout is quarantined and never a tuning surface.
- Any known temporal leak is a hard stop.

See `docs/` for the full charter, gates, and seed hypotheses.