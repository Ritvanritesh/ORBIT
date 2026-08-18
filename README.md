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
- Follow-ups before modeling: universe expansion (5 -> 20 -> 50 -> 100,
  config-only) MUST include delisted names for survivorship control;
  benchmark instruments (SPY + broad/sector ETFs) for excess-return labels;
  ALFRED vintage access (schema-ready; revised macro series are
  point-in-time blocked - reported as snapshot limitations - until then).

## Structure

```
docs/                  Charter, seed hypotheses, gates, universe architecture,
                       phase4_temporal_truth.md (temporal engine conventions)
src/orbit/schemas/     HypothesisSpec / ExperimentSpec / Instrument / data contracts
src/orbit/universe/    Membership rules + delisting-aware reconstruction engine
src/orbit/temporal/    Phase 4: times, rules, adapters, engine, snapshot,
                       features, fixtures, contracts
hypotheses/            Registered seed hypotheses (validated instances)
src/orbit/             Ingestion (Phase 3): providers, parsing, registry, storage, validators
scripts/               phase3_run_all.py - end-to-end ingest + verify + provenance
tests/                 Validation tests
```

## Rules

- Paper trading only. No customer money, ever (Phase 1 policy).
- Pre-register hypotheses before feature exploration.
- The final holdout is quarantined and never a tuning surface.
- Any known temporal leak is a hard stop.

See `docs/` for the full charter, gates, and seed hypotheses.