# ORBIT

**Optimized Research & Behavioral Intelligence Trading**

A research operating system for discovering, testing, falsifying, explaining and
replaying trading hypotheses. Paper-trading first, evidence-gated.

Core thesis: the model is replaceable; the evidence chain is the product.

## Status

- Phase 1 (Research Charter and Falsifiable Success Criteria) - complete:
  charter, HypothesisSpec/ExperimentSpec schemas, 3 registered seed
  hypotheses, promotion gates.
- Phase 2 (Universe and Data Architecture) - complete: instrument master,
  symbol history, corporate actions, benchmark set, versioned membership
  rules, delisting-aware + lagged-liquidity reconstruction engine (23 tests).
- Phase 3 (Raw Historical Data Ingestion) - complete: free-source ingestion
  (Yahoo chart API, SEC EDGAR companyfacts, FRED), immutable raw storage with
  checksums, validation gates, corporate-action reconciliation + records,
  normalized parquet artifacts, DuckDB provenance registry, reproducibility
  check (80 tests). Run: `python scripts/phase3_run_all.py`.
- Follow-ups before modeling: universe expansion (5 -> 20 -> 50 -> 100,
  config-only) MUST include delisted names for survivorship control;
  benchmark instruments (SPY + broad/sector ETFs) for excess-return labels;
  ALFRED vintage access (currently gated - macro series are
  latest-vintage-only until then).

## Structure

```
docs/                  Charter, seed hypotheses, gates, universe architecture
src/orbit/schemas/     HypothesisSpec / ExperimentSpec / Instrument / data contracts
src/orbit/universe/    Membership rules + delisting-aware reconstruction engine
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