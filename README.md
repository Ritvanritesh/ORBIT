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
- Awaiting charter sign-off before Phase 3 (Raw Historical Data Ingestion).

## Structure

```
docs/                  Charter, seed hypotheses, gates, universe architecture
src/orbit/schemas/     HypothesisSpec / ExperimentSpec / Instrument / data contracts
src/orbit/universe/    Membership rules + delisting-aware reconstruction engine
hypotheses/            Registered seed hypotheses (validated instances)
src/orbit/             Future phases: data, temporal, features, models, backtest
tests/                 Validation tests
```

## Rules

- Paper trading only. No customer money, ever (Phase 1 policy).
- Pre-register hypotheses before feature exploration.
- The final holdout is quarantined and never a tuning surface.
- Any known temporal leak is a hard stop.

See `docs/` for the full charter, gates, and seed hypotheses.