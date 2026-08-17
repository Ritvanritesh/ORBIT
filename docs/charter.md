# ORBIT Research Charter

**Optimized Research & Behavioral Intelligence Trading**
Version 1.0 - 17 August 2026 - Status: Phase 1

## 1. Mission

ORBIT is a reproducible scientific instrument for discovering, testing,
falsifying, explaining and replaying trading hypotheses. ORBIT is not a trader.
The model is replaceable; the evidence chain is the product.

Core loop: OBSERVE → HYPOTHESIZE → DEFINE LABEL → BUILD FEATURES → TRAIN →
TEST → FALSIFY → ROBUSTNESS → PORTFOLIO → RISK → PAPER → DIAGNOSE → UPDATE.

## 2. Scope

- Market: US equities, daily/end-of-day frequency first.
- Universe: roughly 50-100 highly liquid names plus broad/sector ETFs,
  reconstructed historically (delisting-aware, survivorship-controlled).
- Data: licensed daily OHLCV + corporate actions, SEC EDGAR/XBRL fundamentals,
  FRED/ALFRED macro vintages. Text and alternative data are deferred (Phases
  21+), and only with a preregistered hypothesis justifying cost.
- Evaluation: sequential walk-forward, quarantined final holdout, after-cost
  metrics, regime-stratified robustness.

## 3. Non-Goals (Phase 1, binding)

- High-frequency trading or exchange colocation.
- Options as the primary research market.
- Order-book prediction before low-frequency integrity is proven.
- A giant end-to-end transformer over all modalities.
- Autonomous LLM agents with unrestricted backtest loops.
- Customer-money trading or personalized automated portfolio management.
- A public strategy marketplace.
- Expensive alternative datasets without a preregistered hypothesis.
- Real-time news before point-in-time fundamentals and event timestamps are
  correct.
- Complex portfolio optimizers before simple constructors are benchmarked.
- A polished mobile app before the research workflow is proven.

## 4. Universe Philosophy

- Universe membership is a *historical fact*, not a today-snapshot fact.
  Selection must be reconstructable for any evaluation date.
- Survivorship bias is controlled via historical membership lists and
  delisting-aware treatment (last price retained, returns terminated).
- Liquidity filters use lagged information only (no look-ahead).
- Universe changes require a versioned `DatasetSnapshot`; every experiment
  pins the exact snapshot it consumed.
- The MVP universe is deliberately narrow; expansion is gated on data and
  execution integrity, never on hope.

## 5. Horizons and Labels

- First horizons: 1, 5, 21, 63 trading days (daily bars).
- Label types: forward return, excess return (vs benchmark), risk-adjusted
  return, volatility, drawdown.
- Every label has: exact mathematical definition, horizon, benchmark, overlap
  policy (purge+embargo), and a version string. Labels are frozen once used;
  a changed label is a new label version, never a silent mutation.
- Golden test cases must match hand calculations before a label is trusted
  (Phase 5).

## 6. Evaluation Windows

- Global evaluation period: **2010-01-01 to the most recent completed
  quarter** before a research cycle locks. Everything before 2010 is
  reserved for feature sanity checks, never for promotion evidence.
- Each hypothesis pins its own evaluation windows at registration (in
  `WindowSpec` per experiment); windows are frozen and cannot shift to
  chase results.
- Evidence must be sequential walk-forward across the global period with
  purge/embargo for overlapping labels.
- The 2007-2009 crisis and the 2020 shock are not primary evaluation windows
  but are mandatory regime-stress scenarios (Phase 13).

## 7. Evidence Definitions

Two distinct evidence types. They are never conflated.

### Research-quality evidence
Statistically meaningful after proper inference: dependence-aware (block
bootstrap, autocorrelation-aware), multiple-comparison-aware, with explicit
effect sizes and confidence intervals. This alone does NOT justify a model.

### Economic evidence
Research-quality evidence that ALSO survives:

- Conservative cost assumptions (default: 2.0 bps spread + 1.0 bps fees +
  2.0 bps slippage per side).
- Sequential walk-forward OOS windows (min 4, default thresholds per
  hypothesis).
- Multiple market regimes (bull/bear/sideways, high/low vol; min 2-3).
- Promotion thresholds preregistered in each `HypothesisSpec` before any
  experiment runs (e.g. OOS rank IC and after-cost annual excess).

Rule: ORBIT never says "statistically significant" and stops. A tiny effect
that disappears after costs is not a trading result.

## 8. Paper-Only Policy (binding)

- No real capital. Ever, under Phase 1. Live-money is optional, gated by
  evidence, reliability, economics and professional legal/compliance review
  (Phase 29), and never uses customer money.
- The canonical simulator is the deterministic internal `PaperBroker`; an
  external paper broker is only a second validation environment behind an
  adapter interface.
- Paper trading precedes any consideration of real capital; any live-money
  phase requires the milestone gates in `docs/gates_and_policy.md`.

## 9. Governance Rules

- Pre-registration: every hypothesis and experiment is fully specified before
  execution. Criteria cannot move after results appear.
- Genealogy: every experiment references a parent; nothing is orphaned.
- Research budgets: maximum trials per hypothesis family (default 20), review
  after 10. No free reruns.
- Holdout quarantine: the final test set is frozen and is never a tuning
  surface.
- Falsification obligation: a promising result requires an adversarial failure
  test (null data, shuffled labels, cost shocks).
- Stop rule: repeated failure of a family requires a new thesis, not more
  tuning.
- A known temporal leak is a hard stop, not a bug ticket.

## 10. Institutional Lines

| Layer | May do | May not do |
|---|---|---|
| Data | Acquire, validate, version, timestamp | Silently mutate historical truth |
| Feature | Compute deterministic values | Read future observations |
| Model | Predict/rank/estimate uncertainty | Place orders |
| Explainability | Attribute and test contributions | Invent facts or causal claims |
| Portfolio | Translate scores to target positions | Ignore risk policy |
| Risk | Allow/reduce/reject | Rewrite model provenance |
| Execution | Simulate orders/fills | Generate alpha |
| Research agent (late) | Draft hypotheses/specifications | Unlock holdouts or mutate until profitable |

## 11. Success and Failure for the Charter Itself

Charter approved when: this document is signed off, three seed hypotheses are
registered (see `docs/seed_hypotheses.md` and `hypotheses/seeds.py`), and
gates are documented (`docs/gates_and_policy.md`).

Charter failed if: labels or evaluation windows are ambiguous, or the research
question moves after results appear.
