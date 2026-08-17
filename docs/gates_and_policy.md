# ORBIT Promotion Gates and Research Policy

Version 1.0 - 17 August 2026 - Phase 1

## 1. Promotion Rule

A good backtest never promotes a model by itself. Promotion requires the
relevant scientific and operational gates below. Gates are prerequisites for
the next set of phases; a failed gate pauses the sequence.

## 2. Milestone Gates (from the roadmap)

| Gate | Phases | Required evidence |
|---|---|---|
| A Research Integrity | 1-6 | No known temporal leak in scope; immutable snapshots; labels frozen; registered experiments |
| B Simulator Integrity | 7-8 | Deterministic replay; accounting invariants; cost monotonicity; baseline controls |
| C Predictive Evidence | 9-14 | At least one simple model beats agreed baseline on locked OOS after costs, or a defensible null |
| D Statistical Integrity | 11-13 | Inference assumptions explicit; robust to sequential windows; no narrow single-regime explanation |
| E Decision Integrity | 15-17 | Every prediction has structured explanation; portfolio construction auditable; no order bypasses risk |
| F Paper Readiness | 18 | Order lifecycle and reconciliation pass failure-injection tests |
| G Live Paper | 19-20 | 30+ sessions, no critical incidents, complete replay and failure classification |
| H Long Paper | 26 | 60-120 sessions, locked config, no rescue tuning, stable operation |
| I Commercial Validation | 27-28 | Users repeatedly complete valuable research workflows faster/reliably than DIY |
| J Real-Money Consideration | 29-31 | Professional legal/compliance review, hardened controls, explicit economic rationale |
| K Strategic Decision | 32 | Predefined scorecard says scale, pivot or stop |

## 3. Model Promotion Policy (gate C and beyond)

A model may move from `candidate` → `validated` only when ALL of:

1. Economic evidence thresholds from its `HypothesisSpec` are met after costs
   on locked OOS data.
2. The effect survives the preregistered number of walk-forward windows and
   regimes.
3. Falsification obligation fulfilled: an adversarial failure test was run and
   reported (null data, shuffled labels, cost/slippage shocks).
4. Registry integrity: dataset snapshot, feature version, model spec, code
   hash and full genealogy are recorded and replayable.
5. The model beats its preregistered baseline, not merely a nominal threshold.

`validated` → `paper` requires gates E and F. `paper` → `live-data paper`
requires gate G. Promotion approval artifacts are archived.

Statuses: `candidate` → `validated` → `paper` → `retired`; `rejected` anytime.

## 4. Research Budgets

- Per hypothesis family: default 20 trials, 50 parameter sets, review after
  10 trials. Hard numbers live in `ResearchBudget` in the schema.
- Null results and failed experiments are registered and preserved. Lost
  negative experiments are a governance violation.
- The final holdout may be accessed at most once per locked research cycle
  (Phase 24).
- Multiple-comparison awareness is mandatory: any report of significance must
  state selection depth and FDR controls.

## 5. Paper-Only Policy

- All evaluation is fake money. No customer money, no real orders.
- Canonical `PaperBroker` is deterministic and internal; external paper
  brokers require an adapter and are never the primary research surface.
- A hard rule: no live-money phase without gate J (professional
  legal/compliance review) and a documented economic rationale.
- Phase 30 (proprietary capital) is optional, tiny, manually approved and
  strictly for measuring the paper/live gap.

## 6. Stop / Abandon Rules

Abandon or pivot if:

- No strategy family shows meaningful incremental value after realistic costs
  and strict OOS tests.
- AI search produces high false-discovery rates on null data or exploits
  known leakage.
- Data licensing costs exceed user value.
- Users prefer cheaper existing platforms and cannot articulate ORBIT's
  unique value.
- Paper operation repeatedly requires manual rescue.

Double down if:

- At least one strategy family shows repeatable after-cost OOS value across
  windows and regimes.
- AI hypothesis generation improves throughput while keeping false discovery
  controlled.
- Paper trading reproduces research characteristics without large unexplained
  gaps.
- Users repeatedly return for provenance/replay workflow rather than only the
  chat interface.
- A narrow buyer segment shows willingness to pay and retention.

## 7. Hard Stops

- Any known temporal leak: full stop on the affected pipeline until fixed.
- Accounting mismatch (cash/position) in the simulator: rebuild before
  research.
- Unexplained model identity or unreplayable experiment: invalid experiment.
