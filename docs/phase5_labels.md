# Phase 5 — Labeling & Outcome Engine

Every prediction target in ORBIT is one immutable, versioned `LabelContract`;
the `LabelEngine` computes exactly one reproducible future outcome per
decision row, and a `LabelSnapshot` pins the batch with a deterministic
digest. The engine runs entirely on Phase 3 canonical market data and
enforces the Phase 4 temporal-truth conventions: it never touches a bar
before its session close, and it never invents an outcome.

## Module layout (`src/orbit/labels/`)

| module       | responsibility |
|--------------|----------------|
| `contract.py`| `LabelContract` + all semantic enums + the Phase 1 `LabelSpec` bridge |
| `outcomes.py`| pure outcome mathematics (golden-testable formulas) |
| `engine.py`  | `LabelEngine`: entry/window resolution, delisting, benchmark, overlap metadata, unavailable reasons |
| `snapshot.py`| `LabelSnapshot`: deterministic content digest + provenance |
| `registry.py`| `LabelVersionRegistry`: immutability and version resolution |
| `seeds.py`   | registered contracts for the seed hypotheses H-001 / H-003 |

## One definition: the contract

A `LabelContract` is a frozen pydantic model. Every semantics-bearing field
is required (nothing inferred from prose, nothing silently defaulted):

- `label_id` `LAB-\d{3}`, `version` `v\d+(\.\d+)*`
- `target_type` — `forward_return`, `excess_return`, `volatility`,
  `drawdown` (Phase 1's `risk_adjusted_return` is REJECTED: it is a
  composite whose assembly is deferred; its components exist here)
- `horizon` ≥ 1, `horizon_semantics = "trading_sessions"` — H1/H5/H21/H63
  from Phase 1 map to 1/5/21/63 **sessions**; calendar gaps (weekends,
  holidays, gaps in the bar series) never count as sessions
- `anchor_mode` — `decision_instant` (last completed bar strictly before
  the decision time) or `post_event` (first completed bar strictly after
  the event's availability instant — the PEAD anchor)
- `return_convention` — `simple_price_return` / `simple_total_return`
  (both close-to-close; total return reinvests ex-date dividends at the
  ex-date close)
- target-specific: `benchmark` (excess), `volatility_estimator` +
  `annualization` + `min_observations` (volatility),
  `drawdown_type` (drawdown: max drawdown or max adverse excursion).
  Volatility and drawdown targets are defined on close-to-close **price**
  returns only: a `SIMPLE_TOTAL_RETURN` convention is rejected by the
  validator (the current estimators have no dividend term).

`content_hash()` = sha256 of the canonical JSON (includes the version);
`definition_identity()` excludes the version so the registry can refuse a
version bump that changes nothing. `contract_from_hypothesis_label()` maps
a Phase 1 `LabelSpec` into a contract without touching the hypothesis
registry.

## One horizon

The outcome window is the next `horizon` sessions **strictly after** the
reference session, counted on the instrument's own session series. If fewer
than `horizon` sessions exist, the label is **unavailable with an explicit
reason** — the horizon is never silently shortened and missing future
prices are never filled. `window_start_session`/`window_end_session`
record the inclusive span [entry, outcome].

## One reproducible outcome

### Prices and corporate actions (canonical basis)

- The engine requires the Phase 3 canonical normalized bars
  (split-continuous closes, `adjustment` ∈ adjusted labels) and rejects a
  raw-basis provider loudly: a raw series would create artificial returns
  across splits.
- Returns are computed on the stored split-continuous basis; splits inside
  the window never create artificial returns.
- `entry_close_as_published` / `outcome_close_as_published` record the
  as-published closes (rebuilt through the same `as_published_bars`
  reconstruction the Phase 4 temporal layer uses) for audit.
- Total-return labels require the corporate-actions events artifact; a
  price-return label ignores dividends exactly as the convention promises
  (a regression test pins that both conventions share one formula and
  differ only in the dividend term). The events artifact is the
  authoritative dividend ledger per instrument: an instrument with no
  dividend rows in the artifact is treated as paying no dividends (the
  artifact's absence is the only "missing dividend data" state, and it is
  a hard error, never a silent price-return label).
- Ex-date dividends inside `(entry, outcome]` count, converted to the
  stored share basis (`raw amount / split factor at the ex-date session`),
  and are reinvested at the ex-date close. `total_dividends` is the audit
  column (0 for price-return labels).
- A corporate-action event missing its ratio/timestamp makes the label
  unavailable (`corporate_action_data_incomplete`) — the basis must be
  established before any return is computed.

### Timing (Phase 4 conventions, verbatim)

- A bar is available at its session close (`session_close_utc`: 16:00
  America/New_York → UTC, DST-aware). The comparison is **strict**:
  `publication_time < decision_time`; a decision at exactly the close
  uses the previous session (golden G10).
- `decision_instant`: entry = last completed bar strictly before the
  decision; a decision on a non-trading day uses the last completed
  session; before the first session → `no_entry_bar`.
- `post_event`: entry = first session whose close is strictly after the
  anchor instant (matches the Phase 4 availability convention for
  filings: a filing available at midnight of day D anchors the window on
  session D).
- Naive datetimes are UTC by convention (same as Phase 4
  `normalize_instant`); date inputs mean start-of-day UTC.

### Delisting, missing data, unavailable reasons

An unavailable label has `outcome_value = NULL` and exactly one reason
(never a fabricated zero, never a substitute):

`no_entry_bar`, `missing_anchor`, `missing_entry_price`,
`insufficient_future_data`, `missing_outcome_price`,
`missing_window_price`, `delisted`, `benchmark_unavailable`,
`insufficient_observations`, `missing_dividend_data`,
`corporate_action_data_incomplete`.

A security whose bars end is `delisted` only when the instrument master
records a delisting and the last bar session is ≤ the delisting date;
otherwise `insufficient_future_data`. Bars extending past the recorded
delisting date contradict the record and are classified as a data
shortfall.

### Excess returns

The benchmark is resolved in the engine's own bars universe under the same
contract (same anchor, same horizon, same convention), so asset and
benchmark always share the identical window. Missing/sparse benchmark
bars → `benchmark_unavailable`. The seed contracts (H-001/H-003) require
the SPY series (Phase 2's benchmark instrument set is a documented
follow-up).

## One reproducible snapshot

`LabelSnapshot` pins a batch: `label_id`, `version`, `contract_digest`,
`engine_version`, `data_refs`, the canonical 34-column label frame, and a
sha256 `content_digest` over the sorted content (wall-clock excluded —
`created_at` is recorded but never part of identity). `equals()` compares
identity; `provenance()` exposes row/available/unavailable counts.
`LabelSnapshot` is a **separate artifact** from `PointInTimeSnapshot`: a
point-in-time snapshot never contains label rows (the outcome is only
finalized at the outcome instant), and features are built strictly from
point-in-time data — the two are never merged into one unrestricted
dataset (integration tests enforce this).

## Versioning (immutability)

`LabelVersionRegistry.register()` refuses:

- re-registration of an existing `(label_id, version)`;
- out-of-order versions (new versions must be strictly newer);
- a version whose definition is identical to a registered one (no silent
  version inflation) — detected via `definition_identity()`, which
  excludes the version number.

Historical experiments pin `(label_id, version)`; `definition()` returns
the exact frozen contract, `definition_digest()` the formula identity.
Label rows carry `contract_digest` and `engine_version`, so every outcome
value can be traced to the exact definition and engine that produced it.

## Overlap metadata

Every row carries the window span; `overlapping_pairs(frame,
sessions_by_instrument)` identifies same-instrument label pairs whose
windows overlap on **session dates** (inclusive intervals; the boundary
session shared by adjacent windows counts). It is identification only —
purging, embargo and statistical machinery are later phases.

## Determinism

Same bars + same contract + same decisions → identical frame row for row
and identical snapshot digest, across engine instances and decision
orders (tests pin this). The only nondeterminism anywhere is `created_at`
wall-clock, excluded from identity.

## Seed contracts

`build_seed_label_registry()` registers:

- **LAB-001** (H-001 momentum): 5-session forward excess total return vs
  SPY, decision-instant anchor.
- **LAB-003** (H-003 PEAD): 5-session forward excess total return vs SPY,
  POST_EVENT anchor (first session after the filing's point-in-time
  publication instant).
- **LAB-002** (H-002 risk-adjusted) is intentionally **not** registered
  (deferred composite).

## Golden tests

`tests/test_phase5_golden.py` pins hand-computed values: ±10% returns,
excess vs benchmark, exact 5-session horizon, calendar-gap semantics,
unavailable-never-shortened, overlap metadata, the 7:1 split canonical
identity (`stored return == as-published ratio × split factor − 1`),
delisting, strict boundary (at-close vs one-microsecond-after), realized
volatility `sqrt(5.04)`, max drawdown `10/105`, MAE `0.05`, ex-date
dividend `1.02 × 103/101 − 1`, and split+dividend basis consistency
(stored-basis and raw-basis totals agree).

## Testing summary (Phase 5)

80 tests: 14 golden hand-calculation, 15 contract validation, 27 engine
behavior, 17 versioning/determinism, 7 Phase 4 integration
(entry-bar agreement, as-published agreement, no-leakage, artifact
separation, POST_EVENT availability convention).