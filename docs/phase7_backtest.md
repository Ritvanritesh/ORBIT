# ORBIT Phase 7: Event-Driven Backtesting Engine

Version 1.0 - 19 August 2026 - Phase 7

## 1. Purpose

Phase 7 implements a deterministic, replayable, long-only daily/EOD execution
and accounting simulator. It produces the event facts that Phase 5 labels
and Phase 6 experiments operate on. Everything is deterministic: the same
bars + same sessions produce the same event stream.

The clock defines exactly, per session: when a market observation becomes
available, when a signal may be generated, when an order may be submitted,
when a simulated order can fill, and when portfolio valuation occurs.

## 2. Core Semantics

### Market Event Clock (`MarketEventClock`)

The deterministic session clock over a canonical bars frame. Key methods:

- `sessions()` - returns all calendar sessions in the data
- `has_bar(instrument_id, session)` - check if instrument has a bar on a session
- `bar(instrument_id, session)` - return the canonical BarRecord (NaN for defective OHLC, never TypeError)
- `last_close(instrument_id, session, strict=False)` - most recent close at or before session; if `strict=True`, close must be strictly before session
- `availability_instant(session)` - session close (Phase 4 convention, `session_close_utc(session)`)
- `execution_instant(session, at_open)` - session open for open fills, session close for close fills
- `window_sessions(start, end)` - sessions in [start, end] (None bounds are open)
- `normalize_signals(signals)` - normalize signals input (polars DataFrame or list of dicts)

### BarRecord

Canonical daily bar as the clock serves it.

- `price_basis` is always `"split_continuous_stored"` (never `adjclose`)
- `volume_basis` is either `"as_published"` or `"provider_stored"`
- OHLC fields served as NaN when null; guarded by `_missing_price`/`_missing_volume`

### Duplicate Bar Detection

`MarketEventClock.__init__` raises `ValueError` if duplicate `(instrument_id, trade_date)` pairs are found in the bars DataFrame.

## 3. Cost Model

Fill cost model: `fill = reference × (1 ± (spread_bps + slippage_bps) / 1e4)`

- Spread: direction-aware (`+` for buys, `-` for sells)
- Slippage: direction-aware (`+` for buys, `-` for sells)
- Fee: `notional × fees_bps / 1e4 + fixed_fee`, floored at `fee_minimum`
- Fees never capitalized into `avg_cost` (avg_cost = execution notional only)
- Equity identity: `equity == initial_cash + realized + unrealized − fees_total`

### Sizing Policies

- `QUANTITY` - signal target shares (whole shares floored)
- `WEIGHT` - fraction of equity, floor to whole shares

## 4. Failure Taxonomy

Orders may be rejected with these reasons:

- `LIQUIDITY_CAP` - participation_fraction * volume cap
- `MAX_ORDER_QUANTITY` - separate cap from participation cap
- `INSUFFICIENT_CASH` - sell cash check: fee > proceeds, no implicit loan
- `ORDER_EXPIRED` - order beyond `order_expiry_sessions` window
- `CANNOT_SELL_MORE_THAN_HELD` - sell exceeds held position
- `ORDER_TOO_LATE_FOR_DELAY` - signal deadline missed

### Cash Invariants

- `cash >= 0` enforced in `validate_invariants()`
- Equity identity: `equity == initial_cash + realized + unrealized - fees_total`
- Partial fills floored to whole shares with `unfilled_quantity`/`unfilled_reason`

## 5. Phase 4 Integration: Temporal Truth

- `validate_signal_temporality()` must be called before lifecycle processing
- Signal `decision_time == session_close_utc(signal_session)` exactly; refused otherwise
- Signal temporal validation gates the full lifecycle

## 6. Phase 5 Integration: Label Engine

- `LabelEngine` entry anchor (`DECISION_INSTANT`) requires a completed bar strictly BEFORE `decision_time`
- `SIMPLE_TOTAL_RETURN` contracts REQUIRE an events artifact (empty schema-valid frame ok)
- `predicted / realized / executed` kept separate in integration output

## 7. Phase 6 Integration: Experiment Lifecycle

- `run_backtest_experiment(config, universe, ...)` refuses any cost config !=
  `CostConfig.from_cost_model(exp.cost_model)`
- `service.result()` records metrics under `metrics_json` (JSON string)
- `service.transitions(id)` reads notes via `service.transitions(id)`
- Experiment status tracking: `DRAFT / RUNNING / COMPLETED / FAILED / REJECTED`
- `run_backtest_experiment()` seed/cost/dataset_snapshot_ids validated
- Artifact attachment: `service.attach_artifact()` events + manifest

### Run Identity

- `run_id = BT-<config8>-<content12>` via `derive_run_id()`
- `signal_set_hash` required (min_length 32), order-independent (sorted by `(session, id)`)
- `content_hash` excludes `created_at`/`run_id`; `canonical_json` uses `model_dump_json(exclude=_OPERATIONAL_FIELDS)`
- Manifest excludes `created_at`; `model_dump_json(exclude={'created_at'})`

## 8. Accounting Identity

```
equity == initial_cash + realized + unrealized - fees_total
```

This invariant is enforced in `ledger.validate_invariants()` and
`result.invariant_violations()`.

## 9. Key Constraints

- Long-only only (shorting refused loudly; `OrderSide.SELL` constrained)
- Order IDs start at `ORD-000000` (sequential integer IDs)
- Default execution: `open, delay=1` (validator refuses `delay=0` with OPEN)
- Expiry: `eligible + order_expiry_sessions`; expired when `session > expiry_session`
- Partial fills floored to whole shares with explicit `unfilled_quantity`/`unfilled_reason`

## 10. API Summary

### Clock

- `MarketEventClock(bars, events=None, volume_basis=None)`
- `sessions()`, `has_bar()`, `bar()`, `last_close()`, `availability_instant()`, `execution_instant()`, `window_sessions()`, `normalize_signals()`

### Cost

- `CostConfig(spread_bps, fees_bps, slippage_bps, fixed_fee, fee_minimum)`

### Config

- `BacktestConfig(universe, window_start, window_end, costs, sizing, execution, benchmark, ...)`

### Result

- `result.invariant_violations()` - checks cash >= 0, accounting identity
- `result.rejections` - list of rejection events
- `result.fills` - list of fill events
- `result.valuations` - valuation events with `benchmark_return`

### Signal Normalization

- `MarketEventClock.normalize_signals(signals)` - returns list of canonical signal dicts