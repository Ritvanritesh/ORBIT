# ORBIT Phase 4: Point-in-Time & Temporal Truth Engine

Version 1.0 - 17 August 2026 - Phase 4

## 1. Purpose

Phase 4 makes the research surface temporally honest. Every historical
prediction must be computable strictly from the information that existed at
the decision time `t`. The core invariant, enforced everywhere:

```
allowed(x, t)  <=>  publication_time(x) < t        (STRICT - ties are rejected)
```

A record that was not public before `t` is never visible at `t`, regardless of
how the data was stored, when it was ingested, or whether a later revision
exists. One known leak is a hard stop (see `docs/gates_and_policy.md`).

## 2. The five temporal fields

Every normalized record is classified at the point it enters the temporal
engine (`TIMING_SCHEMA` in `src/orbit/temporal/adapters.py`):

| Field | Meaning | Default if unknown |
|---|---|---|
| `event_time` | when the event/observation economically occurred (bar session, period end, observation month) | required |
| `publication_time` | when the record became public; never invented | required - missing => REJECTED |
| `publication_precision` | `instant` or `date` (time-of-day unknown) | `date` |
| `effective_time` | when the record becomes applicable (filing: at filing; daily bar: session close) | optional |
| `ingestion_time` | when WE downloaded/stored it; provenance only, never availability | optional |

`vintage_id` / `vintage_date` identify which release of a revised observation a
record is (`vintage_date` = ALFRED vintage date; `null` = single version).

## 3. Timestamp conventions (canonical, in `times.py`)

- All instants are normalized to **naive UTC** through the single point
  `normalize_instant()`. Naive = UTC; aware instants are converted.
- **Date-precision publication** (e.g. SEC `filed`, ALFRED vintage date):
  available only from the **next day 00:00 UTC** (`next_day_midnight`). This is
  the conservative assumption that the record became public at some unknown
  time on its date, and buys the guarantee that a filing is never usable on
  its own filing day.
- **Date-only `as_of`** means the start of the day (00:00 UTC).
- **Market bars** (Yahoo daily): a bar for session `D` is available at session
  close on `D` — 16:00 America/New_York converted to UTC (21:00 UTC in EST,
  20:00 UTC in EDT, via `session_close_utc()`). We never use `ts_utc`, because
  for Yahoo daily bars `ts_utc` is the session **open** (14:30 UTC), which
  would leak the whole session.
- DST is handled by real zone conversion (America/New_York), never by fixed
  offsets.

## 4. Availability decisions (`rules.py`)

Rules are evaluated in order; the first match decides:

1. `NOT_POINT_IN_TIME` - a `revised` series (e.g. CPIAUCSL, UNRATE) with no
   vintage history cannot be used point-in-time (no ALFRED access). Rejected
   and reported as a snapshot limitation. `DFF` is `non_revised`: its releases
   are never revised, so the next-day record is the truth.
2. `MISSING_PUBLICATION_TIME` - publication is unknown; availability is never
   invented. Rejected.
3. `PUBLICATION_AT_OR_AFTER_AS_OF` - publication >= `t`. Rejected
   (STRICT boundary: an exact tie is rejected too).
4. `NO_VINTAGE_AT_AS_OF` - the only known version was released on/after the
   as-of date. Rejected: no version of this observation exists yet.
5. `EVENT_AFTER_AS_OF` - the event itself happened after `t` (forward-dated
   data). Rejected.
6. Allowed - code `ALLOWED_BEFORE_PUBLICATION` (instant precision),
   `ALLOWED_DATE_PRECISION` (next-day convention applied), or
   `ALLOWED_VINTAGE_RESOLVED` (a specific vintage was selected).

Warnings (never exclusion): `INGESTED_AFTER_AS_OF` (delayed ingestion -
ingestion time is provenance, not availability) and `DATE_PRECISION_NOTE`.

Every decision carries a human-readable `decision_detail` and a warning list;
`trace_record()` returns the full rule-by-rule trace for a single record.

## 5. Vintage resolution (`_resolve_vintages`)

For revised series with multiple releases of the same observation:

- Only versions **released AND allowed** at `t` can be the "latest".
- The version with the largest `vintage_date < as_of.date()` wins per
  `(series_id, observation_date)`.
- Older released versions are **audited, not dropped**: they appear in the
  snapshot's `excluded` records with `VINTAGE_SUPERSEDED` and their full
  decision trail, so a reviewer can see exactly which release was discarded.
- A rejected version (not public at `t`) can never become "latest".

## 6. Point-in-Time snapshots (`snapshot.py`, `engine.py`)

`TemporalTruthEngine.snapshot(as_of)` returns a `PointInTimeSnapshot`:

- `records` - the full information set allowed at `as_of` (bars, facts,
  series versions) with the decision trail attached;
- `excluded` - everything rejected, with reasons (audit surface);
- `decision_counts()` - one row per decision code;
- `content_digest` - deterministic SHA-256 over the sorted content
  (`created_at` is excluded), enabling equality checks and replay;
- `provenance()` - the exact `TemporalSource` set (snapshot id, domain,
  provider, checksum, manifest path, artifact paths) that produced the
  information set;
- `to_json()` - full serializable record for the registry.

Sources are resolved from normalized parquet artifacts (bars, facts, series)
or from explicit `TemporalSource.artifact_paths`.

`asof_join(right, left)` implements the phase-16 temporal join: each left row
(decision at `decision_time`) is joined to the most recent right-side
observation available at that instant, with full gating, vintage resolution,
and deterministic tie-breaks (largest `record_id`). Rows with no qualifying
right record keep their left columns with NULL right columns.

## 7. Feature-time rules (`features.py`)

- `completed_bars(bars, as_of, n)`: the `n` most recent bars whose sessions
  COMPLETED before `as_of` (session-close semantics) - never the in-progress
  session.
- `assert_no_future_refs(frame, as_of)`: raises `FutureRefViolation` if any
  bar/fact row's time exceeds `as_of`. `Date` columns are checked with
  session-close semantics; `Datetime` columns with strict instants. This is
  the last line of defense for hand-built features.

## 8. Synthetic leak fixtures (`fixtures.py`)

Permanent regression fixtures - each is a deliberately leaky dataset with a
known violation that MUST be caught:

- `future_earnings_fixture` - earnings announced after the decision time.
- `future_filing_fixture` - 10-K filed after the decision time.
- `future_macro_revision_fixture` - a macro revision published after the
  decision time (must not replace the original).
- `future_price_fixture` - bars whose session closed after the decision time
  (including the in-progress session).
- `delayed_ingestion_fixture` - data ingested after `t`: allowed, with a
  warning; availability must come from publication, not ingestion.
- `missing_publication_fixture` - no publication time: rejected, never
  guessed.
- `future_feature_fixture` - a pre-computed feature column that references
  future bars; must trip `assert_no_future_refs`.

## 9. Macro schema v1.1.0

`normalize_fred_series` gained an optional `vintage_date` (ALFRED vintage
date). v1.0.0 artifacts (no column) remain fully supported: the adapter
detects the column and treats the series as single-version. Until ALFRED
vintage access is wired up, revised series are rejected point-in-time
(`NOT_POINT_IN_TIME`) and recorded as snapshot limitations - the honest
failure mode, not a silent approximation.

## 10. Testing strategy

- `tests/test_phase4_rules.py` - record-level decisions: strict boundary
  (tie rejected), date-precision next-day, DST/session-close conversions,
  tz-aware as-of equivalence, missing publication, forward-dated events,
  revised-without-vintage, superseded auditing, adapter column handling.
- `tests/test_phase4_engine.py` - snapshot information sets, reproducibility
  within/across engines, digest stability, provenance, decision counts,
  latest-available, historical vintage retrieval, as-of joins across the full
  matrix (gates, delayed ingestion, multiple versions, null joins).
- `tests/test_phase4_leaks.py` - every synthetic leak fixture must fail the
  engine; pipeline end-to-end: ingest two vintages of the same observation
  and resolve them point-in-time; SEC pipeline snapshots must exclude a
  future filing; a latest-vintage ingest without ALFRED must be rejected.

Run: `python -m pytest tests -q` (168 tests, Phase 1-4).

## 11. Known limitations (honest list)

- SEC `filed` is date-only: the true intra-day publication instant is
  unknown, so availability is the next day 00:00 UTC (conservative).
- Yahoo daily bar availability is the session close; intraday resolution is
  not available from this source.
- `adjclose` (retroactively dividend/split-adjusted) is EXCLUDED from bar
  timing payloads: an adjclose read at as_of can differ from what was
  publishable then. Point-in-time prices come from `close` plus the
  corporate-action adjustment records.
- Revised FRED series are point-in-time blocked until ALFRED vintage access
  exists (v1.1.0 schema is ready for it).
- Twin releases with an identical vintage_date for the same observation
  share a record_id; the pipeline fingerprints dedupe identical ALFRED
  requests, so this is only reachable through hand-crafted artifacts.
- `ingestion_time` is stored at day granularity (day start): the
  `INGESTED_AFTER_AS_OF` warning fires on a coarse day boundary.
- The as-of join is a grouped cross-product (correctness first); optimize
  only after profiling.