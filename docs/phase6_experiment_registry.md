# Phase 6 - Experiment Registry Foundation

ORBIT's permanent research-control plane: every experiment is registered
**before** it runs, carries an immutable scientific identity, a validated
lifecycle, complete lineage pins, and a reproduction specification that any
future researcher (human or AI) can resolve and replay.

## What an experiment is

An `ExperimentSpec` (Phase 1 schema, extended in Phase 6) is the canonical,
pydantic-validated declaration of one experiment:

- identity: `experiment_id`, `title`, `seed`, `content_hash()`
- hypothesis: `hypothesis_id` (must resolve in the Phase 1 registry),
  `hypothesis_family` (search-family for trial counting)
- genealogy: `parent_id` (acyclic, hypothesis-scoped)
- lineage pins:
  - data: `dataset_snapshot_ids` (exact `DS-xxxxxx` ids resolving in the
    Phase 3 dataset registry; descriptive names alone are refused)
  - temporal: `temporal_config` (Phase 4 engine version + config digest,
    must equal the digest of the loaded contract)
  - labels: `label_id` + `label_version` (Phase 5, pinned, never "latest")
  - features: `FeatureRef` list (`feature_id`, `feature_version`,
    `transformation`)
  - model: `ModelSpec` (family, hyperparameters, preprocessing,
    training_config, model_version)
  - cost model: `cost_model_id`
  - protocol: `evaluation_protocol`, `randomness_policy`
- operational fields (not part of the scientific identity):
  `status`, `code_hash`, `config_hash`, `created_at`, `registered_at`,
  `trial_number`, `number_of_prior_trials`

`content_hash()` hashes the scientific identity **excluding** the
operational fields. The registry stores it as an indexed column of the
FK-referenced `experiments` table, which makes DuckDB refuse every raw-SQL
UPDATE to it (`ConstraintException`) - content tampering is impossible even
with direct database access.

## Lifecycle

```
registered -> running -> completed -> (rejected | promoted)
                |            |
                v            v
              failed -----> retired  (terminal)
```

- `register()`: the only entry point. Registration is validated end-to-end
  (unique id, parent rules, hypothesis exists and is registered, research
  budget not exhausted, every lineage pin resolves) and then writes the
  identity **atomically** with its state row and lineage join rows. The
  ledger refuses every birth status except `draft`/`registered`: an
  experiment is born, never teleported into a later lifecycle state.
- `mark_running()`: requires the executing `code_hash` and `config_hash`
  BEFORE execution; once set, hashes are immutable. They may also be pinned
  at registration.
- `complete()` / `fail()`: normal completion or infrastructure failure.
- `record_decision()`: the ONLY path to `rejected` / `promoted`. Requires a
  completed experiment, a recorded result, a substantive reason (>= 10
  chars, not a placeholder), a decision maker and a policy version. The
  recorded-result requirement is enforced inside the ledger transaction
  (`ExperimentRegistry.record_decision`), not just in the service: a direct
  registry user cannot record a decision that cites no evidence.
- `retire()`: archival. The experiment leaves active research but its full
  history (identity, lineage, results, decisions, artifacts) is retained.
- REJECTED/PROMOTED/RETIRED are terminal; a bare `transition()` refuses the
  decision states with a loud error.

## Registry storage (DuckDB)

`ExperimentRegistry` owns one DuckDB file (`experiments.duckdb`):

- `experiments` - write-once identity rows (PK, FK parent, status CHECK,
  `CHECK (parent_id <> experiment_id)`, content-hash index; id/lineage
  columns carry regex CHECKs - `EXP-[0-9]{5}`, `H-[0-9]{3}`, `DS-[0-9]{6}`,
  `FEAT-[0-9]{3,}`, and `label_id IS NULL OR label_version IS NOT NULL`)
- `experiment_state` - status/code_hash/config_hash/updated_at (indexed,
  updated via optimistic `WHERE status = ?` guards)
- `experiment_datasets`, `experiment_features` - lineage join rows (FK-bound)
- `transitions`, `artifacts`, `results`, `decisions` - FK-bound records
  (`results.experiment_id UNIQUE`: one immutable result per experiment;
  `transitions` CHECKs `from_status`/`to_status` against the lifecycle's
  status set, so raw-SQL garbage rows are refused). Every child record also
  carries a `content_hash` (sha256 over its canonical content, computed at
  write time and CHECK-constrained to `[0-9a-f]{64}`): DuckDB's index trick
  does not protect these FK-referencing tables, so the hash is what makes a
  raw-SQL rewrite of a summary, decision reason, checksum or note detectable.
- `counters`, `trial_counters` - atomic id and per-family trial counters

Design note: DuckDB refuses an UPDATE of a column that carries a secondary
index on a table referenced by a foreign key. The two-table split
(identity vs state) is deliberate: identity columns are indexed (and thus
raw-SQL immutable), while state is separately updateable and protected by
optimistic guards and CHECK constraints.

Concurrency: one `ExperimentRegistry` per process/thread with the same
file is the intended layout; DuckDB serializes writers. The registry retries
lock contention - including `Conflict on tuple deletion` from concurrent
writers on the same row, which is retried until the status re-read settles
the race - and the trial counter is bumped atomically inside the
registration transaction (concurrent first registrations in one family
cannot collide on trial numbers). A writer that loses a race to a rival
write gets a clean, catchable `ValueError` from the post-retry status
check; the lossy DuckDB `TransactionException` never escapes the API.

## Trial and search depth

- `trial_number` / `number_of_prior_trials` are **computed by the registry**
  from an atomic per-family counter; a researcher-declared value that
  disagrees rolls the registration back.
- Trial ordinals never renumber: failed, rejected and retired experiments
  remain part of the search history.
- `count_trials(hypothesis_id)` counts live (non-retired) experiments per
  hypothesis and feeds the Phase 1 research-budget check
  (`research_budget.max_trials`). The budget is enforced atomically inside
  the registration transaction and is hypothesis-scoped: a `hypothesis_family`
  is a search-family label, not a budget-bypass vehicle.
- `list(label_version=...)` filters pinned label versions; the "latest"
  label is never resolvable from the registry or in reproduction.

## Reproduction specification

`ExperimentService.reproduction_spec(experiment_id)` resolves every lineage
element into a frozen `ReproductionSpec`:

- the experiment (content hash, code/config hashes, seed, trial depth)
- dataset snapshots (ids + checksums) - a snapshot that no longer resolves
  is a loud `lineage violation`, not a silent `None`
- temporal contract (engine version + digest)
- pinned label contract (version + content hash)
- feature refs, model spec, windows, cost model
- result, decision and artifact records

The `reproduction_digest` is a stable hash over the immutable core: pin the
code/config identity (or have it captured at `mark_running`) and the digest
never changes afterwards; decision and result records do not move it. A
replay test rebuilds an experiment's configuration from the ledger and
verifies the stored metrics match a fresh deterministic run.

## Invariant validation (audits)

`validate_invariants()` recomputes every content hash, walks the whole graph
for acyclicity, and counts orphans (every value must be 0). `orphan_counts()`
covers state, artifacts, results, decisions, transitions and lineage joins.
It also checks trial-ordinal contiguity: per search family the trial
numbers must be exactly `1..n` with no gaps or duplicates, so a tampered
trial counter is detected even though the counter row itself is mutable.

`validate_invariants()` also verifies state/audit-trail consistency - every
status must be *explained* by the recorded audit trail:

- the transition chain (ordered by `transitioned_at`, `rowid` for
  deterministic tie-breaks) must start at a birth state (`draft` /
  `registered` - an experiment is born, never teleported), stitch
  hop-to-hop (`prev.to_status == next.from_status`), take only
  lifecycle-valid steps, and end exactly at the current state;
- a status with no transition record at all (only `draft`/`registered`
  are legal without one) means the state was rewritten without an audit
  trail - e.g. a raw-SQL UPDATE of `experiment_state` (which no foreign
  key references, so no index protects it);
- decision states (`rejected`/`promoted`) must have exactly one decision
  row whose value matches the status;
- every child record's `content_hash` is recomputed over the stored row and
  compared: a raw-SQL rewrite of `results.summary`/`metrics_json`,
  `decisions.reason` (the cited evidence), `artifacts.checksum` or
  `transitions.note` is flagged as a content mismatch (timestamps are
  canonicalized so the ISO string written and the datetime DuckDB returns
  hash identically);
- the lineage join rows must equal the pinned spec: `experiment_datasets`
  and `experiment_features` are FK-referencing tables, so a raw-SQL UPDATE
  or DELETE of a join row would otherwise silently rewrite (or erase) the
  experiment's real data/feature lineage while the hash-protected `spec_json`
  still pins the original.

`register()` itself refuses any birth status other than `draft`/
`registered`: the ledger cannot create an experiment that already lives
in a later lifecycle state with no history behind it.

## Files

- `src/orbit/schemas/experiment.py` - canonical `ExperimentSpec`,
  `FeatureRef`, `TemporalConfigRef`, extended `FeaturePin`/`ModelSpec`
- `src/orbit/schemas/common.py` - `ExperimentStatus` (adds REJECTED, PROMOTED)
- `src/orbit/experiments/lifecycle.py` - transition table, decision states
- `src/orbit/experiments/registry.py` - DuckDB ledger
- `src/orbit/experiments/service.py` - `ExperimentService` API
- `src/orbit/experiments/reproduction.py` - `ReproductionSpec`
- `tests/test_phase6_registry.py`, `test_phase6_reproducibility.py`,
  `test_phase6_invariants.py`, `test_phase6_concurrency.py`,
  `test_phase6_audit.py`, `test_phase6_revision.py` (audit regression
  suite: label pinning, feature lineage, decision-cites-evidence,
  hypothesis-scoped budgets, lifecycle enforcement, DB CHECKs, trial
  contiguity, registry-level decision guards, decision-vs-retire races,
  state/audit-trail consistency, birth-status restriction, child-record
  content integrity, lineage join-row verification)
- `scripts/phase6_demo.py` - end-to-end example

Run the Phase 6 suite: `python -m pytest tests/test_phase6_*.py -q`