# ORBIT Universe and Data Architecture

Version 1.0 - 17 August 2026 - Phase 2

## 1. Goal

The canonical instrument model and a scientifically defensible first US-equity
universe. The universe is a *historical fact*, not a today-snapshot fact: its
membership must be reconstructable for any evaluation date without
survivorship bias.

## 2. Why This Exists

Universe selection is a major source of survivorship and selection bias. A
static current-day ticker list silently drops delisted names, making history
look easier than it was. This layer makes that impossible by construction.

## 3. Components

### 3.1 Instrument master (`src/orbit/schemas/instrument.py`)

- `Instrument` - stable identity (`INS-xxxxxx`), primary ticker, exchange,
  security type, listing/delisting dates and reasons, sector, currency.
  Identity is immutable and survives ticker changes.
- `SymbolHistory` - every ticker an instrument traded under with effective
  dates; resolves which symbol was valid on any date (ticker-identity errors
  are a stated failure condition of Phase 2).
- `Exchange` - metadata incl. IANA timezone and local session hours, used for
  UTC bar timestamps (Phase 3/4).
- `CorporateAction` - splits, dividends, mergers, spin-offs, delistings,
  symbol/name changes with effective/ex dates (Phase 3 reconciliation).
- `SectorTaxonomy` - GICS-style sector hierarchy.
- `Benchmark` - benchmark instruments (SPY + broad/sector ETFs) referenced by
  hypotheses and evaluation.

### 3.2 Data contracts (`src/orbit/schemas/data.py`)

- `DatasetSnapshot` - immutable, checksummed dataset deliveries
  (`snapshot_id, provider, source_uri, checksum, schema_version,
  available_from/to, ingest_time, license_ref`). Every experiment pins the
  snapshot it consumed.
- `MarketBar` - normalized daily bar (UTC timestamp, OHLC consistency
  validated).

### 3.3 Universe selection (`src/orbit/universe/`)

- `MembershipRule` - one versioned, immutable rule
  (`RULE-xxx/vN`): security types, exchanges, min price, min trailing median
  dollar volume, liquidity window (default 20 days), max names (default 100).
  Any logic change is a new rule version - never a silent mutation.
- `DataAccessor` - protocol for read-only, strictly-lagged market data.
  Implemented over the raw data layer in Phase 3; a synthetic accessor
  proves the engine in Phase 2.
- `UniverseEngine` - deterministic reconstruction. For any `as_of`:
  1. Eligible = listed on or before `as_of` AND (not delisted or delisted
     after `as_of`) - delisting-aware.
  2. Filter by security type / exchange.
  3. Require lagged price history and min price (last close strictly before
     `as_of`).
  4. Require median trailing dollar volume over the window strictly before
     `as_of` (lagged liquidity - a liquidity regime change only enters
     membership once its days actually fill the window).
  5. Rank by dollar volume desc (tie-break by instrument_id for determinism),
     cap at max names.
  - Every exclusion records a machine-readable reason (distinct reasons for
    future listings vs delisted names). Same inputs always produce the same
    snapshot (covered by a determinism test).
  - `data_ref` is REQUIRED: every snapshot pins the dataset/accessor version
    it was computed from - an unreferenced universe is ungovernable.

## 4. Rule v1 (MVP)

- Universe: up to 100 most liquid NYSE/NASDAQ-listed US equities (typically
  50-100 in practice)
- Security types: equity (excluded: ADR/preferred/units for the first pass)
- Min price: $5.00 (lagged last close)
- Min trailing median dollar volume: $20M/day over 20 days (lagged)
- Benchmarks: SPY plus broad/sector ETFs (separate `benchmark` rule class)

## 5. Survivorship Controls

- The instrument master MUST include delisted instruments; the engine refuses
  nothing on that basis - it filters by date.
- Delisted names with full price history appear in membership up to (not
  including) their delisting date - verified by test.
- Returns for delisted names terminate at the delisting price; delisting
  causes are recorded for the outcome store (Phase 5+).

## 6. Done When

- Historical universe can be reconstructed for any evaluation date.
- Delisted names disappear exactly at delisting; future listings never appear
  early; liquidity is strictly lagged; exclusions are always reasoned;
  membership is deterministic.

## 7. Failure Conditions (guarded by tests)

- Static current-day ticker list used as historical truth - impossible:
  engine only consumes instruments + dates + lagged bars.
- Missing delistings - engine filters by delisting_date.
- Identity/ticker mapping errors - SymbolHistory coverage enforced and
  resolved per as_of.

## 8. Next

Phase 3 (Raw Historical Data Ingestion) implements `DataAccessor` over an
immutable raw zone (provider connectors, checksums, idempotent ingestion,
corporate-action reconciliation).
