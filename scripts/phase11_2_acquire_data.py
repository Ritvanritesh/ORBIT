"""Phase 11.2 data acquisition: download expanded universe data.

Downloads historical market data for the ~50 and ~100 symbol universes
using the existing Yahoo Chart API pipeline.

This script:
1. Defines the expanded universes
2. Downloads data via the existing ingestion pipeline
3. Creates new dataset snapshots
4. Validates data quality
5. Persists manifests
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orbit.ingestion.pipeline import IngestionPipeline
from orbit.ingestion.paths import ensure_layout, registry_path
from orbit.ingestion.providers.yahoo import YahooChartConnector
from orbit.ingestion.registry import IngestionRegistry
from orbit.ingestion.storage import RawStore
from orbit.ml.phase11_2_universe import (
    get_symbols_50,
    get_symbols_100,
    get_instrument_id_map_50,
    get_instrument_id_map_100,
    persist_universe_master,
    get_universe_50,
    get_universe_100,
)


def acquire_universe(
    symbols: list[str],
    symbol_map: dict[str, str],
    universe_id: str,
    data_range: str = "30y",
) -> dict:
    """Acquire data for a universe using the existing pipeline."""
    print(f"\n{'='*72}")
    print(f"ACQUIRING DATA FOR {universe_id}: {len(symbols)} symbols")
    print(f"{'='*72}")

    ensure_layout()
    registry = IngestionRegistry(registry_path())
    raw_store = RawStore()
    pipeline = IngestionPipeline(registry, raw_store)
    connector = YahooChartConnector()

    print(f"\nDownloading {len(symbols)} symbols via Yahoo Chart API...")
    print(f"Symbols: {', '.join(symbols[:10])}{'...' if len(symbols) > 10 else ''}")

    result = pipeline.ingest_market(
        connector,
        symbols,
        symbol_map,
        license_ref="DEVELOPMENT - see configs/sources.json",
        request_params={"range": data_range},
    )

    print(f"\nResult: {result.summary()}")

    if result.validation.get("issues"):
        n_issues = len(result.validation["issues"])
        print(f"Validation issues: {n_issues}")
        # Show first few issues
        for issue in result.validation["issues"][:5]:
            print(f"  - {issue.get('level', '?')}: {issue.get('check', '?')}: {issue.get('message', '?')[:100]}")

    if result.reconciliation:
        recon = result.reconciliation
        n_findings = len(recon.get("findings", [])) if isinstance(recon, dict) else 0
        print(f"Reconciliation findings: {n_findings}")

    registry.close()

    return {
        "snapshot_id": result.snapshot_id,
        "reused": result.reused,
        "row_count": result.row_count,
        "validation_status": result.validation.get("status", "unknown"),
        "n_issues": len(result.validation.get("issues", [])),
    }


def validate_snapshot(snapshot_id: str) -> dict:
    """Validate a downloaded snapshot."""
    import polars as pl
    from orbit.ingestion.paths import normalized_dir

    print(f"\nValidating snapshot {snapshot_id}...")

    bars_path = normalized_dir("market", "yahoo_chart_api", snapshot_id) / "bars.parquet"
    if not bars_path.exists():
        print(f"  ERROR: bars.parquet not found at {bars_path}")
        return {"valid": False, "reason": "bars.parquet not found"}

    bars = pl.read_parquet(bars_path)

    checks = {}

    # 1. Schema check
    required_cols = {"trade_date", "instrument_id", "symbol", "open", "high", "low", "close", "volume"}
    missing_cols = required_cols - set(bars.columns)
    checks["schema_valid"] = len(missing_cols) == 0
    if missing_cols:
        print(f"  FAIL: Missing columns: {missing_cols}")

    # 2. No nulls in critical columns
    for col in ["trade_date", "instrument_id", "close"]:
        null_count = bars[col].null_count()
        checks[f"no_nulls_{col}"] = null_count == 0
        if null_count > 0:
            print(f"  FAIL: {null_count} nulls in {col}")

    # 3. No negative prices
    price_cols = ["open", "high", "low", "close"]
    for col in price_cols:
        neg_count = (bars[col] < 0).sum()
        checks[f"no_negative_{col}"] = neg_count == 0
        if neg_count > 0:
            print(f"  FAIL: {neg_count} negative values in {col}")

    # 4. OHLC consistency (high >= low)
    if "high" in bars.columns and "low" in bars.columns:
        inconsistent = (bars["high"] < bars["low"]).sum()
        checks["ohlc_consistent"] = inconsistent == 0
        if inconsistent > 0:
            print(f"  FAIL: {inconsistent} rows with high < low")

    # 5. Duplicate rows
    n_before = bars.height
    n_after = bars.unique().height
    checks["no_duplicates"] = n_before == n_after
    if n_before != n_after:
        print(f"  FAIL: {n_before - n_after} duplicate rows")

    # 6. Chronological ordering per instrument
    is_sorted = True
    for inst_id in bars["instrument_id"].unique().to_list():
        inst_bars = bars.filter(bars["instrument_id"] == inst_id).sort("trade_date")
        dates = inst_bars["trade_date"].to_list()
        if dates != sorted(dates):
            is_sorted = False
            break
    checks["chronological_order"] = is_sorted

    # 7. Instrument count
    n_instruments = bars["instrument_id"].n_unique()
    checks["instrument_count"] = n_instruments

    # 8. Session count
    n_sessions = bars["trade_date"].n_unique()
    checks["session_count"] = n_sessions

    # 9. Date range
    checks["date_min"] = str(bars["trade_date"].min())
    checks["date_max"] = str(bars["trade_date"].max())

    # 10. No BENCH-001 in instrument_id (benchmark not tradable)
    has_benchmark = "BENCH-001" in bars["instrument_id"].to_list()
    checks["benchmark_not_tradable"] = not has_benchmark
    if has_benchmark:
        print(f"  FAIL: BENCH-001 found in instrument_id")

    all_pass = all(v for k, v in checks.items() if isinstance(v, bool))
    checks["all_pass"] = all_pass

    print(f"\n  Instruments: {n_instruments}")
    print(f"  Sessions: {n_sessions}")
    print(f"  Date range: {checks.get('date_min', '?')} to {checks.get('date_max', '?')}")
    print(f"  All checks: {'PASS' if all_pass else 'FAIL'}")

    return checks


def main():
    """Main entry point for data acquisition."""
    print("=" * 72)
    print("PHASE 11.2 - EXPANDED UNIVERSE DATA ACQUISITION")
    print("=" * 72)

    # Step 1: Persist universe masters
    print("\n[1/4] Persisting universe masters...")
    universe_50 = get_universe_50()
    universe_100 = get_universe_100()
    persist_universe_master(universe_50, "UNIVERSE-050")
    persist_universe_master(universe_100, "UNIVERSE-100")
    print(f"  50-universe: {len(universe_50)} instruments")
    print(f"  100-universe: {len(universe_100)} instruments")

    # Step 2: Acquire 50-symbol data
    print("\n[2/4] Acquiring 50-symbol universe data...")
    symbols_50 = get_symbols_50()
    map_50 = get_instrument_id_map_50()
    result_50 = acquire_universe(symbols_50, map_50, "UNIVERSE-050")

    # Step 3: Validate 50-symbol snapshot
    print("\n[3/4] Validating 50-symbol snapshot...")
    validation_50 = validate_snapshot(result_50["snapshot_id"])

    # Step 4: Acquire 100-symbol data
    print("\n[4/4] Acquiring 100-symbol universe data...")
    symbols_100 = get_symbols_100()
    map_100 = get_instrument_id_map_100()
    result_100 = acquire_universe(symbols_100, map_100, "UNIVERSE-100")

    # Validate 100-symbol snapshot
    validation_100 = validate_snapshot(result_100["snapshot_id"])

    # Summary
    print("\n" + "=" * 72)
    print("ACQUISITION COMPLETE")
    print("=" * 72)
    print(f"\n50-symbol snapshot: {result_50['snapshot_id']}")
    print(f"  Rows: {result_50['row_count']}")
    print(f"  Validation: {result_50['validation_status']}")
    print(f"  All checks: {'PASS' if validation_50.get('all_pass') else 'FAIL'}")

    print(f"\n100-symbol snapshot: {result_100['snapshot_id']}")
    print(f"  Rows: {result_100['row_count']}")
    print(f"  Validation: {result_100['validation_status']}")
    print(f"  All checks: {'PASS' if validation_100.get('all_pass') else 'FAIL'}")

    return {
        "universe_50": {"snapshot_id": result_50["snapshot_id"], "validation": validation_50},
        "universe_100": {"snapshot_id": result_100["snapshot_id"], "validation": validation_100},
    }


if __name__ == "__main__":
    result = main()
    sys.exit(0 if result["universe_50"]["validation"].get("all_pass") and result["universe_100"]["validation"].get("all_pass") else 1)
