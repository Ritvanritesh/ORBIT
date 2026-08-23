"""Phase 11.2 robust data acquisition: download expanded universe data.

Handles individual symbol download failures gracefully.
Downloads symbols one at a time to avoid batch failures.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import polars as pl
from orbit.ingestion.providers.yahoo import YahooChartConnector
from orbit.ingestion.parsing import parse_yahoo_chart
from orbit.ingestion.normalizers.market import normalize_market_bars
from orbit.ingestion.paths import normalized_dir, raw_dir
from orbit.ml.phase11_2_universe import (
    get_symbols_50,
    get_symbols_100,
    get_instrument_id_map_50,
    get_instrument_id_map_100,
    persist_universe_master,
    get_universe_50,
    get_universe_100,
    EXISTING_20,
)


def download_single_symbol(
    connector: YahooChartConnector,
    symbol: str,
    data_range: str = "30y",
) -> dict | None:
    """Download data for a single symbol. Returns parsed data or None on failure."""
    try:
        request = {"symbol": symbol, "range": data_range}
        raw_objects = connector.fetch(request)
        if raw_objects:
            raw = raw_objects[0]
            tables = parse_yahoo_chart(raw.body, symbol)
            bars_df = tables[0]
            events_df = tables[1]
            return {
                "symbol": symbol,
                "bars": bars_df,
                "events": events_df,
                "success": True,
                "rows": bars_df.height,
                "sessions": bars_df["ts"].n_unique() if "ts" in bars_df.columns else bars_df["date"].n_unique(),
            }
    except Exception as e:
        return {
            "symbol": symbol,
            "success": False,
            "error": str(e),
        }
    return None


def acquire_universe_individual(
    symbols: list[str],
    symbol_map: dict[str, str],
    universe_id: str,
    data_range: str = "30y",
) -> dict:
    """Acquire data for a universe, downloading one symbol at a time."""
    print(f"\n{'='*72}")
    print(f"ACQUIRING DATA FOR {universe_id}: {len(symbols)} symbols")
    print(f"{'='*72}")

    connector = YahooChartConnector()
    all_bars = []
    all_events = []
    failed_symbols = []
    succeeded = []

    for i, symbol in enumerate(symbols):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"\n  Progress: {i+1}/{len(symbols)} symbols...")

        result = download_single_symbol(connector, symbol, data_range)
        if result and result["success"]:
            all_bars.append(result["bars"])
            all_events.append(result["events"])
            succeeded.append(symbol)
        else:
            error = result["error"] if result else "unknown error"
            failed_symbols.append({"symbol": symbol, "error": error})
            print(f"  FAILED: {symbol}: {error[:80]}")

        # Rate limiting: 100ms between requests
        time.sleep(0.1)

    print(f"\n  Succeeded: {len(succeeded)}/{len(symbols)}")
    print(f"  Failed: {len(failed_symbols)}")
    if failed_symbols:
        print(f"  Failed symbols: {[f['symbol'] for f in failed_symbols]}")

    if not all_bars:
        print("  ERROR: No data downloaded")
        return {"success": False, "failed_symbols": failed_symbols}

    # Concatenate all bars
    combined_bars = pl.concat(all_bars)
    combined_events = pl.concat(all_events) if all_events else pl.DataFrame()

    # Normalize to ORBIT schema
    snapshot_id = f"DS-EXP-{universe_id.replace('UNIVERSE-', '')}"
    normalized_bars = normalize_market_bars(
        {s: {"bars": b, "events": e} for s, b, e in zip(succeeded, all_bars, all_events)},
        symbol_map,
        "yahoo_chart_api",
        f"yahoo_chart_api_{universe_id}",
        snapshot_id,
    )

    # Save to normalized directory
    out_dir = normalized_dir("market", "yahoo_chart_api", snapshot_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    bars_path = out_dir / "bars.parquet"
    events_path = out_dir / "events.parquet"

    normalized_bars["bars"].write_parquet(bars_path)
    if normalized_bars["events"].height > 0:
        normalized_bars["events"].write_parquet(events_path)

    print(f"\n  Saved to: {bars_path}")
    print(f"  Total rows: {normalized_bars['bars'].height}")
    print(f"  Instruments: {normalized_bars['bars']['instrument_id'].n_unique()}")
    print(f"  Sessions: {normalized_bars['bars']['trade_date'].n_unique()}")

    return {
        "success": True,
        "snapshot_id": snapshot_id,
        "bars_path": str(bars_path),
        "rows": normalized_bars["bars"].height,
        "instruments": normalized_bars["bars"]["instrument_id"].n_unique(),
        "sessions": normalized_bars["bars"]["trade_date"].n_unique(),
        "succeeded": succeeded,
        "failed_symbols": failed_symbols,
    }


def validate_snapshot(snapshot_id: str) -> dict:
    """Validate a downloaded snapshot."""
    print(f"\nValidating snapshot {snapshot_id}...")

    bars_path = normalized_dir("market", "yahoo_chart_api", snapshot_id) / "bars.parquet"
    if not bars_path.exists():
        print(f"  ERROR: bars.parquet not found")
        return {"valid": False}

    bars = pl.read_parquet(bars_path)
    checks = {}

    # Schema
    required_cols = {"trade_date", "instrument_id", "symbol", "open", "high", "low", "close", "volume"}
    checks["schema_valid"] = required_cols.issubset(set(bars.columns))

    # No nulls in critical columns
    for col in ["trade_date", "instrument_id", "close"]:
        checks[f"no_nulls_{col}"] = bars[col].null_count() == 0

    # No negative prices
    for col in ["open", "high", "low", "close"]:
        checks[f"no_negative_{col}"] = (bars[col] < 0).sum() == 0

    # OHLC consistency
    checks["ohlc_consistent"] = (bars["high"] < bars["low"]).sum() == 0

    # No duplicates
    checks["no_duplicates"] = bars.height == bars.unique().height

    # Benchmark not tradable
    checks["benchmark_not_tradable"] = "BENCH-001" not in bars["instrument_id"].to_list()

    # Instrument count
    n_instruments = bars["instrument_id"].n_unique()
    checks["instrument_count"] = n_instruments

    # Session count
    n_sessions = bars["trade_date"].n_unique()
    checks["session_count"] = n_sessions

    # Date range
    checks["date_min"] = str(bars["trade_date"].min())
    checks["date_max"] = str(bars["trade_date"].max())

    all_pass = all(v for k, v in checks.items() if isinstance(v, bool))
    checks["all_pass"] = all_pass

    print(f"  Instruments: {n_instruments}, Sessions: {n_sessions}")
    print(f"  Date range: {checks.get('date_min')} to {checks.get('date_max')}")
    print(f"  All checks: {'PASS' if all_pass else 'FAIL'}")

    return checks


def main():
    """Main entry point."""
    print("=" * 72)
    print("PHASE 11.2 - EXPANDED UNIVERSE DATA ACQUISITION")
    print("=" * 72)

    # Persist universe masters
    universe_50 = get_universe_50()
    universe_100 = get_universe_100()
    persist_universe_master(universe_50, "UNIVERSE-050")
    persist_universe_master(universe_100, "UNIVERSE-100")

    # Acquire 50-symbol data
    symbols_50 = get_symbols_50()
    map_50 = get_instrument_id_map_50()
    result_50 = acquire_universe_individual(symbols_50, map_50, "UNIVERSE-050")

    if result_50["success"]:
        validation_50 = validate_snapshot(result_50["snapshot_id"])
    else:
        validation_50 = {"all_pass": False}

    # Acquire 100-symbol data (individual downloads handle failures)
    symbols_100 = get_symbols_100()
    map_100 = get_instrument_id_map_100()
    result_100 = acquire_universe_individual(symbols_100, map_100, "UNIVERSE-100")

    if result_100["success"]:
        validation_100 = validate_snapshot(result_100["snapshot_id"])
    else:
        validation_100 = {"all_pass": False}

    # Summary
    print("\n" + "=" * 72)
    print("ACQUISITION COMPLETE")
    print("=" * 72)

    for label, res, val in [("50-symbol", result_50, validation_50), ("100-symbol", result_100, validation_100)]:
        print(f"\n{label}:")
        if res["success"]:
            print(f"  Snapshot: {res['snapshot_id']}")
            print(f"  Rows: {res['rows']}")
            print(f"  Instruments: {res['instruments']}")
            print(f"  Sessions: {res['sessions']}")
            print(f"  Failed symbols: {len(res['failed_symbols'])}")
            print(f"  Validation: {'PASS' if val.get('all_pass') else 'FAIL'}")
        else:
            print(f"  FAILED: {len(res.get('failed_symbols', []))} symbols failed")

    return {
        "universe_50": result_50,
        "universe_100": result_100,
        "validation_50": validation_50,
        "validation_100": validation_100,
    }


if __name__ == "__main__":
    result = main()
    success = (
        result["universe_50"].get("success", False)
        and result["universe_100"].get("success", False)
    )
    sys.exit(0 if success else 1)
