"""Phase 3: ingest daily market data for a symbol list.

Usage:
    python scripts/ingest_market.py --symbols AAPL MSFT --range 30y
    python scripts/ingest_market.py --provider yahoo_chart_api --symbols AAPL

Providers: yahoo_chart_api (working, DEVELOPMENT), stooq_csv (needs api_key,
RESEARCH CANDIDATE).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from orbit.ingestion.pipeline import IngestionPipeline
from orbit.ingestion.paths import ensure_layout, load_json, registry_path
from orbit.ingestion.providers.stooq import StooqConnector
from orbit.ingestion.providers.yahoo import YahooChartConnector
from orbit.ingestion.registry import IngestionRegistry
from orbit.ingestion.storage import RawStore

REPO = Path(__file__).resolve().parents[1]
DEFAULT_MASTER = REPO / "configs" / "instrument_master_dev.json"


def symbol_map_from_master(master_path: Path) -> dict[str, str]:
    master = load_json(master_path)
    return {
        inst["primary_ticker"]: inst["instrument_id"]
        for inst in master["instruments"]
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", default="yahoo_chart_api",
                        choices=["yahoo_chart_api", "stooq_csv"])
    parser.add_argument("--symbols", nargs="+", required=True)
    parser.add_argument("--range", default="30y")
    parser.add_argument("--api-key", default=None, help="Stooq API key (CAPTCHA-gated)")
    parser.add_argument("--master", default=str(DEFAULT_MASTER))
    args = parser.parse_args()

    ensure_layout()
    registry = IngestionRegistry(registry_path())
    raw_store = RawStore()
    pipeline = IngestionPipeline(registry, raw_store)

    if args.provider == "yahoo_chart_api":
        connector = YahooChartConnector()
    else:
        connector = StooqConnector()

    symbols = args.symbols
    symbol_map = symbol_map_from_master(Path(args.master))
    missing = [s for s in symbols if s not in symbol_map]
    if missing:
        print(f"error: symbols not in instrument master: {missing}")
        return 1

    result = pipeline.ingest_market(
        connector,
        symbols,
        symbol_map,
        license_ref="DEVELOPMENT - see configs/sources.json",
        request_params={"range": args.range, "api_key": args.api_key} if args.provider == "stooq_csv" else {"range": args.range},
    )
    print(result.summary())
    if result.validation.get("issues"):
        print(json.dumps(result.validation["issues"], indent=2)[:2000])
    if result.reconciliation:
        print("reconciliation:", json.dumps(result.reconciliation, indent=2)[:2000])
    registry.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())