"""Phase 3: ingest SEC EDGAR / XBRL company facts for CIK list.

Usage:
    python scripts/ingest_sec.py --symbols AAPL MSFT

The SEC requires a declared User-Agent with a contact address; replace the
default before heavier use.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from orbit.ingestion.pipeline import IngestionPipeline
from orbit.ingestion.paths import ensure_layout, load_json, registry_path
from orbit.ingestion.providers.sec import SecEdgarConnector
from orbit.ingestion.registry import IngestionRegistry
from orbit.ingestion.storage import RawStore

REPO = Path(__file__).resolve().parents[1]
DEFAULT_MASTER = REPO / "configs" / "instrument_master_dev.json"
DEFAULT_UA = "ORBIT-Research-Project research@example.com"

SEC_UA_KEY = "ORBIT_SEC_USER_AGENT"


def cik_map(master_path: Path) -> dict[str, int]:
    master = load_json(master_path)
    return {inst["primary_ticker"]: int(inst["cik"]) for inst in master["instruments"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", nargs="+", required=True)
    parser.add_argument("--user-agent", default=None)
    parser.add_argument("--master", default=str(DEFAULT_MASTER))
    args = parser.parse_args()

    import os

    ua = args.user_agent or os.environ.get(SEC_UA_KEY, DEFAULT_UA)

    ensure_layout()
    registry = IngestionRegistry(registry_path())
    pipeline = IngestionPipeline(registry, RawStore())
    ciks = cik_map(Path(args.master))
    missing = [s for s in args.symbols if s not in ciks]
    if missing:
        print(f"error: symbols not in instrument master: {missing}")
        return 1

    connector = SecEdgarConnector(ua)
    result = pipeline.ingest_sec(
        connector,
        [ciks[s] for s in args.symbols],
        license_ref="AUTHORITATIVE - SEC EDGAR public filings",
    )
    print(result.summary())
    if result.validation.get("issues"):
        print(json.dumps(result.validation["issues"], indent=2)[:2000])
    registry.close()
    return 0 if result.validation.get("status") == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())