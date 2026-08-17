"""Phase 3: ingest FRED macro series (latest published vintage).

Usage:
    python scripts/ingest_fred.py --series DFF CPIAUCSL UNRATE

ALFRED historical vintages are NOT silently substituted; request them
explicitly with --vintage-date (may fail until the ALFRED endpoint is
reachable or a free FRED API key is configured).
"""

from __future__ import annotations

import argparse
import json
import sys

from orbit.ingestion.pipeline import IngestionPipeline
from orbit.ingestion.paths import ensure_layout, registry_path
from orbit.ingestion.providers.fred import FredConnector
from orbit.ingestion.registry import IngestionRegistry
from orbit.ingestion.storage import RawStore


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--series", nargs="+", required=True)
    parser.add_argument("--vintage-date", default=None,
                        help="ALFRED vintage date YYYY-MM-DD (experimental)")
    args = parser.parse_args()

    ensure_layout()
    registry = IngestionRegistry(registry_path())
    pipeline = IngestionPipeline(registry, RawStore())
    result = pipeline.ingest_macro(
        FredConnector(),
        args.series,
        license_ref="AUTHORITATIVE - FRED public data",
        request_params={"vintage_date": args.vintage_date} if args.vintage_date else {},
    )
    print(result.summary())
    if result.validation.get("issues"):
        print(json.dumps(result.validation["issues"], indent=2)[:2000])
    registry.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())