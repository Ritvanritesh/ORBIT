"""SEC EDGAR / XBRL connector (company facts).

AUTHORITATIVE source (official US government filings): one JSON payload per
company at https://data.sec.gov/api/xbrl/companyfacts/CIK########.json.

SEC requires a declared User-Agent with a contact address; ORBIT's default
is set in configs/sources.json and must be replaced with a real contact
before heavier use (SEC asks for <= 10 requests/second).

Phase 3 scope is acquisition + preservation: the raw JSON is stored verbatim
(checksummed), with filing-level metadata (accn, filed, fy, fp) preserved
inside it for Phase 4's point-in-time work. No fundamental feature engine.
"""

from __future__ import annotations

from typing import Any

from orbit.ingestion.downloaders.http import fetch_bytes
from orbit.ingestion.providers.base import RawObject

_COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"


class SecEdgarConnector:
    provider_name = "sec_edgar_companyfacts"

    def __init__(self, user_agent: str):
        self._ua = user_agent

    def fetch(self, request: dict[str, Any]) -> list[RawObject]:
        cik = int(request["cik"])
        url = _COMPANYFACTS_URL.format(cik=cik)
        dl = fetch_bytes(url, headers={"User-Agent": self._ua}, timeout=45)
        import json

        payload = json.loads(dl.body)
        facts = payload.get("facts", {})
        if not facts:
            raise RuntimeError(f"sec returned no facts for CIK {cik}")
        return [
            RawObject(
                filename=f"cik{cik:010d}_companyfacts.json",
                body=dl.body,
                source_uri=url,
                content_type=dl.content_type,
                meta={
                    "cik": cik,
                    "entity_name": payload.get("entityName"),
                    "taxonomies": sorted(facts.keys()),
                },
            )
        ]