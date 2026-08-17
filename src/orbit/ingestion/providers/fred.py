"""FRED / ALFRED connector (macroeconomic series).

AUTHORITATIVE source: fredgraph.csv
(https://fred.stlouisfed.org/graph/fredgraph.csv?id=DFF) is free, keyless,
and reliable. It delivers the *latest published vintage* of each series -
it is NOT a historical-vintage view.

ALFRED (the archival vintage database) is what Phase 4 needs to know what
was known at a historical date. Its programmatic CSV endpoint
(https://alfred.stlouisfed.org/series/downloaddata?series_id=X&vintage_date=D)
was unreachable from this machine on 2026-08-17 (timeout); full vintage
access may require the free FRED API key. The connector accepts an optional
`vintage_date`. ALFRED never degrades silently to today's revision: when a
vintage is requested but the endpoint fails, the fetch ABORTS with
ProviderUnavailable and no snapshot is recorded - the failed attempt is
loud, never silently substituted.

Missing observations appear as "." in FRED CSVs and are preserved as NULL,
never silently filled.
"""

from __future__ import annotations

import csv
import io
from typing import Any

from orbit.ingestion.downloaders.http import DownloadError, fetch_bytes
from orbit.ingestion.providers.base import ProviderUnavailable, RawObject

_FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
_ALFRED_CSV_URL = "https://alfred.stlouisfed.org/series/downloaddata?series_id={series_id}"


class FredConnector:
    provider_name = "fred_csv"

    def fetch(self, request: dict[str, Any]) -> list[RawObject]:
        series_id = request["series_id"]
        vintage = request.get("vintage_date")
        if vintage:
            url = _ALFRED_CSV_URL.format(series_id=series_id) + f"&vintage_date={vintage}"
            vintage_note = "alfred_vintage_requested"
        else:
            url = _FRED_CSV_URL.format(series_id=series_id)
            vintage_note = "latest_published_vintage"
        try:
            dl = fetch_bytes(url, timeout=60)
        except DownloadError as e:
            if vintage:
                raise ProviderUnavailable(
                    f"alfred vintage request for {series_id}@{vintage} failed: {e}. "
                    "No snapshot was recorded - latest-vintage data was NOT "
                    "silently substituted. The ALFRED endpoint may require the "
                    "free FRED API key."
                ) from e
            raise
        rows = list(csv.reader(io.StringIO(dl.body.decode("utf-8", errors="replace"))))
        if not rows or rows[0][0] != "observation_date":
            raise ProviderUnavailable(
                f"fred returned an unexpected payload for {series_id} (no snapshot recorded)"
            )
        return [
            RawObject(
                filename=f"{series_id}.csv",
                body=dl.body,
                source_uri=url,
                content_type=dl.content_type,
                meta={
                    "series_id": series_id,
                    "vintage_date": vintage,
                    "vintage_note": vintage_note,
                    "observations": len(rows) - 1,
                },
            )
        ]