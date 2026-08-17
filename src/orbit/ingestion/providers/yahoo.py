"""Yahoo Finance chart API connector.

DEVELOPMENT source: free, keyless, but unofficial and not licensed for
redistribution - its data may contain errors and its terms may change.
ORBIT uses it to prove the pipeline end-to-end; it is NOT authoritative.

The v8 chart API returns one JSON payload per symbol containing OHLCV,
`adjclose`, and the raw dividend/split event lists. Prices in this endpoint
are split-adjusted by Yahoo; the raw JSON is preserved verbatim and the
adjustment status is recorded in the manifest so no one mistakes it for
unadjusted exchange data.
"""

from __future__ import annotations

import json
from typing import Any

from orbit.ingestion.downloaders.http import fetch_bytes
from orbit.ingestion.providers.base import RawObject

_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"


class YahooChartConnector:
    provider_name = "yahoo_chart_api"

    def __init__(self, user_agent: str | None = None):
        self._ua = user_agent

    def fetch(self, request: dict[str, Any]) -> list[RawObject]:
        symbol = request["symbol"]
        params: list[str] = []
        if request.get("period1") and request.get("period2"):
            params.extend([f"period1={request['period1']}", f"period2={request['period2']}"])
        else:
            params.append(f"range={request.get('range', '30y')}")
        params.append("interval=1d")
        params.append("events=div%2Csplit")
        url = _CHART_URL.format(symbol=symbol) + "?" + "&".join(params)

        headers = {}
        if self._ua:
            headers["User-Agent"] = self._ua
        dl = fetch_bytes(url, headers=headers, timeout=45)
        payload = json.loads(dl.body)
        result = payload.get("chart", {}).get("result")
        if not result:
            error = payload.get("chart", {}).get("error", {})
            raise RuntimeError(f"yahoo returned no data for {symbol}: {error}")

        return [
            RawObject(
                filename=f"{symbol}.json",
                body=dl.body,
                source_uri=url,
                content_type=dl.content_type,
                meta={
                    "symbol": symbol,
                    "bars": len(result[0].get("timestamp", [])),
                    "request": {k: v for k, v in request.items() if k != "symbol"},
                },
            )
        ]