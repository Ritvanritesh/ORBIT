"""Stooq CSV connector.

RESEARCH CANDIDATE source: bulk daily CSV per symbol (e.g.
https://stooq.com/q/d/l/?s=aapl.us&i=d). Verified 2026-08-17: Stooq now
gates programmatic downloads behind an API key obtained via an on-site
CAPTCHA, so automated keyless access is currently unavailable (the site
answers with a JavaScript verification page). The connector therefore
requires `api_key` in the request; without it, ProviderUnavailable is
raised. Stooq's Close column is split-adjusted (no separate unadjusted
series), which the manifest records.

Terms of use must be re-verified before promoting this to an
AUTHORITATIVE source.
"""

from __future__ import annotations

from typing import Any

from orbit.ingestion.downloaders.http import fetch_bytes
from orbit.ingestion.providers.base import ProviderUnavailable, RawObject

_STOOQ_URL = "https://stooq.com/q/d/l/"


class StooqConnector:
    provider_name = "stooq_csv"

    def fetch(self, request: dict[str, Any]) -> list[RawObject]:
        symbol = request["symbol"]
        api_key = request.get("api_key")
        if not api_key:
            raise ProviderUnavailable(
                "stooq_csv requires an api_key (obtained via the on-site CAPTCHA at "
                "https://stooq.com/q/d/?s=spy.us&get_apikey) since early 2026"
            )
        params = f"?s={symbol}&i=d&k={api_key}"
        url = _STOOQ_URL + params
        dl = fetch_bytes(url, timeout=45)
        text = dl.body.decode("utf-8", errors="replace")
        if text.startswith("<!DOCTYPE") or "Access denied" in text:
            raise ProviderUnavailable(
                f"stooq returned a challenge/denial page for {symbol} (key rejected?)"
            )
        return [
            RawObject(
                filename=f"{symbol}.csv",
                body=dl.body,
                source_uri=url,
                content_type=dl.content_type,
                meta={
                    "symbol": symbol,
                    "adjusted": True,
                    "adjustment_note": "Stooq Close is split-adjusted; no unadjusted series",
                },
            )
        ]