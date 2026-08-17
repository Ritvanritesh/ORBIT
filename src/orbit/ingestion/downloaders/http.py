"""Minimal HTTP downloader.

Uses only the standard library. Downloads are logged with their source URI
so a manifest can always answer "where did this byte come from".
Transient failures (timeouts, connection errors, HTTP 429/5xx) are retried
with backoff; definitive 4xx errors fail immediately.
"""

from __future__ import annotations

import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Mapping


class DownloadError(RuntimeError):
    """Raised when a download ultimately fails after retries."""


# 429 = rate limited (SEC asks for <= 10 requests/second); 5xx = provider
# hiccup. Both are worth retrying; other 4xx are definitive.
_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


@dataclass
class Downloaded:
    body: bytes
    url: str
    content_type: str
    headers: Mapping[str, str] = field(default_factory=dict)


def fetch_bytes(
    url: str,
    *,
    headers: Mapping[str, str] | None = None,
    timeout: float = 30.0,
    retries: int = 3,
    backoff_seconds: float = 2.0,
) -> Downloaded:
    """Fetch a URL with retries. Any non-2xx status raises DownloadError."""
    req_headers = {"User-Agent": "ORBIT-Research/0.1 (research; contact: orbit-research@example.com)"}
    if headers:
        req_headers.update(headers)
    last_exc: Exception | None = None
    last_status: int | None = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers=req_headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                status = resp.status
                if status in _RETRYABLE_STATUS and attempt < retries:
                    last_status = status
                    time.sleep(backoff_seconds * attempt)
                    continue
                if not (200 <= status < 300):
                    raise DownloadError(f"HTTP {status} for {url}")
                return Downloaded(
                    body=resp.read(),
                    url=resp.geturl(),
                    content_type=resp.headers.get("Content-Type", ""),
                    headers=dict(resp.headers),
                )
        except DownloadError:
            raise
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last_exc = e
            if attempt < retries:
                time.sleep(backoff_seconds * attempt)
    raise DownloadError(
        f"download failed after {retries} attempts: {url}: {last_exc or f'HTTP {last_status}'}"
    )