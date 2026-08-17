"""Checksums and content fingerprints.

A checksum is a digital fingerprint of a file: if the file changes by even
one byte, the fingerprint changes. ORBIT uses sha256 everywhere so that any
downloaded artifact can later be proven to be byte-for-byte what we stored.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_bytes(data: bytes) -> str:
    """Hex sha256 of a byte payload."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """Hex sha256 of a file on disk, streamed so memory stays flat."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def request_fingerprint(payload: dict[str, Any]) -> str:
    """Deterministic fingerprint of a *request* (not the response).

    Used for idempotency: the same provider, URI, and fetch parameters always
    map to the same fingerprint, so ORBIT can recognize a re-run of the same
    download before it hits the network again.
    """
    canonical = json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()