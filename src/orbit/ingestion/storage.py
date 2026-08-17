"""Raw immutable storage.

Raw means: the bytes exactly as the provider delivered them, before any
ORBIT transformation. Immutable means: ORBIT never overwrites or edits that
copy. Enforcement is three-fold:

1. A snapshot directory is created once and only written during download.
2. Files are only created if they do not already exist (write-once).
3. After sealing, an "immutable" marker file records the seal time and the
   expected checksums; a `verify` helper re-checksums every file and
   reports any drift.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from orbit.ingestion.checksums import sha256_file
from orbit.ingestion.paths import raw_dir, write_json


class RawStore:
    """Write-once store for provider-verbatim payloads."""

    def __init__(self, root: str | Path | None = None):
        self._root = Path(root) if root else None

    def _base(self, domain: str, provider: str, snapshot_id: str) -> Path:
        if self._root is not None:
            return self._root / "raw" / domain / provider / snapshot_id
        return raw_dir(domain, provider, snapshot_id)

    def snapshot_dir(
        self, domain: str, provider: str, snapshot_id: str
    ) -> Path:
        return self._base(domain, provider, snapshot_id)

    def write_once(
        self, domain: str, provider: str, snapshot_id: str,
        filename: str, body: bytes,
    ) -> tuple[Path, str]:
        """Write a payload unless it already exists; returns (path, sha256)."""
        target = self._base(domain, provider, snapshot_id) / filename
        if target.exists():
            return target, sha256_file(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_bytes(body)
        tmp.replace(target)
        return target, sha256_file(target)

    def seal(
        self, domain: str, provider: str, snapshot_id: str, expected: dict[str, str]
    ) -> Path:
        """Write the immutable marker; raises if files drift from `expected`."""
        base = self._base(domain, provider, snapshot_id)
        if not base.exists():
            raise RuntimeError(f"snapshot dir missing, cannot seal: {base}")
        actual = {
            p.name: sha256_file(p)
            for p in sorted(base.iterdir())
            if p.is_file() and p.name != "IMMUTABLE.json"
        }
        for name, checksum in expected.items():
            if actual.get(name) != checksum:
                raise RuntimeError(
                    f"raw seal mismatch for {name}: expected {checksum}, got {actual.get(name)}"
                )
        marker = base / "IMMUTABLE.json"
        write_json(
            marker,
            {
                "sealed_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
                "files": actual,
            },
        )
        return marker

    def verify(self, domain: str, provider: str, snapshot_id: str) -> bool:
        """Re-checksum every raw file against the sealed marker."""
        base = self._base(domain, provider, snapshot_id)
        marker = base / "IMMUTABLE.json"
        if not marker.exists():
            return False
        import json

        sealed = json.loads(marker.read_text())["files"]
        actual = {
            p.name: sha256_file(p)
            for p in sorted(base.iterdir())
            if p.is_file() and p.name != "IMMUTABLE.json"
        }
        return sealed == actual