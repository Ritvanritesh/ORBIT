"""Filesystem layout for the ORBIT data zone.

```
<data_root>/
├── raw/            immutable, provider-verbatim payloads, one dir per snapshot
├── normalized/     derived, reproducible parquet (domain/source/snapshot/)
├── manifests/      one JSON manifest per snapshot
└── registry.duckdb ingestion catalog (idempotency + provenance)
```

The data root defaults to `<repo>/data` and can be overridden with the
ORBIT_DATA_ROOT environment variable (tests use a temp root).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[3]


def data_root() -> Path:
    root = os.environ.get("ORBIT_DATA_ROOT")
    if root:
        return Path(root)
    return _REPO_ROOT / "data"


def raw_dir(domain: str, provider: str, snapshot_id: str) -> Path:
    """Directory holding one immutable snapshot's raw payloads."""
    return data_root() / "raw" / domain / provider / snapshot_id


def normalized_dir(domain: str, provider: str, snapshot_id: str) -> Path:
    return data_root() / "normalized" / domain / provider / snapshot_id


def manifests_dir() -> Path:
    return data_root() / "manifests"


def registry_path() -> Path:
    return data_root() / "registry.duckdb"


def ensure_layout() -> None:
    for d in (
        raw_dir("market", "", "_").parent,
        normalized_dir("market", "", "_").parent,
        manifests_dir(),
    ):
        d.mkdir(parents=True, exist_ok=True)


def load_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=str)
        f.write("\n")