"""Ingestion registry: the durable catalog of what ORBIT downloaded.

A single DuckDB file (`registry.duckdb`) records every raw snapshot and
every normalized artifact with its checksum. Idempotency lives here: a
request fingerprint (provider + URI + fetch parameters) is UNIQUE, so
re-running the same download is recognized and skipped instead of
duplicating data.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb

_SCHEMA = """
-- Domain convention: the registry/raw-store domain is the SOURCE domain
-- (market | sec | macro); the normalized layer uses delivery-domain labels
-- (fundamentals/ for sec raw snapshots). Manifests carry the same source
-- domain so registry, raw zone, and manifest always agree.
CREATE TABLE IF NOT EXISTS snapshots (
    snapshot_id        VARCHAR PRIMARY KEY,
    domain             VARCHAR NOT NULL,
    provider           VARCHAR NOT NULL,
    source_uri         VARCHAR NOT NULL,
    request_fingerprint VARCHAR UNIQUE NOT NULL,
    checksum           VARCHAR NOT NULL,
    file_count         INTEGER NOT NULL,
    row_count          INTEGER,
    downloaded_at      TIMESTAMP NOT NULL,
    schema_version     VARCHAR NOT NULL,
    license_ref        VARCHAR,
    validation_status  VARCHAR,
    manifest_path      VARCHAR
);
CREATE TABLE IF NOT EXISTS normalized_artifacts (
    artifact_id    VARCHAR PRIMARY KEY,
    snapshot_id    VARCHAR NOT NULL,
    domain         VARCHAR NOT NULL,
    path           VARCHAR NOT NULL,
    checksum       VARCHAR NOT NULL,
    row_count      INTEGER,
    schema_version VARCHAR NOT NULL,
    created_at     TIMESTAMP NOT NULL,
    UNIQUE (snapshot_id, path)
);
CREATE TABLE IF NOT EXISTS counters (kind VARCHAR PRIMARY KEY, value INTEGER NOT NULL);
INSERT OR IGNORE INTO counters (kind, value) VALUES ('snapshot', 0), ('artifact', 0);
"""


def _next_id(con: duckdb.DuckDBPyConnection, kind: str, prefix: str) -> str:
    row = con.execute(
        "UPDATE counters SET value = value + 1 WHERE kind = ? RETURNING value", [kind]
    ).fetchone()
    return f"{prefix}{row[0]:06d}"


class IngestionRegistry:
    """Owns the DuckDB catalog. One instance per process; commits per write."""

    def __init__(self, db_path: str | Path):
        self._path = str(db_path)
        self._con = duckdb.connect(self._path)
        self._con.execute(_SCHEMA)

    def close(self) -> None:
        self._con.close()

    # ------------------------------------------------------------------ raw

    def has_fingerprint(self, fingerprint: str) -> bool:
        row = self._con.execute(
            "SELECT snapshot_id FROM snapshots WHERE request_fingerprint = ?",
            [fingerprint],
        ).fetchone()
        return row is not None

    def snapshot_for_fingerprint(self, fingerprint: str) -> dict[str, Any] | None:
        row = self._con.execute(
            "SELECT * FROM snapshots WHERE request_fingerprint = ?", [fingerprint]
        ).fetchone()
        if row is None:
            return None
        cols = [d[0] for d in self._con.description]
        return dict(zip(cols, row))

    def register_snapshot(self, record: dict[str, Any]) -> str:
        """Insert a raw snapshot record; returns its snapshot_id."""
        snapshot_id = _next_id(self._con, "snapshot", "DS-")
        self._con.execute(
            """INSERT INTO snapshots VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                snapshot_id,
                record["domain"],
                record["provider"],
                record["source_uri"],
                record["request_fingerprint"],
                record["checksum"],
                record["file_count"],
                record.get("row_count"),
                record["downloaded_at"],
                record["schema_version"],
                record.get("license_ref"),
                record.get("validation_status"),
                record.get("manifest_path"),
            ],
        )
        return snapshot_id

    def update_validation(self, snapshot_id: str, status: str, row_count: int) -> None:
        self._con.execute(
            "UPDATE snapshots SET validation_status = ?, row_count = ? WHERE snapshot_id = ?",
            [status, row_count, snapshot_id],
        )

    def update_manifest_path(self, snapshot_id: str, manifest_path: str) -> None:
        self._con.execute(
            "UPDATE snapshots SET manifest_path = ? WHERE snapshot_id = ?",
            [manifest_path, snapshot_id],
        )

    def snapshot(self, snapshot_id: str) -> dict[str, Any] | None:
        row = self._con.execute(
            "SELECT * FROM snapshots WHERE snapshot_id = ?", [snapshot_id]
        ).fetchone()
        if row is None:
            return None
        cols = [d[0] for d in self._con.description]
        return dict(zip(cols, row))

    # ----------------------------------------------------------- normalized

    def has_artifact(self, snapshot_id: str, path: str) -> bool:
        row = self._con.execute(
            "SELECT 1 FROM normalized_artifacts WHERE snapshot_id = ? AND path = ?",
            [snapshot_id, path],
        ).fetchone()
        return row is not None

    def register_artifact(
        self, snapshot_id: str, domain: str, path: str, checksum: str,
        row_count: int, schema_version: str,
    ) -> str:
        """Register (or refresh) one normalized artifact.

        Re-deriving a snapshot with changed code produces different bytes;
        the registry must record the NEW checksum, otherwise the artifact
        row silently lies about what is on disk.
        """
        existing = self._con.execute(
            "SELECT artifact_id, checksum FROM normalized_artifacts WHERE snapshot_id = ? AND path = ?",
            [snapshot_id, path],
        ).fetchone()
        if existing:
            if existing[1] != checksum:
                self._con.execute(
                    "UPDATE normalized_artifacts SET checksum = ?, row_count = ?, "
                    "schema_version = ?, created_at = ? WHERE artifact_id = ?",
                    [
                        checksum, row_count, schema_version,
                        datetime.now(timezone.utc).isoformat(), existing[0],
                    ],
                )
            return existing[0]
        artifact_id = _next_id(self._con, "artifact", "NS-")
        self._con.execute(
            """INSERT INTO normalized_artifacts VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                artifact_id, snapshot_id, domain, path, checksum, row_count,
                schema_version, datetime.now(timezone.utc).isoformat(),
            ],
        )
        return artifact_id

    def artifact_checksum(self, snapshot_id: str, path: str) -> str | None:
        row = self._con.execute(
            "SELECT checksum FROM normalized_artifacts WHERE snapshot_id = ? AND path = ?",
            [snapshot_id, path],
        ).fetchone()
        return row[0] if row else None

    def artifacts(self, snapshot_id: str) -> list[dict[str, Any]]:
        rows = self._con.execute(
            "SELECT artifact_id, path, checksum, row_count, schema_version, created_at "
            "FROM normalized_artifacts WHERE snapshot_id = ? ORDER BY artifact_id",
            [snapshot_id],
        ).fetchall()
        cols = ["artifact_id", "path", "checksum", "row_count", "schema_version", "created_at"]
        return [dict(zip(cols, r)) for r in rows]

    def dump(self) -> list[dict[str, Any]]:
        rows = self._con.execute(
            "SELECT snapshot_id, domain, provider, checksum, row_count, validation_status "
            "FROM snapshots ORDER BY snapshot_id"
        ).fetchall()
        cols = ["snapshot_id", "domain", "provider", "checksum", "row_count", "validation_status"]
        return [dict(zip(cols, r)) for r in rows]