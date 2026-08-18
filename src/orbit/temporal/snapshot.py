"""Point-in-time snapshots: the auditable answer to "what was known at T".

A PointInTimeSnapshot is fully reproducible: the same (as_of, sources,
engine version) always yields the same records, exclusions, and content
digest. Provenance links every record back to its dataset snapshot,
checksum, manifest, and raw source.

Provenance chain (prompt section 18):

    PointInTimeSnapshot
        -> TemporalSource (dataset snapshot + checksum + manifest)
        -> DatasetSnapshot (registry record)
        -> raw source / manifest -> source record -> publication info
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any

import polars as pl
from pydantic import BaseModel, ConfigDict

from orbit.temporal.times import normalize_instant


class TemporalSource(BaseModel):
    """One dataset snapshot feeding a point-in-time snapshot.

    `artifact_paths` lets tests point the engine at arbitrary parquet
    files; production use resolves them from the registry/manifests.
    """

    model_config = ConfigDict(frozen=True)

    snapshot_id: str
    domain: str
    provider: str
    checksum: str
    manifest_path: str | None = None
    ingest_time: datetime | None = None
    artifact_paths: list[str] = []

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "domain": self.domain,
            "provider": self.provider,
            "checksum": self.checksum,
            "manifest_path": self.manifest_path,
            "ingest_time": self.ingest_time.isoformat() if self.ingest_time else None,
            "artifact_paths": list(self.artifact_paths),
        }


class PointInTimeSnapshot:
    """The reproducible information set ORBIT was allowed to know at as_of."""

    def __init__(
        self,
        *,
        as_of_time: datetime,
        engine_version: str,
        sources: list[TemporalSource],
        records: pl.DataFrame,
        excluded: pl.DataFrame,
        limitations: list[str],
    ):
        self.as_of_time = normalize_instant(as_of_time)
        if self.as_of_time is None:
            raise ValueError("as_of_time is required")
        self.engine_version = engine_version
        self.sources = list(sources)
        self.records = records
        self.excluded = excluded
        self.limitations = list(limitations)
        self.created_at = datetime.now()
        self.content_digest = self._compute_digest()

    # ------------------------------------------------------------- queries

    def allowed_record_ids(self) -> list[str]:
        return sorted(self.records["record_id"].to_list())

    def excluded_record_ids(self) -> list[str]:
        return sorted(self.excluded["record_id"].to_list())

    def decision_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        if self.records.height:
            for r in self.records.group_by("decision_code").agg(pl.len().alias("n")).iter_rows(named=True):
                counts[r["decision_code"]] = counts.get(r["decision_code"], 0) + r["n"]
        if self.excluded.height:
            for r in self.excluded.group_by("decision_code").agg(pl.len().alias("n")).iter_rows(named=True):
                counts[r["decision_code"]] = counts.get(r["decision_code"], 0) + r["n"]
        return counts

    def record(self, record_id: str) -> dict[str, Any] | None:
        for df in (self.records, self.excluded):
            hit = df.filter(pl.col("record_id") == record_id)
            if hit.height:
                row = hit.row(0, named=True)
                return {k: (v.isoformat() if isinstance(v, datetime) else v) for k, v in row.items()}
        return None

    # ------------------------------------------------------- reproducibility

    def _canonical_records_json(self) -> str:
        """Deterministic JSON of the full content (records + exclusions)."""
        parts: dict[str, Any] = {
            "as_of_time": self.as_of_time.isoformat(),
            "engine_version": self.engine_version,
            "sources": sorted(
                (s.snapshot_id, s.domain, s.provider, s.checksum, s.manifest_path)
                for s in self.sources
            ),
        }
        if self.records.height:
            parts["records"] = json.loads(
                self.records.sort(self.records.columns).write_json()
            )
        if self.excluded.height:
            parts["excluded"] = json.loads(
                self.excluded.sort(self.excluded.columns).write_json()
            )
        parts["limitations"] = sorted(self.limitations)
        return json.dumps(parts, sort_keys=True, default=str, separators=(",", ":"))

    def _compute_digest(self) -> str:
        canonical = self._canonical_records_json().encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    def equals(self, other: "PointInTimeSnapshot") -> bool:
        """Same information set at the same as_of with the same provenance.

        `created_at` is wall-clock and intentionally excluded: two runs of
        the same query must be identical even though they happen at
        different moments.
        """
        return (
            self.as_of_time == other.as_of_time
            and self.engine_version == other.engine_version
            and self.content_digest == other.content_digest
        )

    # ------------------------------------------------------------ provenance

    def provenance(self) -> dict[str, Any]:
        """The traceable chain: snapshot -> source -> checksum -> manifest."""
        return {
            "as_of_time": self.as_of_time.isoformat(),
            "engine_version": self.engine_version,
            "content_digest": self.content_digest,
            "limitations": list(self.limitations),
            "sources": [s.to_dict() for s in self.sources],
            "record_count": self.records.height if self.records.height else 0,
            "excluded_count": self.excluded.height if self.excluded.height else 0,
        }

    def to_json(self) -> dict[str, Any]:
        data = self.provenance()
        data["created_at"] = self.created_at.isoformat()
        if self.records.height:
            data["records"] = json.loads(
                self.records.sort(self.records.columns).write_json()
            )
        if self.excluded.height:
            data["excluded"] = json.loads(
                self.excluded.sort(self.excluded.columns).write_json()
            )
        return data