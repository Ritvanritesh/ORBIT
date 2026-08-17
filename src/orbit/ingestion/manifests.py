"""Ingestion manifests.

A manifest is the answer to the question "exactly what data did ORBIT
download?" - one JSON file per snapshot recording provider, source URI,
per-file checksums, date range, instrument coverage, schema/data versions,
license reference, row counts, and validation outcome. Manifests are the
human-readable half of the registry.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from orbit.ingestion.paths import manifests_dir, write_json


class Manifest(BaseModel):
    """Provenance record for one raw ingestion snapshot."""

    model_config = ConfigDict(frozen=True)

    snapshot_id: str = Field(pattern=r"^DS-\d{6}$")
    domain: str
    provider: str
    source: str
    source_uri: str
    request_fingerprint: str
    checksums: dict[str, str] = Field(
        description="filename -> sha256 hex of every raw file"
    )
    file_paths: dict[str, str]
    downloaded_at: datetime
    date_range: list[str | None] = Field(
        min_length=2, max_length=2, description="[earliest, latest] observation date"
    )
    instruments: list[str] = Field(default_factory=list)
    schema_version: str = Field(pattern=r"^v\d+(\.\d+)*$")
    data_version: str | None = None
    license_ref: str | None = None
    row_count: int | None = None
    validation_status: str | None = None
    validation_issues: list[dict[str, Any]] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _range_ok(self) -> "Manifest":
        if all(self.date_range):
            if self.date_range[1] < self.date_range[0]:
                raise ValueError("date_range end precedes start")
        return self

    def to_json(self) -> dict[str, Any]:
        data = self.model_dump(mode="json")
        data["manifest_created_at"] = datetime.now(timezone.utc).isoformat()
        return data

    def write(self) -> str:
        path = manifests_dir() / f"{self.snapshot_id}.json"
        manifest_dir = manifests_dir()
        manifest_dir.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        import json

        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.to_json(), f, indent=2, sort_keys=True, default=str)
            f.write("\n")
        tmp.replace(path)
        return str(path)


def build_manifest(
    snapshot_id: str,
    domain: str,
    provider: str,
    source: str,
    source_uri: str,
    request_fingerprint: str,
    checksums: dict[str, str],
    file_paths: dict[str, str],
    earliest: date | None,
    latest: date | None,
    instruments: list[str],
    schema_version: str,
    data_version: str | None,
    license_ref: str | None,
    row_count: int | None,
    validation_status: str | None,
    validation_issues: list[dict[str, Any]],
    meta: dict[str, Any],
    downloaded_at: datetime | None = None,
) -> Manifest:
    return Manifest(
        snapshot_id=snapshot_id,
        domain=domain,
        provider=provider,
        source=source,
        source_uri=source_uri,
        request_fingerprint=request_fingerprint,
        checksums=checksums,
        file_paths=file_paths,
        downloaded_at=downloaded_at or datetime.now(timezone.utc),
        date_range=[earliest.isoformat() if earliest else None, latest.isoformat() if latest else None],
        instruments=sorted(set(instruments)),
        schema_version=schema_version,
        data_version=data_version,
        license_ref=license_ref,
        row_count=row_count,
        validation_status=validation_status,
        validation_issues=validation_issues,
        meta=meta,
    )