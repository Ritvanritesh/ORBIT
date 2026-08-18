"""Label snapshots: the reproducible, auditable Phase 5 output.

A LabelSnapshot is one batch of computed labels: a fixed contract version,
one engine version, one source universe (data refs), and the record frame.
The content digest is deterministic over the sorted content (wall-clock is
excluded, like PointInTimeSnapshot), so the same inputs always yield the
same digest - a label batch can be pinned, replayed, and compared.

Structural separation (Phase 4 integration): a LabelSnapshot is a separate
artifact from any PointInTimeSnapshot. A point-in-time snapshot at the
decision instant never contains a label row for that decision (the outcome
is finalized only at the outcome instant); features are built from
point-in-time snapshots, labels from label snapshots - the two are never
merged into one unrestricted dataset.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any

import polars as pl

from orbit.labels.engine import LABEL_OUTPUT_COLUMNS, empty_label_frame


class LabelSnapshot:
    """One deterministic, reproducible batch of computed labels."""

    def __init__(
        self,
        *,
        label_id: str,
        version: str,
        contract_digest: str,
        engine_version: str,
        data_refs: list[str],
        records: pl.DataFrame,
        limitations: list[str] | None = None,
    ):
        self.label_id = label_id
        self.version = version
        self.contract_digest = contract_digest
        self.engine_version = engine_version
        self.data_refs = list(data_refs)
        self.records = records
        if list(records.columns) != [c for c, _ in LABEL_OUTPUT_COLUMNS]:
            raise ValueError(
                "label records must use the canonical label output schema "
                "(exact column order); got "
                f"{list(records.columns)}"
            )
        self.limitations = list(limitations or [])
        self.created_at = datetime.now()
        self.content_digest = self._compute_digest()

    def row_count(self) -> int:
        return self.records.height if self.records.height else 0

    def available_count(self) -> int:
        if self.records.height == 0:
            return 0
        return self.records.filter(pl.col("outcome_status") == "available").height

    def unavailable_count(self) -> int:
        if self.records.height == 0:
            return 0
        return self.records.filter(pl.col("outcome_status") == "unavailable").height

    def unavailable_reason_counts(self) -> dict[str, int]:
        if self.records.height == 0:
            return {}
        counts: dict[str, int] = {}
        for r in self.records.filter(pl.col("outcome_status") == "unavailable").group_by(
            "unavailable_reason"
        ).agg(pl.len().alias("n")).iter_rows(named=True):
            counts[r["unavailable_reason"]] = r["n"]
        return counts

    # ------------------------------------------------------- reproducibility

    def _canonical_json(self) -> str:
        parts: dict[str, Any] = {
            "label_id": self.label_id,
            "version": self.version,
            "contract_digest": self.contract_digest,
            "engine_version": self.engine_version,
            "data_refs": sorted(self.data_refs),
            "limitations": sorted(self.limitations),
        }
        if self.records.height:
            parts["records"] = json.loads(
                self.records.sort(self.records.columns).write_json()
            )
        return json.dumps(parts, sort_keys=True, default=str, separators=(",", ":"))

    def _compute_digest(self) -> str:
        return hashlib.sha256(self._canonical_json().encode("utf-8")).hexdigest()

    def equals(self, other: "LabelSnapshot") -> bool:
        """Same contract version, engine, sources and content (wall-clock
        excluded, like PointInTimeSnapshot.equals). data_refs are compared
        as sets, matching the digest's sorted treatment."""
        return (
            self.label_id == other.label_id
            and self.version == other.version
            and self.contract_digest == other.contract_digest
            and self.engine_version == other.engine_version
            and sorted(self.data_refs) == sorted(other.data_refs)
            and self.content_digest == other.content_digest
        )

    # ------------------------------------------------------------ provenance

    def provenance(self) -> dict[str, Any]:
        return {
            "label_id": self.label_id,
            "version": self.version,
            "contract_digest": self.contract_digest,
            "engine_version": self.engine_version,
            "content_digest": self.content_digest,
            "data_refs": list(self.data_refs),
            "limitations": list(self.limitations),
            "row_count": self.row_count(),
            "available_count": self.available_count(),
            "unavailable_count": self.unavailable_count(),
        }

    def to_json(self) -> dict[str, Any]:
        data = self.provenance()
        data["created_at"] = self.created_at.isoformat()
        if self.records.height:
            data["records"] = json.loads(
                self.records.sort(self.records.columns).write_json()
            )
        return data


def empty_label_snapshot(
    *,
    label_id: str,
    version: str,
    contract_digest: str,
    engine_version: str,
    data_refs: list[str] | None = None,
) -> LabelSnapshot:
    return LabelSnapshot(
        label_id=label_id,
        version=version,
        contract_digest=contract_digest,
        engine_version=engine_version,
        data_refs=data_refs or [],
        records=empty_label_frame(),
    )


__all__ = ["LabelSnapshot", "empty_label_snapshot"]