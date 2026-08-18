"""Label version registry: immutability is the point.

A LabelContract becomes the binding definition of a prediction target only
once it is REGISTERED here. Registration rules (researcher safety):

  - a (label_id, version) pair can be registered exactly once; re-registering
    with a different definition is refused - a definition can never silently
    mutate after use;
  - a new version must be STRICTLY NEWER than every existing version of the
    same label_id (version numbering is monotonic, so a historical experiment
    always resolves the same number to the same definition);
  - a version whose definition is byte-identical to an existing version of
    the same label_id is refused (no silent version inflation - "v2" must
    actually differ from "v1");
  - registration metadata (registered_at, note) is recorded separately from
    the contract: `content_hash` covers exactly the semantic definition, so
    two registrations of the same definition are identical identities.

Historical experiments pin (label_id, version); `definition()` returns the
exact frozen contract they used, forever.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from orbit.labels.contract import LabelContract


class LabelVersionRecord(BaseModel):
    """One registered version: the frozen contract plus creation metadata."""

    model_config = ConfigDict(frozen=True)

    label_id: str = Field(pattern=r"^LAB-\d{3}$")
    version: str = Field(pattern=r"^v\d+(\.\d+)*$")
    contract: LabelContract
    registered_at: date
    note: str | None = None


def _version_key(version: str) -> tuple[int, ...]:
    return tuple(int(p) for p in version[1:].split("."))


class LabelVersionRegistry(BaseModel):
    """Versioned, immutable label definitions."""

    model_config = ConfigDict(frozen=True)

    records: list[LabelVersionRecord] = Field(default_factory=list)

    # ------------------------------------------------------------- register

    def register(
        self,
        contract: LabelContract,
        registered_at: date | None = None,
        note: str | None = None,
    ) -> LabelVersionRecord:
        """Register a contract, freezing it as the definition of
        (label_id, version). Refuses duplicates, out-of-order versions, and
        unchanged-definition re-registrations."""
        if not isinstance(contract, LabelContract):
            raise TypeError(
                "register() requires a LabelContract instance (a definition "
                "cannot be registered as raw dict/config)"
            )
        existing = self.records_for(contract.label_id)
        for rec in existing:
            if rec.version == contract.version:
                raise ValueError(
                    f"label {contract.label_id} version {contract.version} is "
                    "already registered; a definition can never be "
                    "overwritten - register a new version instead"
                )
            if rec.contract.definition_identity() == contract.definition_identity():
                raise ValueError(
                    f"label {contract.label_id}: the definition is identical "
                    f"to registered version {rec.version}; a new version must "
                    "change the definition (no silent version inflation)"
                )
        if existing:
            latest = max(rec.version for rec in existing)
            if _version_key(contract.version) <= _version_key(latest):
                raise ValueError(
                    f"label {contract.label_id}: version {contract.version} is "
                    f"not strictly newer than the registered {latest}; "
                    "versions must be monotonic"
                )
        record = LabelVersionRecord(
            label_id=contract.label_id,
            version=contract.version,
            contract=contract,
            registered_at=registered_at or date.today(),
            note=note,
        )
        self.records.append(record)
        return record

    # --------------------------------------------------------------- lookup

    def records_for(self, label_id: str) -> list[LabelVersionRecord]:
        return [r for r in self.records if r.label_id == label_id]

    def get(
        self, label_id: str, version: str | None = None
    ) -> LabelVersionRecord:
        """The record for (label_id, version); the latest version when
        version is None. Raises KeyError listing the registered versions."""
        candidates = self.records_for(label_id)
        if not candidates:
            raise KeyError(f"unknown label: {label_id} (nothing registered)")
        if version is None:
            return max(candidates, key=lambda r: _version_key(r.version))
        for rec in candidates:
            if rec.version == version:
                return rec
        raise KeyError(
            f"unknown version {version!r} of label {label_id}; registered "
            f"versions: {sorted(c.version for c in candidates)}"
        )

    def definition(
        self, label_id: str, version: str | None = None
    ) -> LabelContract:
        """The frozen contract a historical experiment used."""
        return self.get(label_id, version).contract

    def versions(self, label_id: str) -> list[str]:
        """Registered versions of a label, oldest first."""
        recs = self.records_for(label_id)
        return [r.version for r in sorted(recs, key=lambda r: _version_key(r.version))]

    def definition_digest(self, label_id: str, version: str) -> str:
        """sha256 identity of the registered definition (formula identity
        recorded in experiment metadata)."""
        return self.get(label_id, version).contract.content_hash()

    def definition_summary(self, label_id: str, version: str) -> dict[str, Any]:
        return self.get(label_id, version).contract.definition_summary()


__all__ = ["LabelVersionRecord", "LabelVersionRegistry"]