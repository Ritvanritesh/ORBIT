"""The backtest run manifest (Phase 7).

A backtest result without a manifest is not a trustworthy research result.
The manifest identifies everything needed to reproduce the run:

  - dataset snapshot ids (Phase 3)
  - feature version / refs and model identity (Phase 6 lineage)
  - the cost model (id + executable configuration)
  - the execution configuration (delay, price, liquidity, expiry)
  - the universe and the evaluation window
  - seed / randomness policy
  - the Phase 4 temporal configuration digest and Phase 5 label ref
  - the experiment id (Phase 6), the executing code hash and the
    configuration hash

`content_hash` covers every identity field; `created_at` is wall-clock
and excluded, so two identical runs have identical manifests and the
replay comparison can be exact.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from orbit.backtest.config import BacktestConfig

MANIFEST_VERSION = "v1.0.0"

# run_id is derived from the content hash (BT-<config8>-<content12>) and
# created_at is wall-clock: neither is part of the identity, so two
# identical runs produce byte-identical manifests.
_OPERATIONAL_FIELDS = frozenset({"created_at", "run_id"})


class BacktestManifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    manifest_version: str = Field(default=MANIFEST_VERSION, pattern=r"^v\d+(\.\d+)*$")
    run_id: str
    engine_version: str = Field(pattern=r"^v\d+(\.\d+)*$")
    created_at: datetime | None = None

    experiment_id: str | None = None
    hypothesis_id: str | None = None
    dataset_snapshot_ids: list[str] = Field(min_length=1)
    feature_refs: list[dict[str, Any]] = Field(default_factory=list)
    model: dict[str, Any] | None = None
    label_id: str | None = None
    label_version: str | None = None
    temporal_config_digest: str | None = None

    universe: list[str] = Field(min_length=1)
    window_start: date | None = None
    window_end: date | None = None

    signal_set_hash: str = Field(
        min_length=32,
        description="sha256 of the canonical signal set: a changed signal is "
        "a different run identity, never a silent overwrite",
    )

    initial_cash: float = Field(gt=0)
    seed: int
    randomness_policy: str

    cost_model_id: str | None = None
    costs: dict[str, Any]
    execution: dict[str, Any]
    sizing: str
    long_only: bool
    valuation_price: str
    benchmark: str | None = None

    liquidity_volume_basis: str

    code_hash: str = Field(min_length=32)
    config_hash: str = Field(min_length=32)
    content_hash: str = Field(min_length=32)

    # ------------------------------------------------------------ identity

    def canonical_json(self) -> str:
        payload = json.loads(
            self.model_dump_json(exclude=_OPERATIONAL_FIELDS)
        )
        payload.pop("content_hash", None)
        return json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))

    def compute_content_hash(self) -> str:
        """sha256 of the canonical manifest WITHOUT the stored content_hash,
        run_id and wall-clock fields (the identity a replay must match)."""
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    def derive_run_id(self) -> str:
        """The deterministic run identity: BT-<config8>-<content12>."""
        content = self.compute_content_hash()
        return f"BT-{self.config_hash[:8]}-{content[:12]}"

    def validate_identity(self) -> list[str]:
        """Check that the stored content_hash and run_id match the fields
        (audit: a tampered manifest cannot pass)."""
        violations: list[str] = []
        recomputed = self.compute_content_hash()
        if recomputed != self.content_hash:
            violations.append(
                "content_hash does not match the manifest fields (tampered "
                "manifest?)"
            )
        if self.run_id != self.derive_run_id():
            violations.append(
                f"run_id {self.run_id} does not match the derived identity "
                f"{self.derive_run_id()}"
            )
        return violations


def build_manifest(
    *,
    config: BacktestConfig,
    engine_version: str,
    signal_set_hash: str,
    universe: list[str],
    liquidity_volume_basis: str,
    dataset_snapshot_ids: list[str],
    code_hash: str,
    config_hash: str,
    experiment_id: str | None = None,
    hypothesis_id: str | None = None,
    feature_refs: list[dict[str, Any]] | None = None,
    model: dict[str, Any] | None = None,
    label_id: str | None = None,
    label_version: str | None = None,
    temporal_config_digest: str | None = None,
    cost_model_id: str | None = None,
    created_at: datetime | None = None,
) -> BacktestManifest:
    """Assemble a manifest and compute its content identity."""
    m = BacktestManifest(
        run_id="",
        engine_version=engine_version,
        created_at=created_at,
        experiment_id=experiment_id,
        hypothesis_id=hypothesis_id,
        dataset_snapshot_ids=sorted(set(dataset_snapshot_ids)),
        feature_refs=sorted(feature_refs or [], key=lambda f: json.dumps(f, sort_keys=True)),
        model=model,
        label_id=label_id,
        label_version=label_version,
        temporal_config_digest=temporal_config_digest,
        universe=sorted(set(universe)),
        window_start=config.window_start,
        window_end=config.window_end,
        signal_set_hash=signal_set_hash,
        initial_cash=config.initial_cash,
        seed=config.seed,
        randomness_policy=config.randomness_policy,
        cost_model_id=cost_model_id,
        costs=config.costs.model_dump(mode="json"),
        execution=config.execution.model_dump(mode="json"),
        sizing=config.sizing.value,
        long_only=config.long_only,
        valuation_price=config.valuation_price,
        benchmark=config.benchmark,
        liquidity_volume_basis=liquidity_volume_basis,
        code_hash=code_hash,
        config_hash=config_hash,
        content_hash="0" * 32,  # placeholder; recomputed below
    )
    content_hash = m.compute_content_hash()
    return m.model_copy(update={"content_hash": content_hash, "run_id": m.derive_run_id()})


__all__ = ["MANIFEST_VERSION", "BacktestManifest", "build_manifest"]