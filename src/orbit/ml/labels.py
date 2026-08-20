"""Phase 9 labels: the LAB-004 forward-return benchmark target.

Context: the frozen seed labels LAB-001 and LAB-003 are 5-session EXCESS
returns versus the SPY benchmark (hypotheses H-001 and H-003). The 20-symbol
development universe (DS-000004) contains no SPY series, so the benchmark
term cannot resolve; the excess-return seed labels are therefore
unavailable on the Phase 9 dataset (documented limitation).

LAB-004 v1 keeps the exact horizon (5 sessions) and anchor
(DECISION_INSTANT) of the seed labels but drops the benchmark term:
a pure forward total return. It is registered through the canonical Phase 5
LabelVersionRegistry as a new contract - seed contracts are never mutated.
"""

from __future__ import annotations

from datetime import date

import polars as pl

from orbit.labels import LabelEngine, LabelVersionRegistry
from orbit.labels.contract import (
    AnchorMode,
    CorporateActionPolicy,
    DelistingPolicy,
    LabelContract,
    LabelType,
    MissingDataPolicy,
    OverlapPolicy,
    ReturnConvention,
)
from orbit.labels.engine import ENGINE_VERSION
from orbit.labels.registry import LabelVersionRecord
from orbit.labels.snapshot import LabelSnapshot

LABEL_ID = "LAB-004"
LABEL_VERSION = "v1"
LABEL_HORIZON_SESSIONS = 5

LAB004_FORMULA = (
    "5-session forward total return anchored at the decision instant: "
    "entry is the last completed session strictly before the decision "
    "instant, outcome closes over the next 5 sessions, ex-date dividends "
    "reinvested at the ex-date close (SIMPLE_TOTAL_RETURN). Identical "
    "horizon and anchor to seed LAB-001 without the SPY excess term "
    "(no SPY series in the 20-symbol dev universe)."
)


def build_phase9_label_contract() -> LabelContract:
    return LabelContract(
        label_id=LABEL_ID,
        version=LABEL_VERSION,
        target_type=LabelType.FORWARD_RETURN,
        horizon=LABEL_HORIZON_SESSIONS,
        anchor_mode=AnchorMode.DECISION_INSTANT,
        price_field="close",
        return_convention=ReturnConvention.SIMPLE_TOTAL_RETURN,
        overlap_policy=OverlapPolicy.WINDOWS_TRACKED,
        missing_data_policy=MissingDataPolicy.EXPLICIT_UNAVAILABLE,
        delisting_policy=DelistingPolicy.UNAVAILABLE_WITH_REASON,
        corporate_action_policy=CorporateActionPolicy.CANONICAL_PHASE3,
        formula=LAB004_FORMULA,
        description="Phase 9 baseline ML benchmark target (forward total return)",
        author="orbit-research",
    )


def register_phase9_label(registry: LabelVersionRegistry) -> LabelVersionRecord:
    """Register LAB-004 v1 in the Phase 5 registry (idempotent-friendly:
    raises if an incompatible version already exists)."""
    contract = build_phase9_label_contract()
    record = registry.register(
        contract,
        note="Phase 9 baseline ML benchmark target; benchmark-free forward "
        "return variant of seed LAB-001 (SPY unavailable in dev universe)",
    )
    return record


def build_phase9_label_snapshot(
    bars: pl.DataFrame,
    events: pl.DataFrame | None,
    instruments: list | None,
    decision_rows: pl.DataFrame,
    data_refs: list[str] | None = None,
) -> LabelSnapshot:
    """Compute the LAB-004 v1 label snapshot for the given decision rows.

    `decision_rows` must carry `instrument_id` and `decision_time` (the
    feature snapshot's decision rows). The Phase 5 LabelEngine computes the
    strict-boundary outcome for every row; unavailable rows carry
    EXPLICIT_UNAVAILABLE reasons and are excluded from training by the
    documented policy.
    """
    engine = LabelEngine(bars=bars, events=events, instruments=instruments)
    contract = build_phase9_label_contract()
    records = engine.compute(contract, decision_rows)
    snapshot = LabelSnapshot(
        label_id=contract.label_id,
        version=contract.version,
        contract_digest=contract.content_hash(),
        engine_version=ENGINE_VERSION,
        data_refs=data_refs or [],
        records=records,
    )
    return snapshot


__all__ = [
    "LABEL_ID",
    "LABEL_VERSION",
    "LABEL_HORIZON_SESSIONS",
    "build_phase9_label_contract",
    "register_phase9_label",
    "build_phase9_label_snapshot",
]