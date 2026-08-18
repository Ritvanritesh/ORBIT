"""Phase 5 - Labeling and Outcome Engine.

Every prediction target is one immutable, versioned `LabelContract`; the
`LabelEngine` computes exactly one reproducible future outcome per decision
row, and a `LabelSnapshot` pins the batch with a deterministic digest.

    LabelContract (registered, immutable)
        + decisions + canonical market bars (Phase 3/4)
        -> LabelEngine.compute() -> LabelSnapshot (deterministic)

Module layout:
    contract    LabelContract + all semantic enums + Phase 1 bridge
    outcomes    pure outcome mathematics (golden-testable formulas)
    engine      LabelEngine: entry/window resolution, delisting, benchmark,
                overlap metadata, unavailable reasons
    snapshot    LabelSnapshot: deterministic digest + provenance
    registry    LabelVersionRegistry: immutability and version resolution
    seeds       registered contracts for the seed hypotheses
"""

from orbit.labels.contract import (
    AnchorMode,
    CorporateActionPolicy,
    DelistingPolicy,
    DrawdownType,
    HORIZON_SESSIONS,
    LabelContract,
    MissingDataPolicy,
    OverlapPolicy,
    PriceField,
    ReturnConvention,
    SESSION_ANNUALIZATION,
    VolatilityEstimator,
    contract_from_hypothesis_label,
)
from orbit.labels.engine import (
    ENGINE_VERSION,
    LABEL_OUTPUT_COLUMNS,
    LabelEngine,
    UnavailableReason,
    empty_label_frame,
    overlapping_pairs,
)
from orbit.labels.outcomes import (
    compute_drawdown,
    compute_return,
    excess_return,
    max_adverse_excursion,
    max_drawdown,
    realized_volatility,
    sample_std,
    simple_return,
    window_returns,
    window_total_return,
)
from orbit.labels.registry import LabelVersionRecord, LabelVersionRegistry
from orbit.labels.seeds import SPY_BENCHMARK, build_seed_label_registry
from orbit.labels.snapshot import LabelSnapshot, empty_label_snapshot

__all__ = [
    "AnchorMode",
    "CorporateActionPolicy",
    "DelistingPolicy",
    "DrawdownType",
    "ENGINE_VERSION",
    "HORIZON_SESSIONS",
    "LABEL_OUTPUT_COLUMNS",
    "LabelContract",
    "LabelEngine",
    "LabelSnapshot",
    "LabelVersionRecord",
    "LabelVersionRegistry",
    "MissingDataPolicy",
    "OverlapPolicy",
    "PriceField",
    "ReturnConvention",
    "SESSION_ANNUALIZATION",
    "SPY_BENCHMARK",
    "UnavailableReason",
    "VolatilityEstimator",
    "build_seed_label_registry",
    "compute_drawdown",
    "compute_return",
    "contract_from_hypothesis_label",
    "empty_label_frame",
    "empty_label_snapshot",
    "excess_return",
    "max_adverse_excursion",
    "max_drawdown",
    "overlapping_pairs",
    "realized_volatility",
    "sample_std",
    "simple_return",
    "window_returns",
    "window_total_return",
]