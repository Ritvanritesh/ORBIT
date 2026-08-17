"""Phase 4 - Point-in-Time & Temporal Truth Engine.

Makes ORBIT only use information that would genuinely have been available
at the time it makes a historical prediction. Core rule:

    allowed(x, t) = publication_time(x) < t
                    AND all other effective-time constraints hold

Module layout:
    times       temporal field definitions + documented conventions
    rules       the availability decision function (per-record)
    adapters    normalized parquet -> canonical timing frames
    engine      TemporalTruthEngine: as-of snapshots, vintages, joins
    snapshot    PointInTimeSnapshot: reproducible + provable information set
    features    feature-time rules future features must obey
    fixtures    synthetic future-leak fixtures (permanent regression inputs)
    contracts   machine-readable temporal contract (configs/temporal.json)
"""

from orbit.temporal.contracts import TemporalContract, load_temporal_contract
from orbit.temporal.engine import (
    Evaluation,
    SourceInput,
    TemporalTruthEngine,
    build_temporal_source,
)
from orbit.temporal.rules import (
    AvailabilityDecision,
    RuleTrace,
    decide,
    decide_frame,
    reasons_summary,
    trace_rule,
)
from orbit.temporal.snapshot import PointInTimeSnapshot, TemporalSource
from orbit.temporal.times import (
    DecisionCode,
    TimePrecision,
    Timing,
    normalize_instant,
    session_close_utc,
)

__all__ = [
    "AvailabilityDecision",
    "DecisionCode",
    "Evaluation",
    "PointInTimeSnapshot",
    "RuleTrace",
    "SourceInput",
    "TemporalContract",
    "TemporalSource",
    "TemporalTruthEngine",
    "TimePrecision",
    "Timing",
    "build_temporal_source",
    "decide",
    "decide_frame",
    "load_temporal_contract",
    "normalize_instant",
    "reasons_summary",
    "session_close_utc",
    "trace_rule",
]