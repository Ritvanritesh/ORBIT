"""Phase 6 - Experiment Registry Foundation.

Every meaningful research experiment becomes a reproducible, immutable
research object. The registry records successful, unsuccessful, null, failed
and abandoned experiments alike, preserves parent/child genealogy, pins exact
data/label/feature/model/temporal/cost lineage, computes trial/search depth,
and exposes reproduction specifications - so ORBIT can answer, forever:

    "Given an experiment ID, what was tested, with what data, labels,
     features, model, configuration and cost assumptions; where did it come
     from; how much searching preceded it; what happened; and why was the
     result accepted or rejected?"

Module layout:
    lifecycle    validated experiment state machine
    registry     DuckDB-backed ledger with database constraints
    service      ExperimentService: the research-control API
    reproduction ReproductionSpec: full lineage resolution + digest

The canonical, frozen ExperimentSpec schema lives in `orbit.schemas.experiment`
(Phase 1 contract extended with Phase 6 lineage fields); the Phase 1 in-memory
ExperimentRegistry there remains the governance/budget helper.
"""

from orbit.experiments.lifecycle import (
    DECISION_STATES,
    PARENT_ELIGIBLE_STATES,
    POST_COMPLETION_STATES,
    TERMINAL_STATES,
    TRANSITIONS,
    allowed_targets,
    validate_transition,
)
from orbit.experiments.registry import ExperimentRegistry
from orbit.experiments.reproduction import ReproductionSpec, build_reproduction_spec
from orbit.experiments.service import (
    Decision,
    ExperimentService,
    ResultKind,
    temporal_config_digest,
)

__all__ = [
    "DECISION_STATES",
    "Decision",
    "ExperimentRegistry",
    "ExperimentService",
    "PARENT_ELIGIBLE_STATES",
    "POST_COMPLETION_STATES",
    "ReproductionSpec",
    "ResultKind",
    "TERMINAL_STATES",
    "TRANSITIONS",
    "allowed_targets",
    "build_reproduction_spec",
    "temporal_config_digest",
    "validate_transition",
]