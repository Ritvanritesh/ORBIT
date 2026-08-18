"""Experiment lifecycle state machine (Phase 6).

The lifecycle is explicit and validated: no arbitrary status mutation.
Invalid transitions fail loudly. REJECTED and PROMOTED are DECISION states:
they may only be entered from COMPLETED through a recorded selection decision
(`ExperimentService.record_decision`), never by a bare status update.

    DRAFT -> REGISTERED -> RUNNING -> COMPLETED -> REJECTED
                                     \         \-> PROMOTED
                            \-> FAILED             \
                                     \              -> RETIRED (archival)
    COMPLETED -> RETIRED
    FAILED -> RETIRED
    REJECTED -> RETIRED
    PROMOTED -> RETIRED
    RETIRED (terminal; history is never destroyed, only archived)
"""

from __future__ import annotations

from orbit.schemas.common import ExperimentStatus

TERMINAL_STATES = frozenset(
    {ExperimentStatus.REJECTED, ExperimentStatus.PROMOTED, ExperimentStatus.RETIRED}
)

# States from which an experiment may still take children (a parent that is
# only a draft is not a real experiment; a retired parent is archived and can
# never be the basis of new research - soft deletion of the lineage anchor).
PARENT_ELIGIBLE_STATES = frozenset(
    {
        ExperimentStatus.REGISTERED,
        ExperimentStatus.RUNNING,
        ExperimentStatus.COMPLETED,
        ExperimentStatus.FAILED,
        ExperimentStatus.REJECTED,
        ExperimentStatus.PROMOTED,
    }
)

TRANSITIONS: dict[ExperimentStatus, frozenset[ExperimentStatus]] = {
    ExperimentStatus.DRAFT: frozenset({ExperimentStatus.REGISTERED}),
    ExperimentStatus.REGISTERED: frozenset(
        {ExperimentStatus.RUNNING, ExperimentStatus.FAILED, ExperimentStatus.RETIRED}
    ),
    ExperimentStatus.RUNNING: frozenset(
        {ExperimentStatus.COMPLETED, ExperimentStatus.FAILED}
    ),
    ExperimentStatus.COMPLETED: frozenset(
        {ExperimentStatus.REJECTED, ExperimentStatus.PROMOTED, ExperimentStatus.RETIRED}
    ),
    ExperimentStatus.FAILED: frozenset({ExperimentStatus.RETIRED}),
    ExperimentStatus.REJECTED: frozenset({ExperimentStatus.RETIRED}),
    ExperimentStatus.PROMOTED: frozenset({ExperimentStatus.RETIRED}),
    ExperimentStatus.RETIRED: frozenset(),
}

# REJECTED / PROMOTED are only reachable through a recorded decision.
DECISION_STATES = frozenset({ExperimentStatus.REJECTED, ExperimentStatus.PROMOTED})

# A completed experiment's scientific identity is fixed; REJECTED/PROMOTED are
# the only post-completion transitions besides archival.
POST_COMPLETION_STATES = frozenset(
    {ExperimentStatus.REJECTED, ExperimentStatus.PROMOTED, ExperimentStatus.RETIRED}
)


def validate_transition(current: ExperimentStatus, target: ExperimentStatus) -> None:
    """Raise ValueError when `current -> target` is not a valid transition."""
    if not isinstance(current, ExperimentStatus):
        current = ExperimentStatus(current)
    if not isinstance(target, ExperimentStatus):
        target = ExperimentStatus(target)
    allowed = TRANSITIONS.get(current, frozenset())
    if target not in allowed:
        order = ", ".join(sorted(s.value for s in allowed)) or "none"
        raise ValueError(
            f"invalid experiment transition: {current.value} -> {target.value} "
            f"(allowed from {current.value}: {order})"
        )


def allowed_targets(current: ExperimentStatus) -> list[ExperimentStatus]:
    return sorted(TRANSITIONS.get(current, frozenset()), key=lambda s: s.value)


__all__ = [
    "DECISION_STATES",
    "PARENT_ELIGIBLE_STATES",
    "POST_COMPLETION_STATES",
    "TERMINAL_STATES",
    "TRANSITIONS",
    "allowed_targets",
    "validate_transition",
]