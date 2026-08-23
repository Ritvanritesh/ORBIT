"""Phase 14 — Model Registry & Evidence-Gated Promotion."""

from orbit.model_registry.core import (
    EvidenceType,
    GateResult,
    LifecycleStatus,
    ModelRegistry,
    PHASE_CLOCK,
    PromotionAction,
    RegistryViolation,
    ReplayMode,
    identity_digest,
    validate_model_payload,
)

__all__ = [
    "EvidenceType", "GateResult", "LifecycleStatus", "ModelRegistry",
    "PHASE_CLOCK", "PromotionAction", "RegistryViolation", "ReplayMode",
    "identity_digest", "validate_model_payload",
]
