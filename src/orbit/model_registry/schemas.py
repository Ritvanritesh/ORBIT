"""Phase 14 model registry contracts.

Canonical implementation lives in `orbit.model_registry.core`.
This module re-exports the public contract surface for import stability.
"""

from orbit.model_registry.core import (  # noqa: F401
    EvidenceType,
    GateResult,
    IDENTITY_FIELDS,
    LifecycleStatus,
    ModelRegistry,
    PHASE_CLOCK,
    PromotionAction,
    PROMOTION_POLICY,
    POLICY_VERSION,
    RegistryViolation,
    ReplayMode,
    digest_full,
    digest_short,
    file_sha256,
    identity_digest,
    identity_of,
    validate_model_payload,
)

__all__ = [
    "EvidenceType", "GateResult", "IDENTITY_FIELDS", "LifecycleStatus",
    "ModelRegistry", "PHASE_CLOCK", "PromotionAction", "PROMOTION_POLICY",
    "POLICY_VERSION", "RegistryViolation", "ReplayMode", "digest_full",
    "digest_short", "file_sha256", "identity_digest", "identity_of",
    "validate_model_payload",
]
