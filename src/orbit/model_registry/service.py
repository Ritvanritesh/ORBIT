"""Phase 14 model registry service surface.

The operational engine lives in `orbit.model_registry.core.ModelRegistry`.
This module re-exports it under the service name for API stability.
"""

from orbit.model_registry.core import (  # noqa: F401
    EvidenceType,
    GateResult,
    LifecycleStatus,
    ModelRegistry as ModelRegistryService,
    PromotionAction,
    RegistryViolation,
    ReplayMode,
)

__all__ = ["ModelRegistryService", "LifecycleStatus", "PromotionAction",
           "GateResult", "EvidenceType", "ReplayMode", "RegistryViolation"]
