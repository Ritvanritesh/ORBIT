"""Phase 15 — Core Explanation Schemas v1."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional
from datetime import datetime


@dataclass(frozen=True)
class ExplanationRecord:
    """Immutable explanation record - sufficient to reconstruct without LLM."""
    explanation_id: str
    prediction_id: str
    model_id: str
    model_version: str
    dataset_snapshot_id: str
    feature_set_id: str
    feature_snapshot_id: str
    feature_version_digest: str
    timestamp: str
    instrument_id: str
    prediction_value: float
    prediction_rank: Optional[float]
    explanation_method: str
    explanation_method_version: str
    feature_names: tuple[str, ...]
    feature_values: tuple[float, ...]
    attribution_values: tuple[float, ...]
    attribution_direction: tuple[str, ...]  # "positive", "negative", "neutral"
    baseline_value: Optional[float]
    code_hash: str
    config_hash: str
    deterministic_digest: str

    def to_dict(self) -> dict:
        return {
            "explanation_id": self.explanation_id,
            "prediction_id": self.prediction_id,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "dataset_snapshot_id": self.dataset_snapshot_id,
            "feature_set_id": self.feature_set_id,
            "feature_snapshot_id": self.feature_snapshot_id,
            "feature_version_digest": self.feature_version_digest,
            "timestamp": self.timestamp,
            "instrument_id": self.instrument_id,
            "prediction_value": self.prediction_value,
            "prediction_rank": self.prediction_rank,
            "explanation_method": self.explanation_method,
            "explanation_method_version": self.explanation_method_version,
            "feature_names": list(self.feature_names),
            "feature_values": list(self.feature_values),
            "attribution_values": list(self.attribution_values),
            "attribution_direction": list(self.attribution_direction),
            "baseline_value": self.baseline_value,
            "code_hash": self.code_hash,
            "config_hash": self.config_hash,
            "deterministic_digest": self.deterministic_digest,
        }


@dataclass(frozen=True)
class AttributionRecord:
    """Global or local attribution record."""
    attribution_id: str
    explanation_id: str
    method: str
    method_version: str
    feature_names: tuple[str, ...]
    attribution_values: tuple[float, ...]
    attribution_direction: tuple[str, ...]
    baseline: Optional[float]
    stability_metrics: dict
    deterministic_digest: str

    def to_dict(self) -> dict:
        return {
            "attribution_id": self.attribution_id,
            "explanation_id": self.explanation_id,
            "method": self.method,
            "method_version": self.method_version,
            "feature_names": list(self.feature_names),
            "attribution_values": list(self.attribution_values),
            "attribution_direction": list(self.attribution_direction),
            "baseline": self.baseline,
            "stability_metrics": self.stability_metrics,
            "deterministic_digest": self.deterministic_digest,
        }


@dataclass(frozen=True)
class CounterfactualRecord:
    """Counterfactual analysis record."""
    counterfactual_id: str
    prediction_id: str
    model_id: str
    model_version: str
    original_feature_values: tuple[float, ...]
    counterfactual_feature_values: tuple[float, ...]
    feature_names: tuple[str, ...]
    changed_features: tuple[str, ...]
    original_prediction: float
    counterfactual_prediction: float
    distance: float
    domain_valid: bool
    classification: str  # CLEAR, HIGH_DIMENSIONAL, UNSTABLE, NO_VALID_COUNTERFACTUAL
    deterministic_digest: str

    def to_dict(self) -> dict:
        return {
            "counterfactual_id": self.counterfactual_id,
            "prediction_id": self.prediction_id,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "original_feature_values": list(self.original_feature_values),
            "counterfactual_feature_values": list(self.counterfactual_feature_values),
            "feature_names": list(self.feature_names),
            "changed_features": list(self.changed_features),
            "original_prediction": self.original_prediction,
            "counterfactual_prediction": self.counterfactual_prediction,
            "distance": self.distance,
            "domain_valid": self.domain_valid,
            "classification": self.classification,
            "deterministic_digest": self.deterministic_digest,
        }


@dataclass(frozen=True)
class DecisionRecord:
    """Decision record linking prediction, explanation, and evidence."""
    decision_id: str
    prediction_id: str
    explanation_id: str
    attribution_id: str
    model_id: str
    model_version: str
    feature_snapshot_id: str
    feature_version_digest: str
    registry_evidence_ids: tuple[str, ...]
    current_evidence_status: str  # RESEARCH, VALIDATED, etc.
    known_limitations: tuple[str, ...]
    disagreement_metrics: dict
    explanation_confidence: str  # HIGH, MEDIUM, LOW, UNRELIABLE
    validation_status: str  # PASS, LIMITATION, FAIL
    deterministic_digest: str

    def to_dict(self) -> dict:
        return {
            "decision_id": self.decision_id,
            "prediction_id": self.prediction_id,
            "explanation_id": self.explanation_id,
            "attribution_id": self.attribution_id,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "feature_snapshot_id": self.feature_snapshot_id,
            "feature_version_digest": self.feature_version_digest,
            "registry_evidence_ids": list(self.registry_evidence_ids),
            "current_evidence_status": self.current_evidence_status,
            "known_limitations": list(self.known_limitations),
            "disagreement_metrics": self.disagreement_metrics,
            "explanation_confidence": self.explanation_confidence,
            "validation_status": self.validation_status,
            "deterministic_digest": self.deterministic_digest,
        }


@dataclass(frozen=True)
class ExplanationValidationRecord:
    """Validation of explanation faithfulness and stability."""
    validation_id: str
    explanation_id: str
    faithfulness_deletion_correlation: Optional[float]
    faithfulness_insertion_correlation: Optional[float]
    local_stability_classification: str  # STABLE, SENSITIVE, CLIFF, INVALID_DOMAIN
    counterfactual_classification: str  # CLEAR, HIGH_DIMENSIONAL, UNSTABLE, NO_VALID_COUNTERFACTUAL
    permutation_stability_rank_corr: Optional[float]
    sensitivity_max_delta: Optional[float]
    correlation_instability_flag: bool
    overall_validation: str  # PASS, LIMITATION, FAIL
    deterministic_digest: str

    def to_dict(self) -> dict:
        return {
            "validation_id": self.validation_id,
            "explanation_id": self.explanation_id,
            "faithfulness_deletion_correlation": self.faithfulness_deletion_correlation,
            "faithfulness_insertion_correlation": self.faithfulness_insertion_correlation,
            "local_stability_classification": self.local_stability_classification,
            "counterfactual_classification": self.counterfactual_classification,
            "permutation_stability_rank_corr": self.permutation_stability_rank_corr,
            "sensitivity_max_delta": self.sensitivity_max_delta,
            "correlation_instability_flag": self.correlation_instability_flag,
            "overall_validation": self.overall_validation,
            "deterministic_digest": self.deterministic_digest,
        }


@dataclass(frozen=True)
class ModelDisagreementRecord:
    """Model disagreement analysis record."""
    disagreement_id: str
    prediction_id: str
    model_ids: tuple[str, ...]
    model_versions: tuple[str, ...]
    predictions: tuple[float, ...]
    pairwise_correlations: dict
    sign_agreement: float
    rank_agreement: float
    top_k_overlap: dict
    disagreement_magnitude: float
    classification: str  # HIGH_AGREEMENT, MODERATE_AGREEMENT, LOW_AGREEMENT, SIGN_CONFLICT
    deterministic_digest: str

    def to_dict(self) -> dict:
        return {
            "disagreement_id": self.disagreement_id,
            "prediction_id": self.prediction_id,
            "model_ids": list(self.model_ids),
            "model_versions": list(self.model_versions),
            "predictions": list(self.predictions),
            "pairwise_correlations": self.pairwise_correlations,
            "sign_agreement": self.sign_agreement,
            "rank_agreement": self.rank_agreement,
            "top_k_overlap": self.top_k_overlap,
            "disagreement_magnitude": self.disagreement_magnitude,
            "classification": self.classification,
            "deterministic_digest": self.deterministic_digest,
        }


# Deterministic digest helper
def canonical_json(obj: Any) -> str:
    import json
    return json.dumps(obj, sort_keys=True, default=str)


def deterministic_digest(obj: Any, length: int = 16) -> str:
    import hashlib
    return hashlib.sha256(canonical_json(obj).encode()).hexdigest()[:length]