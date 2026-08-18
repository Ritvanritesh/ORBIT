"""Phase 5 contract tests: the LabelContract is the single source of truth
for what a prediction target means. These tests pin its validation rules:
semantics-bearing fields are required, composite labels are deferred, the
content hash is a stable formula identity, and the Phase 1 bridge never
changes the researcher-facing hypothesis registry.
"""

from __future__ import annotations

from datetime import date

import pytest
from pydantic import ValidationError

from orbit.labels.contract import (
    AnchorMode,
    HORIZON_SESSIONS,
    LabelContract,
    ReturnConvention,
    contract_from_hypothesis_label,
)
from orbit.labels.registry import LabelVersionRegistry
from orbit.labels.seeds import build_seed_label_registry
from orbit.schemas.common import Horizon, LabelType
from orbit.schemas.hypothesis import HypothesisSpec


def _base(**kw) -> LabelContract:
    b = dict(
        label_id="LAB-001",
        version="v1",
        target_type="forward_return",
        horizon=5,
        anchor_mode=AnchorMode.DECISION_INSTANT,
        return_convention=ReturnConvention.SIMPLE_PRICE_RETURN,
        formula="test formula",
    )
    b.update(kw)
    return LabelContract(**b)


# ------------------------------------------------------------ field rules

def test_contract_is_frozen_and_basic_fields():
    c = _base()
    assert c.label_id == "LAB-001"
    assert c.version == "v1"
    assert c.horizon_semantics == "trading_sessions"
    assert c.anchor_mode == AnchorMode.DECISION_INSTANT
    with pytest.raises(ValidationError):
        c.label_id = "LAB-999"  # frozen


def test_horizon_must_be_positive():
    with pytest.raises(ValidationError):
        _base(horizon=0)
    with pytest.raises(ValidationError):
        _base(horizon=-3)


def test_horizon_semantics_is_a_fixed_literal():
    # the engine ONLY ever counts trading sessions; a contract claiming any
    # other semantics would silently misdescribe the computed label
    with pytest.raises(ValidationError):
        _base(horizon_semantics="calendar_days")


def test_label_id_and_version_patterns():
    with pytest.raises(ValidationError):
        _base(label_id="bad-id")
    with pytest.raises(ValidationError):
        _base(label_id="LAB-0012")
    with pytest.raises(ValidationError):
        _base(version="1.0")
    with pytest.raises(ValidationError):
        _base(version="v")


def test_target_types_require_their_semantics_fields():
    # excess requires a benchmark
    with pytest.raises(ValidationError):
        _base(target_type="excess_return", benchmark=None)
    # volatility requires estimator + annualization + min observations
    with pytest.raises(ValidationError):
        _base(target_type="volatility")
    # drawdown requires a drawdown type
    with pytest.raises(ValidationError):
        _base(target_type="drawdown")


def test_risk_adjusted_return_is_deferred_and_rejected():
    with pytest.raises(ValidationError):
        _base(target_type=LabelType.RISK_ADJUSTED_RETURN)


def test_return_convention_required_for_returns():
    with pytest.raises(ValidationError):
        _base(target_type="forward_return", return_convention=None)


def test_min_observations_bounds():
    _base(target_type="volatility", horizon=5,
          volatility_estimator="sample_std_close_to_close_daily_returns",
          annualization=252.0, min_observations=2)
    with pytest.raises(ValidationError):
        _base(target_type="volatility", horizon=5,
              volatility_estimator="sample_std_close_to_close_daily_returns",
              annualization=252.0, min_observations=1)
    with pytest.raises(ValidationError):
        _base(target_type="volatility", horizon=5,
              volatility_estimator="sample_std_close_to_close_daily_returns",
              annualization=252.0, min_observations=6)


def test_annualization_must_be_positive():
    with pytest.raises(ValidationError):
        _base(target_type="volatility", horizon=5,
              volatility_estimator="sample_std_close_to_close_daily_returns",
              annualization=0.0, min_observations=2)


def test_volatility_and_drawdown_reject_total_return_convention():
    # the estimators/drawdown math has no dividend term; a TOTAL_RETURN
    # convention would misdescribe the computed label
    with pytest.raises(ValidationError, match="PRICE returns"):
        _base(target_type="volatility", horizon=5,
              volatility_estimator="sample_std_close_to_close_daily_returns",
              annualization=252.0, min_observations=2,
              return_convention=ReturnConvention.SIMPLE_TOTAL_RETURN)
    with pytest.raises(ValidationError, match="PRICE returns"):
        _base(target_type="drawdown", horizon=5,
              drawdown_type="max_drawdown",
              return_convention=ReturnConvention.SIMPLE_TOTAL_RETURN)


# ------------------------------------------------------------- formula hash

def test_content_hash_is_stable_and_sensitive():
    c1 = _base(formula="same")
    c2 = _base(formula="same")
    assert c1.content_hash() == c2.content_hash()
    c3 = _base(formula="different formula")
    assert c3.content_hash() != c1.content_hash()
    c4 = _base(horizon=10)
    assert c4.content_hash() != c1.content_hash()
    assert len(c1.content_hash()) == 64


def test_canonical_json_is_deterministic():
    c = _base()
    assert c.canonical_json() == _base(formula="test formula").canonical_json()


def test_horizon_session_map_matches_phase1_enum():
    assert HORIZON_SESSIONS == {
        Horizon.H1: 1, Horizon.H5: 5, Horizon.H21: 21, Horizon.H63: 63,
    }


# -------------------------------------------------------- phase 1 bridge

def test_contract_from_hypothesis_label_maps_semantics():
    spec = HypothesisSpec(
        hypothesis_id="H-003",
        title="PEAD",
        statement="post-earnings drift",
        mechanism="slow diffusion",
        baseline=["b1"],
        universe="liquid_equity_50_100",
        label={
            "label_type": "excess_return",
            "horizon": "5D",
            "benchmark": "SPY",
            "definition": "5-session forward excess total return after filing",
        },
        feature_families=["fundamentals"],
        data_sources=["s"],
        economic_evidence={"oos_rank_ic": 0.02},
        falsification_criteria="falsified if OOS rank IC < 0.02",
    )
    c = contract_from_hypothesis_label(
        spec.label, label_id="LAB-003", anchor_mode=AnchorMode.POST_EVENT,
        return_convention=ReturnConvention.SIMPLE_TOTAL_RETURN,
    )
    assert c.label_id == "LAB-003"
    assert c.version == "v1"
    assert c.target_type.value == "excess_return"
    assert c.horizon == 5
    assert c.benchmark == "SPY"
    assert c.anchor_mode == AnchorMode.POST_EVENT
    assert c.return_convention == ReturnConvention.SIMPLE_TOTAL_RETURN
    assert c.formula


def test_contract_from_hypothesis_label_rejects_risk_adjusted():
    spec = HypothesisSpec(
        hypothesis_id="H-002",
        title="Risk-adjusted momentum",
        statement="s",
        mechanism="m",
        baseline=["b1"],
        universe="liquid_equity_50_100",
        label={
            "label_type": "risk_adjusted_return",
            "horizon": "21D",
            "benchmark": "SPY",
            "definition": "forward return over trailing volatility",
        },
        feature_families=["fundamentals"],
        data_sources=["s"],
        economic_evidence={"oos_rank_ic": 0.02},
        falsification_criteria="falsified if OOS rank IC < 0.02",
    )
    with pytest.raises(ValidationError):
        contract_from_hypothesis_label(
            spec.label, label_id="LAB-002",
            anchor_mode=AnchorMode.DECISION_INSTANT,
            return_convention=ReturnConvention.SIMPLE_TOTAL_RETURN,
        )


# ------------------------------------------------------------- seeds

def test_seed_registry_registers_the_computable_hypothesis_labels():
    reg = build_seed_label_registry()
    assert reg.versions("LAB-001") == ["v1"]
    assert reg.versions("LAB-003") == ["v1"]
    assert reg.versions("LAB-002") == []  # deferred, never registered
    lab1 = reg.definition("LAB-001")
    assert lab1.horizon == 5
    assert lab1.target_type.value == "excess_return"
    assert lab1.benchmark == "SPY"
    assert lab1.return_convention == ReturnConvention.SIMPLE_TOTAL_RETURN
    lab3 = reg.definition("LAB-003")
    assert lab3.anchor_mode == AnchorMode.POST_EVENT
    assert lab3.horizon == 5


def test_seed_contracts_are_versioned_and_immutable():
    reg = build_seed_label_registry()
    record = reg.get("LAB-001", "v1")
    assert record.registered_at == date.today()
    with pytest.raises(ValidationError):
        record.contract.version = "v2"  # frozen contract
    with pytest.raises(ValidationError):
        record.version = "v2"  # frozen record