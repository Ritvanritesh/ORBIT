"""Sanity tests for Phase 1 schemas and seed registry."""

import pytest
from pydantic import ValidationError

from orbit.schemas.common import HypothesisStatus
from orbit.schemas.experiment import ExperimentRegistry, ExperimentSpec, WindowSpec
from orbit.schemas.hypothesis import HypothesisRegistry, HypothesisSpec
from hypotheses.seeds import build_seed_registry, register_seeds


def test_three_seed_hypotheses_exist():
    registry = build_seed_registry()
    assert len(registry.hypotheses) == 3
    assert {h.hypothesis_id for h in registry.hypotheses} == {"H-001", "H-002", "H-003"}


def test_seeds_freeze_on_registration():
    registry = register_seeds()
    assert all(
        h.status == HypothesisStatus.REGISTERED and h.registration_date is not None
        for h in registry.hypotheses
    )


def test_falsification_criteria_are_required():
    with pytest.raises(ValidationError):
        HypothesisSpec(
            hypothesis_id="H-999",
            title="no criteria",
            statement="x predicts y",
            mechanism="reason",
            baseline=["b1"],
            universe="liquid_equity_50_100",
            label={
                "label_type": "excess_return",
                "horizon": "5D",
                "benchmark": "SPY",
                "definition": "def",
            },
            feature_families=["momentum"],
            data_sources=["src"],
            economic_evidence={"oos_rank_ic": 0.03},
        )


def test_registered_hypothesis_requires_date():
    with pytest.raises(ValidationError):
        HypothesisSpec(
            hypothesis_id="H-998",
            title="bad status",
            statement="x predicts y",
            mechanism="reason",
            baseline=["b1"],
            universe="liquid_equity_50_100",
            label={
                "label_type": "excess_return",
                "horizon": "5D",
                "benchmark": "SPY",
                "definition": "def",
            },
            feature_families=["momentum"],
            data_sources=["src"],
            economic_evidence={"oos_rank_ic": 0.03},
            falsification_criteria="never",
            status="registered",
        )


def test_window_order_enforced():
    with pytest.raises(ValidationError):
        WindowSpec(
            train_start="2015-01-01",
            train_end="2020-01-01",
            val_start="2019-01-01",  # before train_end
            val_end="2021-01-01",
            test_start="2022-01-01",
            test_end="2023-01-01",
        )


def test_experiment_spec_and_registry():
    exp = ExperimentSpec(
        experiment_id="EXP-00001",
        hypothesis_id="H-001",
        title="momentum baseline",
        datasets=["market_daily_v1"],
        features={"feature_names": ["ret_12m_1m"], "feature_version": "v1"},
        model={"family": "heuristic"},
        windows={
            "train_start": "2015-01-01",
            "train_end": "2020-01-01",
            "val_start": "2020-01-02",
            "val_end": "2021-01-01",
            "test_start": "2021-01-02",
            "test_end": "2022-01-01",
        },
    )
    registry = ExperimentRegistry()
    registry.register(exp)
    assert registry.trials_for("H-001") == 1
    assert len(exp.content_hash()) == 64


def test_experiment_genealogy_enforced():
    registry = ExperimentRegistry()
    with pytest.raises(ValueError):
        registry.register(
            ExperimentSpec(
                experiment_id="EXP-00002",
                hypothesis_id="H-001",
                title="orphan variant",
                parent_id="EXP-99999",
                datasets=["d"],
                features={"feature_names": ["f"], "feature_version": "v1"},
                model={"family": "linear"},
                windows={
                    "train_start": "2015-01-01",
                    "train_end": "2020-01-01",
                    "val_start": "2020-01-02",
                    "val_end": "2021-01-01",
                    "test_start": "2021-01-02",
                    "test_end": "2022-01-01",
                },
            )
        )


def test_seed_specs_are_experiment_safe():
    registry = build_seed_registry()
    for h in registry.hypotheses:
        assert h.economic_evidence.cost_model.spread_bps >= 0
        assert h.label.horizon in ("1D", "5D", "21D", "63D")


def test_registered_status_with_date_is_valid():
    spec = HypothesisSpec(
        hypothesis_id="H-997",
        title="valid direct registration",
        statement="x predicts y",
        mechanism="reason",
        baseline=["b1"],
        universe="liquid_equity_50_100",
        label={
            "label_type": "excess_return",
            "horizon": "5D",
            "benchmark": "SPY",
            "definition": "def",
        },
        feature_families=["momentum"],
        data_sources=["src"],
        economic_evidence={"oos_rank_ic": 0.03},
        falsification_criteria="never",
        status="registered",
        registration_date="2026-01-01",
    )
    assert spec.status == HypothesisStatus.REGISTERED


def test_register_is_strict_and_idempotent():
    registry = build_seed_registry()
    once = registry.register_all()
    assert all(h.status == HypothesisStatus.REGISTERED for h in once.hypotheses)
    twice = once.register_all()
    assert [h.registration_date for h in twice.hypotheses] == [
        h.registration_date for h in once.hypotheses
    ]
    with pytest.raises(ValueError):
        once.hypotheses[0].register()


def test_cost_model_rejects_negative_bps():
    with pytest.raises(ValidationError):
        HypothesisSpec(
            hypothesis_id="H-996",
            title="bad costs",
            statement="x predicts y",
            mechanism="reason",
            baseline=["b1"],
            universe="liquid_equity_50_100",
            label={
                "label_type": "excess_return",
                "horizon": "5D",
                "benchmark": "SPY",
                "definition": "def",
            },
            feature_families=["momentum"],
            data_sources=["src"],
            economic_evidence={"oos_rank_ic": 0.03, "cost_model": {"spread_bps": -1}},
            falsification_criteria="never",
        )


def test_specs_are_immutable_after_construction():
    spec = build_seed_registry().hypotheses[0]
    with pytest.raises(ValidationError):
        spec.title = "mutated"
    exp = ExperimentSpec(
        experiment_id="EXP-00003",
        hypothesis_id="H-001",
        title="frozen",
        datasets=["d"],
        features={"feature_names": ["f"], "feature_version": "v1"},
        model={"family": "linear"},
        windows={
            "train_start": "2015-01-01",
            "train_end": "2020-01-01",
            "val_start": "2020-01-02",
            "val_end": "2021-01-01",
            "test_start": "2021-01-02",
            "test_end": "2022-01-01",
        },
    )
    with pytest.raises(ValidationError):
        exp.title = "mutated"


def test_experiment_registry_enforces_hypothesis_lineage():
    hypotheses = register_seeds()
    registry = ExperimentRegistry()
    orphan = ExperimentSpec(
        experiment_id="EXP-00004",
        hypothesis_id="H-999",
        title="orphan",
        datasets=["d"],
        features={"feature_names": ["f"], "feature_version": "v1"},
        model={"family": "linear"},
        windows={
            "train_start": "2015-01-01",
            "train_end": "2020-01-01",
            "val_start": "2020-01-02",
            "val_end": "2021-01-01",
            "test_start": "2021-01-02",
            "test_end": "2022-01-01",
        },
    )
    with pytest.raises(ValueError):
        registry.register(orphan, hypothesis_registry=hypotheses)


def test_economic_evidence_cannot_be_vacuous():
    with pytest.raises(ValidationError):
        HypothesisSpec(
            hypothesis_id="H-995",
            title="no thresholds",
            statement="x predicts y",
            mechanism="reason",
            baseline=["b1"],
            universe="liquid_equity_50_100",
            label={
                "label_type": "excess_return",
                "horizon": "5D",
                "benchmark": "SPY",
                "definition": "def",
            },
            feature_families=["momentum"],
            data_sources=["src"],
            economic_evidence={},
            falsification_criteria="never",
        )
    # research-quality evidence may skip economic thresholds
    HypothesisSpec(
        hypothesis_id="H-994",
        title="research quality ok",
        statement="x predicts y",
        mechanism="reason",
        baseline=["b1"],
        universe="liquid_equity_50_100",
        label={
            "label_type": "excess_return",
            "horizon": "5D",
            "benchmark": "SPY",
            "definition": "def",
        },
        feature_families=["momentum"],
        data_sources=["src"],
        economic_evidence={},
        falsification_criteria="never",
        evidence_type="research_quality",
    )
