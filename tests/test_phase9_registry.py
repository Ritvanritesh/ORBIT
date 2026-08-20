"""Phase 9 registry tests: register-before-run, lineage pins, grid lock."""

from __future__ import annotations

import pytest

from orbit.ml.grids import SEED
from orbit.ml.registry import (
    control_experiment_id_for,
    experiment_id_for,
    ml_code_hash,
    phase9_config_hash,
    register_control_experiment,
    register_ml_experiment,
)
from orbit.schemas.common import ExperimentStatus


def test_experiment_ids_are_deterministic_and_unique():
    ids = set()
    for a in (0.01, 0.1, 1.0, 10.0):
        ids.add(experiment_id_for("ridge", {"alpha": a}))
    assert len(ids) == 4
    assert experiment_id_for("ridge", {"alpha": 1.0}) == "EXP-90003"


def test_all_grid_points_get_distinct_ids():
    from orbit.ml.grids import MODEL_FAMILIES, PHASE9_GRIDS

    ids = [
        experiment_id_for(family, params)
        for family in MODEL_FAMILIES
        for params in PHASE9_GRIDS[family]
    ]
    assert len(ids) == 20
    assert len(set(ids)) == 20


def test_control_ids_are_distinct_and_stable():
    from orbit.ml.baselines import CONTROL_GRIDS

    ids = []
    for fam in ("buy_and_hold", "equal_weight"):
        ids.append(control_experiment_id_for(fam, {}))
    for family, grid in CONTROL_GRIDS.items():
        for p in grid:
            ids.append(control_experiment_id_for(family, p))
    ids += [control_experiment_id_for("random_null", {}), control_experiment_id_for("null_flat", {})]
    assert len(ids) == len(set(ids))
    assert control_experiment_id_for("buy_and_hold", {}) == "EXP-90021"
    assert control_experiment_id_for("equal_weight", {}) == "EXP-90022"
    assert control_experiment_id_for("null_flat", {}) == "EXP-90036"


def test_ml_registration_pins_full_lineage():
    service, spec = register_ml_experiment(
        experiment_id="EXP-90005", hypothesis_id="H-001", family="ridge", params={"alpha": 1.0}
    )
    assert spec.status == ExperimentStatus.REGISTERED
    assert spec.dataset_snapshot_ids == ["DS-000004"]
    assert spec.label_id == "LAB-004"
    assert spec.label_version == "v1"
    assert spec.cost_model_id == "CM-001"
    assert spec.seed == 42
    assert spec.model.hyperparameters == {"alpha": 1.0}
    assert spec.evaluation_protocol == "fixed_split_v1"
    assert spec.temporal_config is not None and spec.temporal_config.config_digest


def test_all_five_ml_families_register():
    from orbit.ml.grids import MODEL_FAMILIES, PHASE9_GRIDS

    for family in MODEL_FAMILIES:
        params = PHASE9_GRIDS[family][0]
        _, spec = register_ml_experiment(
            experiment_id="EXP-90005", hypothesis_id="H-001", family=family, params=params
        )
        assert spec.model.family == family


def test_ml_registration_validates_grid_before_run():
    with pytest.raises(ValueError, match="not a pre-registered"):
        register_ml_experiment(
            experiment_id="EXP-90005", hypothesis_id="H-001", family="ridge", params={"alpha": 0.007}
        )


def test_ml_registration_rejects_unknown_family():
    with pytest.raises(ValueError, match="unknown model family"):
        register_ml_experiment(
            experiment_id="EXP-90005", hypothesis_id="H-001", family="svm", params={}
        )


def test_control_registration_validates_grid():
    with pytest.raises(ValueError, match="not pre-registered"):
        register_control_experiment(
            experiment_id="EXP-90021", hypothesis_id="H-001", family="momentum", params={"lookback": 77}
        )


def test_control_registration_rejects_unknown_family():
    with pytest.raises(ValueError, match="unknown control family"):
        register_control_experiment(
            experiment_id="EXP-90021", hypothesis_id="H-001", family="bogus", params={}
        )


def test_code_hash_is_stable_and_long():
    h = ml_code_hash()
    assert len(h) == 64
    assert h == ml_code_hash()


def test_config_hash_changes_with_params_and_seed():
    a = phase9_config_hash("ridge", {"alpha": 1.0}, 42)
    b = phase9_config_hash("ridge", {"alpha": 10.0}, 42)
    c = phase9_config_hash("ridge", {"alpha": 1.0}, 7)
    assert a != b and a != c


def test_registration_refuses_duplicate_experiment_id():
    service, _ = register_ml_experiment(
        experiment_id="EXP-90005", hypothesis_id="H-001", family="ridge", params={"alpha": 1.0}
    )
    with pytest.raises(ValueError, match="duplicate experiment id"):
        service.register(service.get("EXP-90005"))