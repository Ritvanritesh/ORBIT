"""Phase 10 pre-registered plan tests: lock, determinism, model-point scope,
experiment-id mapping, and adversarial guardrails."""

from __future__ import annotations

import pytest

from orbit.ml.features import PHASE10_FEATURE_SET_ORDER
from orbit.ml.phase10_plan import (
    EXPERIMENT_COUNT,
    PHASE10_MODEL_ORDER,
    PHASE10_MODEL_POINTS,
    phase10_experiment_id,
    phase10_model_point_for,
    phase10_plan,
    phase10_plan_digest,
    validate_phase10_plan,
)


def test_plan_is_locked_and_deterministic():
    assert phase10_plan_digest() == "16d62bff387704746fe2ac23742045dcf27314109957752473ed4b0edff64910"
    a = phase10_plan()
    b = phase10_plan()
    assert a["plan_digest"] == b["plan_digest"]
    assert a == b


def test_plan_has_52_experiments():
    assert EXPERIMENT_COUNT == 52
    plan = phase10_plan()
    assert plan["experiment_count"] == 52
    assert plan["experiment_id_range"] == "EXP-10001..EXP-10052"


def test_four_model_points_one_per_phase9_family():
    assert PHASE10_MODEL_ORDER == ["ridge", "lasso", "random_forest", "xgboost"]
    assert [m["family"] for m in PHASE10_MODEL_POINTS] == PHASE10_MODEL_ORDER
    # every model point is a subset of the Phase 9 grid (checked in validate)


def test_plan_validates_cleanly():
    validate_phase10_plan()


def test_experiment_ids_cover_the_range_exactly():
    ids = [
        phase10_experiment_id(sid, m["family"], m["params"])
        for sid in PHASE10_FEATURE_SET_ORDER
        for m in PHASE10_MODEL_POINTS
    ]
    assert ids[0] == "EXP-10001"
    assert ids[-1] == "EXP-10052"
    assert len(set(ids)) == 52
    assert sorted(ids) == ids
    for expected, actual in zip(
        [f"EXP-{i:05d}" for i in range(10001, 10053)], ids
    ):
        assert expected == actual


def test_experiment_id_deterministic():
    assert phase10_experiment_id("FS-003", "ridge", {"alpha": 1.0}) == (
        phase10_experiment_id("FS-003", "ridge", {"alpha": 1.0})
    )
    assert phase10_experiment_id("FS-003", "ridge", {"alpha": 1.0}) == "EXP-10009"


def test_non_plan_hyperparameters_refused():
    with pytest.raises(ValueError):
        phase10_experiment_id("FS-003", "ridge", {"alpha": 0.999})
    with pytest.raises(ValueError):
        phase10_model_point_for("ridge", {"alpha": 0.999})
    with pytest.raises(ValueError):
        phase10_model_point_for("xgboost", {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.05})


def test_unknown_feature_set_refused():
    with pytest.raises(ValueError):
        phase10_experiment_id("FS-999", "ridge", {"alpha": 1.0})


def test_model_point_lookup_returns_deep_copy():
    mp = phase10_model_point_for("random_forest", {"n_estimators": 200, "max_depth": 3})
    mp["params"]["n_estimators"] = 999
    again = phase10_model_point_for("random_forest", {"n_estimators": 200, "max_depth": 3})
    assert again["params"]["n_estimators"] == 200


def test_parent_refs_pinned_to_phase9():
    parents = {m["family"]: m["phase9_parent"] for m in PHASE10_MODEL_POINTS}
    assert parents == {
        "ridge": "EXP-90003",
        "lasso": "EXP-90006",
        "random_forest": "EXP-90015",
        "xgboost": "EXP-90019",
    }


def test_set_ordering_matches_phase10_order_constant():
    plan = phase10_plan()
    assert [s["feature_set_id"] for s in plan["feature_sets"]] == PHASE10_FEATURE_SET_ORDER
    assert plan["feature_sets"][0]["role"] == "base"