"""Phase 9 model tests: every family trains, predicts deterministically."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from orbit.ml.grids import MODEL_FAMILIES, PHASE9_GRIDS, SEED, validate_model_parameters
from orbit.ml.models import (
    binary_target,
    predict_with_state,
    train_model,
)
from tests.phase9_testutils import TEST_WINDOWS, build_test_datasets, make_canonical_bars, make_events


@pytest.fixture(scope="module")
def matrices():
    _, _, ds = build_test_datasets(make_canonical_bars(), make_events())
    return ds


@pytest.mark.parametrize("family", MODEL_FAMILIES)
def test_every_family_trains_and_predicts(family, matrices):
    (Xtr, ytr, _, _) = matrices["train"]
    Xte = matrices["test"][0]
    params = PHASE9_GRIDS[family][0]
    model, state = train_model(family, params, Xtr, ytr)
    pred = predict_with_state(model, state, Xte)
    assert pred.shape == (Xte.shape[0],)
    assert np.isfinite(pred).all()


def test_logistic_returns_probabilities(matrices):
    (Xtr, ytr, _, _) = matrices["train"]
    Xte = matrices["test"][0]
    model, state = train_model("logistic", {"C": 1.0}, Xtr, ytr)
    pred = predict_with_state(model, state, Xte)
    assert ((pred >= 0.0) & (pred <= 1.0)).all()


def test_deterministic_training(matrices):
    (Xtr, ytr, _, _) = matrices["train"]
    Xte = matrices["test"][0]
    a, sa = train_model("random_forest", {"n_estimators": 50, "max_depth": 3}, Xtr, ytr)
    b, sb = train_model("random_forest", {"n_estimators": 50, "max_depth": 3}, Xtr, ytr)
    pa = predict_with_state(a, sa, Xte)
    pb = predict_with_state(b, sb, Xte)
    assert np.array_equal(pa, pb)


def test_xgboost_deterministic(matrices):
    (Xtr, ytr, _, _) = matrices["train"]
    Xte = matrices["test"][0]
    a, sa = train_model("xgboost", {"n_estimators": 50, "max_depth": 3, "learning_rate": 0.1}, Xtr, ytr)
    b, sb = train_model("xgboost", {"n_estimators": 50, "max_depth": 3, "learning_rate": 0.1}, Xtr, ytr)
    assert np.array_equal(predict_with_state(a, sa, Xte), predict_with_state(b, sb, Xte))


def test_scaler_is_fit_on_train_only(matrices):
    (Xtr, ytr, _, _) = matrices["train"]
    Xva = matrices["val"][0]
    model, state = train_model("ridge", {"alpha": 1.0}, Xtr, ytr)
    assert state["preprocessing"] == "standardized"
    pred = predict_with_state(model, state, Xva)
    assert pred.shape == (Xva.shape[0],)


def test_grid_validation_accepts_registered_points():
    for family in MODEL_FAMILIES:
        for point in PHASE9_GRIDS[family]:
            assert validate_model_parameters(family, point) == point


def test_grid_validation_rejects_unknown_params():
    with pytest.raises(ValueError, match="not a pre-registered"):
        validate_model_parameters("ridge", {"alpha": 0.007})


def test_grid_validation_rejects_missing_params():
    with pytest.raises(ValueError, match="requires exactly"):
        validate_model_parameters("random_forest", {"n_estimators": 50})


def test_grid_validation_rejects_unknown_family():
    with pytest.raises(ValueError, match="unknown model family"):
        validate_model_parameters("svm", {})


def test_grid_sizes_are_small():
    assert sum(len(g) for g in PHASE9_GRIDS.values()) == 20


def test_seed_is_fixed_at_42():
    assert SEED == 42


def test_binary_target_is_deterministic():
    y = np.array([-0.5, 0.0, 0.1, 1.0])
    assert binary_target(y).tolist() == [0, 0, 1, 1]


def test_linear_coefficients_are_exposed(matrices):
    (Xtr, ytr, _, _) = matrices["train"]
    model, _ = train_model("lasso", {"alpha": 0.001}, Xtr, ytr)
    coefs = model.coefficients()
    assert set(coefs) == set(model.feature_names)
    assert all(np.isfinite(v) for v in coefs.values())


def test_coefficients_refused_for_trees(matrices):
    (Xtr, ytr, _, _) = matrices["train"]
    model, _ = train_model("random_forest", {"n_estimators": 50, "max_depth": 3}, Xtr, ytr)
    with pytest.raises(ValueError, match="linear families"):
        model.coefficients()