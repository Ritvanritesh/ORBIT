"""Phase 9 model families.

Every family is a thin, deterministic wrapper over scikit-learn / xgboost:

  ridge          sklearn Ridge                (deterministic; standardized)
  lasso          sklearn Lasso (CD)           (deterministic; standardized)
  logistic       sklearn LogisticRegression   (lbfgs, deterministic; standardized)
  random_forest  sklearn RandomForestRegressor(random_state=seed, n_jobs=1)
  xgboost        xgboost XGBRegressor         (tree_method=hist, n_jobs=1,
                                               random_state=seed)

Linear families are trained on features standardized with a StandardScaler
fit on the TRAINING split only (recorded in `preprocessing`); tree families
use raw features. The regression target is the LAB-004 forward total return;
the logistic family uses the deterministic binary transform y > 0
(positive forward return) - its scores are probabilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

from orbit.ml.grids import SEED, validate_model_parameters

FEATURE_NAMES_DEFAULT: tuple[str, ...] = (
    "ret_10",
    "ret_20",
    "ret_30",
    "sma_ratio_5_30",
    "sma_ratio_15_40",
    "vol_10",
    "vol_30",
    "log_dv_med_20",
)

LINEAR_FAMILIES = ("ridge", "lasso", "logistic")


@dataclass
class FittedModel:
    family: str
    hyperparameters: dict[str, Any]
    feature_names: list[str]
    preprocessing: str
    seed: int
    estimator: Any
    fitted_train_rows: int
    train_window: tuple[str, str]
    val_window: tuple[str, str]
    test_window: tuple[str, str]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.estimator.predict(X), dtype=np.float64)

    def predict_proba_positive(self, X: np.ndarray) -> np.ndarray:
        if self.family != "logistic":
            raise ValueError("probability output is only defined for the logistic family")
        return np.asarray(self.estimator.predict_proba(X)[:, 1], dtype=np.float64)

    def coefficients(self) -> dict[str, float]:
        if self.family not in LINEAR_FAMILIES:
            raise ValueError(f"coefficients are only defined for linear families; got {self.family}")
        return {name: float(c) for name, c in zip(self.feature_names, self.estimator.coef_.flatten())}


def _build_estimator(family: str, params: dict[str, Any]) -> Any:
    from sklearn.linear_model import Lasso, LogisticRegression, Ridge
    from sklearn.ensemble import RandomForestRegressor
    import xgboost as xgb

    if family == "ridge":
        return Ridge(alpha=params["alpha"])
    if family == "lasso":
        return Lasso(alpha=params["alpha"], max_iter=100000, tol=1e-8, random_state=SEED)
    if family == "logistic":
        return LogisticRegression(
            C=params["C"], solver="lbfgs", max_iter=5000, random_state=SEED
        )
    if family == "random_forest":
        return RandomForestRegressor(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            random_state=SEED,
            n_jobs=1,
        )
    if family == "xgboost":
        return xgb.XGBRegressor(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            tree_method="hist",
            n_jobs=1,
            random_state=SEED,
            objective="reg:squarederror",
        )
    raise ValueError(f"unknown model family {family!r}")


def train_model(
    family: str,
    params: dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str] | None = None,
    windows: dict[str, str] | None = None,
) -> tuple[FittedModel, dict[str, Any]]:
    """Train one pre-registered grid point and return (model, pipeline_state).

    `pipeline_state` carries the fitted scaler (linear families) plus the
    train row count, so prediction can be replayed deterministically.
    """
    validate_model_parameters(family, params)
    feature_names = list(feature_names or FEATURE_NAMES_DEFAULT)
    windows = windows or {
        "train": ("2010-01-04", "2018-12-31"),
        "val": ("2019-01-02", "2021-12-31"),
        "test": ("2022-01-03", "2026-06-30"),
    }

    scaler = None
    preprocessing = "raw"
    if family in LINEAR_FAMILIES:
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X_train = np.asarray(scaler.fit_transform(X_train), dtype=np.float64)
        preprocessing = "standardized"

    estimator = _build_estimator(family, params)
    fit_y = binary_target(y_train) if family == "logistic" else y_train
    estimator.fit(X_train, fit_y)

    model = FittedModel(
        family=family,
        hyperparameters=dict(params),
        feature_names=feature_names,
        preprocessing=preprocessing,
        seed=SEED,
        estimator=estimator,
        fitted_train_rows=int(len(y_train)),
        train_window=tuple(windows["train"]),
        val_window=tuple(windows["val"]),
        test_window=tuple(windows["test"]),
    )
    state = {
        "scaler": scaler,
        "train_rows": int(len(y_train)),
        "feature_names": feature_names,
        "preprocessing": preprocessing,
    }
    return model, state


def predict_with_state(model: FittedModel, state: dict[str, Any], X: np.ndarray) -> np.ndarray:
    """Deterministic prediction through the stored pipeline state.

    Returns the score used for ranking: for the logistic family this is the
    probability of a positive forward return (predict_proba[:, 1]); for the
    regression families it is the predicted forward return itself.
    """
    X = np.asarray(X, dtype=np.float64)
    if state["scaler"] is not None:
        X = np.asarray(state["scaler"].transform(X), dtype=np.float64)
    if model.family == "logistic":
        return np.asarray(model.estimator.predict_proba(X)[:, 1], dtype=np.float64)
    return model.predict(X)


def binary_target(outcome_values: np.ndarray) -> np.ndarray:
    """Deterministic logistic target: positive forward return."""
    return (np.asarray(outcome_values) > 0.0).astype(np.int64)


__all__ = [
    "FittedModel",
    "FEATURE_NAMES_DEFAULT",
    "LINEAR_FAMILIES",
    "train_model",
    "predict_with_state",
    "binary_target",
]