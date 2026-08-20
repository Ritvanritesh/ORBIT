"""Phase 9 reproducibility tests: repeated runs produce identical artifacts."""

from __future__ import annotations

import numpy as np
import polars as pl

from orbit.ml.models import predict_with_state, train_model
from orbit.ml.ranking import cross_sectional_rank, top_k_long
from orbit.ml.signals import predictions_to_signals
from tests.phase9_testutils import TEST_WINDOWS, build_test_datasets, make_canonical_bars, make_events


def test_full_pipeline_is_bitwise_reproducible():
    bars = make_canonical_bars()
    events = make_events()

    def run_once():
        _, _, ds = build_test_datasets(bars, events)
        (Xtr, ytr, _, _) = ds["train"]
        Xte, yte, _, meta_te = ds["test"]
        model, state = train_model("xgboost", {"n_estimators": 50, "max_depth": 3, "learning_rate": 0.1}, Xtr, ytr)
        pred = predict_with_state(model, state, Xte)
        frame = meta_te.with_columns(pl.Series("prediction", pred))
        sig = predictions_to_signals(frame, family="xgboost", params={"n_estimators": 50, "max_depth": 3, "learning_rate": 0.1}, top_k=3)
        return pred, sig

    p1, s1 = run_once()
    p2, s2 = run_once()
    assert np.array_equal(p1, p2)
    assert s1.equals(s2)


def test_ranking_is_reproducible_across_runs():
    from datetime import date

    days = [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4), date(2024, 1, 5)]
    frame = pl.DataFrame(
        {
            "decision_session": days * 4,
            "instrument_id": ["A", "B", "C", "D"] * 4,
            "score": [0.4, 0.1, 0.9, 0.2, 0.3, 0.8, 0.5, 0.6, 0.7, 0.2, 0.1, 0.9, 0.5, 0.5, 0.5, 0.5],
        }
    )
    a = top_k_long(cross_sectional_rank(frame, "score"), k=2)
    b = top_k_long(cross_sectional_rank(frame, "score"), k=2)
    assert a.equals(b)


def test_model_train_repeated_runs_match_for_every_family():
    bars = make_canonical_bars()
    _, _, ds = build_test_datasets(bars, make_events())
    (Xtr, ytr, _, _) = ds["train"]
    Xte = ds["test"][0]
    for family, params in (
        ("ridge", {"alpha": 1.0}),
        ("lasso", {"alpha": 0.01}),
        ("logistic", {"C": 1.0}),
        ("random_forest", {"n_estimators": 50, "max_depth": 3}),
        ("xgboost", {"n_estimators": 50, "max_depth": 3, "learning_rate": 0.1}),
    ):
        m1, s1 = train_model(family, params, Xtr, ytr)
        m2, s2 = train_model(family, params, Xtr, ytr)
        assert np.array_equal(predict_with_state(m1, s1, Xte), predict_with_state(m2, s2, Xte)), family