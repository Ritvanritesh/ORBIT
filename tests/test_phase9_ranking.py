"""Phase 9 ranking tests: cross-sectional, deterministic, tie-safe."""

from __future__ import annotations

import polars as pl
import pytest
from datetime import date

from orbit.ml.ranking import cross_sectional_rank, top_k_long


def _frame():
    return pl.DataFrame(
        {
            "decision_session": [date(2024, 1, 2), date(2024, 1, 2), date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 3), date(2024, 1, 3)],
            "instrument_id": ["A", "B", "C", "A", "B", "C"],
            "score": [0.1, 0.5, 0.3, None, -0.2, 0.4],
        }
    )


def test_rank_within_session_only():
    ranked = cross_sectional_rank(_frame(), "score")
    ranks = (
        ranked.filter(pl.col("decision_session") == date(2024, 1, 2))
        .sort("instrument_id")["rank"]
        .to_list()
    )
    assert ranks == [3.0, 1.0, 2.0]  # A=0.1 -> 3, B=0.5 -> 1, C=0.3 -> 2


def test_short_cross_section_with_null_dropped_entirely():
    """2024-01-03 has only 2 valid scores after the null is dropped, below
    min_obs=3, so the whole degenerate cross-section disappears."""
    ranked = cross_sectional_rank(_frame(), "score")
    assert ranked.filter(pl.col("decision_session") == date(2024, 1, 3)).height == 0
    assert ranked.filter(pl.col("decision_session") == date(2024, 1, 2)).height == 3


def test_ties_get_average_rank():
    f = pl.DataFrame(
        {
            "decision_session": [date(2024, 1, 2)] * 4,
            "instrument_id": ["A", "B", "C", "D"],
            "score": [0.2, 0.2, 0.5, 0.1],
        }
    )
    ranked = cross_sectional_rank(f, "score")
    ranks = ranked.sort("instrument_id")["rank"].to_list()
    assert ranks == [2.5, 2.5, 1.0, 4.0]


def test_rank_is_deterministic():
    a = cross_sectional_rank(_frame(), "score")
    b = cross_sectional_rank(_frame(), "score")
    assert a.equals(b)


def test_short_cross_sections_dropped():
    f = pl.DataFrame(
        {
            "decision_session": [date(2024, 1, 2)] * 2,
            "instrument_id": ["A", "B"],
            "score": [0.1, 0.2],
        }
    )
    ranked = cross_sectional_rank(f, "score", min_obs=3)
    assert ranked.height == 0


def test_top_k_selects_highest_scored():
    ranked = cross_sectional_rank(_frame(), "score")
    top = top_k_long(ranked, k=2)
    top1 = top.filter(pl.col("decision_session") == date(2024, 1, 2)).sort("instrument_id")
    assert top1["instrument_id"].to_list() == ["B", "C"]
    assert (top1["target_weight"] == 0.5).all()


def test_top_k_weight_defaults_to_one_over_k():
    ranked = cross_sectional_rank(_frame(), "score")
    top = top_k_long(ranked, k=3)
    assert all(w == pytest.approx(1.0 / 3.0) for w in top["target_weight"].to_list())


def test_top_k_rejects_k_zero():
    with pytest.raises(ValueError):
        top_k_long(cross_sectional_rank(_frame(), "score"), k=0)


def test_no_cross_session_ranking():
    f = pl.DataFrame(
        {
            "decision_session": [date(2024, 1, 2)] * 4 + [date(2024, 1, 3)] * 4,
            "instrument_id": ["A", "B", "C", "D"] * 2,
            "score": [0.4, 0.1, 0.9, 0.2, 0.3, 0.8, 0.5, 0.6],
        }
    )
    ranked = cross_sectional_rank(f, "score")
    top = top_k_long(ranked, k=1)
    sessions = top["decision_session"].unique().to_list()
    assert len(sessions) == 2  # one winner per session, never merged
    assert top.filter(pl.col("decision_session") == date(2024, 1, 2))["instrument_id"][0] == "C"
    assert top.filter(pl.col("decision_session") == date(2024, 1, 3))["instrument_id"][0] == "B"