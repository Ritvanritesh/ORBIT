"""Cross-sectional ranking (deterministic).

Ranking is per decision session (cross-section): scores are ranked only
against instruments of the SAME session - the cross-sectional character of
the H-001 momentum hypothesis. Ties use the average method (scipy.stats.
rankdata, deterministic); rows with null scores are excluded from the
cross-section. A cross-section with fewer than min_obs valid scores is
dropped entirely and counted, so per-date statistics never see degenerate
dates.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from scipy.stats import rankdata


def cross_sectional_rank(
    frame: pl.DataFrame,
    score_col: str,
    session_col: str = "decision_session",
    min_obs: int = 3,
) -> pl.DataFrame:
    """Add a `rank` column: rank 1 = the highest score of the session.

    Ranks are computed per decision session (cross-section) with the
    average-tie method over the NEGATED score, so the best score always
    receives rank 1 (scipy.stats.rankdata, deterministic). Rows with null
    scores are dropped; cross-sections with fewer than min_obs valid scores
    are dropped entirely and counted nowhere (documented).
    """
    out = frame.drop_nulls(subset=[score_col])
    ranked: list[pl.DataFrame] = []
    for session in out[session_col].unique().sort().to_list():
        cross = out.filter(pl.col(session_col) == session)
        if cross.height < min_obs:
            continue
        scores = cross[score_col].to_numpy()
        ranks = rankdata(-scores, method="average")
        cross = cross.with_columns(pl.Series("rank", ranks, dtype=pl.Float64))
        ranked.append(cross)
    if not ranked:
        return out.with_columns(pl.lit(None, dtype=pl.Float64).alias("rank")).slice(0, 0)
    result = pl.concat(ranked)
    return result.sort([session_col, score_col])


def top_k_long(
    frame: pl.DataFrame,
    k: int = 3,
    rank_col: str = "rank",
    session_col: str = "decision_session",
    weight: float | None = None,
) -> pl.DataFrame:
    """Select the k highest-ranked names per session.

    Weight defaults to 1/k (fully invested, equal-weight top-k). The result
    carries one row per selected name with a `target_weight` column; the
    rest of the cross-section is implicitly flat (zero weight).
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    target_weight = 1.0 / k if weight is None else float(weight)
    ranked = frame.filter(pl.col(rank_col).is_not_null())
    selected: list[pl.DataFrame] = []
    for session in ranked[session_col].unique().sort().to_list():
        cross = ranked.filter(pl.col(session_col) == session).sort(rank_col)
        picked = cross.head(k).with_columns(
            pl.lit(target_weight, dtype=pl.Float64).alias("target_weight")
        )
        selected.append(picked)
    if not selected:
        return frame.slice(0, 0)
    return pl.concat(selected)


__all__ = ["cross_sectional_rank", "top_k_long"]