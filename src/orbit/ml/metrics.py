"""Phase 9 evaluation metrics.

All statistics are computed per decision session (cross-section) and
aggregated as the mean over sessions, matching the cross-sectional
character of the benchmark:

  - OOS IC:      per-session Pearson correlation(prediction, realized label);
                 reported as mean over sessions. Sessions with < min_obs
                 pairs or zero variance on either side are skipped and
                 counted (documented; they cannot contribute a meaningful
                 correlation).
  - rank IC:     per-session Spearman correlation via average-method
                 rankdata (deterministic), aggregated the same way.
  - calibration: ECE + Brier on the calibrated probabilities.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
from scipy.stats import rankdata


def _per_session_correlations(
    frame: pl.DataFrame,
    x_col: str,
    y_col: str,
    session_col: str = "decision_session",
    min_obs: int = 3,
    spearman: bool = False,
) -> dict[str, Any]:
    frame = frame.drop_nulls(subset=[x_col, y_col])
    cors: list[float] = []
    skipped_short = 0
    skipped_variance = 0
    sessions = 0
    for session in frame[session_col].unique().sort().to_list():
        cross = frame.filter(pl.col(session_col) == session)
        if cross.height < min_obs:
            skipped_short += 1
            continue
        x = cross[x_col].to_numpy()
        y = cross[y_col].to_numpy()
        if spearman:
            x = rankdata(x, method="average")
            y = rankdata(y, method="average")
        if np.std(x) == 0.0 or np.std(y) == 0.0:
            skipped_variance += 1
            continue
        corr = float(np.corrcoef(x, y)[0, 1])
        if np.isnan(corr):
            skipped_variance += 1
            continue
        cors.append(corr)
        sessions += 1
    return {
        "value": float(np.mean(cors)) if cors else float("nan"),
        "sessions_used": sessions,
        "sessions_skipped_short": skipped_short,
        "sessions_skipped_variance": skipped_variance,
        "total_sessions": sessions + skipped_short + skipped_variance,
    }


def oos_ic(
    frame: pl.DataFrame,
    pred_col: str,
    label_col: str = "outcome_value",
    session_col: str = "decision_session",
    min_obs: int = 3,
) -> dict[str, Any]:
    """Out-of-sample Information Coefficient (per-session Pearson, mean)."""
    return _per_session_correlations(frame, pred_col, label_col, session_col, min_obs, spearman=False)


def rank_ic(
    frame: pl.DataFrame,
    pred_col: str,
    label_col: str = "outcome_value",
    session_col: str = "decision_session",
    min_obs: int = 3,
) -> dict[str, Any]:
    """Cross-sectional rank IC (per-session Spearman, mean)."""
    return _per_session_correlations(frame, pred_col, label_col, session_col, min_obs, spearman=True)


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))


def hit_rate(frame: pl.DataFrame, pred_col: str, label_col: str = "outcome_value") -> float:
    """Fraction of rows where predicted score sign agrees with the realized
    sign of the forward return (deterministic binary agreement rate)."""
    f = frame.drop_nulls(subset=[pred_col, label_col])
    if f.height == 0:
        return float("nan")
    agreement = (
        (f[pred_col] > 0.0) == (f[label_col] > 0.0)
    ).sum()
    return float(agreement) / float(f.height)


__all__ = ["oos_ic", "rank_ic", "mean_squared_error", "hit_rate"]