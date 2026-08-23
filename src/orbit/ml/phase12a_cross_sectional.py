"""Phase 12A Family C: Cross-Sectional Features.

Computes cross-sectional features from the universe state at each session.

All features satisfy point-in-time: the universe snapshot at session D
uses only instruments and features available at or before D.

Requirements:
- minimum universe population enforced
- ties handled via average ranking
- missing instruments dropped per session
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from orbit.ml.features import attach_decision_times


def compute_cross_sectional_features(
    base_features: pl.DataFrame,
    universe_sessions: pl.DataFrame,
    feature_names: list[str],
    min_population: int = 5,
) -> pl.DataFrame:
    """Compute cross-sectional features from base instrument features.

    Parameters
    ----------
    base_features : pl.DataFrame
        Instrument features with columns: instrument_id, decision_session,
        window_end_session, plus feature_names columns.
    universe_sessions : pl.DataFrame
        DataFrame with columns: instrument_id, decision_session (Date).
    feature_names : list[str]
        Base feature names to derive cross-sectional features from.
        Must include at least 'ret_20' and 'vol_20'.
    min_population : int
        Minimum instruments per session for valid cross-sectional features.

    Returns
    -------
    pl.DataFrame
        Cross-sectional features per instrument per session.
    """
    required = ["ret_20", "vol_10"]
    for f in required:
        if f not in feature_names:
            raise ValueError(f"required base feature {f!r} not in feature_names")

    all_sessions = (
        universe_sessions.select("decision_session")
        .unique()
        .sort("decision_session")
    )

    results = []

    for row in all_sessions.iter_rows(named=True):
        d = row["decision_session"]

        # Get instruments in universe at this session
        session_instruments = (
            universe_sessions.filter(pl.col("decision_session") == d)
            .select("instrument_id")
            .unique()
        )

        # Get base features for this session
        session_features = base_features.filter(
            pl.col("decision_session") == d
        ).select(
            ["instrument_id", "window_end_session"] + [f for f in feature_names if f in base_features.columns]
        )

        # Inner join to get only universe instruments
        cross = session_instruments.join(session_features, on="instrument_id", how="inner")

        if cross.height < min_population:
            continue

        last_session = cross["window_end_session"].max()

        # FEAT-221: xs_rank_ret_20 (percentile rank of ret_20)
        if "ret_20" in cross.columns:
            ret_vals = cross["ret_20"].to_numpy()
            valid_mask = ~np.isnan(ret_vals)
            if valid_mask.sum() >= min_population:
                ranks = np.full_like(ret_vals, np.nan)
                valid_vals = ret_vals[valid_mask]
                # Average ranking
                order = valid_vals.argsort()
                r = np.empty_like(order, dtype=float)
                r[order] = np.arange(1, len(valid_vals) + 1, dtype=float)
                # Handle ties by averaging
                sorted_vals = valid_vals[order]
                i = 0
                while i < len(sorted_vals):
                    j = i
                    while j < len(sorted_vals) - 1 and sorted_vals[j + 1] == sorted_vals[i]:
                        j += 1
                    avg_rank = np.mean(np.arange(i + 1, j + 2, dtype=float))
                    for k in range(i, j + 1):
                        orig_idx = order[k]
                        r[orig_idx] = avg_rank
                    i = j + 1
                ranks[valid_mask] = r / (valid_mask.sum() + 1)  # percentile [0, 1)
                cross = cross.with_columns([
                    pl.Series("xs_rank_ret_20", ranks),
                ])
            else:
                cross = cross.with_columns([
                    pl.lit(None, dtype=pl.Float64).alias("xs_rank_ret_20"),
                ])
        else:
            cross = cross.with_columns([
                pl.lit(None, dtype=pl.Float64).alias("xs_rank_ret_20"),
            ])

        # FEAT-222: xs_rank_vol_10 (percentile rank of vol_10)
        if "vol_10" in cross.columns:
            vol_vals = cross["vol_10"].to_numpy()
            valid_mask = ~np.isnan(vol_vals)
            if valid_mask.sum() >= min_population:
                ranks = np.full_like(vol_vals, np.nan)
                valid_vals = vol_vals[valid_mask]
                order = valid_vals.argsort()
                r = np.empty_like(order, dtype=float)
                r[order] = np.arange(1, len(valid_vals) + 1, dtype=float)
                sorted_vals = valid_vals[order]
                i = 0
                while i < len(sorted_vals):
                    j = i
                    while j < len(sorted_vals) - 1 and sorted_vals[j + 1] == sorted_vals[i]:
                        j += 1
                    avg_rank = np.mean(np.arange(i + 1, j + 2, dtype=float))
                    for k in range(i, j + 1):
                        orig_idx = order[k]
                        r[orig_idx] = avg_rank
                    i = j + 1
                ranks[valid_mask] = r / (valid_mask.sum() + 1)
                cross = cross.with_columns([
                    pl.Series("xs_rank_vol_10", ranks),
                ])
            else:
                cross = cross.with_columns([
                    pl.lit(None, dtype=pl.Float64).alias("xs_rank_vol_10"),
                ])
        else:
            cross = cross.with_columns([
                pl.lit(None, dtype=pl.Float64).alias("xs_rank_vol_10"),
            ])

        # FEAT-223: xs_ret_vs_median_20
        if "ret_20" in cross.columns:
            median_ret = cross["ret_20"].median()
            cross = cross.with_columns([
                (pl.col("ret_20") - median_ret).alias("xs_ret_vs_median_20"),
            ])
        else:
            cross = cross.with_columns([
                pl.lit(None, dtype=pl.Float64).alias("xs_ret_vs_median_20"),
            ])

        # FEAT-224: xs_ret_vs_mean_20
        if "ret_20" in cross.columns:
            mean_ret = cross["ret_20"].mean()
            cross = cross.with_columns([
                (pl.col("ret_20") - mean_ret).alias("xs_ret_vs_mean_20"),
            ])
        else:
            cross = cross.with_columns([
                pl.lit(None, dtype=pl.Float64).alias("xs_ret_vs_mean_20"),
            ])

        # FEAT-225: xs_dispersion_ret_20
        if "ret_20" in cross.columns:
            std_ret = cross["ret_20"].std()
            cross = cross.with_columns([
                pl.lit(std_ret).alias("xs_dispersion_ret_20"),
            ])
        else:
            cross = cross.with_columns([
                pl.lit(None, dtype=pl.Float64).alias("xs_dispersion_ret_20"),
            ])

        # Collect results
        out = cross.select([
            "instrument_id",
            pl.lit(d).alias("decision_session"),
            pl.lit(last_session).alias("window_end_session"),
            "xs_rank_ret_20",
            "xs_rank_vol_10",
            "xs_ret_vs_median_20",
            "xs_ret_vs_mean_20",
            "xs_dispersion_ret_20",
        ])
        results.append(out)

    if not results:
        return pl.DataFrame()

    return pl.concat(results).sort(["instrument_id", "decision_session"])
