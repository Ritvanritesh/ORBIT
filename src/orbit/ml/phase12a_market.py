"""Phase 12A Family A: Market Regime Context Features (vectorized)."""
from __future__ import annotations
from typing import Any
import polars as pl


def compute_market_features(
    benchmark_bars: pl.DataFrame,
    universe_sessions: pl.DataFrame,
) -> pl.DataFrame:
    """Compute market regime features using vectorized rolling operations."""
    bm = benchmark_bars.sort("trade_date").select([
        pl.col("trade_date").alias("session"),
        pl.col("close").alias("bm_close"),
    ])
    bm = bm.with_columns([
        pl.col("bm_close").pct_change().alias("bm_ret"),
    ])

    # Rolling computations on the benchmark series
    bm = bm.with_columns([
        # FEAT-201: mkt_ret_5
        (pl.col("bm_close") / pl.col("bm_close").shift(5) - 1).alias("mkt_ret_5"),
        # FEAT-202: mkt_ret_20
        (pl.col("bm_close") / pl.col("bm_close").shift(20) - 1).alias("mkt_ret_20"),
        # FEAT-203: mkt_vol_20
        pl.col("bm_ret").rolling_std(window_size=20).alias("mkt_vol_20"),
        # FEAT-204: mkt_vol_60
        pl.col("bm_ret").rolling_std(window_size=60).alias("mkt_vol_60"),
        # FEAT-205: mkt_trend_20_50
        (pl.col("bm_close").rolling_mean(window_size=20)
         / pl.col("bm_close").rolling_mean(window_size=50) - 1).alias("mkt_trend_20_50"),
        # FEAT-206: mkt_drawdown_from_peak_60
        (pl.col("bm_close") / pl.col("bm_close").rolling_max(window_size=60) - 1).alias("mkt_drawdown_from_peak_60"),
    ])

    # For point-in-time: features at decision_session D use data up to D-1
    # Shift all features by 1 so that at session D we see D-1's values
    feature_cols = ["mkt_ret_5", "mkt_ret_20", "mkt_vol_20", "mkt_vol_60",
                    "mkt_trend_20_50", "mkt_drawdown_from_peak_60"]
    bm = bm.with_columns([pl.col(c).shift(1).alias(c) for c in feature_cols])

    # Cross-join with all instruments in the universe
    instruments = universe_sessions.select("instrument_id").unique()
    bm = bm.with_columns([pl.lit(1).alias("_jk")])
    instruments = instruments.with_columns([pl.lit(1).alias("_jk")])
    result = instruments.join(bm, on="_jk", how="left").drop("_jk")

    # Keep only sessions that are in the universe
    universe_session_set = set(
        universe_sessions["decision_session"].unique().to_list()
    )
    result = result.filter(pl.col("session").is_in(list(universe_session_set)))
    result = result.rename({"session": "decision_session"})

    return result.sort(["instrument_id", "decision_session"])
