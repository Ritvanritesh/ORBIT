"""Phase 9 controls: the Phase 8 baseline strategies on the real dataset.

Phase 8's own runner executed its documented strategies on SYNTHETIC bars;
Phase 9 executes the same documented rules on the real DS-000004 bars
through the identical canonical Phase 7 path and cost model (CM-001), so the
ML models and the controls are compared under exactly the same conditions.

Two documented adaptations (both make the comparison strictly conservative):

  1. Strict point-in-time boundary: every control metric uses bars strictly
     before the decision instant (the same boundary as the FS-001 features),
     so controls never see information the ML models do not see.
  2. WEIGHT sizing: controls emit weight targets (top-1 -> 1.0, equal-weight
     -> 1/n) exactly like the ML top-k signals; the Phase 8 runner used
     nominal share quantities, which would not be comparable.

Control list (all rules as documented in src/orbit/baselines/strategies.py):
  buy_and_hold        one long on the first session (first instrument)
  equal_weight        all names, weight 1/n, every session
  momentum            lookbacks {10, 20, 30}: long the top-1 by point-in-time
                      lookback return, flat the rest
  mean_reversion      lookbacks {10, 20, 30}: long the single most-extreme
                      name below its rolling mean (most negative deviation),
                      flat otherwise
  moving_average      combos (5,30), (10,30), (15,40): long the single name
                      with the largest positive SMA gap, flat otherwise
  volatility_targeted combos (0.10,10), (0.15,30), (0.20,60): equal weight
                      scaled by min(1, target_vol / est_vol)
  random_null         seeded random longs (seed 42) and the all-flat null
"""

from __future__ import annotations

from datetime import date
from typing import Any, Callable

import polars as pl

from orbit.ml.features import close_utc

CONTROL_GRIDS: dict[str, list[dict[str, Any]]] = {
    "momentum": [{"lookback": lb} for lb in (10, 20, 30)],
    "mean_reversion": [{"lookback": lb} for lb in (10, 20, 30)],
    "moving_average": [
        {"short_window": s, "long_window": l}
        for s, l in ((5, 30), (10, 30), (15, 40))
    ],
    "volatility_targeted": [
        {"target_volatility": tv, "estimation_window": w}
        for tv, w in ((0.10, 10), (0.15, 30), (0.20, 60))
    ],
}

CONTROL_FAMILIES = ("buy_and_hold", "equal_weight", "momentum", "mean_reversion", "moving_average", "volatility_targeted", "random_null", "null_flat")


def control_metrics(bars: pl.DataFrame) -> pl.DataFrame:
    """Per (instrument, session) point-in-time metrics for the controls.

    All metrics use the strict boundary (bars strictly before the decision
    session's close, i.e. sessions <= D-1), identical to FS-001. Columns:
    instrument_id, decision_session, plus ret_{10,20,30}, dev_{10,20,30},
    gap_{5,10,15}_{30,30,40}, vol_{10,30,60}.
    """
    parts = []
    for _, g in bars.group_by("instrument_id"):
        b = g.sort("trade_date")
        b = b.with_columns(
            (pl.col("close") / pl.col("close").shift(1) - 1.0).alias("ret_1")
        )
        feats = {
            f"ret_{lb}": (pl.col("close").shift(1) / pl.col("close").shift(lb) - 1.0)
            for lb in (10, 20, 30)
        }
        feats.update(
            {
                f"dev_{lb}": (
                    pl.col("close").shift(1)
                    / pl.col("close").rolling_mean(window_size=lb).shift(1)
                    - 1.0
                )
                for lb in (10, 20, 30)
            }
        )
        feats.update(
            {
                f"gap_{s}_{l}": (
                    pl.col("close").rolling_mean(window_size=s).shift(1)
                    / pl.col("close").rolling_mean(window_size=l).shift(1)
                    - 1.0
                )
                for s, l in ((5, 30), (10, 30), (15, 40))
            }
        )
        feats.update(
            {
                f"vol_{w}": pl.col("ret_1").rolling_std(window_size=w).shift(1)
                for w in (10, 30, 60)
            }
        )
        out = b.select("instrument_id", "trade_date", **feats)
        parts.append(out)
    frame = pl.concat(parts) if parts else pl.DataFrame(
        schema={
            "instrument_id": pl.Utf8,
            "trade_date": pl.Date,
            **{n: pl.Float64 for n in (
                "ret_10", "ret_20", "ret_30",
                "dev_10", "dev_20", "dev_30",
                "gap_5_30", "gap_10_30", "gap_15_40",
                "vol_10", "vol_30", "vol_60",
            )}
        }
    )
    frame = frame.rename({"trade_date": "decision_session"})
    return frame.drop_nulls()


def _signal_rows(
    frame: pl.DataFrame,
    *,
    selected: pl.DataFrame,
    target_col: str,
    strategy_ref: str,
) -> pl.DataFrame:
    """Merge a per-session selection into complete signal rows (long/flat)."""
    out = frame.sort(["decision_session", "instrument_id"]).with_columns(
        pl.lit("flat", dtype=pl.Utf8).alias("direction"),
        pl.lit(0.0, dtype=pl.Float64).alias("target"),
        pl.lit(None, dtype=pl.Float64).alias("signal_metric"),
    )
    picked = selected.with_columns(
        pl.lit("long", dtype=pl.Utf8).alias("direction"),
        pl.col(target_col).alias("target"),
    )
    pick_cols = ["instrument_id", "decision_session", "direction", "target"]
    if "signal_metric" in picked.columns:
        pick_cols.append("signal_metric")
    if picked.height == 0:
        picked = pl.DataFrame(
            schema={
                "instrument_id": pl.Utf8,
                "decision_session": pl.Date,
                "direction": pl.Utf8,
                "target": pl.Float64,
                "signal_metric": pl.Float64,
            }
        ).slice(0, 0)
    picked = picked.select(pick_cols)
    out = out.join(picked, on=["instrument_id", "decision_session"], how="left")
    out = out.with_columns(
        pl.coalesce(pl.col("direction_right"), pl.col("direction")).alias("direction"),
        pl.coalesce(pl.col("target_right"), pl.col("target")).alias("target"),
    ).drop(["direction_right", "target_right"])
    if "signal_metric_right" in out.columns:
        out = out.with_columns(
            pl.coalesce(pl.col("signal_metric_right"), pl.col("signal_metric")).alias("signal_metric")
        ).drop("signal_metric_right")
    out = out.with_columns(
        pl.col("decision_session").map_elements(close_utc, return_dtype=pl.Datetime("us", None)).alias("decision_time"),
        pl.lit(strategy_ref, dtype=pl.Utf8).alias("strategy_ref"),
    )
    out = out.with_columns(
        (
            "SIG-CTL-"
            + pl.col("instrument_id")
            + "-"
            + pl.col("decision_session").dt.strftime("%Y%m%d")
        ).alias("signal_id")
    )
    return out.select(
        "signal_id",
        "instrument_id",
        pl.col("decision_session").alias("signal_session"),
        "decision_time",
        "direction",
        "target",
        "signal_metric",
        "strategy_ref",
    )


def buy_and_hold_signals(
    bars: pl.DataFrame, sessions: list[date], strategy_ref: str = "buy_and_hold"
) -> pl.DataFrame:
    """One long (weight 0.99) on the first session of the first instrument.

    The 0.99 target is a documented adaptation: at the canonical no-implicit-
    loan rule a weight-1.0 market-on-open buy can be rejected when the open
    gaps up beyond the CM-001 cost buffer (spread 2 bps + slippage 2 bps +
    gap), leaving the control degenerate (zero fills). The 1% residual
    absorbs the entry costs so the hold-to-end strategy actually executes;
    the residual stays in cash for the full window.
    """
    first_instrument = sorted(bars["instrument_id"].unique().to_list())[0]
    first_session = sessions[0]
    rows = [
        {
            "signal_id": f"SIG-CTL-BH-{first_instrument[-6:]}-{first_session.isoformat()}",
            "instrument_id": first_instrument,
            "signal_session": first_session,
            "decision_time": close_utc(first_session),
            "direction": "long",
            "target": 0.99,
            "signal_metric": None,
            "strategy_ref": strategy_ref,
        }
    ]
    return pl.DataFrame(rows)


def equal_weight_signals(
    bars: pl.DataFrame,
    sessions: list[date],
    strategy_ref: str = "equal_weight",
) -> pl.DataFrame:
    """All names, weight 1/n, every session."""
    instruments = sorted(bars["instrument_id"].unique().to_list())
    n = len(instruments)
    rows = []
    for session in sessions:
        for inst in instruments:
            rows.append(
                {
                    "signal_id": f"SIG-CTL-EW-{inst[-6:]}-{session.isoformat()}",
                    "instrument_id": inst,
                    "signal_session": session,
                    "decision_time": close_utc(session),
                    "direction": "long",
                    "target": 1.0 / n,
                    "signal_metric": None,
                    "strategy_ref": strategy_ref,
                }
            )
    return pl.DataFrame(rows)


def momentum_signals(
    bars: pl.DataFrame, sessions: list[date], lookback: int, strategy_ref: str | None = None
) -> pl.DataFrame:
    """Long the top-1 by point-in-time lookback return, flat the rest."""
    metrics = control_metrics(bars).filter(pl.col("decision_session").is_in(sessions))
    ref = strategy_ref or f"momentum_{lookback}"
    selected: list[pl.DataFrame] = []
    for session in sessions:
        cross = metrics.filter(pl.col("decision_session") == session).drop_nulls(subset=[f"ret_{lookback}"])
        if cross.height == 0:
            continue
        top = cross.sort(f"ret_{lookback}", descending=True).head(1)
        selected.append(
            top.with_columns(
                pl.lit(1.0, dtype=pl.Float64).alias("target"),
                pl.col(f"ret_{lookback}").alias("signal_metric"),
            )
        )
    sel = pl.concat(selected) if selected else metrics.slice(0, 0)
    return _signal_rows(metrics, selected=sel, target_col="target", strategy_ref=ref)


def mean_reversion_signals(
    bars: pl.DataFrame, sessions: list[date], lookback: int, strategy_ref: str | None = None
) -> pl.DataFrame:
    """Long the single most-extreme name below its rolling mean, else flat."""
    metrics = control_metrics(bars).filter(pl.col("decision_session").is_in(sessions))
    ref = strategy_ref or f"mean_reversion_{lookback}"
    selected: list[pl.DataFrame] = []
    for session in sessions:
        cross = metrics.filter(pl.col("decision_session") == session).drop_nulls(subset=[f"dev_{lookback}"])
        if cross.height == 0:
            continue
        bottom = cross.sort(f"dev_{lookback}").head(1)
        if float(bottom[f"dev_{lookback}"][0]) >= 0.0:
            continue  # nothing is below its mean: flat
        selected.append(
            bottom.with_columns(
                pl.lit(1.0, dtype=pl.Float64).alias("target"),
                pl.col(f"dev_{lookback}").alias("signal_metric"),
            )
        )
    sel = pl.concat(selected) if selected else metrics.slice(0, 0)
    return _signal_rows(metrics, selected=sel, target_col="target", strategy_ref=ref)


def moving_average_signals(
    bars: pl.DataFrame,
    sessions: list[date],
    short_window: int,
    long_window: int,
    strategy_ref: str | None = None,
) -> pl.DataFrame:
    """Long the single name with the largest positive SMA gap, else flat."""
    metrics = control_metrics(bars).filter(pl.col("decision_session").is_in(sessions))
    gap_col = f"gap_{short_window}_{long_window}"
    ref = strategy_ref or f"moving_average_{short_window}_{long_window}"
    selected: list[pl.DataFrame] = []
    for session in sessions:
        cross = metrics.filter(pl.col("decision_session") == session).drop_nulls(subset=[gap_col])
        if cross.height == 0:
            continue
        best = cross.sort(gap_col, descending=True).head(1)
        if float(best[gap_col][0]) <= 0.0:
            continue  # no positive crossover: flat
        selected.append(
            best.with_columns(
                pl.lit(1.0, dtype=pl.Float64).alias("target"),
                pl.col(gap_col).alias("signal_metric"),
            )
        )
    sel = pl.concat(selected) if selected else metrics.slice(0, 0)
    return _signal_rows(metrics, selected=sel, target_col="target", strategy_ref=ref)


def volatility_targeted_signals(
    bars: pl.DataFrame,
    sessions: list[date],
    target_volatility: float,
    estimation_window: int,
    strategy_ref: str | None = None,
) -> pl.DataFrame:
    """Equal-weight exposure scaled by min(1, target_vol / est_vol)."""
    metrics = control_metrics(bars).filter(pl.col("decision_session").is_in(sessions))
    n = metrics["instrument_id"].n_unique()
    ref = strategy_ref or f"volatility_targeted_{target_volatility}_{estimation_window}"
    vol_col = f"vol_{estimation_window}"
    weighted = metrics.drop_nulls(subset=[vol_col]).with_columns(
        ((1.0 / n) * (target_volatility / pl.col(vol_col)).clip(0.0, 1.0)).alias("target")
    )
    return _signal_rows(metrics, selected=weighted, target_col="target", strategy_ref=ref)


def random_null_signals(
    bars: pl.DataFrame,
    sessions: list[date],
    seed: int = 42,
    strategy_ref: str = "random_null",
) -> pl.DataFrame:
    """Seeded random longs (70%) / flat (30%) with fractional targets."""
    import random

    rng = random.Random(seed)
    instruments = sorted(bars["instrument_id"].unique().to_list())
    rows = []
    for inst in instruments:
        for session in sessions:
            r = rng.random()
            direction = "long" if r < 0.7 else "flat"
            target = round(rng.uniform(0.05, 0.5), 4) if direction == "long" else 0.0
            rows.append(
                {
                    "signal_id": f"SIG-CTL-RAND-{inst[-6:]}-{session.isoformat()}",
                    "instrument_id": inst,
                    "signal_session": session,
                    "decision_time": close_utc(session),
                    "direction": direction,
                    "target": target,
                    "signal_metric": float(rng.random()),
                    "strategy_ref": strategy_ref,
                }
            )
    return pl.DataFrame(rows)


def null_flat_signals(
    bars: pl.DataFrame,
    sessions: list[date],
    strategy_ref: str = "null_flat",
) -> pl.DataFrame:
    """Null control: all-flat signals (no trading activity)."""
    instruments = sorted(bars["instrument_id"].unique().to_list())
    rows = []
    for inst in instruments:
        for session in sessions:
            rows.append(
                {
                    "signal_id": f"SIG-CTL-NULL-{inst[-6:]}-{session.isoformat()}",
                    "instrument_id": inst,
                    "signal_session": session,
                    "decision_time": close_utc(session),
                    "direction": "flat",
                    "target": 0.0,
                    "signal_metric": None,
                    "strategy_ref": strategy_ref,
                }
            )
    return pl.DataFrame(rows)


CONTROL_BUILDERS: dict[str, Callable[..., pl.DataFrame]] = {
    "buy_and_hold": buy_and_hold_signals,
    "equal_weight": equal_weight_signals,
    "momentum": momentum_signals,
    "mean_reversion": mean_reversion_signals,
    "moving_average": moving_average_signals,
    "volatility_targeted": volatility_targeted_signals,
    "random_null": random_null_signals,
    "null_flat": null_flat_signals,
}


def build_control_signals(
    bars: pl.DataFrame, sessions: list[date], family: str, params: dict[str, Any]
) -> pl.DataFrame:
    """Build one control strategy's signals with its pre-registered params."""
    builder = CONTROL_BUILDERS.get(family)
    if builder is None:
        raise ValueError(f"unknown control family {family!r}")
    if family in ("buy_and_hold", "equal_weight", "random_null", "null_flat"):
        if params:
            raise ValueError(f"control {family!r} takes no parameters")
        return builder(bars, sessions)
    return builder(bars, sessions, **params)


__all__ = [
    "CONTROL_GRIDS",
    "CONTROL_FAMILIES",
    "CONTROL_BUILDERS",
    "control_metrics",
    "build_control_signals",
]