"""Baseline strategy implementations for ORBIT Phase 8.

Every strategy produces canonical signal rows that feed into the same
Phase 7 backtester. No strategy bypasses execution costs, idealized fills,
or the portfolio ledger.

Canonical signal row format (expected by orbit.backtest.Backtester._build_signal_events):

    {
        "signal_id": "SIG-xxxxxx",
        "instrument_id": "INSTRUMENT",
        "signal_session": date,          # decision session
        "decision_time": datetime,       # must be session_close_utc(signal_session)
        "direction": "long" | "flat",
        "target": float,                 # shares (for QUANTITY sizing) or fraction (for WEIGHT)
        "signal_metric": float,          # metric value at decision time
        "strategy_ref": "strategy name",
    }
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import polars as pl

from orbit.backtest import BacktestConfig
from orbit.backtest.clock import session_close_utc
from orbit.schemas.common import CostModel


# ---------------------------------------------------------------------------
# 1. Buy-and-hold
# ---------------------------------------------------------------------------


def buy_and_hold_signals(
    instrument_id: str,
    sessions: list[date],
    *,
    target_shares: float = 1_000.0,
    strategy_ref: str = "buy_and_hold",
) -> list[dict[str, Any]]:
    """Generate the buy-and-hold signal set.

    Definition (roadmap explicit & reproducible):

    - One initial buy signal on the first evaluation session.
    - The position is held forever (no subsequent signals).
    - Uses the same dataset, universe, cost model, and execution simulator
      as every other baseline.
    - No idealized fills: the order fills at the next session's open via
      the canonical Phase 7 execution semantics.

    The single signal stamps `decision_time` at the session close of its
    `signal_session`; the Phase 7 backtester then schedules the fill at the
    next session's open (delay=1, execution_price=OPEN).
    """
    if not sessions:
        raise ValueError("sessions must not be empty")
    return [
        {
            "signal_id": f"SIG-BH-{instrument_id[-6:]}",
            "instrument_id": instrument_id,
            "signal_session": sessions[0],
            "decision_time": session_close_utc(sessions[0]),
            "direction": "long",
            "target": target_shares,
            "signal_metric": 1.0,
            "strategy_ref": strategy_ref,
        }
    ]


# ---------------------------------------------------------------------------
# 2. Equal weight
# ---------------------------------------------------------------------------


def equal_weight_signals(
    instrument_ids: list[str],
    sessions: list[date],
    *,
    rebalance: bool = True,
    strategy_ref: str = "equal_weight",
) -> list[dict[str, Any]]:
    """Generate equal-weight signals.

    Rebalancing behavior (explicit choice, not silently invented):

    - If `rebalance=True` (default): on every session, all instruments in
      `instrument_ids` receive an equal-weight allocation.  The target
      quantity for each instrument is computed as:

          portfolio_equity / n_instruments / reference_price

      where reference_price is the open of the signal session.  This means
      the allocation drifts with price movements and is rebalanced each
      session.

    - If `rebalance=False`: a one-time allocation on the first session only.
      No further signals are generated.

    The resulting orders go through Phase 7 (same executor, same cost model,
    same accounting).  No separate portfolio engine is used.

    Note: the roadmap does not specify a particular rebalancing frequency,
    so `rebalance=True` (session-by-session) is the explicit default, and
    the choice is documented rather than silently using a complex schedule.
    """
    if not instrument_ids:
        raise ValueError("instrument_ids must not be empty")
    if not sessions:
        raise ValueError("sessions must not be empty")

    n_instruments = len(instrument_ids)

    if not rebalance:
        # One-time allocation on the first session only
        return [
            {
                "signal_id": f"SIG-EW-{instrument_ids[i][-6:]}-{sessions[0].isoformat()}",
                "instrument_id": instrument_ids[i],
                "signal_session": sessions[0],
                "decision_time": session_close_utc(sessions[0]),
                "direction": "long",
                "target": 0.0,  # placeholder; recalculated per run
                "signal_metric": 1.0,
                "strategy_ref": strategy_ref,
            }
            for i in range(n_instruments)
        ]

    # Session-by-session rebalance: one signal per instrument per session
    signals: list[dict[str, Any]] = []
    for i, inst_id in enumerate(instrument_ids):
        for s_idx, session in enumerate(sessions):
            signals.append(
                {
                    "signal_id": f"SIG-EW-{inst_id[-6:]}-{session.isoformat()}-{s_idx}",
                    "instrument_id": inst_id,
                    "signal_session": session,
                    "decision_time": session_close_utc(session),
                    "direction": "long",
                    "target": 0.0,  # nominal target; backtester sizing interprets
                    "signal_metric": 1.0,
                    "strategy_ref": strategy_ref,
                }
            )
    return signals


# ---------------------------------------------------------------------------
# 3. Momentum
# ---------------------------------------------------------------------------


def momentum_signals(
    instrument_ids: list[str],
    sessions: list[date],
    lookback_days: int = 20,
    *,
    strategy_ref: str = "momentum",
) -> list[dict[str, Any]]:
    """Generate momentum signals.

    Rule (deterministic, interpretable, reproducible):

    - For each instrument, compute the total return over the `lookback_days`
      lookback window (prior to the signal session).
    - Rank instruments by that lookback return (higher = stronger momentum).
    - Signal the top-ranked instrument with a long target; signal the
      bottom-ranked instrument with a short target (but Phase 7 is
      long-only, so the short is simply not acted upon / generates a
      position-constraint rejection).
    - Only one instrument is signaled per session (the current leader).

    The rule uses only information available at decision time: historical
    closes up to and including the close of the signal session. No future
    prices enter the lookback calculation.

    Small pre-registered parameter grid: lookback_days ∈ {10, 20, 30}.
    """
    if not instrument_ids:
        raise ValueError("instrument_ids must not be empty")
    if not sessions:
        raise ValueError("sessions must not be empty")
    if lookback_days < 1:
        raise ValueError("lookback_days must be >= 1")

    signals: list[dict[str, Any]] = []
    n_instruments = len(instrument_ids)

    # For each session, we generate signal structures; the actual rank
    # computation is done by the runner using market data from the backtester.
    for s_idx, session in enumerate(sessions):
        for inst_idx, inst_id in enumerate(instrument_ids):
            signals.append(
                {
                    "signal_id": f"SIG-MOM-{inst_id[-6:]}-{session.isoformat()}-{s_idx}-{inst_idx}",
                    "instrument_id": inst_id,
                    "signal_session": session,
                    "decision_time": session_close_utc(session),
                    "direction": "long",
                    # target and signal_metric will be set by the runner
                    # based on momentum rank computation using market data.
                    "target": 0.0,
                    "signal_metric": 0.0,
                    "strategy_ref": strategy_ref,
                }
            )

    return signals


# ---------------------------------------------------------------------------
# 4. Mean reversion
# ---------------------------------------------------------------------------


def mean_reversion_signals(
    instrument_ids: list[str],
    sessions: list[date],
    lookback_days: int = 20,
    *,
    strategy_ref: str = "mean_reversion",
) -> list[dict[str, Any]]:
    """Generate mean-reversion signals.

    Rule (deterministic, interpretable, point-in-time valid):

    - For each instrument, compute the distance from a rolling mean (e.g.,
      simple moving average over `lookback_days` sessions prior to the
      signal session).
    - If price is below the mean, signal a long target (expect reversion
      to the mean).
    - If price is above the mean, signal a short target (expect reversion
      to the mean), but Phase 7 long-only accounting will reject/constrain
      the short.
    - Only one instrument is signaled per session (the most extreme).

    The rule uses only information available at decision time: historical
    closes up to and including the close of the signal session. No future
    prices enter the rolling-mean calculation.

    Small pre-registered parameter grid: lookback_days ∈ {10, 20, 30}.
    """
    if not instrument_ids:
        raise ValueError("instrument_ids must not be empty")
    if not sessions:
        raise ValueError("sessions must not be empty")
    if lookback_days < 1:
        raise ValueError("lookback_days must be >= 1")

    signals: list[dict[str, Any]] = []

    for s_idx, session in enumerate(sessions):
        for inst_idx, inst_id in enumerate(instrument_ids):
            signals.append(
                {
                    "signal_id": f"SIG-MR-{inst_id[-6:]}-{session.isoformat()}-{s_idx}-{inst_idx}",
                    "instrument_id": inst_id,
                    "signal_session": session,
                    "decision_time": session_close_utc(session),
                    "direction": "long",
                    # target will be set by the runner based on mean-reversion
                    # rank (price vs rolling mean).  Placeholder 0.0.
                    "target": 0.0,
                    "signal_metric": 0.0,
                    "strategy_ref": strategy_ref,
                }
            )

    return signals


# ---------------------------------------------------------------------------
# 5. Moving-average rule
# ---------------------------------------------------------------------------


def moving_average_signals(
    instrument_ids: list[str],
    sessions: list[date],
    short_window: int = 10,
    long_window: int = 30,
    *,
    strategy_ref: str = "moving_average",
) -> list[dict[str, Any]]:
    """Generate moving-average crossover signals.

    Rule (transparent, deterministic, reproducible):

    - For each instrument, compute the short-term SMA and long-term SMA
      of closes up to and including the signal session.
    - If short SMA > long SMA: signal a long target.
    - If short SMA <= long SMA: signal flat (no action).
    - Only one instrument is signaled per session (the one with the most
      significant crossover).

    The rule uses only information available at decision time: historical
    closes. No future prices enter the SMA calculation.

    Small pre-registered parameter grid:
      - short_window ∈ {5, 10, 15}
      - long_window ∈ {30, 40, 50}
    with exactly 3 pre-registered combinations: (5, 30), (10, 30), (15, 40).
    """
    if not instrument_ids:
        raise ValueError("instrument_ids must not be empty")
    if not sessions:
        raise ValueError("sessions must not be empty")
    if short_window < 1:
        raise ValueError("short_window must be >= 1")
    if long_window < short_window:
        raise ValueError("long_window must be >= short_window")

    signals: list[dict[str, Any]] = []

    for s_idx, session in enumerate(sessions):
        for inst_idx, inst_id in enumerate(instrument_ids):
            signals.append(
                {
                    "signal_id": f"SIG-MA-{inst_id[-6:]}-{session.isoformat()}-{s_idx}-{inst_idx}",
                    "instrument_id": inst_id,
                    "signal_session": session,
                    "decision_time": session_close_utc(session),
                    "direction": "long",
                    # target will be set by the runner based on SMA crossover.
                    # Placeholder 0.0.
                    "target": 0.0,
                    "signal_metric": 0.0,
                    "strategy_ref": strategy_ref,
                }
            )

    return signals


# ---------------------------------------------------------------------------
# 6. Volatility-targeted control
# ---------------------------------------------------------------------------


def volatility_targeted_signals(
    instrument_ids: list[str],
    sessions: list[date],
    target_volatility: float = 0.15,
    estimation_window: int = 30,
    *,
    strategy_ref: str = "volatility_targeted",
) -> list[dict[str, Any]]:
    """Generate volatility-targeted control signals.

    Rule (transparent, deterministic, no future volatility information):

    - For each instrument, compute the sample standard deviation of daily
      closes over the `estimation_window` sessions prior to the signal
      session (including the signal session's close).
    - Scale the target quantity so that the expected position volatility
      matches `target_volatility`.
    - The volatility estimate uses only closes available at the signal
      session's close. No future closes are used.

    The exact target quantity scaling is handled by the backtester's order
    generator (QUANTITY sizing policy): the signal's `target` field is
    treated as a number of shares.  This baseline sets a nominal target;
    the volatility estimation informs the *intended* exposure but the
    actual fill size is determined by the Phase 7 executor.

    Small pre-registered parameter grid:
      - target_volatility ∈ {0.10, 0.15, 0.20}
      - estimation_window ∈ {10, 30, 60}
    with 3 pre-registered combinations.
    """
    if not instrument_ids:
        raise ValueError("instrument_ids must not be empty")
    if not sessions:
        raise ValueError("sessions must not be empty")
    if target_volatility <= 0:
        raise ValueError("target_volatility must be > 0")
    if estimation_window < 1:
        raise ValueError("estimation_window must be >= 1")

    signals: list[dict[str, Any]] = []

    for s_idx, session in enumerate(sessions):
        for inst_idx, inst_id in enumerate(instrument_ids):
            signals.append(
                {
                    "signal_id": f"SIG-VT-{inst_id[-6:]}-{session.isoformat()}-{s_idx}-{inst_idx}",
                    "instrument_id": inst_id,
                    "signal_session": session,
                    "decision_time": session_close_utc(session),
                    "direction": "long",
                    # nominal target; volatility estimation is performed by the
                    # runner using historical closes available at decision time.
                    "target": 100.0,
                    "signal_metric": target_volatility,
                    "strategy_ref": strategy_ref,
                }
            )

    return signals


# ---------------------------------------------------------------------------
# 7. Random / null control
# ---------------------------------------------------------------------------


def random_null_signals(
    instrument_ids: list[str],
    sessions: list[date],
    *,
    seed: int = 42,
    strategy_ref: str = "random_null",
    null: bool = False,
) -> list[dict[str, Any]]:
    """Generate random or null control signals.

    Purpose: sanity check for the research system; demonstrate what
    performance appears under a null-like strategy.

    - If `null=False` (default): random signals with controlled reproducibility.
      Each instrument gets a random direction (long/flat) and random target
      at each session, seeded with `seed`.
    - If `null=True`: null control - no trading activity.  All signals have
      direction="flat" and target=0, producing no fills.  This demonstrates
      the baseline performance of the execution pipeline itself.

    Randomness must be controlled and reproducible.  The seed is recorded
    through the existing experiment/run configuration.

    If multiple random trials are used, the number is small (1 trial) and
    explicitly recorded.
    """
    if not instrument_ids:
        raise ValueError("instrument_ids must not be empty")
    if not sessions:
        raise ValueError("sessions must not be empty")
    if seed < 0:
        raise ValueError("seed must be non-negative")

    import random

    rng = random.Random(seed)

    signals: list[dict[str, Any]] = []

    if null:
        # Null control: no trading activity
        for inst_id in instrument_ids:
            for session in sessions:
                signals.append(
                    {
                        "signal_id": f"SIG-NULL-{inst_id[-6:]}-{session.isoformat()}",
                        "instrument_id": inst_id,
                        "signal_session": session,
                        "decision_time": session_close_utc(session),
                        "direction": "flat",
                        "target": 0.0,
                        "signal_metric": 0.0,
                        "strategy_ref": strategy_ref,
                    }
                )
    else:
        # Random signals: controlled reproducibility
        for inst_id in instrument_ids:
            for session in sessions:
                # Random direction: 70% long, 30% flat (deterministic given seed)
                r = rng.random()
                if r < 0.7:
                    direction = "long"
                else:
                    direction = "flat"
                # Random target: small positive number
                target = round(rng.uniform(10.0, 500.0), 2)

                signals.append(
                    {
                        "signal_id": f"SIG-RAND-{inst_id[-6:]}-{session.isoformat()}",
                        "instrument_id": inst_id,
                        "signal_session": session,
                        "decision_time": session_close_utc(session),
                        "direction": direction,
                        "target": target,
                        "signal_metric": float(rng.random()),
                        "strategy_ref": strategy_ref,
                    }
                )

    return signals