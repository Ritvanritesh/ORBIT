"""Pure outcome mathematics for the Phase 5 label engine.

Every function here is a deterministic, documented formula. The engine
(`orbit.labels.engine`) supplies the series (stored split-continuous closes,
ex-date dividend amounts on the same share basis, session dates); these
functions compute the outcome value. Keeping the math pure makes golden
hand-calculation tests direct and unambiguous.

Conventions (all defined in `contract.py` / `docs/phase5_labels.md`):

  - prices are the canonical stored split-continuous closes
    (close = raw / product of all later split ratios); a split inside the
    outcome window never creates an artificial return;
  - dividends are ex-date cash amounts converted to the same share basis;
    a dividend with ex-date session strictly inside (entry, outcome] counts
    and is reinvested at the ex-date close;
  - returns are simple (not logarithmic): P/P0 - 1;
  - drawdowns are non-negative fractions of the running peak / entry close;
  - volatility is the sample standard deviation (ddof=1) of daily
    close-to-close returns, annualized by the contract factor.
"""

from __future__ import annotations

import math

from orbit.labels.contract import DrawdownType, ReturnConvention


def simple_return(entry_value: float, outcome_value: float) -> float:
    """r = outcome / entry - 1 (simple, not log)."""
    return outcome_value / entry_value - 1.0


def window_total_return(
    closes: list[float],
    entry_idx: int,
    horizon: int,
    dividends: dict[int, float],
) -> float:
    """Total return over the H sessions following entry_idx.

    `closes` is the split-continuous close series (indexed by bar position).
    `dividends` maps the WINDOW-RELATIVE position i (1..H, i.e. closes[
    entry_idx + i]) to the ex-date dividend amount expressed on the same
    split-continuous share basis as the closes.

    Formula (reinvestment at the ex-date close):

        r = prod_{i=1..H} ( close[entry+i] + D[i] ) / close[entry+i-1]  -  1

    When no dividend falls on a session, D[i] = 0 and the factor is the
    plain close-to-close price ratio, so SIMPLE_PRICE_RETURN is the special
    case of SIMPLE_TOTAL_RETURN with no dividends.
    """
    r = 1.0
    prev = closes[entry_idx]
    for i in range(1, horizon + 1):
        cur = closes[entry_idx + i]
        d = dividends.get(i, 0.0)
        r *= (cur + d) / prev
        prev = cur
    return r - 1.0


def excess_return(asset_return: float, benchmark_return: float) -> float:
    """excess = asset forward return - benchmark forward return over the
    SAME horizon, anchored at the SAME decision instant."""
    return asset_return - benchmark_return


def sample_std(returns: list[float]) -> float:
    """Sample standard deviation (ddof=1). Requires >= 2 values."""
    if len(returns) < 2:
        raise ValueError("sample_std requires at least 2 observations")
    n = len(returns)
    mean = sum(returns) / n
    variance = sum((x - mean) ** 2 for x in returns) / (n - 1)
    return math.sqrt(variance)


def realized_volatility(returns: list[float], annualization: float) -> float:
    """Annualized realized volatility of a daily-return series.

    value = sample_std(returns) * sqrt(annualization).
    """
    return sample_std(returns) * math.sqrt(annualization)


def window_returns(closes: list[float], entry_idx: int, horizon: int) -> list[float]:
    """The H close-to-close returns of the outcome window following
    entry_idx: r_i = close[entry+i] / close[entry+i-1] - 1 for i = 1..H."""
    return [
        closes[entry_idx + i] / closes[entry_idx + i - 1] - 1.0
        for i in range(1, horizon + 1)
    ]


def max_drawdown(closes: list[float], entry_idx: int, horizon: int) -> float:
    """Maximum peak-to-trough decline over the outcome window.

    The running peak starts at the ENTRY close (the entry close counts as
    the initial peak) and includes every window close up to and including
    the outcome close; the trough is any later close. Reported as a
    non-negative fraction: 0 when the window never falls below its running
    peak.

        mdd = max( 0, max over s in window of
                   1 - close[s] / max(close[t] for t in [entry..s]) )
    """
    peak = closes[entry_idx]
    mdd = 0.0
    for i in range(1, horizon + 1):
        cur = closes[entry_idx + i]
        if cur < peak:
            mdd = max(mdd, 1.0 - cur / peak)
        peak = max(peak, cur)
    return mdd


def max_adverse_excursion(
    entry_close: float, window_closes: list[float]
) -> float:
    """Maximum decline from the ENTRY close only:

        mae = max(0, 1 - min(window closes) / entry close)
    """
    if not window_closes:
        return 0.0
    return max(0.0, 1.0 - min(window_closes) / entry_close)


def compute_drawdown(
    closes: list[float], entry_idx: int, horizon: int, drawdown_type: DrawdownType
) -> float:
    if drawdown_type == DrawdownType.MAX_DRAWDOWN:
        return max_drawdown(closes, entry_idx, horizon)
    return max_adverse_excursion(closes[entry_idx], closes[entry_idx + 1: entry_idx + 1 + horizon])


def compute_return(
    closes: list[float],
    entry_idx: int,
    horizon: int,
    convention: ReturnConvention,
    dividends: dict[int, float] | None = None,
) -> float:
    """The window return per the contract's convention. SIMPLE_PRICE_RETURN
    excludes dividends (dividends are ignored, exactly as the convention
    promises); SIMPLE_TOTAL_RETURN reinvests them at the ex-date close.
    Both are close-to-close on the split-continuous basis, so they share
    one formula and can never diverge except for the dividend term."""
    divs = (
        {}
        if convention == ReturnConvention.SIMPLE_PRICE_RETURN
        else (dividends if dividends is not None else {})
    )
    return window_total_return(closes, entry_idx, horizon, divs)


__all__ = [
    "compute_drawdown",
    "compute_return",
    "excess_return",
    "max_adverse_excursion",
    "max_drawdown",
    "realized_volatility",
    "sample_std",
    "simple_return",
    "window_returns",
    "window_total_return",
]