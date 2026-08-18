"""Phase 5 golden tests: every outcome value is hand-computed from the
contract's formula and asserted against the engine output, so a regression
in the labeling math fails loudly.

Golden cases (prompt 17):
    G1  simple forward return, +10%
    G2  negative forward return, -10%
    G3  excess return vs benchmark, +6%
    G4  multi-session horizon resolves exactly the intended five sessions
    G5  calendar gaps (weekend/holiday) never count as sessions
    G6  incomplete horizon -> unavailable, never shortened
    G7  overlapping windows are tagged with exact overlap metadata
    G8  split scenario: canonical adjustment semantics, as-published audit
    G9  delisting: unavailable with reason, never a silent zero
    G10 timestamp boundary: the label begins exactly where the contract
        says it begins (strict session-close rule)
    G11 realized volatility (hand-computed, sqrt(5.04) case)
    G12 max drawdown and max adverse excursion (hand-computed)
    G13 total return with an ex-date dividend (hand-computed)
    G14 split + dividend: dividends converted to split-continuous basis
        (as-published and stored-basis paths must agree)
"""

from __future__ import annotations

import math
from datetime import date, datetime, timezone

import polars as pl
import pytest

from orbit.labels import (
    AnchorMode,
    DrawdownType,
    LabelContract,
    LabelEngine,
    ReturnConvention,
    VolatilityEstimator,
    overlapping_pairs,
)
from orbit.labels.outcomes import (
    excess_return,
    max_adverse_excursion,
    max_drawdown,
    realized_volatility,
    simple_return,
    window_total_return,
)


# ------------------------------------------------------------------ helpers


def _bars(
    sessions: list[date],
    closes: list[float],
    instrument_id: str = "INS-000001",
) -> pl.DataFrame:
    assert len(sessions) == len(closes)
    return pl.DataFrame(
        {
            "instrument_id": [instrument_id] * len(sessions),
            "trade_date": sessions,
            "open": closes,
            "high": [c + 1.0 for c in closes],
            "low": [c - 1.0 for c in closes],
            "close": closes,
            "volume": [1000] * len(sessions),
        }
    )


def _contract(**kw) -> LabelContract:
    base = dict(
        label_id="LAB-001",
        version="v1",
        target_type="forward_return",
        horizon=1,
        anchor_mode=AnchorMode.DECISION_INSTANT,
        return_convention=ReturnConvention.SIMPLE_PRICE_RETURN,
        formula="hand-computed golden contract",
    )
    base.update(kw)
    return LabelContract(**base)


def _dec(inst: str, t: datetime):
    return {"instrument_id": inst, "decision_time": t}


WINTER_CLOSE = timezone.utc  # 16:00 America/New_York == 21:00 UTC in winter


# ------------------------------------------------------- golden 1, 2, 3: pure

def test_g1_simple_forward_return_plus_10pct():
    assert simple_return(100.0, 110.0) == pytest.approx(0.10)
    bars = _bars([date(2020, 1, 6), date(2020, 1, 7)], [100.0, 110.0])
    row = LabelEngine(bars).compute_one(
        _contract(horizon=1),
        "INS-000001",
        datetime(2020, 1, 6, 21, 0, 1, tzinfo=WINTER_CLOSE),
    )
    assert row["outcome_status"] == "available"
    assert row["outcome_value"] == pytest.approx(0.10)
    assert row["outcome_session"] == date(2020, 1, 7)


def test_g2_negative_forward_return_minus_10pct():
    assert simple_return(100.0, 90.0) == pytest.approx(-0.10)
    bars = _bars([date(2020, 1, 6), date(2020, 1, 7)], [100.0, 90.0])
    row = LabelEngine(bars).compute_one(
        _contract(horizon=1),
        "INS-000001",
        datetime(2020, 1, 6, 21, 0, 1, tzinfo=WINTER_CLOSE),
    )
    assert row["outcome_value"] == pytest.approx(-0.10)


def test_g3_excess_return_vs_benchmark():
    assert excess_return(0.10, 0.04) == pytest.approx(0.06)
    sessions = [date(2020, 1, 6), date(2020, 1, 7)]
    bars = pl.concat(
        [
            _bars(sessions, [100.0, 110.0], "INS-000001"),
            _bars(sessions, [300.0, 312.0], "SPY"),
        ]
    )
    row = LabelEngine(bars).compute_one(
        _contract(
            target_type="excess_return", horizon=1, benchmark="SPY"
        ),
        "INS-000001",
        datetime(2020, 1, 6, 21, 0, 1, tzinfo=WINTER_CLOSE),
    )
    assert row["outcome_status"] == "available"
    assert row["outcome_value"] == pytest.approx(0.06)
    assert row["benchmark_return"] == pytest.approx(0.04)


# -------------------------------------------------- golden 4, 5: horizon truth

def test_g4_multi_session_horizon_is_exactly_five_sessions():
    sessions = [
        date(2020, 1, 6), date(2020, 1, 7), date(2020, 1, 8),
        date(2020, 1, 9), date(2020, 1, 10), date(2020, 1, 13),
        date(2020, 1, 14), date(2020, 1, 15), date(2020, 1, 16),
        date(2020, 1, 17), date(2020, 1, 21),
    ]
    closes = [100.0 + i for i in range(len(sessions))]
    bars = _bars(sessions, closes)
    row = LabelEngine(bars).compute_one(
        _contract(horizon=5),
        "INS-000001",
        datetime(2020, 1, 10, 21, 0, 1, tzinfo=WINTER_CLOSE),
    )
    assert row["outcome_status"] == "available"
    assert row["entry_session"] == date(2020, 1, 10)
    assert row["outcome_session"] == date(2020, 1, 17)
    assert row["sessions_available"] == 5
    # the outcome window spans the entry close through the outcome close
    # (inclusive): [entry, outcome] - exactly the five sessions after entry
    assert row["window_start_session"] == date(2020, 1, 10)
    assert row["window_end_session"] == date(2020, 1, 17)
    # close(01-17)=109, close(01-10)=104 -> 109/104 - 1
    assert row["outcome_value"] == pytest.approx(109.0 / 104.0 - 1.0)
    assert row["entry_close_as_published"] == pytest.approx(104.0)
    assert row["outcome_close_as_published"] == pytest.approx(109.0)


def test_g5_calendar_gaps_never_count_as_sessions():
    # weekend (01-11/12) and MLK holiday (01-20) sit between the sessions;
    # a 2-session window after 01-10 close must land on 01-13, 01-14
    sessions = [
        date(2020, 1, 6), date(2020, 1, 7), date(2020, 1, 8),
        date(2020, 1, 9), date(2020, 1, 10), date(2020, 1, 13),
        date(2020, 1, 14), date(2020, 1, 15), date(2020, 1, 16),
        date(2020, 1, 17), date(2020, 1, 21), date(2020, 1, 22),
    ]
    closes = [100.0 + i for i in range(len(sessions))]
    bars = _bars(sessions, closes)
    eng = LabelEngine(bars)
    row = eng.compute_one(
        _contract(horizon=2),
        "INS-000001",
        datetime(2020, 1, 10, 21, 0, 1, tzinfo=WINTER_CLOSE),
    )
    assert row["outcome_session"] == date(2020, 1, 14)
    # the same contract decided after 01-17 close spans the MLK holiday
    row2 = eng.compute_one(
        _contract(horizon=2),
        "INS-000001",
        datetime(2020, 1, 17, 21, 0, 1, tzinfo=WINTER_CLOSE),
    )
    assert row2["entry_session"] == date(2020, 1, 17)
    assert row2["outcome_session"] == date(2020, 1, 22)
    assert row2["sessions_available"] == 2


def test_g6_incomplete_horizon_is_unavailable_never_shortened():
    sessions = [date(2020, 1, 6), date(2020, 1, 7), date(2020, 1, 8)]
    bars = _bars(sessions, [100.0, 101.0, 102.0])
    row = LabelEngine(bars).compute_one(
        _contract(horizon=5),
        "INS-000001",
        datetime(2020, 1, 6, 21, 0, 1, tzinfo=WINTER_CLOSE),
    )
    assert row["outcome_status"] == "unavailable"
    assert row["unavailable_reason"] == "insufficient_future_data"
    assert row["outcome_value"] is None
    assert row["sessions_available"] == 2
    assert row["outcome_session"] is None


# ----------------------------------------------------------- golden 7: overlap

def test_g7_overlapping_windows_carry_exact_metadata():
    from datetime import timedelta

    sessions = [date(2020, 1, 6) + timedelta(days=o) for o in _weekday_offsets(12)]
    closes = [100.0 + i for i in range(len(sessions))]
    bars = _bars(sessions, closes)
    eng = LabelEngine(bars)
    decisions = [
        _dec("INS-000001", datetime(s.year, s.month, s.day, 21, 0, 1, tzinfo=WINTER_CLOSE))
        for s in sessions[:6]
    ]
    frame = eng.compute(_contract(horizon=5), decisions)
    assert frame.height == 6
    pairs = overlapping_pairs(
        frame, sessions_by_instrument={"INS-000001": eng.instrument_sessions("INS-000001")}
    )
    # windows include the entry session ([entry, outcome]); decisions 1..5
    # sessions apart share window sessions, decisions 6+ apart do not
    expected_pairs = sum(
        1 for i in range(6) for j in range(i + 1, 6) if (j - i) <= 5
    )
    assert len(pairs) == expected_pairs
    # the consecutive pair overlaps on exactly horizon window sessions
    pair = next(
        p for p in pairs
        if p["window_start_a"] == date(2020, 1, 6) and p["window_start_b"] == date(2020, 1, 7)
    )
    assert pair["overlap_sessions"] == 5
    assert pair["label_a"] == "LAB-001"
    assert pair["label_b"] == "LAB-001"
    # the boundary case: windows exactly five sessions apart share the
    # boundary session (inclusive intervals) and must be reported
    boundary = next(
        p for p in pairs
        if p["window_start_a"] == date(2020, 1, 6) and p["window_start_b"] == date(2020, 1, 13)
    )
    assert boundary["overlap_sessions"] == 1


def _weekday_offsets(n: int) -> list[int]:
    from datetime import timedelta

    start = date(2020, 1, 6)
    out = []
    d = start
    while len(out) < n:
        if d.weekday() < 5:
            out.append((d - start).days)
        d += timedelta(days=1)
    return out


# ------------------------------------------------------- golden 8: corporate

def test_g8_split_canonical_adjustment_semantics():
    # AAPL-like: 7:1 split ex-date 2020-08-31; the stored (split-continuous)
    # series has no jump at the split; as-published closes are audited
    sessions = [date(2020, 8, 27), date(2020, 8, 28), date(2020, 8, 31)]
    stored = [425.04 / 7, 499.23 / 7, 129.04]   # split-continuous closes
    as_pub = [425.04, 499.23, 129.04]           # as-published raw closes
    bars = _bars(sessions, stored)
    events = pl.DataFrame(
        {
            "instrument_id": ["INS-000001"],
            "kind": ["splits"],
            "ts": [datetime(2020, 8, 31, 12, 0, tzinfo=timezone.utc)],
            "ratio": [7.0],
        }
    )
    eng = LabelEngine(bars, events=events)
    row = eng.compute_one(
        _contract(horizon=2),
        "INS-000001",
        datetime(2020, 8, 27, 21, 0, 1, tzinfo=WINTER_CLOSE),
    )
    assert row["outcome_status"] == "available"
    # stored-basis return: stored(08-31)/stored(08-27) - 1
    assert row["outcome_value"] == pytest.approx(stored[2] / stored[0] - 1.0)
    # as-published audit columns (the split must NOT create an artificial
    # -74% "return" from the as-published 499.23 -> 129.04 collapse)
    assert row["entry_close_as_published"] == pytest.approx(as_pub[0])
    assert row["outcome_close_as_published"] == pytest.approx(as_pub[2])
    naive = as_pub[2] / as_pub[1] - 1.0
    assert abs(row["outcome_value"] - naive) > 0.5
    # as-published consistency identity: outcome == as_pub ratio * split ratio
    assert row["outcome_value"] == pytest.approx(as_pub[2] / as_pub[0] * 7.0 - 1.0)


# --------------------------------------------------------- golden 9: delisting

def test_g9_delisting_is_never_a_silent_zero():
    sessions = [date(2020, 1, 6), date(2020, 1, 7), date(2020, 1, 8),
                date(2020, 1, 9), date(2020, 1, 10), date(2020, 1, 13),
                date(2020, 1, 14)]
    bars = _bars(sessions, [100.0 + i for i in range(len(sessions))])
    instruments = pl.DataFrame(
        {
            "instrument_id": ["INS-000001"],
            "name": ["Delisted Co"],
            "delisting_date": [date(2020, 1, 15)],
        }
    )
    row = LabelEngine(bars, instruments=instruments).compute_one(
        _contract(horizon=5),
        "INS-000001",
        datetime(2020, 1, 10, 21, 0, 1, tzinfo=WINTER_CLOSE),
    )
    assert row["outcome_status"] == "unavailable"
    assert row["unavailable_reason"] == "delisted"
    assert row["outcome_value"] is None
    # without the instrument master the same data is a data shortfall,
    # not a delisting
    row2 = LabelEngine(bars).compute_one(
        _contract(horizon=5),
        "INS-000001",
        datetime(2020, 1, 10, 21, 0, 1, tzinfo=WINTER_CLOSE),
    )
    assert row2["unavailable_reason"] == "insufficient_future_data"


# ----------------------------------------------------- golden 10: boundary

def test_g10_label_begins_exactly_where_contract_says():
    sessions = [date(2020, 1, 8), date(2020, 1, 9), date(2020, 1, 10)]
    bars = _bars(sessions, [100.0, 101.0, 102.0])
    eng = LabelEngine(bars)
    # exactly at the session close (21:00 UTC winter): the 01-10 bar is NOT
    # yet available -> entry is 01-09
    at_close = eng.compute_one(
        _contract(horizon=1), "INS-000001",
        datetime(2020, 1, 10, 21, 0, 0, tzinfo=WINTER_CLOSE),
    )
    assert at_close["entry_session"] == date(2020, 1, 9)
    # one microsecond after the close: 01-10 is the entry
    after_close = eng.compute_one(
        _contract(horizon=1), "INS-000001",
        datetime(2020, 1, 10, 21, 0, 0, 1, tzinfo=WINTER_CLOSE),
    )
    assert after_close["entry_session"] == date(2020, 1, 10)
    # mid-session (in-progress bar) must NOT be used as entry
    midday = eng.compute_one(
        _contract(horizon=1), "INS-000001",
        datetime(2020, 1, 10, 15, 0, 0, tzinfo=timezone.utc),
    )
    assert midday["entry_session"] == date(2020, 1, 9)
    # naive naive-local time (naive datetime is UTC by convention)
    naive = eng.compute_one(
        _contract(horizon=1), "INS-000001", datetime(2020, 1, 10, 21, 0, 1),
    )
    assert naive["entry_session"] == date(2020, 1, 10)


# ------------------------------------------- golden 11: realized volatility

def test_g11_realized_volatility_hand_computed():
    # realized volatility hand calc: returns +0.10, -0.10 -> mean 0,
    # sample var 0.02, annualized sqrt(0.02 * 252) = sqrt(5.04)
    assert realized_volatility([0.10, -0.10], 252.0) == pytest.approx(math.sqrt(5.04))
    sessions = [date(2020, 1, 6), date(2020, 1, 7), date(2020, 1, 8)]
    bars = _bars(sessions, [100.0, 110.0, 99.0])
    row = LabelEngine(bars).compute_one(
        _contract(
            target_type="volatility",
            horizon=2,
            return_convention=ReturnConvention.SIMPLE_PRICE_RETURN,
            volatility_estimator=VolatilityEstimator.SAMPLE_STD_CLOSE_TO_CLOSE,
            annualization=252.0,
            min_observations=2,
        ),
        "INS-000001",
        datetime(2020, 1, 6, 21, 0, 1, tzinfo=WINTER_CLOSE),
    )
    assert row["outcome_status"] == "available"
    assert row["outcome_value"] == pytest.approx(math.sqrt(5.04))
    # the sample estimator yields exactly H observations for a complete
    # window, so min_observations == H can never make an available window
    # unavailable (the guard is defensive for future estimators)
    row_max = LabelEngine(bars).compute_one(
        _contract(
            target_type="volatility",
            horizon=2,
            return_convention=ReturnConvention.SIMPLE_PRICE_RETURN,
            volatility_estimator=VolatilityEstimator.SAMPLE_STD_CLOSE_TO_CLOSE,
            annualization=252.0,
            min_observations=2,
        ),
        "INS-000001",
        datetime(2020, 1, 6, 21, 0, 1, tzinfo=WINTER_CLOSE),
    )
    assert row_max["outcome_status"] == "available"


# ------------------------------------------------ golden 12: max drawdown/MAE

def test_g12_max_drawdown_and_mae_hand_computed():
    from datetime import timedelta

    # entry close 100; window closes 105, 102, 98, 101, 95
    # peak after s1 is 105 -> trough 95 -> MDD = 1 - 95/105 = 10/105
    # MAE = 1 - 95/100 = 0.05
    closes = [100.0, 105.0, 102.0, 98.0, 101.0, 95.0]
    assert max_drawdown(closes, 0, 5) == pytest.approx(10.0 / 105.0)
    assert max_adverse_excursion(closes[0], closes[1:6]) == pytest.approx(0.05)
    sessions = [date(2020, 1, 6) + timedelta(days=o) for o in _weekday_offsets(6)]
    bars = _bars(sessions, closes)
    row = LabelEngine(bars).compute_one(
        _contract(
            target_type="drawdown",
            horizon=5,
            return_convention=ReturnConvention.SIMPLE_PRICE_RETURN,
            drawdown_type=DrawdownType.MAX_DRAWDOWN,
        ),
        "INS-000001",
        datetime(2020, 1, 6, 21, 0, 1, tzinfo=WINTER_CLOSE),
    )
    assert row["outcome_value"] == pytest.approx(10.0 / 105.0)
    row_mae = LabelEngine(bars).compute_one(
        _contract(
            target_type="drawdown",
            horizon=5,
            return_convention=ReturnConvention.SIMPLE_PRICE_RETURN,
            drawdown_type=DrawdownType.MAX_ADVERSE_EXCURSION,
        ),
        "INS-000001",
        datetime(2020, 1, 6, 21, 0, 1, tzinfo=WINTER_CLOSE),
    )
    assert row_mae["outcome_value"] == pytest.approx(0.05)


# ------------------------------------- golden 13: total return with dividend

def test_g13_total_return_with_exdate_dividend_hand_computed():
    # entry 100; window s1=101 (dividend $1 ex-date), s2=103
    # total return = (101+1)/100 * 103/101 - 1 = 1.02 * 103/101 - 1
    sessions = [date(2020, 1, 6), date(2020, 1, 7), date(2020, 1, 8)]
    bars = _bars(sessions, [100.0, 101.0, 103.0])
    events = pl.DataFrame(
        {
            "instrument_id": ["INS-000001"],
            "kind": ["dividends"],
            "ts": [datetime(2020, 1, 7, 12, 0, tzinfo=timezone.utc)],
            "ratio": [1.0],
        }
    )
    eng = LabelEngine(bars, events=events)
    row = eng.compute_one(
        _contract(
            target_type="forward_return",
            horizon=2,
            return_convention=ReturnConvention.SIMPLE_TOTAL_RETURN,
        ),
        "INS-000001",
        datetime(2020, 1, 6, 21, 0, 1, tzinfo=WINTER_CLOSE),
    )
    assert row["outcome_status"] == "available"
    assert row["outcome_value"] == pytest.approx(1.02 * 103.0 / 101.0 - 1.0)
    assert row["total_dividends"] == pytest.approx(1.0)
    # the same contract with price return excludes the dividend
    row_price = eng.compute_one(
        _contract(horizon=2),
        "INS-000001",
        datetime(2020, 1, 6, 21, 0, 1, tzinfo=WINTER_CLOSE),
    )
    assert row_price["outcome_value"] == pytest.approx(103.0 / 100.0 - 1.0)


# --------------------------- golden 14: split + dividend basis consistency

def test_g14_split_and_dividend_basis_consistency():
    # entry raw 100 / stored 25 (4:1 split ex-date on the SECOND window
    # session); window s1 raw 101 / stored 25.25 with a $1 ex-date dividend;
    # outcome s2 stored 25.5 (post-split raw 25.5)
    # stored-basis total = (25.25+0.25)/25 * 25.5/25.25 - 1
    # raw-basis total   = (101+1)/100 * (4*25.5)/101 - 1  -> must agree
    sessions = [date(2020, 1, 6), date(2020, 1, 7), date(2020, 1, 8)]
    bars = _bars(sessions, [25.0, 25.25, 25.5])
    events = pl.DataFrame(
        {
            "instrument_id": ["INS-000001", "INS-000001"],
            "kind": ["splits", "dividends"],
            "ts": [
                datetime(2020, 1, 8, 12, 0, tzinfo=timezone.utc),
                datetime(2020, 1, 7, 12, 0, tzinfo=timezone.utc),
            ],
            "ratio": [4.0, 1.0],
        }
    )
    row = LabelEngine(bars, events=events).compute_one(
        _contract(
            target_type="forward_return",
            horizon=2,
            return_convention=ReturnConvention.SIMPLE_TOTAL_RETURN,
        ),
        "INS-000001",
        datetime(2020, 1, 6, 21, 0, 1, tzinfo=WINTER_CLOSE),
    )
    assert row["outcome_status"] == "available"
    stored_basis = (25.25 + 0.25) / 25.0 * 25.5 / 25.25 - 1.0
    raw_basis = (101.0 + 1.0) / 100.0 * (4.0 * 25.5) / 101.0 - 1.0
    assert row["outcome_value"] == pytest.approx(stored_basis)
    assert row["outcome_value"] == pytest.approx(raw_basis)
    # the dividend is converted to stored basis (raw $1 / factor 4 = $0.25)
    assert row["total_dividends"] == pytest.approx(0.25)
    # as-published audit: entry was 100 raw, s1 was 101 raw
    assert row["entry_close_as_published"] == pytest.approx(100.0)