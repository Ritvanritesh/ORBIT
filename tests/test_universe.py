"""Phase 2 tests: universe reconstruction with synthetic data.

These prove the selection logic before any real data is licensed:
survivorship control, lagged liquidity, identity resolution, determinism.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from orbit.schemas.instrument import Instrument, SymbolHistory
from orbit.universe.engine import UniverseEngine
from orbit.universe.rules import MembershipRule


def _business_days(start: date, end: date) -> list[date]:
    days = []
    d = start
    while d <= end:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    return days


class SyntheticAccessor:
    """In-memory DataAccessor built from fixtures. All data is lagged-safe."""

    def __init__(self, instruments, bars, symbol_history=None):
        self._instruments = instruments
        self._bars = bars  # {instrument_id: {date: (close, volume)}}
        self._symbol_history = symbol_history or {}

    def instruments(self):
        return self._instruments

    def symbol_history(self, instrument_id):
        return self._symbol_history.get(instrument_id, [])

    def trailing_dollar_volume(self, instrument_id, as_of, window_days):
        bars = sorted(
            (d, c * v)
            for d, (c, v) in self._bars.get(instrument_id, {}).items()
            if d < as_of
        )
        window = [dv for _, dv in bars[-window_days:]]
        if not window:
            return None
        mid = len(window) // 2
        return sorted(window)[mid] if len(window) % 2 else (sorted(window)[mid - 1] + sorted(window)[mid]) / 2

    def last_close(self, instrument_id, as_of):
        closes = [
            c for d, (c, _) in self._bars.get(instrument_id, {}).items() if d < as_of
        ]
        return closes[-1] if closes else None


def _instrument(ins_id, ticker, listed, delisted=None, reason=None, stype="equity"):
    return Instrument(
        instrument_id=ins_id,
        primary_ticker=ticker,
        exchange_id="XNYS",
        name=f"{ticker} Inc",
        security_type=stype,
        listing_date=listed,
        delisting_date=delisted,
        delisting_reason=reason,
    )


@pytest.fixture
def fixtures():
    start = date(2015, 1, 5)
    end = date(2021, 12, 31)
    bars = {}

    def gen(ins_id, listed, last, close, volume, until=None):
        ldate = max(listed, start)
        udate = min(until or last, end)
        bars[ins_id] = {d: (close, volume) for d in _business_days(ldate, udate)}

    instruments = [
        _instrument("INS-000001", "LIQ1", start),                    # active, liquid
        _instrument("INS-000002", "DELIST", start, date(2018, 6, 1), "merger"),  # delisted
        _instrument("INS-000003", "LATE", date(2019, 3, 1)),         # listed later
        _instrument("INS-000004", "ILLIQ", start),       # illiquid
        _instrument("INS-000005", "PENNY", start),                   # penny stock
        _instrument("INS-000006", "SPIKE", start),                   # volume spike on as_of
        _instrument("INS-000007", "OLD", start),                     # ticker change
    ]
    gen("INS-000001", start, end, 100.0, 50_000_000)
    gen("INS-000002", start, end, 90.0, 40_000_000, until=date(2018, 5, 31))
    gen("INS-000003", date(2019, 3, 1), end, 80.0, 30_000_000)
    gen("INS-000004", start, end, 70.0, 50_000)      # $3.5M/day - illiquid
    gen("INS-000005", start, end, 3.0, 25_000_000)
    bars["INS-000006"] = {
        d: (60.0, 50_000) for d in _business_days(start, date(2019, 12, 31))
    }
    # liquidity regime change: high volume from 2020-01-01 onward
    for d in _business_days(date(2020, 1, 1), end):
        bars["INS-000006"][d] = (60.0, 1_700_000)   # $102M/day
    gen("INS-000007", start, end, 55.0, 30_000_000)

    symbol_history = {
        "INS-000007": [
            SymbolHistory(instrument_id="INS-000007", symbol="OLD", effective_from=start, effective_to=date(2017, 2, 28)),
            SymbolHistory(instrument_id="INS-000007", symbol="NEW", effective_from=date(2017, 3, 1)),
        ]
    }
    return instruments, bars, symbol_history


RULE = MembershipRule(
    rule_id="RULE-001",
    version="v1",
    max_names=100,
    min_price=5.0,
    min_trailing_dollar_volume=20_000_000.0,
    liquidity_window_days=20,
)


def _engine(fixtures):
    instruments, bars, sym = fixtures
    return UniverseEngine(SyntheticAccessor(instruments, bars, sym), RULE)


def test_delisted_instrument_excluded_after_delisting(fixtures):
    snap = _engine(fixtures).membership(date(2018, 7, 1))
    assert "INS-000002" not in snap.instrument_ids
    assert any(e.instrument_id == "INS-000002" for e in snap.excluded)


def test_delisted_instrument_included_before_delisting(fixtures):
    snap = _engine(fixtures).membership(date(2018, 5, 15))
    assert "INS-000002" in snap.instrument_ids


def test_future_listing_never_members(fixtures):
    snap = _engine(fixtures).membership(date(2018, 7, 1))
    assert "INS-000003" not in snap.instrument_ids
    snap2 = _engine(fixtures).membership(date(2019, 6, 1))
    assert "INS-000003" in snap2.instrument_ids


def test_liquidity_and_price_filters(fixtures):
    snap = _engine(fixtures).membership(date(2020, 6, 1))
    assert "INS-000004" not in snap.instrument_ids  # illiquid
    assert "INS-000005" not in snap.instrument_ids  # penny


def test_liquidity_is_lagged_not_same_day(fixtures):
    """A liquidity regime change on 2020-01-01 must not affect membership
    on that day: the median window only sees bars strictly before as_of.
    Only after the window fills does the name become a member."""
    engine = _engine(fixtures)
    before = engine.membership(date(2019, 12, 31))
    assert "INS-000006" not in before.instrument_ids
    same_day = engine.membership(date(2020, 1, 2))
    assert "INS-000006" not in same_day.instrument_ids  # window not yet filled
    filled = engine.membership(date(2020, 2, 15))
    assert "INS-000006" in filled.instrument_ids


def test_symbol_history_resolution(fixtures):
    old = _engine(fixtures).membership(date(2017, 1, 15))
    new = _engine(fixtures).membership(date(2017, 6, 15))
    old_sym = next(m for m in old.members if m.instrument_id == "INS-000007")
    new_sym = next(m for m in new.members if m.instrument_id == "INS-000007")
    assert old_sym.symbol_at_asof == "OLD"
    assert new_sym.symbol_at_asof == "NEW"


def test_membership_is_deterministic(fixtures):
    engine = _engine(fixtures)
    a = engine.membership(date(2020, 6, 1))
    b = engine.membership(date(2020, 6, 1))
    assert a == b


def test_ranking_and_cap(fixtures):
    capped = MembershipRule(
        rule_id="RULE-002", version="v1", max_names=2,
        min_price=5.0, min_trailing_dollar_volume=20_000_000.0,
    )
    snap = UniverseEngine(SyntheticAccessor(*fixtures), capped).membership(date(2020, 6, 1))
    assert len(snap.members) == 2
    assert snap.members[0].rank == 1
    assert snap.members[1].rank == 2
    assert any(e.reason == "below_liquidity_cap" for e in snap.excluded)


def test_every_exclusion_is_reasoned(fixtures):
    snap = _engine(fixtures).membership(date(2020, 6, 1))
    assert len(snap.excluded) >= 3
    assert all(e.reason for e in snap.excluded)