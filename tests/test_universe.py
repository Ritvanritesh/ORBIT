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


def _instrument(ins_id, ticker, listed, delisted=None, reason=None, stype="equity", exchange="XNYS"):
    return Instrument(
        instrument_id=ins_id,
        primary_ticker=ticker,
        exchange_id=exchange,
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
        _instrument("INS-000004", "ILLIQ", start),                   # illiquid
        _instrument("INS-000005", "PENNY", start),                   # penny stock
        _instrument("INS-000006", "SPIKE", start),                   # liquidity regime change
        _instrument("INS-000007", "OLD", start),                     # ticker change
        _instrument("INS-000008", "FOREIGN", start, stype="equity", exchange="XAMS"),  # non-NYSE/NASDAQ
        _instrument("INS-000009", "THEETF", start, stype="etf"),     # ETF in equity universe
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
    gen("INS-000008", start, end, 95.0, 60_000_000)
    gen("INS-000009", start, end, 200.0, 80_000_000)
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
    return UniverseEngine(
        SyntheticAccessor(instruments, bars, sym), RULE, data_ref="synthetic_v1"
    )


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
    snap = UniverseEngine(
        SyntheticAccessor(*fixtures), capped, data_ref="synthetic_v1"
    ).membership(date(2020, 6, 1))
    assert len(snap.members) == 2
    assert snap.members[0].rank == 1
    assert snap.members[1].rank == 2
    assert any(e.reason == "below_liquidity_cap" for e in snap.excluded)


def test_every_exclusion_is_reasoned(fixtures):
    snap = _engine(fixtures).membership(date(2020, 6, 1))
    assert len(snap.excluded) >= 3
    assert all(e.reason for e in snap.excluded)


def test_exchange_and_security_type_filters(fixtures):
    instruments, bars, sym = fixtures
    snap = _engine(fixtures).membership(date(2020, 6, 1))
    assert "INS-000008" not in snap.instrument_ids  # non-NYSE/NASDAQ exchange
    assert "INS-000009" not in snap.instrument_ids  # ETF in equity universe
    reasons = {e.instrument_id: e.reason for e in snap.excluded}
    assert reasons["INS-000008"] == "exchange=XAMS"
    assert reasons["INS-000009"] == "security_type=etf"


def test_instrument_requires_delisting_date_for_reason():
    from orbit.schemas.instrument import Instrument
    with pytest.raises(Exception):
        Instrument(
            instrument_id="INS-000100", primary_ticker="X", exchange_id="XNYS",
            name="X", security_type="equity", listing_date="2020-01-01",
            delisting_reason="merger",
        )


def test_dataset_snapshot_rejects_reversed_range():
    from datetime import datetime
    from orbit.schemas.data import DatasetSnapshot
    with pytest.raises(Exception):
        DatasetSnapshot(
            snapshot_id="DS-000001", provider="p", source_uri="u",
            checksum="a" * 32, schema_version="v1",
            available_from="2021-01-01", available_to="2020-01-01",
            ingest_time=datetime(2026, 1, 1),
        )


def test_symbol_history_registry_rejects_overlap():
    from orbit.schemas.instrument import SymbolHistory, SymbolHistoryRegistry
    with pytest.raises(Exception):
        SymbolHistoryRegistry(
            entries=[
                SymbolHistory(instrument_id="INS-000001", symbol="A", effective_from="2020-01-01", effective_to="2020-06-01"),
                SymbolHistory(instrument_id="INS-000001", symbol="B", effective_from="2020-05-01"),
            ]
        )


def test_symbol_history_registry_rejects_gapless_extension_after_open_end():
    from orbit.schemas.instrument import SymbolHistory, SymbolHistoryRegistry
    with pytest.raises(Exception):
        SymbolHistoryRegistry(
            entries=[
                SymbolHistory(instrument_id="INS-000001", symbol="A", effective_from="2020-01-01"),
                SymbolHistory(instrument_id="INS-000001", symbol="B", effective_from="2021-01-01"),
            ]
        )


def test_symbol_history_registry_resolves():
    from orbit.schemas.instrument import SymbolHistory, SymbolHistoryRegistry
    reg = SymbolHistoryRegistry(
        entries=[
            SymbolHistory(instrument_id="INS-000001", symbol="A", effective_from="2020-01-01", effective_to="2020-06-01"),
            SymbolHistory(instrument_id="INS-000001", symbol="B", effective_from="2020-06-02"),
        ]
    )
    assert reg.resolve("INS-000001", date(2020, 3, 1)) == "A"
    assert reg.resolve("INS-000001", date(2020, 9, 1)) == "B"


def test_snapshot_carries_data_ref(fixtures):
    engine = UniverseEngine(SyntheticAccessor(*fixtures), RULE, data_ref="synthetic_v1")
    snap = engine.membership(date(2020, 6, 1))
    assert snap.data_ref == "synthetic_v1"


def test_data_ref_is_required():
    from orbit.universe.engine import UniverseEngine as UE
    with pytest.raises(Exception):
        UE(SyntheticAccessor(*fixtures), RULE)


def test_corporate_action_validation():
    from orbit.schemas.instrument import CorporateAction
    with pytest.raises(Exception):
        CorporateAction(
            action_id="CA-000010", instrument_id="INS-000001",
            action_type="dividend", effective_date="2020-01-01", ratio=-5.0,
        )
    with pytest.raises(Exception):
        CorporateAction(
            action_id="CA-000011", instrument_id="INS-000001",
            action_type="split", effective_date="2020-01-01",
            ex_date="2020-06-01", ratio=2.0,
        )
    CorporateAction(
        action_id="CA-000012", instrument_id="INS-000001",
        action_type="split", effective_date="2020-01-01",
        ex_date="2020-01-01", ratio=2.0,
    )


def test_exchange_session_validation():
    from orbit.schemas.instrument import Exchange
    with pytest.raises(Exception):
        Exchange(
            exchange_id="XAAA", name="x", mic="AAAA", country="US",
            tz="America/New_York", open_local="15:00", close_local="09:30",
        )


def test_engine_accepts_symbol_history_registry_accessor(fixtures):
    from orbit.schemas.instrument import SymbolHistory, SymbolHistoryRegistry
    instruments, bars, _ = fixtures
    registry = SymbolHistoryRegistry(
        entries=[
            SymbolHistory(instrument_id="INS-000007", symbol="OLD", effective_from=date(2015, 1, 5), effective_to=date(2017, 2, 28)),
            SymbolHistory(instrument_id="INS-000007", symbol="NEW", effective_from=date(2017, 3, 1)),
        ]
    )

    class RegistryAccessor(SyntheticAccessor):
        def __init__(self, instruments, bars, registry):
            super().__init__(instruments, bars)
            self.symbol_history = registry

    engine = UniverseEngine(
        RegistryAccessor(instruments, bars, registry), RULE, data_ref="synthetic_v1"
    )
    snap = engine.membership(date(2017, 6, 15))
    member = next(m for m in snap.members if m.instrument_id == "INS-000007")
    assert member.symbol_at_asof == "NEW"


def test_delisting_reason_is_distinct_from_future_listing(fixtures):
    snap = _engine(fixtures).membership(date(2018, 7, 1))
    reasons = {e.instrument_id: e.reason for e in snap.excluded}
    assert reasons["INS-000002"] == "delisted_asof(2018-06-01)"
    assert reasons["INS-000003"] == "listed_after_asof"