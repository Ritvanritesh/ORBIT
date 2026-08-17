"""Phase 4 rules tests: the per-record availability decision.

These tests pin the documented conventions:

  - strict boundary: publication < as_of is required (exact tie rejects)
  - date-precision publication is available the NEXT day
  - market bars are available at the session close, never at ts_utc
  - a market bar's session close depends on DST (21:00 UTC EST / 20:00 EDT)
  - missing publication is never available
  - event time after as_of rejects (forward-dated records)
  - revised macro series without vintage history are never point-in-time
  - ingestion time is provenance, not availability
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import polars as pl
import pytest

from orbit.temporal.adapters import (
    fred_timing_frame,
    market_timing_frame,
    sec_timing_frame,
)
from orbit.temporal.engine import TemporalTruthEngine
from orbit.temporal.rules import AvailabilityDecision
from orbit.temporal.times import (
    DecisionCode,
    TimePrecision,
    Timing,
    normalize_instant,
    session_close_utc,
)

ENGINE = TemporalTruthEngine()


def _timing(
    record_id: str,
    *,
    event: datetime | None = None,
    pub: datetime | None = None,
    precision: TimePrecision = TimePrecision.DATETIME,
    ingested: datetime | None = None,
    vintage_date: date | None = None,
    policy: str | None = None,
    domain: str = "fundamentals",
    kind: str = "fact",
) -> Timing:
    return Timing(
        record_id=record_id,
        domain=domain,
        kind=kind,
        event_time=event,
        publication_time=pub,
        publication_precision=precision,
        effective_time=pub,
        ingestion_time=ingested,
        vintage_id=vintage_date.isoformat() if vintage_date else None,
        vintage_date=vintage_date,
        series_policy=policy,
    )


T = datetime(2018, 1, 10, 16, 0, 0)


# ------------------------------------------------------------ publication


def test_publication_before_decision_allows():
    d = ENGINE.decide_record(
        _timing("r1", event=datetime(2018, 1, 5), pub=datetime(2018, 1, 8, 10)),
        T,
    )
    assert d.allowed
    assert d.code == DecisionCode.ALLOWED_BEFORE_PUBLICATION


def test_publication_after_decision_rejects():
    d = ENGINE.decide_record(
        _timing("r1", event=datetime(2018, 1, 5), pub=datetime(2018, 1, 11, 10)),
        T,
    )
    assert not d.allowed
    assert d.code == DecisionCode.PUBLICATION_AT_OR_AFTER_AS_OF


def test_publication_exactly_at_decision_rejects_strict_boundary():
    """Test 6: the boundary is explicit and strict - publication == as_of is
    NOT available."""
    d = ENGINE.decide_record(_timing("r1", pub=T), T)
    assert not d.allowed
    assert d.code == DecisionCode.PUBLICATION_AT_OR_AFTER_AS_OF


def test_one_minute_before_publication_rejects():
    """Test 7: one minute before publication -> REJECT."""
    d = ENGINE.decide_record(
        _timing("r1", pub=datetime(2018, 1, 10, 16, 1)), T
    )
    assert not d.allowed
    assert d.code == DecisionCode.PUBLICATION_AT_OR_AFTER_AS_OF


def test_one_minute_after_publication_allows():
    """Test 8: one minute after publication -> ALLOW."""
    d = ENGINE.decide_record(
        _timing("r1", pub=datetime(2018, 1, 10, 15, 59)), T
    )
    assert d.allowed


def test_missing_publication_time_rejects():
    """Test 9 + missing-timestamp policy: never invent availability."""
    d = ENGINE.decide_record(
        _timing("r1", event=datetime(2017, 12, 31), pub=None), T
    )
    assert not d.allowed
    assert d.code == DecisionCode.MISSING_PUBLICATION_TIME


# ------------------------------------------------------------------ events


def test_event_before_decision_allows():
    d = ENGINE.decide_record(
        _timing("r1", event=datetime(2018, 1, 5), pub=datetime(2018, 1, 8)),
        T,
    )
    assert d.allowed


def test_event_after_decision_rejects_forward_dated():
    """Test 1 variant: event occurs after the decision - excluded from the
    historical information set even though it was already published."""
    d = ENGINE.decide_record(
        _timing("r1", event=datetime(2018, 1, 15), pub=datetime(2018, 1, 8)),
        T,
    )
    assert not d.allowed
    assert d.code == DecisionCode.EVENT_AFTER_AS_OF


# ---------------------------------------------------------------- filings


def test_fiscal_period_before_but_filing_after_rejects():
    """Test 2: NEVER join by fiscal period. Period ended before t, filing
    came after -> REJECT."""
    d = ENGINE.decide_record(
        _timing(
            "q4_2017",
            event=datetime(2017, 12, 31),
            pub=datetime(2018, 2, 15),
            precision=TimePrecision.DATE,
        ),
        T,
    )
    assert not d.allowed
    assert d.code == DecisionCode.PUBLICATION_AT_OR_AFTER_AS_OF


def test_filing_before_decision_allows():
    d = ENGINE.decide_record(
        _timing(
            "q3_2017",
            event=datetime(2017, 9, 30),
            pub=datetime(2017, 11, 2),
            precision=TimePrecision.DATE,
        ),
        T,
    )
    assert d.allowed
    assert d.code == DecisionCode.ALLOWED_DATE_PRECISION


def test_date_precision_filing_not_available_on_filing_day():
    """A date-precision filing is available the NEXT day: a decision on the
    filing date itself (even at 23:59) must NOT see it."""
    d = ENGINE.decide_record(
        _timing(
            "r1",
            event=datetime(2017, 12, 31),
            pub=datetime(2018, 1, 10),
            precision=TimePrecision.DATE,
        ),
        T,  # 2018-01-10 16:00 - the filing day
    )
    assert not d.allowed
    d2 = ENGINE.decide_record(
        _timing(
            "r1",
            event=datetime(2017, 12, 31),
            pub=datetime(2018, 1, 10),
            precision=TimePrecision.DATE,
        ),
        datetime(2018, 1, 11, 0, 0, 1),
    )
    assert d2.allowed


# ------------------------------------------------------------------ vintage


def test_macro_original_value_available_before_revision():
    """Test 4: a later revision must not replace the historical value."""
    def obs(rid: str, vintage: date, value: float) -> Timing:
        return _timing(
            rid,
            event=datetime(2018, 1, 1),
            pub=datetime(vintage.year, vintage.month, vintage.day),
            precision=TimePrecision.DATE,
            vintage_date=vintage,
            policy="revised",
            domain="macro",
            kind="observation",
        )

    original = obs("jan_original", date(2018, 1, 12), 2.1)
    revision = obs("jan_revision", date(2018, 2, 1), 2.3)

    # as_of 2018-01-15: only the original is known (released 01-12, next-day)
    after_original = ENGINE.decide_record(original, datetime(2018, 1, 15))
    before_revision = ENGINE.decide_record(revision, datetime(2018, 1, 15))
    assert after_original.allowed
    assert before_revision.allowed is False  # revision not yet released

    # as_of 2018-01-10: NEITHER was public -> both unavailable (no invention)
    d_early = ENGINE.decide_record(original, datetime(2018, 1, 10))
    assert not d_early.allowed
    assert d_early.code == DecisionCode.PUBLICATION_AT_OR_AFTER_AS_OF


def test_revised_series_without_vintage_never_available():
    """Missing vintage policy: a revised series served only as the latest
    vintage cannot be made point-in-time -> rejected with NOT_POINT_IN_TIME,
    never substituted."""
    d = ENGINE.decide_record(
        _timing(
            "cpi_jan",
            event=datetime(2018, 1, 1),
            pub=datetime(2018, 1, 1),
            precision=TimePrecision.DATE,
            policy="revised",
            domain="macro",
            kind="observation",
        ),
        datetime(2020, 6, 1),
    )
    assert not d.allowed
    assert d.code == DecisionCode.NOT_POINT_IN_TIME


def test_vintage_released_after_as_of_rejects():
    """A version released after the decision is not available at it."""
    d = ENGINE.decide_record(
        _timing(
            "v1",
            event=datetime(2018, 1, 1),
            pub=datetime(2018, 2, 1),
            precision=TimePrecision.DATE,
            vintage_date=date(2018, 2, 1),
            policy="revised",
            domain="macro",
            kind="observation",
        ),
        datetime(2018, 1, 15),
    )
    assert not d.allowed
    assert d.code == DecisionCode.PUBLICATION_AT_OR_AFTER_AS_OF


def test_non_revised_series_without_vintage_is_point_in_time():
    """A series documented as never-revised IS trustworthy from the latest
    file: its observation for day D is available the day after D."""
    d = ENGINE.decide_record(
        _timing(
            "dff_d",
            event=datetime(2018, 1, 9),
            pub=datetime(2018, 1, 9),
            precision=TimePrecision.DATE,
            policy="non_revised",
            domain="macro",
            kind="observation",
        ),
        T,  # 2018-01-10
    )
    assert d.allowed
    d2 = ENGINE.decide_record(
        _timing(
            "dff_d2",
            event=datetime(2018, 1, 10),
            pub=datetime(2018, 1, 10),
            precision=TimePrecision.DATE,
            policy="non_revised",
            domain="macro",
            kind="observation",
        ),
        T,  # 2018-01-10: observation of the SAME day not yet available
    )
    assert not d2.allowed


# --------------------------------------------------------------- ingestion


def test_ingested_before_decision_published_after_rejects():
    """Test 3: ingestion before the decision does NOT make information
    available - publication decides."""
    d = ENGINE.decide_record(
        _timing(
            "r1",
            event=datetime(2018, 1, 5),
            pub=datetime(2018, 1, 11),
            ingested=datetime(2018, 1, 9),
        ),
        T,
    )
    assert not d.allowed
    assert d.code == DecisionCode.PUBLICATION_AT_OR_AFTER_AS_OF


def test_ingested_after_decision_but_public_before_allows_with_warning():
    """Delayed ingestion: the market knew it, ORBIT downloaded it late.
    Availability follows publication; ingestion is a provenance warning."""
    d = ENGINE.decide_record(
        _timing(
            "r1",
            event=datetime(2017, 12, 31),
            pub=datetime(2018, 1, 4),
            precision=TimePrecision.DATE,
            ingested=datetime(2018, 2, 1),
        ),
        T,
    )
    assert d.allowed
    assert any("ingested" in w and "AFTER" in w for w in d.warnings)


# ------------------------------------------------------------------ market


def test_market_bar_available_at_session_close_not_ts_utc():
    """The daily bar's ts_utc (session open for intraday-stamped feeds) is
    never availability: the close is known only at the session close."""
    # bar for 2020-01-02, ts_utc 14:30 UTC (= 09:30 ET open)
    bars = pl.DataFrame(
        {
            "instrument_id": ["INS-000001"],
            "symbol": ["AAPL"],
            "trade_date": [date(2020, 1, 2)],
            "ts_utc": [datetime(2020, 1, 2, 14, 30)],
            "open": [100.0], "high": [102.0], "low": [99.0],
            "close": [101.0], "volume": [1000], "adjclose": [101.0],
            "adjustment": ["split_adjusted"],
            "provider": ["yahoo_chart_api"],
            "source_uri": ["u"], "snapshot_id": ["DS-000001"],
        }
    )
    frame = market_timing_frame(bars, "DS-000001", datetime(2020, 1, 3, 12, 0))
    row = frame.row(0, named=True)
    assert row["publication_time"] == datetime(2020, 1, 2, 21, 0)  # 16:00 ET EST
    assert row["publication_time"] != datetime(2020, 1, 2, 14, 30)  # NOT ts_utc

    # a decision at 15:00 UTC on 01-02 (before the close) must not see it
    d = ENGINE.decide_record(
        Timing(
            record_id=row["record_id"], domain="market", kind="bar",
            event_time=row["event_time"],
            publication_time=row["publication_time"],
            publication_precision=TimePrecision.DATETIME,
            effective_time=row["effective_time"],
        ),
        datetime(2020, 1, 2, 15, 0),
    )
    assert not d.allowed

    # ... but a decision after the close does
    d2 = ENGINE.decide_record(
        Timing(
            record_id=row["record_id"], domain="market", kind="bar",
            event_time=row["event_time"],
            publication_time=row["publication_time"],
            publication_precision=TimePrecision.DATETIME,
            effective_time=row["effective_time"],
        ),
        datetime(2020, 1, 2, 21, 0, 1),
    )
    assert d2.allowed


def test_session_close_utc_dst_handling():
    """16:00 America/New_York is 21:00 UTC in EST and 20:00 UTC in EDT."""
    assert session_close_utc(date(2020, 1, 2)) == datetime(2020, 1, 2, 21, 0)
    assert session_close_utc(date(2020, 6, 15)) == datetime(2020, 6, 15, 20, 0)
    # spring-forward Sunday and fall-back Sunday
    assert session_close_utc(date(2020, 3, 8)) == datetime(2020, 3, 8, 20, 0)
    assert session_close_utc(date(2020, 11, 1)) == datetime(2020, 11, 1, 21, 0)


def test_market_close_boundary_est_and_edt():
    """The same 'before close / after close' behavior in both DST regimes."""
    for session, close in [
        (date(2020, 1, 2), datetime(2020, 1, 2, 21, 0)),   # EST
        (date(2020, 6, 15), datetime(2020, 6, 15, 20, 0)),  # EDT
    ]:
        timing = Timing(
            record_id="bar", domain="market", kind="bar",
            event_time=datetime(session.year, session.month, session.day),
            publication_time=close,
            publication_precision=TimePrecision.DATETIME,
            effective_time=close,
        )
        assert not ENGINE.decide_record(timing, close).allowed  # exactly at close
        before = ENGINE.decide_record(timing, close - timedelta(seconds=1))
        assert not before.allowed
        after = ENGINE.decide_record(timing, close + timedelta(seconds=1))
        assert after.allowed


def test_date_as_of_sees_only_previous_days_bars():
    """A date-only as_of (start of day) never sees the same-day bar."""
    close = session_close_utc(date(2020, 1, 7))
    timing = Timing(
        record_id="bar", domain="market", kind="bar",
        event_time=datetime(2020, 1, 7),
        publication_time=close,
        publication_precision=TimePrecision.DATETIME,
        effective_time=close,
    )
    assert not ENGINE.decide_record(timing, date(2020, 1, 7)).allowed
    assert ENGINE.decide_record(timing, date(2020, 1, 8)).allowed


# --------------------------------------------------------------- timezones


def test_tz_aware_as_of_matches_naive_utc():
    """The same instant expressed with a timezone must give the same
    decision as the naive-UTC form."""
    timing = _timing("r1", pub=datetime(2018, 1, 10, 15, 59))
    naive = ENGINE.decide_record(timing, datetime(2018, 1, 10, 16, 0))
    aware = ENGINE.decide_record(
        timing,
        datetime(2018, 1, 10, 11, 0, tzinfo=ZoneInfo("America/New_York")),
    )
    assert naive.allowed == aware.allowed
    aware_ny = ENGINE.decide_record(
        timing,
        datetime(2018, 1, 10, 16, 0, tzinfo=ZoneInfo("America/New_York")),
    )
    assert aware_ny.allowed  # 16:00 ET = 21:00 UTC, after the 15:59 UTC publication


def test_midnight_boundary():
    """Midnight is a hard boundary: publication at 23:59:59.999999 of the
    prior day is available; anything at/after midnight is not."""
    timing = _timing(
        "r1",
        event=datetime(2018, 1, 5),
        pub=datetime(2018, 1, 9, 23, 59, 59, 999999),
    )
    assert ENGINE.decide_record(timing, datetime(2018, 1, 10, 0, 0)).allowed
    assert not ENGINE.decide_record(timing, datetime(2018, 1, 9, 23, 59, 59, 999999)).allowed


# --------------------------------------------------------------- precision


def test_dst_transition_day_session_dates():
    """Session dates around the spring transition must map to the right UTC
    closes (03-08-2020 -> EDT starts, so the close shifts an hour)."""
    assert session_close_utc(date(2020, 3, 6)) == datetime(2020, 3, 6, 21, 0)  # EST
    assert session_close_utc(date(2020, 3, 9)) == datetime(2020, 3, 9, 20, 0)  # EDT
    assert session_close_utc(date(2020, 11, 2)) == datetime(2020, 11, 2, 21, 0)  # EST


def test_normalize_instant_forms_agree():
    assert normalize_instant(date(2018, 1, 10)) == datetime(2018, 1, 10, 0, 0)
    assert normalize_instant("2018-01-10T16:00:00") == datetime(2018, 1, 10, 16, 0)
    aware = datetime(2018, 1, 10, 11, 0, tzinfo=timezone.utc)
    assert normalize_instant(aware) == datetime(2018, 1, 10, 11, 0)


# ---------------------------------------------------------------- adapters


def test_sec_adapter_uses_filed_date_precision():
    facts = pl.DataFrame(
        {
            "cik": [320193, 320193],
            "entity_name": ["Apple Inc.", "Apple Inc."],
            "taxonomy": ["us-gaap", "us-gaap"],
            "fact": ["Revenues", "Assets"],
            "unit": ["USD", "USD"],
            "val": [1.0, 2.0],
            "start": ["2017-10-01", "2017-10-01"],
            "end": ["2017-12-31", "2017-12-31"],
            "accn": ["0000320193-18-000010", "0000320193-18-000010"],
            "fy": [2018, 2018],
            "fp": ["Q1", "Q1"],
            "form": ["10-Q", "10-Q"],
            "filed": ["2018-02-01", None],  # second fact has NO filing date
            "frame": ["CY2017Q4", "CY2017Q4"],
        }
    )
    frame = sec_timing_frame(facts, "DS-000009", datetime(2018, 3, 1))
    by_id = {r["record_id"]: r for r in frame.iter_rows(named=True)}
    rev = by_id["fact|DS-000009|0000320193-18-000010|Revenues|USD|2017-10-01|2017-12-31"]
    assert rev["publication_time"] == datetime(2018, 2, 1)
    assert rev["publication_precision"] == "date"
    missing = by_id["fact|DS-000009|0000320193-18-000010|Assets|USD|2017-10-01|2017-12-31"]
    assert missing["publication_time"] is None


def test_fred_adapter_v1_0_0_no_vintage_column():
    """Phase 3 v1.0.0 macro files (no vintage_date column) still adapt."""
    series = pl.DataFrame(
        {
            "series_id": ["DFF", "DFF"],
            "observation_date": [date(2018, 1, 8), date(2018, 1, 9)],
            "value": [1.40, 1.41],
            "vintage_note": ["latest_published_vintage", "latest_published_vintage"],
            "provider": ["fred_csv", "fred_csv"],
            "snapshot_id": ["DS-000003", "DS-000003"],
        }
    )
    frame = fred_timing_frame(
        series, "DS-000003",
        series_policies={"DFF": "non_revised", "CPIAUCSL": "revised"},
        default_policy="revised",
    )
    rows = {r["record_id"]: r for r in frame.iter_rows(named=True)}
    dff = rows["obs|DS-000003|DFF|2018-01-08|latest"]
    assert dff["vintage_date"] is None
    assert dff["series_policy"] == "non_revised"
    assert dff["publication_time"] == datetime(2018, 1, 8)
    assert dff["publication_precision"] == "date"


def test_fred_adapter_v1_1_0_vintage_rows():
    """v1.1.0 files carry vintage_date; those rows are vintage-point-in-time."""
    series = pl.DataFrame(
        {
            "series_id": ["CPIAUCSL", "CPIAUCSL"],
            "observation_date": [date(2018, 1, 1), date(2018, 1, 1)],
            "value": [2.1, 2.3],
            "vintage_date": [date(2018, 1, 12), date(2018, 2, 1)],
            "vintage_note": ["alfred_vintage_requested", "alfred_vintage_requested"],
            "provider": ["fred_csv", "fred_csv"],
            "snapshot_id": ["DS-000010", "DS-000011"],
        }
    )
    frame = fred_timing_frame(series, "DS-000010", series_policies={"CPIAUCSL": "revised"})
    rows = {r["record_id"]: r for r in frame.iter_rows(named=True)}
    original = rows["obs|DS-000010|CPIAUCSL|2018-01-01|2018-01-12"]
    assert original["vintage_date"] == date(2018, 1, 12)
    assert original["publication_time"] == datetime(2018, 1, 12)
    assert original["series_policy"] == "revised"  # vintages override policy


# --------------------------------------------- audit-4 regression findings


def test_series_policy_casing_cannot_admit_revised_data():
    """Audit 4 finding 3: a case-variant policy (e.g. 'Revised') must still
    be treated as revised - no casing accident admits revised data."""
    d = ENGINE.decide_record(
        Timing(
            record_id="r1",
            domain="macro",
            kind="observation",
            event_time=datetime(2018, 1, 1),
            publication_time=datetime(2018, 1, 12),
            publication_precision=TimePrecision.DATE,
            vintage_date=None,
            series_policy="Revised",
        ),
        datetime(2018, 1, 15),
    )
    assert not d.allowed
    assert d.code == DecisionCode.NOT_POINT_IN_TIME


def test_timing_normalizes_string_instants():
    """Audit 4 finding 7: Timing normalizes ISO strings/date values at
    construction, so a comparison can never receive a string."""
    timing = Timing(
        record_id="r1",
        domain="fundamentals",
        kind="fact",
        event_time="2018-01-01T00:00:00",
        publication_time="2018-01-08T15:00:00",
        publication_precision=TimePrecision.DATETIME,
    )
    assert isinstance(timing.publication_time, datetime)
    d = ENGINE.decide_record(timing, T)
    assert d.allowed


def test_market_payload_excludes_retroactive_adjclose():
    """Audit 4 finding 4: adjclose is retroactively adjusted and must not
    ride inside the allowed information set's payload."""
    bars = pl.DataFrame(
        {
            "instrument_id": ["INS-000001"],
            "trade_date": [date(2018, 1, 9)],
            "ts_utc": [datetime(2018, 1, 9, 14, 30)],
            "open": [100.0], "high": [102.0], "low": [99.0], "close": [101.0],
            "volume": [1000], "adjclose": [95.0],  # retroactively adjusted
        }
    )
    frame = market_timing_frame(bars, "DS-000001")
    for r in frame.iter_rows(named=True):
        assert "adjclose" not in json.loads(r["payload_json"])
        assert json.loads(r["payload_json"])["close"] == 101.0