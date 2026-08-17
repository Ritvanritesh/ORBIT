"""Synthetic future-leak fixtures - permanent adversarial regression inputs.

Each fixture represents a KNOWN leakage pattern. The Temporal Truth Engine
must reject the planted future information; if someone later changes the
temporal engine and accidentally allows any of these, the regression tests
catch it. A single surviving leak is a hard stop for Phase 4.

Fixture inventory (prompt section 15):
    future_earnings_fixture        earnings fact published after as_of
    future_macro_revision_fixture  a revision released after as_of must not
                                   replace the historical vintage
    future_filing_fixture          fiscal period before as_of, filing after
    future_price_fixture           a bar whose session has not completed
    future_feature_fixture         a feature referencing a future row
    delayed_ingestion_fixture      public before as_of, ingested after
    missing_publication_fixture    no publication time at all
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import polars as pl

from orbit.temporal.rules import AvailabilityDecision
from orbit.temporal.times import DecisionCode, TimePrecision, Timing

# Fixed reference instants (all naive UTC; 2018-01-10 16:00 is the running
# example of the Phase 4 prompt).
T_2018_01_10_16 = datetime(2018, 1, 10, 16, 0, 0)


def _dt(y: int, m: int, d: int, h: int = 0, minute: int = 0) -> datetime:
    return datetime(y, m, d, h, minute)


def _fact_timing(
    record_id: str,
    *,
    period_end: date,
    filed: date | None,
    ingested: datetime | None = None,
    kind: str = "fact",
) -> Timing:
    return Timing(
        record_id=record_id,
        domain="fundamentals",
        kind=kind,
        event_time=_dt(period_end.year, period_end.month, period_end.day),
        publication_time=_dt(filed.year, filed.month, filed.day) if filed else None,
        publication_precision=TimePrecision.DATE,
        effective_time=_dt(filed.year, filed.month, filed.day) if filed else None,
        ingestion_time=ingested,
        payload={"cik": 320193, "fact": "Revenues", "val": 84_000_000_000.0},
    )


@dataclass(frozen=True)
class LeakFixture:
    """One adversarial scenario: records + decision time + expected outcome."""

    name: str
    as_of: datetime
    timings: list[Timing]
    expected_allowed: list[str]
    expected_rejected: list[str]
    expected_code: dict[str, DecisionCode] | None = None
    description: str = ""


def future_earnings_fixture() -> LeakFixture:
    """Q1 2018 earnings (period end 2018-03-31) filed AFTER the decision."""
    return LeakFixture(
        name="future_earnings",
        as_of=T_2018_01_10_16,
        timings=[
            _fact_timing(
                "earnings_Q1_2018",
                period_end=date(2018, 3, 31),
                filed=date(2018, 4, 25),
            ),
            _fact_timing(
                "earnings_Q4_2017_ok",
                period_end=date(2017, 12, 31),
                filed=date(2018, 1, 4),
            ),
        ],
        expected_allowed=["earnings_Q4_2017_ok"],
        expected_rejected=["earnings_Q1_2018"],
        expected_code={
            "earnings_Q1_2018": DecisionCode.PUBLICATION_AT_OR_AFTER_AS_OF,
            "earnings_Q4_2017_ok": DecisionCode.ALLOWED_DATE_PRECISION,
        },
        description=(
            "the classic leak: the fact describes a past period but was "
            "filed after the decision - it must be rejected"
        ),
    )


def future_filing_fixture() -> LeakFixture:
    """Fiscal period ended BEFORE the decision, but the filing came after."""
    return LeakFixture(
        name="future_filing",
        as_of=T_2018_01_10_16,
        timings=[
            _fact_timing(
                "filing_after_decision",
                period_end=date(2017, 12, 31),
                filed=date(2018, 2, 15),
            ),
            _fact_timing(
                "filing_before_decision",
                period_end=date(2017, 9, 30),
                filed=date(2017, 11, 2),
            ),
        ],
        expected_allowed=["filing_before_decision"],
        expected_rejected=["filing_after_decision"],
        expected_code={
            "filing_after_decision": DecisionCode.PUBLICATION_AT_OR_AFTER_AS_OF,
            "filing_before_decision": DecisionCode.ALLOWED_DATE_PRECISION,
        },
        description=(
            "fiscal-period joins are forbidden: the period ended before the "
            "decision but the filing was not public - reject"
        ),
    )


def future_macro_revision_fixture() -> LeakFixture:
    """January 2018 macro observation: original release 2.1%, later revision
    2.3% released 2018-02-01. A decision on 2018-01-10 must see the ORIGINAL
    value only - and there is no version released before 2018-01-10, so the
    observation is unavailable there (no invented availability)."""
    def obs(record_id: str, vintage: date, value: float) -> Timing:
        return Timing(
            record_id=record_id,
            domain="macro",
            kind="observation",
            event_time=_dt(2018, 1, 1),
            publication_time=_dt(vintage.year, vintage.month, vintage.day),
            publication_precision=TimePrecision.DATE,
            effective_time=_dt(vintage.year, vintage.month, vintage.day),
            vintage_id=f"vintage_{vintage.isoformat()}",
            vintage_date=vintage,
            series_policy="revised",
            payload={"series_id": "CPIAUCSL", "observation_date": "2018-01-01", "value": value},
        )

    return LeakFixture(
        name="future_macro_revision",
        as_of=T_2018_01_10_16,
        timings=[
            obs("jan2018_original", date(2018, 1, 12), 2.1),
            obs("jan2018_revision", date(2018, 2, 1), 2.3),
        ],
        expected_allowed=[],
        expected_rejected=["jan2018_original", "jan2018_revision"],
        expected_code={
            "jan2018_original": DecisionCode.PUBLICATION_AT_OR_AFTER_AS_OF,
            "jan2018_revision": DecisionCode.PUBLICATION_AT_OR_AFTER_AS_OF,
        },
        description=(
            "NEITHER version was public at 2018-01-10: the revision must not "
            "replace history, and the original was not yet released either"
        ),
    )


def future_price_fixture() -> LeakFixture:
    """Bars around the decision: the same-day bar (session not complete) and
    a future bar must be rejected; prior completed bars allowed."""
    def bar(record_id: str, session: date) -> Timing:
        return Timing(
            record_id=record_id,
            domain="market",
            kind="bar",
            event_time=_dt(session.year, session.month, session.day),
            publication_time=_dt(session.year, session.month, session.day, 21, 0),
            publication_precision=TimePrecision.DATETIME,
            effective_time=_dt(session.year, session.month, session.day, 21, 0),
            payload={"instrument_id": "INS-000001", "trade_date": session.isoformat(), "close": 1.0},
        )

    return LeakFixture(
        name="future_price",
        as_of=T_2018_01_10_16,  # 2018-01-10 16:00 UTC; NY close 21:00 UTC
        timings=[
            bar("bar_2018_01_09", date(2018, 1, 9)),   # close 21:00 UTC 01-09 < 16:00 01-10
            bar("bar_2018_01_10", date(2018, 1, 10)),  # closes 21:00 UTC, AFTER as_of
            bar("bar_2018_01_11", date(2018, 1, 11)),  # future session
        ],
        expected_allowed=["bar_2018_01_09"],
        expected_rejected=["bar_2018_01_10", "bar_2018_01_11"],
        expected_code={
            "bar_2018_01_10": DecisionCode.PUBLICATION_AT_OR_AFTER_AS_OF,
            "bar_2018_01_11": DecisionCode.PUBLICATION_AT_OR_AFTER_AS_OF,
        },
        description=(
            "today's final close is not available before the close; a bar is "
            "available only after its session close"
        ),
    )


def delayed_ingestion_fixture() -> LeakFixture:
    """The information was PUBLIC before as_of but ORBIT downloaded it
    later. Availability follows publication; ingestion is a warning."""
    return LeakFixture(
        name="delayed_ingestion",
        as_of=T_2018_01_10_16,
        timings=[
            _fact_timing(
                "filing_public_before_ingested_after",
                period_end=date(2017, 9, 30),
                filed=date(2017, 11, 2),
                ingested=_dt(2018, 2, 1, 18, 0),
            ),
        ],
        expected_allowed=["filing_public_before_ingested_after"],
        expected_rejected=[],
        description=(
            "ingestion_time is provenance, not availability: a late download "
            "of an old-publication record is still known history"
        ),
    )


def missing_publication_fixture() -> LeakFixture:
    """No publication time at all: availability must never be invented."""
    return LeakFixture(
        name="missing_publication",
        as_of=T_2018_01_10_16,
        timings=[
            Timing(
                record_id="no_publication_fact",
                domain="fundamentals",
                kind="fact",
                event_time=_dt(2017, 12, 31),
                publication_time=None,
                publication_precision=TimePrecision.DATE,
                payload={"cik": 320193, "fact": "Assets", "val": 1.0},
            ),
        ],
        expected_allowed=[],
        expected_rejected=["no_publication_fact"],
        expected_code={
            "no_publication_fact": DecisionCode.MISSING_PUBLICATION_TIME,
        },
        description=(
            "a record with an unknown publication time is NEVER available - "
            "do not treat missing timestamps as available"
        ),
    )


def future_feature_fixture() -> dict[str, Any]:
    """A feature frame that references a future bar (the future-price leak
    in feature form). The feature-time rules must flag it."""
    bars = pl.DataFrame(
        {
            "record_id": [f"bar_{d}" for d in ("2018_01_08", "2018_01_09", "2018_01_10")],
            "trade_date": [date(2018, 1, 8), date(2018, 1, 9), date(2018, 1, 10)],
            "close": [100.0, 101.0, 102.0],
            "feature_time": [T_2018_01_10_16] * 3,
        }
    )
    return {
        "name": "future_feature",
        "as_of": T_2018_01_10_16,
        "bars": bars,
        "expected_violations": ["bar_2018_01_10"],
    }


ALL_LEAK_FIXTURES: list[LeakFixture] = [
    future_earnings_fixture(),
    future_filing_fixture(),
    future_macro_revision_fixture(),
    future_price_fixture(),
    delayed_ingestion_fixture(),
    missing_publication_fixture(),
]


def run_fixture(fixture: LeakFixture, engine: Any) -> list[AvailabilityDecision]:
    """Evaluate every record of a fixture with the engine's decide_record."""
    return [engine.decide_record(t, fixture.as_of) for t in fixture.timings]