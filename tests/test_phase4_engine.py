"""Phase 4 engine tests: as-of snapshots, vintages, joins, reproducibility,
and provenance, all hermetic (parquet written to a temp data root).

The engine is tested against the SAME normalized shapes Phase 3 produces,
so passing these tests means the real DS-000001/2/3 artifacts work too.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

import polars as pl
import pytest

from orbit.ingestion.normalizers.market import normalize_market_bars
from orbit.ingestion.parsing import parse_yahoo_chart
from orbit.temporal.adapters import (
    fred_timing_frame,
    market_timing_frame,
    sec_timing_frame,
)
from orbit.temporal.engine import (
    TemporalTruthEngine,
    build_temporal_source,
)
from orbit.temporal.snapshot import PointInTimeSnapshot, TemporalSource
from orbit.temporal.times import DecisionCode, TimePrecision, Timing


def _yahoo_payload():
    import json as _json

    ts = [
        int((datetime(2020, 1, 2, 14, 30) - datetime(1970, 1, 1)).total_seconds()),
        int((datetime(2020, 1, 3, 14, 30) - datetime(1970, 1, 1)).total_seconds()),
        int((datetime(2020, 1, 6, 14, 30) - datetime(1970, 1, 1)).total_seconds()),
        int((datetime(2020, 1, 7, 14, 30) - datetime(1970, 1, 1)).total_seconds()),
    ]
    return _json.dumps(
        {
            "chart": {
                "result": [
                    {
                        "meta": {"currency": "USD", "exchangeName": "NMS"},
                        "timestamp": ts,
                        "indicators": {
                            "quote": [{
                                "open": [100.0, 101.0, 99.0, 102.0],
                                "high": [102.0, 103.0, 100.5, 103.0],
                                "low": [99.0, 100.0, 98.0, 101.0],
                                "close": [101.0, 101.5, 100.5, 102.5],
                                "volume": [1000, 1100, 900, 1200],
                            }],
                            "adjclose": [{"adjclose": [100.0, 100.5, 99.5, 101.5]}],
                        },
                        "events": {"dividends": {}, "splits": {}},
                    }
                ],
                "error": None,
            }
        }
    ).encode()


def _bars_frame() -> pl.DataFrame:
    bars, events = parse_yahoo_chart(_yahoo_payload(), "AAPL")
    normalized = normalize_market_bars(
        {"AAPL": {"bars": bars, "events": events}}, {"AAPL": "INS-000001"},
        "yahoo_chart_api", "https://example/chart/AAPL", "DS-000001",
    )
    return normalized["bars"]


def _facts_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "cik": [320193, 320193],
            "entity_name": ["Apple Inc.", "Apple Inc."],
            "taxonomy": ["us-gaap", "us-gaap"],
            "fact": ["Revenues", "Assets"],
            "unit": ["USD", "USD"],
            "val": [111_439_000_000.0, 354_000_000_000.0],
            "start": ["2019-09-29", "2019-09-29"],
            "end": ["2019-12-28", "2019-12-28"],
            "accn": ["0000320193-20-000010", "0000320193-20-000010"],
            "fy": [2020, 2020],
            "fp": ["Q1", "Q1"],
            "form": ["10-Q", "10-Q"],
            "filed": ["2020-01-29", "2020-01-29"],
            "frame": ["CY2019Q4", "CY2019Q4"],
        }
    )


def _fred_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "series_id": ["DFF", "DFF", "CPIAUCSL"],
            "observation_date": [date(2020, 1, 2), date(2020, 1, 3), date(2020, 1, 1)],
            "value": [1.55, 1.56, 2.5],
            "vintage_note": ["latest_published_vintage"] * 3,
            "provider": ["fred_csv"] * 3,
            "snapshot_id": ["DS-000003"] * 3,
        }
    )


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    return tmp_path


def _write_market(data_root: Path) -> Path:
    d = data_root / "normalized" / "market" / "yahoo_chart_api" / "DS-000001"
    d.mkdir(parents=True, exist_ok=True)
    p = d / "bars.parquet"
    _bars_frame().write_parquet(p)
    return p


def _write_sec(data_root: Path) -> Path:
    d = data_root / "normalized" / "fundamentals" / "sec_edgar_companyfacts" / "DS-000002"
    d.mkdir(parents=True, exist_ok=True)
    p = d / "facts.parquet"
    _facts_frame().write_parquet(p)
    return p


def _write_macro(data_root: Path) -> Path:
    d = data_root / "normalized" / "macro" / "fred_csv" / "DS-000003"
    d.mkdir(parents=True, exist_ok=True)
    p = d / "series.parquet"
    _fred_frame().write_parquet(p)
    return p


def _sources(data_root: Path, ingest_time: datetime | None = None) -> list[TemporalSource]:
    return [
        TemporalSource(
            snapshot_id="DS-000001", domain="market", provider="yahoo_chart_api",
            checksum="a" * 64, manifest_path="m1.json", ingest_time=ingest_time,
            artifact_paths=[str(_write_market(data_root))],
        ),
        TemporalSource(
            snapshot_id="DS-000002", domain="sec", provider="sec_edgar_companyfacts",
            checksum="b" * 64, manifest_path="m2.json", ingest_time=ingest_time,
            artifact_paths=[str(_write_sec(data_root))],
        ),
        TemporalSource(
            snapshot_id="DS-000003", domain="macro", provider="fred_csv",
            checksum="c" * 64, manifest_path="m3.json", ingest_time=ingest_time,
            artifact_paths=[str(_write_macro(data_root))],
        ),
    ]


# ------------------------------------------------------------ as-of snapshot


def test_snapshot_information_set_at_as_of(data_root):
    engine = TemporalTruthEngine(sources=_sources(data_root))
    # 2020-01-08 16:00 UTC: after the 01-07 close (21:00 UTC), after the
    # filings (filed 2020-01-29 -> NOT yet available), DFF known through
    # 01-07... but 01-02/01-03 are the observations here; CPIAUCSL revised.
    snap = engine.snapshot(datetime(2020, 1, 8, 16, 0))
    assert isinstance(snap, PointInTimeSnapshot)
    assert snap.as_of_time == datetime(2020, 1, 8, 16, 0)

    # all 4 bars allowed (01-02, 01-03, 01-06, 01-07 all closed before as_of)
    bars = snap.records.filter(pl.col("kind") == "bar")
    assert bars.height == 4
    dates = bars["payload_json"].map_elements(
        lambda j: json.loads(j)["trade_date"], return_dtype=pl.Utf8
    ).to_list()
    assert dates == sorted(dates)

    # filings are NOT available (filed 2020-01-29 > as_of)
    facts = snap.records.filter(pl.col("kind") == "fact")
    assert facts.height == 0
    assert snap.excluded.filter(pl.col("kind") == "fact").height == 2

    # DFF (non_revised): 01-02 and 01-03 observations available by 01-08
    macro = snap.records.filter(pl.col("kind") == "observation")
    assert macro.height == 2

    # CPIAUCSL (revised, no vintage): excluded as NOT_POINT_IN_TIME and
    # reported as a limitation (never silently substituted)
    codes = set(snap.excluded["decision_code"].to_list())
    assert DecisionCode.NOT_POINT_IN_TIME.value in codes
    assert any("CPIAUCSL" in l for l in snap.limitations)


def test_snapshot_rejects_same_day_bar_and_future_facts(data_root):
    engine = TemporalTruthEngine(sources=_sources(data_root))
    snap = engine.snapshot(datetime(2020, 1, 7, 15, 0))  # before 01-07 close
    bars = snap.records.filter(pl.col("kind") == "bar")
    assert bars.height == 3  # 01-02, 01-03, 01-06; NOT 01-07
    assert snap.excluded.filter(pl.col("kind") == "bar").height == 1
    rej = snap.excluded.filter(pl.col("kind") == "bar").row(0, named=True)
    assert rej["decision_code"] == DecisionCode.PUBLICATION_AT_OR_AFTER_AS_OF.value


def test_date_precision_filing_available_next_day(data_root):
    engine = TemporalTruthEngine(sources=_sources(data_root))
    # filings filed 2020-01-29: available from 2020-01-30 00:00 UTC
    before = engine.snapshot(datetime(2020, 1, 29, 23, 59, 59))
    assert before.records.filter(pl.col("kind") == "fact").height == 0
    at = engine.snapshot(datetime(2020, 1, 30, 0, 0, 1))
    assert at.records.filter(pl.col("kind") == "fact").height == 2


def test_delayed_ingestion_warning_does_not_exclude(data_root):
    engine = TemporalTruthEngine(
        sources=_sources(data_root, ingest_time=datetime(2020, 2, 15))
    )
    snap = engine.snapshot(datetime(2020, 2, 1))
    facts = snap.records.filter(pl.col("kind") == "fact")
    assert facts.height == 2
    warned = facts.filter(pl.col("warn_ingested_after_as_of")).height
    assert warned == 2  # public before as_of but downloaded after: a warning


def test_ingestion_never_gates_availability(data_root):
    """A snapshot downloaded AFTER as_of still contributes its OLD
    publication data; nothing newer is invented."""
    engine = TemporalTruthEngine(
        sources=_sources(data_root, ingest_time=datetime(2020, 3, 1))
    )
    snap = engine.snapshot(datetime(2020, 1, 8, 16, 0))
    assert snap.records.filter(pl.col("kind") == "bar").height == 4
    # no bar after 01-07 sneaks in just because ingestion happened later
    assert snap.records.filter(pl.col("kind") == "bar")["event_time"].max() == datetime(2020, 1, 7)


# -------------------------------------------------------------- vintages


def _vintage_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "series_id": ["CPIAUCSL", "CPIAUCSL", "CPIAUCSL"],
            "observation_date": [date(2018, 1, 1), date(2018, 1, 1), date(2018, 2, 1)],
            "value": [2.1, 2.3, 2.2],
            "vintage_date": [date(2018, 1, 12), date(2018, 2, 1), date(2018, 3, 1)],
            "vintage_note": ["alfred"] * 3,
            "provider": ["fred_csv"] * 3,
            "snapshot_id": ["DS-000020", "DS-000021", "DS-000022"],
        }
    )


def test_historical_vintage_retrieval_uses_version_known_at_t():
    """Test 4: the revision (2.3, released 2018-02-01) must not replace the
    original (2.1, released 2018-01-12) for a decision before the revision."""
    frame = fred_timing_frame(
        _vintage_frame(), "DS-000020", series_policies={"CPIAUCSL": "revised"}
    )
    engine = TemporalTruthEngine()

    as_of_jan = engine.historical_vintage(frame, datetime(2018, 1, 15))
    jan = as_of_jan.filter(pl.col("source_key") == "CPIAUCSL")
    rows = {json.loads(r["payload_json"])["observation_date"]: json.loads(r["payload_json"])["value"]
            for r in jan.iter_rows(named=True)}
    assert rows["2018-01-01"] == 2.1  # the ORIGINAL, not the revision

    as_of_mar = engine.historical_vintage(frame, datetime(2018, 3, 5))
    mar = as_of_mar.filter(pl.col("source_key") == "CPIAUCSL")
    rows2 = {json.loads(r["payload_json"])["observation_date"]: json.loads(r["payload_json"])["value"]
             for r in mar.iter_rows(named=True)}
    assert rows2["2018-01-01"] == 2.3  # by March the revision IS the known value
    assert rows2["2018-02-01"] == 2.2


def test_observation_without_released_version_is_unavailable():
    """2018-02 observation's only vintage (2018-03-01) is after the as_of:
    no version exists yet -> observation unavailable, never approximated."""
    frame = fred_timing_frame(
        _vintage_frame(), "DS-000020", series_policies={"CPIAUCSL": "revised"}
    )
    engine = TemporalTruthEngine()
    as_of_feb = engine.historical_vintage(frame, datetime(2018, 2, 15))
    obs = {json.loads(r["payload_json"])["observation_date"] for r in as_of_feb.iter_rows(named=True)}
    assert obs == {"2018-01-01"}


def test_superseded_versions_are_audited_not_dropped():
    frame = fred_timing_frame(
        _vintage_frame(), "DS-000020", series_policies={"CPIAUCSL": "revised"}
    )
    engine = TemporalTruthEngine()
    # the snapshot resolve path must audit the superseded original version
    # instead of silently dropping it
    resolved = engine._resolve_vintages(engine.evaluate(frame, datetime(2018, 3, 5)).frame)
    superseded = resolved.filter(pl.col("decision_code") == DecisionCode.VINTAGE_SUPERSEDED.value)
    assert superseded.height == 1
    assert json.loads(superseded.row(0, named=True)["payload_json"])["value"] == 2.1
    allowed = resolved.filter(pl.col("allowed"))
    jan = allowed.filter(pl.col("event_time") == datetime(2018, 1, 1))
    assert json.loads(jan.row(0, named=True)["payload_json"])["value"] == 2.3


# ------------------------------------------------------- latest available


def test_latest_available_picks_latest_publication_per_key():
    facts = _facts_frame()
    frame = sec_timing_frame(facts, "DS-000002", datetime(2020, 2, 1))
    engine = TemporalTruthEngine()
    latest = engine.latest_available(frame, datetime(2020, 2, 5), keys=["source_key"])
    # both facts are for the same filing/cik, so the latest per cik is 1 row;
    # both published at the same instant -> deterministic tie-break by the
    # largest record_id ("Revenues" > "Assets")
    assert latest.height == 1
    row = json.loads(latest.row(0, named=True)["payload_json"])
    assert row["fact"] == "Revenues"


def test_duplicate_releases_tie_breaks_deterministically():
    """Two versions released on the SAME instant: the tie is broken by the
    largest record_id, deterministically (never random)."""
    facts = pl.DataFrame(
        {
            "cik": [320193, 320193],
            "entity_name": ["Apple Inc.", "Apple Inc."],
            "taxonomy": ["us-gaap", "us-gaap"],
            "fact": ["Revenues", "Revenues"],
            "unit": ["USD", "USD"],
            "val": [100.0, 101.0],
            "start": ["2019-09-29", "2019-09-29"],
            "end": ["2019-12-28", "2019-12-28"],
            "accn": ["0000320193-20-000010", "0000320193-20-000011"],
            "fy": [2020, 2020],
            "fp": ["Q1", "Q1"],
            "form": ["10-Q", "10-Q"],
            "filed": ["2020-01-29", "2020-01-29"],
            "frame": ["CY2019Q4", "CY2019Q4"],
        }
    )
    frame = sec_timing_frame(facts, "DS-000002", datetime(2020, 2, 1))
    engine = TemporalTruthEngine()
    latest = engine.latest_available(frame, datetime(2020, 2, 5), keys=["source_key"])
    assert latest.height == 1
    chosen = json.loads(latest.row(0, named=True)["payload_json"])
    assert chosen["accn"] == "0000320193-20-000011"  # largest record_id wins
    again = engine.latest_available(frame, datetime(2020, 2, 5), keys=["source_key"])
    assert again.equals(latest)


# -------------------------------------------------------------- asof joins


def _right_frame() -> pl.DataFrame:
    """Mixed right side: filings + vintages + delayed ingestion + missing
    publication - the full §16 matrix."""
    return pl.concat(
        [
            sec_timing_frame(_facts_frame(), "DS-000002", datetime(2020, 2, 1)),
            fred_timing_frame(
                _vintage_frame(), "DS-000020", series_policies={"CPIAUCSL": "revised"}
            ),
        ]
    )


def test_asof_join_attaches_latest_observation_available_at_t():
    right = _right_frame()
    left = pl.DataFrame(
        {
            "record_id": ["L1", "L2", "L3"],
            "decision_time": [
                datetime(2018, 1, 10),   # before any vintage release
                datetime(2018, 2, 15),   # original CPIAUCSL version known
                datetime(2020, 2, 5),    # filings + everything known
            ],
        }
    )
    engine = TemporalTruthEngine()
    joined = engine.asof_join(right, left)

    l1 = joined.filter(pl.col("record_id") == "L1")
    assert l1.height == 1  # no right record available at 2018-01-10 -> null join
    assert l1["event_time"][0] is None

    l2 = joined.filter(pl.col("record_id") == "L2")
    assert l2.height == 1
    l2row = l2.row(0, named=True)
    # by 2018-02-15 the revision (released 2018-02-01, available 2018-02-02)
    # IS the known value; the original is no longer the information set
    assert json.loads(l2row["payload_json"])["value"] == 2.3
    assert l2row["vintage_date"] == date(2018, 2, 1)

    l3 = joined.filter(pl.col("record_id") == "L3")
    assert l3.height == 1
    l3row = l3.row(0, named=True)
    # filings (filed 2020-01-29) are available; among available records the
    # most recent OBSERVATION is the 2019-12-28 period end; both facts tie on
    # event_time -> deterministic tie-break by largest record_id ("Revenues")
    assert json.loads(l3row["payload_json"])["fact"] == "Revenues"


def test_asof_join_gates_missing_publication_and_future():
    right = pl.concat(
        [
            sec_timing_frame(_facts_frame(), "DS-000002", datetime(2020, 2, 1)),
        ]
    )
    # a left decision BEFORE the filing publication -> nothing to join
    left = pl.DataFrame(
        {
            "record_id": ["L_early"],
            "decision_time": [datetime(2020, 1, 10)],
        }
    )
    engine = TemporalTruthEngine()
    joined = engine.asof_join(right, left)
    assert joined["event_time"][0] is None

    # same-day publication (filed 2020-01-29, date precision): a decision at
    # 2020-01-29 23:59 still must not join it (next-day convention)
    left2 = pl.DataFrame(
        {
            "record_id": ["L_filing_day"],
            "decision_time": [datetime(2020, 1, 29, 23, 59, 59)],
        }
    )
    joined2 = engine.asof_join(right, left2)
    assert joined2["event_time"][0] is None
    left3 = pl.DataFrame(
        {
            "record_id": ["L_after"],
            "decision_time": [datetime(2020, 1, 30, 0, 0, 1)],
        }
    )
    joined3 = engine.asof_join(right, left3)
    assert joined3["event_time"][0] == datetime(2019, 12, 28)


def test_asof_join_delayed_ingestion_does_not_block_join():
    """The §16 'delayed ingestion' case: public long ago, ingested late -
    the join still attaches it (availability follows publication)."""
    facts = _facts_frame()
    frame = sec_timing_frame(facts, "DS-000002", datetime(2020, 5, 1))
    left = pl.DataFrame(
        {
            "record_id": ["L"],
            "decision_time": [datetime(2020, 3, 1)],
        }
    )
    engine = TemporalTruthEngine()
    joined = engine.asof_join(frame, left)
    assert joined["event_time"][0] == datetime(2019, 12, 28)


def test_asof_join_multiple_versions_uses_revision():
    """§16 'revisions' + 'multiple versions': with two released versions the
    join uses the value known at t (the latest released BEFORE t)."""
    frame = fred_timing_frame(
        _vintage_frame(), "DS-000020", series_policies={"CPIAUCSL": "revised"}
    )
    left = pl.DataFrame(
        {
            "record_id": ["L"],
            "decision_time": [datetime(2018, 2, 15)],
        }
    )
    engine = TemporalTruthEngine()
    joined = engine.asof_join(frame, left)
    row = joined.row(0, named=True)
    # at 2018-02-15 the latest released version of the Jan observation is 2.3
    # (vintage 2018-02-01, available 2018-02-02); the 2018-02 observation has
    # no released version yet (vintage 2018-03-01) and is unavailable
    assert json.loads(row["payload_json"])["value"] == 2.3
    assert row["vintage_date"] == date(2018, 2, 1)


def test_asof_join_null_decision_time_null_joins():
    """A left row with no decision instant has no moment to be asked about:
    it must null-join (never crash, never attach a record)."""
    frame = fred_timing_frame(
        _vintage_frame(), "DS-000020", series_policies={"CPIAUCSL": "revised"}
    )
    left = pl.DataFrame(
        {
            "record_id": ["L_null", "L_ok"],
            "decision_time": [None, datetime(2018, 2, 15)],
        }
    )
    engine = TemporalTruthEngine()
    joined = engine.asof_join(frame, left)
    by_id = {r["record_id"]: r["event_time"] for r in joined.iter_rows(named=True)}
    assert by_id["L_null"] is None
    assert by_id["L_ok"] == datetime(2018, 1, 1)


def test_asof_join_left_event_time_collision_does_not_gate():
    """A left frame carrying its OWN event_time column must not shadow the
    right side's event_time in the <= t filter (audit 4 finding 1)."""
    frame = fred_timing_frame(
        _vintage_frame(), "DS-000020", series_policies={"CPIAUCSL": "revised"}
    )
    left = pl.DataFrame(
        {
            "record_id": ["L"],
            "decision_time": [datetime(2020, 2, 5)],
            "event_time": [datetime(2025, 1, 1)],  # far future, must be ignored
        }
    )
    engine = TemporalTruthEngine()
    joined = engine.asof_join(frame, left)
    assert joined.height == 1
    # the left's own event_time (2025) must never gate or shadow the join;
    # the right side's observation is attached under its renamed column
    assert joined["event_time"][0] == datetime(2025, 1, 1)  # left column intact
    assert joined["right_event_time"][0] == datetime(2018, 2, 1)  # right obs


def test_asof_join_null_decision_time_with_colliding_columns():
    """Colliding left columns + a null decision_time row must not crash
    (audit 4 finding 2): the null row joins with null right columns."""
    frame = fred_timing_frame(
        _vintage_frame(), "DS-000020", series_policies={"CPIAUCSL": "revised"}
    )
    left = pl.DataFrame(
        {
            "record_id": ["L_null", "L_ok"],
            "decision_time": [None, datetime(2018, 2, 15)],
            "publication_time": [datetime(2018, 1, 1), datetime(2018, 1, 1)],
        }
    )
    engine = TemporalTruthEngine()
    joined = engine.asof_join(frame, left)
    assert joined.height == 2
    by_id = {r["record_id"]: r["event_time"] for r in joined.iter_rows(named=True)}
    assert by_id["L_null"] is None
    assert by_id["L_ok"] == datetime(2018, 1, 1)


def test_asof_join_duplicate_left_record_ids_all_kept():
    """Two left rows sharing a record_id must both be joined (one right
    record per LEFT ROW, audit 4 finding 5)."""
    frame = fred_timing_frame(
        _vintage_frame(), "DS-000020", series_policies={"CPIAUCSL": "revised"}
    )
    left = pl.DataFrame(
        {
            "record_id": ["dup", "dup"],
            "decision_time": [datetime(2018, 2, 15), datetime(2018, 2, 15)],
        }
    )
    engine = TemporalTruthEngine()
    joined = engine.asof_join(frame, left)
    assert joined.height == 2
    assert all(json.loads(r["payload_json"])["value"] == 2.3 for r in joined.iter_rows(named=True))


def test_asof_join_left_colliding_with_decision_columns():
    """A left frame carrying its OWN allowed/decision_code columns must not
    crash (revision review R1): the right side's decision columns are
    renamed, and null joins keep identical schemas."""
    frame = fred_timing_frame(
        _vintage_frame(), "DS-000020", series_policies={"CPIAUCSL": "revised"}
    )
    left = pl.DataFrame(
        {
            "record_id": ["L1", "L2"],
            "decision_time": [datetime(2018, 2, 15), None],
            "allowed": [True, False],
            "decision_code": ["x", "y"],
        }
    )
    engine = TemporalTruthEngine()
    joined = engine.asof_join(frame, left)
    assert joined.height == 2
    by_id = {r["record_id"]: r for r in joined.iter_rows(named=True)}
    assert by_id["L1"]["event_time"] == datetime(2018, 1, 1)  # right's (no collision)
    assert by_id["L1"]["allowed"] is True  # left column untouched
    assert by_id["L2"]["event_time"] is None
    assert by_id["L2"]["right_allowed"] is None
    assert by_id["L2"]["right_decision_code"] is None


def test_asof_join_left_without_record_id_column():
    """An arbitrary signal frame (no record_id of its own) must join cleanly
    (revision review R4): the right side's record_id stays unrenamed and the
    tie-break skips it."""
    frame = fred_timing_frame(
        _vintage_frame(), "DS-000020", series_policies={"CPIAUCSL": "revised"}
    )
    left = pl.DataFrame({"decision_time": [datetime(2018, 2, 15)]})
    engine = TemporalTruthEngine()
    joined = engine.asof_join(frame, left)
    assert joined.height == 1
    assert joined["event_time"][0] == datetime(2018, 1, 1)
    assert joined["allowed"][0] is True


def test_asof_join_left_right_prefixed_column_cannot_gate():
    """Second review S1: a left frame carrying its own right_event_time must
    not hijack the renamed right event column - the rename escalates and the
    join still attaches the right observation."""
    frame = fred_timing_frame(
        _vintage_frame(), "DS-000020", series_policies={"CPIAUCSL": "revised"}
    )
    left = pl.DataFrame(
        {
            "record_id": ["L1"],
            "decision_time": [datetime(2018, 2, 15)],
            "event_time": [datetime(2019, 1, 1)],
            "right_event_time": [datetime(2026, 1, 1)],
        }
    )
    joined = TemporalTruthEngine().asof_join(frame, left)
    assert joined.height == 1
    row = joined.row(0, named=True)
    # the left's own right_event_time forces the rename to escalate
    assert row["right_event_time__2"] == datetime(2018, 1, 1)
    assert row["event_time"] == datetime(2019, 1, 1)  # left's own, untouched


def test_asof_join_left_with_own_row_index_column():
    """Second review S2: a left frame with its own _join_idx column joins
    without collision (the engine's internal index escalates)."""
    frame = fred_timing_frame(
        _vintage_frame(), "DS-000020", series_policies={"CPIAUCSL": "revised"}
    )
    left = pl.DataFrame(
        {
            "_join_idx": ["mine"],
            "record_id": ["L1"],
            "decision_time": [datetime(2018, 2, 15)],
        }
    )
    joined = TemporalTruthEngine().asof_join(frame, left)
    assert joined.height == 1
    assert joined["_join_idx"][0] == "mine"
    assert joined["event_time"][0] == datetime(2018, 1, 1)


def test_asof_join_right_with_right_prefixed_rename_target():
    """Second review S3: a right frame carrying a column whose name equals a
    rename target (right_source_key while left has source_key) must not
    crash with a duplicate rename."""
    frame = fred_timing_frame(
        _vintage_frame(), "DS-000020", series_policies={"CPIAUCSL": "revised"}
    )
    frame = frame.with_columns(pl.lit("mine").alias("right_source_key"))
    left = pl.DataFrame(
        {
            "record_id": ["L1"],
            "decision_time": [datetime(2018, 2, 15)],
            "source_key": ["left"],
        }
    )
    joined = TemporalTruthEngine().asof_join(frame, left)
    assert joined.height == 1
    assert joined["event_time"][0] == datetime(2018, 1, 1)


def test_asof_join_tie_break_without_left_record_id():
    """Second review S4: without a left record_id column the tie-break must
    still pick the largest right record_id, independent of right input
    order."""
    frame = pl.DataFrame(
        {
            "record_id": ["AA001", "BB002"],
            "source_key": ["s", "s"], "domain": ["m", "m"], "kind": ["k", "k"],
            "event_time": [datetime(2018, 1, 1), datetime(2018, 1, 1)],
            "publication_time": [datetime(2018, 1, 2, 15, 0)] * 2,
            "publication_precision": ["datetime"] * 2,
            "effective_time": [None] * 2, "ingestion_time": [None] * 2,
            "vintage_id": [None] * 2, "vintage_date": [None] * 2,
            "series_policy": [None] * 2, "payload_json": ["{}"] * 2,
        }
    )
    left = pl.DataFrame({"decision_time": [datetime(2018, 2, 15)]})
    engine = TemporalTruthEngine()
    w1 = engine.asof_join(frame, left).row(0, named=True)["record_id"]
    w2 = engine.asof_join(frame.reverse(), left).row(0, named=True)["record_id"]
    assert w1 == w2 == "BB002"


def test_asof_join_empty_right_null_joins_consistently():
    """Second review MINOR 9: an empty right frame yields the same null-join
    schema as 'no qualifying record' instead of silently dropping the
    right-side columns."""
    left = pl.DataFrame({"record_id": ["L1"], "decision_time": [datetime(2018, 2, 15)]})
    joined = TemporalTruthEngine().asof_join(pl.DataFrame(), left)
    assert joined.height == 1
    assert joined["allowed"][0] is None
    assert joined["decision_code"][0] is None


# --------------------------------------------------------- reproducibility


def test_snapshot_reproducible_within_process(data_root):
    engine = TemporalTruthEngine(sources=_sources(data_root))
    s1 = engine.snapshot(datetime(2020, 1, 8, 16, 0))
    s2 = engine.snapshot(datetime(2020, 1, 8, 16, 0))
    assert s1.content_digest == s2.content_digest
    assert s1.equals(s2)
    assert s1.allowed_record_ids() == s2.allowed_record_ids()


def test_snapshot_reproducible_across_engine_restart(data_root):
    """Simulates a process restart: a brand-new engine reading the same
    files must produce the identical information set and digest."""
    s1 = TemporalTruthEngine(sources=_sources(data_root)).snapshot(
        datetime(2020, 1, 8, 16, 0)
    )
    s2 = TemporalTruthEngine(sources=_sources(data_root)).snapshot(
        datetime(2020, 1, 8, 16, 0)
    )
    assert s1.content_digest == s2.content_digest
    assert s1.equals(s2)
    assert s1.to_json()["records"] == s2.to_json()["records"]


def test_snapshot_digest_changes_when_content_changes(data_root):
    engine = TemporalTruthEngine(sources=_sources(data_root))
    s1 = engine.snapshot(datetime(2020, 1, 8, 16, 0))
    s2 = engine.snapshot(datetime(2020, 1, 9, 16, 0))  # later as_of
    assert s1.content_digest != s2.content_digest


def test_snapshot_digest_excludes_wall_clock(data_root):
    s1 = TemporalTruthEngine(sources=_sources(data_root)).snapshot(
        datetime(2020, 1, 8, 16, 0)
    )
    assert s1.created_at != s1.as_of_time  # created_at is wall clock
    assert "created_at" not in s1.provenance()  # excluded from content


# --------------------------------------------------------------- provenance


def test_provenance_traceable_to_sources_and_checksums(data_root):
    engine = TemporalTruthEngine(sources=_sources(data_root))
    snap = engine.snapshot(datetime(2020, 1, 8, 16, 0))
    prov = snap.provenance()
    assert prov["as_of_time"] == "2020-01-08T16:00:00"
    assert prov["engine_version"] == "v1.0.0"
    ids = {s["snapshot_id"] for s in prov["sources"]}
    assert ids == {"DS-000001", "DS-000002", "DS-000003"}
    assert all(s["checksum"] and len(s["checksum"]) == 64 for s in prov["sources"])


def test_record_lookup_returns_decision_and_payload(data_root):
    engine = TemporalTruthEngine(sources=_sources(data_root))
    snap = engine.snapshot(datetime(2020, 1, 8, 16, 0))
    bar_id = snap.allowed_record_ids()[0]
    rec = snap.record(bar_id)
    assert rec is not None
    assert rec["decision_code"].startswith("allowed")
    assert rec["domain"] == "market"
    rej = snap.record(snap.excluded_record_ids()[0])
    assert rej is not None and not rej["allowed"]


def test_build_temporal_source_from_registry(tmp_path):
    """The engine consumes Phase 3's registry: build_temporal_source() maps
    a registry record into a TemporalSource with provenance fields."""
    import duckdb

    from orbit.ingestion.registry import IngestionRegistry

    db = tmp_path / "registry.duckdb"
    reg = IngestionRegistry(db)
    reg.register_snapshot(
        {
            "domain": "market", "provider": "yahoo_chart_api",
            "source_uri": "https://example", "request_fingerprint": "fp",
            "checksum": "z" * 64, "file_count": 1, "row_count": 4,
            "downloaded_at": "2020-02-01T12:00:00", "schema_version": "v1.0.0",
            "license_ref": "test", "validation_status": "ok",
            "manifest_path": str(tmp_path / "m.json"),
        }
    )
    src = build_temporal_source(reg, "DS-000001")
    assert src.domain == "market"
    assert src.checksum == "z" * 64
    assert src.ingest_time == datetime(2020, 2, 1, 12, 0)
    assert src.manifest_path == str(tmp_path / "m.json")
    reg.close()


# ---------------------------------------------------------- traceability


def test_rule_trace_exposes_every_intermediate_value():
    engine = TemporalTruthEngine()
    timing = Timing(
        record_id="r1", domain="fundamentals", kind="fact",
        event_time=datetime(2017, 12, 31),
        publication_time=datetime(2018, 2, 15),
        publication_precision=TimePrecision.DATE,
        ingestion_time=datetime(2018, 2, 20),
    )
    trace = engine.trace_record(timing, datetime(2018, 1, 10, 16, 0))
    assert trace.normalized_as_of == datetime(2018, 1, 10, 16, 0)
    assert trace.publication_precision == "date"
    assert trace.available_instant == datetime(2018, 2, 16)
    assert not trace.decision.allowed
    assert trace.decision.code == DecisionCode.PUBLICATION_AT_OR_AFTER_AS_OF


def test_decision_counts_include_all_codes(data_root):
    engine = TemporalTruthEngine(sources=_sources(data_root))
    snap = engine.snapshot(datetime(2020, 1, 8, 16, 0))
    counts = snap.decision_counts()
    assert counts[DecisionCode.ALLOWED_BEFORE_PUBLICATION.value] == 4  # bars
    assert counts[DecisionCode.NOT_POINT_IN_TIME.value] == 1  # CPIAUCSL
    assert counts[DecisionCode.PUBLICATION_AT_OR_AFTER_AS_OF.value] == 2  # facts