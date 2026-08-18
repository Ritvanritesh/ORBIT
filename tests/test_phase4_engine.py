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


# ------------------------------------------------ audit findings regressions


def test_rejected_revision_preserved_in_audit_trail():
    """Audit finding: _resolve_vintages used to DROP rejected vintage rows
    when another vintage of the same observation was allowed - the rejected
    revision vanished from the snapshot's excluded set. It must stay."""
    frame = fred_timing_frame(
        _vintage_frame(), "DS-000020", series_policies={"CPIAUCSL": "revised"}
    )
    engine = TemporalTruthEngine()
    # as_of 2018-01-15: original released 2018-01-12 (allowed next day),
    # revision released 2018-02-01 (rejected - not yet public)
    resolved = engine._resolve_vintages(engine.evaluate(frame, datetime(2018, 1, 15)).frame)
    excluded = resolved.filter(~pl.col("allowed"))
    codes = {r["record_id"]: r["decision_code"] for r in excluded.iter_rows(named=True)}
    assert "obs|DS-000020|CPIAUCSL|2018-01-01|2018-02-01" in codes
    assert codes["obs|DS-000020|CPIAUCSL|2018-02-01|2018-03-01"] == (
        DecisionCode.PUBLICATION_AT_OR_AFTER_AS_OF.value
    )
    # the audit trail is complete: rejected + superseded + allowed all present
    assert resolved.height == 3


def test_rejected_revision_preserved_in_snapshot_excluded(data_root):
    """Snapshot-level: the same guarantee through the public API."""
    v = _vintage_frame()
    d = data_root / "normalized" / "macro" / "fred_csv" / "DS-000020"
    d.mkdir(parents=True, exist_ok=True)
    v.write_parquet(d / "series.parquet")
    src = TemporalSource(
        snapshot_id="DS-000020", domain="macro", provider="fred_csv",
        checksum="c" * 64, manifest_path="m.json",
        artifact_paths=[str(d / "series.parquet")],
    )
    engine = TemporalTruthEngine(sources=[src])
    snap = engine.snapshot(datetime(2018, 1, 15))
    excl = {r["record_id"]: r["decision_code"] for r in snap.excluded.iter_rows(named=True)}
    assert "obs|DS-000020|CPIAUCSL|2018-01-01|2018-02-01" in excl
    assert excl["obs|DS-000020|CPIAUCSL|2018-01-01|2018-02-01"] == (
        DecisionCode.PUBLICATION_AT_OR_AFTER_AS_OF.value
    )
    # the original vintage is the only allowed version of the Jan observation
    jan = snap.records.filter(pl.col("event_time") == datetime(2018, 1, 1))
    assert jan.height == 1
    assert json.loads(jan.row(0, named=True)["payload_json"])["value"] == 2.1


def test_market_payload_as_published_prices_from_split_events():
    """Audit finding: stored bars are retroactively split-adjusted; the
    payload must carry AS-PUBLISHED prices reconstructed from the sibling
    events artifact - never the post-split values."""
    bars = pl.DataFrame(
        {
            "instrument_id": ["INS-000001"] * 3,
            "symbol": ["AAPL"] * 3,
            "trade_date": [date(2014, 6, 6), date(2014, 6, 9), date(2014, 6, 10)],
            "ts_utc": [datetime(2014, 6, 6, 14, 30)] * 3,
            "open": [23.1, 23.17, 23.68],
            "high": [23.2, 23.47, 23.76],
            "low": [23.0, 22.94, 23.39],
            "close": [23.05607, 23.424999, 23.5625],
            "volume": [349_938_400, 62_766_800, 53_930_400],
            "adjclose": [20.2, 20.5, 20.6],
            "adjustment": ["split_adjusted"] * 3,
        }
    )
    events = pl.DataFrame(
        {
            "instrument_id": ["INS-000001"] * 2,
            "symbol": ["AAPL"] * 2,
            "kind": ["splits", "splits"],
            "ts": [datetime(2014, 6, 9, 13, 30), datetime(2020, 8, 31, 13, 30)],
            "ratio": [7.0, 4.0],
        }
    )
    frame = market_timing_frame(bars, "DS-000001", events=events)
    rows = {json.loads(r["payload_json"])["trade_date"]: json.loads(r["payload_json"])
            for r in frame.iter_rows(named=True)}
    # pre-split: raw = adjusted * 7 * 4 (both splits after the bar)
    assert rows["2014-06-06"]["close"] == pytest.approx(23.05607 * 28, abs=1e-6)
    assert rows["2014-06-06"]["price_basis"] == "as_published"
    assert rows["2014-06-06"]["volume"] == int(349_938_400 / 28)
    # ex-date bar: only the 2020 split remains -> factor 4
    assert rows["2014-06-09"]["close"] == pytest.approx(23.424999 * 4, abs=1e-6)
    assert rows["2014-06-10"]["close"] == pytest.approx(23.5625 * 4, abs=1e-6)
    # adjclose and adjustment never ride in the payload
    for r in rows.values():
        assert "adjclose" not in r
        assert "adjustment" not in r


def test_market_payload_without_events_marks_provider_basis():
    """No events artifact -> the provider values stay verbatim and the
    payload says so: never silently presented as historical truth."""
    bars = pl.DataFrame(
        {
            "instrument_id": ["INS-000001"],
            "trade_date": [date(2018, 1, 9)],
            "ts_utc": [datetime(2018, 1, 9, 14, 30)],
            "open": [100.0], "high": [102.0], "low": [99.0], "close": [101.0],
            "volume": [1000], "adjclose": [95.0],
        }
    )
    frame = market_timing_frame(bars, "DS-000001")
    r = json.loads(frame.row(0, named=True)["payload_json"])
    assert r["close"] == 101.0
    assert r["price_basis"] == "provider_split_adjusted"


def _split_events() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "instrument_id": ["INS-000001"] * 2,
            "symbol": ["AAPL"] * 2,
            "kind": ["splits", "splits"],
            "ts": [datetime(2014, 6, 9, 13, 30), datetime(2020, 8, 31, 13, 30)],
            "ratio": [7.0, 4.0],
        }
    )


def test_market_payload_raw_volume_provider_keeps_verbatim_shares():
    """Audit finding (volume basis): stooq_csv volume is RAW shares (the
    yahoo volume is split-adjusted). Dividing the raw volume by the split
    factor would corrupt the as-published share count - OHLC is still
    reconstructed, volume is not."""
    bars = pl.DataFrame(
        {
            "instrument_id": ["INS-000001"] * 3,
            "symbol": ["AAPL"] * 3,
            "trade_date": [date(2014, 6, 6), date(2014, 6, 9), date(2014, 6, 10)],
            "ts_utc": [datetime(2014, 6, 6, 14, 30)] * 3,
            "open": [23.1, 23.17, 23.68],
            "high": [23.2, 23.47, 23.76],
            "low": [23.0, 22.94, 23.39],
            "close": [23.05607, 23.424999, 23.5625],
            "volume": [12_497_800, 62_766_800, 53_930_400],
            "adjclose": [20.2, 20.5, 20.6],
            "adjustment": ["split_adjusted"] * 3,
            "provider": ["stooq_csv"] * 3,
        }
    )
    frame = market_timing_frame(bars, "DS-000001", events=_split_events())
    rows = {json.loads(r["payload_json"])["trade_date"]: json.loads(r["payload_json"])
            for r in frame.iter_rows(named=True)}
    # OHLC reconstructed (stooq OHLC is split-adjusted too)
    assert rows["2014-06-06"]["close"] == pytest.approx(23.05607 * 28, abs=1e-6)
    # volume is raw shares: NEVER divided by the split factor
    assert rows["2014-06-06"]["volume"] == 12_497_800
    assert rows["2014-06-06"]["price_basis"] == "as_published"
    # the split-adjusted-volume path still divides (regression guard)
    y = bars.with_columns(pl.lit("yahoo_chart_api").alias("provider"))
    yf = market_timing_frame(y, "DS-000001", events=_split_events())
    yrows = {json.loads(r["payload_json"])["trade_date"]: json.loads(r["payload_json"])
             for r in yf.iter_rows(named=True)}
    assert yrows["2014-06-06"]["volume"] == int(12_497_800 / 28)
    # an explicit volume_basis overrides the provider inference
    vf = market_timing_frame(y, "DS-000001", events=_split_events(), volume_basis="raw")
    vrows = {json.loads(r["payload_json"])["trade_date"]: json.loads(r["payload_json"])
             for r in vf.iter_rows(named=True)}
    assert vrows["2014-06-06"]["volume"] == 12_497_800


def test_market_payload_ohlc_reconstruction_guarded_by_adjustment_basis():
    """Audit finding (adjustment guard): multiplying OHLC by the split
    factor is only valid for SPLIT-ADJUSTED stored bars. A provider that
    delivers raw OHLC must keep its verbatim as-published prices."""
    bars = pl.DataFrame(
        {
            "instrument_id": ["INS-000001"],
            "symbol": ["AAPL"],
            "trade_date": [date(2014, 6, 6)],
            "ts_utc": [datetime(2014, 6, 6, 14, 30)],
            "open": [645.5], "high": [646.0], "low": [643.0], "close": [645.57],
            "volume": [12_497_800],
            "adjclose": [645.57],
            "adjustment": ["raw"],
        }
    )
    frame = market_timing_frame(bars, "DS-000001", events=_split_events())
    r = json.loads(frame.row(0, named=True)["payload_json"])
    # already as-published: not multiplied by 28
    assert r["close"] == 645.57
    assert r["open"] == 645.5
    assert r["volume"] == 12_497_800
    assert r["price_basis"] == "as_published"


def test_market_payload_factor_column_collision_is_safe():
    """Second-pass finding: a bars frame that already carries its own
    'factor' column used to make the polars join produce 'factor_right',
    and the multiplier silently read the BARS' factor (999.0) instead of
    the split factor - corrupting every reconstructed price. The joined
    factor is now collision-free."""
    bars = pl.DataFrame(
        {
            "instrument_id": ["INS-000001"],
            "symbol": ["AAPL"],
            "trade_date": [date(2014, 6, 6)],
            "ts_utc": [datetime(2014, 6, 6, 14, 30)],
            "open": [23.1], "high": [23.2], "low": [23.0], "close": [23.05607],
            "volume": [349_938_400],
            "adjclose": [20.2],
            "adjustment": ["split_adjusted"],
            "factor": [999.0],
        }
    )
    frame = market_timing_frame(bars, "DS-000001", events=_split_events())
    r = json.loads(frame.row(0, named=True)["payload_json"])
    assert r["close"] == pytest.approx(23.05607 * 28, abs=1e-6)
    assert r["volume"] == int(349_938_400 / 28)


def test_engine_loads_events_artifact_for_market_source(data_root):
    """The engine finds the sibling events.parquet of a market source and
    the snapshot payload then carries as-published prices."""
    d = data_root / "normalized" / "market" / "yahoo_chart_api" / "DS-000099"
    d.mkdir(parents=True, exist_ok=True)
    bars = pl.DataFrame(
        {
            "instrument_id": ["INS-000001"] * 2,
            "symbol": ["AAPL"] * 2,
            "trade_date": [date(2014, 6, 6), date(2014, 6, 9)],
            "ts_utc": [datetime(2014, 6, 6, 14, 30)] * 2,
            "open": [23.1, 23.17], "high": [23.2, 23.47],
            "low": [23.0, 22.94], "close": [23.05607, 23.424999],
            "volume": [349_938_400, 62_766_800],
            "adjclose": [20.2, 20.5],
            "adjustment": ["split_adjusted"] * 2,
            "provider": ["yahoo_chart_api"] * 2,
            "source_uri": ["u"] * 2,
            "snapshot_id": ["DS-000099"] * 2,
        }
    )
    bars.write_parquet(d / "bars.parquet")
    pl.DataFrame(
        {
            "instrument_id": ["INS-000001"],
            "symbol": ["AAPL"],
            "kind": ["splits"],
            "ts": [datetime(2014, 6, 9, 13, 30)],
            "ratio": [7.0],
        }
    ).write_parquet(d / "events.parquet")
    src = TemporalSource(
        snapshot_id="DS-000099", domain="market", provider="yahoo_chart_api",
        checksum="a" * 64,
        artifact_paths=[str(d / "bars.parquet")],
    )
    engine = TemporalTruthEngine(sources=[src])
    snap = engine.snapshot(datetime(2014, 6, 10, 16, 0))
    rows = {json.loads(r["payload_json"])["trade_date"]: json.loads(r["payload_json"])
            for r in snap.records.iter_rows(named=True)}
    assert rows["2014-06-06"]["close"] == pytest.approx(23.05607 * 7, abs=1e-6)  # as-published
    assert rows["2014-06-06"]["price_basis"] == "as_published"
    assert rows["2014-06-09"]["close"] == 23.424999  # post-split: factor 1


def test_contract_rejects_mismatched_convention(tmp_path):
    """A config that promises a convention the code does not implement must
    fail loudly at load time (audit finding: conventions used to be parsed
    and ignored)."""
    from orbit.temporal.contracts import load_temporal_contract

    p = tmp_path / "temporal.json"
    p.write_text(
        json.dumps(
            {
                "engine_version": "v1.0.0",
                "session_close": "15:30",
                "series_policies": {"DFF": "non_revised"},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="session_close"):
        load_temporal_contract(p)
    # the default config is consistent with the implementation
    from orbit.temporal.contracts import load_temporal_contract as _load

    ok = _load()
    assert ok.exchange_tz == "America/New_York"


def test_contract_rejects_non_revised_default_policy(tmp_path):
    """Audit finding: the documented conservative default is that an
    UNKNOWN series counts as revised (rejected point-in-time). A config
    flipping the default to non_revised would silently admit revised series
    without vintage history - a leak-by-config - so it must fail loudly."""
    from orbit.temporal.contracts import load_temporal_contract

    p = tmp_path / "temporal.json"
    p.write_text(
        json.dumps(
            {
                "engine_version": "v1.0.0",
                "default_series_policy": "non_revised",
                "series_policies": {"DFF": "non_revised"},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="default_series_policy"):
        load_temporal_contract(p)


def test_contract_normalizes_series_policy_values(tmp_path):
    """Audit finding: garbage policy values were accepted silently. Values
    are casefolded and normalized to the two canonical labels; anything else
    raises instead of silently changing a series' revision status."""
    from orbit.temporal.contracts import load_temporal_contract

    p = tmp_path / "temporal.json"
    p.write_text(
        json.dumps(
            {
                "engine_version": "v1.0.0",
                "series_policies": {"DFF": "Non_Revised", "CPIAUCSL": "revised "},
            }
        ),
        encoding="utf-8",
    )
    c = load_temporal_contract(p)
    assert c.series_policies == {
        "DFF": "non_revised", "CPIAUCSL": "revised", "UNRATE": "revised",
    }
    p2 = tmp_path / "bad.json"
    p2.write_text(
        json.dumps({"series_policies": {"DFF": "sometimes_revised"}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="invalid series policy"):
        load_temporal_contract(p2)


def test_snapshot_does_not_mutate_source_frames(data_root):
    """Invariant (prompt 16): requesting snapshots must never mutate the
    underlying source data."""
    engine = TemporalTruthEngine(sources=_sources(data_root))
    before = {
        src.snapshot_id: pl.read_parquet(src.artifact_paths[0]).hash_rows().to_list()
        for src in engine.sources()
    }
    engine.snapshot(datetime(2020, 1, 8, 16, 0))
    engine.snapshot(datetime(2020, 1, 9, 16, 0))
    engine.snapshot(datetime(2019, 1, 1))
    for src in engine.sources():
        after = pl.read_parquet(src.artifact_paths[0]).hash_rows().to_list()
        assert after == before[src.snapshot_id]


def test_engine_data_root_redirects_artifact_resolution(data_root):
    """Fifth-review finding: TemporalTruthEngine(data_root=...) was stored
    but inert - artifact resolution always used the global data zone, so a
    sandboxed engine silently read the WRONG data. data_root now redirects
    the normalized artifact lookup, keeping sandboxed evaluation hermetic."""
    _write_market(data_root)  # writes into data_root/normalized/market/...
    engine = TemporalTruthEngine(
        data_root=data_root,
        sources=[
            TemporalSource(
                snapshot_id="DS-000001", domain="market",
                provider="yahoo_chart_api",
                checksum="a" * 64, manifest_path="m1.json",
            )
        ],
    )
    snap = engine.snapshot(datetime(2020, 1, 8, 16, 0))
    assert snap.records.height == 4
    # artifact resolution went to the sandboxed root, never the global zone
    src = engine.sources()[0]
    assert engine._artifact_paths(src) == [
        str(data_root / "normalized" / "market" / "yahoo_chart_api"
            / "DS-000001" / "bars.parquet")
    ]


def test_monotonic_information_set_no_future_backtravel():
    """Invariants (prompt 16) over a deterministic corpus:

      - no future visibility: every allowed record's available instant is
        strictly before as_of;
      - monotonic information: a non-vintage record allowed at t1 stays
        allowed at every later t2;
      - an observation available at t1 (some version) stays available at
        every later t2;
      - a future revision is never the version selected at t1.
    """
    from orbit.temporal.adapters import TIMING_SCHEMA
    from orbit.temporal.times import Timing

    def fact(rid: str, pub: datetime, event: datetime, ingest: datetime | None = None) -> Timing:
        return Timing(
            record_id=rid, domain="fundamentals", kind="fact",
            event_time=event, publication_time=pub,
            publication_precision=TimePrecision.DATETIME,
            effective_time=pub, ingestion_time=ingest,
        )

    def vintage(rid: str, obs: date, released: date, value: float) -> Timing:
        return Timing(
            record_id=rid, domain="macro", kind="observation",
            event_time=datetime(obs.year, obs.month, obs.day),
            publication_time=datetime(released.year, released.month, released.day),
            publication_precision=TimePrecision.DATE,
            effective_time=datetime(released.year, released.month, released.day),
            vintage_id=f"v{released.isoformat()}", vintage_date=released,
            series_policy="revised",
            payload={"series_id": "CPIAUCSL", "value": value},
        )

    corpus = [
        fact("f_early", datetime(2018, 1, 2, 15), datetime(2018, 1, 1)),
        fact("f_late", datetime(2018, 1, 15, 15), datetime(2018, 1, 10)),
        fact("f_ingested_late", datetime(2018, 1, 5, 15), datetime(2018, 1, 4),
             ingest=datetime(2018, 3, 1)),
        vintage("v1", date(2018, 1, 1), date(2018, 1, 10), 2.1),
        vintage("v2", date(2018, 1, 1), date(2018, 2, 5), 2.3),
        vintage("v3", date(2018, 2, 1), date(2018, 3, 1), 2.2),
    ]

    def timing_frame(corpus: list[Timing]) -> pl.DataFrame:
        return pl.DataFrame(
            [
                {
                    "record_id": c.record_id,
                    "source_key": c.payload.get("series_id", "s") if c.payload else "s",
                    "domain": c.domain, "kind": c.kind,
                    "event_time": c.event_time, "publication_time": c.publication_time,
                    "publication_precision": c.publication_precision.value,
                    "effective_time": c.effective_time, "ingestion_time": c.ingestion_time,
                    "vintage_id": c.vintage_id, "vintage_date": c.vintage_date,
                    "series_policy": c.series_policy,
                    "payload_json": json.dumps(c.payload or {}),
                }
                for c in corpus
            ],
            schema=TIMING_SCHEMA,
        )

    engine = TemporalTruthEngine()
    times = [
        datetime(2018, 1, 1),
        datetime(2018, 1, 8, 16, 0),
        datetime(2018, 1, 12, 16, 0),
        datetime(2018, 2, 7, 16, 0),
        datetime(2018, 3, 5, 16, 0),
    ]
    resolved_by_t: list[tuple[datetime, pl.DataFrame]] = []
    for t in times:
        resolved = engine._resolve_vintages(engine.evaluate(timing_frame(corpus), t).frame)
        resolved_by_t.append((t, resolved))

    # (a) no future visibility: allowed records are strictly before as_of
    for t, resolved in resolved_by_t:
        allowed = resolved.filter(pl.col("allowed"))
        if allowed.height:
            assert (allowed["available_instant"] < t).all()

    # (b) non-vintage records: allowed at t1 stays allowed at every later t2
    def allowed_at(rid: str, t: datetime) -> bool:
        for t2, resolved in resolved_by_t:
            if t2 == t:
                return resolved.filter(pl.col("record_id") == rid)["allowed"][0]
        raise AssertionError(t)

    for i in range(len(times) - 1):
        for rid in ("f_early", "f_ingested_late"):
            if allowed_at(rid, times[i]):
                assert allowed_at(rid, times[i + 1]), f"{rid} regressed at {times[i+1]}"

    # (c) the January observation exists (in some version) from 01-12 on
    jan_at = {
        t: resolved.filter(
            (pl.col("source_key") == "CPIAUCSL")
            & (pl.col("event_time") == datetime(2018, 1, 1))
            & pl.col("allowed")
        ).height
        for t, resolved in resolved_by_t
    }
    assert jan_at[datetime(2018, 1, 1)] == 0
    assert jan_at[datetime(2018, 1, 8, 16, 0)] == 0   # nothing released yet
    assert jan_at[datetime(2018, 1, 12, 16, 0)] == 1  # original only
    assert jan_at[datetime(2018, 2, 7, 16, 0)] == 1   # revision supersedes
    assert jan_at[datetime(2018, 3, 5, 16, 0)] == 1

    # (d) the version selected at t is never the future revision
    t1_rows = {
        json.loads(r["payload_json"])["value"]: r
        for r in resolved_by_t[2][1]
        .filter(
            (pl.col("source_key") == "CPIAUCSL")
            & (pl.col("event_time") == datetime(2018, 1, 1))
            & pl.col("allowed")
        )
        .iter_rows(named=True)
    }
    assert list(t1_rows) == [2.1]
    t2_rows = {
        json.loads(r["payload_json"])["value"]: r
        for r in resolved_by_t[4][1]
        .filter(
            (pl.col("source_key") == "CPIAUCSL")
            & (pl.col("event_time") == datetime(2018, 1, 1))
            & pl.col("allowed")
        )
        .iter_rows(named=True)
    }
    assert list(t2_rows) == [2.3]

    # (e) the rejected revision stays in the audit trail at 01-12
    codes = {
        r["record_id"]: r["decision_code"]
        for r in resolved_by_t[2][1].filter(~pl.col("allowed")).iter_rows(named=True)
    }
    assert codes["v2"] == DecisionCode.PUBLICATION_AT_OR_AFTER_AS_OF.value
    assert codes["v3"] == DecisionCode.PUBLICATION_AT_OR_AFTER_AS_OF.value