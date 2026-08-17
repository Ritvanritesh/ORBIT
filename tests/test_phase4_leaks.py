"""Phase 4 leak tests: the permanent adversarial regression suite.

Every fixture plants FUTURE information next to a decision time. The
engine must reject ALL of it. If someone later changes the temporal
engine and accidentally allows ANY planted leak, these tests fail loudly.
A single surviving leak is a hard stop for Phase 4 (roadmap rule).

Also covered here: feature-time rules (future row / future close / future
fundamental), and the macro vintage ingestion path through the Phase 3
pipeline (hermetic, no network).
"""

from __future__ import annotations

import json
from datetime import date, datetime

import polars as pl
import pytest

from orbit.ingestion.pipeline import IngestionPipeline
from orbit.ingestion.providers.base import RawObject
from orbit.ingestion.registry import IngestionRegistry
from orbit.ingestion.storage import RawStore
from orbit.temporal.engine import TemporalTruthEngine
from orbit.temporal.features import (
    assert_no_future_refs,
    completed_bars,
    FutureRefViolation,
)
from orbit.temporal.fixtures import ALL_LEAK_FIXTURES, future_feature_fixture, run_fixture
from orbit.temporal.times import DecisionCode, Timing

ENGINE = TemporalTruthEngine()


# -------------------------------------------- the permanent leak fixtures


@pytest.mark.parametrize("fixture", ALL_LEAK_FIXTURES, ids=lambda f: f.name)
def test_leak_fixture_rejects_all_future_information(fixture):
    """Every fixture's future records must be rejected; every expected
    allowed record must be allowed."""
    decisions = run_fixture(fixture, ENGINE)
    allowed = {d.record_id for d in decisions if d.allowed}
    rejected = {d.record_id for d in decisions if not d.allowed}
    assert allowed == set(fixture.expected_allowed)
    assert rejected == set(fixture.expected_rejected)
    if fixture.expected_code:
        for d in decisions:
            exp = fixture.expected_code.get(d.record_id)
            if exp is not None:
                assert d.code == exp, f"{fixture.name}/{d.record_id}: {d.code}"


def test_future_earnings_fixture_is_the_canonical_example():
    """The prompt's running example, verified at the record level:
    period ended 2018-03-31, filed 2018-04-25, prediction 2018-01-10."""
    from orbit.temporal.fixtures import future_earnings_fixture

    fx = future_earnings_fixture()
    decisions = run_fixture(fx, ENGINE)
    d = next(d for d in decisions if d.record_id == "earnings_Q1_2018")
    assert not d.allowed
    assert d.code == DecisionCode.PUBLICATION_AT_OR_AFTER_AS_OF


def test_future_macro_revision_never_replaces_history():
    """The 2018-02-01 revision of a 2018-01 observation is not available to
    a 2018-01-10 decision; NEITHER is the not-yet-released original. No
    invented availability."""
    from orbit.temporal.fixtures import future_macro_revision_fixture

    fx = future_macro_revision_fixture()
    decisions = run_fixture(fx, ENGINE)
    assert all(not d.allowed for d in decisions)
    # and, critically, the decision at 2018-01-15 DOES see the original:
    original = fx.timings[0]
    assert original.payload["value"] == 2.1
    d = ENGINE.decide_record(original, datetime(2018, 1, 15))
    assert d.allowed


def test_future_price_bar_is_rejected_before_session_close():
    from orbit.temporal.fixtures import future_price_fixture

    fx = future_price_fixture()
    decisions = run_fixture(fx, ENGINE)
    by_id = {d.record_id: d for d in decisions}
    assert not by_id["bar_2018_01_10"].allowed  # today's close
    assert not by_id["bar_2018_01_11"].allowed  # future session
    assert by_id["bar_2018_01_09"].allowed


def test_delayed_ingestion_is_allowed_with_warning():
    """§14 test 3 inversion: public before t but ingested after t is still
    known history - a warning, never a rejection."""
    from orbit.temporal.fixtures import delayed_ingestion_fixture

    fx = delayed_ingestion_fixture()
    (d,) = run_fixture(fx, ENGINE)
    assert d.allowed
    assert any("ingested" in w for w in d.warnings)


def test_missing_publication_is_never_available():
    from orbit.temporal.fixtures import missing_publication_fixture

    fx = missing_publication_fixture()
    (d,) = run_fixture(fx, ENGINE)
    assert not d.allowed
    assert d.code == DecisionCode.MISSING_PUBLICATION_TIME


# ------------------------------------------------------ feature-time leaks


def test_future_feature_fixture_flagged():
    ff = future_feature_fixture()
    violations = assert_no_future_refs(
        ff["bars"], ff["as_of"], time_col="trade_date", id_col="record_id"
    )
    assert {v.record_id for v in violations} == set(ff["expected_violations"])
    assert all(isinstance(v, FutureRefViolation) for v in violations)


def test_completed_bars_never_includes_same_day_bar():
    bars = pl.DataFrame(
        {
            "instrument_id": ["INS-000001"] * 3,
            "trade_date": [date(2018, 1, 8), date(2018, 1, 9), date(2018, 1, 10)],
            "close": [100.0, 101.0, 102.0],
        }
    )
    # decision at 2018-01-10 15:00 UTC: the 01-10 session has NOT closed
    window = completed_bars(bars, datetime(2018, 1, 10, 15, 0), 5)
    assert window["trade_date"].to_list() == [date(2018, 1, 8), date(2018, 1, 9)]

    # decision after the 01-10 close (21:00 UTC): the 01-10 bar is complete
    window2 = completed_bars(bars, datetime(2018, 1, 10, 21, 0, 1), 5)
    assert window2["trade_date"].to_list() == [date(2018, 1, 8), date(2018, 1, 9), date(2018, 1, 10)]

    # exactly at the close: still NOT complete (strict boundary)
    window3 = completed_bars(bars, datetime(2018, 1, 10, 21, 0), 5)
    assert window3["trade_date"].to_list() == [date(2018, 1, 8), date(2018, 1, 9)]


def test_completed_bars_window_is_capped_at_n():
    bars = pl.DataFrame(
        {
            "instrument_id": ["INS-000001"] * 5,
            "trade_date": [date(2018, 1, 2 + i) for i in range(5)],
            "close": [100.0 + i for i in range(5)],
        }
    )
    window = completed_bars(bars, datetime(2018, 1, 10, 15, 0), 2)
    assert window.height == 2
    assert window["trade_date"].to_list() == [date(2018, 1, 5), date(2018, 1, 6)]


def test_future_fundamental_leak_rejected():
    """A 'feature' that uses a fundamental filed after the decision must be
    rejected by the engine and flagged by the feature-time rules."""
    from orbit.temporal.fixtures import future_earnings_fixture

    fx = future_earnings_fixture()
    earnings = fx.timings[0]
    d = ENGINE.decide_record(earnings, fx.as_of)
    assert not d.allowed
    # feature-level: any feature referencing that record at t is a violation
    feat = pl.DataFrame(
        {
            "record_id": ["feat_1"],
            "ref_instant": [d.as_of_time],
            "feature_time": [fx.as_of],
        }
    )
    violations = assert_no_future_refs(feat, fx.as_of, time_col="ref_instant")
    assert len(violations) == 1


# --------------------------------------------- pipeline -> engine end-to-end


class FakeFredVintageConnector:
    """Emits ALFRED-shaped payloads: one vintage per request."""

    provider_name = "fred_csv"

    def __init__(self, vintages: dict[str, str]):
        self._vintages = vintages  # vintage_date -> csv body

    def fetch(self, request):
        v = request["vintage_date"]
        sid = request["series_id"]
        return [
            RawObject(
                filename=f"{sid}.csv", body=self._vintages[v].encode(),
                source_uri=f"https://alfred.local/{sid}?vintage_date={v}",
                content_type="text/csv",
                meta={"series_id": sid, "vintage_date": v, "vintage_note": "alfred_vintage_requested"},
            )
        ]


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("ORBIT_DATA_ROOT", str(tmp_path))
    registry = IngestionRegistry(tmp_path / "registry.duckdb")
    pipeline = IngestionPipeline(registry, RawStore())
    yield {"root": tmp_path, "registry": registry, "pipeline": pipeline}
    registry.close()


def _fred_csv(rows: list[tuple[str, float]]) -> str:
    body = "observation_date,CPIAUCSL\n"
    body += "\n".join(f"{d},{v}" for d, v in rows)
    return body


def test_pipeline_ingests_two_vintages_and_engine_resolves_pit(env):
    """ALFRED vintages flow through the Phase 3 pipeline into the Phase 4
    engine: at a mid-January as_of the ORIGINAL value is the known one."""
    csv_orig = _fred_csv([("2018-01-01", 2.1), ("2017-12-01", 2.0)])
    csv_rev = _fred_csv([("2018-01-01", 2.3), ("2017-12-01", 2.0)])

    r1 = env["pipeline"].ingest_macro(
        FakeFredVintageConnector({"2018-01-12": csv_orig}),
        ["CPIAUCSL"], license_ref="test", request_params={"vintage_date": "2018-01-12"},
    )
    r2 = env["pipeline"].ingest_macro(
        FakeFredVintageConnector({"2018-02-01": csv_rev}),
        ["CPIAUCSL"], license_ref="test", request_params={"vintage_date": "2018-02-01"},
    )
    assert r1.validation["status"] == "ok"
    assert r2.validation["status"] == "ok"
    assert r1.snapshot_id != r2.snapshot_id

    from orbit.temporal.snapshot import TemporalSource

    def src(snapshot_id: str) -> TemporalSource:
        return TemporalSource(
            snapshot_id=snapshot_id, domain="macro", provider="fred_csv",
            checksum=env["registry"].snapshot(snapshot_id)["checksum"],
            manifest_path=env["registry"].snapshot(snapshot_id)["manifest_path"],
            ingest_time=datetime(2018, 3, 1),
        )

    engine = TemporalTruthEngine()
    from orbit.temporal.adapters import fred_timing_frame

    frame = fred_timing_frame(
        pl.concat(
            [
                pl.read_parquet(env["root"] / "normalized" / "macro" / "fred_csv" / r1.snapshot_id / "series.parquet"),
                pl.read_parquet(env["root"] / "normalized" / "macro" / "fred_csv" / r2.snapshot_id / "series.parquet"),
            ]
        ),
        r1.snapshot_id, series_policies={"CPIAUCSL": "revised"},
    )
    pit = engine.historical_vintage(frame, datetime(2018, 1, 15))
    jan = pit.filter(pl.col("event_time") == datetime(2018, 1, 1))
    assert jan.height == 1
    val = json.loads(jan.row(0, named=True)["payload_json"])["value"]
    assert val == 2.1  # original, NOT the 2018-02-01 revision

    # later as_of: the revision is now the known value
    pit2 = engine.historical_vintage(frame, datetime(2018, 2, 5))
    jan2 = pit2.filter(pl.col("event_time") == datetime(2018, 1, 1))
    val2 = json.loads(jan2.row(0, named=True)["payload_json"])["value"]
    assert val2 == 2.3

    # a full snapshot over both vintages audits the superseded version
    snap = engine.snapshot(datetime(2018, 2, 5), sources=[src(r1.snapshot_id), src(r2.snapshot_id)])
    codes = set(snap.excluded["decision_code"].to_list())
    assert DecisionCode.VINTAGE_SUPERSEDED.value in codes


def test_latest_vintage_ingest_without_alfred_is_not_point_in_time(env):
    """The Phase 3 default (latest published vintage, no vintage_date):
    a revised series must be excluded by the engine and reported, and the
    manifest documents the limitation."""
    csv = _fred_csv([("2018-01-01", 2.5)])

    class FakeLatestConnector:
        provider_name = "fred_csv"

        def fetch(self, request):
            return [
                RawObject(
                    filename="CPIAUCSL.csv", body=csv.encode(),
                    source_uri="https://fred.local/CPIAUCSL",
                    content_type="text/csv",
                    meta={"series_id": "CPIAUCSL"},
                )
            ]

    r = env["pipeline"].ingest_macro(FakeLatestConnector(), ["CPIAUCSL"], license_ref="test")
    assert r.validation["status"] == "ok"

    import json as _json

    from orbit.temporal.adapters import fred_timing_frame

    frame = fred_timing_frame(
        pl.read_parquet(env["root"] / "normalized" / "macro" / "fred_csv" / r.snapshot_id / "series.parquet"),
        r.snapshot_id, series_policies={"CPIAUCSL": "revised"},
    )
    engine = TemporalTruthEngine()
    ev = engine.evaluate(frame, datetime(2020, 1, 1))
    assert ev.allowed.height == 0
    assert ev.excluded["decision_code"].to_list() == [DecisionCode.NOT_POINT_IN_TIME.value]
    assert any("CPIAUCSL" in l for l in ev.limitations)

    manifest = _json.loads(
        (env["root"] / "manifests" / f"{r.snapshot_id}.json").read_text(encoding="utf-8")
    )
    assert manifest["meta"]["vintage_semantics"]


def test_sec_pipeline_engine_snapshot_excludes_future_filing(env):
    """End-to-end §8 example: period ended 2019-03-31, filed 2019-05-02,
    prediction 2019-04-15 -> the filing must NOT be in the information set."""
    from orbit.ingestion.normalizers.fundamentals import FUNDAMENTALS_SCHEMA_VERSION
    from orbit.ingestion.parsing import parse_sec_companyfacts

    payload = json.dumps(
        {
            "cik": 320193,
            "entityName": "Apple Inc.",
            "facts": {
                "us-gaap": {
                    "Revenues": {
                        "units": {
                            "USD": [
                                {"start": "2019-01-01", "end": "2019-03-31",
                                 "val": 58_000_000_000, "accn": "0000320193-19-000010",
                                 "fy": 2019, "fp": "Q2", "form": "10-Q",
                                 "filed": "2019-05-02", "frame": "CY2019Q1"},
                            ]
                        }
                    }
                }
            },
        }
    ).encode()

    class FakeSecConnector:
        provider_name = "sec_edgar_companyfacts"

        def fetch(self, request):
            cik = int(request["cik"])
            return [
                RawObject(
                    filename=f"cik{cik:010d}_companyfacts.json", body=payload,
                    source_uri="https://data.sec.gov.local",
                    content_type="application/json",
                    meta={"cik": cik},
                )
            ]

    r = env["pipeline"].ingest_sec(FakeSecConnector(), [320193], license_ref="test")
    assert r.validation["status"] == "ok"

    from orbit.temporal.snapshot import TemporalSource

    src = TemporalSource(
        snapshot_id=r.snapshot_id, domain="sec", provider="sec_edgar_companyfacts",
        checksum=env["registry"].snapshot(r.snapshot_id)["checksum"],
        manifest_path=env["registry"].snapshot(r.snapshot_id)["manifest_path"],
        ingest_time=datetime(2019, 6, 1),
    )
    engine = TemporalTruthEngine(sources=[src])
    snap = engine.snapshot(datetime(2019, 4, 15, 16, 0))
    assert snap.records.height == 0  # filing not yet public
    assert snap.excluded.height == 1
    rej = snap.excluded.row(0, named=True)
    assert rej["decision_code"] == DecisionCode.PUBLICATION_AT_OR_AFTER_AS_OF.value

    after = engine.snapshot(datetime(2019, 5, 3, 0, 0, 1))
    assert after.records.height == 1  # available the day after filing