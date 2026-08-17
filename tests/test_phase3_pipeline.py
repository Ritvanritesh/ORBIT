"""Phase 3 pipeline tests: end-to-end ingestion with a synthetic connector.

These prove the Phase 3 success criteria hermetically (no network):
idempotent re-ingestion, byte-identical re-derivation, immutability of the
raw zone, validation gating, and manifests that answer "what did ORBIT
download?". The same pipeline code runs against real providers in
scripts/phase3_run_all.py.
"""

from __future__ import annotations

import json
import os
from datetime import date, datetime

import polars as pl
import pytest

from orbit.ingestion.checksums import sha256_bytes, sha256_file
from orbit.ingestion.normalizers.market import normalize_market_bars
from orbit.ingestion.parsing import parse_yahoo_chart
from orbit.ingestion.paths import normalized_dir, raw_dir
from orbit.ingestion.pipeline import IngestionPipeline
from orbit.ingestion.providers.base import RawObject
from orbit.ingestion.registry import IngestionRegistry
from orbit.ingestion.snapshot import MarketDataAccessor, build_dataset_snapshot
from orbit.ingestion.storage import RawStore
from orbit.ingestion.validators import validate_market_bars


def _yahoo_payload(symbol: str = "AAPL") -> bytes:
    ts = [
        int((datetime(2020, 1, 2, 14, 30) - datetime(1970, 1, 1)).total_seconds()),
        int((datetime(2020, 1, 3, 14, 30) - datetime(1970, 1, 1)).total_seconds()),
        int((datetime(2020, 1, 6, 14, 30) - datetime(1970, 1, 1)).total_seconds()),
        int((datetime(2020, 1, 7, 14, 30) - datetime(1970, 1, 1)).total_seconds()),
    ]
    return json.dumps(
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


class FakeConnector:
    """Synthetic provider with deterministic payloads per symbol."""

    provider_name = "yahoo_chart_api"

    def __init__(self, payloads: dict[str, bytes]):
        self._payloads = payloads

    def fetch(self, request):
        symbol = request["symbol"]
        body = self._payloads[symbol]
        return [
            RawObject(
                filename=f"{symbol}.json", body=body,
                source_uri=f"https://fake.local/chart/{symbol}",
                content_type="application/json",
                meta={"symbol": symbol, "bars": 4},
            )
        ]


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("ORBIT_DATA_ROOT", str(tmp_path))
    registry = IngestionRegistry(tmp_path / "registry.duckdb")
    pipeline = IngestionPipeline(registry, RawStore())
    yield {
        "root": tmp_path,
        "registry": registry,
        "pipeline": pipeline,
        "symbol_map": {"AAPL": "INS-000001"},
    }
    registry.close()


def _ingest(env, payload=None, request_params=None):
    payload = payload or _yahoo_payload()
    return env["pipeline"].ingest_market(
        FakeConnector({"AAPL": payload}), ["AAPL"], env["symbol_map"],
        license_ref="test-license",
        request_params=request_params or {"range": "30y"},
    )


def test_ingest_creates_raw_normalized_manifest(env):
    result = _ingest(env)
    assert not result.reused
    assert result.validation["status"] == "ok"
    root = env["root"]

    raw_file = raw_dir("market", "yahoo_chart_api", result.snapshot_id) / "AAPL.json"
    assert raw_file.exists()
    assert raw_file.read_bytes() == _yahoo_payload()  # verbatim
    assert (raw_dir("market", "yahoo_chart_api", result.snapshot_id) / "IMMUTABLE.json").exists()

    bars_path = normalized_dir("market", "yahoo_chart_api", result.snapshot_id) / "bars.parquet"
    assert bars_path.exists()
    bars = pl.read_parquet(bars_path)
    assert bars.height == 4
    assert bars["instrument_id"].unique().to_list() == ["INS-000001"]

    manifest_path = env["root"] / "manifests" / f"{result.snapshot_id}.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["validation_status"] == "ok"
    assert manifest["checksums"]["AAPL.json"] == sha256_bytes(_yahoo_payload())
    assert manifest["date_range"] == ["2020-01-02", "2020-01-07"]
    assert manifest["instruments"] == ["INS-000001"]


def test_ingest_is_idempotent(env):
    first = _ingest(env)
    second = _ingest(env)
    assert first.snapshot_id == second.snapshot_id
    assert second.reused
    rows = env["registry"].dump()
    assert len(rows) == 1
    bars = pl.read_parquet(normalized_dir("market", "yahoo_chart_api", first.snapshot_id) / "bars.parquet")
    assert bars.height == 4  # no duplicates from the second run


def test_reconciliation_refreshes_on_reuse(env):
    """Reconciliation is a review step, not an ingestion artifact: a reused
    snapshot must carry freshly computed findings and the manifest must be
    updated, so a rule change is visible without a re-download."""
    first = _ingest(env)
    manifest_path = env["root"] / "manifests" / f"{first.snapshot_id}.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["meta"]["reconciliation"]["finding_count"] == 0

    # stale the manifest, then re-ingest (reuse path, no re-download)
    manifest["meta"]["reconciliation"] = {"finding_count": 99, "findings": []}
    manifest_path.write_text(json.dumps(manifest))
    second = _ingest(env)
    assert second.reused
    assert second.reconciliation is not None
    refreshed = json.loads(manifest_path.read_text())
    assert refreshed["meta"]["reconciliation"]["finding_count"] == 0
    assert second.reconciliation["finding_count"] == 0


def test_reuse_refuses_tampered_raw_zone(env):
    """Regression: an ok snapshot must not be reused when the raw zone was
    modified after sealing - the trust anchor is the manifest, so tampering
    the payload AND the seal marker together is still caught."""
    first = _ingest(env)
    raw_file = raw_dir("market", "yahoo_chart_api", first.snapshot_id) / "AAPL.json"
    raw_file.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="tampered"):
        _ingest(env)

    # even tampering the seal marker together with the payload is caught
    marker = raw_dir("market", "yahoo_chart_api", first.snapshot_id) / "IMMUTABLE.json"
    marker.write_text("{}")
    with pytest.raises(RuntimeError, match="tampered"):
        _ingest(env)


def test_reprocessing_incomplete_raw_zone_is_refused(env):
    """Regression: a snapshot whose raw zone holds only SOME of the
    requested files must not be reprocessed into a silently-truncated
    dataset (e.g. a fetch interrupted after 2 of 5 symbols)."""
    payload = _yahoo_payload()
    result = env["pipeline"].ingest_market(
        FakeConnector({"AAPL": payload, "MSFT": payload}),
        ["AAPL", "MSFT"],
        {"AAPL": "INS-000001", "MSFT": "INS-000002"},
        license_ref="test-license",
        request_params={"range": "30y"},
    )
    zone = raw_dir("market", "yahoo_chart_api", result.snapshot_id)
    # simulate an interrupted retry snapshot: failed validation, partial zone
    env["registry"]._con.execute(
        "UPDATE snapshots SET validation_status = 'failed' WHERE snapshot_id = ?",
        [result.snapshot_id],
    )
    (zone / "AAPL.json").unlink()
    (zone / "IMMUTABLE.json").unlink()
    with pytest.raises(RuntimeError, match="INCOMPLETE"):
        env["pipeline"].ingest_market(
            FakeConnector({"AAPL": payload, "MSFT": payload}),
            ["AAPL", "MSFT"],
            {"AAPL": "INS-000001", "MSFT": "INS-000002"},
            license_ref="test-license",
            request_params={"range": "30y"},
        )


def test_stooq_duplicate_dates_are_caught(env):
    """Regression: the stooq path used date_col 'Date' but parse lowercases
    to 'date', so duplicate/continuity checks silently skipped."""
    csv = (
        "Date,Open,High,Low,Close,Volume\n"
        "2020-01-02,100,102,99,101,1000\n"
        "2020-01-02,101,103,100,101.5,1100\n"  # duplicate trading date
    )

    class FakeStooqConnector:
        provider_name = "stooq_csv"

        def fetch(self, request):
            symbol = request["symbol"]
            return [
                RawObject(
                    filename=f"{symbol}.csv", body=csv.encode(),
                    source_uri=f"https://fake.local/stooq/{symbol}",
                    content_type="text/csv",
                    meta={"symbol": symbol, "adjusted": True},
                )
            ]

    result = env["pipeline"].ingest_market(
        FakeStooqConnector(), ["AAPL"], env["symbol_map"],
        license_ref="test-license",
        request_params={"range": "1y"},
    )
    assert result.validation["status"] == "failed"
    codes = {i["code"] for i in result.validation["issues"]}
    assert "duplicate_records" in codes
    assert "price_outside_range" not in codes  # failed for the RIGHT reason
    assert result.snapshot_record["validation_status"] == "failed"


def test_failed_snapshot_raw_zone_is_still_sealed(env, tmp_path):
    """A snapshot that fails validation must still be tamper-evident, so a
    tampered retry can never reprocess into 'ok' unnoticed."""
    bad = _yahoo_payload().replace(b'"close": [101.0, 101.5, 100.5, 102.5]',
                                   b'"close": [101.0, 101.5, 100.5, 500.0]')
    result = _ingest(env, payload=bad)
    assert result.validation["status"] == "failed"
    store = RawStore()
    assert store.verify("market", result.provider, result.snapshot_id)
    raw_file = raw_dir("market", result.provider, result.snapshot_id) / "AAPL.json"
    raw_file.write_bytes(b"tampered")
    assert not store.verify("market", result.provider, result.snapshot_id)


def test_artifact_checksum_refreshes_after_re_derivation(env):
    """Registry artifact checksums must track re-derived bytes, or the
    provenance catalog silently lies about what is on disk."""
    result = _ingest(env)
    path = str(normalized_dir("market", result.provider, result.snapshot_id) / "bars.parquet")
    old = env["registry"].artifact_checksum(result.snapshot_id, path)
    env["registry"].register_artifact(
        result.snapshot_id, "market", path, "0" * 64, 99, "v9.9.9"
    )
    new = env["registry"].artifact_checksum(result.snapshot_id, path)
    assert new == "0" * 64
    assert new != old
    bars_artifacts = [
        a for a in env["registry"].artifacts(result.snapshot_id) if a["path"] == path
    ]
    assert len(bars_artifacts) == 1  # refreshed, not duplicated


def test_reproducibility_same_raw_snapshot_same_output(env):
    result = _ingest(env)
    snapshot_id, provider = result.snapshot_id, result.provider
    raw_dir_path = raw_dir("market", provider, snapshot_id)
    payload = (raw_dir_path / "AAPL.json").read_bytes()
    stored_bars = pl.read_parquet(normalized_dir("market", provider, snapshot_id) / "bars.parquet")
    source_uri = stored_bars["source_uri"][0]

    def derive() -> bytes:
        bars, events = parse_yahoo_chart(payload, "AAPL")
        report = validate_market_bars(bars, "AAPL", date_col="ts", provider=provider)
        assert report.passed
        normalized = normalize_market_bars(
            {"AAPL": {"bars": bars, "events": events}}, env["symbol_map"],
            provider, source_uri, snapshot_id,
        )
        buf = __import__("io").BytesIO()
        normalized["bars"].write_parquet(buf)
        return buf.getvalue()

    first, second = derive(), derive()
    assert first == second  # identical inputs -> identical normalized bytes
    stored = sha256_file(normalized_dir("market", provider, snapshot_id) / "bars.parquet")
    assert sha256_bytes(first) == stored


def test_validation_failure_blocks_normalization(env):
    bad = _yahoo_payload().replace(b'"close": [101.0, 101.5, 100.5, 102.5]',
                                   b'"close": [101.0, 101.5, 100.5, 500.0]')
    result = _ingest(env, payload=bad)
    assert result.validation["status"] == "failed"
    assert not normalized_dir("market", "yahoo_chart_api", result.snapshot_id).exists()
    assert result.snapshot_record["validation_status"] == "failed"
    # a retry with the SAME bad data is content-addressed to the SAME
    # snapshot (no duplicate download, no duplicate records) and still fails
    retry = _ingest(env, payload=bad)
    assert retry.snapshot_id == result.snapshot_id
    assert retry.validation["status"] == "failed"
    assert not retry.reused
    # ...a DIFFERENT request (different date range) is a new snapshot
    good = _ingest(env, payload=_yahoo_payload(), request_params={"range": "5y"})
    assert good.snapshot_id != result.snapshot_id
    assert good.validation["status"] == "ok"


def test_raw_zone_is_immutable_and_detects_tampering(env, tmp_path):
    result = _ingest(env)
    store = RawStore()
    assert store.verify("market", result.provider, result.snapshot_id)
    raw_file = raw_dir("market", result.provider, result.snapshot_id) / "AAPL.json"
    raw_file.write_bytes(b"tampered")
    assert not store.verify("market", result.provider, result.snapshot_id)


def test_manifest_and_dataset_snapshot_are_consistent(env):
    result = _ingest(env)
    ds = build_dataset_snapshot(
        env["registry"], result.snapshot_id, "market", result.provider,
        instrument_count=1,
    )
    assert ds.snapshot_id == result.snapshot_id
    assert ds.validation_status == "ok"
    assert ds.available_from == date(2020, 1, 2)
    assert ds.available_to == date(2020, 1, 7)
    assert ds.row_count == 4
    assert ds.instrument_count == 1
    assert ds.manifest_path and os.path.exists(ds.manifest_path)


def test_pipeline_can_feed_universe_engine(env):
    """Close the Phase 2 loop: UniverseEngine on ingested real-shaped data."""
    from orbit.schemas.instrument import Instrument
    from orbit.universe.engine import UniverseEngine
    from orbit.universe.rules import MembershipRule

    _ingest(env)
    inst = Instrument(
        instrument_id="INS-000001", primary_ticker="AAPL", exchange_id="XNAS",
        name="Apple", security_type="equity", listing_date=date(1980, 12, 12),
    )
    accessor = MarketDataAccessor([inst], "DS-000001", data_root=env["root"])
    rule = MembershipRule(
        rule_id="RULE-001", version="v1", security_types=["equity"],
        exchanges=["XNAS"], max_names=100,
        min_price=None, min_trailing_dollar_volume=None,
    )
    snapshot = UniverseEngine(accessor, rule, data_ref="DS-000001").membership(date(2020, 1, 8))
    assert snapshot.members[0].instrument_id == "INS-000001"
    assert snapshot.members[0].last_close == 102.5
    # as_of before any data -> instrument excluded, not crashed
    early = UniverseEngine(accessor, rule, data_ref="DS-000001").membership(date(1990, 1, 1))
    assert early.members == []


def _sec_payload(cik: int, entity: str, revenue: int) -> bytes:
    return json.dumps(
        {
            "cik": cik,
            "entityName": entity,
            "facts": {
                "us-gaap": {
                    "Revenues": {
                        "units": {
                            "USD": [
                                {"start": "2020-01-01", "end": "2020-12-31",
                                 "val": revenue, "accn": f"0000{cik}-20-000001",
                                 "fy": 2020, "fp": "FY", "form": "10-K",
                                 "filed": "2021-02-01", "frame": "CY2020"},
                            ]
                        }
                    }
                }
            },
        }
    ).encode()


class FakeSecConnector:
    provider_name = "sec_edgar_companyfacts"

    def __init__(self, payloads: dict[int, bytes]):
        self._payloads = payloads

    def fetch(self, request):
        cik = int(request["cik"])
        return [
            RawObject(
                filename=f"cik{cik:010d}_companyfacts.json",
                body=self._payloads[cik],
                source_uri=f"https://fake.local/sec/{cik}",
                content_type="application/json",
                meta={"cik": cik, "entity_name": f"entity-{cik}"},
            )
        ]


def test_sec_reproduction_is_order_independent(env):
    """The SEC snapshot must be byte-reproducible from disk: the first pass
    sorts raw files (canonical order), so a re-derivation from globbed disk
    files matches the stored artifact exactly."""
    from orbit.ingestion.normalizers.fundamentals import normalize_sec_facts
    from orbit.ingestion.parsing import parse_sec_companyfacts
    from orbit.ingestion.validators import validate_sec_facts

    payloads = {
        320193: _sec_payload(320193, "Apple Inc.", 111_000_000_000),
        200406: _sec_payload(200406, "Johnson & Johnson", 20_000_000_000),
    }
    # request order deliberately UNSORTED vs filename order
    result = env["pipeline"].ingest_sec(
        FakeSecConnector(payloads), [320193, 200406], license_ref="test-license"
    )
    assert result.validation["status"] == "ok"
    assert result.domain == "sec"

    src_dir = raw_dir("sec", result.provider, result.snapshot_id)
    files = sorted(p for p in src_dir.glob("*.json") if p.name != "IMMUTABLE.json")
    frames = []
    for f in files:
        facts = parse_sec_companyfacts(f.read_bytes())
        cik = int(f.stem[3:13])
        assert validate_sec_facts(facts, cik).passed
        frames.append(facts)
    rederived = normalize_sec_facts(pl.concat(frames), result.provider, result.snapshot_id)
    stored = pl.read_parquet(
        normalized_dir("fundamentals", result.provider, result.snapshot_id) / "facts.parquet"
    )
    # stored rows come from cik 200406 (sorted filename first), NOT request
    # order (320193 first) - that is the whole point of canonical ordering
    assert stored["cik"].to_list() == [200406, 320193]
    assert rederived.equals(stored)
    import io

    buf = io.BytesIO()
    rederived.write_parquet(buf)
    stored_bytes = open(
        normalized_dir("fundamentals", result.provider, result.snapshot_id) / "facts.parquet", "rb"
    ).read()
    assert buf.getvalue() == stored_bytes