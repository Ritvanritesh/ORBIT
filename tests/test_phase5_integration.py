"""Phase 4 integration tests for Phase 5: labels must live EXACTLY on the
temporal truth layer.

  1. ENTRY-BAR AGREEMENT - for a decision at t, the LabelEngine's reference
     session is the same most-recent-completed bar the TemporalTruthEngine
     snapshot at t allows (both use the strict session-close rule).
  2. AS-PUBLISHED AGREEMENT - the label's audit close equals the Phase 4
     snapshot's as-published close for the same bar (the label engine
     rebuilds as-published prices with the same events the temporal layer
     uses).
  3. NO LEAKAGE - a point-in-time snapshot at t never contains a label
     record; a label row's outcome timestamp is always strictly after its
     decision time; feature bars never include the outcome window.
  4. SEPARATE ARTIFACTS - a LabelSnapshot is a distinct artifact from a
     PointInTimeSnapshot (label digest != snapshot digest, distinct record
     kinds), so a leaked merged dataset is detectable.
  5. POST_EVENT = PHASE 4 AVAILABILITY - the PEAD entry session is the
     first session after the filing's point-in-time availability instant.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

import polars as pl
import pytest

from orbit.ingestion.normalizers.market import normalize_market_bars
from orbit.ingestion.parsing import parse_yahoo_chart
from orbit.labels import (
    AnchorMode,
    LabelEngine,
    LabelSnapshot,
    ReturnConvention,
    build_seed_label_registry,
)
from orbit.labels.engine import LABEL_OUTPUT_COLUMNS
from orbit.temporal.engine import TemporalTruthEngine
from orbit.temporal.snapshot import PointInTimeSnapshot, TemporalSource
from orbit.temporal.features import completed_bars
from orbit.temporal.times import session_close_utc

WINTER_CLOSE = timezone.utc


def _market_payload():
    # 10 sessions: 01-02, 01-03, 01-06..01-10, 01-13..01-17 (Jan 2020, EST)
    ts = [
        int((datetime(2020, 1, 2, 14, 30) - datetime(1970, 1, 1)).total_seconds()),
        int((datetime(2020, 1, 3, 14, 30) - datetime(1970, 1, 1)).total_seconds()),
        int((datetime(2020, 1, 6, 14, 30) - datetime(1970, 1, 1)).total_seconds()),
        int((datetime(2020, 1, 7, 14, 30) - datetime(1970, 1, 1)).total_seconds()),
        int((datetime(2020, 1, 8, 14, 30) - datetime(1970, 1, 1)).total_seconds()),
        int((datetime(2020, 1, 9, 14, 30) - datetime(1970, 1, 1)).total_seconds()),
        int((datetime(2020, 1, 10, 14, 30) - datetime(1970, 1, 1)).total_seconds()),
        int((datetime(2020, 1, 13, 14, 30) - datetime(1970, 1, 1)).total_seconds()),
        int((datetime(2020, 1, 14, 14, 30) - datetime(1970, 1, 1)).total_seconds()),
        int((datetime(2020, 1, 15, 14, 30) - datetime(1970, 1, 1)).total_seconds()),
        int((datetime(2020, 1, 16, 14, 30) - datetime(1970, 1, 1)).total_seconds()),
        int((datetime(2020, 1, 17, 14, 30) - datetime(1970, 1, 1)).total_seconds()),
    ]
    n = len(ts)
    aapl_close = [101.0 + 0.5 * i for i in range(n)]
    spy_close = [300.0 + 0.5 * i for i in range(n)]

    def _chart(closes):
        return {
            "chart": {
                "result": [
                    {
                        "meta": {"currency": "USD", "exchangeName": "NMS"},
                        "timestamp": ts,
                        "indicators": {
                            "quote": [{
                                "open": closes,
                                "high": [c + 1 for c in closes],
                                "low": [c - 1 for c in closes],
                                "close": closes,
                                "volume": [1000] * len(closes),
                            }],
                            "adjclose": [{"adjclose": closes}],
                        },
                        "events": {"dividends": {}, "splits": {}},
                    }
                ],
                "error": None,
            }
        }

    return {
        "AAPL": json.dumps(_chart(aapl_close)).encode(),
        "SPY": json.dumps(_chart(spy_close)).encode(),
    }


@pytest.fixture
def market() -> dict[str, pl.DataFrame]:
    all_bars, all_events = [], None
    for symbol, payload in _market_payload().items():
        bars, events = parse_yahoo_chart(payload, symbol)
        inst = {"AAPL": "INS-000001", "SPY": "SPY"}[symbol]
        normalized = normalize_market_bars(
            {symbol: {"bars": bars, "events": events}}, {symbol: inst},
            "yahoo_chart_api", "https://example/chart", "DS-000001",
        )
        all_bars.append(normalized["bars"])
        if normalized.get("events") is not None:
            all_events = normalized["events"]
    return {"bars": pl.concat(all_bars), "events": all_events}


@pytest.fixture
def data_root(tmp_path: Path, market: dict[str, pl.DataFrame]) -> Path:
    d = tmp_path / "normalized" / "market" / "yahoo_chart_api" / "DS-000001"
    d.mkdir(parents=True, exist_ok=True)
    market["bars"].write_parquet(d / "bars.parquet")
    events = market.get("events")
    if events is not None and events.height:
        events.write_parquet(d / "events.parquet")
    return tmp_path


def _temporal_sources(root: Path) -> list[TemporalSource]:
    return [
        TemporalSource(
            snapshot_id="DS-000001", domain="market", provider="yahoo_chart_api",
            checksum="a" * 64, manifest_path="m1.json",
            ingest_time=datetime(2020, 1, 11, 0, 0),
            artifact_paths=[str(root / "normalized" / "market" / "yahoo_chart_api" / "DS-000001" / "bars.parquet")],
        )
    ]


def _bars_frame(market: dict[str, pl.DataFrame]) -> pl.DataFrame:
    return market["bars"]


def _events_frame(market: dict[str, pl.DataFrame]) -> pl.DataFrame | None:
    return market.get("events")


def _seed_contracts():
    return build_seed_label_registry()


# ----------------------------------------- 1. entry-bar agreement

def test_entry_bar_agrees_with_temporal_snapshot(data_root, market):
    t = datetime(2020, 1, 8, 21, 0, 1)  # one microsecond after the 01-08 close
    bars = _bars_frame(market)
    label_engine = LabelEngine(bars, events=_events_frame(market))
    entry = label_engine.entry_bar("INS-000001", t)
    assert entry["trade_date"] == date(2020, 1, 8)

    temporal = TemporalTruthEngine(sources=_temporal_sources(data_root))
    snap = temporal.snapshot(t)
    assert isinstance(snap, PointInTimeSnapshot)
    allowed = snap.records.filter(pl.col("kind") == "bar")
    payloads = [
        json.loads(j) for j in allowed["payload_json"].to_list()
    ]
    latest = max(p["trade_date"] for p in payloads)
    assert latest == entry["trade_date"].isoformat()
    # the strict boundary: exactly AT the close, the temporal layer and the
    # label engine both refuse the same-day bar
    t_at = datetime(2020, 1, 8, 21, 0, 0)
    assert label_engine.entry_bar("INS-000001", t_at)["trade_date"] == date(2020, 1, 7)
    snap_at = temporal.snapshot(t_at)
    latest_at = max(
        json.loads(j)["trade_date"]
        for j in snap_at.records.filter(pl.col("kind") == "bar")["payload_json"].to_list()
    )
    assert latest_at == "2020-01-07"


# ------------------------------------------ 2. as-published agreement

def test_as_published_close_agrees_with_temporal_layer(data_root, market):
    t = datetime(2020, 1, 8, 21, 0, 1)
    bars = _bars_frame(market)
    row = LabelEngine(bars, events=_events_frame(market)).compute_one(
        _seed_contracts().definition("LAB-001"),
        "INS-000001", t,
    )
    snap = TemporalTruthEngine(sources=_temporal_sources(data_root)).snapshot(t)
    bar_rows = snap.records.filter(pl.col("kind") == "bar")
    entry_payload = next(
        json.loads(j) for j in bar_rows["payload_json"].to_list()
        if json.loads(j)["trade_date"] == "2020-01-08"
    )
    # the snapshot carries as-published prices (raw, unadjusted basis)
    assert row["entry_close_as_published"] == pytest.approx(
        entry_payload["close"] * 1.0
    )
    # stored close equals the canonical adjusted close
    assert row["entry_close"] == pytest.approx(
        bars.filter(pl.col("trade_date") == date(2020, 1, 8))["close"][0]
    )


# --------------------------------------------------- 3. no leakage

def test_point_in_time_snapshot_never_contains_labels(data_root):
    t = datetime(2020, 1, 8, 21, 0, 1)
    snap = TemporalTruthEngine(sources=_temporal_sources(data_root)).snapshot(t)
    assert snap.records.height > 0
    assert snap.records["kind"].unique().to_list() == ["bar"]
    assert "label" not in snap.records["kind"].to_list()
    # no label columns leak into the snapshot schema
    assert "outcome_value" not in snap.records.columns
    assert "label_id" not in snap.records.columns


def test_label_outcome_is_always_strictly_after_the_decision(data_root, market):
    bars = _bars_frame(market)
    reg = _seed_contracts()
    frame = LabelEngine(bars, events=_events_frame(market)).compute(
        reg.definition("LAB-001"),
        [
            {"instrument_id": "INS-000001", "decision_time": datetime(2020, 1, 6, 21, 0, 1)},
            {"instrument_id": "INS-000001", "decision_time": datetime(2020, 1, 7, 21, 0, 1)},
        ],
    )
    for r in frame.to_dicts():
        assert r["outcome_status"] == "available"
        assert r["outcome_timestamp"] > r["decision_time"]
        # the outcome is finalized at a session close, never intraday
        assert r["outcome_timestamp"] == session_close_utc(r["outcome_session"])


def test_features_at_decision_time_never_see_the_outcome_window(data_root, market):
    t = datetime(2020, 1, 6, 21, 0, 1)
    bars = _bars_frame(market)
    # feature bars available at the decision time (Phase 4 completed_bars)
    feature_bars = completed_bars(bars, t, window=3, instrument_id="INS-000001")
    assert set(feature_bars["trade_date"].to_list()) == {
        date(2020, 1, 2), date(2020, 1, 3), date(2020, 1, 6),
    }
    # the label for the same decision resolves over the NEXT five sessions
    reg = _seed_contracts()
    row = LabelEngine(bars, events=_events_frame(market)).compute_one(
        reg.definition("LAB-001"), "INS-000001", t,
    )
    assert row["entry_session"] == date(2020, 1, 6)
    assert row["outcome_session"] == date(2020, 1, 13)
    outcome_dates = {
        r["trade_date"] for r in bars.filter(
            pl.col("trade_date") > date(2020, 1, 6)
        ).to_dicts()
    }
    # the outcome session is strictly in the future: no feature ever saw it
    assert set(feature_bars["trade_date"].to_list()).isdisjoint(
        {date(2020, 1, 13)}
    )
    assert row["outcome_session"] in outcome_dates  # outcome is future data


# ---------------------------------------------- 4. separate artifacts

def test_label_snapshot_is_distinct_from_point_in_time_snapshot(data_root, market):
    t = datetime(2020, 1, 8, 21, 0, 1)
    bars = _bars_frame(market)
    reg = _seed_contracts()
    contract = reg.definition("LAB-001")
    frame = LabelEngine(bars, events=_events_frame(market)).compute(
        contract,
        [{"instrument_id": "INS-000001", "decision_time": t}],
    )
    label_snap = LabelSnapshot(
        label_id=contract.label_id, version=contract.version,
        contract_digest=contract.content_hash(), engine_version="v1.0.0",
        data_refs=["DS-000001"], records=frame,
    )
    pt_snap = TemporalTruthEngine(sources=_temporal_sources(data_root)).snapshot(t)
    # different artifacts: different record kinds, different identities
    assert pt_snap.records["kind"].unique().to_list() == ["bar"]
    assert label_snap.records.columns == [c for c, _ in LABEL_OUTPUT_COLUMNS]
    assert label_snap.row_count() == 1
    assert len(label_snap.content_digest) == 64
    # a label digest is not a point-in-time digest (no cross-contamination)
    assert label_snap.content_digest != pt_snap.content_digest


# ------------------------------------------- 5. post_event availability

def test_post_event_entry_uses_phase4_availability_convention(data_root, market):
    # a filing that becomes available on 2020-01-08 00:00 UTC (midnight:
    # the day after it was filed) anchors the PEAD window on 01-08
    bars = _bars_frame(market)
    reg = _seed_contracts()
    contract = reg.definition("LAB-003")
    assert contract.anchor_mode == AnchorMode.POST_EVENT
    assert contract.return_convention == ReturnConvention.SIMPLE_TOTAL_RETURN
    anchor = datetime(2020, 1, 8, 0, 0, 0)
    row = LabelEngine(bars, events=_events_frame(market)).compute_one(
        contract, "INS-000001",
        datetime(2020, 1, 8, 0, 0, 0), anchor_instant=anchor,
    )
    assert row["outcome_status"] == "available"
    assert row["entry_session"] == date(2020, 1, 8)
    # Phase 4 confirms: at the anchor instant, the 01-08 bar is NOT yet
    # available (its close is 21:00), but it is the FIRST session whose
    # close is strictly after the anchor - exactly the PEAD convention
    temporal = TemporalTruthEngine(sources=_temporal_sources(data_root))
    snap = temporal.snapshot(anchor)
    latest = max(
        json.loads(j)["trade_date"]
        for j in snap.records.filter(pl.col("kind") == "bar")["payload_json"].to_list()
    )
    assert latest == "2020-01-07"
    assert row["entry_session"].isoformat() > latest