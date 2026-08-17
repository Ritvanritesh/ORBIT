"""Phase 3 unit tests: checksums, storage immutability, parsing, validation,
normalization, reconciliation, and the snapshot accessor.

All tests are hermetic (fixtures, no network). A synthetic FakeConnector
drives the full pipeline in test_phase3_pipeline.py.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, timedelta

import polars as pl
import pytest

from orbit.ingestion.checksums import request_fingerprint, sha256_bytes, sha256_file
from orbit.ingestion.downloaders.http import DownloadError, fetch_bytes
from orbit.ingestion.normalizers.market import (
    BAR_COLUMNS,
    MARKET_SCHEMA_VERSION,
    build_corporate_actions,
    normalize_market_bars,
)
from orbit.ingestion.parsing import (
    parse_fred_csv,
    parse_sec_companyfacts,
    parse_stooq_csv,
    parse_yahoo_chart,
)
from orbit.ingestion.pipeline import IngestResult
from orbit.ingestion.reconciliation import reconcile_market
from orbit.ingestion.storage import RawStore
from orbit.ingestion.validators import (
    ValidationReport,
    validate_continuity,
    validate_duplicates,
    validate_market_bars,
    validate_numerical,
    validate_structure,
)
from orbit.schemas.data import MarketBar


# ------------------------------------------------------------------ checksums


def test_sha256_roundtrip():
    a = b"hello orbit"
    assert sha256_bytes(a) == hashlib.sha256(a).hexdigest()
    assert sha256_bytes(b"hello orbit!") != sha256_bytes(a)


def test_request_fingerprint_is_deterministic_and_order_insensitive():
    f1 = request_fingerprint({"a": 1, "b": [2, 3]})
    f2 = request_fingerprint({"b": [2, 3], "a": 1})
    assert f1 == f2
    assert request_fingerprint({"a": 1, "b": [2, 4]}) != f1


def test_sha256_file_matches_bytes(tmp_path):
    p = tmp_path / "f.bin"
    p.write_bytes(b"payload")
    assert sha256_file(p) == sha256_bytes(b"payload")


# -------------------------------------------------------------------- storage


def test_http_fetch_retries_on_429_and_5xx(monkeypatch):
    import time
    import urllib.request

    class _Resp:
        def __init__(self, status, body=b"ok"):
            self.status = status
            self._body = body
            self.headers = {"Content-Type": "text/plain"}

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return self._body

        def geturl(self):
            return "https://provider.local/x"

    responses = [_Resp(429), _Resp(503), _Resp(200, b"final")]
    calls = []

    def fake_urlopen(req, timeout=None):
        calls.append(req.full_url)
        r = responses.pop(0)
        if r.status == 200:
            return r
        raise urllib.error.HTTPError(req.full_url, r.status, "retry", {}, None)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(time, "sleep", lambda s: None)
    out = fetch_bytes("https://provider.local/x", retries=3, backoff_seconds=1)
    assert out.body == b"final"
    assert len(calls) == 3  # 429 + 503 retried, then success


def test_http_fetch_gives_up_after_retries(monkeypatch):
    import time
    import urllib.request

    class _Resp:
        status = 429
        headers = {}

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return b""

        def geturl(self):
            return "https://provider.local/x"

    def fake_urlopen(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, 429, "rate limited", {}, None)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(time, "sleep", lambda s: None)
    with pytest.raises(DownloadError):
        fetch_bytes("https://provider.local/x", retries=2)


def test_http_fetch_fails_fast_on_404(monkeypatch):
    import urllib.request

    def fake_urlopen(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, 404, "missing", {}, None)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(DownloadError, match="404"):
        fetch_bytes("https://provider.local/missing", retries=3)


def test_raw_store_write_once_does_not_overwrite(tmp_path):
    store = RawStore(tmp_path)
    path, c1 = store.write_once("market", "p", "DS-000001", "a.csv", b"v1")
    path2, c2 = store.write_once("market", "p", "DS-000001", "a.csv", b"v2-tampered")
    assert path == path2
    assert path.read_bytes() == b"v1"
    assert c1 == c2


def test_raw_store_seal_detects_drift(tmp_path):
    store = RawStore(tmp_path)
    store.write_once("market", "p", "DS-000001", "a.csv", b"v1")
    checksum = sha256_bytes(b"v1")
    store.seal("market", "p", "DS-000001", {"a.csv": checksum})
    assert store.verify("market", "p", "DS-000001")
    (tmp_path / "raw" / "market" / "p" / "DS-000001" / "a.csv").write_bytes(b"CHANGED")
    assert not store.verify("market", "p", "DS-000001")


def test_raw_store_seal_rejects_missing_file(tmp_path):
    store = RawStore(tmp_path)
    with pytest.raises(RuntimeError):
        store.seal("market", "p", "DS-000002", {"a.csv": sha256_bytes(b"x")})


def test_raw_store_verify_false_without_marker(tmp_path):
    store = RawStore(tmp_path)
    store.write_once("market", "p", "DS-000003", "a.csv", b"x")
    assert not store.verify("market", "p", "DS-000003")


# -------------------------------------------------------------------- parsing


def _yahoo_payload():
    timestamps = [
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
                        "timestamp": timestamps,
                        "indicators": {
                            "quote": [
                                {
                                    "open": [100.0, 101.0, 99.0, 102.0],
                                    "high": [102.0, 103.0, 100.5, 103.0],
                                    "low": [99.0, 100.0, 98.0, 101.0],
                                    "close": [101.0, 101.5, 100.5, 102.5],
                                    "volume": [1000, 1100, 900, 1200],
                                }
                            ],
                            "adjclose": [{"adjclose": [100.0, 100.5, 99.5, 101.5]}],
                        },
                        "events": {
                            "splits": {
                                "1578000000": {"date": 1578000000, "numerator": 2.0, "denominator": 1.0, "splitRatio": "2:1"}
                            },
                            "dividends": {
                                "1577000000": {"date": 1577000000, "amount": 0.2}
                            },
                        },
                    }
                ],
                "error": None,
            }
        }
    ).encode()


def test_parse_yahoo_chart():
    bars, events = parse_yahoo_chart(_yahoo_payload(), "AAPL")
    assert bars.height == 4
    assert bars.columns == [
        "symbol", "ts_epoch", "open", "high", "low", "close", "volume",
        "adjclose", "currency", "exchange_name", "ts",
    ]
    assert events.height == 2
    kinds = events["kind"].to_list()
    assert sorted(kinds) == ["dividends", "splits"]


def test_parse_stooq_csv():
    csv = "Date,Open,High,Low,Close,Volume\n2020-01-02,100,102,99,101,1000\n2020-01-03,101,103,100,99.5,1100\n"
    bars, events = parse_stooq_csv(csv.encode(), "AAPL")
    assert bars.height == 2
    assert bars["symbol"].to_list() == ["AAPL", "AAPL"]
    assert events.height == 0  # stooq ships no split/dividend events


def test_parse_fred_csv_keeps_missing_as_null():
    csv = "observation_date,DFF\n1954-07-01,1.13\n1954-07-02,.\n1954-07-03,1.25\n"
    df = parse_fred_csv(csv.encode(), "DFF")
    assert df["value"].to_list() == [1.13, None, 1.25]
    assert df["series_id"].to_list() == ["DFF"] * 3


def test_parse_sec_companyfacts():
    payload = json.dumps(
        {
            "cik": 320193,
            "entityName": "Apple Inc.",
            "facts": {
                "us-gaap": {
                    "Revenues": {
                        "label": "Revenues",
                        "units": {
                            "USD": [
                                {"start": "2020-09-26", "end": "2020-12-26", "val": 111439000000,
                                 "accn": "0000320193-21-000010", "fy": 2021, "fp": "Q1",
                                 "form": "10-Q", "filed": "2021-01-27", "frame": "CY2020Q4"},
                            ]
                        },
                    }
                }
            },
        }
    ).encode()
    df = parse_sec_companyfacts(payload)
    assert df.height == 1
    row = df.row(0, named=True)
    assert row["fact"] == "Revenues"
    assert row["filed"] == "2021-01-27"
    assert row["val"] == "111439000000"


# ---------------------------------------------------------------- validation


def _market_df(rows):
    return pl.DataFrame(rows, schema={"open": pl.Float64, "high": pl.Float64, "low": pl.Float64, "close": pl.Float64, "volume": pl.Int64, "date": pl.Date})


def test_structural_missing_column_is_error():
    report = ValidationReport("x")
    df = pl.DataFrame({"open": [1.0], "high": [2.0]})
    validate_structure(df, report, ["open", "high", "low", "close"], numeric_columns=["open", "high", "low", "close"])
    assert not report.passed
    assert report.errors[0].code == "missing_columns"


def test_numerical_high_below_low_is_error():
    report = ValidationReport("x")
    df = _market_df([{"open": 8, "high": 8, "low": 9, "close": 8, "volume": 1, "date": date(2020, 1, 2)}])
    validate_numerical(df, report, symbol="T")
    assert any(i.code == "high_below_low" for i in report.errors)


def test_numerical_negative_volume_is_error():
    report = ValidationReport("x")
    df = _market_df([{"open": 10, "high": 11, "low": 9, "close": 10, "volume": -5, "date": date(2020, 1, 2)}])
    validate_numerical(df, report, symbol="T")
    assert report.errors[0].code == "negative_volume"


def test_duplicate_dates_are_errors():
    report = ValidationReport("x")
    df = _market_df(
        [
            {"open": 10, "high": 11, "low": 9, "close": 10, "volume": 5, "date": date(2020, 1, 2)},
            {"open": 11, "high": 12, "low": 10, "close": 11, "volume": 5, "date": date(2020, 1, 2)},
        ]
    )
    validate_duplicates(df, report, keys=["date"], context="T")
    assert report.errors[0].code == "duplicate_records"


def test_continuity_flags_gap_and_missing_weekday():
    report = ValidationReport("x")
    dates = [date(2020, 1, 2), date(2020, 2, 3)]  # ~32 calendar days apart
    df = _market_df([{"open": 10, "high": 11, "low": 9, "close": 10, "volume": 5, "date": d} for d in dates])
    validate_continuity(df, report, date_col="date", symbol="T")
    codes = {i.code for i in report.issues}
    assert "suspicious_gap" in codes
    assert "missing_weekday" in codes


def test_market_validation_accepts_clean_data():
    bars, _ = parse_yahoo_chart(_yahoo_payload(), "AAPL")
    report = validate_market_bars(bars, "AAPL", date_col="ts", provider="yahoo_chart_api")
    assert report.passed, report.errors


# --------------------------------------------------------------- normalization


def test_normalize_yahoo_bars_schema_and_timezone_rule():
    bars, events = parse_yahoo_chart(_yahoo_payload(), "AAPL")
    normalized = normalize_market_bars(
        {"AAPL": {"bars": bars, "events": events}}, {"AAPL": "INS-000001"},
        "yahoo_chart_api", "https://example/chart/AAPL", "DS-000001",
    )
    out = normalized["bars"]
    assert list(out.columns) == list(BAR_COLUMNS)
    assert out.height == 4
    assert out["instrument_id"].to_list() == ["INS-000001"] * 4
    # 2020-01-02 14:30 UTC -> 09:30 America/New_York session date 2020-01-02
    assert out["trade_date"].to_list() == [date(2020, 1, 2), date(2020, 1, 3), date(2020, 1, 6), date(2020, 1, 7)]
    assert out["adjustment"].unique().to_list() == ["split_adjusted"]
    assert out["snapshot_id"].unique().to_list() == ["DS-000001"]
    assert normalized["events"].height == 2
    ev_sessions = {ts.astimezone(__import__("zoneinfo").ZoneInfo("America/New_York")).date() for ts in normalized["events"]["ts"].to_list()}
    assert ev_sessions == {date(2019, 12, 21), date(2020, 1, 2)}


def test_normalize_stooq_bars_uses_local_close_rule():
    csv = "Date,Open,High,Low,Close,Volume\n2020-01-02,100,102,99,101,1000\n"
    bars, events = parse_stooq_csv(csv.encode(), "AAPL")
    normalized = normalize_market_bars(
        {"AAPL": {"bars": bars, "events": events}}, {"AAPL": "INS-000001"},
        "stooq_csv", "https://stooq.com/q/d/l/", "DS-000002",
    )
    out = normalized["bars"]
    # 2020-01-02 16:00 America/New_York = 2020-01-02 21:00 UTC
    assert out["ts_utc"][0] == datetime(2020, 1, 2, 21, 0)
    assert out["trade_date"][0] == date(2020, 1, 2)


def test_normalized_row_validates_as_marketbar():
    bars, events = parse_yahoo_chart(_yahoo_payload(), "AAPL")
    normalized = normalize_market_bars(
        {"AAPL": {"bars": bars, "events": events}}, {"AAPL": "INS-000001"},
        "yahoo_chart_api", "u", "DS-000001",
    )
    row = normalized["bars"].row(0, named=True)
    MarketBar(
        instrument_id=row["instrument_id"], ts=row["ts_utc"], open=row["open"],
        high=row["high"], low=row["low"], close=row["close"], volume=row["volume"],
    )


def test_corporate_actions_materialize_from_events():
    """Roadmap: corporate actions (splits/dividends) surface as records."""
    from orbit.schemas.instrument import CorporateAction

    bars, events = parse_yahoo_chart(_yahoo_payload(), "AAPL")
    normalized = normalize_market_bars(
        {"AAPL": {"bars": bars, "events": events}}, {"AAPL": "INS-000001"},
        "yahoo_chart_api", "u", "DS-000001",
    )
    actions = build_corporate_actions(normalized["events"], "DS-000001", "yahoo_chart_api")
    assert len(actions) == 2
    kinds = sorted(a["action_type"] for a in actions)
    assert kinds == ["dividend", "split"]
    for a in actions:
        # every record passes the Phase 2 CorporateAction contract
        CorporateAction(**a)
        assert a["source"] == "yahoo_chart_api/DS-000001"
    split = [a for a in actions if a["action_type"] == "split"][0]
    assert split["ratio"] == 2.0
    assert split["effective_date"] == "2020-01-02"  # NY session date of event ts
    dividend = [a for a in actions if a["action_type"] == "dividend"][0]
    assert dividend["ratio"] == 0.2
    # deterministic: same input -> same ids
    again = build_corporate_actions(normalized["events"], "DS-000001", "yahoo_chart_api")
    assert [a["action_id"] for a in again] == [a["action_id"] for a in actions]


# ------------------------------------------------------------- reconciliation


def _recon_bars(closes, adjustment="split_adjusted"):
    return pl.DataFrame(
        {
            "instrument_id": ["INS-000001"] * len(closes),
            "trade_date": [date(2020, 1, 2) + timedelta(days=i) for i in range(len(closes))],
            "close": closes,
            "adjustment": [adjustment] * len(closes),
        }
    )


def _recon_events(kind="splits", ts=None, ratio=2.0):
    return pl.DataFrame(
        {
            "instrument_id": ["INS-000001"],
            "kind": [kind],
            "ts": [ts or datetime(2020, 1, 3)],
            "ratio": [ratio],
        }
    )


def test_reconciliation_flags_unexplained_jump():
    bars = _recon_bars([100.0, 150.0])  # +50% overnight, no event
    report = reconcile_market(bars, pl.DataFrame(schema={"instrument_id": pl.Utf8, "kind": pl.Utf8, "ts": pl.Datetime("us"), "ratio": pl.Float64}))
    codes = {f.code for f in report.findings}
    assert "unexplained_discontinuity" in codes


def test_reconciliation_accepts_split_adjusted_series():
    """Regression: a split-adjusted series is CONTINUOUS across the ex-date.
    The pre-fix check expected close_after = close_before * ratio, which
    flagged every real split (the dev run recorded 15 false findings).
    """
    bars = _recon_bars([100.0, 100.5, 100.6])  # continuous across the 2:1 split
    events = _recon_events(ratio=2.0)
    report = reconcile_market(bars, events)
    assert not report.findings


def test_reconciliation_accepts_unadjusted_series_with_ratio_jump():
    bars = _recon_bars([100.0, 50.0, 50.5], adjustment="unadjusted")  # 2:1 raw
    events = _recon_events(ratio=2.0)
    report = reconcile_market(bars, events)
    codes = {f.code for f in report.findings}
    assert "split_adjustment_inconsistency" not in codes
    assert "explained_by_corporate_action" in codes  # the -50% move is explained


def test_reconciliation_flags_unadjusted_series_that_is_continuous():
    bars = _recon_bars([100.0, 100.5, 100.6], adjustment="unadjusted")  # no ratio jump
    events = _recon_events(ratio=2.0)
    report = reconcile_market(bars, events)
    assert any(f.code == "split_adjustment_inconsistency" for f in report.findings)


def test_reconciliation_real_shaped_split_does_not_false_positive():
    """AAPL 2020-08-31 4:1 split shape: continuous adjusted closes, 4.0 ratio."""
    bars = _recon_bars([124.8, 129.0, 127.5], adjustment="split_adjusted")
    events = _recon_events(ratio=4.0, ts=datetime(2020, 8, 31))
    report = reconcile_market(bars, events)
    assert not report.findings


def test_reconciliation_split_does_not_explain_move_on_adjusted_series():
    """A split causes NO price jump in a split-adjusted series, so a large
    overnight move near a split must stay unexplained there. The pre-fix
    check counted ANY corporate action in the window, so a split (or a
    tiny dividend) wrongly 'explained' the move."""
    bars = _recon_bars([100.0, 130.0, 129.0])  # +30% on the split ex-date
    events = _recon_events(ratio=2.0)
    report = reconcile_market(bars, events)
    codes = {f.code for f in report.findings}
    assert "unexplained_discontinuity" in codes
    assert "explained_by_corporate_action" not in codes


def test_reconciliation_dividend_does_not_explain_large_move():
    """A small cash dividend never moves price 25%+; a move near one must
    be unexplained, not attributed to it."""
    bars = _recon_bars([100.0, 130.0, 129.0])
    events = _recon_events(kind="dividends", ratio=0.5, ts=datetime(2020, 1, 3))
    report = reconcile_market(bars, events)
    codes = {f.code for f in report.findings}
    assert "unexplained_discontinuity" in codes
    assert "explained_by_corporate_action" not in codes


def test_reconciliation_unadjusted_split_still_explains_move():
    """On an UNADJUSTED series a split mechanically halves the price, so the
    same move IS explained there."""
    bars = _recon_bars([100.0, 50.0, 50.5], adjustment="unadjusted")
    events = _recon_events(ratio=2.0)
    report = reconcile_market(bars, events)
    codes = {f.code for f in report.findings}
    assert "explained_by_corporate_action" in codes


def test_reconciliation_flags_volume_spike():
    vols = [100_000_000] * 59 + [7_400_000_000]  # 74x the trailing median
    bars = pl.DataFrame(
        {
            "instrument_id": ["INS-000001"] * 60,
            "trade_date": [date(2020, 1, 2) + timedelta(days=i) for i in range(60)],
            "close": [100.0] * 60,
            "volume": vols,
            "adjustment": ["split_adjusted"] * 60,
        }
    )
    report = reconcile_market(bars, pl.DataFrame(schema={"instrument_id": pl.Utf8, "kind": pl.Utf8, "ts": pl.Datetime("us"), "ratio": pl.Float64}))
    assert any(f.code == "volume_spike" for f in report.findings)
    spike = [f for f in report.findings if f.code == "volume_spike"]
    assert spike[0].event_date == (date(2020, 1, 2) + timedelta(days=59)).isoformat()


def test_reconciliation_collapses_contiguous_spikes_not_distant_ones():
    """Regression: run collapsing must merge CONSECUTIVE flagged days but
    keep distant spikes separate. The pre-fix check subtracted dates in the
    wrong order, so a negative gap (-212 days) satisfied `<= 3` and every
    spike of an instrument collapsed into one finding spanning years."""
    vols = [100_000_000] * 59 + [7_400_000_000, 7_100_000_000] + [100_000_000] * 4 + [7_000_000_000]
    bars = pl.DataFrame(
        {
            "instrument_id": ["INS-000001"] * 66,
            "trade_date": [date(2020, 1, 2) + timedelta(days=i) for i in range(66)],
            "close": [100.0] * 66,
            "volume": vols,
            "adjustment": ["split_adjusted"] * 66,
        }
    )
    report = reconcile_market(bars, pl.DataFrame(schema={"instrument_id": pl.Utf8, "kind": pl.Utf8, "ts": pl.Datetime("us"), "ratio": pl.Float64}))
    spikes = [f for f in report.findings if f.code == "volume_spike"]
    assert len(spikes) == 2  # days 59-60 merged; day 65 separate
    merged = spikes[0].message
    assert ".." in merged  # range form for the merged run
    assert spikes[1].event_date == (date(2020, 1, 2) + timedelta(days=65)).isoformat()


# ------------------------------------------------------------------ accessor


def test_snapshot_accessor_is_strictly_lagged(tmp_path):
    from orbit.ingestion.snapshot import MarketDataAccessor
    from orbit.schemas.instrument import Instrument

    out = tmp_path / "normalized" / "market" / "yahoo_chart_api" / "DS-000001"
    out.mkdir(parents=True)
    bars, events = parse_yahoo_chart(_yahoo_payload(), "AAPL")
    normalized = normalize_market_bars(
        {"AAPL": {"bars": bars, "events": events}}, {"AAPL": "INS-000001"},
        "yahoo_chart_api", "u", "DS-000001",
    )
    normalized["bars"].write_parquet(out / "bars.parquet")
    inst = Instrument(
        instrument_id="INS-000001", primary_ticker="AAPL", exchange_id="XNAS",
        name="Apple", security_type="equity", listing_date=date(1980, 12, 12),
    )
    accessor = MarketDataAccessor([inst], "DS-000001", data_root=tmp_path)

    as_of = date(2020, 1, 7)  # last bar day: must NOT be visible at as_of
    close = accessor.last_close("INS-000001", as_of)
    assert close == 100.5  # 2020-01-06 close, strictly before 2020-01-07
    dv = accessor.trailing_dollar_volume("INS-000001", as_of, 20)
    assert dv == pytest.approx(101000.0)  # median of 101000, 101000, 111650, 90450
    assert accessor.last_close("INS-000001", date(2020, 1, 2)) is None


def test_normalize_market_bars_sorts_by_instrument_and_date(tmp_path):
    """Canonical bars are date-ascending even when the provider delivers
    rows in reverse order; order-dependent consumers rely on this."""
    bars, events = parse_yahoo_chart(_yahoo_payload(), "AAPL")
    reversed_bars = bars.reverse()  # simulate a provider delivering newest-first
    normalized = normalize_market_bars(
        {"AAPL": {"bars": reversed_bars, "events": events}},
        {"AAPL": "INS-000001"},
        "yahoo_chart_api", "u", "DS-000001",
    )
    dates = normalized["bars"]["trade_date"].to_list()
    assert dates == sorted(dates)
    assert dates[0] == date(2020, 1, 2)


def test_accessor_trailing_dollar_volume_is_order_independent(tmp_path):
    """The accessor must not depend on parquet row order: a reversed file
    yields the same strictly-lagged median."""
    from orbit.ingestion.snapshot import MarketDataAccessor
    from orbit.schemas.instrument import Instrument

    out = tmp_path / "normalized" / "market" / "yahoo_chart_api" / "DS-000001"
    out.mkdir(parents=True)
    bars, events = parse_yahoo_chart(_yahoo_payload(), "AAPL")
    normalized = normalize_market_bars(
        {"AAPL": {"bars": bars, "events": events}}, {"AAPL": "INS-000001"},
        "yahoo_chart_api", "u", "DS-000001",
    )
    normalized["bars"].reverse().write_parquet(out / "bars.parquet")
    inst = Instrument(
        instrument_id="INS-000001", primary_ticker="AAPL", exchange_id="XNAS",
        name="Apple", security_type="equity", listing_date=date(1980, 12, 12),
    )
    accessor = MarketDataAccessor([inst], "DS-000001", data_root=tmp_path)

    as_of = date(2020, 1, 7)
    assert accessor.last_close("INS-000001", as_of) == 100.5
    assert accessor.trailing_dollar_volume("INS-000001", as_of, 20) == pytest.approx(101000.0)


def test_ingest_market_script_exit_code_reflects_validation(monkeypatch, tmp_path):
    """The CLI must fail loudly (non-zero) when a snapshot is not promoted,
    so CI cannot report green on a failed validation."""
    import importlib.util
    import sys

    repo = __file__.replace("\\", "/").split("/tests/")[0]
    spec = importlib.util.spec_from_file_location(
        "ingest_market_cli", f"{repo}/scripts/ingest_market.py"
    )
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "ingest_market_cli", mod)

    class _Registry:
        def close(self):
            pass

    class _StubPipeline:
        def __init__(self, *a, **k):
            pass

        def ingest_market(self, *a, **k):
            return IngestResult(
                snapshot_id="DS-000001", reused=False, domain="market",
                provider="yahoo_chart_api",
                validation={"status": "failed", "issues": [{"severity": "error"}]},
            )

    spec.loader.exec_module(mod)

    monkeypatch.setattr(mod, "ensure_layout", lambda: None)
    monkeypatch.setattr(mod, "registry_path", lambda: str(tmp_path / "r.duckdb"))
    monkeypatch.setattr(mod, "IngestionRegistry", lambda *a, **k: _Registry())
    monkeypatch.setattr(mod, "RawStore", lambda: None)
    monkeypatch.setattr(mod, "IngestionPipeline", _StubPipeline)
    monkeypatch.setattr(mod, "symbol_map_from_master", lambda p: {"AAPL": "INS-000001"})
    monkeypatch.setattr(mod, "YahooChartConnector", lambda: None)

    monkeypatch.setattr(
        sys, "argv", ["ingest_market.py", "--symbols", "AAPL", "--range", "1y"]
    )
    assert mod.main() == 1

    monkeypatch.setattr(
        _StubPipeline,
        "ingest_market",
        lambda self, *a, **k: IngestResult(
            snapshot_id="DS-000001", reused=False, domain="market",
            provider="yahoo_chart_api", validation={"status": "ok"},
        ),
    )
    assert mod.main() == 0