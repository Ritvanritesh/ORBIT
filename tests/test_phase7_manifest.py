"""Phase 7 manifest and export tests: the run's scientific identity, its
tamper-resistance, and the serializable artifacts."""

from __future__ import annotations

import json
from datetime import date

import pytest
from pydantic import ValidationError

from orbit.backtest import (
    BacktestConfig,
    BacktestManifest,
    CostConfig,
    build_manifest,
)
from orbit.backtest.config import ExecutionConfig
from orbit.temporal.times import session_close_utc

from phase7_testutils import make_bars, run_default, signals, weekdays

DATES = weekdays(date(2024, 1, 2), 10)
_BARS = make_bars(DATES)
_SIGNALS = signals("INS-000001", DATES[:2])


def _manifest() -> BacktestManifest:
    res = run_default(_BARS, []).run(_BARS, _SIGNALS)
    return res.manifest


def test_manifest_records_execution_and_cost_identity():
    m = _manifest()
    assert m.execution["execution_price"] == "open"
    assert m.execution["execution_delay"] == 1
    assert m.costs == {"spread_bps": 0.0, "fees_bps": 0.0, "slippage_bps": 0.0,
                       "fixed_fee_per_order": 0.0, "fee_minimum": 0.0}
    assert m.long_only is True
    assert m.valuation_price == "close"
    assert m.sizing == "quantity"
    assert m.liquidity_volume_basis in {"as_published", "provider_stored"}


def test_manifest_requires_dataset_snapshot_ids():
    with pytest.raises(ValueError, match="dataset_snapshot_ids"):
        build_manifest(
            config=BacktestConfig(),
            engine_version="v1.0.0",
            signal_set_hash="s" * 64,
            universe=["INS-000001"],
            liquidity_volume_basis="as_published",
            dataset_snapshot_ids=[],
            code_hash="c" * 64,
            config_hash="g" * 64,
        )


def test_manifest_run_id_is_derived_from_content():
    m = _manifest()
    assert m.run_id == f"BT-{m.config_hash[:8]}-{m.content_hash[:12]}"
    assert m.derive_run_id() == m.run_id
    assert m.validate_identity() == []


def test_manifest_content_hash_covers_signal_set():
    a = run_default(_BARS, []).run(_BARS, _SIGNALS)
    changed = _SIGNALS + [
        dict(
            _SIGNALS[0],
            signal_id="SIG-EXTRA",
            signal_session=DATES[5],
            decision_time=session_close_utc(DATES[5]),
        )
    ]
    b = run_default(_BARS, []).run(_BARS, changed)
    assert a.manifest.signal_set_hash != b.manifest.signal_set_hash
    assert a.manifest.content_hash != b.manifest.content_hash


def test_manifest_content_hash_covers_config():
    a = run_default(_BARS, []).run(_BARS, _SIGNALS)
    b = run_default(_BARS, [], config=BacktestConfig(
        execution=ExecutionConfig(execution_delay=2)
    )).run(_BARS, _SIGNALS)
    assert a.manifest.config_hash != b.manifest.config_hash
    assert a.manifest.content_hash != b.manifest.content_hash


def test_tampered_manifest_is_detected():
    m = _manifest()
    tampered = m.model_copy(update={"initial_cash": 999.0})
    violations = tampered.validate_identity()
    assert any("content_hash" in v for v in violations)
    forged = m.model_copy(update={"run_id": "BT-forged"})
    assert any("run_id" in v for v in forged.validate_identity())


def test_created_at_and_run_id_excluded_from_canonical_json():
    m = _manifest()
    payload = json.loads(m.canonical_json())
    assert "created_at" not in payload
    assert "run_id" not in payload
    assert "content_hash" not in payload


def test_manifest_requires_min_length_hashes():
    with pytest.raises(ValidationError):
        build_manifest(
            config=BacktestConfig(),
            engine_version="v1.0.0",
            signal_set_hash="short",
            universe=["INS-000001"],
            liquidity_volume_basis="as_published",
            dataset_snapshot_ids=["DS-000001"],
            code_hash="c" * 64,
            config_hash="g" * 64,
        )


def test_events_frame_schema_is_stable():
    res = run_default(_BARS, []).run(_BARS, _SIGNALS)
    frame = res.events_frame()
    required = {
        "event_id", "run_id", "event_type", "sequence", "session",
        "timestamp", "source", "config_ref",
    }
    assert required.issubset(frame.columns)
    assert frame.height == len(res.events)
    types = set(frame["event_type"].to_list())
    assert {"run_start", "market", "signal", "order", "fill", "ledger",
            "valuation", "outcome", "run_end"} <= types


def test_summary_shape_and_consistency():
    res = run_default(_BARS, []).run(_BARS, _SIGNALS)
    s = res.summary()
    assert s["final_equity"] == pytest.approx(1_000_000.0, rel=1e-9)
    assert s["n_signals"] == 2
    assert s["n_orders"] == 1
    assert s["n_fills"] == 1
    assert s["n_rejects"] == 0
    assert s["total_return"] == pytest.approx(0.0, abs=1e-9)
    assert s["invariant_violations"] == []
    assert isinstance(s["positions"], dict)


def test_exported_jsonl_matches_events_frame(tmp_path):
    import polars as pl

    res = run_default(_BARS, []).run(_BARS, _SIGNALS)
    path = res.to_jsonl(tmp_path / "manifest_events.jsonl")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    frame = pl.DataFrame(rows)
    assert frame["event_id"].to_list() == res.events_frame()["event_id"].to_list()
    assert frame["sequence"].to_list() == res.events_frame()["sequence"].to_list()


def test_cost_config_from_phase1_cost_model():
    from orbit.schemas.common import CostModel

    model = CostModel()
    cc = CostConfig.from_cost_model(model)
    assert cc.spread_bps == model.spread_bps
    assert cc.fees_bps == model.fees_bps
    assert cc.slippage_bps == model.slippage_bps