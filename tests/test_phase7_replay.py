"""Phase 7 replay / determinism tests: identical inputs must produce
byte-identical event streams, manifests and summaries - the reproducibility
contract of the engine."""

from __future__ import annotations

import json
from datetime import date

from orbit.backtest import BacktestConfig, CostConfig

from phase7_testutils import make_bars, run_default, signals, weekdays

DATES = weekdays(date(2024, 1, 2), 10)


def _result(config=None, **kw):
    bt = run_default(make_bars(DATES), [], config=config, **kw)
    return bt.run(make_bars(DATES), signals("INS-000001", DATES[:4], target=500))


def test_reruns_produce_identical_event_streams():
    a = _result()
    b = _result()
    assert a.events_frame().write_json() == b.events_frame().write_json()
    assert len(a.events) == len(b.events)


def test_reruns_produce_identical_manifests_and_run_ids():
    a = _result()
    b = _result()
    assert a.manifest.run_id == b.manifest.run_id
    assert a.manifest.content_hash == b.manifest.content_hash
    assert a.manifest.model_dump_json(exclude={"created_at"}) == b.manifest.model_dump_json(
        exclude={"created_at"}
    )


def test_equals_passes_for_rerun_and_fails_for_different_signals():
    a = _result()
    b = _result()
    assert a.equals(b)
    changed = signals("INS-000001", DATES[:4], target=501)
    c = _result()
    c_bt = run_default(make_bars(DATES), [])
    c2 = c_bt.run(make_bars(DATES), changed)
    assert not a.equals(c2)
    # a different signal set is a different run identity
    assert a.manifest.signal_set_hash != c2.manifest.signal_set_hash


def test_config_change_is_a_different_run_identity():
    a = _result()
    b = _result(config=BacktestConfig(costs=CostConfig(spread_bps=5)))
    assert a.manifest.run_id != b.manifest.run_id
    assert a.manifest.config_hash != b.manifest.config_hash
    assert not a.equals(b)


def test_created_at_is_excluded_from_identity():
    from datetime import datetime, timezone

    from orbit.backtest import Backtester

    def _with_created_at(ts):
        bt = Backtester(
            config=BacktestConfig(),
            universe=["INS-000001", "INS-000002"],
            dataset_snapshot_ids=["DS-000001"],
            code_hash="c" * 64,
            created_at=ts,
        )
        return bt.run(make_bars(DATES), signals("INS-000001", DATES[:1]))

    a = _with_created_at(datetime(2024, 1, 1, tzinfo=timezone.utc))
    b = _with_created_at(datetime(2024, 6, 1, tzinfo=timezone.utc))
    # same run: identical identity even though created_at differs
    assert a.manifest.run_id == b.manifest.run_id
    assert a.manifest.created_at != b.manifest.created_at
    assert a.equals(b)


def test_event_sequences_are_gap_free_and_strict():
    res = _result()
    seqs = [e.sequence for e in res.events]
    assert seqs == list(range(len(seqs)))
    ids = [e.event_id for e in res.events]
    assert len(set(ids)) == len(ids)


def test_to_jsonl_round_trip(tmp_path):
    res = _result()
    path = res.to_jsonl(tmp_path / "replay_events.jsonl")
    assert path.exists()
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == len(res.events)
    first = json.loads(lines[0])
    assert first["run_id"] == res.run_id
    assert first["event_type"] == "run_start"


def test_to_dict_is_serializable_and_complete():
    res = _result()
    d = res.to_dict()
    assert d["manifest"]["run_id"] == res.run_id
    assert len(d["events"]) == len(res.events)
    json.dumps(d)  # fully JSON-serializable


def test_manifest_identity_validates():
    res = _result()
    assert res.manifest.validate_identity() == []


def test_summary_is_deterministic():
    a = _result().summary()
    b = _result().summary()
    assert json.dumps(a, sort_keys=True, default=str) == json.dumps(
        b, sort_keys=True, default=str
    )