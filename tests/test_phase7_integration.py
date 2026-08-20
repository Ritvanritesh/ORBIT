"""Phase 7 integration tests: the Phase 4 temporal gate, the Phase 5
predicted/realized/executed separation, and the Phase 6 experiment
lifecycle around a backtest run."""

from __future__ import annotations

from datetime import date
import json

import polars as pl
import pytest

from conftest import make_spec

from orbit.backtest import BacktestConfig
from orbit.backtest.events import FailureKind
from orbit.backtest.integration import (
    predicted_vs_executed,
    realized_outcome,
    run_backtest_experiment,
    validate_signal_temporality,
)
from orbit.experiments import ExperimentService, ResultKind
from orbit.labels.engine import LabelEngine
from orbit.schemas.common import ExperimentStatus
from orbit.temporal.times import session_close_utc

from phase7_testutils import make_bars, signals, weekdays

DATES = weekdays(date(2024, 1, 2), 12)
_BARS = make_bars(
    DATES,
    instruments=["INS-000001", "SPY"],
    base_prices={"INS-000001": 100.0, "SPY": 400.0},
)


def _spec_config(spec):
    """The backtest config consistent with the registered experiment's
    pinned cost model (lineage: a different cost assumption is a different
    experiment)."""
    from orbit.backtest import CostConfig

    return BacktestConfig(costs=CostConfig.from_cost_model(spec.cost_model))


# ---------------------------------------------------------------- Phase 4

def test_signal_temporality_accepts_only_session_close_instants():
    ok = validate_signal_temporality(signals("INS-000001", DATES[:1]))
    assert ok[0]["decision_time"] == session_close_utc(DATES[0])
    bad = signals("INS-000001", DATES[:1])
    bad[0]["decision_time"] = session_close_utc(DATES[0]).replace(hour=9, minute=30)
    with pytest.raises(ValueError, match="session close"):
        validate_signal_temporality(bad)


def test_backtester_refuses_non_session_close_decision_times():
    from phase7_testutils import run_default

    bad = signals("INS-000001", DATES[:1])
    bad[0]["decision_time"] = session_close_utc(DATES[0]).replace(hour=23)
    bt = run_default(_BARS, [])
    with pytest.raises(ValueError, match="session close"):
        bt.run(_BARS, bad)


# ---------------------------------------------------------------- Phase 5

def test_predicted_vs_executed_keeps_columns_separate():
    from phase7_testutils import run_default

    res = run_default(_BARS, []).run(_BARS, signals("INS-000001", DATES[:3]))
    frame = predicted_vs_executed(res)
    cols = frame.columns
    assert "predicted_metric" in cols and "executed_vwap" in cols
    assert frame.height == res.summary()["n_signals"]
    executed = frame.filter(pl.col("executed_quantity") > 0)
    assert executed.height == 1
    assert executed["signal_id"].to_list() == ["SIG-000001"]
    assert executed["predicted_metric"].to_list() == [0.05]
    assert executed["executed_vwap"].to_list() == [100.0]


def test_realized_outcome_computed_by_phase5_engine(labels, temporal):
    from orbit.backtest import BacktestConfig, CostConfig
    from phase7_testutils import run_default

    # the Phase 5 engine requires the corporate-actions artifact for
    # total-return contracts; an empty (but schema-valid) artifact means
    # no dividends -> total return == price return
    events = pl.DataFrame(
        schema={
            "instrument_id": pl.Utf8,
            "kind": pl.Utf8,
            "ts": pl.Utf8,
            "ratio": pl.Float64,
        }
    )
    contract = labels.definition("LAB-001", "v1")
    engine = LabelEngine(_BARS, events=events)
    # signals from day 1 onward: the label engine's decision-instant anchor
    # resolves the entry to the last completed bar strictly before the
    # decision, so a decision at day 0's close has no entry bar
    res = run_default(_BARS, []).run(_BARS, signals("INS-000001", DATES[1:5]))
    frame = realized_outcome(res, engine, contract, label_version="v1")
    assert "predicted_metric" in frame.columns
    assert "executed_vwap" in frame.columns
    assert "realized_metric" in frame.columns
    executed = frame.filter(pl.col("executed_vwap").is_not_null())
    assert executed.height == 1
    row = executed.row(0, named=True)
    # predicted metric and realized label are different columns and
    # different numbers here: price stays flat, so the 5-session return
    # is 0 (SPY also flat) -> the realized excess is exactly 0
    assert row["realized_metric"] == pytest.approx(0.0, abs=1e-9)
    assert row["label_version"] == "v1"


# ---------------------------------------------------------------- Phase 6

def _service_spec(service, temporal_digest, **overrides):
    spec = make_spec(temporal_digest=temporal_digest, **overrides)
    service.register(spec, registered_at=__import__(
        "datetime", fromlist=["datetime"]
    ).datetime(2026, 1, 1, tzinfo=__import__("datetime", fromlist=["timezone"]).timezone.utc))
    return spec


def test_full_experiment_lifecycle_around_a_backtest(service, temporal_digest, tmp_path):
    spec = _service_spec(service, temporal_digest)
    result, meta = run_backtest_experiment(
        service,
        spec.experiment_id,
        config=_spec_config(spec),
        bars=_BARS,
        signals=signals("INS-000001", DATES[:3]),
        artifacts_dir=tmp_path / "artifacts",
    )
    assert meta["run_id"] == result.run_id
    exp = service.get(spec.experiment_id)
    assert exp.status == ExperimentStatus.COMPLETED
    assert exp.code_hash is not None and exp.config_hash is not None
    artifacts = service.artifacts(spec.experiment_id)
    kinds = {a["kind"] for a in artifacts}
    assert kinds == {"backtest_event_stream_jsonl", "backtest_manifest_json"}
    for a in artifacts:
        assert a["checksum"], "artifacts must carry checksums"
    rec = service.result(spec.experiment_id)
    assert rec is not None
    assert rec["kind"] == ResultKind.SUPPORTED.value
    metrics = json.loads(rec["metrics_json"])
    assert metrics["run_id"] == result.run_id
    # the manifest records the lineage of the registered experiment
    manifest = result.manifest
    assert manifest.experiment_id == spec.experiment_id
    assert manifest.hypothesis_id == spec.hypothesis_id
    assert manifest.dataset_snapshot_ids == ["DS-000001"]
    assert manifest.label_id == "LAB-001" and manifest.label_version == "v1"
    assert manifest.temporal_config_digest == temporal_digest
    assert manifest.cost_model_id == "CM-001"
    assert manifest.validate_identity() == []
    assert result.invariant_violations() == []


def test_seed_mismatch_is_refused_before_any_run(service, temporal_digest, tmp_path):
    spec = _service_spec(service, temporal_digest)
    with pytest.raises(ValueError, match="different seed"):
        run_backtest_experiment(
            service,
            spec.experiment_id,
            config=BacktestConfig(seed=7),
            bars=_BARS,
            signals=signals("INS-000001", DATES[:1]),
            artifacts_dir=tmp_path / "artifacts",
        )
    # nothing started: the experiment is untouched
    assert service.get(spec.experiment_id).status == ExperimentStatus.REGISTERED


def test_cost_model_mismatch_is_refused(service, temporal_digest, tmp_path):
    from orbit.backtest import CostConfig

    spec = _service_spec(service, temporal_digest)
    with pytest.raises(ValueError, match="different cost assumption"):
        run_backtest_experiment(
            service,
            spec.experiment_id,
            config=BacktestConfig(costs=CostConfig(fees_bps=50)),
            bars=_BARS,
            signals=signals("INS-000001", DATES[:1]),
            artifacts_dir=tmp_path / "artifacts",
        )


def test_unknown_experiment_is_refused(service, temporal_digest, tmp_path):
    with pytest.raises(ValueError, match="unknown experiment"):
        run_backtest_experiment(
            service,
            "EXP-99999",
            config=BacktestConfig(),
            bars=_BARS,
            signals=signals("INS-000001", DATES[:1]),
            artifacts_dir=tmp_path / "artifacts",
        )


def test_temporally_invalid_signals_are_refused_before_any_transition(
    service, temporal_digest, tmp_path
):
    # the Phase 4 gate is a request validation, not an experiment failure:
    # a temporally invalid input must not burn a lifecycle transition
    spec = _service_spec(service, temporal_digest)
    bad_signals = signals("INS-000001", DATES[:1])
    bad_signals[0]["decision_time"] = session_close_utc(DATES[0]).replace(hour=8)
    with pytest.raises(ValueError, match="session close"):
        run_backtest_experiment(
            service,
            spec.experiment_id,
            config=_spec_config(spec),
            bars=_BARS,
            signals=bad_signals,
            artifacts_dir=tmp_path / "artifacts",
        )
    exp = service.get(spec.experiment_id)
    assert exp.status == ExperimentStatus.REGISTERED
    assert service.transitions(spec.experiment_id) == []


def test_failing_run_records_experiment_as_failed(service, temporal_digest, tmp_path):
    # an in-run failure (after mark_running) is a lifecycle failure: the
    # experiment must never rest in RUNNING
    spec = _service_spec(service, temporal_digest)
    bad_signals = signals("NOT-IN-UNIVERSE", DATES[:1])
    with pytest.raises(ValueError, match="not in the backtest universe"):
        run_backtest_experiment(
            service,
            spec.experiment_id,
            config=_spec_config(spec),
            bars=_BARS,
            signals=bad_signals,
            artifacts_dir=tmp_path / "artifacts",
        )
    exp = service.get(spec.experiment_id)
    assert exp.status == ExperimentStatus.FAILED
    notes = [t["note"] for t in service.transitions(spec.experiment_id)]
    assert any("backtest failed" in (n or "") for n in notes)


def test_run_records_only_one_result(service, temporal_digest, tmp_path):
    spec = _service_spec(service, temporal_digest)
    run_backtest_experiment(
        service,
        spec.experiment_id,
        config=_spec_config(spec),
        bars=_BARS,
        signals=signals("INS-000001", DATES[:1]),
        artifacts_dir=tmp_path / "artifacts",
    )
    with pytest.raises(ValueError, match="invalid experiment transition"):
        run_backtest_experiment(
            service,
            spec.experiment_id,
            config=_spec_config(spec),
            bars=_BARS,
            signals=signals("INS-000001", DATES[:1]),
            artifacts_dir=tmp_path / "artifacts",
        )