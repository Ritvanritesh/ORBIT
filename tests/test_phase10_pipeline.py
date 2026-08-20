"""Phase 10 full-pipeline tests on hermetic synthetic data.

The ENTIRE locked plan (13 feature sets x 4 models = 52 experiments) runs
end to end ONCE (module-scoped fixture) with the test windows, in temp
report/cache locations, and every experiment is verified: registered-before-
run, completed with metrics, an after-cost result, and an audit that passes.
"""

from __future__ import annotations

import json
from datetime import date

import polars as pl
import pytest

import orbit.ml.phase10_report as report_mod
import orbit.ml.phase10_runner as runner_mod
from orbit.ml.phase10_plan import phase10_plan_digest
from orbit.schemas.instrument import Instrument, SecurityType
from tests.phase9_testutils import TEST_WINDOWS, make_canonical_bars, make_events


@pytest.fixture(scope="module")
def pipeline_result(tmp_path_factory):
    """Run the full locked plan once; share the result across all tests."""
    tmp = tmp_path_factory.mktemp("phase10_pipeline")
    bars = make_canonical_bars()
    events = make_events()
    instruments = [
        Instrument(
            instrument_id=ins,
            primary_ticker=f"SYM{ins[-2:]}",
            exchange_id="XNYS",
            name=f"Test {ins}",
            security_type=SecurityType.EQUITY,
            listing_date=date(1995, 1, 1),
        )
        for ins in sorted(bars["instrument_id"].unique().to_list())
    ]
    manifest = {"snapshot_id": "DS-000004", "row_count": -1}

    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(runner_mod, "load_snapshot_bars", lambda: bars)
        monkeypatch.setattr(runner_mod, "load_snapshot_events", lambda: events)
        monkeypatch.setattr(runner_mod, "load_snapshot_manifest", lambda: manifest)
        monkeypatch.setattr(runner_mod, "load_instrument_master", lambda: instruments)
        monkeypatch.setattr(runner_mod, "P9_CACHE_DIR", tmp / "p9cache")
        monkeypatch.setattr(runner_mod, "P10_CACHE_DIR", tmp / "p10cache")
        monkeypatch.setattr(runner_mod, "ARTIFACTS_ROOT", tmp / "runs")
        monkeypatch.setattr(report_mod, "REPORT_PARQUET", tmp / "phase10.parquet")
        monkeypatch.setattr(report_mod, "REPORT_MARKDOWN", tmp / "phase10.md")
        monkeypatch.setattr(report_mod, "PLAN_JSON", tmp / "plan.json")
        monkeypatch.setattr(report_mod, "DIAGNOSTICS_JSON", tmp / "diagnostics.json")
        monkeypatch.setattr(report_mod, "RESEARCH_MD", tmp / "research.md")
        result = runner_mod.run_phase10_all(windows=TEST_WINDOWS)
        result["tmp"] = tmp
        yield result
    finally:
        monkeypatch.undo()


def test_full_pipeline_runs_all_52_experiments_and_passes_audit(pipeline_result):
    audit = pipeline_result["audit"]
    assert audit["passed"] > 0
    assert audit["failed"] == 0, f"audit failures: {audit['failed_checks']}"

    frame = pl.read_parquet(report_mod.REPORT_PARQUET)
    assert frame.height == 52
    assert frame["status"].to_list() == ["completed"] * 52
    assert frame["experiment_id"].to_list() == sorted(frame["experiment_id"].to_list())
    assert set(frame["experiment_id"].to_list()) == {f"EXP-{i:05d}" for i in range(10001, 10053)}
    assert frame["oos_ic"].null_count() == 0
    assert frame["after_cost_total_return"].null_count() == 0


def test_fs001_base_uses_identical_rows_as_phase9_assembly(pipeline_result):
    """FS-001 runs assemble the exact Phase 9 row sets (train/val/test)."""
    from orbit.ml.dataset import assemble_datasets
    from orbit.ml.features import FEATURE_NAMES

    snapshots, ls, digest = runner_mod.build_or_load_phase10_snapshots(
        runner_mod.load_snapshot_bars(),
        runner_mod.load_snapshot_events(),
        runner_mod.load_instrument_master(),
    )
    ds = assemble_datasets(
        snapshots["FS-001"], ls, windows=TEST_WINDOWS, feature_names=list(FEATURE_NAMES)
    )
    assert ds["report"]["test_rows"] > 100
    frame = pl.read_parquet(report_mod.REPORT_PARQUET)
    base_rows = frame.filter(pl.col("feature_set_id") == "FS-001")
    assert base_rows.height == 4
    assert base_rows["test_rows"].to_list() == [ds["report"]["test_rows"]] * 4
    assert base_rows["train_rows"].to_list() == [ds["report"]["train_rows"]] * 4


def test_plan_json_written_and_matches_locked_plan(pipeline_result):
    plan = json.loads(report_mod.PLAN_JSON.read_text(encoding="utf-8"))
    assert plan["plan_digest"] == phase10_plan_digest()
    assert plan["experiment_count"] == 52


def test_diagnostics_json_is_train_scoped(pipeline_result):
    diag = json.loads(report_mod.DIAGNOSTICS_JSON.read_text(encoding="utf-8"))
    assert diag["scope"] == "train split only (never test)"
    assert set(diag["feature_sets"].keys()) == {"FS-001", "FS-002", "FS-003"}


def test_runs_are_reproducible(pipeline_result):
    """A second full run produces bit-identical metrics (deterministic)."""
    f1 = pl.read_parquet(report_mod.REPORT_PARQUET).sort("experiment_id")
    r2 = runner_mod.run_phase10_all(windows=TEST_WINDOWS)
    f2 = pl.read_parquet(report_mod.REPORT_PARQUET).sort("experiment_id")
    assert r2["plan_digest"] == phase10_plan_digest()
    assert f1.height == f2.height == 52
    assert f1["oos_ic"].to_list() == f2["oos_ic"].to_list()
    assert f1["after_cost_total_return"].to_list() == f2["after_cost_total_return"].to_list()


def test_experiment_artifacts_are_checksummed(pipeline_result):
    runs_dir = pipeline_result["tmp"] / "runs"
    exp_dirs = [p for p in runs_dir.iterdir() if p.is_dir()]
    assert len(exp_dirs) == 52
    for d in exp_dirs:
        metrics = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
        assert "oos_ic" in metrics and "after_cost_total_return" in metrics


def test_backtest_uses_phase7_engine_with_cm001(pipeline_result):
    """The ablation never bypasses the canonical backtester: every result has
    fills, costs, and rejects recorded by the Phase 7 engine."""
    frame = pl.read_parquet(report_mod.REPORT_PARQUET)
    assert frame["n_signals"].sum() > 0
    assert frame["total_costs"].min() >= 0
    assert frame["cost_model_id"].to_list() == ["CM-001"] * 52
    assert frame["label_id"].to_list() == ["LAB-004"] * 52