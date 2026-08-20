"""Phase 10 report tests: permanent artifacts are written, deterministic,
never hide failures, and embed the locked plan digest."""

from __future__ import annotations

import json
import polars as pl
import pytest

import orbit.ml.phase10_report as report_mod
from orbit.ml.phase10_plan import phase10_plan


@pytest.fixture()
def patched_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(report_mod, "REPORT_PARQUET", tmp_path / "p10.parquet")
    monkeypatch.setattr(report_mod, "REPORT_MARKDOWN", tmp_path / "p10.md")
    monkeypatch.setattr(report_mod, "PLAN_JSON", tmp_path / "plan.json")
    monkeypatch.setattr(report_mod, "DIAGNOSTICS_JSON", tmp_path / "diag.json")
    monkeypatch.setattr(report_mod, "RESEARCH_MD", tmp_path / "research.md")
    return tmp_path


def _row(experiment_id="EXP-10001", **kw):
    row = {
        "experiment_id": experiment_id,
        "feature_set_id": "FS-003",
        "feature_set_version": "v1",
        "set_role": "all",
        "set_family": None,
        "n_features": 23,
        "family": "ridge",
        "params": '{"alpha": 1.0}',
        "seed": 42,
        "status": "completed",
        "oos_ic": 0.02,
        "rank_ic": 0.03,
        "ece": 0.1,
        "brier": 0.25,
        "mse": 1.0,
        "hit_rate": 0.5,
        "after_cost_total_return": 0.01,
        "after_cost_final_equity": 1_010_000.0,
        "turnover": 3.0,
        "total_costs": 500.0,
        "n_fills": 300,
        "n_rejects": 0,
        "n_signals": 400,
        "label_id": "LAB-004",
        "dataset_snapshot_ids": "DS-000004",
        "cost_model_id": "CM-001",
        "evaluation_window": "2022-01-03..2026-06-30",
        "train_rows": 1000,
        "val_rows": 500,
        "test_rows": 500,
        "feature_set_digest": "a" * 64,
        "definitions_digest": "b" * 64,
        "notes": "test",
    }
    row.update(kw)
    return row


def test_append_and_upsert_rows(patched_paths):
    report_mod.append_report_rows([_row("EXP-10001", oos_ic=0.01)])
    report_mod.append_report_rows([_row("EXP-10001", oos_ic=0.99)])
    frame = pl.read_parquet(report_mod.REPORT_PARQUET)
    assert frame.height == 1
    assert frame["oos_ic"][0] == 0.99  # upsert keeps the latest


def test_report_keeps_failed_rows(patched_paths):
    failed = _row("EXP-10002", status="failed", oos_ic=None,
                  after_cost_total_return=None, notes="boom")
    report_mod.append_report_rows([failed])
    frame = pl.read_parquet(report_mod.REPORT_PARQUET)
    assert frame.height == 1
    assert frame["status"][0] == "failed"
    assert frame["oos_ic"][0] is None


def test_plan_json_written(patched_paths):
    report_mod.write_plan(phase10_plan())
    plan = json.loads(report_mod.PLAN_JSON.read_text(encoding="utf-8"))
    assert plan["plan_digest"] == "16d62bff387704746fe2ac23742045dcf27314109957752473ed4b0edff64910"
    assert plan["experiment_count"] == 52


def test_diagnostics_json_written(patched_paths):
    report_mod.write_diagnostics({"scope": "train", "x": 1})
    diag = json.loads(report_mod.DIAGNOSTICS_JSON.read_text(encoding="utf-8"))
    assert diag["scope"] == "train"


def test_markdown_report_regenerates_from_parquet(patched_paths):
    report_mod.append_report_rows([_row()])
    md = report_mod.write_markdown_report()
    text = md.read_text(encoding="utf-8")
    assert "# Phase 10" in text
    assert "EXP-10001" in text
    assert "CM-001" in text


def test_markdown_requires_parquet(patched_paths):
    with pytest.raises(FileNotFoundError):
        report_mod.write_markdown_report()


def test_research_report_embeds_digest_and_inventory(patched_paths):
    report_mod.write_plan(phase10_plan())
    report_mod.append_report_rows([_row()])
    from orbit.ml.features import build_feature_snapshot
    from tests.phase9_testutils import make_canonical_bars

    fs001 = build_feature_snapshot(make_canonical_bars(), data_refs=["DS-000001"])
    from orbit.ml.features import build_feature_snapshot_phase10

    fs002 = build_feature_snapshot_phase10(make_canonical_bars(), data_refs=["DS-000001"])
    from orbit.ml.features import build_phase10_feature_set_snapshot, build_phase10_all_feature_frame

    allf = build_phase10_all_feature_frame(make_canonical_bars())
    snapshots = {"FS-001": fs001, "FS-002": fs002, "FS-003": build_phase10_feature_set_snapshot("FS-003", allf, data_refs=["DS-000001"])}
    md = report_mod.write_research_report(
        plan=phase10_plan(),
        diagnostics={"scope": "train"},
        snapshots=snapshots,
        phase9_fs001_digest=fs001.content_digest,
    )
    text = md.read_text(encoding="utf-8")
    assert "# ORBIT Phase 10" in text
    assert "Feature inventory" in text
    assert fs001.content_digest[:16] in text
    assert "EXP-10001" in text


def test_research_report_inventory_has_no_placeholders(patched_paths):
    """The permanent inventory must name the family of every feature (never
    '?') and snapshot versions must render as 'v1' (never 'vv1')."""
    from orbit.ml.features import (
        build_feature_snapshot,
        build_feature_snapshot_phase10,
        build_phase10_all_feature_frame,
        build_phase10_feature_set_snapshot,
    )
    from tests.phase9_testutils import make_canonical_bars

    fs001 = build_feature_snapshot(make_canonical_bars(), data_refs=["DS-000001"])
    allf = build_phase10_all_feature_frame(make_canonical_bars())
    snapshots = {
        "FS-001": fs001,
        "FS-002": build_feature_snapshot_phase10(make_canonical_bars(), data_refs=["DS-000001"]),
        "FS-003": build_phase10_feature_set_snapshot("FS-003", allf, data_refs=["DS-000001"]),
        "FS-004": build_phase10_feature_set_snapshot("FS-004", allf, data_refs=["DS-000001"]),
    }
    md = report_mod.write_research_report(
        plan=phase10_plan(),
        diagnostics={"scope": "train"},
        snapshots=snapshots,
        phase9_fs001_digest=fs001.content_digest,
    )
    text = md.read_text(encoding="utf-8")
    assert "| ? |" not in text
    assert "vv1" not in text
    assert "| FEAT-101 | ret_5 | momentum |" in text
    assert "| FEAT-106 | price_distance_200ma | trend |" in text
    assert "| FEAT-115 | normalized_range_20 | range |" in text
    assert "| FS-001 v1 | base (frozen) |" in text
    assert "| FS-002 v1 | new |" in text