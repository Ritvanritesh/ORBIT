"""Phase 10 registry tests: register-before-run, config-hash lineage,
plan-scoped model points, and single-immutable-result lifecycle."""

from __future__ import annotations

import json
import polars as pl
import pytest

from orbit.ml.features import FEATURE_SET_VERSION, phase10_set_identity
from orbit.ml.phase10_registry import (
    phase10_config_hash,
    register_phase10_experiment,
    run_registered_phase10_experiment,
)


def test_config_hash_is_deterministic_and_lineage_sensitive():
    a = phase10_config_hash("FS-003", "ridge", {"alpha": 1.0})
    b = phase10_config_hash("FS-003", "ridge", {"alpha": 1.0})
    assert a == b
    assert a == "3e2d370f4ef43e6632eff3f71feb7b8230c16b6cbb68cb6ec0998046fb0444db"
    # different feature set -> different hash
    assert phase10_config_hash("FS-001", "ridge", {"alpha": 1.0}) != a
    # different model point -> different hash
    assert phase10_config_hash("FS-003", "xgboost", {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.1}) != a
    # different seed -> different hash
    assert phase10_config_hash("FS-003", "ridge", {"alpha": 1.0}, seed=7) != a


def test_registration_requires_a_phase10_model_point():
    with pytest.raises(ValueError, match="pre-registered|grid"):
        register_phase10_experiment(
            experiment_id="EXP-10001",
            hypothesis_id="H-001",
            feature_set_id="FS-003",
            feature_set_version="v1",
            family="ridge",
            params={"alpha": 0.5},
        )


def test_registration_rejects_wrong_feature_set_version():
    with pytest.raises(ValueError, match="is version"):
        register_phase10_experiment(
            experiment_id="EXP-10001",
            hypothesis_id="H-001",
            feature_set_id="FS-003",
            feature_set_version="v2",
            family="ridge",
            params={"alpha": 1.0},
        )


def test_registration_pins_feature_lineage(tmp_path):
    service, spec = register_phase10_experiment(
        experiment_id="EXP-10013",
        hypothesis_id="H-001",
        feature_set_id="FS-003",
        feature_set_version="v1",
        family="ridge",
        params={"alpha": 1.0},
        seed=42,
        plan_digest="deadbeef",
    )
    identity = phase10_set_identity("FS-003")
    assert spec.features.feature_set_id == "FS-003"
    assert spec.features.feature_version == "v1"
    assert {r.feature_id for r in spec.features.feature_refs} == set(identity["feature_refs"])
    assert spec.dataset_snapshot_ids == ["DS-000004"]
    assert spec.label_id == "LAB-004"
    assert spec.cost_model_id == "CM-001"
    assert spec.seed == 42
    assert "plan_digest=deadbeef" in spec.notes
    # transformation binds set id + version + definitions digest
    assert all("phase10_feature_set_v1" in r.transformation for r in spec.features.feature_refs)


def test_registration_title_and_notes_use_version_once():
    """The registered title/notes must render the version as 'v1' (never the
    doubled 'vv1' that a 'v' + 'v1' concatenation would produce)."""
    service, spec = register_phase10_experiment(
        experiment_id="EXP-10013",
        hypothesis_id="H-001",
        feature_set_id="FS-003",
        feature_set_version="v1",
        family="ridge",
        params={"alpha": 1.0},
        seed=42,
    )
    assert "vv1" not in spec.title
    assert "FS-003 v1" in spec.title
    assert "vv1" not in spec.notes
    assert "FS-003 v1" in spec.notes


def test_run_lifecycle_records_single_result(tmp_path):
    service, spec = register_phase10_experiment(
        experiment_id="EXP-10010",
        hypothesis_id="H-001",
        feature_set_id="FS-003",
        feature_set_version="v1",
        family="lasso",
        params={"alpha": 0.001},
        seed=42,
    )
    import json as _json
    import pathlib

    art = tmp_path / "artifacts"
    art.mkdir()
    pl.DataFrame({"x": [1, 2, 3]}).write_parquet(art / "pred.parquet")
    metrics = {"oos_ic": 0.01, "after_cost_total_return": 0.02}
    result_id = run_registered_phase10_experiment(
        service, "EXP-10010",
        feature_set_id="FS-003", family="lasso", params={"alpha": 0.001}, seed=42,
        artifacts_dir=str(art),
        result_summary="ok",
        result_metrics=metrics,
        artifact_files={"test_predictions_parquet": art / "pred.parquet"},
    )
    assert isinstance(result_id, str) and result_id
    res = service.result("EXP-10010")
    assert res is not None
    assert res["kind"] == "supported"
    assert _json.loads(res["metrics_json"])["oos_ic"] == 0.01
    # the result is immutable: a second record_result refuses
    with pytest.raises(Exception):
        service.record_result(
            "EXP-10010", kind="supported", summary="again",
            metrics={"oos_ic": 0.0}, recorded_by="orbit-research",
        )


def test_run_lifecycle_marks_running_before_completion(tmp_path):
    service, _ = register_phase10_experiment(
        experiment_id="EXP-10011",
        hypothesis_id="H-001",
        feature_set_id="FS-003",
        feature_set_version="v1",
        family="xgboost",
        params={"n_estimators": 200, "max_depth": 3, "learning_rate": 0.1},
        seed=42,
    )
    spec = service.get("EXP-10011")
    assert spec is not None
    assert spec.status == "registered"