"""Independent audit functions for the Phase 9 benchmark.

`run_phase9_audit` executes the adversarial/structural checks required by
the roadmap Section 35 (second independent review) against the artifacts of
a completed run: feature point-in-time validity, label availability, split
integrity, calibration leakage, grid lock, test lock, ranking determinism,
backtest uniformity, and registry lineage - plus the scope guards that keep
Phase 9 from leaking into Phase 10+ (feature-zoo guard, model-scope guard,
data-expansion guard). Every check returns PASS/FAIL with evidence; any FAIL
blocks the benchmark verdict.
"""

from __future__ import annotations

from typing import Any

import polars as pl

from orbit.ml.calibration import assert_no_test_fit
from orbit.ml.features import (
    FEATURE_DEFINITIONS,
    FEATURE_NAMES,
    FEATURE_SET_ID,
    FEATURE_SET_VERSION,
    assert_features_point_in_time,
)
from orbit.ml.grids import MODEL_FAMILIES, validate_model_parameters
from orbit.ml.splits import PHASE9_WINDOWS, assert_split_integrity


def _check(name: str, passed: bool, evidence: str) -> dict[str, Any]:
    return {"check": name, "status": "PASS" if passed else "FAIL", "evidence": evidence}


def run_phase9_audit(
    *,
    feature_snapshot: Any,
    label_snapshot: Any,
    datasets: dict[str, Any] | None = None,
    fitted_model: Any | None = None,
    calibration_map: Any | None = None,
    ranking_frame: pl.DataFrame | None = None,
    backtest_config: Any | None = None,
    experiment_spec: Any | None = None,
    test_predictions: pl.DataFrame | None = None,
) -> list[dict[str, Any]]:
    """Run the full independent audit. Returns a list of check dicts."""
    checks: list[dict[str, Any]] = []

    # 1. feature point-in-time validity
    if feature_snapshot is not None:
        try:
            assert_features_point_in_time(feature_snapshot.records)
            checks.append(
                _check(
                    "feature_point_in_time",
                    True,
                    f"all {feature_snapshot.records.height} feature rows reference "
                    "sessions strictly before their decision session",
                )
            )
        except AssertionError as exc:
            checks.append(_check("feature_point_in_time", False, str(exc)))
        # 1b. scope guard: no Phase 10 feature zoo (Section 35 / requirement 29)
        expected_ids = [f["feature_id"] for f in FEATURE_DEFINITIONS]
        actual_ids = sorted(getattr(feature_snapshot, "feature_refs", []) or [])
        identity_cols = {
            "instrument_id", "decision_session", "decision_time", "window_end_session"
        }
        extra_cols = sorted(set(feature_snapshot.records.columns) - identity_cols)
        scope_ok = (
            feature_snapshot.feature_set_id == FEATURE_SET_ID
            and feature_snapshot.feature_set_version == FEATURE_SET_VERSION
            and actual_ids == sorted(expected_ids)
            and extra_cols == sorted(FEATURE_NAMES)
        )
        checks.append(
            _check(
                "feature_scope_guard",
                scope_ok,
                f"feature set {feature_snapshot.feature_set_id} "
                f"v{feature_snapshot.feature_set_version}: exactly "
                f"{len(actual_ids)} documented refs, feature columns "
                f"{extra_cols}; no feature zoo",
            )
        )

    # 2. label availability + contract identity
    if label_snapshot is not None:
        checks.append(
            _check(
                "label_contract",
                label_snapshot.label_id == "LAB-004" and label_snapshot.version == "v1",
                f"label {label_snapshot.label_id} v{label_snapshot.version} "
                f"(digest {label_snapshot.contract_digest[:16]}...)",
            )
        )
        checks.append(
            _check(
                "label_availability",
                label_snapshot.unavailable_count() == 0
                or label_snapshot.unavailable_reason_counts() != {},
                f"{label_snapshot.available_count()} available / "
                f"{label_snapshot.unavailable_count()} unavailable",
            )
        )

    # 3. split integrity (exact adversarial assertion, not just row counts)
    if datasets is not None:
        meta = datasets["report"]
        split_ok = True
        evidence = f"train {meta['train_rows']} / val {meta['val_rows']} / test {meta['test_rows']} rows"
        split_frame = datasets.get("split_frame")
        if split_frame is not None:
            try:
                assert_split_integrity(split_frame)
                evidence += "; exact purge boundaries verified"
            except AssertionError as exc:
                split_ok = False
                evidence += f"; PURGE VIOLATION: {exc}"
        checks.append(
            _check(
                "split_integrity",
                split_ok
                and meta["train_rows"] >= 0
                and meta["val_rows"] >= 0
                and meta["test_rows"] > 0,
                evidence,
            )
        )
        if meta["unavailable_rows"]:
            checks.append(
                _check(
                    "unavailable_documented",
                    True,
                    f"{meta['unavailable_rows']} unavailable rows "
                    f"({meta['unavailable_reasons']})",
                )
            )

    # 4. calibration leakage
    if calibration_map is not None:
        try:
            assert_no_test_fit(calibration_map)
            checks.append(
                _check("calibration_val_only", True, "calibration map fitted on validation only")
            )
        except AssertionError as exc:
            checks.append(_check("calibration_val_only", False, str(exc)))

    # 5. grid lock
    if fitted_model is not None:
        try:
            validate_model_parameters(fitted_model.family, fitted_model.hyperparameters)
            checks.append(
                _check(
                    "grid_lock",
                    True,
                    f"{fitted_model.family} params {fitted_model.hyperparameters} "
                    "are a pre-registered grid point",
                )
            )
        except ValueError as exc:
            checks.append(_check("grid_lock", False, str(exc)))
        checks.append(
            _check(
                "seed_lock",
                fitted_model.seed == 42,
                f"seed {fitted_model.seed}",
            )
        )
        # 5b. scope guard: only the 5 Phase 9 families (no Phase 22+ models)
        checks.append(
            _check(
                "model_scope_guard",
                fitted_model.family in MODEL_FAMILIES,
                f"family {fitted_model.family!r} in the Phase 9 families "
                f"{sorted(MODEL_FAMILIES)}",
            )
        )

    # 6. test lock: test window is the locked holdout
    if test_predictions is not None and test_predictions.height:
        sessions = test_predictions["decision_session"].unique().to_list()
        outside = [
            s for s in sessions
            if s < PHASE9_WINDOWS["test_start"] or s > PHASE9_WINDOWS["test_end"]
        ]
        checks.append(
            _check(
                "test_lock",
                not outside,
                f"{len(sessions)} test sessions, all within the locked window"
                if not outside
                else f"test predictions outside locked window: {outside[:5]}",
            )
        )

    # 7. backtest uniformity
    if backtest_config is not None:
        costs = backtest_config.costs
        checks.append(
            _check(
                "backtest_uniformity",
                costs.spread_bps == 2.0
                and costs.fees_bps == 1.0
                and costs.slippage_bps == 2.0
                and backtest_config.sizing.value == "weight"
                and backtest_config.long_only is True,
                f"costs {costs.spread_bps}/{costs.fees_bps}/{costs.slippage_bps} bps, "
                f"sizing {backtest_config.sizing.value}, long_only {backtest_config.long_only}",
            )
        )

    # 8. registry lineage + data-expansion guard (no new data acquisition)
    if experiment_spec is not None:
        checks.append(
            _check(
                "registry_lineage",
                experiment_spec.dataset_snapshot_ids == ["DS-000004"]
                and experiment_spec.label_id == "LAB-004"
                and experiment_spec.cost_model_id == "CM-001",
                f"snapshots {experiment_spec.dataset_snapshot_ids}, "
                f"label {experiment_spec.label_id} v{experiment_spec.label_version}, "
                f"cost model {experiment_spec.cost_model_id}",
            )
        )
        checks.append(
            _check(
                "data_expansion_guard",
                set(experiment_spec.dataset_snapshot_ids or []) <= {"DS-000004"},
                f"experiment pins only the validated dev snapshot "
                f"{experiment_spec.dataset_snapshot_ids}; no data expansion",
            )
        )

    # 9. data-expansion guard from the feature snapshot's data refs
    if feature_snapshot is not None:
        refs = sorted(getattr(feature_snapshot, "data_refs", []) or [])
        checks.append(
            _check(
                "data_expansion_guard",
                set(refs) <= {"DS-000004"},
                f"feature snapshot data refs {refs}; no data expansion",
            )
        )

    return checks


def audit_summary(checks: list[dict[str, Any]]) -> dict[str, Any]:
    failed = [c for c in checks if c["status"] == "FAIL"]
    return {
        "checks": len(checks),
        "passed": len(checks) - len(failed),
        "failed": len(failed),
        "blocked": bool(failed),
        "failed_checks": [c["check"] for c in failed],
    }


__all__ = ["run_phase9_audit", "audit_summary"]