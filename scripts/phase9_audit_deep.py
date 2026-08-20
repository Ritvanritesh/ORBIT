"""Phase 9 - Deep-input independent audit (roadmap Section 35, Review 1).

The benchmark runner's structural audit receives only the snapshots and the
dataset report (a subset of the checks). This script performs the FULL
deep-input audit: it trains one representative pre-registered grid point per
model family on the real DS-000004 data, calibrates on validation, ranks the
test predictions, runs the canonical Phase 7 backtest, registers the
experiment, and then invokes `run_phase9_audit` with EVERY artifact input so
all 13 checks execute:

  feature_point_in_time, feature_scope_guard, label_contract,
  label_availability, split_integrity, unavailable_documented,
  calibration_val_only, grid_lock, seed_lock, model_scope_guard,
  test_lock, backtest_uniformity, registry_lineage, data_expansion_guard

Any FAIL blocks the benchmark verdict. Results are written permanently to
benchmarks/phase9_audit_results.json. The process exits 0 only if every
check PASSes.

Run:  python scripts/phase9_audit_deep.py [--family ridge|...]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "src"))

from orbit.ml.audit import audit_summary, run_phase9_audit  # noqa: E402
from orbit.ml.calibration import fit_platt  # noqa: E402
from orbit.ml.data import (  # noqa: E402
    bars_meta,
    load_instrument_master,
    load_snapshot_bars,
    load_snapshot_events,
)
from orbit.ml.dataset import assemble_datasets  # noqa: E402
from orbit.ml.grids import MODEL_FAMILIES, PHASE9_GRIDS, params_identity  # noqa: E402
from orbit.ml.models import predict_with_state, train_model  # noqa: E402
from orbit.ml.ranking import cross_sectional_rank  # noqa: E402
from orbit.ml.registry import (  # noqa: E402
    experiment_id_for,
    register_ml_experiment,
)
from orbit.ml.signals import (  # noqa: E402
    build_backtest_config,
    predictions_to_signals,
    run_backtest,
)
from orbit.ml.snapshot_cache import build_or_load_snapshots  # noqa: E402
from orbit.ml.splits import PHASE9_WINDOWS  # noqa: E402

AUDIT_RESULTS = _REPO_ROOT / "benchmarks" / "phase9_audit_results.json"

# One representative pre-registered grid point per family for the audit.
REPRESENTATIVE_POINTS: dict[str, dict] = {
    "ridge": {"alpha": 1.0},
    "lasso": {"alpha": 0.001},
    "logistic": {"C": 1.0},
    "random_forest": {"n_estimators": 50, "max_depth": 3},
    "xgboost": {"n_estimators": 50, "max_depth": 3, "learning_rate": 0.1},
}

EXPECTED_CHECKS = {
    "feature_point_in_time",
    "feature_scope_guard",
    "label_contract",
    "label_availability",
    "split_integrity",
    "unavailable_documented",
    "calibration_val_only",
    "grid_lock",
    "seed_lock",
    "model_scope_guard",
    "test_lock",
    "backtest_uniformity",
    "registry_lineage",
    "data_expansion_guard",
}


def deep_audit_one(family: str, params: dict, bars, events, instruments, datasets) -> list[dict]:
    """Run the full 14-check audit for one experiment with all artifacts."""
    (Xtr, ytr, _, _), (Xva, _, ybin_va, _), (Xte, yte, _, meta_te) = (
        datasets["train"],
        datasets["val"],
        datasets["test"],
    )
    exp_id = experiment_id_for(family, params)
    service, spec = register_ml_experiment(
        experiment_id=exp_id, hypothesis_id="H-001", family=family, params=params
    )

    model, state = train_model(family, params, Xtr, ytr, windows={
        "train": ("2010-01-04", "2018-12-31"),
        "val": ("2019-01-02", "2021-12-31"),
        "test": ("2022-01-03", "2026-06-30"),
    })
    pred_val = predict_with_state(model, state, Xva)
    pred_test = predict_with_state(model, state, Xte)
    calibrator = fit_platt(pred_val, ybin_va)

    test_frame = meta_te.with_columns(pl.Series("prediction", pred_test))
    test_frame = test_frame.with_columns(
        pl.Series("calibrated_prob", calibrator.apply(pred_test))
    )
    ranked = cross_sectional_rank(test_frame, "prediction")

    signals = predictions_to_signals(
        test_frame, family=family, params=params, top_k=3,
        strategy_ref=f"phase9:{family}:{params_identity(family, params)}:topk3",
    )
    result = run_backtest(
        bars, signals,
        window_start=PHASE9_WINDOWS["test_start"],
        window_end=PHASE9_WINDOWS["test_end"],
        experiment_id=exp_id, hypothesis_id="H-001", events=events,
        feature_refs=[{"feature_id": f, "feature_version": "v1"} for f in
                      ("FEAT-001", "FEAT-008")],
        model={"family": family, "hyperparameters": params},
        label_id="LAB-004", label_version="v1",
    )
    backtest_config = build_backtest_config(
        window_start=PHASE9_WINDOWS["test_start"],
        window_end=PHASE9_WINDOWS["test_end"],
    )

    return run_phase9_audit(
        feature_snapshot=datasets["feature_snapshot"],
        label_snapshot=datasets["label_snapshot"],
        datasets=datasets,
        fitted_model=model,
        calibration_map=calibrator,
        ranking_frame=ranked,
        backtest_config=backtest_config,
        experiment_spec=spec,
        test_predictions=test_frame,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 9 deep-input independent audit")
    parser.add_argument("--family", choices=list(MODEL_FAMILIES) + ["all"], default="all")
    args = parser.parse_args()

    print("[phase9 audit] deep-input independent audit (Review 1)")
    t_start = time.time()

    bars = load_snapshot_bars()
    events = load_snapshot_events()
    instruments = load_instrument_master()
    print(f"[phase9 audit] data: {bars_meta(bars)}")

    fs, ls = build_or_load_snapshots(bars, events, instruments)
    datasets = assemble_datasets(fs, ls)
    datasets["feature_snapshot"] = fs
    datasets["label_snapshot"] = ls
    print(f"[phase9 audit] datasets: {datasets['report']}")

    families = [args.family] if args.family != "all" else list(MODEL_FAMILIES)
    all_checks: list[dict] = []
    summary_rows: list[dict] = []
    overall_pass = True
    for family in families:
        params = REPRESENTATIVE_POINTS[family]
        checks = deep_audit_one(family, params, bars, events, instruments, datasets)
        summary = audit_summary(checks)
        all_checks.append({"family": family, "params": params, "checks": checks, "summary": summary})
        print(f"[phase9 audit] {family}: {summary}")
        overall_pass = overall_pass and not summary["blocked"]

    # the audit must have exercised every expected check at least once
    exercised = {c["check"] for run in all_checks for c in run["checks"]}
    missing = sorted(EXPECTED_CHECKS - exercised)
    if missing:
        overall_pass = False
        print(f"[phase9 audit] MISSING CHECKS: {missing}")

    payload = {
        "protocol": "phase9_deep_audit_v1",
        "dataset_snapshot_ids": ["DS-000004"],
        "feature_set_id": "FS-001",
        "label_id": "LAB-004",
        "windows": {
            "train": "2010-01-04..2018-12-31",
            "val": "2019-01-02..2021-12-31",
            "test": "2022-01-03..2026-06-30",
        },
        "families_audited": families,
        "all_checks_exercised": list(sorted(exercised)),
        "missing_checks": missing,
        "runs": all_checks,
        "verdict": "PASS" if overall_pass else "FAIL",
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    AUDIT_RESULTS.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_RESULTS.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    print(f"[phase9 audit] results written: {AUDIT_RESULTS}")
    print(f"[phase9 audit] VERDICT: {'PASS' if overall_pass else 'FAIL'} "
          f"({time.time() - t_start:.1f}s)")
    sys.exit(0 if overall_pass else 1)


if __name__ == "__main__":
    main()