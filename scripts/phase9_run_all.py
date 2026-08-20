"""Phase 9 - Baseline ML Benchmark: full run.

Run:  python scripts/phase9_run_all.py

Pipeline (register-before-run throughout):
  1. load DS-000004 + instrument master
  2. build or load (digest-verified) FS-001 feature + LAB-004 label snapshots
  3. assemble strict train/val/test matrices with outcome-window purge
  4. run all 20 pre-registered ML grid points (ridge, lasso, logistic,
     random_forest, xgboost): train on train, calibrate on validation,
     evaluate OOS IC / rank IC / ECE / Brier / MSE / hit rate on the locked
     test window, convert top-3 predictions to canonical signals, backtest
     through the canonical Phase 7 engine with CM-001 costs (identical to
     the controls), and record every experiment through the Phase 6 lifecycle
  5. run the Phase 8 controls on the real dataset through the same path
  6. structural + reproducibility audit (Phase 9 audit, independent pass)
  7. write the permanent benchmark report (parquet + markdown)

Every experiment id is deterministic (EXP-9xxxx). All runs, including null
and failed ones, are recorded in the report.
"""

from __future__ import annotations

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
from orbit.ml.baselines import CONTROL_FAMILIES, CONTROL_GRIDS, build_control_signals  # noqa: E402
from orbit.ml.calibration import brier_score, expected_calibration_error, fit_platt  # noqa: E402
from orbit.ml.data import (  # noqa: E402
    bars_meta,
    load_instrument_master,
    load_snapshot_bars,
    load_snapshot_events,
)
from orbit.ml.dataset import assemble_datasets  # noqa: E402
from orbit.ml.grids import PHASE9_GRIDS, MODEL_FAMILIES, params_identity  # noqa: E402
from orbit.ml.metrics import hit_rate, mean_squared_error, oos_ic, rank_ic  # noqa: E402
from orbit.ml.models import predict_with_state, train_model  # noqa: E402
from orbit.ml.registry import (  # noqa: E402
    control_experiment_id_for,
    experiment_id_for,
    register_control_experiment,
    register_ml_experiment,
    run_registered_experiment,
)
from orbit.ml.report import append_report_rows, write_markdown_report  # noqa: E402
from orbit.ml.signals import predictions_to_signals, run_backtest  # noqa: E402
from orbit.ml.snapshot_cache import build_or_load_snapshots  # noqa: E402
from orbit.ml.splits import PHASE9_WINDOWS  # noqa: E402

ARTIFACTS_ROOT = _REPO_ROOT / "benchmarks" / "phase9_runs"
TOP_K = 3


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _test_sessions(bars: pl.DataFrame) -> list:
    return sorted(
        bars.filter(pl.col("trade_date") >= PHASE9_WINDOWS["test_start"])
        .filter(pl.col("trade_date") <= PHASE9_WINDOWS["test_end"])["trade_date"]
        .unique()
        .to_list()
    )


def run_one_ml(
    family: str,
    params: dict[str, object],
    bars: pl.DataFrame,
    events: pl.DataFrame | None,
    datasets: dict,
) -> dict:
    exp_id = experiment_id_for(family, params)
    _log(f"ML experiment {exp_id}: {family} {params}")
    (Xtr, ytr, _, _), (Xva, _, ybin_va, meta_va), (Xte, yte, ybin_te, meta_te) = (
        datasets["train"],
        datasets["val"],
        datasets["test"],
    )
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

    # calibration: fit Platt on validation only, apply to test
    calibrator = fit_platt(pred_val, ybin_va)
    cal_test = calibrator.apply(pred_test)
    calib_eval = expected_calibration_error(cal_test, ybin_te, n_bins=10)
    brier = brier_score(cal_test, ybin_te)

    test_frame = meta_te.with_columns(pl.Series("prediction", pred_test))
    ic = oos_ic(test_frame, "prediction")
    ric = rank_ic(test_frame, "prediction")
    mse = mean_squared_error(yte, pred_test)
    hit = hit_rate(test_frame, "prediction")

    signals = predictions_to_signals(
        test_frame, family=family, params=params, top_k=TOP_K,
        strategy_ref=f"phase9:{family}:{params_identity(family, params)}:topk{TOP_K}",
    )
    result = run_backtest(
        bars, signals,
        window_start=PHASE9_WINDOWS["test_start"],
        window_end=PHASE9_WINDOWS["test_end"],
        experiment_id=exp_id, hypothesis_id="H-001", events=events,
        feature_refs=[{"feature_id": f, "feature_version": "v1"} for f in ("FEAT-001", "FEAT-008")],
        model={"family": family, "hyperparameters": params},
        label_id="LAB-004", label_version="v1",
    )
    summary = result.summary()

    metrics = {
        "model_family": family,
        "hyperparameters": params,
        "seed": 42,
        "oos_ic": ic["value"],
        "rank_ic": ric["value"],
        "ic_sessions_used": ic["sessions_used"],
        "ece": calib_eval["ece"],
        "brier": brier,
        "mse": mse,
        "hit_rate": hit,
        "after_cost_total_return": summary["total_return"],
        "after_cost_final_equity": summary["final_equity"],
        "turnover": summary["turnover"],
        "total_costs": summary["total_fees"],
        "n_fills": summary["n_fills"],
        "n_rejects": summary["n_rejects"],
        "n_signals": summary["n_signals"],
        "coefs": model.coefficients() if family in ("ridge", "lasso", "logistic") else None,
        "calibration_slope": calibrator.slope,
        "calibration_intercept": calibrator.intercept,
        "run_id": result.run_id,
    }

    exp_dir = ARTIFACTS_ROOT / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    pred_file = exp_dir / "test_predictions.parquet"
    test_frame.with_columns(
        pl.Series("calibrated_prob", cal_test)
    ).write_parquet(pred_file)
    metrics_file = exp_dir / "metrics.json"
    metrics_file.write_text(json.dumps(metrics, sort_keys=True, default=str), encoding="utf-8")

    result_summary = (
        f"Phase 9 {family} {params}: OOS IC {ic['value']:.4f}, rank IC "
        f"{ric['value']:.4f}, ECE {calib_eval['ece']:.4f}, after-cost return "
        f"{summary['total_return']:.4%}"
    )
    run_registered_experiment(
        service, exp_id, family=family, params=params, seed=42,
        artifacts_dir=ARTIFACTS_ROOT,
        result_summary=result_summary,
        result_metrics=metrics,
        artifact_files={
            "test_predictions_parquet": pred_file,
            "metrics_json": metrics_file,
        },
    )

    return {
        "run_kind": "ml",
        "experiment_id": exp_id,
        "family": family,
        "params": json.dumps(params, sort_keys=True),
        "seed": 42,
        "status": "completed",
        "oos_ic": ic["value"],
        "rank_ic": ric["value"],
        "ece": calib_eval["ece"],
        "brier": brier,
        "mse": mse,
        "hit_rate": hit,
        "after_cost_total_return": summary["total_return"],
        "after_cost_final_equity": summary["final_equity"],
        "turnover": summary["turnover"],
        "total_costs": summary["total_fees"],
        "n_fills": summary["n_fills"],
        "n_rejects": summary["n_rejects"],
        "n_signals": summary["n_signals"],
        "feature_set_id": "FS-001",
        "label_id": "LAB-004",
        "dataset_snapshot_ids": "DS-000004",
        "cost_model_id": "CM-001",
        "evaluation_window": "2022-01-03..2026-06-30",
        "notes": f"top-{TOP_K} long, weight 1/{TOP_K}; calibration fit on validation",
    }


def run_one_control(
    family: str,
    params: dict[str, object],
    bars: pl.DataFrame,
    events: pl.DataFrame | None,
    sessions: list,
) -> dict:
    exp_id = control_experiment_id_for(family, params)
    _log(f"Control experiment {exp_id}: {family} {params}")
    service, spec = register_control_experiment(
        experiment_id=exp_id, hypothesis_id="H-001", family=family, params=params
    )
    signals = build_control_signals(bars, sessions, family, params)
    result = run_backtest(
        bars, signals,
        window_start=PHASE9_WINDOWS["test_start"],
        window_end=PHASE9_WINDOWS["test_end"],
        experiment_id=exp_id, hypothesis_id="H-001", events=events,
        feature_refs=[{"feature_id": "FEAT-901", "feature_version": "v1", "transformation": "phase8_documented_rules"}],
        model={"family": "baseline", "hyperparameters": params},
        label_id="LAB-004", label_version="v1",
    )
    summary = result.summary()
    metrics = {
        "model_family": family,
        "hyperparameters": params,
        "seed": 42,
        "after_cost_total_return": summary["total_return"],
        "after_cost_final_equity": summary["final_equity"],
        "turnover": summary["turnover"],
        "total_costs": summary["total_fees"],
        "n_fills": summary["n_fills"],
        "n_rejects": summary["n_rejects"],
        "n_signals": summary["n_signals"],
        "run_id": result.run_id,
    }
    exp_dir = ARTIFACTS_ROOT / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    sig_file = exp_dir / "signals.parquet"
    signals.write_parquet(sig_file)
    metrics_file = exp_dir / "metrics.json"
    metrics_file.write_text(json.dumps(metrics, sort_keys=True, default=str), encoding="utf-8")

    run_registered_experiment(
        service, exp_id, family=family, params=params, seed=42,
        artifacts_dir=ARTIFACTS_ROOT,
        result_summary=(
            f"Phase 9 control {family} {params}: after-cost return "
            f"{summary['total_return']:.4%}"
        ),
        result_metrics=metrics,
        artifact_files={
            "signals_parquet": sig_file,
            "metrics_json": metrics_file,
        },
    )
    return {
        "run_kind": "control",
        "experiment_id": exp_id,
        "family": family,
        "params": json.dumps(params, sort_keys=True),
        "seed": 42,
        "status": "completed",
        "oos_ic": None,
        "rank_ic": None,
        "ece": None,
        "brier": None,
        "mse": None,
        "hit_rate": None,
        "after_cost_total_return": summary["total_return"],
        "after_cost_final_equity": summary["final_equity"],
        "turnover": summary["turnover"],
        "total_costs": summary["total_fees"],
        "n_fills": summary["n_fills"],
        "n_rejects": summary["n_rejects"],
        "n_signals": summary["n_signals"],
        "feature_set_id": "phase8_controls",
        "label_id": "LAB-004",
        "dataset_snapshot_ids": "DS-000004",
        "cost_model_id": "CM-001",
        "evaluation_window": "2022-01-03..2026-06-30",
        "notes": "Phase 8 documented rules on the real dataset; strict point-in-time boundary",
    }


def main() -> None:
    _log("Phase 9 baseline ML benchmark - full run")
    t_start = time.time()

    bars = load_snapshot_bars()
    events = load_snapshot_events()
    instruments = load_instrument_master()
    _log(f"data: {bars_meta(bars)}")

    fs, ls = build_or_load_snapshots(bars, events, instruments)
    _log(f"snapshots: features {fs.provenance()['row_count']} rows, "
         f"labels {ls.provenance()['row_count']} rows")

    datasets = assemble_datasets(fs, ls)
    _log(f"datasets: {datasets['report']}")

    sessions = _test_sessions(bars)
    _log(f"test window sessions: {len(sessions)} ({sessions[0]} .. {sessions[-1]})")

    rows = []
    for family in MODEL_FAMILIES:
        for params in PHASE9_GRIDS[family]:
            try:
                rows.append(run_one_ml(family, params, bars, events, datasets))
            except Exception as exc:  # noqa: BLE001 - never hide a failed run
                _log(f"ML experiment FAILED: {family} {params}: {type(exc).__name__}: {exc}")
                rows.append({
                    "run_kind": "ml",
                    "experiment_id": experiment_id_for(family, params),
                    "family": family,
                    "params": json.dumps(params, sort_keys=True),
                    "seed": 42,
                    "status": "failed",
                    "notes": f"{type(exc).__name__}: {exc}",
                })

    for family in CONTROL_FAMILIES:
        grid = CONTROL_GRIDS.get(family)
        points = grid if grid else [{}]
        for params in points:
            try:
                rows.append(run_one_control(family, params, bars, events, sessions))
            except Exception as exc:  # noqa: BLE001
                _log(f"Control experiment FAILED: {family} {params}: {type(exc).__name__}: {exc}")
                rows.append({
                    "run_kind": "control",
                    "experiment_id": control_experiment_id_for(family, params),
                    "family": family,
                    "params": json.dumps(params, sort_keys=True),
                    "seed": 42,
                    "status": "failed",
                    "notes": f"{type(exc).__name__}: {exc}",
                })

    _log("writing benchmark report")
    report_path = append_report_rows(rows)
    md_path = write_markdown_report()
    _log(f"report written: {report_path} / {md_path}")

    # structural + reproducibility audit
    audit_checks = run_phase9_audit(
        feature_snapshot=fs,
        label_snapshot=ls,
        datasets=datasets,
        fitted_model=None,
        experiment_spec=None,
    )
    audit_summary_result = audit_summary(audit_checks)
    _log(f"audit: {audit_summary_result}")

    _log(f"total wall time: {time.time() - t_start:.1f}s")
    _log("PHASE 9 STATUS: benchmark report generated; verdict must follow the "
         "full checklist in docs/phase9_ml_benchmark.md")


if __name__ == "__main__":
    main()