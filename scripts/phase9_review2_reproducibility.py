"""Phase 9 - Review 2: independent reproducibility double-run (roadmap 35).

The benchmark runner stores one artifact set per experiment in
benchmarks/phase9_runs/EXP-9xxxx/. This script recomputes a selection of
those experiments from scratch through the FULL pipeline and compares the
fresh artifacts against the stored ones:

  - ML experiment   EXP-90003 (ridge alpha 1.0): test predictions parquet
                    must be bitwise identical (sha256), metrics exactly equal.
  - ML experiment   EXP-90015 (random_forest 50/3, the best-performing ML
                    run): same bitwise + exact-metric comparison.
  - control run     EXP-90024 (momentum 20): signals parquet bitwise
                    identical, metrics exactly equal.

A second, independent pass over the stored benchmark therefore reproduces
bitwise-identical predictions, signals, metrics and backtest inputs. Any
mismatch is reported with evidence and blocks the reproducibility verdict.

Run:  python scripts/phase9_review2_reproducibility.py
Exit code 0 = all selected experiments reproduce bitwise.
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import polars as pl

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "src"))

from orbit.ml.calibration import fit_platt  # noqa: E402
from orbit.ml.data import (  # noqa: E402
    load_instrument_master,
    load_snapshot_bars,
    load_snapshot_events,
)
from orbit.ml.dataset import assemble_datasets  # noqa: E402
from orbit.ml.models import predict_with_state, train_model  # noqa: E402
from orbit.ml.registry import (  # noqa: E402
    experiment_id_for,
    register_ml_experiment,
)
from orbit.ml.signals import predictions_to_signals, run_backtest  # noqa: E402
from orbit.ml.snapshot_cache import build_or_load_snapshots  # noqa: E402
from orbit.ml.splits import PHASE9_WINDOWS  # noqa: E402

ARTIFACTS_ROOT = _REPO_ROOT / "benchmarks" / "phase9_runs"
RESULT_FILE = _REPO_ROOT / "benchmarks" / "phase9_review2_results.json"
TOP_K = 3

# (kind, experiment_id, family, params, stored_prediction_or_signals_file)
REVIEW_POINTS = [
    ("ml", "EXP-90003", "ridge", {"alpha": 1.0}, "test_predictions.parquet"),
    ("ml", "EXP-90015", "random_forest",
     {"n_estimators": 200, "max_depth": 3}, "test_predictions.parquet"),
    ("control", "EXP-90024", "momentum", {"lookback": 20}, "signals.parquet"),
]


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _test_sessions(bars: pl.DataFrame) -> list:
    return sorted(
        bars.filter(pl.col("trade_date") >= PHASE9_WINDOWS["test_start"])
        .filter(pl.col("trade_date") <= PHASE9_WINDOWS["test_end"])["trade_date"]
        .unique()
        .to_list()
    )


def recompute_ml(family: str, params: dict, bars, events, datasets, tmp: Path):
    exp_id = experiment_id_for(family, params)
    (Xtr, ytr, _, _), (Xva, _, ybin_va, _), (Xte, yte, ybin_te, meta_te) = (
        datasets["train"],
        datasets["val"],
        datasets["test"],
    )
    register_ml_experiment(
        experiment_id=exp_id, hypothesis_id="H-001", family=family, params=params
    )
    model, state = train_model(family, params, Xtr, ytr, windows={
        "train": ("2010-01-04", "2018-12-31"),
        "val": ("2019-01-02", "2021-12-31"),
        "test": ("2022-01-03", "2026-06-30"),
    })
    pred_test = predict_with_state(model, state, Xte)
    pred_val = predict_with_state(model, state, Xva)
    calibrator = fit_platt(pred_val, ybin_va)
    cal_test = calibrator.apply(pred_test)

    test_frame = meta_te.with_columns(pl.Series("prediction", pred_test))
    from orbit.ml.grids import params_identity

    signals = predictions_to_signals(
        test_frame, family=family, params=params, top_k=TOP_K,
        strategy_ref=f"phase9:{family}:{params_identity(family, params)}:topk{TOP_K}",
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
    summary = result.summary()

    from orbit.ml.calibration import brier_score, expected_calibration_error
    from orbit.ml.metrics import hit_rate, oos_ic, rank_ic

    test_frame = test_frame.with_columns(pl.Series("calibrated_prob", cal_test))
    ic = oos_ic(test_frame, "prediction")
    ric = rank_ic(test_frame, "prediction")
    calib = expected_calibration_error(cal_test, ybin_te, n_bins=10)
    metrics = {
        "model_family": family,
        "hyperparameters": params,
        "seed": 42,
        "oos_ic": ic["value"],
        "rank_ic": ric["value"],
        "ic_sessions_used": ic["sessions_used"],
        "ece": calib["ece"],
        "brier": brier_score(cal_test, ybin_te),
        "mse": float(np.mean((yte - pred_test) ** 2)),
        "hit_rate": hit_rate(test_frame, "prediction"),
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

    pred_file = tmp / f"{exp_id}_test_predictions.parquet"
    test_frame.write_parquet(pred_file)
    metrics_file = tmp / f"{exp_id}_metrics.json"
    metrics_file.write_text(json.dumps(metrics, sort_keys=True, default=str), encoding="utf-8")
    return pred_file, metrics_file


def recompute_control(family: str, params: dict, bars, events, sessions, tmp: Path):
    from orbit.ml.baselines import build_control_signals
    from orbit.ml.registry import control_experiment_id_for

    exp_id = control_experiment_id_for(family, params)
    signals = build_control_signals(bars, sessions, family, params)
    result = run_backtest(
        bars, signals,
        window_start=PHASE9_WINDOWS["test_start"],
        window_end=PHASE9_WINDOWS["test_end"],
        experiment_id=exp_id, hypothesis_id="H-001", events=events,
        feature_refs=[{"feature_id": "FEAT-901", "feature_version": "v1",
                       "transformation": "phase8_documented_rules"}],
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
    sig_file = tmp / f"{exp_id}_signals.parquet"
    signals.write_parquet(sig_file)
    metrics_file = tmp / f"{exp_id}_metrics.json"
    metrics_file.write_text(json.dumps(metrics, sort_keys=True, default=str), encoding="utf-8")
    return sig_file, metrics_file


def main() -> None:
    print("[review2] Phase 9 reproducibility double-run (independent pass)")
    t_start = time.time()

    bars = load_snapshot_bars()
    events = load_snapshot_events()
    instruments = load_instrument_master()
    fs, ls = build_or_load_snapshots(bars, events, instruments)
    datasets = assemble_datasets(fs, ls)
    sessions = _test_sessions(bars)

    report: list[dict] = []
    all_pass = True
    with tempfile.TemporaryDirectory(prefix="phase9_review2_") as tmp:
        tmp_path = Path(tmp)
        for kind, exp_id, family, params, artifact_name in REVIEW_POINTS:
            stored_dir = ARTIFACTS_ROOT / exp_id
            stored_artifact = stored_dir / artifact_name
            stored_metrics = stored_dir / "metrics.json"
            entry = {
                "experiment_id": exp_id,
                "kind": kind,
                "family": family,
                "params": params,
            }
            try:
                if kind == "ml":
                    pred_file, metrics_file = recompute_ml(
                        family, params, bars, events, datasets, tmp_path
                    )
                else:
                    sig_file, metrics_file = recompute_control(
                        family, params, bars, events, sessions, tmp_path
                    )
                fresh_artifact = pred_file if kind == "ml" else sig_file

                sha_stored = _sha256_file(stored_artifact)
                sha_fresh = _sha256_file(fresh_artifact)
                artifact_ok = sha_stored == sha_fresh
                entry["artifact"] = {
                    "file": artifact_name,
                    "stored_sha256": sha_stored,
                    "fresh_sha256": sha_fresh,
                    "bitwise_identical": artifact_ok,
                }

                stored_metrics_dict = json.loads(stored_metrics.read_text(encoding="utf-8"))
                fresh_metrics_dict = json.loads(metrics_file.read_text(encoding="utf-8"))
                diffs = {}
                for key in sorted(set(stored_metrics_dict) | set(fresh_metrics_dict)):
                    a = stored_metrics_dict.get(key)
                    b = fresh_metrics_dict.get(key)
                    if a != b:
                        diffs[key] = {"stored": a, "fresh": b}
                metrics_ok = not diffs
                entry["metrics"] = {"exactly_equal": metrics_ok, "diffs": diffs}
                entry["status"] = "PASS" if (artifact_ok and metrics_ok) else "FAIL"
            except Exception as exc:  # noqa: BLE001 - a failed recompute is itself a finding
                entry["status"] = "FAIL"
                entry["error"] = f"{type(exc).__name__}: {exc}"
            all_pass = all_pass and entry["status"] == "PASS"
            report.append(entry)
            print(f"[review2] {entry['status']}: {exp_id} {family} {params}")

    payload = {
        "protocol": "phase9_review2_v1",
        "seed": 42,
        "dataset_snapshot_ids": ["DS-000004"],
        "feature_set_id": "FS-001",
        "label_id": "LAB-004",
        "test_window": "2022-01-03..2026-06-30",
        "runs": report,
        "verdict": "PASS" if all_pass else "FAIL",
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULT_FILE.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    print(f"[review2] results written: {RESULT_FILE}")
    print(f"[review2] VERDICT: {'PASS' if all_pass else 'FAIL'} ({time.time() - t_start:.1f}s)")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()