"""Phase 10 - Review 2: independent reproducibility double-run.

The benchmark runner stores one artifact set per experiment in
benchmarks/phase10_runs/EXP-1xxxx/. This script recomputes a selection of
those experiments from scratch through the FULL pipeline (digest-verified
snapshot cache -> assemble -> train -> predict -> calibrate -> signals ->
Phase 7 backtest) and compares the fresh artifacts against the stored ones:

  - EXP-10001  FS-001 BASE   ridge alpha 1.0   (the frozen Phase 9 baseline)
  - EXP-10009  FS-003 ALL    ridge alpha 1.0   (full feature representation)
  - EXP-10052  FS-013 ALL-range xgboost 200/3/lr0.1 (a tree family point)

For each: test_predictions.parquet must be bitwise identical (sha256) and
metrics exactly equal (run_id is content-derived, so backtest identity is
included). Any mismatch blocks the reproducibility verdict.

Run:  python scripts/phase10_review2.py
Exit code 0 = all selected experiments reproduce bitwise.
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import time
from pathlib import Path

import polars as pl

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "src"))

from orbit.ml.calibration import (  # noqa: E402
    brier_score,
    expected_calibration_error,
    fit_platt,
)
from orbit.ml.data import (  # noqa: E402
    load_instrument_master,
    load_snapshot_bars,
    load_snapshot_events,
)
from orbit.ml.dataset import assemble_datasets  # noqa: E402
from orbit.ml.features import (  # noqa: E402
    PHASE10_FEATURE_SETS,
)
from orbit.ml.metrics import hit_rate, mean_squared_error, oos_ic, rank_ic  # noqa: E402
from orbit.ml.models import predict_with_state, train_model  # noqa: E402
from orbit.ml.phase10_registry import (  # noqa: E402
    register_phase10_experiment,
)
from orbit.ml.phase10_plan import phase10_model_point_for  # noqa: E402
from orbit.ml.phase10_runner import _feature_names  # noqa: E402
from orbit.ml.signals import predictions_to_signals, run_backtest  # noqa: E402
from orbit.ml.snapshot_cache import (  # noqa: E402
    PHASE10_SNAPSHOT_CACHE_DIR,
    load_cached_feature_snapshot,
    load_cached_phase10_snapshot,
)
from orbit.ml.splits import PHASE9_WINDOWS  # noqa: E402

ARTIFACTS_ROOT = _REPO_ROOT / "benchmarks" / "phase10_runs"
RESULT_FILE = _REPO_ROOT / "benchmarks" / "phase10_review2_results.json"
TOP_K = 3

# (experiment_id, feature_set_id, family, params)
REVIEW_POINTS = [
    ("EXP-10001", "FS-001", "ridge", {"alpha": 1.0}),
    ("EXP-10009", "FS-003", "ridge", {"alpha": 1.0}),
    ("EXP-10052", "FS-013", "xgboost", {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.1}),
]

_WINDOWS = {
    "train": ("2010-01-04", "2018-12-31"),
    "val": ("2019-01-02", "2021-12-31"),
    "test": ("2022-01-03", "2026-06-30"),
}


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def recompute(exp_id: str, feature_set_id: str, family: str, params: dict, bars, events, snapshots, ls, tmp: Path):
    snap = snapshots[feature_set_id]
    names = list(_feature_names(snap))
    datasets = assemble_datasets(snap, ls, feature_names=names)
    (Xtr, ytr, _, _), (Xva, _, ybin_va, _), (Xte, yte, ybin_te, meta_te) = (
        datasets["train"], datasets["val"], datasets["test"]
    )

    register_phase10_experiment(
        experiment_id=exp_id, hypothesis_id="H-001",
        feature_set_id=feature_set_id,
        feature_set_version=snap.feature_set_version,
        family=family, params=params, seed=42,
    )
    model, state = train_model(family, params, Xtr, ytr, feature_names=names, windows=_WINDOWS)
    pred_val = predict_with_state(model, state, Xva)
    pred_test = predict_with_state(model, state, Xte)
    calibrator = fit_platt(pred_val, ybin_va)
    cal_test = calibrator.apply(pred_test)
    calib = expected_calibration_error(cal_test, ybin_te, n_bins=10)

    test_frame = meta_te.with_columns(pl.Series("prediction", pred_test))
    ic = oos_ic(test_frame, "prediction")
    ric = rank_ic(test_frame, "prediction")
    mse = mean_squared_error(yte, pred_test)
    hit = hit_rate(test_frame, "prediction")

    signals = predictions_to_signals(
        test_frame, family=family, params=params, top_k=TOP_K,
        strategy_ref=f"phase10:{feature_set_id}:{family}:"
        f"{'-'.join(f'{k}{v}' for k, v in sorted(params.items()))}:topk{TOP_K}",
    )
    result = run_backtest(
        bars, signals,
        window_start=PHASE9_WINDOWS["test_start"],
        window_end=PHASE9_WINDOWS["test_end"],
        experiment_id=exp_id, hypothesis_id="H-001", events=events,
        feature_refs=[
            {"feature_id": fid, "feature_version": "v1"}
            for fid in snap.feature_refs
        ],
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
        "ece": calib["ece"],
        "brier": brier_score(cal_test, ybin_te),
        "mse": mse,
        "hit_rate": hit,
        "after_cost_total_return": summary["total_return"],
        "after_cost_final_equity": summary["final_equity"],
        "turnover": summary["turnover"],
        "total_costs": summary["total_fees"],
        "n_fills": summary["n_fills"],
        "n_rejects": summary["n_rejects"],
        "n_signals": summary["n_signals"],
        "calibration_slope": calibrator.slope,
        "calibration_intercept": calibrator.intercept,
        "run_id": result.run_id,
    }

    pred_file = tmp / f"{exp_id}_test_predictions.parquet"
    test_frame.with_columns(pl.Series("calibrated_prob", cal_test)).write_parquet(pred_file)
    metrics_file = tmp / f"{exp_id}_metrics.json"
    metrics_file.write_text(json.dumps(metrics, sort_keys=True, default=str), encoding="utf-8")
    return pred_file, metrics_file


def main() -> None:
    print("[review2] Phase 10 reproducibility double-run (independent pass)")
    t_start = time.time()

    bars = load_snapshot_bars()
    events = load_snapshot_events()
    instruments = load_instrument_master()

    snapshots = {}
    fs001 = load_cached_feature_snapshot()
    if fs001 is not None:
        snapshots["FS-001"] = fs001
    for sid in PHASE10_FEATURE_SETS:
        s = load_cached_phase10_snapshot(sid, PHASE10_SNAPSHOT_CACHE_DIR)
        if s is not None:
            snapshots[sid] = s
    if "FS-001" not in snapshots:
        raise SystemExit("[review2] FS-001 snapshot not found in the Phase 9 cache")
    decisions = snapshots["FS-001"].records.select("instrument_id", "decision_time")
    from orbit.ml.labels import build_phase9_label_snapshot

    ls = build_phase9_label_snapshot(bars, events, instruments, decisions, data_refs=["DS-000004"])

    report: list[dict] = []
    all_pass = True
    with tempfile.TemporaryDirectory(prefix="phase10_review2_") as tmp:
        tmp_path = Path(tmp)
        for exp_id, feature_set_id, family, params in REVIEW_POINTS:
            stored_dir = ARTIFACTS_ROOT / exp_id
            stored_artifact = stored_dir / "test_predictions.parquet"
            stored_metrics = stored_dir / "metrics.json"
            entry = {
                "experiment_id": exp_id,
                "feature_set_id": feature_set_id,
                "family": family,
                "params": params,
            }
            try:
                pred_file, metrics_file = recompute(
                    exp_id, feature_set_id, family, params, bars, events, snapshots, ls, tmp_path
                )
                sha_stored = _sha256_file(stored_artifact)
                sha_fresh = _sha256_file(pred_file)
                artifact_ok = sha_stored == sha_fresh
                entry["artifact"] = {
                    "file": "test_predictions.parquet",
                    "stored_sha256": sha_stored,
                    "fresh_sha256": sha_fresh,
                    "bitwise_identical": artifact_ok,
                }
                stored_dict = json.loads(stored_metrics.read_text(encoding="utf-8"))
                fresh_dict = json.loads(metrics_file.read_text(encoding="utf-8"))
                diffs = {}
                for key in sorted(set(stored_dict) | set(fresh_dict)):
                    if stored_dict.get(key) != fresh_dict.get(key):
                        diffs[key] = {"stored": stored_dict.get(key), "fresh": fresh_dict.get(key)}
                metrics_ok = not diffs
                entry["metrics"] = {"exactly_equal": metrics_ok, "diffs": diffs}
                entry["status"] = "PASS" if (artifact_ok and metrics_ok) else "FAIL"
            except Exception as exc:  # noqa: BLE001 - a failed recompute is itself a finding
                entry["status"] = "FAIL"
                entry["error"] = f"{type(exc).__name__}: {exc}"
            all_pass = all_pass and entry["status"] == "PASS"
            report.append(entry)
            print(f"[review2] {entry['status']}: {exp_id} {feature_set_id} {family} {params}")

    payload = {
        "protocol": "phase10_review2_v1",
        "seed": 42,
        "dataset_snapshot_ids": ["DS-000004"],
        "label_id": "LAB-004",
        "test_window": "2022-01-03..2026-06-30",
        "runs": report,
        "verdict": "PASS" if all_pass else "FAIL",
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    RESULT_FILE.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    print(f"[review2] results written: {RESULT_FILE}")
    print(f"[review2] VERDICT: {'PASS' if all_pass else 'FAIL'} ({time.time() - t_start:.1f}s)")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()