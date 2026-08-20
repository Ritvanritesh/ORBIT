"""Phase 10 runner: executes the locked ablation plan end to end.

Pipeline (register-before-run throughout, identical execution path to Phase 9):

  1. load DS-000004 + instrument master + events; verify the dataset is
     unchanged (data-expansion guard)
  2. build/load (digest-verified) the FS-001 base snapshot (frozen Phase 9
     artifact), the FS-002..FS-013 feature-set snapshots, and the LAB-004
     label snapshot
  3. run feature quality + redundancy diagnostics on the TRAINING split
  4. for every (feature_set, model) in the locked plan (52 experiments):
     register -> train on train -> predict val/test -> calibrate on val ->
     evaluate OOS IC / rank IC / ECE / Brier / MSE / hit rate on the locked
     test window -> top-3 signals -> canonical Phase 7 backtest with CM-001
     costs -> record through the Phase 6 lifecycle
  5. write the permanent report (parquet + markdown + diagnostics + plan)
  6. run the Phase 10 independent audit

Nothing in Phase 10 mutates Phase 9 artifacts: the Phase 9 snapshot cache,
the Phase 9 benchmark report, and DS-000004 are read-only inputs here.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from orbit.ml.calibration import brier_score, expected_calibration_error, fit_platt
from orbit.ml.data import (
    load_instrument_master,
    load_snapshot_bars,
    load_snapshot_events,
    load_snapshot_manifest,
)
from orbit.ml.dataset import assemble_datasets
from orbit.ml.features import (
    FEATURE_NAMES,
    FEATURE_NAMES_PHASE10,
    PHASE10_FEATURE_SETS,
    PHASE10_FEATURE_SET_ORDER,
    build_feature_snapshot,
    build_feature_snapshot_phase10,
    build_phase10_all_feature_frame,
    build_phase10_feature_set_snapshot,
    phase10_set_identity,
)
from orbit.ml.labels import build_phase9_label_snapshot
from orbit.ml.metrics import hit_rate, mean_squared_error, oos_ic, rank_ic
from orbit.ml.models import predict_with_state, train_model
from orbit.ml.phase10_audit import audit_summary, run_phase10_audit, verify_dataset_unchanged
from orbit.ml.phase10_diagnostics import feature_quality_report, redundancy_report
from orbit.ml.phase10_plan import (
    PHASE10_MODEL_POINTS,
    phase10_experiment_id,
    phase10_plan,
)
from orbit.ml.phase10_registry import (
    register_phase10_experiment,
    run_registered_phase10_experiment,
)
from orbit.ml.phase10_report import (
    ARTIFACTS_ROOT,
    append_report_rows,
    write_diagnostics,
    write_markdown_report,
    write_plan,
    write_research_report,
)
from orbit.ml.signals import predictions_to_signals, run_backtest
from orbit.ml.snapshot_cache import (
    PHASE10_SNAPSHOT_CACHE_DIR,
    cache_label_snapshot,
    cache_phase10_snapshot,
    load_cached_label_snapshot,
    load_cached_phase10_snapshot,
)
from orbit.ml.splits import PHASE9_WINDOWS

TOP_K = 3

# Cache locations (module-level so tests can redirect them to hermetic dirs).
P9_CACHE_DIR = Path(__file__).resolve().parents[3] / "data" / "cache" / "phase9_snapshots"
P10_CACHE_DIR = PHASE10_SNAPSHOT_CACHE_DIR


def _log(msg: str, *, flush: bool = False) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _fmt_dur(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def _test_sessions(bars: pl.DataFrame, windows: dict | None = None) -> list:
    w = windows or PHASE9_WINDOWS
    return sorted(
        bars.filter(pl.col("trade_date") >= w["test_start"])
        .filter(pl.col("trade_date") <= w["test_end"])["trade_date"]
        .unique()
        .to_list()
    )


def _windows_dict(windows: dict | None = None) -> dict[str, Any]:
    w = windows or PHASE9_WINDOWS
    return {
        "train": (w["train_start"].isoformat(), w["train_end"].isoformat()),
        "val": (w["val_start"].isoformat(), w["val_end"].isoformat()),
        "test": (w["test_start"].isoformat(), w["test_end"].isoformat()),
    }


def load_and_verify_data() -> tuple[pl.DataFrame, pl.DataFrame | None, list, dict[str, Any]]:
    """Load DS-000004 + master + events; enforce the data-expansion guard."""
    bars = load_snapshot_bars()
    events = load_snapshot_events()
    instruments = load_instrument_master()
    manifest = load_snapshot_manifest()
    guard = verify_dataset_unchanged(bars, instruments, manifest)
    failed = [c for c in guard["checks"] if c["status"] == "FAIL"]
    if failed:
        raise RuntimeError(
            "Phase 10 data guard FAILED: " + "; ".join(f"{c['check']}: {c['evidence']}" for c in failed)
        )
    _log(f"data guard PASS: {guard['summary']}")
    return bars, events, instruments, manifest


def build_or_load_phase10_snapshots(
    bars: pl.DataFrame,
    events: pl.DataFrame | None,
    instruments: list,
    *,
    force_rebuild: bool = False,
) -> tuple[dict[str, Any], Any, str]:
    """Build/load all feature-set snapshots + the LAB-004 snapshot.

    Returns (snapshots_by_set, label_snapshot, phase9_fs001_digest). The
    FS-001 snapshot is the frozen Phase 9 artifact (loaded from the Phase 9
    cache when present, so its digest is preserved byte-for-byte).
    """
    from orbit.ml.snapshot_cache import (
        load_cached_feature_snapshot,
    )

    snapshots: dict[str, Any] = {}

    # FS-001: frozen Phase 9 artifact (never rebuilt/redefined here).
    fs001 = load_cached_feature_snapshot(P9_CACHE_DIR) if not force_rebuild else None
    if fs001 is None:
        fs001 = build_feature_snapshot(bars, data_refs=["DS-000004"])
    phase9_fs001_digest = fs001.content_digest
    snapshots["FS-001"] = fs001

    # LAB-004 over the FS-001 decision rows (identical to the Phase 9 label
    # snapshot; cached and digest-verified).
    ls = load_cached_label_snapshot(P10_CACHE_DIR) if not force_rebuild else None
    if ls is None:
        decisions = fs001.records.select("instrument_id", "decision_time")
        ls = build_phase9_label_snapshot(
            bars, events, instruments, decisions, data_refs=["DS-000004"]
        )
        cache_label_snapshot(ls, P10_CACHE_DIR)

    # FS-002..FS-013 from the FS-003 superset frame.
    all_frame = build_phase10_all_feature_frame(bars)
    for sid in PHASE10_FEATURE_SET_ORDER:
        if sid == "FS-001":
            continue
        cached = None if force_rebuild else load_cached_phase10_snapshot(sid, P10_CACHE_DIR)
        if cached is not None:
            snapshots[sid] = cached
            continue
        if sid == "FS-002":
            snap = build_feature_snapshot_phase10(bars, data_refs=["DS-000004"])
        else:
            snap = build_phase10_feature_set_snapshot(
                sid, all_frame, data_refs=["DS-000004"]
            )
        cache_phase10_snapshot(snap, P10_CACHE_DIR)
        snapshots[sid] = snap

    return snapshots, ls, phase9_fs001_digest


def _datasets_for_set(snapshots, ls, feature_set_id: str, windows: dict | None = None) -> dict[str, Any]:
    snap = snapshots[feature_set_id]
    names = _feature_names(snap)
    return assemble_datasets(snap, ls, windows=windows, feature_names=names)


def _feature_names(snap: Any) -> list[str]:
    """Map the snapshot's feature refs to column names (deterministic)."""
    from orbit.ml.features import (
        ALL_PHASE10_DEFINITIONS,
        FEATURE_DEFINITIONS,
    )

    by_id = {
        f["feature_id"]: f["name"]
        for f in FEATURE_DEFINITIONS + ALL_PHASE10_DEFINITIONS
    }
    return [by_id[fid] for fid in sorted(snap.feature_refs)]


def run_one_experiment(
    *,
    feature_set_id: str,
    model_point: dict[str, Any],
    bars: pl.DataFrame,
    events: pl.DataFrame | None,
    snapshots: dict[str, Any],
    ls: Any,
    plan_digest: str,
    windows: dict | None = None,
) -> dict:
    """Run one pre-registered (feature_set, model) experiment end to end.

    Register-before-run: the experiment is registered (REGISTERED state)
    before any training or evaluation happens.
    """
    w = windows or PHASE9_WINDOWS
    family = model_point["family"]
    params = model_point["params"]
    exp_id = phase10_experiment_id(feature_set_id, family, params)
    snap = snapshots[feature_set_id]
    identity = phase10_set_identity(feature_set_id)

    _log(f"experiment {exp_id}: {feature_set_id} x {family} {params}")
    datasets = _datasets_for_set(snapshots, ls, feature_set_id, windows=w)
    (Xtr, ytr, _, _), (Xva, _, ybin_va, meta_va), (Xte, yte, ybin_te, meta_te) = (
        datasets["train"],
        datasets["val"],
        datasets["test"],
    )

    service, spec = register_phase10_experiment(
        experiment_id=exp_id,
        hypothesis_id="H-001",
        feature_set_id=feature_set_id,
        feature_set_version=identity["feature_set_version"],
        family=family,
        params=params,
        seed=42,
        notes=(
            f"dataset {datasets['report']['train_rows']}/{datasets['report']['val_rows']}"
            f"/{datasets['report']['test_rows']} rows"
        ),
        plan_digest=plan_digest,
    )

    model, state = train_model(
        family, params, Xtr, ytr,
        feature_names=datasets["report"]["feature_names"],
        windows=_windows_dict(w),
    )
    pred_val = predict_with_state(model, state, Xva)
    pred_test = predict_with_state(model, state, Xte)

    calibrator = fit_platt(pred_val, ybin_va)
    cal_test = calibrator.apply(pred_test)
    calib_eval = expected_calibration_error(cal_test, ybin_te, n_bins=10)
    brier = brier_score(cal_test, ybin_te)

    test_frame = meta_te.with_columns(pl.Series("prediction", pred_test))
    ic = oos_ic(test_frame, "prediction")
    ric = rank_ic(test_frame, "prediction")
    mse = mean_squared_error(yte, pred_test)
    hit = hit_rate(test_frame, "prediction")

    strategy_ref = (
        f"phase10:{feature_set_id}:{family}:"
        f"{'-'.join(f'{k}{v}' for k, v in sorted(params.items()))}:topk{TOP_K}"
    )
    signals = predictions_to_signals(
        test_frame, family=family, params=params, top_k=TOP_K,
        strategy_ref=strategy_ref,
    )
    result = run_backtest(
        bars, signals,
        window_start=w["test_start"],
        window_end=w["test_end"],
        experiment_id=exp_id, hypothesis_id="H-001", events=events,
        feature_refs=[
            {"feature_id": fid, "feature_version": "v1"}
            for fid in identity["feature_refs"]
        ],
        model={"family": family, "hyperparameters": params},
        label_id="LAB-004", label_version="v1",
    )
    summary = result.summary()

    metrics = {
        "model_family": family,
        "hyperparameters": params,
        "seed": 42,
        "feature_set_id": feature_set_id,
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
        "calibration_slope": calibrator.slope,
        "calibration_intercept": calibrator.intercept,
        "run_id": result.run_id,
    }

    exp_dir = ARTIFACTS_ROOT / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    pred_file = exp_dir / "test_predictions.parquet"
    test_frame.with_columns(pl.Series("calibrated_prob", cal_test)).write_parquet(pred_file)
    metrics_file = exp_dir / "metrics.json"
    metrics_file.write_text(json.dumps(metrics, sort_keys=True, default=str), encoding="utf-8")

    result_summary = (
        f"Phase 10 {feature_set_id} {family} {params}: OOS IC {ic['value']:.4f}, "
        f"rank IC {ric['value']:.4f}, after-cost return {summary['total_return']:.4%}"
    )
    run_registered_phase10_experiment(
        service, exp_id, feature_set_id=feature_set_id, family=family, params=params,
        seed=42, artifacts_dir=ARTIFACTS_ROOT,
        result_summary=result_summary, result_metrics=metrics,
        artifact_files={
            "test_predictions_parquet": pred_file,
            "metrics_json": metrics_file,
        },
    )

    entry = PHASE10_FEATURE_SETS.get(feature_set_id, {})
    return {
        "experiment_id": exp_id,
        "feature_set_id": feature_set_id,
        "feature_set_version": identity["feature_set_version"],
        "set_role": "base" if feature_set_id == "FS-001" else entry.get("role"),
        "set_family": entry.get("family"),
        "n_features": len(identity["feature_refs"]),
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
        "label_id": "LAB-004",
        "dataset_snapshot_ids": "DS-000004",
        "cost_model_id": "CM-001",
        "evaluation_window": f"{w['test_start'].isoformat()}..{w['test_end'].isoformat()}",
        "train_rows": datasets["report"]["train_rows"],
        "val_rows": datasets["report"]["val_rows"],
        "test_rows": datasets["report"]["test_rows"],
        "feature_set_digest": snap.content_digest,
        "definitions_digest": (
            __import__("orbit.ml.features", fromlist=["_feature_definitions_digest"])._feature_definitions_digest(
                identity["feature_refs"]
            )
        ),
        "notes": f"top-{TOP_K} long, weight 1/{TOP_K}; calibration fit on validation; phase9_parent={model_point['phase9_parent']}",
    }


def run_diagnostics(
    snapshots: dict[str, Any], ls: Any, windows: dict | None = None
) -> dict[str, Any]:
    """Feature quality + redundancy diagnostics, computed on TRAIN only."""
    from orbit.ml.features import FEATURE_NAMES, FEATURE_NAMES_PHASE10

    result: dict[str, Any] = {
        "scope": "train split only (never test)",
        "feature_sets": {},
        "redundancy": {},
    }
    for sid in ("FS-001", "FS-002", "FS-003"):
        snap = snapshots[sid]
        names = list(FEATURE_NAMES) if sid == "FS-001" else (
            FEATURE_NAMES_PHASE10 if sid == "FS-002" else list(FEATURE_NAMES) + list(FEATURE_NAMES_PHASE10)
        )
        datasets = assemble_datasets(snap, ls, windows=windows, feature_names=names)
        train_meta = datasets["train"][3]
        frame = train_meta.join(
            snap.records.select(
                "instrument_id", "decision_session", *names
            ),
            on=["instrument_id", "decision_session"],
            how="inner",
        )
        frame = frame.with_columns(pl.lit("train").alias("split"))
        result["feature_sets"][sid] = {
            "quality": feature_quality_report(frame, names),
            "redundancy": redundancy_report(frame, names),
        }
        result["redundancy"][sid] = {
            "duplicates": result["feature_sets"][sid]["redundancy"]["duplicates"],
            "high_correlation_pairs": result["feature_sets"][sid]["redundancy"]["high_correlation_pairs"],
        }
    return result


def run_phase10_all(windows: dict | None = None) -> dict[str, Any]:
    """Execute the full Phase 10 pipeline. Returns the audit summary.

    `windows` defaults to the locked Phase 9 protocol; tests inject tighter
    windows for speed (the experiment id space and plan are unaffected).
    """
    w = windows or PHASE9_WINDOWS
    t_start = time.time()
    _log("Phase 10 feature engineering + ablation - full run")

    plan = phase10_plan()
    plan_digest = plan["plan_digest"]
    _log(f"locked plan digest: {plan_digest[:16]}... ({plan['experiment_count']} experiments)")

    bars, events, instruments, manifest = load_and_verify_data()
    snapshots, ls, phase9_fs001_digest = build_or_load_phase10_snapshots(bars, events, instruments)
    _log(
        f"snapshots: FS-001 {snapshots['FS-001'].records.height} rows (frozen), "
        f"FS-003 {snapshots['FS-003'].records.height} rows, "
        f"labels {ls.provenance()['row_count']} rows"
    )

    diagnostics = run_diagnostics(snapshots, ls, windows=w)
    write_diagnostics(diagnostics)

    rows = []
    _total = plan["experiment_count"]
    _done = 0
    _t_exp_start = time.time()
    for feature_set_id in PHASE10_FEATURE_SET_ORDER:
        for model_point in PHASE10_MODEL_POINTS:
            _t_exp_start = time.time()
            try:
                rows.append(
                    run_one_experiment(
                        feature_set_id=feature_set_id,
                        model_point=model_point,
                        bars=bars,
                        events=events,
                        snapshots=snapshots,
                        ls=ls,
                        plan_digest=plan_digest,
                        windows=w,
                    )
                )
            except Exception as exc:  # noqa: BLE001 - never hide a failed run
                _log(f"experiment FAILED: {feature_set_id} {model_point}: {type(exc).__name__}: {exc}")
                exp_id = phase10_experiment_id(
                    feature_set_id, model_point["family"], model_point["params"]
                )
                entry = PHASE10_FEATURE_SETS.get(feature_set_id, {})
                identity = phase10_set_identity(feature_set_id)
                rows.append(
                    {
                        "experiment_id": exp_id,
                        "feature_set_id": feature_set_id,
                        "feature_set_version": identity["feature_set_version"],
                        "set_role": "base" if feature_set_id == "FS-001" else entry.get("role"),
                        "set_family": entry.get("family"),
                        "n_features": len(identity["feature_refs"]),
                        "family": model_point["family"],
                        "params": json.dumps(model_point["params"], sort_keys=True),
                        "seed": 42,
                        "status": "failed",
                        "feature_set_digest": snapshots.get(feature_set_id).content_digest
                        if feature_set_id in snapshots else None,
                        "definitions_digest": (
                            __import__("orbit.ml.features", fromlist=["_feature_definitions_digest"])._feature_definitions_digest(
                                identity["feature_refs"]
                            )
                        ),
                        "notes": f"{type(exc).__name__}: {exc}",
                    }
                )
            _done += 1
            _t_exp = time.time() - _t_exp_start
            _t_elapsed = time.time() - t_start
            _avg = _t_elapsed / _done
            _remaining = max(1, _total - _done)
            _eta_s = _avg * _remaining
            _log(
                f"[progress] {_done}/{_total} ({_done / _total:.0%}) "
                f"last={feature_set_id}/{model_point['family']} {_t_exp:.1f}s "
                f"elapsed={_fmt_dur(_t_elapsed)} avg={_avg:.1f}s/exp "
                f"ETA={_fmt_dur(_eta_s)}",
                flush=True,
            )

    write_plan(plan)
    report_path = append_report_rows(rows)
    md_path = write_markdown_report()
    write_research_report(
        plan=plan,
        diagnostics=diagnostics,
        snapshots=snapshots,
        phase9_fs001_digest=phase9_fs001_digest,
    )
    _log(f"reports written: {report_path} / {md_path}")

    datasets_by_set = {}
    for sid in ("FS-001", "FS-003"):
        names = list(FEATURE_NAMES) if sid == "FS-001" else list(FEATURE_NAMES) + list(FEATURE_NAMES_PHASE10)
        datasets_by_set[sid] = assemble_datasets(snapshots[sid], ls, windows=w, feature_names=names)

    checks = run_phase10_audit(
        snapshots=snapshots,
        base_snapshot=snapshots.get("FS-001"),
        label_snapshot=ls,
        datasets_by_set=datasets_by_set,
        phase9_fs001_digest=phase9_fs001_digest,
        bars=bars,
        test_predictions=datasets_by_set["FS-001"]["test"][3],
        windows=w,
    )
    summary = audit_summary(checks)
    _log(f"audit: {summary}")
    _log(f"total wall time: {time.time() - t_start:.1f}s")
    return {"audit": summary, "plan_digest": plan_digest, "report_path": str(report_path)}


__all__ = [
    "TOP_K",
    "load_and_verify_data",
    "build_or_load_phase10_snapshots",
    "run_one_experiment",
    "run_diagnostics",
    "run_phase10_all",
]