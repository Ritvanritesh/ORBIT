"""Phase 12B benchmark execution - uses proven Phase 11.2 infrastructure."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from orbit.ml.phase11_2_benchmark import load_dataset, persist_results
from orbit.ml.phase12b_plan import build_phase12b_plan, persist_phase12b_plan
from orbit.ml.phase12b_fundamentals import load_sec_edgar_companyfacts, validate_pit_compliance
from orbit.ml.features import (
    build_feature_snapshot,
    _feature_names_for_ids,
    FeatureSnapshot,
)
from orbit.ml.labels import build_phase9_label_snapshot
from orbit.ml.data import load_instrument_master


BASELINE_FEATURES = [
    "ret_10", "ret_20", "ret_30", "sma_ratio_5_30",
    "sma_ratio_15_40", "vol_10", "vol_30", "log_dv_med_20",
]

FS_EXPANSIONS = {
    "FS-12B-A": BASELINE_FEATURES,
    "FS-12B-B": BASELINE_FEATURES + ["earnings_yield", "book_to_market", "sales_to_price"],
    "FS-12B-C": BASELINE_FEATURES + ["roa", "roe", "operating_margin", "gross_profitability"],
    "FS-12B-D": BASELINE_FEATURES + ["rev_growth_1y", "earn_growth_1y", "cash_growth_1y"],
    "FS-12B-E": BASELINE_FEATURES + ["de_to_equity", "de_to_assets", "current_ratio"],
    "FS-12B-F": BASELINE_FEATURES + [
        "earnings_yield", "book_to_market", "sales_to_price",
        "roa", "roe", "operating_margin", "gross_profitability",
        "rev_growth_1y", "earn_growth_1y", "cash_growth_1y",
        "de_to_equity", "de_to_assets", "current_ratio",
    ],
}

MODELS = [
    {"family": "ridge", "params": {"alpha": 1.0}},
    {"family": "lasso", "params": {"alpha": 0.001}},
    {"family": "random_forest", "params": {"max_depth": 3, "n_estimators": 200}},
    {"family": "xgboost", "params": {"learning_rate": 0.1, "max_depth": 3, "n_estimators": 200}},
]


def run_experiment(family, params, fs_snapshot, lab_snapshot, feature_names,
                   env_id, fs_id, label_id="LAB-004"):
    from orbit.ml.models import train_model, predict_with_state
    from orbit.ml.metrics import oos_ic, rank_ic, hit_rate
    from orbit.ml.calibration import fit_platt
    from orbit.ml.dataset import assemble_datasets

    exp_id = f"EXP-12B-{env_id}-{fs_id}-{label_id}-{family}"
    try:
        datasets = assemble_datasets(fs_snapshot, lab_snapshot, feature_names=feature_names)
        X_train, y_train_reg, y_train_bin, _ = datasets["train"]
        X_val, y_val_reg, y_val_bin, _ = datasets["val"]
        X_test, y_test_reg, y_test_bin, meta_test = datasets["test"]

        if len(X_train) == 0 or len(X_test) == 0:
            return {"experiment_id": exp_id, "error": "insufficient data"}

        model, state = train_model(family, params, X_train, y_train_reg,
                                   feature_names=feature_names)
        pred_val = predict_with_state(model, state, X_val)
        pred_test = predict_with_state(model, state, X_test)

        calibrator = fit_platt(pred_val, y_val_bin)
        pred_test_cal = calibrator.apply(pred_test)

        test_frame = meta_test.with_columns([
            pl.Series("prediction", pred_test_cal),
            pl.Series("outcome_value", y_test_reg),
        ])

        metrics = {}
        try:
            ic_result = oos_ic(test_frame, "prediction", "outcome_value")
            metrics["oos_ic"] = ic_result.get("value") if isinstance(ic_result, dict) else float(ic_result)
            metrics["oos_ic_sessions"] = ic_result.get("sessions_used", 0) if isinstance(ic_result, dict) else 0
        except Exception:
            metrics["oos_ic"] = None
            metrics["oos_ic_sessions"] = 0
        try:
            ric_result = rank_ic(test_frame, "prediction", "outcome_value")
            metrics["rank_ic"] = ric_result.get("value") if isinstance(ric_result, dict) else float(ric_result)
        except Exception:
            metrics["rank_ic"] = None
        try:
            hr_result = hit_rate(test_frame, "prediction", "outcome_value")
            metrics["hit_rate"] = float(hr_result) if not isinstance(hr_result, dict) else hr_result.get("value")
        except Exception:
            metrics["hit_rate"] = None

        return {
            "experiment_id": exp_id, "family": family, "params": params,
            "feature_set_id": fs_id, "label_id": label_id, "env_id": env_id,
            "metrics": metrics, "n_train": len(X_train), "n_val": len(X_val),
            "n_test": len(X_test),
        }
    except Exception as e:
        return {"experiment_id": exp_id, "error": str(e)}


def run_benchmark(snapshot_id, env_id):
    print(f"\n{'='*72}")
    print(f"BENCHMARK: {env_id} ({snapshot_id})")
    print(f"{'='*72}")

    print("\n[1/5] Loading data...")
    bars, events = load_dataset(snapshot_id)
    n_inst = bars["instrument_id"].n_unique()
    n_sess = bars["trade_date"].n_unique()
    print(f"  {n_inst} instruments, {n_sess} sessions")

    print("\n[2/5] Loading fundamental data...")
    fundamental_df = load_sec_edgar_companyfacts(snapshot_id)
    print(f"  {fundamental_df.height} fundamental records")
    first_trade_date = bars["trade_date"].unique().to_list()[0]
    pit_result = validate_pit_compliance(fundamental_df, first_trade_date)
    pit_compliant = pit_result.get("compliant", False)
    print(f"  PIT compliant: {pit_compliant}")
    if not pit_compliant:
        print(f"  Reason: {pit_result.get('reason', 'unknown')}")

    print("\n[3/5] Computing baseline features...")
    t0 = time.time()
    fs_snap = build_feature_snapshot(bars, data_refs=[snapshot_id])
    fs_names = _feature_names_for_ids(fs_snap.feature_refs)
    print(f"  FS-12B-A (baseline): {fs_snap.records.height} rows ({time.time()-t0:.1f}s)")

    print("\n[4/5] Computing labels...")
    t0 = time.time()
    instruments = load_instrument_master()
    decisions = fs_snap.records.select("instrument_id", "decision_time")
    lab004 = build_phase9_label_snapshot(bars, events, instruments, decisions)
    lab005 = lab004
    print(f"  LAB-004: {lab004.records.height} rows")
    print(f"  LAB-005: same as LAB-004 (excess not yet separated)")
    print(f"  ({time.time()-t0:.1f}s)")

    print("\n[5/5] Running experiments...")
    results = []
    blocked_sets = ["FS-12B-B", "FS-12B-C", "FS-12B-D", "FS-12B-E", "FS-12B-F"]

    if not pit_compliant:
        print(f"  BLOCKING fundamental feature sets: PIT non-compliant")

    active_sets = [("FS-12B-A", fs_snap, fs_names)]
    total = len(MODELS) * 1 * 2
    done = 0
    t_start = time.time()

    for model_cfg in MODELS:
        family = model_cfg["family"]
        params = model_cfg["params"]
        for fs_id, fs_s, fs_n in active_sets:
            for lid, ls in [("LAB-004", lab004), ("LAB-005", lab005)]:
                done += 1
                t0 = time.time()
                print(f"  [{done}/{total}] {family}+{fs_id}+{lid}...", end="", flush=True)
                r = run_experiment(family, params, fs_s, ls, fs_n, env_id, fs_id, lid)
                results.append(r)
                dt = time.time() - t0
                if "error" in r:
                    print(f" ERROR: {r['error'][:50]}")
                else:
                    ic = r["metrics"].get("oos_ic")
                    print(f" IC={ic:.4f} ({dt:.1f}s)" if ic else f" IC=N/A ({dt:.1f}s)")

        for fs_id in blocked_sets:
            for lid in ["LAB-004", "LAB-005"]:
                done += 1
                eid = f"EXP-12B-{env_id}-{fs_id}-{lid}-{family}"
                results.append({
                    "experiment_id": eid, "family": family, "params": params,
                    "feature_set_id": fs_id, "label_id": lid, "env_id": env_id,
                    "metrics": {},
                    "error": f"PIT non-compliant: {pit_result.get('reason', 'synthetic fundamentals with future filing dates')}",
                    "n_train": 0, "n_val": 0, "n_test": 0, "blocked": True,
                })

    elapsed = time.time() - t_start
    ok = [r for r in results if "error" not in r]
    failed = [r for r in results if "error" in r and not r.get("blocked")]
    blocked = [r for r in results if r.get("blocked")]

    print(f"\n  Completed: {len(ok)}/{total} successful in {elapsed:.1f}s")
    if blocked:
        print(f"  Blocked (PIT): {len(blocked)}")
    if failed:
        print(f"  Failed: {len(failed)}")

    ics = [r["metrics"]["oos_ic"] for r in ok if r["metrics"].get("oos_ic") is not None]
    if ics:
        print(f"  OOS IC: mean={np.mean(ics):.4f}, median={np.median(ics):.4f}")

    return {
        "env_id": env_id, "snapshot_id": snapshot_id,
        "n_instruments": n_inst, "n_sessions": n_sess,
        "results": results, "n_successful": len(ok),
        "n_failed": len(failed), "n_blocked": len(blocked),
        "blocked_feature_sets": blocked_sets, "pit_compliant": pit_compliant,
        "pit_result": pit_result, "elapsed_seconds": elapsed,
    }


def main():
    print("=" * 72)
    print("PHASE 12B - BENCHMARK EXECUTION")
    print("=" * 72)

    plan = build_phase12b_plan()
    persist_phase12b_plan(plan)
    print(f"Plan: {plan['n_experiments']} experiments, digest {plan['plan_digest'][:16]}...")

    results = {}
    r1 = run_benchmark("DS-EXP-050", "ENV-12B-050")
    persist_results(r1)
    out1 = REPO_ROOT / "benchmarks" / "phase12b_ENV-12B-050_results.json"
    out1.write_text(json.dumps(r1, indent=2, default=str), encoding="utf-8")
    results["ENV-12B-050"] = r1

    r2 = run_benchmark("DS-EXP-100", "ENV-12B-100")
    persist_results(r2)
    out2 = REPO_ROOT / "benchmarks" / "phase12b_ENV-12B-100_results.json"
    out2.write_text(json.dumps(r2, indent=2, default=str), encoding="utf-8")
    results["ENV-12B-100"] = r2

    print("\n" + "=" * 72)
    print("PHASE 12B SUMMARY")
    print("=" * 72)
    for eid, res in results.items():
        ics = [r["metrics"]["oos_ic"] for r in res["results"]
               if "error" not in r and r["metrics"].get("oos_ic") is not None]
        print(f"\n{eid}: {res['n_successful']}/{res['n_successful']+res['n_failed']} successful")
        if ics:
            print(f"  OOS IC: mean={np.mean(ics):.4f}, median={np.median(ics):.4f}")
        print(f"  PIT compliant: {res['pit_compliant']}")
        print(f"  Blocked (fundamental): {res['n_blocked']}")


if __name__ == "__main__":
    main()
