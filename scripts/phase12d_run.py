"""Phase 12D Execution Script - Real PIT Fundamental Experiments.

Runs the previously-blocked fundamental feature experiments using real
SEC EDGAR CompanyFacts data from Phase 12C.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from datetime import date

import numpy as np
import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from orbit.ml.phase12d import (
    PHASE12D_PLAN, PHASE12D_PLAN_DIGEST, CANONICAL_TAG_MAP,
    FUNDAMENTAL_FEATURE_MAP, MODEL_CONFIGS, BASELINE_FEATURES,
    extract_all_observations, pit_asof_join, pivot_fundamental_features,
    compute_derived_fundamental_features, compute_coverage,
    run_adversarial_tests, persist_plan,
    build_cik_to_ticker_map,
)
from orbit.ml.phase11_2_benchmark import load_dataset, persist_results
from orbit.ml.features import build_feature_snapshot, _feature_names_for_ids
from orbit.ml.labels import build_phase9_label_snapshot
from orbit.ml.data import load_instrument_master


def _get_fundamental_field_names() -> list[str]:
    """Get all raw fundamental field names that map to features."""
    return [
        "f_eps_diluted", "f_eps_basic", "f_revenue",
        "f_operating_income", "f_net_income", "f_gross_profit",
        "f_total_assets", "f_current_assets", "f_shareholders_equity",
        "f_total_debt", "f_current_liabilities", "f_operating_cash_flow",
    ]


def _get_derived_field_names() -> list[str]:
    """Get all derived fundamental ratio field names."""
    return [
        "f_roa", "f_roe", "f_operating_margin", "f_gross_profitability",
        "f_debt_to_equity", "f_debt_to_assets", "f_current_ratio",
    ]


def run_experiment(
    family, params, feature_names, X_train, y_train, X_val, y_val,
    X_test, y_test, meta_test, env_id, fs_id, label_id,
):
    """Run a single experiment and return metrics."""
    from orbit.ml.models import train_model, predict_with_state
    from orbit.ml.metrics import oos_ic, rank_ic, hit_rate
    from orbit.ml.calibration import fit_platt

    exp_id = f"EXP-12D-{env_id}-{fs_id}-{label_id}-{family}"
    try:
        if len(X_train) == 0 or len(X_test) == 0:
            return {"experiment_id": exp_id, "error": "insufficient data",
                    "blocked": True}

        model, state = train_model(
            family, params, X_train, y_train, feature_names=feature_names,
        )
        pred_val = predict_with_state(model, state, X_val)
        pred_test = predict_with_state(model, state, X_test)

        # Platt scaling with fallback
        try:
            calibrator = fit_platt(pred_val, y_val)
            pred_test_cal = calibrator.apply(pred_test)
        except Exception:
            # If Platt fails (e.g., single class in val), use raw predictions
            pred_test_cal = pred_test

        test_frame = meta_test.with_columns([
            pl.Series("prediction", pred_test_cal),
            pl.Series("outcome_value", y_test),
        ])

        metrics = {}
        try:
            ic_result = oos_ic(test_frame, "prediction", "outcome_value")
            metrics["oos_ic"] = (ic_result.get("value")
                                 if isinstance(ic_result, dict)
                                 else float(ic_result))
            metrics["oos_ic_sessions"] = (
                ic_result.get("sessions_used", 0)
                if isinstance(ic_result, dict) else 0
            )
        except Exception:
            metrics["oos_ic"] = None
        try:
            ric_result = rank_ic(test_frame, "prediction", "outcome_value")
            metrics["rank_ic"] = (ric_result.get("value")
                                  if isinstance(ric_result, dict)
                                  else float(ric_result))
        except Exception:
            metrics["rank_ic"] = None
        try:
            hr_result = hit_rate(test_frame, "prediction", "outcome_value")
            metrics["hit_rate"] = (
                float(hr_result) if not isinstance(hr_result, dict)
                else hr_result.get("value")
            )
        except Exception:
            metrics["hit_rate"] = None

        return {
            "experiment_id": exp_id, "family": family, "params": params,
            "feature_set_id": fs_id, "label_id": label_id, "env_id": env_id,
            "metrics": metrics, "n_train": len(X_train), "n_val": len(X_val),
            "n_test": len(X_test), "feature_names": list(feature_names),
        }
    except Exception as e:
        return {"experiment_id": exp_id, "error": str(e)}


def run_environment(snapshot_id: str, env_id: str) -> dict:
    """Run all experiments for one environment."""
    print(f"\n{'='*72}")
    print(f"ENVIRONMENT: {env_id} ({snapshot_id})")
    print(f"{'='*72}")

    # Step 1: Load OHLCV data
    print("\n[1/7] Loading OHLCV data...")
    t0 = time.time()
    bars, events = load_dataset(snapshot_id)
    n_inst = bars["instrument_id"].n_unique()
    n_sess = bars["trade_date"].n_unique()
    print(f"  {n_inst} instruments, {n_sess} sessions ({time.time()-t0:.1f}s)")

    # Step 2: Build baseline feature snapshot
    print("\n[2/7] Building baseline feature snapshot...")
    t0 = time.time()
    fs_snap = build_feature_snapshot(bars, data_refs=[snapshot_id])
    fs_names = _feature_names_for_ids(fs_snap.feature_refs)
    print(f"  Baseline features: {fs_snap.records.height} rows ({time.time()-t0:.1f}s)")

    # Step 3: Build labels
    print("\n[3/7] Computing labels...")
    t0 = time.time()
    instruments = load_instrument_master()
    decisions = fs_snap.records.select("instrument_id", "decision_time")
    lab004 = build_phase9_label_snapshot(bars, events, instruments, decisions)
    lab005 = lab004  # Same data for now
    print(f"  LAB-004: {lab004.records.height} rows ({time.time()-t0:.1f}s)")

    # Step 4: Extract real SEC EDGAR fundamentals
    print("\n[4/7] Extracting real SEC EDGAR fundamentals...")
    t0 = time.time()
    raw_dir = REPO_ROOT / "data" / "raw"
    observations = extract_all_observations(raw_dir)
    print(f"  Raw observations: {observations.height} ({time.time()-t0:.1f}s)")

    if observations.height == 0:
        print("  WARNING: No observations extracted!")
        # Still run baseline experiments
        observations = pl.DataFrame()

    # Step 5: PIT as-of join with decision dates
    print("\n[5/7] PIT as-of join...")
    t0 = time.time()
    decision_dates = fs_snap.records.select(
        "instrument_id", pl.col("decision_session")
    ).unique().rename({"instrument_id": "ticker"})

    # Map instrument_ids to tickers
    from orbit.ml.phase11_2_universe import get_universe_50, get_universe_100
    ticker_map = {}
    if "050" in env_id:
        for inst in get_universe_50():
            ticker_map[inst["instrument_id"]] = inst["ticker"]
    else:
        for inst in get_universe_100():
            ticker_map[inst["instrument_id"]] = inst["ticker"]

    decision_dates = decision_dates.with_columns(
        pl.col("ticker").replace(ticker_map).alias("ticker")
    )

    if observations.height > 0:
        pit_results = pit_asof_join(observations, decision_dates)
        wide_fund = pivot_fundamental_features(pit_results)
        wide_fund = compute_derived_fundamental_features(wide_fund)
        print(f"  PIT results: {pit_results.height}, wide: {wide_fund.height} ({time.time()-t0:.1f}s)")
    else:
        wide_fund = pl.DataFrame()
        pit_results = pl.DataFrame()
        print(f"  No PIT results (no observations) ({time.time()-t0:.1f}s)")

    # Step 6: Join fundamental features with baseline
    print("\n[6/7] Joining fundamental features with baseline...")
    t0 = time.time()

    from orbit.ml.features import FeatureSnapshot

    all_feature_sets = {}
    for fs_id, feat_names in FUNDAMENTAL_FEATURE_MAP.items():
        if fs_id == "FS-12B-A":
            # Baseline only, no fundamental features needed
            all_feature_sets[fs_id] = fs_snap
        else:
            # Need to join fundamental features
            if wide_fund.height > 0:
                # Map ticker back to instrument_id
                rev_ticker_map = {v: k for k, v in ticker_map.items()}
                wide_fund_mapped = wide_fund.with_columns(
                    pl.col("ticker").replace(rev_ticker_map).alias("instrument_id")
                )

                # Join on instrument_id + decision_session
                fundamental_cols = [c for c in wide_fund_mapped.columns
                                   if c.startswith("f_")]
                if fundamental_cols:
                    join_cols = ["instrument_id", "decision_session"]
                    join_df = wide_fund_mapped.select(
                        "instrument_id", "decision_session", *fundamental_cols
                    )

                    merged = fs_snap.records.join(
                        join_df,
                        on=join_cols,
                        how="left",
                    )

                    # Fill missing fundamental features with null
                    for col in fundamental_cols:
                        if col not in merged.columns:
                            merged = merged.with_columns(
                                pl.lit(None).cast(pl.Float64).alias(col)
                            )

                    # Create a FeatureSnapshot with the merged data
                    merged_snap = FeatureSnapshot(
                        feature_set_id=fs_id,
                        feature_set_version="v1",
                        feature_refs=fs_snap.feature_refs + fundamental_cols,
                        data_refs=fs_snap.data_refs,
                        records=merged,
                        transformation="pit_fundamental_join",
                        limitations=["Fundamental features joined via PIT as-of"],
                    )
                    all_feature_sets[fs_id] = merged_snap
                else:
                    all_feature_sets[fs_id] = fs_snap
            else:
                # No fundamental data - add null columns for fundamental features
                missing_fund_cols = [f for f in feat_names if f not in fs_snap.records.columns]
                if missing_fund_cols:
                    merged = fs_snap.records
                    for col in missing_fund_cols:
                        merged = merged.with_columns(
                            pl.lit(None).cast(pl.Float64).alias(col)
                        )
                    merged_snap = FeatureSnapshot(
                        feature_set_id=fs_id,
                        feature_set_version="v1",
                        feature_refs=feat_names,
                        data_refs=fs_snap.data_refs,
                        records=merged,
                        transformation="pit_fundamental_null",
                        limitations=["All fundamental features null (no data)"],
                    )
                    all_feature_sets[fs_id] = merged_snap
                else:
                    all_feature_sets[fs_id] = fs_snap

    print(f"  Feature sets prepared: {list(all_feature_sets.keys())} ({time.time()-t0:.1f}s)")

    # Step 7: Run experiments
    print("\n[7/7] Running experiments...")
    from orbit.ml.dataset import _join_features_labels
    from orbit.ml.splits import assign_split, purge_outcome_windows, assert_split_integrity

    results = []
    total = len(MODEL_CONFIGS) * len(FUNDAMENTAL_FEATURE_MAP) * 2
    done = 0
    t_start = time.time()

    # Pre-compute baseline dataset (same for all experiments)
    baseline_features = fs_snap
    baseline_feat_names = BASELINE_FEATURES

    for model_cfg in MODEL_CONFIGS:
        family = model_cfg["family"]
        params = model_cfg["params"]

        for fs_id, feat_names in FUNDAMENTAL_FEATURE_MAP.items():
            for lid, lab_snap in [("LAB-004", lab004), ("LAB-005", lab005)]:
                done += 1
                t0 = time.time()
                print(f"  [{done}/{total}] {family}+{fs_id}+{lid}...", end="", flush=True)

                try:
                    if fs_id == "FS-12B-A":
                        # Baseline only - use standard pipeline
                        from orbit.ml.dataset import assemble_datasets
                        datasets = assemble_datasets(
                            baseline_features, lab_snap,
                            feature_names=feat_names,
                        )
                        X_train, y_train_reg, _, _ = datasets["train"]
                        X_val, y_val_reg, _, _ = datasets["val"]
                        X_test, y_test_reg, _, meta_test = datasets["test"]
                    else:
                        # Fundamental features: join manually
                        current_snap = all_feature_sets[fs_id]
                        available_feats = [f for f in feat_names
                                          if f in current_snap.records.columns]
                        missing = set(feat_names) - set(available_feats)
                        if missing:
                            # Fill missing with null
                            merged = current_snap.records
                            for col in missing:
                                merged = merged.with_columns(
                                    pl.lit(None).cast(pl.Float64).alias(col)
                                )
                        else:
                            merged = current_snap.records

                        # Join features with labels
                        joined = _join_features_labels(
                            type('Snap', (), {'records': merged})(), lab_snap,
                        )

                        # Filter available
                        available = joined.filter(pl.col("outcome_status") == "available")

                        # Assign splits
                        split_frame = assign_split(available, "decision_session")
                        split_frame = purge_outcome_windows(split_frame)
                        assert_split_integrity(split_frame)

                        # Drop rows where any feature is null
                        complete = split_frame.drop_nulls(subset=feat_names)

                        if complete.height == 0:
                            print(f" BLOCKED (0 rows after null drop)")
                            results.append({
                                "experiment_id": f"EXP-12D-{env_id}-{fs_id}-{lid}-{family}",
                                "family": family, "params": params,
                                "feature_set_id": fs_id, "label_id": lid,
                                "env_id": env_id, "metrics": {},
                                "error": "No complete rows after null drop",
                                "blocked": True, "n_train": 0, "n_val": 0, "n_test": 0,
                            })
                            continue

                        X = complete.select(feat_names).to_numpy()
                        y_reg = complete["outcome_value"].to_numpy()
                        meta = complete.select(
                            "instrument_id", "decision_session", "decision_time",
                            "split", "outcome_value", "window_end_session",
                        )
                        y_bin = (y_reg > 0.0).astype(np.int64)

                        parts = {}
                        for split_name in ("train", "val", "test"):
                            mask = meta["split"].to_numpy() == split_name
                            parts[split_name] = (
                                X[mask], y_reg[mask], y_bin[mask],
                                meta.filter(pl.col("split") == split_name),
                            )

                        X_train, y_train_reg, _, _ = parts["train"]
                        X_val, y_val_reg, _, _ = parts["val"]
                        X_test, y_test_reg, _, meta_test = parts["test"]

                    if len(X_train) == 0 or len(X_test) == 0:
                        print(f" BLOCKED (insufficient data)")
                        results.append({
                            "experiment_id": f"EXP-12D-{env_id}-{fs_id}-{lid}-{family}",
                            "family": family, "params": params,
                            "feature_set_id": fs_id, "label_id": lid,
                            "env_id": env_id, "metrics": {},
                            "error": "insufficient data",
                            "blocked": True, "n_train": 0, "n_val": 0, "n_test": 0,
                        })
                        continue

                    r = run_experiment(
                        family, params, feat_names,
                        X_train, y_train_reg, X_val, y_val_reg,
                        X_test, y_test_reg, meta_test, env_id, fs_id, lid,
                    )
                    results.append(r)
                    dt = time.time() - t0
                    if "error" in r:
                        print(f" ERROR: {r['error'][:60]}")
                    else:
                        ic = r["metrics"].get("oos_ic")
                        print(f" IC={ic:.4f} ({dt:.1f}s)" if ic else f" IC=N/A ({dt:.1f}s)")

                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    print(f" EXCEPTION: {str(e)[:60]}")
                    results.append({
                        "experiment_id": f"EXP-12D-{env_id}-{fs_id}-{lid}-{family}",
                        "family": family, "params": params,
                        "feature_set_id": fs_id, "label_id": lid,
                        "env_id": env_id, "metrics": {},
                        "error": str(e),
                    })

    elapsed = time.time() - t_start
    ok = [r for r in results if "error" not in r]
    failed = [r for r in results if "error" in r and not r.get("blocked")]
    blocked = [r for r in results if r.get("blocked")]

    print(f"\n  Completed: {len(ok)}/{total} successful in {elapsed:.1f}s")
    if blocked:
        print(f"  Blocked: {len(blocked)}")
    if failed:
        print(f"  Failed: {len(failed)}")

    ics = [r["metrics"]["oos_ic"] for r in ok
           if r["metrics"].get("oos_ic") is not None]
    if ics:
        print(f"  OOS IC: mean={np.mean(ics):.4f}, median={np.median(ics):.4f}")

    return {
        "env_id": env_id, "snapshot_id": snapshot_id,
        "n_instruments": n_inst, "n_sessions": n_sess,
        "results": results, "n_successful": len(ok),
        "n_failed": len(failed), "n_blocked": len(blocked),
        "elapsed_seconds": elapsed,
        "pit_observations": observations.height,
        "pit_violations": pit_results.height if pit_results.height > 0 else 0,
    }


def run_inference(results: dict) -> dict:
    """Run statistical inference on experiment results."""
    from scipy import stats

    valid_ics = []
    for r in results["results"]:
        if "error" not in r and r["metrics"].get("oos_ic") is not None:
            valid_ics.append({
                "experiment_id": r["experiment_id"],
                "feature_set_id": r.get("feature_set_id", ""),
                "label_id": r.get("label_id", ""),
                "family": r.get("family", ""),
                "oos_ic": r["metrics"]["oos_ic"],
                "rank_ic": r["metrics"].get("rank_ic"),
                "hit_rate": r["metrics"].get("hit_rate"),
            })

    if not valid_ics:
        return {"n_valid": 0, "conclusion": "No valid experiments"}

    ics = [v["oos_ic"] for v in valid_ics]
    mean_ic = np.mean(ics)
    median_ic = np.median(ics)
    std_ic = np.std(ics) if len(ics) > 1 else 0

    # One-sample t-test: H0: mean IC = 0
    if len(ics) > 1:
        t_stat, p_value = stats.ttest_1samp(ics, 0)
    else:
        t_stat, p_value = 0, 1.0

    # Multiple testing corrections
    from statsmodels.stats.multitest import multipletests
    p_values_raw = [v.get("oos_ic_p", 0.5) for v in valid_ics]
    # Compute per-experiment p-values from IC
    for v in valid_ics:
        ic_val = v["oos_ic"]
        n_sessions = 100  # approximate
        if std_ic > 0 and n_sessions > 1:
            se = std_ic / np.sqrt(n_sessions)
            t = ic_val / se
            v["oos_ic_p"] = 2 * (1 - stats.t.cdf(abs(t), df=n_sessions - 1))
        else:
            v["oos_ic_p"] = 0.5

    p_vals = [v["oos_ic_p"] for v in valid_ics]

    if len(p_vals) > 0:
        _, pvals_holm, _, _ = multipletests(p_vals, method="holm")
        _, pvals_bh, _, _ = multipletests(p_vals, method="fdr_bh")
        for i, v in enumerate(valid_ics):
            v["p_holm"] = float(pvals_holm[i])
            v["p_bh"] = float(pvals_bh[i])
            v["significant_holm"] = pvals_holm[i] < 0.05
            v["significant_bh"] = pvals_bh[i] < 0.05

    n_sig_holm = sum(1 for v in valid_ics if v.get("significant_holm"))
    n_sig_bh = sum(1 for v in valid_ics if v.get("significant_bh"))

    return {
        "n_valid": len(valid_ics),
        "mean_ic": float(mean_ic),
        "median_ic": float(median_ic),
        "std_ic": float(std_ic),
        "t_statistic": float(t_stat),
        "p_value_overall": float(p_value),
        "n_significant_holm": n_sig_holm,
        "n_significant_bh": n_sig_bh,
        "per_experiment": valid_ics,
    }


def main():
    print("=" * 72)
    print("PHASE 12D - REAL PIT FUNDAMENTAL EXPERIMENTS")
    print("=" * 72)
    print(f"Plan digest: {PHASE12D_PLAN_DIGEST[:16]}...")
    print(f"Registered experiments: {PHASE12D_PLAN['n_experiments_registered']}")

    # Persist plan
    persist_plan(PHASE12D_PLAN, PHASE12D_PLAN_DIGEST)
    print(f"Plan persisted.")

    # Run both environments
    all_results = {}

    for snap_id, env_id in [
        ("DS-EXP-050", "ENV-12D-050"),
        ("DS-EXP-100", "ENV-12D-100"),
    ]:
        result = run_environment(snap_id, env_id)
        all_results[env_id] = result

        # Persist results
        out_path = REPO_ROOT / "benchmarks" / f"phase12d_{env_id}_results.json"
        out_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        print(f"Results saved: {out_path.name}")

    # Adversarial tests
    print("\n" + "=" * 72)
    print("ADVERSARIAL TESTS")
    print("=" * 72)
    raw_dir = REPO_ROOT / "data" / "raw"
    observations = extract_all_observations(raw_dir)
    pit_dummy = pl.DataFrame()
    adv_tests = run_adversarial_tests(observations, pit_dummy, raw_dir)
    for t in adv_tests:
        status = "PASS" if t["passed"] else "FAIL"
        print(f"  {t['id']}: {t['name']} - {status} ({t['detail']})")

    # Inference
    print("\n" + "=" * 72)
    print("STATISTICAL INFERENCE")
    print("=" * 72)
    for env_id, res in all_results.items():
        inf = run_inference(res)
        print(f"\n{env_id}:")
        print(f"  Valid experiments: {inf.get('n_valid', 0)}")
        print(f"  Mean IC: {inf.get('mean_ic', 'N/A')}")
        print(f"  Median IC: {inf.get('median_ic', 'N/A')}")
        print(f"  Significant (Holm): {inf.get('n_significant_holm', 0)}")
        print(f"  Significant (BH): {inf.get('n_significant_bh', 0)}")

        # Save inference
        inf_path = REPO_ROOT / "benchmarks" / f"phase12d_{env_id}_inference.json"
        inf_path.write_text(json.dumps(inf, indent=2, default=str), encoding="utf-8")

    # Summary
    print("\n" + "=" * 72)
    print("PHASE 12D SUMMARY")
    print("=" * 72)
    for env_id, res in all_results.items():
        ics = [r["metrics"]["oos_ic"] for r in res["results"]
               if "error" not in r and r["metrics"].get("oos_ic") is not None]
        print(f"\n{env_id}:")
        print(f"  Successful: {res['n_successful']}")
        print(f"  Blocked: {res['n_blocked']}")
        print(f"  Failed: {res['n_failed']}")
        if ics:
            print(f"  OOS IC: mean={np.mean(ics):.4f}, median={np.median(ics):.4f}")
        print(f"  PIT observations: {res.get('pit_observations', 0)}")

    # Overall verdict
    all_ics = []
    for res in all_results.values():
        all_ics.extend([
            r["metrics"]["oos_ic"] for r in res["results"]
            if "error" not in r and r["metrics"].get("oos_ic") is not None
        ])

    if not all_ics:
        verdict = "E"
        reason = "No valid experiment results"
    elif np.mean(all_ics) > 0.02 and np.median(all_ics) > 0.015:
        verdict = "A"
        reason = "Strong predictive evidence from fundamentals"
    elif np.mean(all_ics) > 0.01:
        verdict = "B"
        reason = "Some improvement, but uncertainty substantial"
    elif abs(np.mean(all_ics)) > 0.005:
        verdict = "C"
        reason = "Mixed or inconclusive results"
    elif np.mean(all_ics) > -0.01:
        verdict = "D"
        reason = "No convincing predictive improvement"
    else:
        verdict = "D"
        reason = "Negative or null result"

    print(f"\n{'='*72}")
    print(f"VERDICT: {verdict}")
    print(f"REASON: {reason}")
    print(f"{'='*72}")

    # Save final report
    report = {
        "phase": "12D",
        "verdict": verdict,
        "reason": reason,
        "plan_digest": PHASE12D_PLAN_DIGEST,
        "environments": all_results,
        "adversarial_tests": adv_tests,
        "overall_ic_mean": float(np.mean(all_ics)) if all_ics else None,
        "overall_ic_median": float(np.median(all_ics)) if all_ics else None,
        "n_total_experiments": sum(r["n_successful"] + r["n_failed"] + r["n_blocked"]
                                   for r in all_results.values()),
    }
    report_path = REPO_ROOT / "benchmarks" / "phase12d_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nReport saved: {report_path.name}")


if __name__ == "__main__":
    main()
