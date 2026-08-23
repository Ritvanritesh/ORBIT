"""Phase 12.9B - Phase 12D replication (LAB-004 + fundamental features)."""
def rep_phase12d():
    print("\n" + "=" * 72)
    print("PHASE 12D CLEAN-RUN REPLICATION")
    print("=" * 72)
    from orbit.ml.phase11_2_benchmark import load_dataset
    from orbit.ml.features import build_feature_snapshot
    from orbit.ml.labels import build_phase9_label_snapshot
    from orbit.ml.data import load_instrument_master
    from orbit.ml.dataset import assemble_datasets
    from orbit.ml.features import FEATURE_NAMES
    from orbit.ml.phase12d import pit_asof_join, pivot_fundamental_features, FUNDAMENTAL_FIELDS

    bars, events = load_dataset("DS-EXP-050")
    fs_snap = build_feature_snapshot(bars, data_refs=["DS-EXP-050"])
    instruments = load_instrument_master()
    decisions = fs_snap.records.select("instrument_id", "decision_time")
    lab = build_phase9_label_snapshot(bars, events, instruments, decisions, data_refs=["DS-EXP-050"])

    # Load fundamentals
    fund_path = DATA / "normalized" / "fundamentals" / "sec_edgar_companyfacts" / "DS-EXP-050"
    fund_rows = []
    for f in sorted(fund_path.glob("INS-*.json")):
        with open(f, encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, list):
            fund_rows.extend(data)
        elif isinstance(data, dict) and "observations" in data:
            fund_rows.extend(data["observations"])
    fund_df = pl.DataFrame(fund_rows)
    print(f"  Fundamentals: {fund_df.height} rows")

    # PIT join + pivot
    pit_df = pit_asof_join(fs_snap.records, fund_df, FUNDAMENTAL_FIELDS)
    pivot_df = pivot_fundamental_features(pit_df, FUNDAMENTAL_FIELDS)

    feature_sets = {
        "FS-12B-A": FEATURE_NAMES,
        "FS-12B-B": FEATURE_NAMES + [c for c in pivot_df.columns if c.startswith("fund_valuation")],
        "FS-12B-C": FEATURE_NAMES + [c for c in pivot_df.columns if c.startswith("fund_profit")],
        "FS-12B-D": FEATURE_NAMES + [c for c in pivot_df.columns if c.startswith("fund_income")],
        "FS-12B-E": FEATURE_NAMES + [c for c in pivot_df.columns if c.startswith("fund_leverage")],
        "FS-12B-F": FEATURE_NAMES + [c for c in pivot_df.columns if c.startswith("fund_")],
    }

    # Merge features with pivot
    merged = fs_snap.records.join(pivot_df, on=["instrument_id", "decision_session"], how="left")
    merged_fs = type(fs_snap)(feature_set_id=fs_snap.feature_set_id,
                               feature_set_version=fs_snap.feature_set_version,
                               feature_refs=fs_snap.feature_refs,
                               data_refs=fs_snap.data_refs, records=merged)

    models = [("ridge", {"alpha": 1.0}), ("lasso", {"alpha": 0.01}),
              ("random_forest", {"n_estimators": 50, "max_depth": 3}),
              ("xgboost", {"n_estimators": 50, "max_depth": 3, "learning_rate": 0.1})]

    results = []
    t0 = time.time()
    for fs_id, fnames in feature_sets.items():
        valid_fnames = [f for f in fnames if f in merged.columns]
        if len(valid_fnames) < 2:
            continue
        ds = assemble_datasets(merged_fs, lab, feature_names=valid_fnames)
        for model, params in models:
            eid = f"EXP-12.9B-ENV-12D-050-{fs_id}-LAB-004-{model}"
            try:
                m = run_experiment(model, params, valid_fnames, ds)
                if m:
                    results.append({"experiment_id": eid, "family": model,
                                    "feature_set_id": fs_id, "label_id": "LAB-004",
                                    "metrics": m, **m})
                    print(f"  [{len(results)}/24] {eid}: IC={m['oos_ic']:+.4f}")
            except Exception as e:
                print(f"  [{len(results)}/24] {eid}: ERROR {e}")

    elapsed = time.time() - t0
    clean_stats = experiment_stats(results)

    hist = load_json(BENCH / "phase12d_ENV-12D-050_results.json")
    hist_stats = experiment_stats(hist["results"])
    hist_pvals = compute_pvals(hist["results"])
    clean_pvals = compute_pvals(results)

    comparison = []
    h_map = {(e["feature_set_id"], e["family"]): e for e in hist["results"]}
    for c in results:
        key = (c["feature_set_id"], c["family"])
        h = h_map.get(key)
        h_ic = h["metrics"].get("oos_ic") if h else None
        c_ic = c["metrics"].get("oos_ic")
        comp = classify(h_ic, c_ic)
        comparison.append({"eid": c["experiment_id"], "hist_ic": h_ic,
                           "clean_ic": c_ic, "classification": comp})

    status = {}
    for c in comparison:
        status[c["classification"]] = status.get(c["classification"], 0) + 1

    output = {"phase": "12D", "env": "ENV-12D-050", "dataset": "DS-EXP-050",
              "n_experiments": len(results), "elapsed_s": round(elapsed, 1),
              "historical_stats": hist_stats, "clean_stats": clean_stats,
              "historical_significance": {"holm": hist_pvals["n_sig_holm"], "bh": hist_pvals["n_sig_bh"]},
              "clean_significance": {"holm": clean_pvals["n_sig_holm"], "bh": clean_pvals["n_sig_bh"]},
              "comparison": comparison, "status_counts": status}
    save_json("phase12_9b_phase12d_replication.json", output)
    print(f"\n  Phase 12D: {status}")
    return output
