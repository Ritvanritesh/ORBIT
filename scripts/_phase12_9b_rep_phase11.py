"""Phase 12.9B - Phase 11.2 baseline replication (ENV-1 through ENV-4)."""
def rep_phase11():
    print("\n" + "=" * 72)
    print("PHASE 11.2 CLEAN-RUN REPLICATION (ENV-1 through ENV-4)")
    print("=" * 72)
    from orbit.ml.phase11_2_benchmark import load_dataset
    from orbit.ml.features import build_feature_snapshot
    from orbit.ml.labels import build_phase9_label_snapshot
    from orbit.ml.data import load_instrument_master
    from orbit.ml.dataset import assemble_datasets
    from orbit.ml.features import FEATURE_NAMES

    bars, events = load_dataset("DS-000004")
    fs = build_feature_snapshot(bars, data_refs=["DS-000004"])
    instruments = load_instrument_master()
    decisions = fs.records.select("instrument_id", "decision_time")
    lab = build_phase9_label_snapshot(bars, events, instruments, decisions, data_refs=["DS-000004"])
    ds = assemble_datasets(fs, lab, feature_names=FEATURE_NAMES)

    models = [("ridge", {"alpha": 1.0}), ("lasso", {"alpha": 0.01}),
              ("random_forest", {"n_estimators": 50, "max_depth": 3}),
              ("xgboost", {"n_estimators": 50, "max_depth": 3, "learning_rate": 0.1})]
    feat_sets = [("FS-001", FEATURE_NAMES)]

    results = []
    t0 = time.time()
    for fi, (fs_id, fnames) in enumerate(feat_sets):
        for mi, (model, params) in enumerate(models):
            eid = f"EXP-12.9B-ENV-1-{fs_id}-LAB-004-{model}"
            try:
                m = run_experiment(model, params, fnames, ds)
                if m:
                    results.append({"experiment_id": eid, "family": model,
                                    "feature_set_id": fs_id, "label_id": "LAB-004",
                                    "metrics": m, **m})
                    print(f"  [{len(results)}/4] {eid}: IC={m['oos_ic']:+.4f}")
            except Exception as e:
                print(f"  [{len(results)}/4] {eid}: ERROR {e}")

    elapsed = time.time() - t0
    clean_stats = experiment_stats(results)

    hist = load_json(BENCH / "phase11_2_ENV-1_results.json")
    hist_stats = experiment_stats(hist["results"])
    hist_pvals = compute_pvals(hist["results"])
    clean_pvals = compute_pvals(results)

    comparison = []
    for h, c in zip(hist["results"], results):
        h_ic = h["metrics"].get("oos_ic")
        c_ic = c["metrics"].get("oos_ic") if c.get("metrics") else None
        comp = classify(h_ic, c_ic)
        comparison.append({"eid": h["experiment_id"], "hist_ic": h_ic,
                           "clean_ic": c_ic, "classification": comp})

    status = {"EXACT_MATCH": 0, "NUMERICALLY_EQUIVALENT": 0, "MINOR_DRIFT": 0,
              "MATERIAL_DRIFT": 0, "FAILED_TO_REPRODUCE": 0}
    for c in comparison:
        status[c["classification"]] = status.get(c["classification"], 0) + 1

    output = {"phase": "11.2", "env": "ENV-1", "dataset": "DS-000004",
              "n_experiments": len(results), "elapsed_s": round(elapsed, 1),
              "historical_stats": hist_stats, "clean_stats": clean_stats,
              "historical_significance": {"holm": hist_pvals["n_sig_holm"], "bh": hist_pvals["n_sig_bh"]},
              "clean_significance": {"holm": clean_pvals["n_sig_holm"], "bh": clean_pvals["n_sig_bh"]},
              "comparison": comparison, "status_counts": status}
    save_json("phase12_9b_phase11_replication.json", output)
    print(f"\n  Phase 11.2: {status}")
    return output
