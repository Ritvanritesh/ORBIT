"""Phase 12.9B - Phase 12A replication (FS-001, FS-101, FS-103)."""
def rep_phase12a():
    print("\n" + "=" * 72)
    print("PHASE 12A CLEAN-RUN REPLICATION")
    print("=" * 72)
    from orbit.ml.phase11_2_benchmark import load_dataset
    from orbit.ml.features import build_feature_snapshot
    from orbit.ml.labels import build_phase9_label_snapshot
    from orbit.ml.data import load_instrument_master
    from orbit.ml.dataset import assemble_datasets
    from orbit.ml.features import FEATURE_NAMES, PHASE10_FEATURE_SETS

    bars, events = load_dataset("DS-EXP-050")
    fs_snap = build_feature_snapshot(bars, data_refs=["DS-EXP-050"])
    instruments = load_instrument_master()
    decisions = fs_snap.records.select("instrument_id", "decision_time")
    lab = build_phase9_label_snapshot(bars, events, instruments, decisions, data_refs=["DS-EXP-050"])

    target_sets = ["FS-001", "FS-101", "FS-103"]
    models = [("ridge", {"alpha": 1.0}), ("lasso", {"alpha": 0.01}),
              ("random_forest", {"n_estimators": 50, "max_depth": 3}),
              ("xgboost", {"n_estimators": 50, "max_depth": 3, "learning_rate": 0.1})]

    results = []
    t0 = time.time()
    for fs_id in target_sets:
        # FS-001 is the base set defined in features.py
        if fs_id == "FS-001":
            fnames = FEATURE_NAMES
        else:
            fs_def = PHASE10_FEATURE_SETS.get(fs_id)
            if not fs_def:
                print(f"  {fs_id}: NOT FOUND in PHASE10_FEATURE_SETS, skipping")
                continue
            fid_list = fs_def.get("members", [])
            from orbit.ml.features import FEATURE_ID_BY_NAME, FEATURE_ID_BY_NAME_PHASE10
            all_id_map = {**FEATURE_ID_BY_NAME, **FEATURE_ID_BY_NAME_PHASE10}
            fnames = [n for fid, n in all_id_map.items() if fid in fid_list]
            if not fnames:
                print(f"  {fs_id}: no feature names resolved, skipping")
                continue

        ds = assemble_datasets(fs_snap, lab, feature_names=fnames)
        for model, params in models:
            eid = f"EXP-12.9B-ENV-12A-050-{fs_id}-LAB-004-{model}"
            try:
                m = run_experiment(model, params, fnames, ds)
                if m:
                    results.append({"experiment_id": eid, "family": model,
                                    "feature_set_id": fs_id, "label_id": "LAB-004",
                                    "metrics": m, **m})
                    print(f"  [{len(results)}/12] {eid}: IC={m['oos_ic']:+.4f}")
            except Exception as e:
                print(f"  [{len(results)}/12] {eid}: ERROR {e}")

    elapsed = time.time() - t0
    clean_stats = experiment_stats(results)

    hist = load_json(BENCH / "phase12a_ENV-12A-050_results.json")
    target_hist = [e for e in hist["results"] if e.get("feature_set_id") in target_sets and "error" not in e]
    hist_stats = experiment_stats(target_hist)
    hist_pvals = compute_pvals(target_hist)
    clean_pvals = compute_pvals(results)

    comparison = []
    h_map = {(e["feature_set_id"], e["family"]): e for e in target_hist}
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

    output = {"phase": "12A", "env": "ENV-12A-050", "dataset": "DS-EXP-050",
              "feature_sets_tested": target_sets, "n_experiments": len(results),
              "elapsed_s": round(elapsed, 1),
              "historical_stats": hist_stats, "clean_stats": clean_stats,
              "historical_significance": {"holm": hist_pvals["n_sig_holm"], "bh": hist_pvals["n_sig_bh"]},
              "clean_significance": {"holm": clean_pvals["n_sig_holm"], "bh": clean_pvals["n_sig_bh"]},
              "comparison": comparison, "status_counts": status}
    save_json("phase12_9b_phase12a_replication.json", output)
    print(f"\n  Phase 12A: {status}")
    return output
