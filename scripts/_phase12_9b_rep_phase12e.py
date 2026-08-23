"""Phase 12.9B - Phase 12E replication (LAB-006 + all combinations)."""
def rep_phase12e():
    print("\n" + "=" * 72)
    print("PHASE 12E CLEAN-RUN REPLICATION")
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
    lab004 = build_phase9_label_snapshot(bars, events, instruments, decisions, data_refs=["DS-EXP-050"])

    # Build LAB-006 vectorized
    print("  Building LAB-006 (excess)...")
    bench = pl.read_parquet(DATA / "normalized" / "benchmark" / "BENCH-001" / "bars.parquet")
    avail = lab004.records.filter(pl.col("outcome_status") == "available")
    avail_dated = avail.with_columns(pl.col("decision_time").dt.date().alias("entry_date"))
    inst_dates = bars.select("instrument_id", "trade_date").unique().sort(["instrument_id", "trade_date"])
    inst_dates = inst_dates.with_columns(pl.col("trade_date").shift(-5).over("instrument_id").alias("outcome_date"))
    aj = avail_dated.join(inst_dates, left_on=["instrument_id", "entry_date"],
                          right_on=["instrument_id", "trade_date"], how="left")
    aj = aj.join(bench.select(pl.col("trade_date").alias("entry_date"), pl.col("close").alias("bench_entry")),
                 on="entry_date", how="left")
    aj = aj.join(bench.select(pl.col("trade_date").alias("outcome_date"), pl.col("close").alias("bench_outcome")),
                 on="outcome_date", how="left")
    aj = aj.filter(pl.col("bench_entry").is_not_null() & pl.col("bench_outcome").is_not_null() &
                   pl.col("outcome_date").is_not_null() & (pl.col("bench_entry") > 0))
    aj = aj.with_columns(
        ((pl.col("bench_outcome") / pl.col("bench_entry")) - 1.0).alias("bench_ret"),
        (pl.col("outcome_value") - ((pl.col("bench_outcome") / pl.col("bench_entry")) - 1.0)).alias("excess_ret"),
    )
    lab006_records = aj.select(
        "instrument_id", "decision_time",
        pl.col("excess_ret").alias("outcome_value"),
    ).with_columns(
        pl.lit("available").alias("outcome_status"),
        pl.lit(None).alias("unavailable_reason"),
    )
    print(f"  LAB-006: {lab006_records.height} observations")

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

    pit_df = pit_asof_join(fs_snap.records, fund_df, FUNDAMENTAL_FIELDS)
    pivot_df = pivot_fundamental_features(pit_df, FUNDAMENTAL_FIELDS)
    merged = fs_snap.records.join(pivot_df, on=["instrument_id", "decision_session"], how="left")
    merged_fs = type(fs_snap)(feature_set_id=fs_snap.feature_set_id,
                               feature_set_version=fs_snap.feature_set_version,
                               feature_refs=fs_snap.feature_refs,
                               data_refs=fs_snap.data_refs, records=merged)

    feature_sets = {
        "FS-12B-A": FEATURE_NAMES,
        "FS-12B-B": FEATURE_NAMES + [c for c in pivot_df.columns if c.startswith("fund_valuation")],
        "FS-12B-C": FEATURE_NAMES + [c for c in pivot_df.columns if c.startswith("fund_profit")],
        "FS-12B-D": FEATURE_NAMES + [c for c in pivot_df.columns if c.startswith("fund_income")],
        "FS-12B-E": FEATURE_NAMES + [c for c in pivot_df.columns if c.startswith("fund_leverage")],
        "FS-12B-F": FEATURE_NAMES + [c for c in pivot_df.columns if c.startswith("fund_")],
    }

    models = [("ridge", {"alpha": 1.0}), ("lasso", {"alpha": 0.01}),
              ("random_forest", {"n_estimators": 50, "max_depth": 3}),
              ("xgboost", {"n_estimators": 50, "max_depth": 3, "learning_rate": 0.1})]

    results = []
    t0 = time.time()
    for fs_id, fnames in feature_sets.items():
        valid_fnames = [f for f in fnames if f in merged.columns]
        if len(valid_fnames) < 2:
            continue
        # Create a label snapshot from lab006_records
        from orbit.labels.snapshot import LabelSnapshot
        lab006_snap = LabelSnapshot(label_id="LAB-006", version="v1",
                                     contract_digest="clean-run", engine_version="clean-run",
                                     data_refs=["DS-EXP-050"], records=lab006_records)
        ds = assemble_datasets(merged_fs, lab006_snap, feature_names=valid_fnames)
        for model, params in models:
            eid = f"EXP-12.9B-ENV-12E-050-{fs_id}-LAB-006-{model}"
            try:
                m = run_experiment(model, params, valid_fnames, ds)
                if m:
                    results.append({"experiment_id": eid, "family": model,
                                    "feature_set_id": fs_id, "label_id": "LAB-006",
                                    "metrics": m, **m})
                    print(f"  [{len(results)}/24] {eid}: IC={m['oos_ic']:+.4f}")
            except Exception as e:
                print(f"  [{len(results)}/24] {eid}: ERROR {e}")

    elapsed = time.time() - t0
    clean_stats = experiment_stats(results)

    hist = load_json(BENCH / "phase12e_ENV-12E-050_results.json")
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

    output = {"phase": "12E", "env": "ENV-12E-050", "dataset": "DS-EXP-050",
              "n_experiments": len(results), "elapsed_s": round(elapsed, 1),
              "historical_stats": hist_stats, "clean_stats": clean_stats,
              "historical_significance": {"holm": hist_pvals["n_sig_holm"], "bh": hist_pvals["n_sig_bh"]},
              "clean_significance": {"holm": clean_pvals["n_sig_holm"], "bh": clean_pvals["n_sig_bh"]},
              "comparison": comparison, "status_counts": status}
    save_json("phase12_9b_phase12e_replication.json", output)
    print(f"\n  Phase 12E: {status}")
    return output
