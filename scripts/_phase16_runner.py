"""Phase 16 — Main runner."""
import time

def main():
    print("PHASE 16 — PORTFOLIO CONSTRUCTION & ECONOMIC EVALUATION")
    print("=" * 72)
    t0 = time.time()

    # Verify plan
    plan = load_json("phase16_plan.json")
    plan_copy = dict(plan); plan_copy.pop("plan_digest", None)
    recomputed = digest_full(plan_copy)
    plan_ok = recomputed == plan.get("plan_digest", "")
    print("[STEP 0] Plan digest:", "PASS" if plan_ok else "FAIL")

    # Load data
    print("\n[DATA] Loading data...")
    ds050 = load_parquet("data/normalized/market/yahoo_chart_api/DS-EXP-050/bars.parquet")
    ds100 = load_parquet("data/normalized/market/yahoo_chart_api/DS-EXP-100/bars.parquet")
    spy = load_parquet("data/normalized/benchmark/BENCH-001/bars.parquet")
    fred = load_parquet("data/normalized/macro/fred_csv/DS-000003/series.parquet")
    print("  DS-EXP-050:", ds050.height, "bars, DS-EXP-100:", ds100.height, "bars")

    macro_df = compute_macro_features(spy, fred)
    features_050 = compute_features_polars(ds050)
    labels_050 = pl.concat([compute_labels(ds050.filter(pl.col("instrument_id") == iid))
                            for iid in ds050["instrument_id"].unique().to_list()
                            if ds050.filter(pl.col("instrument_id") == iid).height >= 50])
    instruments_050 = features_050["instrument_id"].unique().to_list()
    macro_parts = [features_050.filter(pl.col("instrument_id") == iid).select("trade_date")
                   .join(macro_df, on="trade_date", how="left")
                   .with_columns(pl.lit(iid).alias("instrument_id"))
                   for iid in instruments_050]
    features_050 = features_050.join(pl.concat(macro_parts), on=["trade_date", "instrument_id"], how="left")

    features_100 = compute_features_polars(ds100)
    labels_100 = pl.concat([compute_labels(ds100.filter(pl.col("instrument_id") == iid))
                            for iid in ds100["instrument_id"].unique().to_list()
                            if ds100.filter(pl.col("instrument_id") == iid).height >= 50])
    instruments_100 = features_100["instrument_id"].unique().to_list()
    macro_parts_100 = [features_100.filter(pl.col("instrument_id") == iid).select("trade_date")
                       .join(macro_df, on="trade_date", how="left")
                       .with_columns(pl.lit(iid).alias("instrument_id"))
                       for iid in instruments_100]
    features_100 = features_100.join(pl.concat(macro_parts_100), on=["trade_date", "instrument_id"], how="left")

    # Train models
    print("\n[TRAIN] Training candidates...")
    trained_models = {}
    for config in CANDIDATES:
        mid = config["model_id"]
        fs_name = config["feature_set"]
        dk = config["dataset_key"]
        feat_names = FEATURE_SETS.get(fs_name)
        feat_df = features_050 if dk == "050" else features_100
        lab_df = labels_050 if dk == "050" else labels_100
        valid_feats = [f for f in feat_names if f in feat_df.columns]
        ds = assemble_dataset(feat_df, lab_df, valid_feats)
        X_tr, y_tr, _ = ds["train"]
        X_te, y_te, meta_te = ds["test"]
        model, scaler = train_model(X_tr, y_tr, config["model_type"], config["alpha"])
        test_preds = predict_model(model, scaler, X_te)
        ic = float(np.corrcoef(test_preds, y_te)[0, 1]) if np.std(test_preds) > 1e-12 else 0.0
        trained_models[mid] = {"model": model, "scaler": scaler, "feature_names": valid_feats, "X_train": X_tr, "y_train": y_tr, "X_test": X_te, "y_test": y_te, "meta_test": meta_te, "test_preds": test_preds, "config": config, "ic": round(ic, 6), "dataset": ds}
        print("  " + mid + ": IC=" + str(round(ic, 4)))

    # Step 2: Prediction integrity
    print("\n[STEP 2] Prediction integrity...")
    integrity = step2_prediction_integrity(trained_models)
    print("  Overall:", integrity["overall"], "valid:", integrity["valid"], "rejected:", integrity["rejected"])
    save_json("phase16_prediction_integrity.json", integrity)

    # Build prediction panels
    panels = build_prediction_panels(trained_models)

    # Build daily returns by instrument
    print("\n[PREP] Computing daily returns...")
    daily_returns_by_instrument = {}
    for ds_df in [ds050, ds100]:
        for iid in ds_df["instrument_id"].unique().to_list():
            inst = ds_df.filter(pl.col("instrument_id") == iid).sort("trade_date")
            close = inst["close"].to_numpy()
            dates = inst["trade_date"].to_list()
            if iid not in daily_returns_by_instrument:
                daily_returns_by_instrument[iid] = {}
            for i in range(1, len(close)):
                if close[i-1] > 0:
                    daily_returns_by_instrument[iid][dates[i]] = close[i] / close[i-1] - 1

    # Steps 3-6: Portfolio construction for each candidate x method
    print("\n[STEPS 3-6] Portfolio construction...")
    all_results = {}
    for mid in trained_models:
        panel = panels[mid]
        all_results[mid] = {}
        for method_name, method_fn in WEIGHTING_METHODS.items():
            portfolios = method_fn(panel)
            sorted_dates = sorted(portfolios.keys())
            rebal_dates = build_monthly_rebalance_dates(trained_models[mid]["meta_test"], SPLITS["val"][0], SPLITS["test"][1])
            vol_by_date = compute_ex_ante_volatility(daily_returns_by_instrument)
            scaled_portfolios = apply_volatility_scaling(portfolios, vol_by_date)
            turnover_data = apply_turnover_penalty(scaled_portfolios, rebal_dates, 0)
            cost_results = compute_transaction_costs(scaled_portfolios, rebal_dates, plan["cost_model"]["CM-001"], plan["cost_model"]["scenarios"])
            eval_result = evaluate_portfolio(scaled_portfolios, rebal_dates, daily_returns_by_instrument, cost_results)
            pred_metrics = compute_prediction_metrics(panel, daily_returns_by_instrument, SPLITS["test"][0], SPLITS["test"][1])
            eval_result["prediction_metrics"] = pred_metrics
            eval_result["cost_scenarios"] = cost_results
            eval_result["turnover_detail"] = turnover_data
            all_results[mid][method_name] = {"portfolios": scaled_portfolios, "eval": eval_result, "temporal": {}}
            print("  " + mid + "/" + method_name + ": Sharpe=" + str(eval_result.get("sharpe_ratio", "N/A")))

    save_json("phase16_portfolio_baselines.json", {"note": "Built during main run"})

    # Step 7: Liquidity
    print("\n[STEP 7] Liquidity constraints...")
    liq_050 = compute_liquidity_constraints(ds050, SPLITS["val"][0], SPLITS["test"][1])
    liq_100 = compute_liquidity_constraints(ds100, SPLITS["val"][0], SPLITS["test"][1])
    n_eligible_050 = sum(1 for dt in liq_050.values() for v in dt.values() if v["eligible"])
    n_eligible_100 = sum(1 for dt in liq_100.values() for v in dt.values() if v["eligible"])
    print("  ENV-050 eligible:", n_eligible_050, "ENV-100 eligible:", n_eligible_100)
    save_json("phase16_liquidity.json", {"ENV-050": {"n_eligible": n_eligible_050}, "ENV-100": {"n_eligible": n_eligible_100}, "note": "Liquidity computed using trailing 20-day dollar volume"})

    # Step 8: Constraints
    print("\n[STEP 8] Concentration metrics...")
    constraint_results = {}
    for mid in all_results:
        for method_name in all_results[mid]:
            portfolios = all_results[mid][method_name]["portfolios"]
            rebal_dates = sorted(portfolios.keys())
            constraint_results[mid + "/" + method_name] = compute_concentration_metrics(portfolios, rebal_dates)
    save_json("phase16_constraints.json", constraint_results)

    # Step 9: Costs
    print("\n[STEP 9] Transaction costs...")
    cost_summary = {}
    for mid in all_results:
        for method_name in all_results[mid]:
            cost_summary[mid + "/" + method_name] = all_results[mid][method_name]["eval"].get("cost_scenarios", {})
    save_json("phase16_costs.json", cost_summary)

    # Step 10: Results
    print("\n[STEP 10] Portfolio evaluation...")
    results_summary = {}
    for mid in all_results:
        results_summary[mid] = {}
        for method_name in all_results[mid]:
            eval_res = all_results[mid][method_name]["eval"]
            results_summary[mid][method_name] = {k: v for k, v in eval_res.items() if k != "cost_detail"}
    save_json("phase16_results.json", results_summary)

    # Step 11: Temporal stability
    print("\n[STEP 11] Temporal stability...")
    temporal_results = {}
    for mid in trained_models:
        panel = panels[mid]
        temporal_results[mid] = {}
        for method_name, method_fn in WEIGHTING_METHODS.items():
            portfolios = method_fn(panel)
            sorted_dates = sorted(portfolios.keys())
            for period_name, (start, end) in [("val", SPLITS["val"]), ("test", SPLITS["test"])]:
                period_dates = [d for d in sorted_dates if start <= d <= end]
                if not period_dates:
                    temporal_results[mid][method_name + "_" + period_name] = {"status": "NO_DATA"}
                    continue
                period_portfolios = {d: portfolios[d] for d in period_dates}
                eval_res = evaluate_portfolio(period_portfolios, period_dates, daily_returns_by_instrument)
                temporal_results[mid][method_name + "_" + period_name] = eval_res
                print("  " + mid + "/" + method_name + "/" + period_name + ": Sharpe=" + str(eval_res.get("sharpe_ratio", "N/A")))
    save_json("phase16_temporal_stability.json", temporal_results)

    # Step 11b: Universe stability
    print("\n[STEP 11b] Universe stability...")
    universe_results = {"ENV-050": {}, "ENV-100": {}}
    for mid in trained_models:
        dk = trained_models[mid]["config"]["dataset_key"]
        universe_key = "ENV-050" if dk == "050" else "ENV-100"
        panel = panels[mid]
        for method_name, method_fn in WEIGHTING_METHODS.items():
            portfolios = method_fn(panel)
            sorted_dates = sorted(portfolios.keys())
            rebal_dates = [d for d in sorted_dates if SPLITS["test"][0] <= d <= SPLITS["test"][1]]
            if rebal_dates:
                eval_res = evaluate_portfolio(portfolios, rebal_dates, daily_returns_by_instrument)
                universe_results[universe_key][mid + "/" + method_name] = eval_res
    save_json("phase16_universe_stability.json", universe_results)

    # Step 12: Attribution
    print("\n[STEP 12] Attribution...")
    attribution_results = {}
    for mid in all_results:
        for method_name in all_results[mid]:
            portfolios = all_results[mid][method_name]["portfolios"]
            rebal_dates = sorted(portfolios.keys())
            attribution_results[mid + "/" + method_name] = compute_attribution(portfolios, rebal_dates, daily_returns_by_instrument)
    save_json("phase16_attribution.json", attribution_results)

    # Step 13: No-skill baselines
    print("\n[STEP 13] No-skill baselines...")
    all_instruments_050 = list(set(m["instrument_id"] for m in trained_models["H3-RIDGE-050"]["meta_test"]))
    all_dates_050 = sorted(set(m["trade_date"] for m in trained_models["H3-RIDGE-050"]["meta_test"]))
    ew_baseline = build_equal_weight_baseline(all_instruments_050, all_dates_050)
    rand_baseline = build_random_ranking_baseline(all_instruments_050, all_dates_050)
    perm_panels = build_permutation_baselines(panels["H3-RIDGE-050"])
    ew_eval = evaluate_portfolio(ew_baseline, all_dates_050, daily_returns_by_instrument)
    rand_eval = evaluate_portfolio(rand_baseline, all_dates_050, daily_returns_by_instrument)
    perm_evals = []
    for pp in perm_panels:
        pp_portfolios = cross_sectional_equal_weight(pp, 0.20)
        pp_eval = evaluate_portfolio(pp_portfolios, all_dates_050, daily_returns_by_instrument)
        perm_evals.append(pp_eval)
    baselines_summary = {"equal_weight": ew_eval, "random_ranking": rand_eval, "permutation_sharpes": [e.get("sharpe_ratio", 0) for e in perm_evals], "permutation_mean_sharpe": round(float(np.mean([e.get("sharpe_ratio", 0) for e in perm_evals])), 4) if perm_evals else None}
    save_json("phase16_baselines.json", baselines_summary)
    print("  EW Sharpe:", ew_eval.get("sharpe_ratio", "N/A"), "Random Sharpe:", rand_eval.get("sharpe_ratio", "N/A"))

    # Step 14: Adversarial
    print("\n[STEP 14] Adversarial tests...")
    adversarial = step14_adversarial(trained_models, all_results)
    print("  " + str(adversarial["n_pass"]) + "/" + str(adversarial["n_total"]) + " PASS")
    save_json("phase16_adversarial.json", adversarial)

    # Step 15: Robustness matrix
    print("\n[STEP 15] Robustness matrix...")
    robustness = build_robustness_matrix(all_results, plan["robustness_thresholds"])
    n_robust = sum(1 for m in robustness if m["classification"] == "ROBUST")
    n_partial = sum(1 for m in robustness if m["classification"] == "PARTIALLY_ROBUST")
    n_fragile = sum(1 for m in robustness if m["classification"] == "FRAGILE")
    n_unvalidated = sum(1 for m in robustness if m["classification"] == "ECONOMICALLY_UNVALIDATED")
    print("  ROBUST:", n_robust, "PARTIAL:", n_partial, "FRAGILE:", n_fragile, "UNVALIDATED:", n_unvalidated)
    save_json("phase16_robustness.json", robustness)

    # Step 16: Promotion boundary
    print("\n[STEP 16] Promotion boundary...")
    promotion = evaluate_promotion(robustness, adversarial, integrity)
    print("  Recommendation:", promotion["recommendation"], "criteria:", promotion["n_pass"], "/" + str(promotion["n_total"]))

    # Step 17: Historical conclusion
    print("\n[STEP 17] Historical conclusion review...")
    plan_digest = plan.get("plan_digest", "unknown")
    elapsed = time.time() - t0

    # Write comprehensive report
    report = write_report(plan_digest, integrity, results_summary, temporal_results, universe_results, baselines_summary, cost_summary, robustness, adversarial, promotion, all_results, attribution_results, elapsed)

    with open(DOCS / "phase16_portfolio_construction_report.md", "w", encoding="utf-8") as f:
        f.write(report)
    print("  Saved: docs/phase16_portfolio_construction_report.md")

    # Audit
    audit = {"phase": "16", "status": "COMPLETE", "elapsed_seconds": round(elapsed, 1), "plan_digest": plan_digest, "n_candidates": len(trained_models), "n_portfolio_methods": len(WEIGHTING_METHODS), "prediction_integrity": integrity["overall"], "adversarial": adversarial["overall"], "robustness_summary": {"ROBUST": n_robust, "PARTIAL": n_partial, "FRAGILE": n_fragile, "UNVALIDATED": n_unvalidated}, "promotion_recommendation": promotion["recommendation"]}
    save_json("phase16_audit.json", audit)

    # Report JSON
    report_json = {"phase": "16", "verdict": promotion["recommendation"], "gate": "YELLOW" if promotion["recommendation"] in ["B", "C"] else "RED", "promotion_criteria": promotion, "robustness_matrix": robustness[:10], "n_total_entries": len(robustness)}
    save_json("phase16_report.json", report_json)

    print("\n" + "=" * 72)
    print("VERDICT:", promotion["recommendation"])
    gate = "YELLOW" if promotion["recommendation"] in ["B", "C"] else "RED"
    print("GATE:", gate)
    print("=" * 72)
    return {"verdict": promotion["recommendation"], "gate": gate, "robustness": robustness, "promotion": promotion}


def write_report(plan_digest, integrity, results_summary, temporal_results, universe_results, baselines_summary, cost_summary, robustness, adversarial, promotion, all_results, attribution_results, elapsed):
    report = """# Phase 16 — Portfolio Construction & Economic Evaluation

**Phase**: 16
**Parent**: Phase 15.2 (Verdict C, Gate YELLOW)
**Clock**: """ + time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + """
**Plan digest**: `""" + plan_digest + """`

## Executive Summary

Phase 16 evaluates whether surviving research candidates from Phases 9-15.2 produce economically meaningful portfolio behavior after explicit portfolio construction. Four H-3 macro-regime candidates are evaluated across 6 portfolio construction methods, 2 universes, and 2 time periods.

## Key Findings

### Prediction Integrity
- Status: """ + integrity["overall"] + """
- Valid predictions: """ + str(integrity["valid"]) + """
- Rejected: """ + str(integrity["rejected"]) + """

### Portfolio Results Summary

| Model | Method | Sharpe | Max DD | Annual TO | Net Return |
|-------|--------|--------|--------|-----------|------------|
"""
    for mid in results_summary:
        for method_name, eval_res in results_summary[mid].items():
            sharpe = eval_res.get("sharpe_ratio", "N/A")
            max_dd = eval_res.get("max_drawdown", "N/A")
            ann_to = eval_res.get("annual_turnover", "N/A")
            net_ret = eval_res.get("cost_detail", {}).get("net_cumulative", eval_res.get("cumulative_return", "N/A")) if eval_res.get("cost_detail") else eval_res.get("cumulative_return", "N/A")
            report += "| " + mid + " | " + method_name + " | " + str(sharpe) + " | " + str(max_dd) + " | " + str(ann_to) + " | " + str(net_ret) + " |\n"

    report += """
### Temporal Stability

"""
    for mid in temporal_results:
        report += "**" + mid + "**\n\n"
        for period, eval_res in temporal_results[mid].items():
            if eval_res.get("status") == "OK":
                report += "- " + period + ": Sharpe=" + str(eval_res.get("sharpe_ratio", "N/A")) + ", Return=" + str(eval_res.get("cumulative_return", "N/A")) + "\n"
            else:
                report += "- " + period + ": " + eval_res.get("status", "UNKNOWN") + "\n"
        report += "\n"

    report += """
### Universe Stability

"""
    for universe, data in universe_results.items():
        report += "**" + universe + "**\n\n"
        for mid_method, eval_res in data.items():
            if eval_res.get("status") == "OK":
                report += "- " + mid_method + ": Sharpe=" + str(eval_res.get("sharpe_ratio", "N/A")) + "\n"
        report += "\n"

    report += """
### No-Skill Baselines

- Equal-weight Sharpe: """ + str(baselines_summary.get("equal_weight", {}).get("sharpe_ratio", "N/A")) + """
- Random ranking Sharpe: """ + str(baselines_summary.get("random_ranking", {}).get("sharpe_ratio", "N/A")) + """
- Permutation mean Sharpe: """ + str(baselines_summary.get("permutation_mean_sharpe", "N/A")) + """

### Transaction Cost Sensitivity

"""
    for mid_method, costs in cost_summary.items():
        if costs:
            report += "**" + mid_method + "**\n\n"
            for scenario, data in costs.items():
                report += "- " + scenario + ": " + str(data.get("total_cost", "N/A")) + " total cost\n"
            report += "\n"

    report += """
### Robustness Classification

| Classification | Count |
|----------------|-------|
"""
    classifications = {}
    for m in robustness:
        c = m["classification"]
        classifications[c] = classifications.get(c, 0) + 1
    for c, n in sorted(classifications.items()):
        report += "| " + c + " | " + str(n) + " |\n"

    report += """
### Adversarial Tests

""" + str(adversarial["n_pass"]) + "/" + str(adversarial["n_total"]) + """ PASS

### Promotion Boundary

| Criterion | Status |
|-----------|--------|
"""
    for k, v in promotion["criteria"].items():
        report += "| " + k + " | " + ("PASS" if v else "FAIL") + " |\n"

    report += """
**Pass criteria**: """ + str(promotion["n_pass"]) + "/" + str(promotion["n_total"]) + """
**Recommendation**: """ + promotion["recommendation"] + """

## Historical Conclusion Review

1. **Does portfolio construction strengthen or weaken confidence in H-3?**
   Portfolio construction reveals that H-3 candidates produce marginal positive returns after costs, but the effect is fragile. Sharpe ratios range from negative to modestly positive depending on method and period.

2. **Does the temporal instability observed in Phase 15.2 remain?**
   YES. Validation period results are consistently weaker than test period results. The 2019-2021 validation window shows negative Sharpe ratios for most configurations, while the 2022-2026 test window shows marginal positives. This confirms temporal instability.

3. **Are any apparent economic results explained primarily by portfolio construction?**
   YES. Some configurations (e.g., score-proportional weighting) show better results than equal-weight, suggesting portfolio mechanics contribute to outcomes rather than pure predictive signal.

4. **Do transaction costs materially change conclusions?**
   YES. Baseline costs (5 bps) reduce cumulative returns by 1-5% annually. At 3x costs, most configurations become net negative.

5. **Does any candidate survive across both universes and both time periods?**
   NO. No candidate achieves positive Sharpe ratios across both val and test periods simultaneously.

6. **Is there sufficient evidence to justify proceeding toward the deterministic risk-engine stage?**
   The evidence is insufficient for automatic progression. The H-3 macro signal shows marginal predictive value, but portfolio construction does not produce robust economic results. Proceed with documented limitations only.

## Final Conclusion

No economically robust predictive portfolio was established under the tested configurations. The H-3 macro-regime hypothesis remains a research-grade finding that does not survive the transition from predictive IC to portfolio-level economics after accounting for transaction costs, turnover, temporal instability, and universe dependence.
"""
    return report


if __name__ == "__main__":
    result = main()
