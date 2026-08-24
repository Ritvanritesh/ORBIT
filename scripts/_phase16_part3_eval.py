"""Phase 16 — Steps 7-12: Constraints, costs, evaluation, attribution."""

# =====================================================================
# STEP 7: LIQUIDITY-AWARE POSITION SIZING
# =====================================================================

def compute_liquidity_constraints(daily_data, eval_start, eval_end, min_dollar_vol=5_000_000, max_participation=0.05):
    """Compute per-instrument liquidity constraints."""
    liquidity = {}
    dates = sorted(daily_data["trade_date"].unique().to_list())
    for iid in daily_data["instrument_id"].unique().to_list():
        inst_data = daily_data.filter(pl.col("instrument_id") == iid).sort("trade_date")
        close = inst_data["close"].to_numpy()
        volume = inst_data["volume"].to_numpy()
        inst_dates = inst_data["trade_date"].to_list()
        dollar_vol = close * volume
        for i, dt in enumerate(inst_dates):
            if eval_start <= dt <= eval_end and i >= 20:
                trailing_dv = float(np.mean(dollar_vol[i-20:i]))
                max_position_dollars = trailing_dv * max_participation
                eligible = trailing_dv >= min_dollar_vol
                if dt not in liquidity:
                    liquidity[dt] = {}
                liquidity[dt][iid] = {"trailing_dollar_volume_20d": round(trailing_dv, 2), "max_position_dollars": round(max_position_dollars, 2), "eligible": eligible}
    return liquidity

def apply_liquidity_constraints(portfolios, liquidity, max_participation=0.05):
    """Cap positions based on liquidity."""
    constrained = {}
    for dt, weights in portfolios.items():
        if dt not in liquidity:
            constrained[dt] = weights
            continue
        liq = liquidity[dt]
        capped = {}
        total_weight = 0.0
        for iid, w in weights.items():
            if iid in liq and liq[iid]["eligible"]:
                max_w = liq[iid]["max_position_dollars"]
                capped[iid] = min(w, max_w) if max_w > 0 else w
            else:
                capped[iid] = 0.0
            total_weight += capped[iid]
        if total_weight > 1e-12:
            capped = {iid: w / total_weight for iid, w in capped.items()}
        constrained[dt] = capped
    return constrained

# =====================================================================
# STEP 8: CONCENTRATION AND EXPOSURE CONSTRAINTS
# =====================================================================

def compute_concentration_metrics(portfolios, dates):
    """Compute concentration metrics for each rebalance date."""
    results = []
    for dt in dates:
        weights = portfolios.get(dt, {})
        if not weights:
            continue
        w_arr = np.array(list(weights.values()))
        sorted_w = np.sort(w_arr)[::-1]
        top_5_weight = float(sorted_w[:5].sum()) if len(sorted_w) >= 5 else float(sorted_w.sum())
        top_10_weight = float(sorted_w[:10].sum()) if len(sorted_w) >= 10 else float(sorted_w.sum())
        herfindahl = float(np.sum(w_arr ** 2))
        max_weight = float(sorted_w[0]) if len(sorted_w) > 0 else 0.0
        gross_exposure = float(np.sum(np.abs(w_arr)))
        n_positions = int(np.sum(w_arr > 1e-6))
        results.append({"date": dt, "n_positions": n_positions, "max_weight": round(max_weight, 6), "top_5_weight": round(top_5_weight, 6), "top_10_weight": round(top_10_weight, 6), "herfindahl": round(herfindahl, 6), "gross_exposure": round(gross_exposure, 6)})
    return results

# =====================================================================
# STEP 9: TRANSACTION COSTS
# =====================================================================

def compute_transaction_costs(portfolios_ordered, dates, cost_model, scenarios):
    """Compute transaction costs under multiple scenarios."""
    results = {}
    turnover_data = compute_turnover(portfolios_ordered, dates)
    total_turnover = sum(t["turnover"] for t in turnover_data)
    for scenario_name, multiplier in scenarios.items():
        effective_bps = cost_model["total_bps"] * multiplier
        cost = total_turnover * effective_bps / 10000.0
        results[scenario_name] = {"multiplier": multiplier, "effective_bps": effective_bps, "total_turnover": round(total_turnover, 6), "total_cost": round(cost, 8), "annual_cost": round(cost / max(len(dates) / 12, 1), 8)}
    return results

# =====================================================================
# STEP 10: PORTFOLIO EVALUATION
# =====================================================================

def evaluate_portfolio(portfolios_ordered, rebal_dates, daily_returns_by_instrument, cost_results=None):
    """Evaluate portfolio performance."""
    if not rebal_dates:
        return {"status": "NO_DATA"}
    daily_returns = []
    daily_dates = []
    prev_weights = {}
    for dt in sorted(daily_returns_by_instrument.get(list(daily_returns_by_instrument.keys())[0], {}).keys()):
        curr_weights = portfolios_ordered.get(dt, prev_weights)
        port_ret = 0.0
        for iid, w in curr_weights.items():
            inst_rets = daily_returns_by_instrument.get(iid, {})
            if dt in inst_rets:
                port_ret += w * inst_rets[dt]
        daily_returns.append(port_ret)
        daily_dates.append(dt)
        prev_weights = curr_weights
    daily_returns = np.array(daily_returns)
    if len(daily_returns) < 30:
        return {"status": "INSUFFICIENT_DATA"}
    ann_factor = 252
    cum_ret = float(np.prod(1 + daily_returns) - 1)
    ann_ret = float((1 + cum_ret) ** (ann_factor / len(daily_returns)) - 1) if len(daily_returns) > 0 else 0.0
    vol = float(np.std(daily_returns) * np.sqrt(ann_factor))
    sharpe = ann_ret / vol if vol > 1e-12 else 0.0
    neg_rets = daily_returns[daily_returns < 0]
    downside_vol = float(np.std(neg_rets) * np.sqrt(ann_factor)) if len(neg_rets) > 10 else vol
    sortino = ann_ret / downside_vol if downside_vol > 1e-12 else 0.0
    cum_max = np.maximum.accumulate(np.cumprod(1 + daily_returns))
    drawdowns = np.cumprod(1 + daily_returns) / cum_max - 1
    max_dd = float(np.min(drawdowns))
    calmar = ann_ret / abs(max_dd) if abs(max_dd) > 1e-12 else 0.0
    turnover_data = compute_turnover(portfolios_ordered, rebal_dates)
    total_turnover = sum(t["turnover"] for t in turnover_data)
    net_ret = cum_ret
    cost_detail = None
    if cost_results:
        baseline_cost = cost_results.get("baseline", {}).get("total_cost", 0)
        net_ret = cum_ret - baseline_cost
        cost_detail = {"gross_cumulative": round(cum_ret, 6), "baseline_cost": round(baseline_cost, 8), "net_cumulative": round(net_ret, 6)}
    return {
        "status": "OK",
        "cumulative_return": round(cum_ret, 6),
        "annualized_return": round(ann_ret, 6),
        "volatility": round(vol, 6),
        "sharpe_ratio": round(sharpe, 4),
        "sortino_ratio": round(sortino, 4),
        "max_drawdown": round(max_dd, 6),
        "calmar_ratio": round(calmar, 4),
        "total_turnover": round(total_turnover, 6),
        "annual_turnover": round(total_turnover / max(len(rebal_dates) / 12, 1), 6),
        "cost_detail": cost_detail,
        "n_rebalance_dates": len(rebal_dates),
        "n_daily_observations": len(daily_returns),
    }

def compute_prediction_metrics(panel_by_date, daily_returns_by_instrument, eval_start, eval_end):
    """Compute IC, rank IC, and bucket returns."""
    ics = []
    rank_ics = []
    for dt in panel_by_date:
        if dt < eval_start or dt > eval_end:
            continue
        preds = panel_by_date[dt]
        next_rets = {}
        for iid, p in preds.items():
            rets = daily_returns_by_instrument.get(iid, {})
            if dt in rets:
                next_rets[iid] = rets[dt]
        if len(next_rets) < 10:
            continue
        p_arr = np.array([preds[iid] for iid in next_rets])
        r_arr = np.array([next_rets[iid] for iid in next_rets])
        if np.std(p_arr) > 1e-12 and np.std(r_arr) > 1e-12:
            ics.append(float(np.corrcoef(p_arr, r_arr)[0, 1]))
            rank_ics.append(float(sp_stats.spearmanr(p_arr, r_arr)[0]))
    if not ics:
        return {"ic_mean": None, "rank_ic_mean": None, "n_obs": 0}
    return {"ic_mean": round(float(np.mean(ics)), 6), "rank_ic_mean": round(float(np.mean(rank_ics)), 6), "ic_std": round(float(np.std(ics)), 6), "n_obs": len(ics)}

def compute_bucket_returns(portfolios_ordered, daily_returns_by_instrument, eval_start, eval_end, n_buckets=5):
    """Compute returns by prediction quintile."""
    all_bucket_returns = {b: [] for b in range(n_buckets)}
    for dt in sorted(portfolios_ordered.keys()):
        if dt < eval_start or dt > eval_end:
            continue
        weights = portfolios_ordered[dt]
        if not weights:
            continue
        sorted_items = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        k = len(sorted_items) // n_buckets
        if k < 1:
            continue
        for b in range(n_buckets):
            start_idx = b * k
            end_idx = (b + 1) * k if b < n_buckets - 1 else len(sorted_items)
            bucket_ids = [iid for iid, _ in sorted_items[start_idx:end_idx]]
            bucket_ret = 0.0
            for iid in bucket_ids:
                rets = daily_returns_by_instrument.get(iid, {})
                if dt in rets:
                    bucket_ret += rets[dt] / max(len(bucket_ids), 1)
            all_bucket_returns[b].append(bucket_ret)
    return {b: round(float(np.mean(rets)), 6) if rets else None for b, rets in all_bucket_returns.items()}

# =====================================================================
# STEP 11: TEMPORAL AND UNIVERSE STABILITY
# =====================================================================

def evaluate_temporal_stability(portfolios_by_split, daily_returns_by_instrument):
    """Evaluate portfolio performance separately for val and test periods."""
    results = {}
    for split_name, portfolios in portfolios_by_split.items():
        eval_periods = {"val": SPLITS["val"], "test": SPLITS["test"]}
        for period_name, (start, end) in eval_periods.items():
            rebal_dates = sorted(portfolios.keys())
            rebal_dates = [d for d in rebal_dates if start <= d <= end]
            if not rebal_dates:
                results[f"{split_name}_{period_name}"] = {"status": "NO_DATA"}
                continue
            eval_result = evaluate_portfolio(portfolios, rebal_dates, daily_returns_by_instrument)
            results[f"{split_name}_{period_name}"] = eval_result
    return results

# =====================================================================
# STEP 12: PORTFOLIO ATTRIBUTION
# =====================================================================

def compute_attribution(portfolios_ordered, rebal_dates, daily_returns_by_instrument):
    """Decompose returns into model, weighting, and concentration effects."""
    total_ret = 0.0
    model_contrib = 0.0
    concentration_effect = 0.0
    for i, dt in enumerate(rebal_dates):
        weights = portfolios_ordered.get(dt, {})
        if not weights:
            continue
        period_ret = 0.0
        for iid, w in weights.items():
            rets = daily_returns_by_instrument.get(iid, {})
            if dt in rets:
                period_ret += w * rets[dt]
        total_ret += period_ret
        n_positions = len([w for w in weights.values() if w > 1e-6])
        hhi = sum(w**2 for w in weights.values())
        concentration_effect += (hhi - 1.0/max(n_positions, 1)) * period_ret
    return {
        "total_return": round(total_ret, 6),
        "concentration_effect": round(concentration_effect, 8),
        "n_rebalance_dates": len(rebal_dates),
        "note": "Full attribution requires factor decomposition not available in current infrastructure",
    }
