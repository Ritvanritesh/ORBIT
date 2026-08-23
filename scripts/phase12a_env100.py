"""Phase 12A benchmark - ENV-12A-100."""
import sys, time, json
from pathlib import Path
import numpy as np
import polars as pl

sys.path.insert(0, ".")
REPO_ROOT = Path.cwd()

from orbit.ml.phase11_2_benchmark import load_dataset, load_benchmark_bars, persist_results
from orbit.ml.phase12a_plan import PHASE12A_FEATURE_SETS, PHASE12A_FEATURE_NAMES
from orbit.ml.phase12a_market import compute_market_features
from orbit.ml.phase12a_sector import compute_sector_features, load_sector_mapping
from orbit.ml.phase12a_cross_sectional import compute_cross_sectional_features
from orbit.ml.features import build_feature_snapshot, FEATURE_NAMES, attach_decision_times, FeatureSnapshot, _feature_names_for_ids
from orbit.ml.labels import build_phase9_label_snapshot
from orbit.ml.data import load_instrument_master
from orbit.ml.dataset import assemble_datasets

def p(msg): print(msg, flush=True)

def get_feature_names(snap):
    from orbit.ml.features import FEATURE_DEFINITIONS, ALL_PHASE10_DEFINITIONS
    all_defs = {f["feature_id"]: f["name"] for f in FEATURE_DEFINITIONS + ALL_PHASE10_DEFINITIONS}
    all_defs.update(PHASE12A_FEATURE_NAMES)
    return [all_defs.get(f, f) for f in snap.feature_refs]

def run_single(family, params, fs_snap, lab_snap, fs_names, env_id, fs_id):
    from orbit.ml.models import train_model, predict_with_state
    from orbit.ml.metrics import oos_ic, rank_ic, hit_rate
    from orbit.ml.calibration import fit_platt
    exp_id = f"EXP-12A-{env_id}-{fs_id}-LAB-004-{family}"
    try:
        ds = assemble_datasets(fs_snap, lab_snap, feature_names=fs_names)
        X_tr, y_tr, yb_tr, _ = ds["train"]
        X_v, y_v, yb_v, _ = ds["val"]
        X_te, y_te, yb_te, meta_te = ds["test"]
        if len(X_tr) == 0 or len(X_te) == 0:
            return {"experiment_id": exp_id, "error": "insufficient data"}
        model, state = train_model(family, params, X_tr, y_tr, feature_names=fs_names)
        pred_v = predict_with_state(model, state, X_v)
        pred_te = predict_with_state(model, state, X_te)
        cal = fit_platt(pred_v, yb_v)
        pred_te_c = cal.apply(pred_te)
        tf = meta_te.with_columns([pl.Series("prediction", pred_te_c), pl.Series("outcome_value", y_te)])
        m = {}
        try:
            r = oos_ic(tf, "prediction", "outcome_value")
            m["oos_ic"] = r.get("value") if isinstance(r, dict) else float(r)
        except: m["oos_ic"] = None
        try:
            r = rank_ic(tf, "prediction", "outcome_value")
            m["rank_ic"] = r.get("value") if isinstance(r, dict) else float(r)
        except: m["rank_ic"] = None
        try:
            r = hit_rate(tf, "prediction", "outcome_value")
            m["hit_rate"] = float(r) if not isinstance(r, dict) else r.get("value")
        except: m["hit_rate"] = None
        return {"experiment_id": exp_id, "family": family, "params": params,
                "feature_set_id": fs_id, "label_id": "LAB-004", "env_id": env_id,
                "metrics": m, "n_train": len(X_tr), "n_val": len(X_v), "n_test": len(X_te)}
    except Exception as e:
        return {"experiment_id": exp_id, "error": str(e)}

def build_combined(fs001, mf, sf, xf):
    snaps = {"FS-001": fs001}
    mkt_c = ["mkt_ret_5","mkt_ret_20","mkt_vol_20","mkt_vol_60","mkt_trend_20_50","mkt_drawdown_from_peak_60"]
    sec_c = ["sector_ret_20","sector_vol_20","sector_ret_5","sector_trend_5_30","sector_dispersion_20"]
    xs_c = ["xs_rank_ret_20","xs_rank_vol_10","xs_ret_vs_median_20","xs_ret_vs_mean_20","xs_dispersion_ret_20"]
    for fs_id, cols, src, name in [("FS-101",mkt_c,mf,"market"),("FS-102",sec_c,sf,"sector"),("FS-103",xs_c,xf,"xs")]:
        t0 = time.time()
        print(f"    {fs_id} (+{name})...", end=" ", flush=True)
        frame = fs001.records
        valid = [c for c in cols if c in src.columns]
        if valid and src.height > 0:
            frame = frame.join(src.select("instrument_id","decision_session",*valid), on=["instrument_id","decision_session"], how="left")
        frame = attach_decision_times(frame)
        snaps[fs_id] = FeatureSnapshot(feature_set_id=fs_id, feature_set_version="v1", feature_refs=PHASE12A_FEATURE_SETS[fs_id]["feature_refs"], data_refs=[], records=frame, transformation=f"phase12a_{name}_v1")
        p(f"{frame.height} rows ({time.time()-t0:.1f}s)")
    t0 = time.time()
    print("    FS-104 (+all)...", end=" ", flush=True)
    frame = fs001.records
    for cols_list, src in [(mkt_c,mf),(sec_c,sf),(xs_c,xf)]:
        valid = [c for c in cols_list if c in src.columns]
        if valid and src.height > 0:
            frame = frame.join(src.select("instrument_id","decision_session",*valid), on=["instrument_id","decision_session"], how="left")
    frame = attach_decision_times(frame)
    snaps["FS-104"] = FeatureSnapshot(feature_set_id="FS-104", feature_set_version="v1", feature_refs=PHASE12A_FEATURE_SETS["FS-104"]["feature_refs"], data_refs=[], records=frame, transformation="phase12a_all_v1")
    p(f"{frame.height} rows ({time.time()-t0:.1f}s)")
    return snaps

def main():
    snapshot_id = "DS-EXP-100"
    env_id = "ENV-12A-100"
    p(f"\nLoading {snapshot_id}...")
    bars, events = load_dataset(snapshot_id)
    benchmark_bars = load_benchmark_bars()
    instruments = load_instrument_master()
    p(f"  {bars['instrument_id'].n_unique()} instruments, {bars['trade_date'].n_unique()} sessions")
    p("\n[1/5] FS-001 baseline...")
    t0 = time.time()
    fs001 = build_feature_snapshot(bars, data_refs=[snapshot_id])
    p(f"  {fs001.records.height} rows ({time.time()-t0:.1f}s)")
    universe = fs001.records.select("instrument_id","decision_session").unique()
    sector_map = load_sector_mapping(instruments)
    p("\n[2/5] Context features...")
    t0 = time.time(); print("  Market...", end=" ", flush=True); mf = compute_market_features(benchmark_bars, universe); p(f"{mf.height} rows ({time.time()-t0:.1f}s)")
    t0 = time.time(); print("  Sector...", end=" ", flush=True); sf = compute_sector_features(bars, sector_map, universe); p(f"{sf.height} rows ({time.time()-t0:.1f}s)")
    t0 = time.time(); print("  Cross-sectional...", end=" ", flush=True); xf = compute_cross_sectional_features(fs001.records, universe, FEATURE_NAMES); p(f"{xf.height} rows ({time.time()-t0:.1f}s)")
    p("\n[3/5] Combined snapshots...")
    snapshots = build_combined(fs001, mf, sf, xf)
    p("\n[4/5] Labels...")
    t0 = time.time()
    decisions = fs001.records.select("instrument_id","decision_time")
    lab = build_phase9_label_snapshot(bars, events, instruments, decisions)
    p(f"  LAB-004: {lab.records.height} rows ({time.time()-t0:.1f}s)")
    p("\n[5/5] Experiments...")
    MODELS = [("ridge",{"alpha":1.0}),("lasso",{"alpha":0.001}),("random_forest",{"max_depth":3,"n_estimators":200}),("xgboost",{"learning_rate":0.1,"max_depth":3,"n_estimators":200})]
    fsets = ["FS-001","FS-101","FS-102","FS-103","FS-104"]
    total = len(MODELS)*len(fsets)
    results = []
    done = 0
    t_start = time.time()
    for fam, par in MODELS:
        for fs_id in fsets:
            done += 1
            snap = snapshots[fs_id]
            fnames = get_feature_names(snap)
            elapsed = time.time()-t_start
            eta = (elapsed/max(done-1,1))*(total-done) if done > 1 else 0
            print(f"  [{done}/{total}] {fam}+{fs_id} ({elapsed:.0f}s, ETA {eta:.0f}s)", end=" ", flush=True)
            r = run_single(fam, par, snap, lab, fnames, env_id, fs_id)
            results.append(r)
            ic = r.get("metrics",{}).get("oos_ic")
            if "error" in r: p(f"ERR: {r['error'][:40]}")
            else: p(f"IC={ic:.4f}" if ic else "IC=N/A")
    elapsed = time.time()-t_start
    ok = [r for r in results if "error" not in r]
    ics = [r["metrics"]["oos_ic"] for r in ok if r["metrics"].get("oos_ic") is not None]
    p(f"\n  {len(ok)}/{total} succeeded in {elapsed:.1f}s")
    if ics: p(f"  OOS IC: mean={np.mean(ics):.4f}, median={np.median(ics):.4f}, min={np.min(ics):.4f}, max={np.max(ics):.4f}")
    p("\n--- Per Feature Set ---")
    for fs_id in fsets:
        fic = [r["metrics"]["oos_ic"] for r in results if r.get("feature_set_id")==fs_id and "error" not in r and r["metrics"].get("oos_ic") is not None]
        if fic: p(f"  {fs_id}: mean={np.mean(fic):.4f}, median={np.median(fic):.4f}")
        else: p(f"  {fs_id}: no ICs")
    res = {"env_id":env_id,"snapshot_id":snapshot_id,"n_instruments":int(bars["instrument_id"].n_unique()),"n_sessions":int(bars["trade_date"].n_unique()),"results":results,"n_successful":len(ok),"n_failed":len(results)-len(ok),"elapsed_seconds":elapsed}
    persist_results(res)
    out = REPO_ROOT/"benchmarks"/f"phase12a_{env_id}_results.json"
    out.write_text(json.dumps(res,indent=2,default=str),encoding="utf-8")
    p(f"\nSaved: {out.name}")

if __name__ == "__main__":
    main()
