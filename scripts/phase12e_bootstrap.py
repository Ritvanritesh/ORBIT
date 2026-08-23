"""Phase 12E bootstrap: generates and runs the corrected excess-return experiments."""
import subprocess, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "_phase12e_impl.py"

# Generate the implementation script
impl = r'''"""Phase 12E implementation: corrected LAB-006 excess-return revalidation."""
import hashlib, json, sys, time
from pathlib import Path
import numpy as np
import polars as pl

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from orbit.ml.phase11_2_benchmark import load_dataset, load_benchmark_bars
from orbit.ml.phase11_2_universe import get_universe_50, get_universe_100
from orbit.ml.features import build_feature_snapshot
from orbit.ml.labels import build_phase9_label_snapshot
from orbit.ml.data import load_instrument_master

BENCH = REPO / "benchmarks"
BENCH.mkdir(exist_ok=True)

BASELINE = ["ret_10","ret_20","ret_30","sma_ratio_5_30","sma_ratio_15_40","vol_10","vol_30","log_dv_med_20"]
FS_MAP = {
    "FS-12B-A": BASELINE,
    "FS-12B-B": BASELINE + ["f_eps_diluted","f_shareholders_equity","f_revenue"],
    "FS-12B-C": BASELINE + ["f_roa","f_roe","f_operating_margin","f_gross_profitability"],
    "FS-12B-D": BASELINE + ["f_net_income","f_operating_cash_flow","f_total_assets"],
    "FS-12B-E": BASELINE + ["f_debt_to_equity","f_debt_to_assets","f_current_ratio"],
    "FS-12B-F": BASELINE + ["f_eps_diluted","f_shareholders_equity","f_revenue",
        "f_roa","f_roe","f_operating_margin","f_gross_profitability",
        "f_net_income","f_operating_cash_flow","f_total_assets",
        "f_debt_to_equity","f_debt_to_assets","f_current_ratio"],
}
MODELS = [
    {"family":"ridge","params":{"alpha":1.0}},
    {"family":"lasso","params":{"alpha":0.001}},
    {"family":"random_forest","params":{"max_depth":3,"n_estimators":200}},
    {"family":"xgboost","params":{"learning_rate":0.1,"max_depth":3,"n_estimators":200}},
]

def compute_excess_labels(inst_bars, bench_bars, decisions, horizon=5):
    bench_close = {r["trade_date"]: r["close"] for r in bench_bars.sort("trade_date").to_dicts()}
    bench_dates = sorted(bench_close.keys())
    inst_series = {}
    for iid in inst_bars["instrument_id"].unique().to_list():
        sub = inst_bars.filter(pl.col("instrument_id")==iid).sort("trade_date")
        inst_series[iid] = {r["trade_date"]: r["close"] for r in sub.to_dicts()}
    results = []
    for row in decisions.to_dicts():
        iid = row["instrument_id"]
        dt = row["decision_time"]
        if hasattr(dt, "date"): dt = dt.date()
        dates = sorted(inst_series.get(iid, {}).keys())
        entry_date = entry_close = None
        for d in reversed(dates):
            if d < dt:
                entry_date, entry_close = d, inst_series[iid][d]
                break
        if entry_date is None:
            results.append({"instrument_id":iid,"decision_time":row["decision_time"],
                "outcome_value":None,"instrument_return":None,"benchmark_return":None,
                "label_available":False,"outcome_status":"unavailable","unavailable_reason":"no_entry"})
            continue
        eidx = dates.index(entry_date) if entry_date in dates else None
        if eidx is None or eidx+horizon>=len(dates):
            results.append({"instrument_id":iid,"decision_time":row["decision_time"],
                "outcome_value":None,"instrument_return":None,"benchmark_return":None,
                "label_available":False,"outcome_status":"unavailable","unavailable_reason":"short_window"})
            continue
        out_date = dates[eidx+horizon]
        out_close = inst_series[iid][out_date]
        if not entry_close or entry_close<=0 or not out_close:
            results.append({"instrument_id":iid,"decision_time":row["decision_time"],
                "outcome_value":None,"instrument_return":None,"benchmark_return":None,
                "label_available":False,"outcome_status":"unavailable","unavailable_reason":"bad_price"})
            continue
        inst_ret = out_close/entry_close - 1.0
        be = bc = None
        for bd in reversed(bench_dates):
            if bd<=entry_date: be=bench_close[bd]; break
        for bd in reversed(bench_dates):
            if bd<=out_date: bc=bench_close[bd]; break
        if be is None or bc is None or be<=0:
            results.append({"instrument_id":iid,"decision_time":row["decision_time"],
                "outcome_value":None,"instrument_return":None,"benchmark_return":None,
                "label_available":False,"outcome_status":"unavailable","unavailable_reason":"bench_unavail"})
            continue
        bench_ret = bc/be - 1.0
        excess = inst_ret - bench_ret
        results.append({"instrument_id":iid,"decision_time":row["decision_time"],
            "outcome_value":excess,"instrument_return":inst_ret,"benchmark_return":bench_ret,
            "label_available":True,"outcome_status":"available","unavailable_reason":None})
    return pl.DataFrame(results)


def assemble_ds(feat_records, lab_records, feat_names):
    from orbit.ml.splits import assign_split, purge_outcome_windows, assert_split_integrity
    from orbit.ml.dataset import _join_features_labels
    class _S:
        def __init__(s, r): s.records = r
    joined = _join_features_labels(_S(feat_records), _S(lab_records))
    avail = joined.filter(pl.col("outcome_status")=="available")
    sf = assign_split(avail, "decision_session")
    sf = purge_outcome_windows(sf)
    assert_split_integrity(sf)
    ds = {}
    for sn in ["train","val","test"]:
        sub = sf.filter(pl.col("split")==sn)
        af = [f for f in feat_names if f in sub.columns]
        X = sub.select(af).to_numpy().astype(np.float32)
        y = sub["outcome_value"].to_numpy().astype(np.float64)
        yb = (y>0).astype(np.float64)
        meta = sub.select("instrument_id","decision_session","outcome_value")
        ds[sn] = (X, y, yb, meta)
    return ds


def train_eval(family, params, feat_names, ds, seed=42):
    from orbit.ml.models import train_model, predict_with_state
    from orbit.ml.metrics import oos_ic, rank_ic, hit_rate
    Xt,yt,_,_ = ds["train"]; Xv,yv,_,_ = ds["val"]; Xs,ys,_, mt = ds["test"]
    if len(Xt)==0 or len(Xs)==0: return None
    model, state = train_model(family, params, Xt, yt, feature_names=feat_names, seed=seed)
    pred = predict_with_state(model, state, Xs)
    tf = mt.with_columns([pl.Series("prediction",pred), pl.Series("outcome_value",ys)])
    return {"oos_ic":float(oos_ic(tf)),"rank_ic":float(rank_ic(tf)),
            "hit_rate":float(hit_rate(tf)),"n_train":len(Xt),"n_val":len(Xv),"n_test":len(Xs)}


def run_env(env_id, snap_id):
    print(f"\n{'='*72}\nENVIRONMENT: {env_id} ({snap_id})\n{'='*72}")
    t_total = time.time()

    print("\n[1/7] Loading data...")
    bars, events = load_dataset(snap_id)
    benchmark = load_benchmark_bars()
    print(f"  Instruments: {bars['instrument_id'].n_unique()}, Benchmark rows: {benchmark.height}")

    print("\n[2/7] Building features...")
    fs = build_feature_snapshot(bars, data_refs=[snap_id])
    print(f"  Features: {fs.records.height} rows")

    print("\n[3/7] Computing LAB-004 (absolute)...")
    instruments = load_instrument_master()
    decisions = fs.records.select("instrument_id","decision_time")
    lab004 = build_phase9_label_snapshot(bars, events, instruments, decisions)
    print(f"  LAB-004: {lab004.records.height} rows")

    print("\n[4/7] Computing LAB-006 (excess vs SPY)...")
    t0 = time.time()
    lab006 = compute_excess_labels(bars, benchmark, decisions)
    avail = lab006.filter(pl.col("label_available")==True)
    print(f"  LAB-006: {lab006.height} total, {avail.height} available ({time.time()-t0:.1f}s)")

    print("\n[5/7] Validating label divergence...")
    l4a = lab004.records.filter(pl.col("outcome_status")=="available").select("instrument_id","decision_time","outcome_value")
    merged = l4a.join(avail.select("instrument_id","decision_time","outcome_value"), on=["instrument_id","decision_time"], how="inner", suffix="_excess")
    if merged.height > 0:
        ndiff = merged.filter((pl.col("outcome_value")-pl.col("outcome_value_excess")).abs()>1e-10).height
        pct = ndiff/merged.height*100
        print(f"  Matched: {merged.height}, Different: {ndiff} ({pct:.1f}%)")
        print(f"  {'PASS' if pct>=1.0 else 'FAIL'}: Labels are {'materially different' if pct>=1.0 else 'suspiciously similar'}")
    else:
        print("  WARNING: No matched observations")

    print("\n[6/7] Loading fundamentals...")
    from orbit.ml.phase12d import extract_all_observations, pit_asof_join, pivot_fundamental_features, compute_derived_fundamental_features
    raw_dir = REPO / "data" / "raw"
    obs = extract_all_observations(raw_dir)
    ticker_map = {}
    u = get_universe_50() if "050" in env_id else get_universe_100()
    for inst in u: ticker_map[inst["instrument_id"]] = inst["ticker"]
    if obs.height > 0:
        dd = fs.records.select("instrument_id", pl.col("decision_session")).unique().rename({"instrument_id":"ticker"})
        dd = dd.with_columns(pl.col("ticker").replace(ticker_map).alias("ticker"))
        raw_t = set(obs["ticker"].unique().to_list())
        dd = dd.filter(pl.col("ticker").is_in(list(raw_t)))
        pit = pit_asof_join(obs, dd)
        wide = pivot_fundamental_features(pit)
        wide = compute_derived_fundamental_features(wide)
        print(f"  Wide: {wide.height} rows")
    else:
        wide = pl.DataFrame()
        print("  No observations")

    print("\n[7/7] Running experiments...")
    from orbit.ml.features import FeatureSnapshot
    all_fs = {}
    for fid, fnames in FS_MAP.items():
        if fid=="FS-12B-A":
            all_fs[fid] = fs
        elif wide.height > 0:
            rev = {v:k for k,v in ticker_map.items()}
            wm = wide.with_columns(pl.col("ticker").replace(rev).alias("instrument_id"))
            fcols = [c for c in wm.columns if c.startswith("f_")]
            if fcols:
                jdf = wm.select("instrument_id","decision_session",*fcols)
                m = fs.records.join(jdf, on=["instrument_id","decision_session"], how="left")
                for c in fcols:
                    if c not in m.columns:
                        m = m.with_columns(pl.lit(None).cast(pl.Float64).alias(c))
                all_fs[fid] = FeatureSnapshot(feature_set_id=fid,feature_set_version="v1",
                    feature_refs=fs.feature_refs+fcols, data_refs=fs.data_refs, records=m,
                    transformation="pit_fundamental", limitations=["PIT join"])
            else:
                all_fs[fid] = fs
        else:
            all_fs[fid] = fs

    results = []
    total = len(MODELS)*len(FS_MAP)
    done = 0
    for mc in MODELS:
        fam, par = mc["family"], mc["params"]
        for fid, fnames in FS_MAP.items():
            done += 1
            t0 = time.time()
            print(f"  [{done}/{total}] {fam}+{fid}+LAB-006...", end="", flush=True)
            try:
                snap = all_fs[fid]
                avail_f = [f for f in fnames if f in snap.records.columns]
                m = snap.records
                for c in set(fnames)-set(avail_f):
                    m = m.with_columns(pl.lit(None).cast(pl.Float64).alias(c))
                ds = assemble_ds(m, lab006, fnames)
                met = train_eval(fam, par, fnames, ds)
                if met:
                    print(f" IC={met['oos_ic']:.4f} ({time.time()-t0:.1f}s)")
                    results.append({"experiment_id":f"EXP-12E-{env_id}-{fid}-LAB-006-{fam}",
                        "family":fam,"params":par,"feature_set_id":fid,"label_id":"LAB-006",
                        "env_id":env_id,"metrics":{"oos_ic":met["oos_ic"],"rank_ic":met["rank_ic"],
                        "hit_rate":met["hit_rate"]},"n_train":met["n_train"],"n_val":met["n_val"],
                        "n_test":met["n_test"],"feature_names":fnames})
                else:
                    print(f" BLOCKED ({time.time()-t0:.1f}s)")
                    results.append({"experiment_id":f"EXP-12E-{env_id}-{fid}-LAB-006-{fam}",
                        "family":fam,"params":par,"feature_set_id":fid,"label_id":"LAB-006",
                        "env_id":env_id,"metrics":{"oos_ic":None,"rank_ic":None,"hit_rate":None},
                        "n_train":0,"n_val":0,"n_test":0,"feature_names":fnames,"blocked":True})
            except Exception as e:
                print(f" ERROR ({e}) ({time.time()-t0:.1f}s)")
                results.append({"experiment_id":f"EXP-12E-{env_id}-{fid}-LAB-006-{fam}",
                    "family":fam,"params":par,"feature_set_id":fid,"label_id":"LAB-006",
                    "env_id":env_id,"metrics":{"oos_ic":None,"rank_ic":None,"hit_rate":None},
                    "n_train":0,"n_val":0,"n_test":0,"feature_names":fnames,"error":str(e)})

    ok = [r for r in results if r["metrics"]["oos_ic"] is not None]
    blocked = [r for r in results if r.get("blocked")]
    failed = [r for r in results if r.get("error")]
    ics = [r["metrics"]["oos_ic"] for r in ok]
    elapsed = time.time()-t_total
    print(f"\n  Completed: {len(ok)}/{total} in {elapsed:.1f}s")
    print(f"  Blocked: {len(blocked)}, Failed: {len(failed)}")
    if ics:
        print(f"  OOS IC: mean={np.mean(ics):.4f}, median={np.median(ics):.4f}")

    out = {"env_id":env_id,"snapshot_id":snap_id,
        "n_instruments":bars["instrument_id"].n_unique(),
        "n_sessions":bars["trade_date"].n_unique(),
        "results":results,"n_successful":len(ok),"n_blocked":len(blocked),
        "n_failed":len(failed),"elapsed_seconds":elapsed,
        "label_id":"LAB-006","label_version":"v1","benchmark":"BENCH-001"}
    p = BENCH/f"phase12e_{env_id}_results.json"
    with open(p,"w") as f: json.dump(out,f,indent=2)
    print(f"  Saved: {p.name}")
    return out


if __name__=="__main__":
    print("="*72)
    print("PHASE 12E - CORRECTED EXCESS-RETURN REVALIDATION")
    print("="*72)
    print("Label: LAB-006 v1 (corrected excess return vs BENCH-001/SPY)")
    print("Experiments: 48 per environment (4 models x 6 feature sets)")
    print()

    # Lock plan
    plan = {"phase":"12E","label_id":"LAB-006","label_version":"v1",
        "benchmark":"BENCH-001","horizon":5,"environments":["ENV-12E-050","ENV-12E-100"],
        "models":[{"family":"ridge","params":{"alpha":1.0}},
                  {"family":"lasso","params":{"alpha":0.001}},
                  {"family":"random_forest","params":{"max_depth":3,"n_estimators":200}},
                  {"family":"xgboost","params":{"learning_rate":0.1,"max_depth":3,"n_estimators":200}}],
        "feature_sets":list(FS_MAP.keys()),
        "n_experiments_per_env":48,"total_experiments":96}
    plan_json = json.dumps(plan,sort_keys=True)
    plan_hash = hashlib.sha256(plan_json.encode()).hexdigest()
    plan["plan_hash"] = plan_hash
    with open(BENCH/"phase12e_plan.json","w") as f: json.dump(plan,f,indent=2)
    print(f"Plan locked: {plan_hash[:16]}...")

    r050 = run_env("ENV-12E-050","DS-EXP-050")
    r100 = run_env("ENV-12E-100","DS-EXP-100")

    # Summary
    print("\n" + "="*72)
    print("PHASE 12E SUMMARY")
    print("="*72)
    for r in [r050, r100]:
        ics = [e["metrics"]["oos_ic"] for e in r["results"] if e["metrics"]["oos_ic"] is not None]
        sig = sum(1 for ic in ics if abs(ic) > 0.01)
        print(f"\n  {r['env_id']}: {r['n_successful']}/{len(r['results'])} completed")
        print(f"    OOS IC: mean={np.mean(ics):.4f}, median={np.median(ics):.4f}" if ics else "    No results")
        print(f"    |IC|>0.01: {sig}/{len(ics)}")

    # Save comparison
    comp = {
        "phase_12e_label": "LAB-006 v1 (corrected excess return vs SPY)",
        "phase_12d_label": "LAB-005 (was identical to LAB-004 - DEFECT)",
        "env_050": {"lab006_mean_ic": float(np.mean([e["metrics"]["oos_ic"] for e in r050["results"] if e["metrics"]["oos_ic"] is not None])) if r050["results"] else None},
        "env_100": {"lab006_mean_ic": float(np.mean([e["metrics"]["oos_ic"] for e in r100["results"] if e["metrics"]["oos_ic"] is not None])) if r100["results"] else None},
    }
    with open(BENCH/"phase12e_comparison.json","w") as f: json.dump(comp,f,indent=2)
    print("\nComparison saved.")
'''

SCRIPT.write_text(impl, encoding='utf-8')
print(f"Implementation script written: {SCRIPT}")
print(f"Lines: {impl.count(chr(10))+1}")
