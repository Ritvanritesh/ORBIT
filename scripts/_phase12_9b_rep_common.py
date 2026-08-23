"""Phase 12.9B - Common utilities and data loading."""
from __future__ import annotations
import hashlib, json, os, sys, time
from datetime import datetime
from pathlib import Path
import numpy as np
import polars as pl
from scipy import stats

REPO = Path(r"E:\Orbit (Optimized Research & Behavioral Intelligence Trading)")
BENCH = REPO / "benchmarks"
DATA = REPO / "data"
OUT = BENCH
sys.path.insert(0, str(REPO / "src"))

def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def save_json(name, data):
    with open(OUT / name, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str, ensure_ascii=False)
    print(f"  Saved: {name}")

TOLERANCE = {"EXACT_MATCH": 1e-10, "NUMERICALLY_EQUIVALENT": 0.001,
             "MINOR_DRIFT": 0.005, "MATERIAL_DRIFT": 0.01}

def classify(historical, clean):
    if historical is None or clean is None:
        return "MISSING"
    diff = abs(historical - clean)
    if diff < TOLERANCE["EXACT_MATCH"]:
        return "EXACT_MATCH"
    if diff < TOLERANCE["NUMERICALLY_EQUIVALENT"]:
        return "NUMERICALLY_EQUIVALENT"
    if diff < TOLERANCE["MINOR_DRIFT"]:
        return "MINOR_DRIFT"
    if diff < TOLERANCE["MATERIAL_DRIFT"]:
        return "MATERIAL_DRIFT"
    return "FAILED_TO_REPRODUCE"

def compute_pvals(results):
    n = len(results)
    alpha = 0.05
    rwp = []
    for exp in results:
        ic = exp["metrics"].get("oos_ic")
        nt = exp.get("n_test", 0)
        if ic is None or abs(ic) < 1e-10 or nt < 3:
            pv = 1.0
        else:
            t = ic * np.sqrt(nt - 2) / np.sqrt(max(1 - ic**2, 1e-20))
            pv = 2 * (1 - stats.t.cdf(abs(t), df=nt - 2))
        rwp.append({"eid": exp["experiment_id"], "ic": ic, "pval": pv})
    rwp.sort(key=lambda x: x["pval"])
    holm = set()
    for i, r in enumerate(rwp):
        if r["pval"] <= alpha / (n - i):
            holm.add(r["eid"])
        else:
            break
    bh = set()
    for i, r in enumerate(rwp):
        if r["pval"] <= alpha * (i + 1) / n:
            bh.add(r["eid"])
    return {"holm": holm, "bh": bh, "n_sig_holm": len(holm), "n_sig_bh": len(bh)}

def experiment_stats(results):
    ics = [e["metrics"]["oos_ic"] for e in results if e["metrics"].get("oos_ic") is not None]
    if not ics:
        return {"n": 0, "mean": None, "median": None, "best": None}
    return {"n": len(ics), "mean": float(np.mean(ics)),
            "median": float(np.median(ics)), "best": float(np.max(ics))}

def run_experiment(model, params, feat_names, ds, label_col="outcome_value"):
    from orbit.ml.models import train_model, predict_with_state
    from orbit.ml.metrics import oos_ic, rank_ic, hit_rate
    from orbit.ml.calibration import fit_platt
    from orbit.ml.grids import validate_model_parameters
    params = validate_model_parameters(model, params)
    Xt, yt, _, _ = ds["train"]
    Xv, yv, _, _ = ds["val"]
    Xs, ys, _, mt = ds["test"]
    if len(Xt) == 0 or len(Xs) == 0:
        return None
    model_obj, state = train_model(model, params, Xt, yt, feature_names=feat_names)
    pred_val = predict_with_state(model_obj, state, Xv)
    pred_test = predict_with_state(model_obj, state, Xs)
    try:
        cal = fit_platt(pred_val, yv)
        pred_test = cal.apply(pred_test)
    except Exception:
        pass
    tf = mt.with_columns([
        pl.Series("prediction", pred_test),
        pl.Series("outcome_value", ys),
    ])
    ic_r = oos_ic(tf, "prediction", "outcome_value")
    ric_r = rank_ic(tf, "prediction", "outcome_value")
    hr_r = hit_rate(tf, "prediction", "outcome_value")
    ic_val = ic_r.get("value") if isinstance(ic_r, dict) else float(ic_r)
    ric_val = ric_r.get("value") if isinstance(ric_r, dict) else float(ric_r)
    hr_val = float(hr_r) if not isinstance(hr_r, dict) else hr_r.get("value")
    return {"oos_ic": ic_val, "rank_ic": ric_val, "hit_rate": hr_val,
            "n_train": len(Xt), "n_val": len(Xv), "n_test": len(Xs)}

def build_clean_dataset(bars, events, instruments, fs, lab, feature_names):
    from orbit.ml.dataset import assemble_datasets
    return assemble_datasets(fs, lab, feature_names=feature_names)

print("Phase 12.9B common utilities loaded.")
