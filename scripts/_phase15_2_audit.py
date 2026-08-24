"""Phase 15.2 — Model Stability & Signal Reconciliation Audit.

Investigates remaining limitations from Phase 15.1:
1. Macro feature correlation (r=0.818)
2. CLIFF sensitivity (H-3-RIDGE-050)
3. Directional disagreement (100% sign conflict)
4. Lasso degeneracy
"""
from __future__ import annotations
import hashlib, json, sys, time, warnings
from datetime import date as _date
from pathlib import Path
import numpy as np
import polars as pl
from scipy import stats as sp_stats
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")

REPO = Path(r"E:\Orbit (Optimized Research & Behavioral Intelligence Trading)")
BENCH = REPO / "benchmarks"
DOCS = REPO / "docs"
sys.path.insert(0, str(REPO / "src"))
SEED = 42
SPLITS = {"train": (_date(2010, 1, 4), _date(2018, 12, 31)),
          "val": (_date(2019, 1, 2), _date(2021, 12, 31)),
          "test": (_date(2022, 1, 3), _date(2026, 6, 30))}
BASELINE = ["ret_10", "ret_20", "ret_30", "sma_ratio_5_30", "sma_ratio_15_40", "vol_10", "vol_30", "log_dv_med_20"]
H3 = ["macro_dff_level", "macro_dff_change_3m", "macro_unemployment_level", "macro_cpi_yoy"]
FEATURE_SETS = {"FS-BASELINE": BASELINE, "FS-H3": BASELINE + H3}
CANDIDATES = [
    {"model_id": "H3-RIDGE-050", "model_type": "ridge", "alpha": 1.0, "feature_set": "FS-H3", "dataset_key": "050"},
    {"model_id": "H3-LASSO-050", "model_type": "lasso", "alpha": 0.001, "feature_set": "FS-H3", "dataset_key": "050"},
    {"model_id": "H3-RIDGE-100", "model_type": "ridge", "alpha": 1.0, "feature_set": "FS-H3", "dataset_key": "100"},
    {"model_id": "H3-LASSO-100", "model_type": "lasso", "alpha": 0.001, "feature_set": "FS-H3", "dataset_key": "100"},
]
LASSO_ALPHAS = [1e-5, 1e-4, 1e-3, 0.01, 0.1, 1.0]

def save_json(name, data):
    with open(BENCH / name, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str, ensure_ascii=False)
    print(f"  Saved: {name}")

def load_json(name):
    with open(BENCH / name, encoding="utf-8") as f:
        return json.load(f)

def canonical(obj):
    return json.dumps(obj, sort_keys=True, default=str, ensure_ascii=False)

def digest_full(obj):
    return hashlib.sha256(canonical(obj).encode()).hexdigest()

def load_parquet(rel):
    return pl.read_parquet(REPO / rel)

def compute_features_polars(df):
    pdf = df.sort("trade_date").with_row_index("_row_idx")
    pdf = pdf.with_columns((pl.col("close") / pl.col("close").shift(1) - 1).alias("daily_ret"))
    pdf = pdf.with_columns([(pl.col("close") / pl.col("close").shift(10) - 1).alias("ret_10"),
        (pl.col("close") / pl.col("close").shift(20) - 1).alias("ret_20"),
        (pl.col("close") / pl.col("close").shift(30) - 1).alias("ret_30")])
    pdf = pdf.with_columns([
        (pl.col("close").rolling_mean(5) / pl.col("close").rolling_mean(30) - 1).alias("sma_ratio_5_30"),
        (pl.col("close").rolling_mean(15) / pl.col("close").rolling_mean(40) - 1).alias("sma_ratio_15_40")])
    pdf = pdf.with_columns([pl.col("daily_ret").rolling_std(10).alias("vol_10"),
        pl.col("daily_ret").rolling_std(30).alias("vol_30")])
    pdf = pdf.with_columns([((pl.col("close") * pl.col("volume")).rolling_median(20) + 1).log().alias("log_dv_med_20")])
    pdf = pdf.with_columns(pl.col("close").rolling_max(20).alias("_peak"))
    pdf = pdf.with_columns((pl.col("close") / pl.col("_peak") - 1).rolling_min(20).alias("path_max_drawdown_20")).drop("_peak")
    pdf = pdf.with_columns([pl.when(pl.col("daily_ret") > 0).then(1).otherwise(0).rolling_sum(20).alias("_n_up"),
        pl.when(pl.col("daily_ret") < 0).then(1).otherwise(0).rolling_sum(20).alias("_n_down")])
    pdf = pdf.with_columns((pl.col("_n_up") / pl.max_horizontal(pl.col("_n_down"), 1)).alias("path_up_down_ratio_20")).drop(["_n_up", "_n_down"])
    pdf = pdf.with_columns(pl.col("daily_ret").abs().rolling_max(20).alias("path_largest_move_20"))
    pdf = pdf.with_columns([pl.col("daily_ret").rolling_mean(20).alias("_mu20"), pl.col("daily_ret").rolling_std(20).alias("_std20")])
    pdf = pdf.with_columns([((pl.col("daily_ret") - pl.col("_mu20")).pow(3).rolling_mean(20)).alias("_m3"),
        ((pl.col("daily_ret") - pl.col("_mu20")).pow(4).rolling_mean(20)).alias("_m4")])
    pdf = pdf.with_columns([(pl.col("_m3") / pl.col("_std20").pow(3)).alias("return_skew_20"),
        (pl.col("_m4") / pl.col("_std20").pow(4) - 3).alias("return_kurt_20")]).drop(["_mu20", "_std20", "_m3", "_m4"])
    pdf = pdf.with_columns(pl.when(pl.col("daily_ret") < 0).then(pl.col("daily_ret")).otherwise(None).rolling_std(20).alias("downside_vol_20"))
    pdf = pdf.with_columns([pl.col("daily_ret").rolling_std(5).alias("_vol5"), pl.col("daily_ret").rolling_std(10).alias("_vol10")])
    pdf = pdf.with_columns(pl.col("_vol5").rolling_std(20).alias("vol_of_vol_20"))
    pdf = pdf.with_columns((pl.col("_vol10") - pl.col("_vol10").shift(20)).alias("vol_change_20")).drop(["_vol5", "_vol10"])
    return pdf.drop("_row_idx")

def compute_macro_features(spy_df, fred_df):
    spy = spy_df.sort("trade_date")
    fred = fred_df.sort("observation_date")
    fred_wide = fred.pivot(index="observation_date", on="series_id", values="value")
    dates = spy["trade_date"].to_list()
    n = len(dates)
    result = {}
    for sid in ["DFF", "UNRATE", "CPIAUCSL"]:
        if sid in fred_wide.columns:
            vf = fred_wide["observation_date"].to_list()
            vd = fred_wide[sid].to_list()
            out = np.full(n, np.nan)
            fi = 0
            for di, d in enumerate(dates):
                while fi < len(vf) - 1 and vf[fi + 1] <= d:
                    fi += 1
                if vf[fi] <= d:
                    out[di] = vd[fi]
            result[f"raw_{sid}"] = out
        else:
            result[f"raw_{sid}"] = np.full(n, np.nan)
    raw_dff = result.get("raw_DFF", np.full(n, np.nan))
    raw_unrate = result.get("raw_UNRATE", np.full(n, np.nan))
    raw_cpi = result.get("raw_CPIAUCSL", np.full(n, np.nan))
    dff_chg = np.full(n, np.nan)
    for i in range(63, n):
        if not np.isnan(raw_dff[i]) and not np.isnan(raw_dff[i - 63]):
            dff_chg[i] = raw_dff[i] - raw_dff[i - 63]
    cpi_yoy = np.full(n, np.nan)
    for i in range(252, n):
        if not np.isnan(raw_cpi[i]) and not np.isnan(raw_cpi[i - 252]) and raw_cpi[i - 252] > 0:
            cpi_yoy[i] = raw_cpi[i] / raw_cpi[i - 252] - 1
    out = spy[["trade_date"]].clone()
    return out.with_columns([pl.Series("macro_dff_level", raw_dff), pl.Series("macro_dff_change_3m", dff_chg),
        pl.Series("macro_unemployment_level", raw_unrate), pl.Series("macro_cpi_yoy", cpi_yoy)])

def compute_labels(df, horizon=5):
    pdf = df.sort("trade_date")
    close = pdf["close"].to_numpy()
    n = len(close)
    fwd_ret = np.full(n, np.nan)
    for i in range(n - horizon):
        if close[i] > 0:
            fwd_ret[i] = close[i + horizon] / close[i] - 1
    out = pdf[["trade_date", "instrument_id"]].clone()
    return out.with_columns(pl.Series("label", fwd_ret))

def assemble_dataset(features_df, labels_df, feature_names):
    merged = features_df.join(labels_df, on=["trade_date", "instrument_id"], how="inner")
    for col in feature_names + ["label"]:
        merged = merged.filter(pl.col(col).is_not_null())
    merged = merged.sort("trade_date")
    result = {}
    for sn, (start, end) in SPLITS.items():
        mask = (merged["trade_date"] >= start) & (merged["trade_date"] <= end)
        sdf = merged.filter(mask)
        X = sdf.select(feature_names).to_numpy().astype(np.float64)
        y = sdf["label"].to_numpy().astype(np.float64)
        meta = sdf.select(["trade_date", "instrument_id"]).to_dicts()
        valid = np.isfinite(X).all(axis=1) & np.isfinite(y)
        result[sn] = (X[valid], y[valid], [m for m, v in zip(meta, valid) if v])
    return result

def train_model(X_tr, y_tr, model_type, alpha):
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    model = Ridge(alpha=alpha, random_state=SEED) if model_type == "ridge" else Lasso(alpha=alpha, random_state=SEED, max_iter=100000)
    model.fit(X_tr_s, y_tr)
    return model, scaler

def predict_model(model, scaler, X):
    return model.predict(scaler.transform(X))

def orthogonalize_features(X, feature_names):
    """Gram-Schmidt orthogonalization of macro features against baseline."""
    baseline_cols = [i for i, f in enumerate(feature_names) if f in BASELINE]
    macro_cols = [i for i, f in enumerate(feature_names) if f in H3]
    if not baseline_cols or not macro_cols:
        return X.copy()
    X_orth = X.copy()
    B = X[:, baseline_cols]
    B_mean = B.mean(axis=0)
    B_centered = B - B_mean
    for mc in macro_cols:
        v = X[:, mc].copy()
        v_centered = v - v.mean()
        for bc_idx in range(len(baseline_cols)):
            b = B_centered[:, bc_idx]
            b_norm = np.dot(b, b)
            if b_norm > 1e-12:
                proj = np.dot(v_centered, b) / b_norm
                v_centered = v_centered - proj * b
        X_orth[:, mc] = v_centered
    return X_orth

def demean_features(X, feature_names, X_train_mean):
    """Demean macro features using training set mean."""
    X_dm = X.copy()
    for i, fn in enumerate(feature_names):
        if fn in H3:
            X_dm[:, i] = X[:, i] - X_train_mean[i]
    return X_dm
