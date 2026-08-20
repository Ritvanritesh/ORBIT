"""Phase 10 feature diagnostics: quality checks and redundancy analysis.

All diagnostics are deterministic. Correlation / duplicate / distribution
diagnostics are computed on the TRAINING split only - the test split is
never used to compute statistics that could influence any decision
(diagnostics are reports, but a test-set-fitted diagnostic is still test-set
snooping, so the train-only rule is applied strictly). Split-level
distribution reports (train/val/test means etc.) are reported separately as
stability evidence, never as fitted statistics.

Redundancy policy (documented, per the Phase 10 requirements): correlation
and duplicate diagnostics IDENTIFY redundancy; they never auto-remove a
feature. Whether a redundant feature adds incremental OOS value is tested by
the registered ablation experiments, not by a correlation threshold.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import polars as pl
from scipy.stats import rankdata

IDENTITY_COLUMNS = ("instrument_id", "decision_session", "decision_time", "window_end_session")


def feature_columns(frame: pl.DataFrame) -> list[str]:
    return [c for c in frame.columns if c not in IDENTITY_COLUMNS]


def _per_split_stats(frame: pl.DataFrame, feature_names: list[str]) -> list[dict[str, Any]]:
    out = []
    if "split" not in frame.columns:
        splits: tuple[str, ...] = ("all",)
        get_split = lambda _row: "all"
    else:
        splits = ("train", "val", "test")
        get_split = lambda row: row["split"]
    for split in splits:
        if "split" in frame.columns:
            sub = frame.filter(pl.col("split") == split)
        else:
            sub = frame
        if sub.height == 0:
            continue
        for name in feature_names:
            s = sub[name]
            out.append(
                {
                    "split": split,
                    "feature": name,
                    "rows": int(sub.height),
                    "n_null": int(s.null_count()),
                    "mean": float(s.mean()) if s.null_count() < sub.height else None,
                    "std": float(s.std()) if (s.null_count() < sub.height and (sub.height - s.null_count()) > 1) else None,
                    "min": float(s.min()) if s.null_count() < sub.height else None,
                    "max": float(s.max()) if s.null_count() < sub.height else None,
                }
            )
    return out


def feature_quality_report(
    frame: pl.DataFrame,
    feature_names: list[str],
) -> dict[str, Any]:
    """Per-feature quality diagnostics on a labeled split frame.

    Reports: missingness, constant / near-constant detection, duplicate-value
    frequency, extreme-value percentiles, and per-split distribution stats.
    Nothing is dropped or imputed here - this is a report only.
    """
    feats = [f for f in feature_names if f in frame.columns]
    n_rows = frame.height
    rows = []
    for name in feats:
        s = frame[name].drop_nulls()
        n_valid = s.len()
        n_null = frame[name].null_count()
        n_unique = s.n_unique()
        frac_most_common = 0.0
        if n_valid:
            counts = s.value_counts(sort=True)
            frac_most_common = float(counts["count"][0]) / n_valid
        pcts = {}
        if n_valid:
            arr = s.to_numpy().astype(np.float64)
            finite = arr[np.isfinite(arr)]
            if len(finite):
                for q in (0.01, 0.05, 0.50, 0.95, 0.99):
                    pcts[q] = float(np.quantile(finite, q))
        rows.append(
            {
                "feature": name,
                "rows": n_rows,
                "n_null": int(n_null),
                "missing_frac": float(n_null / n_rows) if n_rows else 0.0,
                "n_unique": int(n_unique),
                "frac_most_common_value": float(frac_most_common),
                "is_constant": n_valid > 0 and n_unique == 1,
                "is_near_constant": n_valid > 0 and frac_most_common >= 0.99,
                "percentiles": pcts,
            }
        )
    return {
        "n_rows": n_rows,
        "features": rows,
        "split_stats": _per_split_stats(frame, feats),
    }


def _corr_matrix(
    frame: pl.DataFrame,
    feature_names: list[str],
    spearman: bool,
) -> dict[str, Any]:
    feats = [f for f in feature_names if f in frame.columns]
    valid = frame.drop_nulls(subset=feats)
    n = len(feats)
    mat = np.full((n, n), np.nan, dtype=np.float64)
    pairs: list[dict[str, Any]] = []
    if valid.height >= 3:
        arr = valid.select(feats).to_numpy().astype(np.float64)
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    mat[i, j] = 1.0
                    continue
                x = arr[:, i]
                y = arr[:, j]
                if np.std(x) == 0.0 or np.std(y) == 0.0:
                    continue
                if spearman:
                    x = rankdata(x, method="average")
                    y = rankdata(y, method="average")
                corr = float(np.corrcoef(x, y)[0, 1])
                if np.isnan(corr):
                    continue
                mat[i, j] = mat[j, i] = corr
                pairs.append(
                    {
                        "feature_a": feats[i],
                        "feature_b": feats[j],
                        "correlation": round(corr, 6),
                    }
                )
    pairs.sort(key=lambda p: (abs(p["correlation"]), p["feature_a"], p["feature_b"]), reverse=True)
    return {
        "method": "spearman" if spearman else "pearson",
        "rows_used": int(valid.height),
        "features": feats,
        "matrix": mat,
        "pairs": pairs,
    }


def _duplicate_columns(frame: pl.DataFrame, feature_names: list[str]) -> list[dict[str, Any]]:
    feats = [f for f in feature_names if f in frame.columns]
    hashes: dict[bytes, list[str]] = {}
    for name in feats:
        arr = frame[name].to_numpy()
        h = hashlib.sha256(arr.tobytes()).digest()
        hashes.setdefault(h, []).append(name)
    groups = [sorted(v) for v in hashes.values() if len(v) > 1]
    groups.sort(key=lambda g: (g[0],))
    return [
        {"features": g, "duplicate": True, "reason": "bitwise-identical column values"}
        for g in groups
    ]


def redundancy_report(
    frame: pl.DataFrame,
    feature_names: list[str],
) -> dict[str, Any]:
    """Redundancy diagnostics on the TRAINING split (never test).

    Returns Pearson and Spearman correlation matrices plus exact-duplicate
    column detection. No feature is removed here - the report only identifies
    redundancy; incremental value is judged by the ablation experiments.
    """
    feats = [f for f in feature_names if f in frame.columns]
    return {
        "train_rows": int(frame.height),
        "pearson": _corr_matrix(frame, feats, spearman=False),
        "spearman": _corr_matrix(frame, feats, spearman=True),
        "duplicates": _duplicate_columns(frame, feats),
        "high_correlation_pairs": _high_correlation_pairs(
            _corr_matrix(frame, feats, spearman=False),
            _corr_matrix(frame, feats, spearman=True),
        ),
    }


def _high_correlation_pairs(pearson: dict, spearman: dict) -> list[dict[str, Any]]:
    """Pairs with |Pearson| or |Spearman| >= 0.95 on the training split."""
    seen: dict[tuple[str, str], dict[str, float]] = {}
    for method, report in (("pearson", pearson), ("spearman", spearman)):
        for p in report["pairs"]:
            if p["feature_a"] == p["feature_b"]:
                continue
            key = tuple(sorted((p["feature_a"], p["feature_b"])))
            seen.setdefault(key, {})[method] = p["correlation"]
    out = []
    for (a, b), vals in seen.items():
        if any(abs(v) >= 0.95 for v in vals.values() if v == v):
            out.append(
                {
                    "feature_a": a,
                    "feature_b": b,
                    "pearson": vals.get("pearson"),
                    "spearman": vals.get("spearman"),
                }
            )
    out.sort(key=lambda r: (r["feature_a"], r["feature_b"]))
    return out


__all__ = [
    "feature_columns",
    "feature_quality_report",
    "redundancy_report",
]