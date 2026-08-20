"""Training/validation/test matrix assembly for the Phase 9 benchmark.

`assemble_datasets` joins the FS-001 feature snapshot with the LAB-004 label
snapshot at the decision-row level (exact join on instrument_id +
decision_time), assigns splits chronologically, purges outcome windows at the
split boundaries, and applies the documented data policy:

  - only rows the label engine marks AVAILABLE are used (explicit
    unavailable rows are excluded; their reasons are counted and reported)
  - rows with any NaN feature are excluded (incomplete warm-up history);
    the count is reported
  - all exclusions are deterministic and auditable via `dataset_report`
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from orbit.ml.features import FEATURE_NAMES
from orbit.ml.splits import assign_split, assert_split_integrity, purge_outcome_windows


def _join_features_labels(
    feature_snapshot: Any, label_snapshot: Any
) -> pl.DataFrame:
    features = feature_snapshot.records
    labels = label_snapshot.records
    joined = labels.join(
        features,
        on=["instrument_id", "decision_time"],
        how="inner",
    )
    return joined


def assemble_datasets(
    feature_snapshot: Any,
    label_snapshot: Any,
    windows: dict | None = None,
    feature_names: list[str] | None = None,
) -> dict[str, Any]:
    """Assemble (train, val, test) matrices with metadata and a report.

    Returns {"train": (X, y_reg, y_bin, meta), "val": ..., "test": ...,
    "report": {...}}. meta is a polars frame with instrument_id,
    decision_session, decision_time, split, outcome_value, window_end_session
    (row-ordered exactly like X/y). `windows` defaults to the Phase 9
    protocol; tests inject tighter windows for speed. `feature_names`
    defaults to the FS-001 set (Phase 9 behavior unchanged); Phase 10 passes
    the member columns of the feature snapshot it assembled.
    """
    names = list(feature_names or FEATURE_NAMES)
    joined = _join_features_labels(feature_snapshot, label_snapshot)
    total = joined.height

    available = joined.filter(pl.col("outcome_status") == "available")
    unavailable = total - available.height
    unavailable_reasons = (
        joined.filter(pl.col("outcome_status") == "unavailable")
        .group_by("unavailable_reason")
        .agg(pl.len().alias("n"))
        .sort("unavailable_reason")
        .to_dicts()
    )

    split_frame = assign_split(available, "decision_session", windows)
    split_frame = purge_outcome_windows(split_frame, windows=windows)
    assert_split_integrity(split_frame, windows=windows)

    complete = split_frame.drop_nulls(subset=names)
    incomplete = split_frame.height - complete.height

    X = complete.select(names).to_numpy()
    y_reg = complete["outcome_value"].to_numpy()
    meta = complete.select(
        "instrument_id",
        "decision_session",
        "decision_time",
        "split",
        "outcome_value",
        "window_end_session",
    )
    y_bin = (y_reg > 0.0).astype(np.int64)

    parts: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, pl.DataFrame]] = {}
    for split_name in ("train", "val", "test"):
        mask = meta["split"].to_numpy() == split_name
        parts[split_name] = (
            X[mask],
            y_reg[mask],
            y_bin[mask],
            meta.filter(pl.col("split") == split_name),
        )

    report = {
        "total_rows": int(total),
        "available_rows": int(available.height),
        "unavailable_rows": int(unavailable),
        "unavailable_reasons": unavailable_reasons,
        "post_purge_rows": int(split_frame.height),
        "purged_rows": int(available.height - split_frame.height),
        "nan_feature_rows_dropped": int(incomplete),
        "train_rows": int(parts["train"][0].shape[0]),
        "val_rows": int(parts["val"][0].shape[0]),
        "test_rows": int(parts["test"][0].shape[0]),
        "feature_names": list(names),
        "feature_set_id": getattr(feature_snapshot, "feature_set_id", None),
        "feature_set_version": getattr(feature_snapshot, "feature_set_version", None),
    }
    return {
        "train": parts["train"],
        "val": parts["val"],
        "test": parts["test"],
        "report": report,
        "split_frame": complete.select(
            "instrument_id",
            "decision_session",
            "split",
            "outcome_value",
            "window_end_session",
        ),
    }


__all__ = ["assemble_datasets"]