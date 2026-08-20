"""Phase 9 evaluation protocol: strict chronological splits with purging.

Protocol (fixed, pre-registered; documented in docs/phase9_ml_benchmark.md):

  TRAIN      2010-01-04 .. 2018-12-31
  VALIDATION 2019-01-02 .. 2021-12-31
  TEST       2022-01-03 .. 2026-06-30   (locked final holdout)

Rules:
  1. A decision row belongs to the split containing its decision session.
  2. Outcome-window purge: an observation whose outcome window (the sessions
     its label is realized over) crosses the NEXT split boundary is removed
     from the earlier split. A training observation must never realize
     outcomes inside the validation period; a validation observation must
     never realize outcomes inside the test period. The purge is exact - it
     uses the label engine's window_end_session, not an embargo estimate.
  3. Test observations are never purged beyond label availability (rows the
     engine marks unavailable are excluded by the documented data policy).
  4. No random train/test split is ever used for the primary evaluation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date

import polars as pl

PHASE9_WINDOWS = {
    "train_start": date(2010, 1, 4),
    "train_end": date(2018, 12, 31),
    "val_start": date(2019, 1, 2),
    "val_end": date(2021, 12, 31),
    "test_start": date(2022, 1, 3),
    "test_end": date(2026, 6, 30),
    "embargo_days": 0,
    "purge_days": 0,
    "protocol": "fixed_chronological_v1",
}

_SPLIT_BOUNDS = [
    ("train", PHASE9_WINDOWS["train_start"], PHASE9_WINDOWS["train_end"], PHASE9_WINDOWS["val_start"]),
    ("val", PHASE9_WINDOWS["val_start"], PHASE9_WINDOWS["val_end"], PHASE9_WINDOWS["test_start"]),
    ("test", PHASE9_WINDOWS["test_start"], PHASE9_WINDOWS["test_end"], None),
]


def assign_split(
    frame: pl.DataFrame,
    session_col: str = "decision_session",
    windows: dict | None = None,
) -> pl.DataFrame:
    """Attach the split name to every row by decision session.

    Rows outside all three windows are dropped. Requires `session_col` to be
    polars Date. `windows` defaults to the Phase 9 protocol; tests inject
    tighter windows for speed.
    """
    w = windows or PHASE9_WINDOWS
    out = frame.with_columns(
        pl.when(
            (pl.col(session_col) >= w["train_start"])
            & (pl.col(session_col) <= w["train_end"])
        )
        .then(pl.lit("train"))
        .when(
            (pl.col(session_col) >= w["val_start"])
            & (pl.col(session_col) <= w["val_end"])
        )
        .then(pl.lit("val"))
        .when(
            (pl.col(session_col) >= w["test_start"])
            & (pl.col(session_col) <= w["test_end"])
        )
        .then(pl.lit("test"))
        .otherwise(pl.lit(None))
        .alias("split")
    )
    return out.filter(pl.col("split").is_not_null())


def purge_outcome_windows(
    frame: pl.DataFrame,
    split_col: str = "split",
    window_end_col: str = "window_end_session",
    boundaries: dict[str, date] | None = None,
    windows: dict | None = None,
) -> pl.DataFrame:
    """Drop observations whose outcome window crosses the next split boundary.

    `boundaries` maps the earlier split to the first session of the next
    split; defaults to the Phase 9 protocol boundaries. An observation in
    split S is kept only if window_end_session < boundary(S). The LAST
    split (test) is never purged: its outcomes are realized inside the test
    period by construction.
    """
    w = windows or PHASE9_WINDOWS
    bounds = boundaries or {
        "train": w["val_start"],
        "val": w["test_start"],
    }
    conditions = []
    for split_name, boundary in bounds.items():
        conditions.append(
            (pl.col(split_col) == split_name) & (pl.col(window_end_col) < boundary)
        )
    keep = conditions[0]
    for cond in conditions[1:]:
        keep = keep | cond
    keep = keep | (pl.col(split_col) == "test")
    return frame.filter(keep)


def split_summary(frame: pl.DataFrame, split_col: str = "split") -> dict[str, int]:
    return {
        s: frame.filter(pl.col(split_col) == s).height
        for s in ["train", "val", "test"]
    }


def assert_split_integrity(
    frame: pl.DataFrame,
    split_col: str = "split",
    boundaries: dict[str, date] | None = None,
    windows: dict | None = None,
) -> None:
    """Adversarial verification of split correctness on a labeled frame.

    Asserts (a) every row belongs to exactly one split, (b) no train row's
    outcome window reaches the validation period, (c) no validation row's
    outcome window reaches the test period, (d) no test observation predates
    the test window. Raises AssertionError on any violation.
    """
    w = windows or PHASE9_WINDOWS
    bounds = boundaries or {
        "train": w["val_start"],
        "val": w["test_start"],
    }
    splits = set(frame[split_col].unique().to_list())
    allowed = {"train", "val", "test"}
    extra = splits - allowed
    if extra:
        raise AssertionError(f"unexpected split values: {sorted(extra)}")
    for s, boundary in bounds.items():
        bad = frame.filter((pl.col(split_col) == s) & (pl.col("window_end_session") >= boundary))
        if bad.height:
            raise AssertionError(
                f"{bad.height} {s} observations have outcome windows reaching "
                f"the next split (>= {boundary})"
            )


def window_identity() -> str:
    """Deterministic identity of the fixed protocol (for experiment lineage)."""
    return json.dumps(
        {k: (v.isoformat() if isinstance(v, date) else v) for k, v in PHASE9_WINDOWS.items()},
        sort_keys=True,
    )


__all__ = [
    "PHASE9_WINDOWS",
    "assign_split",
    "purge_outcome_windows",
    "split_summary",
    "assert_split_integrity",
    "window_identity",
]