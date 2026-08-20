"""Phase 9 split protocol tests: strict chronology, exact purge, integrity."""

from __future__ import annotations

import polars as pl
import pytest
from datetime import date

from orbit.ml.splits import (
    PHASE9_WINDOWS,
    assign_split,
    assert_split_integrity,
    purge_outcome_windows,
    split_summary,
    window_identity,
)
from tests.phase9_testutils import TEST_WINDOWS, build_test_datasets, make_canonical_bars, make_events


def test_protocol_windows_are_contiguous_and_ordered():
    w = PHASE9_WINDOWS
    assert w["train_start"] < w["train_end"] < w["val_start"] < w["val_end"] < w["test_start"] < w["test_end"]
    assert w["test_end"] <= date(2026, 8, 14)  # inside the data sample


def test_assign_split_maps_sessions_and_drops_outside():
    f = pl.DataFrame({"decision_session": [date(2010, 3, 1), date(2019, 6, 1), date(2024, 3, 1), date(2000, 1, 1)]})
    out = assign_split(f)
    assert out["split"].to_list() == ["train", "val", "test"]


def test_assign_split_accepts_injected_windows():
    f = pl.DataFrame({"decision_session": [date(2011, 6, 1), date(2013, 6, 1), date(2015, 6, 1)]})
    out = assign_split(f, windows=TEST_WINDOWS)
    assert out["split"].to_list() == ["train", "val", "test"]


def test_purge_drops_boundary_crossing_observations():
    f = pl.DataFrame(
        {
            "split": ["train", "train", "val", "val", "test"],
            "window_end_session": [date(2012, 6, 29), date(2012, 7, 2), date(2014, 6, 30), date(2014, 7, 2), date(2016, 6, 30)],
        }
    )
    out = purge_outcome_windows(f, windows=TEST_WINDOWS)
    assert out.height == 3
    assert "train" in out["split"].to_list()
    assert (out["split"] == "val").sum() == 1
    assert (out["split"] == "test").sum() == 1


def test_purge_never_drops_test_rows():
    f = pl.DataFrame(
        {
            "split": ["test", "test"],
            "window_end_session": [date(2016, 6, 30), date(2016, 8, 30)],
        }
    )
    out = purge_outcome_windows(f, windows=TEST_WINDOWS)
    assert out.height == 2


def test_assert_split_integrity_passes_on_clean_frame():
    f = pl.DataFrame(
        {
            "split": ["train", "val", "test"],
            "window_end_session": [date(2012, 6, 29), date(2014, 6, 30), date(2016, 6, 30)],
        }
    )
    assert_split_integrity(f, windows=TEST_WINDOWS)


def test_assert_split_integrity_detects_leakage():
    f = pl.DataFrame(
        {
            "split": ["train"],
            "window_end_session": [date(2012, 7, 3)],
        }
    )
    with pytest.raises(AssertionError, match="outcome windows reaching"):
        assert_split_integrity(f, windows=TEST_WINDOWS)


def test_assert_split_integrity_detects_unknown_split():
    f = pl.DataFrame(
        {
            "split": ["bogus"],
            "window_end_session": [date(2012, 6, 29)],
        }
    )
    with pytest.raises(AssertionError, match="unexpected split"):
        assert_split_integrity(f, windows=TEST_WINDOWS)


def test_end_to_end_dataset_splits_are_chronological_and_purged():
    _, _, ds = build_test_datasets(make_canonical_bars(), make_events())
    meta_tr = ds["train"][3]
    meta_va = ds["val"][3]
    meta_te = ds["test"][3]
    assert meta_tr["decision_session"].max() < meta_va["decision_session"].min()
    assert meta_va["decision_session"].max() < meta_te["decision_session"].min()
    assert meta_te["decision_session"].min() >= TEST_WINDOWS["test_start"]
    assert meta_te["decision_session"].max() <= TEST_WINDOWS["test_end"]
    # train outcomes never reach the val period
    assert (meta_tr["window_end_session"] < TEST_WINDOWS["val_start"]).all()
    # val outcomes never reach the test period
    assert (meta_va["window_end_session"] < TEST_WINDOWS["test_start"]).all()


def test_split_summary_counts():
    f = pl.DataFrame({"split": ["train", "train", "val"]})
    assert split_summary(f) == {"train": 2, "val": 1, "test": 0}


def test_window_identity_is_stable():
    assert window_identity() == window_identity()