"""Phase 10 dataset assembly tests: parameterized feature columns, identical
split rows across feature sets, train-only preprocessing inputs, and
reproducibility."""

from __future__ import annotations

import polars as pl
import pytest

from orbit.ml.dataset import assemble_datasets
from orbit.ml.features import (
    FEATURE_NAMES,
    FEATURE_NAMES_PHASE10,
    build_feature_snapshot,
    build_feature_snapshot_phase10,
    build_phase10_all_feature_frame,
    build_phase10_feature_set_snapshot,
)
from orbit.ml.labels import build_phase9_label_snapshot
from tests.phase9_testutils import (
    TEST_WINDOWS,
    make_canonical_bars,
    make_events,
    weekdays,
)


@pytest.fixture(scope="module")
def snapshots():
    # Start bars in 2005 so every feature set's warm-up (200 sessions) is
    # complete well before the 2010 train window - all sets then resolve
    # IDENTICAL rows in every split (fair ablation comparison).
    sessions = weekdays(__import__("datetime").date(2005, 1, 3), 3000)
    bars = make_canonical_bars(sessions=sessions)
    allf = build_phase10_all_feature_frame(bars)
    fs001 = build_feature_snapshot(bars, data_refs=["DS-000001"])
    fs002 = build_feature_snapshot_phase10(bars, data_refs=["DS-000001"])
    fs003 = build_phase10_feature_set_snapshot("FS-003", allf, data_refs=["DS-000001"])
    decisions = fs001.records.select("instrument_id", "decision_time")
    ls = build_phase9_label_snapshot(bars, make_events(), [], decisions, data_refs=["DS-000001"])
    return {"bars": bars, "fs001": fs001, "fs002": fs002, "fs003": fs003, "ls": ls}


def test_default_feature_names_preserves_phase9_behavior(snapshots):
    """Omitting feature_names must yield exactly the FS-001 matrix (Phase 9
    behavior unchanged)."""
    ds_default = assemble_datasets(snapshots["fs001"], snapshots["ls"], windows=TEST_WINDOWS)
    ds_explicit = assemble_datasets(
        snapshots["fs001"], snapshots["ls"], windows=TEST_WINDOWS,
        feature_names=list(FEATURE_NAMES),
    )
    assert ds_default["report"]["feature_names"] == list(FEATURE_NAMES)
    for split in ("train", "val", "test"):
        assert ds_default[split][0].shape == ds_explicit[split][0].shape
        assert (ds_default[split][0] == ds_explicit[split][0]).all()


def test_all_sets_share_identical_split_rows(snapshots):
    """FS-001, FS-002, FS-003 must resolve to the SAME test/val/train row
    identities (fair ablation comparison)."""
    key_cols = ["instrument_id", "decision_time"]
    rows_by_set = {}
    for sid, snap in (("FS-001", snapshots["fs001"]), ("FS-002", snapshots["fs002"]), ("FS-003", snapshots["fs003"])):
        names = (
            list(FEATURE_NAMES)
            if sid == "FS-001"
            else (list(FEATURE_NAMES_PHASE10) if sid == "FS-002" else list(FEATURE_NAMES) + list(FEATURE_NAMES_PHASE10))
        )
        ds = assemble_datasets(snap, snapshots["ls"], windows=TEST_WINDOWS, feature_names=names)
        rows_by_set[sid] = {
            split: ds[split][3].select(key_cols).sort(key_cols).to_dicts()
            for split in ("train", "val", "test")
        }
    for split in ("train", "val", "test"):
        assert rows_by_set["FS-001"][split] == rows_by_set["FS-002"][split]
        assert rows_by_set["FS-001"][split] == rows_by_set["FS-003"][split]


def test_feature_columns_match_the_requested_set(snapshots):
    ds = assemble_datasets(
        snapshots["fs003"], snapshots["ls"], windows=TEST_WINDOWS,
        feature_names=list(FEATURE_NAMES) + list(FEATURE_NAMES_PHASE10),
    )
    assert ds["report"]["feature_names"] == list(FEATURE_NAMES) + list(FEATURE_NAMES_PHASE10)
    assert ds["train"][0].shape[1] == 23
    assert ds["val"][0].shape[1] == 23
    assert ds["test"][0].shape[1] == 23


def test_dataset_is_deterministic(snapshots):
    a = assemble_datasets(
        snapshots["fs003"], snapshots["ls"], windows=TEST_WINDOWS,
        feature_names=list(FEATURE_NAMES) + list(FEATURE_NAMES_PHASE10),
    )
    b = assemble_datasets(
        snapshots["fs003"], snapshots["ls"], windows=TEST_WINDOWS,
        feature_names=list(FEATURE_NAMES) + list(FEATURE_NAMES_PHASE10),
    )
    for split in ("train", "val", "test"):
        assert (a[split][0] == b[split][0]).all()
        assert (a[split][1] == b[split][1]).all()


def test_meta_is_row_aligned_with_matrices(snapshots):
    ds = assemble_datasets(
        snapshots["fs002"], snapshots["ls"], windows=TEST_WINDOWS,
        feature_names=list(FEATURE_NAMES_PHASE10),
    )
    for split in ("train", "val", "test"):
        X, y_reg, y_bin, meta = ds[split]
        assert X.shape[0] == meta.height == len(y_reg) == len(y_bin)
        # outcome_value ordering matches meta exactly
        assert (meta["outcome_value"].to_numpy() == y_reg).all()