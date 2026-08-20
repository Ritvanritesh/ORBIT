"""Phase 10 diagnostics tests: quality report + redundancy report.

The redundancy/duplicate diagnostics must be computed on the TRAINING split
only (test-set snooping is forbidden). Nothing auto-removes a feature.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from orbit.ml.features import (
    FEATURE_NAMES,
    FEATURE_NAMES_PHASE10,
    build_feature_snapshot,
    build_feature_snapshot_phase10,
    build_phase10_feature_set_snapshot,
    build_phase10_all_feature_frame,
)
from orbit.ml.labels import build_phase9_label_snapshot
from orbit.ml.phase10_diagnostics import (
    feature_columns,
    feature_quality_report,
    redundancy_report,
)
from tests.phase9_testutils import TEST_WINDOWS, make_canonical_bars, make_events


@pytest.fixture(scope="module")
def train_frame():
    bars = make_canonical_bars()
    allf = build_phase10_all_feature_frame(bars)
    fs001 = build_feature_snapshot(bars, data_refs=["DS-000001"])
    fs003 = build_phase10_feature_set_snapshot("FS-003", allf, data_refs=["DS-000001"])
    decisions = fs001.records.select("instrument_id", "decision_time")
    ls = build_phase9_label_snapshot(bars, make_events(), [], decisions, data_refs=["DS-000001"])

    from orbit.ml.dataset import assemble_datasets

    ds = assemble_datasets(
        fs003, ls, windows=TEST_WINDOWS,
        feature_names=list(FEATURE_NAMES) + list(FEATURE_NAMES_PHASE10),
    )
    meta = ds["train"][3]
    frame = meta.join(
        fs003.records.select("instrument_id", "decision_session", *FEATURE_NAMES, *FEATURE_NAMES_PHASE10),
        on=["instrument_id", "decision_session"],
        how="inner",
    ).with_columns(pl.lit("train").alias("split"))
    return frame


def test_feature_columns_excludes_identity():
    frame = pl.DataFrame(
        {
            "instrument_id": [1],
            "decision_session": ["2010-01-04"],
            "decision_time": ["x"],
            "window_end_session": ["2009-12-31"],
            "ret_5": [0.1],
        }
    )
    assert feature_columns(frame) == ["ret_5"]


def test_quality_report_has_every_feature(train_frame):
    names = list(FEATURE_NAMES) + list(FEATURE_NAMES_PHASE10)
    rep = feature_quality_report(train_frame, names)
    assert rep["n_rows"] == train_frame.height
    assert {f["feature"] for f in rep["features"]} == set(names)
    for f in rep["features"]:
        assert f["n_null"] == 0
        assert f["is_constant"] is False
        assert 0.0 <= f["missing_frac"] <= 1.0
        assert f["percentiles"][0.5] is not None


def test_quality_report_detects_constant_column():
    frame = pl.DataFrame(
        {
            "instrument_id": [1] * 10,
            "decision_session": [str(i) for i in range(10)],
            "decision_time": ["x"] * 10,
            "window_end_session": ["x"] * 10,
            "flat": [3.0] * 10,
        }
    )
    rep = feature_quality_report(frame, ["flat"])
    row = rep["features"][0]
    assert row["is_constant"] is True
    assert row["frac_most_common_value"] == 1.0


def test_redundancy_report_is_train_only_shape(train_frame):
    rep = redundancy_report(train_frame, list(FEATURE_NAMES) + list(FEATURE_NAMES_PHASE10))
    assert rep["pearson"]["method"] == "pearson"
    assert rep["spearman"]["method"] == "spearman"
    n = len(rep["pearson"]["features"])
    assert rep["pearson"]["matrix"].shape == (n, n)
    assert rep["spearman"]["matrix"].shape == (n, n)
    for p in rep["pearson"]["pairs"]:
        assert -1.0 - 1e-9 <= p["correlation"] <= 1.0 + 1e-9


def test_highly_correlated_pairs_are_reported_not_removed(train_frame):
    rep = redundancy_report(train_frame, list(FEATURE_NAMES) + list(FEATURE_NAMES_PHASE10))
    # momentum ret features are strongly correlated by construction; the report
    # must IDENTIFY them without dropping anything.
    ids = [f["feature"] for f in rep["features"]] if "features" in rep else None
    assert rep["high_correlation_pairs"] is not None
    # nothing to auto-remove: the API surface only reports
    assert "drop" not in rep and "remove" not in rep


def test_duplicate_detection_finds_exact_copies():
    frame = pl.DataFrame(
        {
            "instrument_id": [1] * 6,
            "decision_session": [str(i) for i in range(6)],
            "decision_time": ["x"] * 6,
            "window_end_session": ["x"] * 6,
            "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "b": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "c": [6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
        }
    )
    rep = redundancy_report(frame, ["a", "b", "c"])
    dups = rep["duplicates"]
    assert len(dups) == 1
    assert set(dups[0]["features"]) == {"a", "b"}


def test_high_correlation_pairs_consistent_with_published_matrices(train_frame):
    """The high-correlation pair list must be derived from the SAME matrices
    the report publishes: every |r| >= 0.95 pair appears, and no phantom pair
    (|r| < 0.95 in both published matrices) is listed."""
    names = list(FEATURE_NAMES) + list(FEATURE_NAMES_PHASE10)
    rep = redundancy_report(train_frame, names)
    listed = {
        tuple(sorted((r["feature_a"], r["feature_b"]))): r
        for r in rep["high_correlation_pairs"]
    }
    published = {method: {} for method in ("pearson", "spearman")}
    for method in ("pearson", "spearman"):
        for p in rep[method]["pairs"]:
            published[method][tuple(sorted((p["feature_a"], p["feature_b"])))] = p["correlation"]
    for key, r in listed.items():
        assert key in published["pearson"] and key in published["spearman"]
        assert any(
            v is not None and abs(v) >= 0.95
            for v in (r.get("pearson"), r.get("spearman"))
        )
    for key in published["pearson"]:
        if any(
            v is not None and abs(v) >= 0.95
            for v in (published["pearson"][key], published["spearman"][key])
        ):
            assert key in listed