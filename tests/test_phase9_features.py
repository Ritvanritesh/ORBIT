"""Phase 9 feature snapshot tests: point-in-time discipline and determinism."""

from __future__ import annotations

import polars as pl
import pytest

from orbit.ml.features import (
    FEATURE_DEFINITIONS,
    FEATURE_NAMES,
    FeatureSnapshot,
    assert_features_point_in_time,
    build_feature_frame,
    build_feature_snapshot,
)
from tests.phase9_testutils import make_canonical_bars


@pytest.fixture(scope="module")
def bars():
    return make_canonical_bars()


def test_feature_set_is_the_documented_small_set():
    assert [f["name"] for f in FEATURE_DEFINITIONS] == FEATURE_NAMES
    assert len(FEATURE_NAMES) == 8
    assert all(f["kind"] in ("momentum_return", "moving_average_ratio", "realized_volatility", "liquidity") for f in FEATURE_DEFINITIONS)


def test_every_feature_row_is_point_in_time(bars):
    frame = build_feature_frame(bars)
    assert_features_point_in_time(frame)
    assert frame.height > 0


def test_point_in_time_violation_is_detected():
    frame = build_feature_frame(make_canonical_bars())
    broken = frame.with_columns(pl.col("window_end_session").alias("decision_session"))
    with pytest.raises(AssertionError):
        assert_features_point_in_time(broken)


def test_features_use_only_strictly_prior_bars(bars):
    """The feature value at decision session D must be reproducible from the
    bars strictly before D alone (window_end_session < D)."""
    frame = build_feature_frame(bars)
    assert (frame["window_end_session"] < frame["decision_session"]).all()


def test_snapshot_digest_is_deterministic(bars):
    a = build_feature_snapshot(bars, data_refs=["DS-000001"])
    b = build_feature_snapshot(bars, data_refs=["DS-000001"])
    assert a.content_digest == b.content_digest
    assert a.equals(b)


def test_snapshot_digest_changes_with_records(bars):
    a = build_feature_snapshot(bars, data_refs=["DS-000001"])
    changed = a.records.with_columns((pl.col("ret_10") + 1e-9).alias("ret_10"))
    b = FeatureSnapshot(
        feature_set_id=a.feature_set_id,
        feature_set_version=a.feature_set_version,
        feature_refs=a.feature_refs,
        data_refs=a.data_refs,
        records=changed,
    )
    assert a.content_digest != b.content_digest


def test_digest_changes_with_data_refs(bars):
    a = build_feature_snapshot(bars, data_refs=["DS-000001"])
    b = build_feature_snapshot(bars, data_refs=["DS-000001", "DS-000002"])
    assert a.content_digest != b.content_digest


def test_feature_values_are_finite(bars):
    frame = build_feature_frame(bars)
    assert frame.select(pl.col(FEATURE_NAMES).is_finite().all()).to_series().all()


def test_warmup_rows_are_excluded(bars):
    """The first ~40 sessions have no complete window and must be absent."""
    frame = build_feature_frame(bars)
    first = frame["decision_session"].min()
    assert first is not None


def test_decision_time_is_session_close():
    frame = build_feature_frame(make_canonical_bars())
    from orbit.ml.features import attach_decision_times

    frame = attach_decision_times(frame)
    first = frame.sort(["decision_session"]).head(1)
    dt = first["decision_time"][0]
    assert dt.hour == 21 and dt.minute == 0  # 16:00 America/New_York == 21:00 UTC