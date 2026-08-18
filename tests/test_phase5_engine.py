"""Phase 5 engine tests: horizon semantics, strict timing boundaries,
missing-data reasons, delisting, corporate-action guards, POST_EVENT
anchors, and the canonical-schema guards - everything except the
hand-calculated golden values (test_phase5_golden.py).
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import polars as pl
import pytest

from orbit.labels import (
    AnchorMode,
    LABEL_OUTPUT_COLUMNS,
    LabelContract,
    LabelEngine,
    ReturnConvention,
    UnavailableReason,
    empty_label_frame,
)


def _bars(sessions, closes, inst="INS-000001", high=None, low=None):
    return pl.DataFrame(
        {
            "instrument_id": [inst] * len(sessions),
            "trade_date": sessions,
            "open": closes,
            "high": high or [c + 1 for c in closes],
            "low": low or [c - 1 for c in closes],
            "close": closes,
            "volume": [1000] * len(sessions),
        }
    )


def _sessions(n: int, start: date = date(2020, 1, 6)) -> list[date]:
    out = []
    d = start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def _contract(**kw) -> LabelContract:
    base = dict(
        label_id="LAB-001", version="v1", target_type="forward_return",
        horizon=1, anchor_mode=AnchorMode.DECISION_INSTANT,
        return_convention=ReturnConvention.SIMPLE_PRICE_RETURN,
        formula="engine test contract",
    )
    base.update(kw)
    return LabelContract(**base)


WINTER = timezone.utc


# ------------------------------------------------------------ guards

def test_engine_requires_canonical_bar_columns():
    bad = pl.DataFrame({"instrument_id": ["x"], "trade_date": [date(2020, 1, 6)]})
    with pytest.raises(ValueError, match="canonical normalized bar columns"):
        LabelEngine(bad)


def test_engine_rejects_duplicate_session_bars():
    bars = _bars([date(2020, 1, 6), date(2020, 1, 6)], [100.0, 101.0])
    with pytest.raises(ValueError, match="duplicate"):
        LabelEngine(bars)


def test_engine_rejects_non_canonical_price_basis():
    bars = _bars(_sessions(3), [100.0, 101.0, 102.0]).with_columns(
        pl.lit("raw_unadjusted").alias("adjustment")
    )
    with pytest.raises(ValueError, match="split-continuous"):
        LabelEngine(bars)


def test_total_return_contract_requires_events_artifact():
    bars = _bars(_sessions(3), [100.0, 101.0, 102.0])
    c = _contract(return_convention=ReturnConvention.SIMPLE_TOTAL_RETURN)
    with pytest.raises(ValueError, match="corporate-actions events"):
        LabelEngine(bars).compute_one(c, "INS-000001", datetime(2020, 1, 6, 21, 0, 1, tzinfo=WINTER))


def test_compute_requires_a_registered_contract():
    bars = _bars(_sessions(3), [100.0, 101.0, 102.0])
    with pytest.raises(TypeError, match="LabelContract"):
        LabelEngine(bars).compute({"target_type": "forward_return"}, [])


def test_events_must_have_canonical_columns():
    bars = _bars(_sessions(3), [100.0, 101.0, 102.0])
    events = pl.DataFrame({"kind": ["dividends"]})
    with pytest.raises(ValueError, match="events artifact"):
        LabelEngine(bars, events=events)


def test_empty_label_frame_has_canonical_schema():
    f = empty_label_frame()
    assert f.height == 0
    assert f.columns == [c for c, _ in LABEL_OUTPUT_COLUMNS]


# -------------------------------------------------------- horizon semantics

def test_horizon_positive_and_short_window_unavailable():
    bars = _bars(_sessions(3), [100.0, 101.0, 102.0])
    eng = LabelEngine(bars)
    with pytest.raises(ValueError, match="horizon"):
        eng.outcome_window("INS-000001", date(2020, 1, 6), 0)
    row = eng.compute_one(_contract(horizon=5), "INS-000001",
                          datetime(2020, 1, 6, 21, 0, 1, tzinfo=WINTER))
    assert row["outcome_status"] == "unavailable"
    assert row["unavailable_reason"] == "insufficient_future_data"


def test_horizon_gaps_are_never_sessions():
    # a 2-session window across a long gap still lands on the next two bars
    sessions = [date(2020, 1, 6), date(2020, 2, 3), date(2020, 2, 4)]
    bars = _bars(sessions, [100.0, 101.0, 102.0])
    row = LabelEngine(bars).compute_one(_contract(horizon=2), "INS-000001",
                                        datetime(2020, 1, 6, 21, 0, 1, tzinfo=WINTER))
    assert row["outcome_session"] == date(2020, 2, 4)
    assert row["sessions_available"] == 2


# ---------------------------------------------------- strict timing boundaries

def test_decision_after_close_uses_that_session():
    sessions = _sessions(4)
    bars = _bars(sessions, [100.0, 101.0, 102.0, 103.0])
    eng = LabelEngine(bars)
    # winter close is 21:00 UTC; exactly at the close the bar is unavailable
    at = datetime(2020, 1, 8, 21, 0, 0, tzinfo=WINTER)
    assert eng.entry_bar("INS-000001", at)["trade_date"] == date(2020, 1, 7)
    after = datetime(2020, 1, 8, 21, 0, 0, 1, tzinfo=WINTER)
    assert eng.entry_bar("INS-000001", after)["trade_date"] == date(2020, 1, 8)


def test_summer_dst_close_is_20_utc():
    sessions = [date(2020, 7, 6), date(2020, 7, 7), date(2020, 7, 8)]
    bars = _bars(sessions, [100.0, 101.0, 102.0])
    eng = LabelEngine(bars)
    # 16:00 America/New_York == 20:00 UTC in July
    at = datetime(2020, 7, 7, 20, 0, 0, tzinfo=WINTER)
    assert eng.entry_bar("INS-000001", at)["trade_date"] == date(2020, 7, 6)
    after = datetime(2020, 7, 7, 20, 0, 0, 1, tzinfo=WINTER)
    assert eng.entry_bar("INS-000001", after)["trade_date"] == date(2020, 7, 7)


def test_decision_on_non_trading_day_uses_last_completed_session():
    sessions = [date(2020, 1, 6), date(2020, 1, 7), date(2020, 1, 8)]
    bars = _bars(sessions, [100.0, 101.0, 102.0])
    eng = LabelEngine(bars)
    # Saturday 2020-01-11
    row = eng.compute_one(_contract(horizon=1), "INS-000001",
                          datetime(2020, 1, 11, 12, 0, tzinfo=WINTER))
    assert row["entry_session"] == date(2020, 1, 8)


def test_decision_before_first_session_is_no_entry_bar():
    bars = _bars(_sessions(3), [100.0, 101.0, 102.0])
    row = LabelEngine(bars).compute_one(
        _contract(horizon=1), "INS-000001",
        datetime(2020, 1, 5, 12, 0, tzinfo=WINTER),
    )
    assert row["outcome_status"] == "unavailable"
    assert row["unavailable_reason"] == "no_entry_bar"


# --------------------------------------------------------------- POST_EVENT

def test_post_event_entry_is_first_session_after_anchor():
    sessions = _sessions(5)
    bars = _bars(sessions, [100.0 + i for i in range(5)])
    eng = LabelEngine(bars)
    anchor = datetime(2020, 1, 8, 0, 0, 0, tzinfo=WINTER)
    entry = eng.entry_bar("INS-000001", None, anchor_instant=anchor,
                          anchor_mode=AnchorMode.POST_EVENT)
    assert entry["trade_date"] == date(2020, 1, 8)
    # anchor exactly at a session close -> the next session
    anchor2 = datetime(2020, 1, 8, 21, 0, 0, tzinfo=WINTER)
    entry2 = eng.entry_bar("INS-000001", None, anchor_instant=anchor2,
                           anchor_mode=AnchorMode.POST_EVENT)
    assert entry2["trade_date"] == date(2020, 1, 9)
    # anchor after the last session -> no entry bar
    anchor3 = datetime(2020, 1, 20, 21, 0, 0, tzinfo=WINTER)
    assert eng.entry_bar("INS-000001", None, anchor_instant=anchor3,
                         anchor_mode=AnchorMode.POST_EVENT) is None


def test_post_event_without_anchor_is_missing_anchor():
    bars = _bars(_sessions(5), [100.0 + i for i in range(5)])
    c = _contract(anchor_mode=AnchorMode.POST_EVENT)
    row = LabelEngine(bars).compute_one(
        c, "INS-000001", datetime(2020, 1, 8, 21, 0, 1, tzinfo=WINTER),
    )
    assert row["outcome_status"] == "unavailable"
    assert row["unavailable_reason"] == "missing_anchor"


# ------------------------------------------------------------ missing data

def test_missing_entry_price():
    sessions = _sessions(4)
    bars = _bars(sessions, [100.0, 101.0, 102.0, 103.0]).with_columns(
        pl.when(pl.col("trade_date") == date(2020, 1, 7))
        .then(None)
        .otherwise(pl.col("close"))
        .alias("close")
    )
    row = LabelEngine(bars).compute_one(_contract(horizon=1), "INS-000001",
                                        datetime(2020, 1, 7, 21, 0, 1, tzinfo=WINTER))
    assert row["unavailable_reason"] == "missing_entry_price"


def test_missing_outcome_price():
    sessions = _sessions(4)
    bars = _bars(sessions, [100.0, 101.0, 102.0, 103.0]).with_columns(
        pl.when(pl.col("trade_date") == date(2020, 1, 8))
        .then(None)
        .otherwise(pl.col("close"))
        .alias("close")
    )
    row = LabelEngine(bars).compute_one(_contract(horizon=1), "INS-000001",
                                        datetime(2020, 1, 7, 21, 0, 1, tzinfo=WINTER))
    assert row["unavailable_reason"] == "missing_outcome_price"


def test_missing_window_price_mid_window():
    sessions = _sessions(6)
    bars = _bars(sessions, [100.0 + i for i in range(6)]).with_columns(
        pl.when(pl.col("trade_date") == date(2020, 1, 9))
        .then(None)
        .otherwise(pl.col("close"))
        .alias("close")
    )
    row = LabelEngine(bars).compute_one(_contract(horizon=3), "INS-000001",
                                        datetime(2020, 1, 7, 21, 0, 1, tzinfo=WINTER))
    assert row["unavailable_reason"] == "missing_window_price"


def test_benchmark_unavailable_reason():
    sessions = _sessions(6)
    bars = _bars(sessions, [100.0 + i for i in range(6)], inst="INS-000001")
    c = _contract(target_type="excess_return", benchmark="SPY")
    row = LabelEngine(bars).compute_one(c, "INS-000001",
                                        datetime(2020, 1, 6, 21, 0, 1, tzinfo=WINTER))
    assert row["outcome_status"] == "unavailable"
    assert row["unavailable_reason"] == "benchmark_unavailable"


def test_corporate_action_incomplete_blocks_return():
    sessions = _sessions(4)
    bars = _bars(sessions, [100.0, 101.0, 102.0, 103.0])
    events = pl.DataFrame(
        {
            "instrument_id": ["INS-000001"],
            "kind": ["dividends"],
            "ts": [datetime(2020, 1, 7, 12, 0, tzinfo=timezone.utc)],
            "ratio": [None],  # incomplete corporate-action record
        }
    )
    row = LabelEngine(bars, events=events).compute_one(
        _contract(return_convention=ReturnConvention.SIMPLE_TOTAL_RETURN),
        "INS-000001", datetime(2020, 1, 6, 21, 0, 1, tzinfo=WINTER),
    )
    assert row["unavailable_reason"] == "corporate_action_data_incomplete"


def test_unknown_instrument_is_no_entry_bar():
    bars = _bars(_sessions(4), [100.0, 101.0, 102.0, 103.0])
    row = LabelEngine(bars).compute_one(_contract(horizon=1), "INS-000002",
                                        datetime(2020, 1, 7, 21, 0, 1, tzinfo=WINTER))
    assert row["unavailable_reason"] == "no_entry_bar"


# --------------------------------------------------------------- delisting

def test_delisting_classification_uses_instrument_master():
    sessions = _sessions(5)
    bars = _bars(sessions, [100.0 + i for i in range(5)])
    instruments = pl.DataFrame(
        {
            "instrument_id": ["INS-000001"],
            "name": ["Defunct Co"],
            "delisting_date": [date(2020, 1, 13)],
        }
    )
    eng = LabelEngine(bars, instruments=instruments)
    # last bar (01-13) <= delisting date -> DELISTED
    row = eng.compute_one(_contract(horizon=3), "INS-000001",
                          datetime(2020, 1, 9, 21, 0, 1, tzinfo=WINTER))
    assert row["unavailable_reason"] == "delisted"
    # an instrument master without delisting info -> data shortfall
    instruments2 = instruments.with_columns(pl.lit(None).alias("delisting_date"))
    row2 = LabelEngine(bars, instruments=instruments2).compute_one(
        _contract(horizon=3), "INS-000001",
        datetime(2020, 1, 9, 21, 0, 1, tzinfo=WINTER),
    )
    assert row2["unavailable_reason"] == "insufficient_future_data"
    # bars extending PAST the delisting date contradict the delisting
    # record -> the reason is a data shortfall, not a delisting
    instruments3 = instruments.with_columns(
        pl.lit(date(2020, 1, 9)).alias("delisting_date")
    )
    row3 = LabelEngine(bars, instruments=instruments3).compute_one(
        _contract(horizon=3), "INS-000001",
        datetime(2020, 1, 9, 21, 0, 1, tzinfo=WINTER),
    )
    assert row3["unavailable_reason"] == "insufficient_future_data"


# ------------------------------------------------------------- decisions API

def test_decisions_frame_and_list_agree():
    sessions = _sessions(4)
    bars = _bars(sessions, [100.0, 101.0, 102.0, 103.0])
    eng = LabelEngine(bars)
    c = _contract(horizon=1)
    t1 = datetime(2020, 1, 6, 21, 0, 1, tzinfo=WINTER)
    t2 = datetime(2020, 1, 7, 21, 0, 1, tzinfo=WINTER)
    from_list = eng.compute(c, [{"instrument_id": "INS-000001", "decision_time": t1},
                                {"instrument_id": "INS-000001", "decision_time": t2}])
    from_frame = eng.compute(
        c, pl.DataFrame({"instrument_id": ["INS-000001", "INS-000001"],
                         "decision_time": [t1, t2]})
    )
    assert from_list.height == 2
    assert from_frame.height == 2
    assert from_list.select("decision_id", "outcome_value").to_dicts() == \
        from_frame.select("decision_id", "outcome_value").to_dicts()


def test_decision_id_default_is_deterministic():
    sessions = _sessions(4)
    bars = _bars(sessions, [100.0, 101.0, 102.0, 103.0])
    row = LabelEngine(bars).compute_one(
        _contract(horizon=1), "INS-000001",
        datetime(2020, 1, 7, 21, 0, 1, tzinfo=WINTER),
    )
    assert row["decision_id"] == "INS-000001|2020-01-07T21:00:01"


def test_compute_output_sorted_by_instrument_then_time():
    sessions = _sessions(4)
    bars = _bars(sessions, [100.0, 101.0, 102.0, 103.0])
    t1 = datetime(2020, 1, 7, 21, 0, 1, tzinfo=WINTER)
    t2 = datetime(2020, 1, 6, 21, 0, 1, tzinfo=WINTER)
    frame = LabelEngine(bars).compute(
        _contract(horizon=1),
        [{"instrument_id": "INS-000001", "decision_time": t1},
         {"instrument_id": "INS-000001", "decision_time": t2}],
    )
    # stored decision_time is normalized naive UTC (Phase 4 convention)
    assert frame["decision_time"].to_list() == [
        t2.replace(tzinfo=None), t1.replace(tzinfo=None),
    ]


# ----------------------------------------------- non-finite / degenerate prices

def test_nan_close_is_missing_price_never_a_value():
    # a NaN close inside the window must make the outcome explicitly
    # unavailable - a 'nan' outcome_value with status available would
    # silently corrupt every downstream statistic
    bars = _bars(_sessions(3), [100.0, float("nan"), 110.0])
    row = LabelEngine(bars).compute_one(
        _contract(horizon=2), "INS-000001",
        datetime(2020, 1, 6, 21, 0, 1, tzinfo=WINTER),
    )
    assert row["outcome_status"] == "unavailable"
    assert row["unavailable_reason"] == "missing_window_price"
    assert row["outcome_value"] is None


def test_zero_close_is_missing_price_not_a_minus_100_percent_return():
    # 0/100 - 1 = -1.0 would fabricate a -100% crash that never happened
    bars = _bars(_sessions(2), [100.0, 0.0])
    row = LabelEngine(bars).compute_one(
        _contract(horizon=1), "INS-000001",
        datetime(2020, 1, 6, 21, 0, 1, tzinfo=WINTER),
    )
    assert row["outcome_status"] == "unavailable"
    assert row["unavailable_reason"] == "missing_outcome_price"


def test_infinite_close_is_missing_price():
    bars = _bars(_sessions(2), [100.0, float("inf")])
    row = LabelEngine(bars).compute_one(
        _contract(horizon=1), "INS-000001",
        datetime(2020, 1, 6, 21, 0, 1, tzinfo=WINTER),
    )
    assert row["unavailable_reason"] == "missing_outcome_price"
    assert row["outcome_value"] is None


def test_non_finite_entry_close_is_missing_entry_price():
    bars = _bars(_sessions(2), [float("nan"), 100.0])
    row = LabelEngine(bars).compute_one(
        _contract(horizon=1), "INS-000001",
        datetime(2020, 1, 6, 21, 0, 1, tzinfo=WINTER),
    )
    assert row["unavailable_reason"] == "missing_entry_price"


def test_non_finite_benchmark_close_is_benchmark_unavailable():
    sessions = _sessions(3)
    bars = pl.concat([
        _bars(sessions, [100.0, 101.0, 102.0], inst="INS-000001"),
        _bars(sessions, [300.0, float("nan"), 310.0], inst="SPY"),
    ])
    row = LabelEngine(bars).compute_one(
        _contract(target_type="excess_return", benchmark="SPY", horizon=1),
        "INS-000001", datetime(2020, 1, 6, 21, 0, 1, tzinfo=WINTER),
    )
    assert row["unavailable_reason"] == "benchmark_unavailable"
    assert "defective" in row["outcome_detail"]


# ------------------------------------------------ corporate-action robustness

def test_event_ts_accepts_iso_strings_and_aware_instants():
    # a string ts used to crash _event_session; an aware ts outside UTC
    # used to be re-labeled as UTC and could land on the wrong day
    sessions = _sessions(3)
    bars = _bars(sessions, [100.0, 110.0, 121.0])
    for ts, expect in [
        ("2020-01-07T12:00:00Z", 0.11),       # string, naive UTC
        (datetime(2020, 1, 7, 12, 0), 0.11),  # naive datetime
    ]:
        events = pl.DataFrame({
            "instrument_id": ["INS-000001"], "kind": ["dividends"],
            "ts": [ts], "ratio": [1.0],
        })
        row = LabelEngine(bars, events=events).compute_one(
            _contract(horizon=1,
                      return_convention=ReturnConvention.SIMPLE_TOTAL_RETURN),
            "INS-000001", datetime(2020, 1, 6, 21, 0, 1, tzinfo=WINTER),
        )
        assert abs(row["outcome_value"] - expect) < 1e-9, row["outcome_value"]
    # 2020-01-08T02:00:00-05:00 = 07:00 UTC = 01-08 in New York; the old
    # re-label-as-UTC behavior would have read the ex-date as 01-07 and
    # counted the dividend a day early
    aware = datetime(2020, 1, 8, 2, 0, tzinfo=timezone(timedelta(hours=-5)))
    events2 = pl.DataFrame({
        "instrument_id": ["INS-000001"], "kind": ["dividends"],
        "ts": [aware], "ratio": [1.0],
    })
    row2 = LabelEngine(bars, events=events2).compute_one(
        _contract(horizon=2,
                  return_convention=ReturnConvention.SIMPLE_TOTAL_RETURN),
        "INS-000001", datetime(2020, 1, 6, 21, 0, 1, tzinfo=WINTER),
    )
    assert abs(row2["outcome_value"] - 0.22) < 1e-9, row2["outcome_value"]


def test_nan_event_ratio_marks_instrument_incomplete():
    # NaN <= 0 is False, so the old `ratio <= 0` guard silently accepted a
    # NaN ratio and poisoned the dividend/factor math
    bars = _bars(_sessions(2), [100.0, 110.0])
    events = pl.DataFrame({
        "instrument_id": ["INS-000001"], "kind": ["dividends"],
        "ts": [datetime(2020, 1, 7, 12, 0)], "ratio": [float("nan")],
    })
    row = LabelEngine(bars, events=events).compute_one(
        _contract(horizon=1), "INS-000001",
        datetime(2020, 1, 6, 21, 0, 1, tzinfo=WINTER),
    )
    assert row["unavailable_reason"] == "corporate_action_data_incomplete"
    events2 = events.with_columns(pl.lit(float("inf")).alias("ratio"))
    row2 = LabelEngine(bars, events=events2).compute_one(
        _contract(horizon=1), "INS-000001",
        datetime(2020, 1, 6, 21, 0, 1, tzinfo=WINTER),
    )
    assert row2["unavailable_reason"] == "corporate_action_data_incomplete"


# ------------------------------------------------- instrument master robustness

def test_delisting_date_as_string_is_normalized():
    # a Utf8 delisting_date in an instruments frame used to crash the
    # date <= str comparison on the shortfall path
    bars = _bars(_sessions(5), [100.0 + i for i in range(5)])
    instruments = pl.DataFrame({
        "instrument_id": ["INS-000001"],
        "delisting_date": ["2020-01-13"],
    })
    row = LabelEngine(bars, instruments=instruments).compute_one(
        _contract(horizon=3), "INS-000001",
        datetime(2020, 1, 9, 21, 0, 1, tzinfo=WINTER),
    )
    assert row["unavailable_reason"] == "delisted"


# ------------------------------------------------------ benchmark diagnostics

def test_benchmark_unavailable_detail_names_the_cause():
    sessions = _sessions(3)
    bars = _bars(sessions, [100.0, 101.0, 102.0])
    row = LabelEngine(bars).compute_one(
        _contract(target_type="excess_return", benchmark="SPY", horizon=1),
        "INS-000001", datetime(2020, 1, 6, 21, 0, 1, tzinfo=WINTER),
    )
    assert "no bars for benchmark" in row["outcome_detail"]
    # benchmark window shortfall is a distinct cause from a missing series
    sessions2 = _sessions(2)
    bars2 = pl.concat([
        _bars(sessions, [100.0, 101.0, 102.0], inst="INS-000001"),
        _bars(sessions2, [300.0, 301.0], inst="SPY"),
    ])
    row2 = LabelEngine(bars2).compute_one(
        _contract(target_type="excess_return", benchmark="SPY", horizon=2),
        "INS-000001", datetime(2020, 1, 6, 21, 0, 1, tzinfo=WINTER),
    )
    assert "only 1 of 2 sessions" in row2["outcome_detail"]


def test_entry_bar_requires_decision_time_for_decision_anchor():
    bars = _bars(_sessions(2), [100.0, 101.0])
    with pytest.raises(ValueError, match="decision_time is required"):
        LabelEngine(bars).entry_bar("INS-000001", None)


def test_decision_instant_rows_never_record_an_anchor_instant():
    # a supplied anchor_instant is IGNORED for DECISION_INSTANT contracts;
    # recording it on the row would misdescribe what was computed
    bars = _bars(_sessions(2), [100.0, 101.0])
    row = LabelEngine(bars).compute_one(
        _contract(horizon=1), "INS-000001",
        datetime(2020, 1, 6, 21, 0, 1, tzinfo=WINTER),
        anchor_instant=datetime(2020, 1, 6, 12, 0, tzinfo=WINTER),
    )
    assert row["anchor_instant"] is None
    assert row["unavailable_reason"] is None  # computed normally


def test_price_basis_column_is_honest_about_the_audit_closes():
    # without an events artifact the *_as_published closes are the
    # provider's split-adjusted values; the row must SAY so instead of
    # silently presenting them as historical truth (Phase 4 convention)
    bars = _bars(_sessions(2), [100.0, 101.0])
    row = LabelEngine(bars).compute_one(
        _contract(horizon=1), "INS-000001",
        datetime(2020, 1, 6, 21, 0, 1, tzinfo=WINTER),
    )
    assert row["price_basis"] == "provider_split_adjusted"
    events = pl.DataFrame({
        "instrument_id": ["INS-000001"], "kind": ["splits"],
        "ts": [datetime(2020, 1, 8, 12, 0)], "ratio": [2.0],
    })
    row2 = LabelEngine(bars, events=events).compute_one(
        _contract(horizon=1), "INS-000001",
        datetime(2020, 1, 6, 21, 0, 1, tzinfo=WINTER),
    )
    assert row2["price_basis"] == "as_published"


# -------------------------------------------------------------- overlap info

def test_overlap_requires_available_labels_only():
    sessions = _sessions(8)
    bars = _bars(sessions, [100.0 + i for i in range(8)])
    eng = LabelEngine(bars)
    t = datetime(2020, 1, 6, 21, 0, 1, tzinfo=WINTER)
    frame = eng.compute(_contract(horizon=5), [
        {"instrument_id": "INS-000001", "decision_time": t},
        {"instrument_id": "INS-000001",
         "decision_time": datetime(2020, 1, 10, 21, 0, 1, tzinfo=WINTER)},
    ])
    # the second decision has only 2 sessions left -> unavailable -> no pair
    from orbit.labels import overlapping_pairs
    pairs = overlapping_pairs(frame, sessions_by_instrument={
        "INS-000001": eng.instrument_sessions("INS-000001")})
    assert pairs == []


def test_overlap_ignores_different_instruments():
    sessions = _sessions(6)
    bars = pl.concat([
        _bars(sessions, [100.0 + i for i in range(6)], inst="INS-000001"),
        _bars(sessions, [200.0 + i for i in range(6)], inst="INS-000002"),
    ])
    eng = LabelEngine(bars)
    t1 = datetime(2020, 1, 6, 21, 0, 1, tzinfo=WINTER)
    t2 = datetime(2020, 1, 7, 21, 0, 1, tzinfo=WINTER)
    frame = eng.compute(_contract(horizon=3), [
        {"instrument_id": "INS-000001", "decision_time": t1},
        {"instrument_id": "INS-000002", "decision_time": t2},
    ])
    from orbit.labels import overlapping_pairs
    assert overlapping_pairs(frame) == []