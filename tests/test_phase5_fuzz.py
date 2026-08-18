"""Phase 5 targeted third-pass audit: differential fuzz and property tests.

Differential fuzz: the public `entry_bar` query and the internal
`_entry_index` used by `compute_one` must agree on every randomized
instant (both anchor modes, boundary instants included) - the two paths
share logic by delegation and must never drift.

Property tests: `overlapping_pairs` invariants (determinism under input
order, symmetry, exact overlap-session counts, availability-only
participation, non-overlap detection).

All randomness is seeded: failures are reproducible.
"""

from __future__ import annotations

import random
from datetime import date, datetime, timedelta, timezone

import polars as pl
import pytest

from orbit.labels import (
    AnchorMode,
    LabelContract,
    LabelEngine,
    ReturnConvention,
    overlapping_pairs,
)

WINTER = timezone.utc

_SESSIONS = [date(2020, 1, 2) + timedelta(days=i) for i in range(14)]  # incl. weekends


def _bars(inst="INS-000001"):
    return pl.DataFrame(
        {
            "instrument_id": [inst] * len(_SESSIONS),
            "trade_date": _SESSIONS,
            "open": [100.0 + i for i in range(len(_SESSIONS))],
            "high": [101.0 + i for i in range(len(_SESSIONS))],
            "low": [99.0 + i for i in range(len(_SESSIONS))],
            "close": [100.0 + i for i in range(len(_SESSIONS))],
            "volume": [1000] * len(_SESSIONS),
        }
    )


def _contract(anchor: AnchorMode, horizon: int = 3, **kw) -> LabelContract:
    base = dict(
        label_id="LAB-001", version="v1", target_type="forward_return",
        horizon=horizon, anchor_mode=anchor,
        return_convention=ReturnConvention.SIMPLE_PRICE_RETURN,
        formula="fuzz contract",
    )
    base.update(kw)
    return LabelContract(**base)


def _random_instants(rng: random.Random, n: int) -> list[datetime]:
    """Random instants spread over the session span, including boundary
    hits: exactly at a session close, one microsecond after, before the
    first bar, after the last bar, and weekend instants."""
    out: list[datetime] = []
    for _ in range(n):
        roll = rng.random()
        if roll < 0.3:
            session = rng.choice(_SESSIONS)
            micro = rng.choice([0, 1, 999999])
            out.append(datetime(2020, 1, session.day, 21, 0, 0, micro))
        elif roll < 0.4:
            out.append(datetime(2020, 1, 2, 20, 59, 59))  # before the first close
        elif roll < 0.5:
            out.append(datetime(2020, 1, 17, 0, 0, 1))  # after the last session
        else:
            day = rng.randint(1, 17)
            hour = rng.randint(0, 23)
            minute = rng.randint(0, 59)
            second = rng.randint(0, 59)
            out.append(datetime(2020, 1, day, hour, minute, second))
    return out


# ------------------------------------------------------------- differential fuzz

def test_entry_bar_matches_compute_one_on_random_instants():
    rng = random.Random(20260118)
    eng = LabelEngine(_bars())
    for anchor in (AnchorMode.DECISION_INSTANT, AnchorMode.POST_EVENT):
        contract = _contract(anchor)
        for t in _random_instants(rng, 120):
            if anchor == AnchorMode.POST_EVENT:
                anchor_instant = datetime(2020, 1, rng.randint(1, 16), 12, 0, 0)
                bar = eng.entry_bar(
                    "INS-000001", t, anchor_instant=anchor_instant,
                    anchor_mode=anchor,
                )
                row = eng.compute_one(
                    contract, "INS-000001", t, anchor_instant=anchor_instant,
                )
            else:
                bar = eng.entry_bar("INS-000001", t, anchor_mode=anchor)
                row = eng.compute_one(contract, "INS-000001", t)
            entry_session = bar["trade_date"] if bar is not None else None
            assert row["entry_session"] == entry_session, (
                f"{anchor}: {t} -> query {entry_session}, "
                f"compute {row['entry_session']}"
            )
            if bar is not None:
                assert row["entry_close"] == bar["close"]


def test_compute_one_entry_equals_entry_bar_for_post_event_anchor():
    rng = random.Random(7)
    eng = LabelEngine(_bars())
    contract = _contract(AnchorMode.POST_EVENT)
    for _ in range(60):
        anchor = datetime(2020, 1, rng.randint(1, 16), rng.randint(0, 23),
                          rng.randint(0, 59), rng.randint(0, 59))
        bar = eng.entry_bar("INS-000001", None, anchor_instant=anchor,
                            anchor_mode=AnchorMode.POST_EVENT)
        row = eng.compute_one(contract, "INS-000001", anchor,  # decision_time unused
                              anchor_instant=anchor)
        assert row["entry_session"] == (
            bar["trade_date"] if bar is not None else None
        )


def test_fuzz_determinism_across_shuffled_decision_order():
    rng = random.Random(99)
    eng = LabelEngine(_bars())
    contract = _contract(AnchorMode.DECISION_INSTANT, horizon=2)
    instants = _random_instants(rng, 40)
    decisions = [{"instrument_id": "INS-000001", "decision_time": t}
                 for t in instants]
    a = eng.compute(contract, decisions)
    b = eng.compute(contract, list(reversed(decisions)))
    assert a.to_dicts() == b.to_dicts()


# ------------------------------------------------------------------ properties

def test_overlapping_pairs_is_symmetric_and_deterministic():
    rng = random.Random(5)
    eng = LabelEngine(_bars())
    contract = _contract(AnchorMode.DECISION_INSTANT, horizon=5)
    decisions = [
        {"instrument_id": "INS-000001", "decision_time": t}
        for t in _random_instants(rng, 25)
    ]
    frame = eng.compute(contract, decisions)
    sessions = {"INS-000001": eng.instrument_sessions("INS-000001")}
    pairs = overlapping_pairs(frame, sessions_by_instrument=sessions)
    # symmetric: (a, b) and (b, a) never both appear; a != b; window fields
    seen: set[tuple[str, str]] = set()
    for p in pairs:
        assert p["decision_id_a"] != p["decision_id_b"]
        key = (p["decision_id_a"], p["decision_id_b"])
        assert key not in seen
        seen.add(key)
        assert p["instrument_id"] == "INS-000001"
    # determinism: reversed decision order in the frame yields the same pairs
    frame_rev = frame.sort(["decision_time"], descending=True)
    pairs_rev = overlapping_pairs(frame_rev, sessions_by_instrument=sessions)
    assert {(p["decision_id_a"], p["decision_id_b"]) for p in pairs} == \
        {(p["decision_id_a"], p["decision_id_b"]) for p in pairs_rev}


def test_overlap_sessions_equals_the_actual_session_intersection():
    rng = random.Random(11)
    eng = LabelEngine(_bars())
    contract = _contract(AnchorMode.DECISION_INSTANT, horizon=6)
    decisions = [
        {"instrument_id": "INS-000001", "decision_time": t}
        for t in _random_instants(rng, 30)
    ]
    frame = eng.compute(contract, decisions)
    sessions = eng.instrument_sessions("INS-000001")
    pairs = overlapping_pairs(frame, sessions_by_instrument={
        "INS-000001": sessions})
    available = [r for r in frame.iter_rows(named=True)
                 if r["outcome_status"] == "available"]
    for p in pairs:
        a = next(r for r in available if r["decision_id"] == p["decision_id_a"])
        b = next(r for r in available if r["decision_id"] == p["decision_id_b"])
        lo = max(a["window_start_session"], b["window_start_session"])
        hi = min(a["window_end_session"], b["window_end_session"])
        expected = sum(1 for s in sessions if lo <= s <= hi)
        assert p["overlap_sessions"] == expected, p
        # every reported pair genuinely overlaps
        assert a["window_end_session"] >= b["window_start_session"]
        assert b["window_end_session"] >= a["window_start_session"]


def test_overlap_count_matches_brute_force_on_random_decisions():
    rng = random.Random(23)
    eng = LabelEngine(_bars())
    contract = _contract(AnchorMode.DECISION_INSTANT, horizon=4)
    decisions = [
        {"instrument_id": "INS-000001", "decision_time": t}
        for t in _random_instants(rng, 35)
    ]
    frame = eng.compute(contract, decisions)
    sessions = eng.instrument_sessions("INS-000001")
    pairs = overlapping_pairs(frame, sessions_by_instrument={
        "INS-000001": sessions})
    available = [r for r in frame.iter_rows(named=True)
                 if r["outcome_status"] == "available"]
    expected: list[dict] = []
    for i in range(len(available)):
        for j in range(i + 1, len(available)):
            a, b = available[i], available[j]
            if (a["window_end_session"] >= b["window_start_session"]
                    and b["window_end_session"] >= a["window_start_session"]):
                lo = max(a["window_start_session"], b["window_start_session"])
                hi = min(a["window_end_session"], b["window_end_session"])
                expected.append((a["decision_id"], b["decision_id"],
                                 sum(1 for s in sessions if lo <= s <= hi)))
    got = {(p["decision_id_a"], p["decision_id_b"], p["overlap_sessions"])
           for p in pairs}
    assert got == set(expected), (got, expected)


def test_overlap_rejects_non_overlapping_and_unavailable_rows():
    rng = random.Random(31)
    eng = LabelEngine(_bars())
    contract = _contract(AnchorMode.DECISION_INSTANT, horizon=2)
    # decisions far apart (day 2 vs day 16) and one past the last session
    decisions = [
        {"instrument_id": "INS-000001", "decision_time": datetime(2020, 1, 2, 21, 0, 1)},
        {"instrument_id": "INS-000001", "decision_time": datetime(2020, 1, 16, 21, 0, 1)},
        {"instrument_id": "INS-000001", "decision_time": datetime(2020, 1, 17, 0, 0, 1)},
    ]
    frame = eng.compute(contract, decisions)
    pairs = overlapping_pairs(frame, sessions_by_instrument={
        "INS-000001": eng.instrument_sessions("INS-000001")})
    # the last decision has no window (unavailable) and the first two do not
    # overlap; the only candidate pair must be empty
    assert pairs == []