"""Phase 5 versioning tests: labels are immutable and reproducible.

Three properties are pinned here:
  1. REGISTRY IMMUTABILITY - a (label_id, version) definition is frozen on
     registration: no re-registration, no out-of-order versions, no
     byte-identical re-registration under a new version number.
  2. DETERMINISM - the same contract + bars + decisions always produce the
     identical label frame (row for row) and the identical snapshot digest,
     including across engine instances.
  3. IDENTITY STABILITY - the content hash is a formula identity: historical
     experiments pin (label_id, version) and always resolve the same
     definition with the same digest.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, timezone

import polars as pl
import pytest
from pydantic import ValidationError

from orbit.labels import (
    AnchorMode,
    LabelContract,
    LabelEngine,
    LabelSnapshot,
    LabelVersionRecord,
    LabelVersionRegistry,
    ReturnConvention,
)


def _contract(**kw) -> LabelContract:
    base = dict(
        label_id="LAB-001", version="v1", target_type="forward_return",
        horizon=5, anchor_mode=AnchorMode.DECISION_INSTANT,
        return_convention=ReturnConvention.SIMPLE_PRICE_RETURN,
        formula="versioning test contract",
    )
    base.update(kw)
    return LabelContract(**base)


def _bars(n=8):
    import datetime as dt

    sessions = []
    d = dt.date(2020, 1, 6)
    while len(sessions) < n:
        if d.weekday() < 5:
            sessions.append(d)
        d += dt.timedelta(days=1)
    return pl.DataFrame(
        {
            "instrument_id": ["INS-000001"] * n,
            "trade_date": sessions,
            "open": [100.0 + i for i in range(n)],
            "high": [101.0 + i for i in range(n)],
            "low": [99.0 + i for i in range(n)],
            "close": [100.0 + i for i in range(n)],
            "volume": [1000] * n,
        }
    )


def _decisions():
    return [
        {"instrument_id": "INS-000001", "decision_time": datetime(2020, 1, 6, 21, 0, 1, tzinfo=timezone.utc)},
        {"instrument_id": "INS-000001", "decision_time": datetime(2020, 1, 7, 21, 0, 1, tzinfo=timezone.utc)},
        {"instrument_id": "INS-000001", "decision_time": datetime(2020, 1, 8, 21, 0, 1, tzinfo=timezone.utc)},
    ]


# ------------------------------------------------- registry immutability

def test_register_then_version_bump_is_strictly_newer():
    reg = LabelVersionRegistry()
    reg.register(_contract(), note="v1")
    reg.register(_contract(version="v2", formula="v2 definition"), note="v2")
    assert reg.versions("LAB-001") == ["v1", "v2"]
    assert reg.get("LAB-001", "v1").note == "v1"
    # the v1 definition is the same object as registered (frozen)
    assert reg.definition("LAB-001", "v1").content_hash() == _contract().content_hash()


def test_duplicate_registration_is_refused():
    reg = LabelVersionRegistry()
    reg.register(_contract())
    with pytest.raises(ValueError, match="already registered"):
        reg.register(_contract())


def test_identical_definition_under_new_version_is_refused():
    reg = LabelVersionRegistry()
    reg.register(_contract())
    with pytest.raises(ValueError, match="identical"):
        reg.register(_contract(version="v2"))


def test_out_of_order_versions_are_refused():
    reg = LabelVersionRegistry()
    reg.register(_contract(version="v2", formula="v2"))
    with pytest.raises(ValueError, match="strictly newer"):
        reg.register(_contract(version="v1", formula="v1"))


def test_registration_requires_a_contract_object():
    reg = LabelVersionRegistry()
    with pytest.raises(TypeError, match="LabelContract"):
        reg.register({"label_id": "LAB-001", "version": "v1"})


def test_get_unknown_label_and_version_errors():
    reg = LabelVersionRegistry()
    reg.register(_contract())
    with pytest.raises(KeyError, match="LAB-999"):
        reg.get("LAB-999")
    with pytest.raises(KeyError, match="registered versions"):
        reg.get("LAB-001", "v9")


def test_get_latest_version():
    reg = LabelVersionRegistry()
    reg.register(_contract())
    reg.register(_contract(version="v2", formula="v2"))
    assert reg.get("LAB-001").version == "v2"


def test_definition_digest_is_the_formula_identity():
    reg = LabelVersionRegistry()
    c = _contract()
    reg.register(c)
    assert reg.definition_digest("LAB-001", "v1") == c.content_hash()
    assert reg.definition_digest("LAB-001", "v1") == _contract().content_hash()


# ----------------------------------------------------------- determinism

def test_compute_is_deterministic_across_engine_instances():
    bars = _bars()
    c = _contract()
    decs = _decisions()
    f1 = LabelEngine(bars).compute(c, decs)
    f2 = LabelEngine(bars).compute(c, decs)
    assert f1.equals(f2)
    assert f1.to_dicts() == f2.to_dicts()


def test_compute_row_order_is_irrelevant_to_the_result():
    bars = _bars()
    c = _contract()
    decs = _decisions()
    f1 = LabelEngine(bars).compute(c, decs)
    f2 = LabelEngine(bars).compute(c, list(reversed(decs)))
    assert f1.to_dicts() == f2.to_dicts()


def test_snapshot_digest_is_deterministic_and_provenance_rich():
    bars = _bars()
    c = _contract()
    frame = LabelEngine(bars).compute(c, _decisions())
    snap1 = LabelSnapshot(
        label_id=c.label_id, version=c.version,
        contract_digest=c.content_hash(), engine_version="v1.0.0",
        data_refs=["DS-000001"], records=frame,
    )
    snap2 = LabelSnapshot(
        label_id=c.label_id, version=c.version,
        contract_digest=c.content_hash(), engine_version="v1.0.0",
        data_refs=["DS-000001"], records=frame,
    )
    assert snap1.content_digest == snap2.content_digest
    assert snap1.equals(snap2)
    assert len(snap1.content_digest) == 64
    assert snap1.row_count() == 3
    assert snap1.available_count() == 3
    assert snap1.unavailable_count() == 0
    prov = snap1.provenance()
    assert prov["label_id"] == "LAB-001"
    assert prov["data_refs"] == ["DS-000001"]
    # wall-clock is excluded from identity but present in the record
    assert "created_at" not in prov
    assert "created_at" in snap1.to_json()


def test_snapshot_digest_differs_when_data_or_definition_differs():
    bars = _bars()
    c = _contract()
    frame = LabelEngine(bars).compute(c, _decisions())
    snap = LabelSnapshot(label_id=c.label_id, version=c.version,
                         contract_digest=c.content_hash(),
                         engine_version="v1.0.0", data_refs=["DS-000001"],
                         records=frame)
    c2 = _contract(formula="other formula")
    snap2 = LabelSnapshot(label_id=c2.label_id, version=c2.version,
                          contract_digest=c2.content_hash(),
                          engine_version="v1.0.0", data_refs=["DS-000001"],
                          records=frame)
    assert snap.content_digest != snap2.content_digest


def test_snapshot_equals_is_order_insensitive_on_data_refs():
    bars = _bars()
    c = _contract()
    frame = LabelEngine(bars).compute(c, _decisions())
    snap1 = LabelSnapshot(label_id=c.label_id, version=c.version,
                          contract_digest=c.content_hash(),
                          engine_version="v1.0.0",
                          data_refs=["DS-000001", "DS-000002"], records=frame)
    snap2 = LabelSnapshot(label_id=c.label_id, version=c.version,
                          contract_digest=c.content_hash(),
                          engine_version="v1.0.0",
                          data_refs=["DS-000002", "DS-000001"], records=frame)
    assert snap1.content_digest == snap2.content_digest
    assert snap1.equals(snap2)
    # a genuinely different source set is not equal
    snap3 = LabelSnapshot(label_id=c.label_id, version=c.version,
                          contract_digest=c.content_hash(),
                          engine_version="v1.0.0", data_refs=["DS-000001"],
                          records=frame)
    assert not snap1.equals(snap3)


def test_snapshot_requires_canonical_label_schema():
    c = _contract()
    junk = pl.DataFrame({"a": [1], "b": [2]})
    with pytest.raises(ValueError, match="canonical label output schema"):
        LabelSnapshot(label_id=c.label_id, version=c.version,
                      contract_digest=c.content_hash(),
                      engine_version="v1.0.0", data_refs=[], records=junk)
    # the empty frame is validated the same way: junk columns on a 0-row
    # frame are also refused (never a silent schema change)
    with pytest.raises(ValueError, match="canonical label output schema"):
        LabelSnapshot(label_id=c.label_id, version=c.version,
                      contract_digest=c.content_hash(),
                      engine_version="v1.0.0", data_refs=[],
                      records=pl.DataFrame({"a": []}))


def test_snapshot_available_counts_per_unavailable_reason():
    bars = _bars(3)  # only 3 sessions: the 5-session labels are unavailable
    c = _contract()
    frame = LabelEngine(bars).compute(c, _decisions())
    snap = LabelSnapshot(label_id=c.label_id, version=c.version,
                         contract_digest=c.content_hash(),
                         engine_version="v1.0.0", data_refs=[], records=frame)
    assert snap.available_count() == 0
    assert snap.unavailable_count() == 3
    assert snap.unavailable_reason_counts() == {"insufficient_future_data": 3}


# ------------------------------------------------------ digest stability

def test_content_hash_is_a_sha256_of_canonical_json():
    c = _contract()
    canonical = c.canonical_json()
    assert c.content_hash() == hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    # the canonical json is compact and stable
    assert json.loads(canonical)["label_id"] == "LAB-001"
    assert json.loads(canonical)["horizon"] == 5


def test_contract_never_changes_after_registration():
    reg = LabelVersionRegistry()
    reg.register(_contract())
    c1 = reg.definition("LAB-001", "v1")
    h1 = c1.content_hash()
    # a second registration path returns an equal definition
    c2 = _contract()
    h2 = c2.content_hash()
    assert h1 == h2
    # the registry's record is frozen: no field can be mutated
    with pytest.raises(ValidationError):
        reg.get("LAB-001", "v1").contract.horizon = 10
    with pytest.raises(ValidationError):
        reg.get("LAB-001", "v1").note = "changed"


def test_historical_experiment_resolution_is_stable():
    reg = LabelVersionRegistry()
    reg.register(_contract())
    reg.register(_contract(version="v2", formula="v2 definition"))
    # an experiment pinned (LAB-001, v1) always resolves the v1 formula
    assert reg.definition("LAB-001", "v1").formula == "versioning test contract"
    assert reg.definition("LAB-001", "v2").formula == "v2 definition"
    # and its digest matches what the label frame records
    bars = _bars()
    frame = LabelEngine(bars).compute(reg.definition("LAB-001", "v1"), _decisions())
    recorded = frame["contract_digest"].unique().to_list()
    assert recorded == [reg.definition_digest("LAB-001", "v1")]