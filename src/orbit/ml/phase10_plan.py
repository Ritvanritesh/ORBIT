"""Phase 10 pre-registered ablation plan (locked before any execution).

The plan defines the EXACT set of feature-set x model experiments Phase 10 is
allowed to run. It is registered and digest-locked before any training or
evaluation happens; a later idea requires a NEW registered experiment, never a
mutation of the plan. The plan is deliberately small and bounded (no
uncontrolled combinatorial search):

  feature sets (13, incl. the frozen FS-001 base control):
    FS-001 BASE (Phase 9 baseline, 8 features)
    FS-002 NEW (Phase 10 candidate families only, 15)
    FS-003 ALL (FS-001 + FS-002, 23)
    FS-004..FS-008  BASE + one family (11 each)   [base-plus-family ablation]
    FS-009..FS-013  ALL - one family (20 each)    [leave-one-family-out]

  models (4, one per Phase 9 family, chosen BEFORE Phase 10 results):
    ridge         alpha=1.0        (linear; grid midpoint; the Phase 9
                                    Review-1 representative point EXP-90003)
    lasso         alpha=0.001      (sparse linear; grid midpoint)
    random_forest n_estimators=200, max_depth=3  (nonlinear tree; highest
                                    pre-registered capacity; Phase 9 Review-1
                                    representative point EXP-90015)
    xgboost       n_estimators=200, max_depth=3, learning_rate=0.1
                                    (gradient boosting; mirrors the RF point
                                    within the pre-registered XGB grid)

  13 x 4 = 52 experiments, EXP-10001..EXP-10052 (deterministic ordering:
  feature sets in the order above, models in the order above). Logistic is
  intentionally excluded from the ablation subset: its score is a monotone
  transform of the ridge regression sign for the binary target and adds no
  independent ranking information for the feature-representation question.
  The infrastructure (validate_model_parameters + the Phase 9 registry path)
  supports it if a later registered experiment needs it.

  Every experiment is run through the exact Phase 9 protocol (same dataset
  DS-000004, label LAB-004 v1, locked split, CM-001 costs, Phase 7 backtest,
  top-3 WEIGHT signals, seed 42, calibration on validation only).
"""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any

from orbit.ml.features import (
    ALL_PHASE10_DEFINITIONS,
    FEATURE_DEFINITIONS,
    PHASE10_FAMILY_DEFINITIONS,
    PHASE10_FEATURE_SETS,
    PHASE10_FEATURE_SET_ORDER,
    _feature_definitions_digest,
    phase10_set_identity,
)
from orbit.ml.grids import PHASE9_GRIDS, validate_model_parameters

PHASE10_MODEL_POINTS: list[dict[str, Any]] = [
    {
        "family": "ridge",
        "params": {"alpha": 1.0},
        "justification": (
            "linear family; midpoint of the Phase 9 ridge grid and the "
            "representative point used by the Phase 9 Review-1 deep audit "
            "(EXP-90003)"
        ),
        "phase9_parent": "EXP-90003",
    },
    {
        "family": "lasso",
        "params": {"alpha": 0.001},
        "justification": (
            "sparse linear family; midpoint of the Phase 9 lasso grid "
            "(1e-4 .. 1e-1)"
        ),
        "phase9_parent": "EXP-90006",
    },
    {
        "family": "random_forest",
        "params": {"n_estimators": 200, "max_depth": 3},
        "justification": (
            "nonlinear tree family; highest-capacity pre-registered RF point, "
            "the representative point used by the Phase 9 Review-1 deep audit "
            "(EXP-90015)"
        ),
        "phase9_parent": "EXP-90015",
    },
    {
        "family": "xgboost",
        "params": {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.1},
        "justification": (
            "gradient boosting family; mirrors the random_forest point within "
            "the pre-registered XGB grid (n_estimators=200, max_depth=3, lr=0.1)"
        ),
        "phase9_parent": "EXP-90019",
    },
]

PHASE10_MODEL_ORDER: list[str] = [m["family"] for m in PHASE10_MODEL_POINTS]

_EXPERIMENT_ID_BASE = 10001
EXPERIMENT_COUNT = len(PHASE10_FEATURE_SET_ORDER) * len(PHASE10_MODEL_POINTS)

_BASE_IDS = [f["feature_id"] for f in FEATURE_DEFINITIONS]
_ALL_IDS = [f["feature_id"] for f in FEATURE_DEFINITIONS] + [
    f["feature_id"] for f in ALL_PHASE10_DEFINITIONS
]


def _set_members(feature_set_id: str) -> list[str]:
    if feature_set_id == "FS-001":
        return list(_BASE_IDS)
    return list(PHASE10_FEATURE_SETS[feature_set_id]["members"])


def phase10_experiment_id(feature_set_id: str, family: str, params: dict[str, Any]) -> str:
    """Deterministic Phase 10 experiment id (EXP-10001..EXP-10052).

    Order: feature sets in PHASE10_FEATURE_SET_ORDER, then models in
    PHASE10_MODEL_ORDER. The params are validated against the Phase 9 grid so
    an unregistered hyperparameter set can never be assigned an id.
    """
    validate_model_parameters(family, params)
    set_index = PHASE10_FEATURE_SET_ORDER.index(feature_set_id)
    for i, m in enumerate(PHASE10_MODEL_POINTS):
        if m["family"] == family and m["params"] == dict(params):
            return f"EXP-{_EXPERIMENT_ID_BASE + set_index * len(PHASE10_MODEL_POINTS) + i}"
    raise ValueError(
        f"model point {family} {params} is not part of the pre-registered "
        "Phase 10 ablation subset"
    )


def phase10_model_point_for(family: str, params: dict[str, Any]) -> dict[str, Any]:
    validate_model_parameters(family, params)
    for m in PHASE10_MODEL_POINTS:
        if m["family"] == family and m["params"] == dict(params):
            return copy.deepcopy(m)
    raise ValueError(
        f"model point {family} {params} is not part of the pre-registered "
        "Phase 10 ablation subset"
    )


def _plan_payload() -> dict[str, Any]:
    return {
        "protocol": "phase10_v1",
        "dataset_snapshot_ids": ["DS-000004"],
        "label_id": "LAB-004",
        "label_version": "v1",
        "cost_model_id": "CM-001",
        "seed": 42,
        "windows": "fixed_chronological_v1 (Phase 9 protocol, locked)",
        "splits": {
            "train": "2010-01-04..2018-12-31",
            "val": "2019-01-02..2021-12-31",
            "test": "2022-01-03..2026-06-30",
        },
        "signal_construction": "top-3 long, equal weight 1/3 (Phase 9 path)",
        "feature_sets": [
            {
                "feature_set_id": sid,
                "members": _set_members(sid),
                "definitions_digest": _feature_definitions_digest(_set_members(sid)),
                "role": (
                    "base"
                    if sid == "FS-001"
                    else PHASE10_FEATURE_SETS[sid]["role"]
                ),
                "family": (
                    None if sid == "FS-001" else PHASE10_FEATURE_SETS[sid].get("family")
                ),
            }
            for sid in PHASE10_FEATURE_SET_ORDER
        ],
        "models": PHASE10_MODEL_POINTS,
        "experiment_count": EXPERIMENT_COUNT,
        "experiment_id_base": _EXPERIMENT_ID_BASE,
        "experiment_id_range": f"EXP-{_EXPERIMENT_ID_BASE}..EXP-{_EXPERIMENT_ID_BASE + EXPERIMENT_COUNT - 1}",
    }


def phase10_plan_digest() -> str:
    """sha256 over the full pre-registered plan (locked identity)."""
    raw = json.dumps(_plan_payload(), sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def phase10_plan() -> dict[str, Any]:
    """The complete, locked plan (payload + digest)."""
    payload = _plan_payload()
    payload["plan_digest"] = phase10_plan_digest()
    return payload


def validate_phase10_plan() -> None:
    """Adversarial verification of the locked plan before any execution."""
    # 1. every feature set exists and its members resolve to documented features
    for sid in PHASE10_FEATURE_SET_ORDER:
        if sid == "FS-001":
            continue
        entry = PHASE10_FEATURE_SETS[sid]
        members = list(entry["members"])
        if not members:
            raise ValueError(f"feature set {sid} has no members")
        if len(set(members)) != len(members):
            raise ValueError(f"feature set {sid} has duplicate members")
        if entry["role"] in ("base_plus_family", "all_minus_family"):
            family_ids = [
                f["feature_id"] for f in PHASE10_FAMILY_DEFINITIONS[entry["family"]]
            ]
        else:
            family_ids = []
        if entry["role"] == "base_plus_family":
            if set(members) != set(_BASE_IDS) | set(family_ids):
                raise ValueError(f"set {sid} is not BASE+{entry['family']}")
        elif entry["role"] == "all_minus_family":
            if set(members) != set(_ALL_IDS) - set(family_ids):
                raise ValueError(f"set {sid} is not ALL-{entry['family']}")
        elif entry["role"] == "all":
            if set(members) != set(_ALL_IDS):
                raise ValueError(f"set {sid} is not ALL")
        elif entry["role"] == "new":
            if set(members) != set(_ALL_IDS) - set(_BASE_IDS):
                raise ValueError(f"set {sid} is not NEW-only")

    # 2. every model point is a pre-registered Phase 9 grid point
    for m in PHASE10_MODEL_POINTS:
        validate_model_parameters(m["family"], m["params"])
        if m["params"] not in PHASE9_GRIDS[m["family"]]:
            raise ValueError(f"model {m} is not in the Phase 9 grid")

    # 3. experiment ids are unique and within the declared range
    ids = [
        phase10_experiment_id(sid, m["family"], m["params"])
        for sid in PHASE10_FEATURE_SET_ORDER
        for m in PHASE10_MODEL_POINTS
    ]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate experiment ids in the Phase 10 plan")
    if min(ids) != f"EXP-{_EXPERIMENT_ID_BASE}" or max(ids) != (
        f"EXP-{_EXPERIMENT_ID_BASE + EXPERIMENT_COUNT - 1}"
    ):
        raise ValueError("experiment id range mismatch in the Phase 10 plan")


__all__ = [
    "PHASE10_MODEL_POINTS",
    "PHASE10_MODEL_ORDER",
    "EXPERIMENT_COUNT",
    "phase10_experiment_id",
    "phase10_model_point_for",
    "phase10_plan",
    "phase10_plan_digest",
    "validate_phase10_plan",
]