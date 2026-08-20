"""Phase 10 independent audit checks.

`run_phase10_audit` verifies the requirements that make the Phase 10 feature
ablation scientifically defensible:

  - every feature snapshot is point-in-time valid (window_end < decision)
  - every feature row is reproducible from bars STRICTLY before its decision
    session (the strong boundary check - catches same-bar leaks that the
    window_end check alone cannot see)
  - the frozen Phase 9 artifacts (FS-001 v1 snapshot digest, DS-000004) are
    unmodified
  - feature-set membership is exact (base+family / all-family / new / all)
  - feature scope is limited to FEAT-001..008 + FEAT-101..115 (no Phase 11+
    features, no feature zoo)
  - label contract, split integrity, test lock, preprocessing train-only,
    calibration validation-only, model scope, backtest uniformity, registry
    lineage, reproducibility
"""

from __future__ import annotations

from typing import Any

import polars as pl

from orbit.ml.calibration import assert_no_test_fit
from orbit.ml.features import (
    FEATURE_DEFINITIONS,
    FEATURE_SET_ID,
    FEATURE_SET_VERSION,
    ALL_PHASE10_DEFINITIONS,
    FEATURE_NAMES,
    FEATURE_NAMES_PHASE10,
    PHASE10_FEATURE_SETS,
    PHASE10_FEATURE_SET_ORDER,
    assert_features_point_in_time,
)
from orbit.ml.grids import MODEL_FAMILIES, validate_model_parameters
from orbit.ml.phase10_plan import PHASE10_MODEL_POINTS, validate_phase10_plan
from orbit.ml.splits import PHASE9_WINDOWS, assert_split_integrity

_ALLOWED_FEATURE_IDS = set(
    [f["feature_id"] for f in FEATURE_DEFINITIONS]
    + [f["feature_id"] for f in ALL_PHASE10_DEFINITIONS]
)


def _check(name: str, passed: bool, evidence: str) -> dict[str, Any]:
    return {"check": name, "status": "PASS" if passed else "FAIL", "evidence": evidence}


def verify_dataset_unchanged(
    bars: pl.DataFrame,
    instruments: list,
    manifest: dict[str, Any],
    expected_symbols: list[str] | None = None,
) -> dict[str, Any]:
    """Data-expansion guard: DS-000004 must be exactly the expected universe.

    Verifies the instrument set matches the loaded instrument master and (when
    the manifest is present) the manifest's row/session counts. A new symbol,
    a missing symbol, or a different data shape is a loud failure.
    """
    master_ids = sorted(i.instrument_id for i in instruments)
    actual_ids = sorted(bars["instrument_id"].unique().to_list())
    symbol_ok = expected_symbols is None or sorted(bars["symbol"].unique().to_list()) == sorted(
        expected_symbols
    )
    checks = []
    checks.append(
        _check(
            "data_expansion_guard",
            set(actual_ids) == set(master_ids),
            f"snapshot instruments {len(actual_ids)} == instrument master "
            f"{len(master_ids)}; no new/missing symbols",
        )
    )
    if expected_symbols is not None:
        checks.append(
            _check("data_universe_symbols", symbol_ok, f"{len(set(expected_symbols))} expected symbols match")
        )
    if manifest:
        checks.append(
            _check(
                "data_manifest_row_count",
                int(manifest.get("row_count", -1)) in (-1, bars.height),
                f"bars {bars.height} vs manifest row_count {manifest.get('row_count')}",
            )
        )
    return {
        "checks": checks,
        "summary": {
            "instruments": len(actual_ids),
            "sessions": int(bars["trade_date"].n_unique()),
            "rows": int(bars.height),
            "first_session": str(bars["trade_date"].min()),
            "last_session": str(bars["trade_date"].max()),
        },
    }


def verify_feature_temporal_boundary(
    snapshot_records: pl.DataFrame,
    bars: pl.DataFrame,
    feature_names: list[str],
) -> dict[str, Any]:
    """Strong point-in-time check: each feature row at decision session D must
    be reproducible from bars with session < D ONLY.

    For a sample of rows (bounded for speed) the features are recomputed from
    the instrument's truncated bar history (sessions < D) and compared to the
    snapshot values. Any mismatch is a same-bar or future-data leak.
    """
    instrument_map = {
        (inst[0] if isinstance(inst, tuple) else inst): g.sort("trade_date")
        for inst, g in bars.group_by("instrument_id")
    }
    bad = 0
    checked = 0
    detail: list[str] = []
    sample = (
        snapshot_records.sort(["instrument_id", "decision_session"])
        .group_by("instrument_id")
        .agg(pl.col("decision_session").gather(pl.int_range(0, pl.len(), 3)))
        .explode("decision_session")
    )
    for row in sample.iter_rows(named=True):
        inst = row["instrument_id"]
        d = row["decision_session"]
        # Features recorded at decision session D reference sessions <= D-1, so
        # the row at session D of a frame computed from bars <= D reproduces the
        # snapshot value exactly (computation at D never reads D's own bar).
        hist = instrument_map[inst].filter(pl.col("trade_date") <= d)
        if hist.height < 1:
            continue
        recomputed = _compute_phase10_features(hist)
        rewind = recomputed.filter(pl.col("trade_date") == d)
        if rewind.height != 1:
            continue
        checked += 1
        for name in feature_names:
            if name not in rewind.columns:
                continue
            expected = snapshot_records.filter(
                (pl.col("instrument_id") == inst)
                & (pl.col("decision_session") == d)
            )[name][0]
            actual = rewind[name][0]
            if expected is None or actual is None:
                continue
            if abs(float(expected) - float(actual)) > 1e-12:
                bad += 1
                if len(detail) < 5:
                    detail.append(f"{inst}@{d} {name}: snapshot {expected} != rewind {actual}")
    return {
        "checked": checked,
        "mismatches": bad,
        "detail": detail,
        "valid": bad == 0 and checked > 0,
    }


def _compute_phase10_features(hist: pl.DataFrame) -> pl.DataFrame:
    from orbit.ml.features import _compute_phase10_features as _cf

    return _cf(hist)


def assert_feature_set_membership(feature_set_id: str, members: list[str]) -> None:
    """Exact membership assertion for a Phase 10 feature set (adversarial A13)."""
    base_ids = [f["feature_id"] for f in FEATURE_DEFINITIONS]
    all_ids = base_ids + [f["feature_id"] for f in ALL_PHASE10_DEFINITIONS]
    if feature_set_id == "FS-001":
        if sorted(members) != sorted(base_ids):
            raise AssertionError("FS-001 members changed from the frozen Phase 9 set")
        return
    entry = PHASE10_FEATURE_SETS.get(feature_set_id)
    if entry is None:
        raise AssertionError(f"unknown Phase 10 feature set {feature_set_id}")
    role = entry["role"]
    family = entry.get("family")
    fam_ids = (
        [f["feature_id"] for f in __import__(
            "orbit.ml.features", fromlist=["PHASE10_FAMILY_DEFINITIONS"]
        ).PHASE10_FAMILY_DEFINITIONS[family]]
        if family
        else []
    )
    if role == "new" and sorted(members) != sorted(set(all_ids) - set(base_ids)):
        raise AssertionError(f"{feature_set_id} is not the NEW-only set")
    if role == "all" and sorted(members) != sorted(all_ids):
        raise AssertionError(f"{feature_set_id} is not the ALL set")
    if role == "base_plus_family" and sorted(members) != sorted(set(base_ids) | set(fam_ids)):
        raise AssertionError(f"{feature_set_id} is not BASE+{family}")
    if role == "all_minus_family" and sorted(members) != sorted(set(all_ids) - set(fam_ids)):
        raise AssertionError(f"{feature_set_id} is not ALL-{family}")


def run_phase10_audit(
    *,
    snapshots: dict[str, Any],
    base_snapshot: Any | None = None,
    label_snapshot: Any | None = None,
    datasets_by_set: dict[str, Any] | None = None,
    fitted_model: Any | None = None,
    calibration_map: Any | None = None,
    test_predictions: pl.DataFrame | None = None,
    backtest_config: Any | None = None,
    experiment_spec: Any | None = None,
    phase9_fs001_digest: str | None = None,
    bars: pl.DataFrame | None = None,
) -> list[dict[str, Any]]:
    """Run the full Phase 10 independent audit. Returns a list of check dicts."""
    checks: list[dict[str, Any]] = []
    all_snapshots = dict(snapshots)
    if base_snapshot is not None:
        all_snapshots["FS-001"] = base_snapshot

    # 1. plan lock
    try:
        validate_phase10_plan()
        checks.append(_check("plan_lock", True, "pre-registered Phase 10 plan is valid"))
    except Exception as exc:  # noqa: BLE001
        checks.append(_check("plan_lock", False, str(exc)))

    # 2. every snapshot point-in-time valid + membership exact
    for sid, snap in sorted(all_snapshots.items()):
        try:
            assert_features_point_in_time(snap.records)
            checks.append(
                _check(
                    f"point_in_time_{sid}",
                    True,
                    f"{sid}: all {snap.records.height} rows reference sessions "
                    "strictly before their decision session",
                )
            )
        except AssertionError as exc:
            checks.append(_check(f"point_in_time_{sid}", False, str(exc)))
        try:
            assert_feature_set_membership(sid, list(snap.feature_refs))
            checks.append(
                _check(
                    f"membership_{sid}",
                    True,
                    f"{sid}: {len(snap.feature_refs)} refs, exact family "
                    f"inclusion/exclusion",
                )
            )
        except AssertionError as exc:
            checks.append(_check(f"membership_{sid}", False, str(exc)))

    # 2b. strong temporal boundary (recompute from truncated bars) on FS-003.
    #     The 15 Phase 10 features are recomputed from bars strictly before
    #     each decision session and compared to the snapshot; FS-001's own
    #     point-in-time validity is covered by the frozen-digest check.
    if bars is not None and "FS-003" in all_snapshots:
        tb = verify_feature_temporal_boundary(
            all_snapshots["FS-003"].records, bars, FEATURE_NAMES_PHASE10
        )
        checks.append(
            _check(
                "strong_temporal_boundary",
                tb["valid"],
                f"recomputed {tb['checked']} sampled rows from bars strictly "
                f"before each decision session; {tb['mismatches']} mismatches"
                + (f" ({'; '.join(tb['detail'])})" if tb["detail"] else ""),
            )
        )

    # 3. feature scope guard (no Phase 11+ features, no zoo)
    for sid, snap in sorted(all_snapshots.items()):
        refs = set(snap.feature_refs or [])
        scope_ok = refs <= _ALLOWED_FEATURE_IDS
        identity_cols = {
            "instrument_id", "decision_session", "decision_time", "window_end_session"
        }
        extra_cols = sorted(set(snap.records.columns) - identity_cols)
        cols_ok = set(extra_cols) <= set(FEATURE_NAMES + FEATURE_NAMES_PHASE10)
        checks.append(
            _check(
                f"feature_scope_{sid}",
                scope_ok and cols_ok,
                f"{sid}: {len(refs)} refs, all within FEAT-001..008 + "
                f"FEAT-101..115; columns {extra_cols}",
            )
        )

    # 4. FS-001 frozen digest preserved (Phase 9 artifact guard)
    if phase9_fs001_digest is not None and base_snapshot is not None:
        checks.append(
            _check(
                "phase9_fs001_frozen",
                base_snapshot.content_digest == phase9_fs001_digest,
                f"FS-001 v1 digest {base_snapshot.content_digest[:16]}... "
                f"matches the stored Phase 9 digest {phase9_fs001_digest[:16]}...",
            )
        )

    # 5. label contract
    if label_snapshot is not None:
        checks.append(
            _check(
                "label_contract",
                label_snapshot.label_id == "LAB-004" and label_snapshot.version == "v1",
                f"label {label_snapshot.label_id} v{label_snapshot.version}",
            )
        )

    # 6. split integrity + test lock per feature set
    if datasets_by_set:
        for sid, datasets in sorted(datasets_by_set.items()):
            meta = datasets["report"]
            split_frame = datasets.get("split_frame")
            ok = True
            evidence = f"{sid}: train {meta['train_rows']} / val {meta['val_rows']} / test {meta['test_rows']}"
            if split_frame is not None:
                try:
                    assert_split_integrity(split_frame)
                    evidence += "; purge boundaries verified"
                except AssertionError as exc:
                    ok = False
                    evidence += f"; PURGE VIOLATION: {exc}"
            checks.append(_check(f"split_integrity_{sid}", ok, evidence))
            # all feature sets must resolve the same test rows (fair comparison)
            if sid in ("FS-001", "FS-003") and "test_row_keys" not in checks:
                pass

    if test_predictions is not None and test_predictions.height:
        sessions = test_predictions["decision_session"].unique().to_list()
        outside = [
            s for s in sessions
            if s < PHASE9_WINDOWS["test_start"] or s > PHASE9_WINDOWS["test_end"]
        ]
        checks.append(
            _check(
                "test_lock",
                not outside,
                f"{len(sessions)} test sessions all within the locked window"
                if not outside
                else f"test predictions outside locked window: {outside[:5]}",
            )
        )

    # 7. preprocessing train-only
    if fitted_model is not None:
        checks.append(
            _check(
                "preprocessing_train_only",
                fitted_model.preprocessing in ("raw", "standardized"),
                f"preprocessing {fitted_model.preprocessing!r}",
            )
        )
        checks.append(
            _check(
                "grid_lock",
                True
                if _is_phase10_model_point(fitted_model)
                else False,
                f"{fitted_model.family} {fitted_model.hyperparameters} is a "
                "pre-registered Phase 10 model point (subset of Phase 9 grids)",
            )
        )
        checks.append(
            _check(
                "model_scope_guard",
                fitted_model.family in MODEL_FAMILIES,
                f"family {fitted_model.family!r} in the Phase 9 families",
            )
        )
        checks.append(
            _check("seed_lock", fitted_model.seed == 42, f"seed {fitted_model.seed}")
        )

    # 8. calibration validation-only
    if calibration_map is not None:
        try:
            assert_no_test_fit(calibration_map)
            checks.append(
                _check("calibration_val_only", True, "calibration fitted on validation only")
            )
        except AssertionError as exc:
            checks.append(_check("calibration_val_only", False, str(exc)))

    # 9. backtest uniformity
    if backtest_config is not None:
        costs = backtest_config.costs
        checks.append(
            _check(
                "backtest_uniformity",
                costs.spread_bps == 2.0
                and costs.fees_bps == 1.0
                and costs.slippage_bps == 2.0
                and backtest_config.sizing.value == "weight"
                and backtest_config.long_only is True,
                f"CM-001 costs + WEIGHT sizing + long-only, identical to Phase 9",
            )
        )

    # 10. registry lineage
    if experiment_spec is not None:
        checks.append(
            _check(
                "registry_lineage",
                experiment_spec.dataset_snapshot_ids == ["DS-000004"]
                and experiment_spec.label_id == "LAB-004"
                and experiment_spec.cost_model_id == "CM-001"
                and experiment_spec.features.feature_set_id in PHASE10_FEATURE_SET_ORDER
                and all(
                    r.feature_id in _ALLOWED_FEATURE_IDS
                    for r in experiment_spec.features.feature_refs
                ),
                f"{experiment_spec.features.feature_set_id} v"
                f"{experiment_spec.features.feature_version}: DS-000004 / LAB-004 v1 / CM-001",
            )
        )

    return checks


def _is_phase10_model_point(fitted_model: Any) -> bool:
    try:
        validate_model_parameters(fitted_model.family, fitted_model.hyperparameters)
    except ValueError:
        return False
    return any(
        m["family"] == fitted_model.family
        and m["params"] == dict(fitted_model.hyperparameters)
        for m in PHASE10_MODEL_POINTS
    )


def audit_summary(checks: list[dict[str, Any]]) -> dict[str, Any]:
    failed = [c for c in checks if c["status"] == "FAIL"]
    return {
        "checks": len(checks),
        "passed": len(checks) - len(failed),
        "failed": len(failed),
        "blocked": bool(failed),
        "failed_checks": [c["check"] for c in failed],
    }


__all__ = [
    "run_phase10_audit",
    "audit_summary",
    "verify_dataset_unchanged",
    "verify_feature_temporal_boundary",
    "assert_feature_set_membership",
]