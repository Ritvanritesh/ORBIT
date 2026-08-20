"""Digest-verified snapshot cache for the Phase 9 benchmark.

The FS-001 feature snapshot and the LAB-004 label snapshot are pure
deterministic functions of the DS-000004 artifacts (they take ~90s to
compute), so the runner caches their record frames to
data/cache/phase9_snapshots/. Loading verifies the sha256 content digest
against the stored digest: a cache entry whose source data or code changed
produces a digest mismatch and is rebuilt, never silently reused.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl

from orbit.ml.features import FEATURE_SET_ID, FEATURE_SET_VERSION, FeatureSnapshot
from orbit.ml.labels import LABEL_ID, LABEL_VERSION, build_phase9_label_snapshot

SNAPSHOT_CACHE_DIR = Path(__file__).resolve().parents[3] / "data" / "cache" / "phase9_snapshots"


def _write(dir_path: Path, name: str, records: pl.DataFrame, meta: dict[str, Any]) -> Path:
    dir_path.mkdir(parents=True, exist_ok=True)
    records.write_parquet(dir_path / f"{name}_records.parquet")
    (dir_path / f"{name}_meta.json").write_text(
        json.dumps(meta, sort_keys=True, indent=2), encoding="utf-8"
    )
    return dir_path


def _read(dir_path: Path, name: str) -> tuple[pl.DataFrame, dict[str, Any]]:
    records = pl.read_parquet(dir_path / f"{name}_records.parquet")
    meta = json.loads((dir_path / f"{name}_meta.json").read_text(encoding="utf-8"))
    return records, meta


def cache_feature_snapshot(
    snapshot: FeatureSnapshot, dir_path: Path = SNAPSHOT_CACHE_DIR
) -> Path:
    meta = snapshot.provenance()
    meta["content_digest"] = snapshot.content_digest
    return _write(dir_path, "feature_snapshot", snapshot.records, meta)


def load_cached_feature_snapshot(
    dir_path: Path = SNAPSHOT_CACHE_DIR,
) -> FeatureSnapshot | None:
    if not (dir_path / "feature_snapshot_records.parquet").exists():
        return None
    records, meta = _read(dir_path, "feature_snapshot")
    snapshot = FeatureSnapshot(
        feature_set_id=meta["feature_set_id"],
        feature_set_version=meta["feature_set_version"],
        feature_refs=meta["feature_refs"],
        data_refs=meta["data_refs"],
        records=records,
        transformation=meta["transformation"],
        limitations=meta.get("limitations") or [],
    )
    if snapshot.content_digest != meta["content_digest"]:
        raise RuntimeError(
            "cached feature snapshot digest mismatch - source data or code "
            "changed; delete data/cache/phase9_snapshots to rebuild"
        )
    return snapshot


def cache_label_snapshot(snapshot: Any, dir_path: Path = SNAPSHOT_CACHE_DIR) -> Path:
    meta = snapshot.provenance()
    return _write(dir_path, "label_snapshot", snapshot.records, meta)


def load_cached_label_snapshot(
    dir_path: Path = SNAPSHOT_CACHE_DIR,
) -> Any | None:
    if not (dir_path / "label_snapshot_records.parquet").exists():
        return None
    records, meta = _read(dir_path, "label_snapshot")
    from orbit.labels.snapshot import LabelSnapshot

    snapshot = LabelSnapshot(
        label_id=meta["label_id"],
        version=meta["version"],
        contract_digest=meta["contract_digest"],
        engine_version=meta["engine_version"],
        data_refs=meta["data_refs"],
        records=records,
        limitations=meta.get("limitations") or [],
    )
    if snapshot.content_digest != meta["content_digest"]:
        raise RuntimeError(
            "cached label snapshot digest mismatch - source data or code "
            "changed; delete data/cache/phase9_snapshots to rebuild"
        )
    return snapshot


def build_or_load_snapshots(
    bars: pl.DataFrame,
    events: pl.DataFrame | None,
    instruments: list,
    *,
    force_rebuild: bool = False,
    dir_path: Path = SNAPSHOT_CACHE_DIR,
) -> tuple[FeatureSnapshot, Any]:
    """Cache-aware snapshot build; returns (feature_snapshot, label_snapshot)."""
    from orbit.ml.features import build_feature_snapshot

    if not force_rebuild:
        fs = load_cached_feature_snapshot(dir_path)
        ls = load_cached_label_snapshot(dir_path)
        if fs is not None and ls is not None:
            return fs, ls

    fs = build_feature_snapshot(bars, data_refs=["DS-000004"])
    decisions = fs.records.select("instrument_id", "decision_time")
    ls = build_phase9_label_snapshot(bars, events, instruments, decisions)
    cache_feature_snapshot(fs, dir_path)
    cache_label_snapshot(ls, dir_path)
    return fs, ls


PHASE10_SNAPSHOT_CACHE_DIR = Path(__file__).resolve().parents[3] / "data" / "cache" / "phase10_snapshots"


def cache_phase10_snapshot(
    snapshot: FeatureSnapshot, dir_path: Path = PHASE10_SNAPSHOT_CACHE_DIR
) -> Path:
    """Cache one Phase 10 feature-set snapshot (digest-verified on load)."""
    meta = snapshot.provenance()
    meta["content_digest"] = snapshot.content_digest
    return _write(dir_path, f"feature_set_{snapshot.feature_set_id}", snapshot.records, meta)


def load_cached_phase10_snapshot(
    feature_set_id: str, dir_path: Path = PHASE10_SNAPSHOT_CACHE_DIR
) -> FeatureSnapshot | None:
    name = f"feature_set_{feature_set_id}"
    if not (dir_path / f"{name}_records.parquet").exists():
        return None
    records, meta = _read(dir_path, name)
    snapshot = FeatureSnapshot(
        feature_set_id=meta["feature_set_id"],
        feature_set_version=meta["feature_set_version"],
        feature_refs=meta["feature_refs"],
        data_refs=meta["data_refs"],
        records=records,
        transformation=meta["transformation"],
        limitations=meta.get("limitations") or [],
    )
    if snapshot.content_digest != meta["content_digest"]:
        raise RuntimeError(
            f"cached Phase 10 snapshot {feature_set_id} digest mismatch - "
            "source data or code changed; delete data/cache/phase10_snapshots "
            "to rebuild"
        )
    return snapshot


__all__ = [
    "SNAPSHOT_CACHE_DIR",
    "cache_feature_snapshot",
    "load_cached_feature_snapshot",
    "cache_label_snapshot",
    "load_cached_label_snapshot",
    "build_or_load_snapshots",
    "PHASE10_SNAPSHOT_CACHE_DIR",
    "cache_phase10_snapshot",
    "load_cached_phase10_snapshot",
]