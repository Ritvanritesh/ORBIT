"""Phase 11.1 benchmark ingestion.

Downloads benchmark data (SPY) using the existing Yahoo Chart API pipeline,
normalizes it to ORBIT's standard bar schema, and persists it as a
first-class research artifact with its own manifest, checksum, and lineage.

The benchmark series is stored separately from instrument data and is NOT
automatically part of the tradable universe.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

from orbit.ingestion.paths import data_root
from orbit.ml.phase11_1_benchmark import BENCH_001_CONFIG, BenchmarkManifest

_REPO_ROOT = Path(__file__).resolve().parents[3]
BENCHMARK_RAW_DIR = data_root() / "raw" / "benchmark"
BENCHMARK_NORM_DIR = data_root() / "normalized" / "benchmark"
BENCHMARK_MANIFESTS_DIR = data_root() / "manifests" / "benchmark"


def _ensure_dirs() -> None:
    for d in (BENCHMARK_RAW_DIR, BENCHMARK_NORM_DIR, BENCHMARK_MANIFESTS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def download_benchmark_raw(symbol: str = "SPY", snapshot_id: str = "BENCH-001") -> Path:
    """Download benchmark data using Yahoo Chart API. Returns path to raw JSON."""
    import requests
    _ensure_dirs()
    out_dir = BENCHMARK_RAW_DIR / snapshot_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{symbol}.json"
    if out_file.exists():
        return out_file
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    params = {"period1": 0, "period2": int(datetime.now().timestamp()),
              "interval": "1d", "events": "history"}
    headers = {"User-Agent": "ORBIT-Research/1.0"}
    resp = requests.get(url, params=params, headers=headers, timeout=60)
    resp.raise_for_status()
    out_file.write_text(resp.text, encoding="utf-8")
    return out_file


def parse_benchmark_yahoo(raw_path: Path) -> pl.DataFrame:
    """Parse Yahoo Chart API JSON into ORBIT-format bars."""
    data = json.loads(raw_path.read_text(encoding="utf-8"))
    result = data["chart"]["result"][0]
    timestamps = result["timestamp"]
    quote = result["indicators"]["quote"][0]
    df = pl.DataFrame({
        "trade_date": [datetime.fromtimestamp(t).date() for t in timestamps],
        "open": quote["open"], "high": quote["high"],
        "low": quote["low"], "close": quote["close"], "volume": quote["volume"],
    })
    df = df.drop_nulls(subset=["close"])
    df = df.sort("trade_date")
    return df


def normalize_benchmark(raw_df: pl.DataFrame, benchmark_id: str = "BENCH-001",
                        symbol: str = "SPY") -> pl.DataFrame:
    """Normalize benchmark bars to ORBIT standard schema."""
    df = raw_df.with_columns([
        pl.lit(benchmark_id).alias("instrument_id"),
        pl.lit(symbol).alias("symbol"),
    ])
    cols = ["trade_date", "instrument_id", "symbol", "open", "high", "low", "close", "volume"]
    return df.select(cols)


def compute_benchmark_checksum(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def load_benchmark_bars(snapshot_id: str = "BENCH-001") -> pl.DataFrame | None:
    """Load persisted benchmark bars. Returns None if not yet ingested."""
    norm_path = BENCHMARK_NORM_DIR / snapshot_id / "bars.parquet"
    if not norm_path.exists():
        return None
    return pl.read_parquet(norm_path)


def load_benchmark_manifest(snapshot_id: str = "BENCH-001") -> dict[str, Any] | None:
    """Load the benchmark manifest. Returns None if not yet created."""
    manifest_path = BENCHMARK_MANIFESTS_DIR / f"{snapshot_id}.json"
    if not manifest_path.exists():
        return None
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def ingest_benchmark(config: Any = BENCH_001_CONFIG,
                     snapshot_id: str = "BENCH-001") -> BenchmarkManifest:
    """Full ingestion pipeline for a benchmark series."""
    _ensure_dirs()
    raw_path = download_benchmark_raw(config.benchmark_symbol, snapshot_id)
    raw_df = parse_benchmark_yahoo(raw_path)
    norm_df = normalize_benchmark(raw_df, config.benchmark_id, config.benchmark_symbol)
    norm_dir = BENCHMARK_NORM_DIR / snapshot_id
    norm_dir.mkdir(parents=True, exist_ok=True)
    norm_path = norm_dir / "bars.parquet"
    norm_df.write_parquet(norm_path)
    checksum = compute_benchmark_checksum(norm_path)
    n_rows = norm_df.height
    n_sessions = norm_df["trade_date"].n_unique()
    date_min = str(norm_df["trade_date"].min())
    date_max = str(norm_df["trade_date"].max())
    null_close = norm_df["close"].null_count()
    validation_status = "warning" if null_close > 0 else "ok"
    notes = f"{null_close} null close values dropped" if null_close > 0 else None
    manifest = BenchmarkManifest(
        benchmark_id=config.benchmark_id, benchmark_symbol=config.benchmark_symbol,
        snapshot_id=snapshot_id, source=config.source, ingestion_time=datetime.now(),
        date_range=[date_min, date_max], row_count=n_rows, session_count=n_sessions,
        checksum=checksum, config_hash=config.content_hash(),
        validation_status=validation_status, notes=notes,
    )
    manifest_path = BENCHMARK_MANIFESTS_DIR / f"{snapshot_id}.json"
    manifest_path.write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    return manifest


__all__ = [
    "BENCHMARK_RAW_DIR", "BENCHMARK_NORM_DIR", "BENCHMARK_MANIFESTS_DIR",
    "download_benchmark_raw", "parse_benchmark_yahoo", "normalize_benchmark",
    "compute_benchmark_checksum", "load_benchmark_bars", "load_benchmark_manifest",
    "ingest_benchmark",
]
