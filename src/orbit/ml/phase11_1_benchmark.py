"""Phase 11.1 benchmark configuration.

Defines the benchmark as first-class research data with explicit role,
source lineage, alignment policy, and return convention. The benchmark
is NOT part of the tradable stock universe unless explicitly configured.

Benchmark identity follows ORBIT conventions: BENCH-NNN pattern.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class BenchmarkRole(str, Enum):
    """How the benchmark series is used in research."""

    BROAD_MARKET = "broad_market"
    SECTOR_BENCHMARK = "sector_benchmark"
    REFERENCE_INDEX = "reference_index"


class ReturnDefinition(str, Enum):
    """How benchmark returns are computed."""

    SIMPLE_TOTAL_RETURN = "simple_total_return"
    SIMPLE_PRICE_RETURN = "simple_price_return"


class AlignmentPolicy(str, Enum):
    """How benchmark observations align with instrument observations."""

    SAME_DAY = "same_day"
    LAGGED_1_DAY = "lagged_1_day"


class AdjustedPricePolicy(str, Enum):
    """How benchmark prices are adjusted for corporate actions."""

    SPLIT_CONTINUOUS = "split_continuous"
    DIVIDEND_ADJUSTED = "dividend_adjusted"
    RAW = "raw"


class BenchmarkConfig(BaseModel):
    """Complete, immutable benchmark configuration."""

    model_config = ConfigDict(frozen=True)

    benchmark_id: str = Field(pattern=r"^BENCH-\d{3}$")
    benchmark_symbol: str = Field(description="Yahoo Finance ticker symbol")
    benchmark_role: BenchmarkRole
    market: str = Field(description="e.g. US, EU, Asia")
    currency: str = Field(description="ISO 4217 currency code")
    return_definition: ReturnDefinition
    alignment_policy: AlignmentPolicy
    adjusted_price_policy: AdjustedPricePolicy
    source: str = Field(description="Data provider identifier")
    source_version: str | None = None
    description: str | None = None
    author: str = "orbit-research"

    def canonical_json(self) -> str:
        return json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        )

    def content_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    def summary(self) -> dict[str, Any]:
        return {
            "benchmark_id": self.benchmark_id,
            "benchmark_symbol": self.benchmark_symbol,
            "benchmark_role": self.benchmark_role.value,
            "market": self.market,
            "currency": self.currency,
            "return_definition": self.return_definition.value,
            "alignment_policy": self.alignment_policy.value,
            "adjusted_price_policy": self.adjusted_price_policy.value,
            "source": self.source,
            "content_hash": self.content_hash(),
        }


class BenchmarkManifest(BaseModel):
    """Immutable manifest for a benchmark data artifact."""

    model_config = ConfigDict(frozen=True)

    benchmark_id: str
    benchmark_symbol: str
    snapshot_id: str
    schema_version: str = "v1"
    source: str
    source_version: str | None = None
    ingestion_time: datetime
    date_range: list[str] = Field(min_length=2, max_length=2)
    row_count: int
    session_count: int
    checksum: str = Field(min_length=32)
    config_hash: str = Field(min_length=32)
    validation_status: str = "ok"
    notes: str | None = None

    def summary(self) -> dict[str, Any]:
        return {
            "benchmark_id": self.benchmark_id,
            "benchmark_symbol": self.benchmark_symbol,
            "snapshot_id": self.snapshot_id,
            "date_range": self.date_range,
            "row_count": self.row_count,
            "session_count": self.session_count,
            "checksum": self.checksum[:16] + "...",
            "config_hash": self.config_hash[:16] + "...",
            "validation_status": self.validation_status,
        }


# ──────────────────────────────────────────────────────────────
# LOCKED CONFIGURATION
# ──────────────────────────────────────────────────────────────

BENCH_001_CONFIG = BenchmarkConfig(
    benchmark_id="BENCH-001",
    benchmark_symbol="SPY",
    benchmark_role=BenchmarkRole.BROAD_MARKET,
    market="US",
    currency="USD",
    return_definition=ReturnDefinition.SIMPLE_TOTAL_RETURN,
    alignment_policy=AlignmentPolicy.SAME_DAY,
    adjusted_price_policy=AdjustedPricePolicy.SPLIT_CONTINUOUS,
    source="yahoo_chart_api",
    description=(
        "SPDR S&P 500 ETF Trust (SPY) - broad U.S. equity market benchmark. "
        "Used as the benchmark for excess-return label construction in Phase 11.1. "
        "NOT part of the tradable stock universe."
    ),
)


def lock_benchmark_config() -> dict[str, Any]:
    """Persist and return the locked benchmark configuration."""
    return {
        "phase": "11.1",
        "protocol": "phase11_1_benchmark_v1",
        "config": BENCH_001_CONFIG.model_dump(mode="json"),
        "config_hash": BENCH_001_CONFIG.content_hash(),
        "locked_at": datetime.now().isoformat(timespec="seconds"),
        "notes": (
            "Phase 11.1 locked benchmark configuration. "
            "BENCH-001 (SPY) is the sole broad-market benchmark. "
            "This configuration must not change after results exist."
        ),
    }


__all__ = [
    "BenchmarkRole",
    "ReturnDefinition",
    "AlignmentPolicy",
    "AdjustedPricePolicy",
    "BenchmarkConfig",
    "BenchmarkManifest",
    "BENCH_001_CONFIG",
    "lock_benchmark_config",
]
