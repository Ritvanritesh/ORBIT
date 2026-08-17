"""Shared enums and value types for ORBIT research schemas."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class CostModel(BaseModel):
    """Explicit, conservative execution cost assumptions (per side)."""

    spread_bps: float = Field(default=2.0, ge=0)
    fees_bps: float = Field(default=1.0, ge=0)
    slippage_bps: float = Field(default=2.0, ge=0)

    def total_bps(self) -> float:
        return self.spread_bps + self.fees_bps + self.slippage_bps


class Horizon(str, Enum):
    """Label horizons. Daily/end-of-day first per charter scope."""

    H1 = "1D"
    H5 = "5D"
    H21 = "21D"
    H63 = "63D"


class LabelType(str, Enum):
    """Prediction target families."""

    FORWARD_RETURN = "forward_return"
    EXCESS_RETURN = "excess_return"          # relative to benchmark
    RISK_ADJUSTED_RETURN = "risk_adjusted_return"
    VOLATILITY = "volatility"
    DRAWDOWN = "drawdown"


class UniverseScope(str, Enum):
    """Universe philosophy buckets from the charter."""

    LIQUID_EQUITY_50_100 = "liquid_equity_50_100"
    SECTOR_ETF = "sector_etf"
    BROAD_ETF = "broad_etf"
    SNAPSHOT_RECONSTRUCTED = "snapshot_reconstructed"


class SecurityType(str, Enum):
    """Security types the instrument master distinguishes."""

    EQUITY = "equity"
    ETF = "etf"
    BENCHMARK = "benchmark"
    PREFERRED = "preferred"
    ADR = "adr"
    UNIT = "unit"


class LeakageClass(str, Enum):
    """Leakage class a feature family must defend against.

    Declared at hypothesis registration so Phase 4's temporal-truth engine
    knows which adversarial fixtures to apply (synthetic future-leak tests).
    A NONE declaration still requires the standard leak fixtures to pass.
    """

    NONE = "none"
    FUTURE_PUBLICATION = "future_publication"   # info published after t
    SURVIVORSHIP = "survivorship"
    REVISED_DATA = "revised_data"               # revised macro/fundamental values
    VENDOR_TIMESTAMP_GAP = "vendor_timestamp_gap"


class EvidenceType(str, Enum):
    """Economic vs research-quality evidence (charter, roadmap 12)."""

    RESEARCH_QUALITY = "research_quality"   # statistical, not necessarily tradable
    ECONOMIC = "economic"                   # survives costs, risk-adjusted, OOS


class ExperimentStatus(str, Enum):
    """Lifecycle per roadmap governance."""

    DRAFT = "draft"
    REGISTERED = "registered"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETIRED = "retired"


class HypothesisStatus(str, Enum):
    """Lifecycle per roadmap governance."""

    DRAFT = "draft"
    PROPOSED = "proposed"
    REGISTERED = "registered"
    ACTIVE = "active"
    FALSIFIED = "falsified"
    ABANDONED = "abandoned"
    PROMOTED = "promoted"