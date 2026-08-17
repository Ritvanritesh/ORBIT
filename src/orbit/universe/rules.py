"""Membership rules: deterministic, versioned universe selection logic."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from orbit.schemas.common import UniverseScope


class MembershipRule(BaseModel):
    """One versioned universe-selection rule.

    Rules are immutable and versioned: any change to selection logic is a new
    rule version, so membership is reconstructable for any evaluation date.
    All data inputs are lagged (strictly before as_of) - no look-ahead.
    """

    model_config = ConfigDict(frozen=True)

    rule_id: str = Field(pattern=r"^RULE-\d{3}$")
    version: str = Field(pattern=r"^v\d+(\.\d+)*$")
    universe_class: UniverseScope = UniverseScope.LIQUID_EQUITY_50_100

    security_types: list[str] = Field(
        default_factory=lambda: ["equity"],
        description="Allowed security types (e.g. equity, etf).",
    )
    exchanges: list[str] | None = Field(
        default=None,
        description="Allowed exchange_ids; None = all.",
    )
    min_price: float | None = Field(
        default=5.0,
        gt=0,
        description="Minimum last close (lagged). None = no price floor.",
    )
    min_trailing_dollar_volume: float | None = Field(
        default=20_000_000.0,
        gt=0,
        description="Minimum median trailing dollar volume over the window (lagged).",
    )
    liquidity_window_days: int = Field(default=20, ge=1)
    max_names: int | None = Field(
        default=100,
        ge=1,
        description="Top-N by liquidity; None = no cap.",
    )