"""Phase 7 backtest configuration: every execution and cost assumption is
explicit, configurable, deterministic and recorded in the run manifest.

Two cost identities are kept separate on purpose:

  - `CostConfig` is the executable cost model of the Phase 7 simulator
    (spread, fees, slippage; direction-aware, per side). It can be built
    from the Phase 1 `CostModel` (schemas.common) so an experiment's
    `cost_model_id` remains the registered research identity.
  - `ExecutionConfig` is the execution semantics: which bar field is the
    fill price, how many sessions pass between a signal and order
    eligibility, the liquidity participation cap, partial-fill behavior,
    and order expiry.

The whole `BacktestConfig` is hashed into the run manifest (`config_hash`);
changing any assumption creates a distinct run identity. Phase 7 is
long-only: `long_only=False` is refused loudly at construction - short
selling is not part of the Phase 1-6 contract and is never silently
invented.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from orbit.schemas.common import CostModel


class ExecutionPrice(str, Enum):
    """Which bar field is the execution price of a market order.

    OPEN  - a market order fills at the OPEN of the fill session. This is
            the canonical EOD convention: an order placed after the close
            of session D executes at the next session's open. Because the
            open precedes the close, the execution delay must be >= 1
            session - a same-session open fill would be a look-ahead.
    CLOSE - a market order fills at the CLOSE of the fill session. The
            fill session is always `signal_session + execution_delay`: even
            with delay 0 the fill lands at the NEXT session's close (the
            signal session's own close is not a valid fill instant - the
            order is only submitted at that close, after the bar is known).
            The effective minimum delay is therefore 1 session, which
            keeps close fills a strictly conservative ideal fill under an
            explicit configuration, never the default.
    """

    OPEN = "open"
    CLOSE = "close"


class SizingPolicy(str, Enum):
    """How an order quantity is derived from a signal target.

    QUANTITY - the signal's `target` is a number of shares.
    WEIGHT   - the signal's `target` is a fraction of the portfolio equity
               at the previous valuation; the quantity is floored to whole
               shares. Note: flooring bounds the notional, but the real
               protection against over-spending is the executor's
               INSUFFICIENT_CASH rejection - a weight-sized order can still
               exceed its budget weight when the fill open gaps above the
               sizing close or fees are charged, and is then either
               rejected or fills into an overweight position.
    """

    QUANTITY = "quantity"
    WEIGHT = "weight"


class CostConfig(BaseModel):
    """Explicit, deterministic, direction-aware cost assumptions.

    Semantics (per side, matching the Phase 1 CostModel docstring):

      spread_bps      - buy fills pay +spread_bps/1e4 over the reference
                        price; sell fills receive -spread_bps/1e4. This is
                        the half-spread cost each side bears.
      fees_bps        - commission charged on the filled notional
                        (fill_quantity * reference price), per side.
      slippage_bps    - direction-aware execution impact: buys pay
                        +slippage_bps/1e4, sells receive -slippage_bps/1e4.
      fixed_fee_per_order - constant commission per filled order.
      fee_minimum     - the total fee of a fill is never below this.

    All values are non-negative; zero is the honest zero-cost baseline.
    """

    model_config = ConfigDict(frozen=True)

    spread_bps: float = Field(default=0.0, ge=0)
    fees_bps: float = Field(default=0.0, ge=0)
    slippage_bps: float = Field(default=0.0, ge=0)
    fixed_fee_per_order: float = Field(default=0.0, ge=0)
    fee_minimum: float = Field(default=0.0, ge=0)

    @classmethod
    def from_cost_model(cls, model: CostModel) -> "CostConfig":
        """The simulator's executable form of the Phase 1/6 CostModel."""
        return cls(
            spread_bps=model.spread_bps,
            fees_bps=model.fees_bps,
            slippage_bps=model.slippage_bps,
        )

    def is_zero(self) -> bool:
        return (
            self.spread_bps == 0.0
            and self.fees_bps == 0.0
            and self.slippage_bps == 0.0
            and self.fixed_fee_per_order == 0.0
            and self.fee_minimum == 0.0
        )


class ExecutionConfig(BaseModel):
    """Execution semantics of the simulator (deterministic, no randomness)."""

    model_config = ConfigDict(frozen=True)

    execution_price: ExecutionPrice = ExecutionPrice.OPEN
    execution_delay: int = Field(
        default=1,
        ge=0,
        description="sessions between the signal session and order eligibility; "
        "with OPEN fills the minimum is 1 (an open is always before a close)",
    )
    participation_fraction: float = Field(
        default=0.05,
        gt=0,
        le=1,
        description="maximum fill quantity = participation_fraction x the fill "
        "session's volume. Daily volume is a liquidity PROXY, never an exact "
        "measure of executable liquidity (documented limitation).",
    )
    max_order_quantity: float | None = Field(
        default=None,
        gt=0,
        description="absolute per-order quantity cap in addition to the "
        "participation cap, when set",
    )
    partial_fills: bool = Field(
        default=True,
        description="True: an order above the liquidity cap fills up to the cap "
        "and the remainder is recorded UNFILLED with an explicit reason. "
        "False: an order above the cap is rejected outright.",
    )
    order_expiry_sessions: int = Field(
        default=5,
        ge=0,
        description="sessions an eligible order may wait for an execution bar "
        "before it is rejected as EXPIRED_ORDER",
    )

    @model_validator(mode="after")
    def _open_fill_requires_delay(self) -> "ExecutionConfig":
        if (
            self.execution_price == ExecutionPrice.OPEN
            and self.execution_delay < 1
        ):
            raise ValueError(
                "execution_price=open requires execution_delay >= 1: an order "
                "placed after a session close can only fill at the NEXT "
                "session's open; a same-session open fill would be a "
                "look-ahead"
            )
        return self


class BacktestConfig(BaseModel):
    """The complete, immutable configuration of one backtest run."""

    model_config = ConfigDict(frozen=True)

    initial_cash: float = Field(default=1_000_000.0, gt=0)
    costs: CostConfig = Field(default_factory=CostConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    sizing: SizingPolicy = SizingPolicy.QUANTITY
    long_only: bool = Field(
        default=True,
        description="Phase 7 implements long-only accounting only; short "
        "selling is refused until a later phase explicitly defines and "
        "tests it",
    )
    valuation_price: Literal["close"] = "close"
    benchmark: str | None = Field(
        default=None,
        description="instrument_id valued separately from the trading "
        "portfolio (analytical comparison only, never a ledger transaction)",
    )
    seed: int = Field(default=42)
    randomness_policy: Literal["seeded", "nondeterministic"] = "seeded"
    window_start: date | None = None
    window_end: date | None = None

    @model_validator(mode="after")
    def _long_only_only(self) -> "BacktestConfig":
        if not self.long_only:
            raise ValueError(
                "Phase 7 is long-only: short-selling accounting is not "
                "implemented and is never silently invented. Keep "
                "long_only=True or implement and test short accounting in a "
                "later phase."
            )
        return self

    @model_validator(mode="after")
    def _window_order(self) -> "BacktestConfig":
        if (
            self.window_start is not None
            and self.window_end is not None
            and self.window_end < self.window_start
        ):
            raise ValueError("window_end cannot precede window_start")
        return self

    # ------------------------------------------------------------ identity

    def canonical_json(self) -> str:
        """Deterministic JSON of the full configuration (the identity)."""
        return json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        )

    def config_hash(self) -> str:
        """sha256 of the canonical configuration. Two runs with different
        assumptions have different hashes - a changed assumption is a
        distinct run, never a silent overwrite."""
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


def backtest_code_hash() -> str:
    """sha256 over the executing Phase 7 code (all modules of the backtest
    package), so the manifest can pin the exact code that produced a run.

    Deterministic: paths are sorted, contents are read as bytes, and the
    package root is resolved relative to this module.
    """
    import pathlib

    pkg_root = pathlib.Path(__file__).resolve().parent
    h = hashlib.sha256()
    for path in sorted(pkg_root.glob("*.py")):
        rel = str(path.relative_to(pkg_root))
        h.update(rel.encode("utf-8"))
        h.update(b"\x00")
        h.update(path.read_bytes())
        h.update(b"\x00")
    return h.hexdigest()


__all__ = [
    "BacktestConfig",
    "CostConfig",
    "ExecutionConfig",
    "ExecutionPrice",
    "SizingPolicy",
    "backtest_code_hash",
]