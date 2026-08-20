"""Phase 7 - the event-driven backtesting engine.

Deterministic, replayable execution + accounting simulation for the
daily/EOD, long-only ORBIT research MVP. See `backtester.py`, `clock.py`,
`execution.py`, `ledger.py`, `orders.py`, `manifest.py`, `result.py` and
`integration.py` for the module-level contracts.
"""

from orbit.backtest.backtester import Backtester
from orbit.backtest.config import (
    BacktestConfig,
    CostConfig,
    ExecutionConfig,
    ExecutionPrice,
    SizingPolicy,
    backtest_code_hash,
)
from orbit.backtest.clock import CANONICAL_PRICE_BASIS, MarketEventClock
from orbit.backtest.events import (
    EventType,
    FailureKind,
    OrderSide,
    OrderStatus,
    SignalDirection,
)
from orbit.backtest.ledger import PortfolioLedger
from orbit.backtest.manifest import BacktestManifest, build_manifest
from orbit.backtest.result import BacktestResult

__all__ = [
    "BacktestConfig",
    "BacktestManifest",
    "BacktestResult",
    "Backtester",
    "CANONICAL_PRICE_BASIS",
    "CostConfig",
    "EventType",
    "ExecutionConfig",
    "ExecutionPrice",
    "FailureKind",
    "MarketEventClock",
    "OrderSide",
    "OrderStatus",
    "PortfolioLedger",
    "SignalDirection",
    "SizingPolicy",
    "backtest_code_hash",
    "build_manifest",
]
