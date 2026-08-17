"""Data access interface for the universe engine.

Phase 3+ will implement this over the raw immutable data layer. For Phase 2
the engine is tested against a synthetic in-memory accessor so the selection
logic is proven before any real data is licensed or ingested.
"""

from __future__ import annotations

from datetime import date
from typing import Protocol

from orbit.schemas.instrument import Instrument


class DataAccessor(Protocol):
    """Read-only access to instrument metadata and lagged market data.

    All market data returned MUST be strictly before as_of (lagged).
    Phase 4's temporal-truth engine owns the adversarial leak tests that
    prove an accessor obeys this; Phase 2's engine trusts the contract and
    is tested against a synthetic accessor that enforces it.
    """

    def instruments(self) -> list[Instrument]:
        """Full instrument master (including delisted names)."""

    def trailing_dollar_volume(
        self, instrument_id: str, as_of: date, window_days: int
    ) -> float | None:
        """Median daily dollar volume over the window ending strictly before as_of.

        Returns None if the instrument has no bars in the window.
        """

    def last_close(self, instrument_id: str, as_of: date) -> float | None:
        """Most recent close strictly before as_of. None if no history."""