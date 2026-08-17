"""Universe reconstruction engine (Phase 2).

Given a rule and a data accessor, rebuild the membership that was knowable
for any evaluation date: listed-and-not-yet-delisted instruments that pass
lagged price and liquidity filters. Survivorship bias is structurally
impossible here because the instrument master includes delisted names and
the engine never uses post-as_of information.
"""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, ConfigDict, Field

from orbit.schemas.instrument import Instrument, SymbolHistory
from orbit.universe.accessor import DataAccessor
from orbit.universe.rules import MembershipRule


class UniverseMember(BaseModel):
    """A single instrument in a reconstructed universe snapshot."""

    model_config = ConfigDict(frozen=True)

    instrument_id: str
    symbol_at_asof: str | None = Field(
        description="Ticker the instrument traded under on as_of (None = unresolved)."
    )
    rank: int
    trailing_dollar_volume: float
    last_close: float


class Exclusion(BaseModel):
    """Why an instrument was excluded - full auditability."""

    model_config = ConfigDict(frozen=True)

    instrument_id: str
    reason: str


class UniverseSnapshot(BaseModel):
    """The complete, reconstructable membership for one evaluation date."""

    model_config = ConfigDict(frozen=True)

    as_of: date
    rule: MembershipRule
    data_ref: str | None = Field(
        default=None,
        description="Dataset snapshot / accessor version the membership was computed from.",
    )
    members: list[UniverseMember] = Field(default_factory=list)
    excluded: list[Exclusion] = Field(default_factory=list)

    @property
    def instrument_ids(self) -> list[str]:
        return [m.instrument_id for m in self.members]


class UniverseEngine:
    """Deterministic membership reconstruction. Same inputs -> same snapshot."""

    def __init__(
        self,
        accessor: DataAccessor,
        rule: MembershipRule,
        data_ref: str | None = None,
    ):
        self.accessor = accessor
        self.rule = rule
        self.data_ref = data_ref

    def membership(self, as_of: date) -> UniverseSnapshot:
        members: list[UniverseMember] = []
        excluded: list[Exclusion] = []

        for inst in self.accessor.instruments():
            if not self._eligible(inst, as_of):
                excluded.append(
                    Exclusion(instrument_id=inst.instrument_id, reason="not_listed_or_delisted")
                )
                continue
            if inst.security_type not in self.rule.security_types:
                excluded.append(
                    Exclusion(
                        instrument_id=inst.instrument_id,
                        reason=f"security_type={inst.security_type.value}",
                    )
                )
                continue
            if inst.exchange_id not in self.rule.exchanges:
                excluded.append(
                    Exclusion(instrument_id=inst.instrument_id, reason=f"exchange={inst.exchange_id}")
                )
                continue

            dv = self.accessor.trailing_dollar_volume(
                inst.instrument_id, as_of, self.rule.liquidity_window_days
            )
            close = self.accessor.last_close(inst.instrument_id, as_of)
            if dv is None or close is None:
                excluded.append(
                    Exclusion(instrument_id=inst.instrument_id, reason="no_lagged_price_history")
                )
                continue
            if self.rule.min_price is not None and close < self.rule.min_price:
                excluded.append(
                    Exclusion(instrument_id=inst.instrument_id, reason=f"price={close:.2f}<{self.rule.min_price}")
                )
                continue
            if self.rule.min_trailing_dollar_volume is not None and dv < self.rule.min_trailing_dollar_volume:
                excluded.append(
                    Exclusion(
                        instrument_id=inst.instrument_id,
                        reason=f"dollar_volume={dv:,.0f}<{self.rule.min_trailing_dollar_volume:,.0f}",
                    )
                )
                continue

            members.append(
                UniverseMember(
                    instrument_id=inst.instrument_id,
                    symbol_at_asof=self._resolve_symbol(inst, as_of),
                    rank=0,  # assigned after sort
                    trailing_dollar_volume=dv,
                    last_close=close,
                )
            )

        members.sort(
            key=lambda m: (-m.trailing_dollar_volume, m.instrument_id)
        )
        if self.rule.max_names is not None:
            cut = members[self.rule.max_names :]
            members = members[: self.rule.max_names]
            excluded.extend(
                Exclusion(instrument_id=m.instrument_id, reason="below_liquidity_cap")
                for m in cut
            )
        members = [
            m.model_copy(update={"rank": i + 1}) for i, m in enumerate(members)
        ]

        return UniverseSnapshot(
            as_of=as_of, rule=self.rule, data_ref=self.data_ref,
            members=members, excluded=excluded,
        )

    def _eligible(self, inst: Instrument, as_of: date) -> bool:
        if inst.listing_date > as_of:
            return False
        if inst.delisting_date is not None and inst.delisting_date <= as_of:
            return False
        return True

    def _resolve_symbol(self, inst: Instrument, as_of: date) -> str | None:
        history = getattr(self.accessor, "symbol_history", None)
        if history is None:
            return inst.primary_ticker
        if callable(history):
            entries = history(inst.instrument_id)
        else:
            entries = history.get(inst.instrument_id, [])
        for entry in entries:
            if entry.covers(as_of):
                return entry.symbol
        return None