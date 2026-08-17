"""Instrument model: identity, symbol history, exchanges, corporate actions.

Identity is the core anti-bias control: an instrument_id is stable across
ticker changes, and symbol history is a first-class, time-aware fact so that
any evaluation date can resolve which symbol an instrument traded under.
"""

from __future__ import annotations

from datetime import date, datetime, time

from pydantic import BaseModel, ConfigDict, Field, model_validator

from orbit.schemas.common import SecurityType, UniverseScope


class Exchange(BaseModel):
    """Exchange metadata. Timezone matters for bar timestamps (Phase 3/4)."""

    model_config = ConfigDict(frozen=True)

    exchange_id: str = Field(pattern=r"^X[A-Z]{3}$")  # e.g. XNYS, XNAS
    name: str
    mic: str = Field(pattern=r"^[A-Z]{4}$")
    country: str
    tz: str = Field(description="IANA timezone, e.g. America/New_York")
    open_local: time
    close_local: time

    def local_to_utc(self, ts_local: datetime) -> datetime:
        """Convert a local timestamp to UTC. Raised in Phase 4 with as-of rules."""
        from zoneinfo import ZoneInfo

        return ts_local.replace(tzinfo=ZoneInfo(self.tz)).astimezone(ZoneInfo("UTC"))


class SectorTaxonomy(BaseModel):
    """GICS-style hierarchy. Depth-1 sectors; industry groups can extend."""

    model_config = ConfigDict(frozen=True)

    sector_id: str = Field(pattern=r"^S\d{2}$")  # e.g. S35
    name: str
    industry_group: str | None = None


class Instrument(BaseModel):
    """A tradable instrument with a stable identity through time."""

    model_config = ConfigDict(frozen=True)

    instrument_id: str = Field(pattern=r"^INS-\d{6}$")
    primary_ticker: str
    exchange_id: str = Field(pattern=r"^X[A-Z]{3}$")
    name: str
    security_type: SecurityType
    universe_class: UniverseScope = UniverseScope.LIQUID_EQUITY_50_100

    listing_date: date
    delisting_date: date | None = None
    delisting_reason: str | None = Field(
        default=None,
        description="e.g. merger, bankruptcy, voluntary, regulatory",
    )

    sector_id: str | None = None
    currency: str = "USD"

    @property
    def is_active(self) -> bool:
        return self.delisting_date is None

    @model_validator(mode="after")
    def _check_delisting(self) -> "Instrument":
        if (
            self.delisting_date is not None
            and self.delisting_date < self.listing_date
        ):
            raise ValueError("delisting_date cannot precede listing_date")
        if self.delisting_reason is not None and self.delisting_date is None:
            raise ValueError("delisting_reason requires delisting_date")
        return self


class SymbolHistory(BaseModel):
    """Every ticker an instrument has traded under, with effective dates."""

    model_config = ConfigDict(frozen=True)

    instrument_id: str = Field(pattern=r"^INS-\d{6}$")
    symbol: str
    effective_from: date
    effective_to: date | None = None

    @model_validator(mode="after")
    def _check_dates(self) -> "SymbolHistory":
        if self.effective_to is not None and self.effective_to < self.effective_from:
            raise ValueError("effective_to cannot precede effective_from")
        return self

    def covers(self, d: date) -> bool:
        return self.effective_from <= d and (
            self.effective_to is None or d <= self.effective_to
        )


class SymbolHistoryRegistry(BaseModel):
    """Validated symbol history for the whole master.

    Identity errors are a Phase 2 failure condition, so overlaps and
    out-of-order periods are rejected at the registry level: a ticker can
    never be ambiguous for any date.
    """

    model_config = ConfigDict(frozen=True)

    entries: list[SymbolHistory] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_no_overlaps(self) -> "SymbolHistoryRegistry":
        by_inst: dict[str, list[SymbolHistory]] = {}
        for e in self.entries:
            by_inst.setdefault(e.instrument_id, []).append(e)
        for inst_id, hist in by_inst.items():
            ordered = sorted(hist, key=lambda h: h.effective_from)
            for prev, nxt in zip(ordered, ordered[1:]):
                if prev.effective_to is not None and nxt.effective_from <= prev.effective_to:
                    raise ValueError(
                        f"overlapping symbol history for {inst_id}: "
                        f"{prev.symbol}({prev.effective_from}..{prev.effective_to}) vs "
                        f"{nxt.symbol}({nxt.effective_from}..{nxt.effective_to})"
                    )
                if prev.effective_to is None:
                    raise ValueError(
                        f"open-ended history for {inst_id} followed by another entry"
                    )
        return self

    def resolve(self, instrument_id: str, as_of: date) -> str | None:
        for e in self.entries:
            if e.instrument_id == instrument_id and e.covers(as_of):
                return e.symbol
        return None


class CorporateAction(BaseModel):
    """Corporate actions that affect returns, prices or identity."""

    model_config = ConfigDict(frozen=True)

    action_id: str = Field(pattern=r"^CA-\d{6}$")
    instrument_id: str = Field(pattern=r"^INS-\d{6}$")
    action_type: str = Field(
        description="split | reverse_split | dividend | merger | spin_off | delisting | symbol_change | name_change"
    )
    effective_date: date
    ex_date: date | None = None
    ratio: float | None = Field(
        default=None,
        description="split ratio (new/old shares) or dividend per share",
    )
    note: str | None = None
    source: str | None = None

    @model_validator(mode="after")
    def _check_ratio(self) -> "CorporateAction":
        if self.action_type in ("split", "reverse_split") and (
            self.ratio is None or self.ratio <= 0
        ):
            raise ValueError("split actions require a positive ratio")
        return self


class Benchmark(BaseModel):
    """Benchmark instruments referenced by hypotheses and evaluation."""

    model_config = ConfigDict(frozen=True)

    benchmark_id: str = Field(pattern=r"^BENCH-\d{3}$")
    instrument_id: str = Field(pattern=r"^INS-\d{6}$")
    name: str
    category: str = Field(
        description="broad | sector | style | risk_free"
    )
    sector_id: str | None = None