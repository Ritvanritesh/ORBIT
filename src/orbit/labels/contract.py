"""Label contracts: the immutable, versioned definition of every prediction
target (Phase 5).

A `LabelContract` is the complete, machine-readable promise that defines ONE
prediction target:

    - what is being predicted (target_type)
    - over which horizon (horizon trading sessions, never calendar days)
    - how the outcome is measured (return convention, estimator, drawdown
      definition)
    - which prices are used (close, on the canonical split-continuous basis)
    - how the outcome window is anchored (decision instant or publication
      event instant)
    - what happens when data is missing, the instrument delists, or a
      corporate action occurs (explicit policies - never silent)
    - which benchmark the excess return is measured against

Semantics-bearing fields are REQUIRED (no hidden defaults that could change
between experiments): target_type, horizon, anchor_mode, return_convention
(where relevant), benchmark (excess), drawdown_type (drawdown),
volatility_estimator / annualization / min_observations (volatility), and
formula. The remaining fields are FIXED constants naming the single
implemented convention and are part of the content identity anyway.

The contract is frozen at construction. It is immutable once registered in a
LabelVersionRegistry; a changed definition is a NEW VERSION, never a mutation
of an existing one. `content_hash` is the formula/config identity recorded in
every label row and experiment metadata.

The Phase 1 `orbit.schemas.hypothesis.LabelSpec` remains the hypothesis-side
summary (frozen, part of HypothesisSpec); `contract_from_hypothesis_label()`
bridges a Phase 1 label into the operational Phase 5 contract.

Not implemented in Phase 5: RISK_ADJUSTED_RETURN (a composite label that
combines a future outcome with a point-in-time trailing-volatility
denominator; the components are implemented here, the composite is assembled
in a later phase) - the validator rejects it loudly so it can never be used
with silent semantics.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from orbit.schemas.common import Horizon, LabelType

# Documented constant: trading sessions per calendar year for US equities
# (the annualization ORBIT uses for volatility outcomes).
SESSION_ANNUALIZATION = 252.0


class AnchorMode(str, Enum):
    """How the label's reference session is chosen.

    DECISION_INSTANT - the reference session is the LAST completed session
        strictly before the decision instant (a daily bar for session D is
        completed at the session close, 16:00 America/New_York; a decision at
        exactly the close does NOT see that day's bar - strict boundary, the
        Phase 4 convention).
    POST_EVENT       - the reference session is the FIRST completed session
        strictly AFTER the event's point-in-time availability instant (e.g.
        a filing becomes available the day after its filed date; the entry
        is the first session closing after that instant). This is the PEAD
        anchor (seed H-003): "from the trading day after the earliest
        point-in-time publication timestamp".
    """

    DECISION_INSTANT = "decision_instant"
    POST_EVENT = "post_event"


class ReturnConvention(str, Enum):
    """How a forward/excess return is computed.

    SIMPLE_PRICE_RETURN - close-to-close price return on the canonical
        split-continuous basis: r = P(H)/P(0) - 1 with both prices on the
        same share basis, so a stock split inside the window NEVER creates
        an artificial return. Dividends are ignored.
    SIMPLE_TOTAL_RETURN - the same, plus cash dividends whose ex-date
        session is strictly after the entry session and on or before the
        outcome session, reinvested at the ex-date close:
        r = prod((P(s) + D(s)) / P(s-1)) - 1 over the window sessions,
        where D(s) is the ex-date dividend amount converted to the same
        split-continuous share basis as P. Requires the corporate-actions
        events artifact; the engine refuses total return without it.
    """

    SIMPLE_PRICE_RETURN = "simple_price_return"
    SIMPLE_TOTAL_RETURN = "simple_total_return"


class PriceField(str, Enum):
    """Which bar field defines the price. Only the close is implemented
    (ORBIT's EOD scope); high/low based outcomes are future work."""

    CLOSE = "close"


class DrawdownType(str, Enum):
    """The two drawdown definitions are NOT interchangeable; the contract
    must name exactly one.

    MAX_DRAWDOWN           - peak-to-trough decline over the outcome window,
        where the peak is the running maximum of closes starting from the
        ENTRY close (the entry close counts as the initial peak) and the
        trough is any later close in the window (including the outcome
        close). Reported as a non-negative fraction (0.10 = a 10% decline;
        0 when the window never falls below its running peak).
    MAX_ADVERSE_EXCURSION  - the maximum decline from the ENTRY close only:
        1 - min(window closes) / entry close, floored at 0.
    """

    MAX_DRAWDOWN = "max_drawdown"
    MAX_ADVERSE_EXCURSION = "max_adverse_excursion"


class VolatilityEstimator(str, Enum):
    """The only implemented estimator: sample standard deviation (ddof=1) of
    close-to-close simple returns over the outcome window's H sessions,
    annualized by the contract's annualization factor."""

    SAMPLE_STD_CLOSE_TO_CLOSE = "sample_std_close_to_close_daily_returns"


class MissingDataPolicy(str, Enum):
    """The single implemented policy: an outcome that cannot be computed
    exactly is marked unavailable with an explicit reason. Missing future
    prices are NEVER filled with zero, NEVER shortened silently, and NEVER
    replaced with substitutes."""

    EXPLICIT_UNAVAILABLE = "explicit_unavailable"


class DelistingPolicy(str, Enum):
    """The single implemented policy: a security that disappears from the
    dataset is never assigned a fabricated zero (or any other) return. If
    the instrument master records a delisting and the outcome window cannot
    complete, the label is unavailable with reason DELISTED (the delisting
    date is recorded in the detail); otherwise INSUFFICIENT_FUTURE_DATA."""

    UNAVAILABLE_WITH_REASON = "unavailable_with_reason"


class CorporateActionPolicy(str, Enum):
    """The single implemented policy: labels use Phase 3's canonical
    split-continuous price basis (the stored bars are marked
    'split_adjusted'; consecutive-close ratios are continuous across
    splits). A split inside the outcome window therefore never creates an
    artificial return. As-published closes (Phase 4 reconstruction) are
    recorded on every label row for audit, but the return itself is
    computed on the shared-basis series."""

    CANONICAL_PHASE3 = "phase3_canonical_split_continuous"


class OverlapPolicy(str, Enum):
    """The single implemented policy: every label row carries its outcome
    window (window_start_session .. window_end_session) so overlapping
    outcomes are always identifiable; the engine exposes overlapping_pairs()
    for later phases (purging, embargo, inference). No statistical
    machinery lives in Phase 5."""

    WINDOWS_TRACKED = "outcome_windows_tracked"


# Horizon values of the Phase 1 enum interpreted as TRADING SESSIONS. The
# "D" in the enum value denotes ORBIT's daily/EOD scope; it does NOT mean
# calendar days. 21 ≈ a month, 63 ≈ a quarter of US trading sessions.
HORIZON_SESSIONS: dict[Horizon, int] = {
    Horizon.H1: 1,
    Horizon.H5: 5,
    Horizon.H21: 21,
    Horizon.H63: 63,
}

_SUPPORTED_TARGET_TYPES = frozenset(
    {
        LabelType.FORWARD_RETURN,
        LabelType.EXCESS_RETURN,
        LabelType.VOLATILITY,
        LabelType.DRAWDOWN,
    }
)


class LabelContract(BaseModel):
    """One prediction target, fully and immutably defined."""

    model_config = ConfigDict(frozen=True)

    label_id: str = Field(pattern=r"^LAB-\d{3}$")
    version: str = Field(pattern=r"^v\d+(\.\d+)*$")

    target_type: LabelType
    horizon: int = Field(ge=1, description="number of trading sessions")
    horizon_semantics: str = "trading_sessions"

    anchor_mode: AnchorMode
    price_field: PriceField = PriceField.CLOSE

    return_convention: ReturnConvention | None = Field(
        default=None,
        description="required for FORWARD_RETURN / EXCESS_RETURN",
    )
    benchmark: str | None = Field(
        default=None,
        description="instrument_id of the benchmark; required for EXCESS_RETURN",
    )

    volatility_estimator: VolatilityEstimator | None = Field(
        default=None, description="required for VOLATILITY"
    )
    annualization: float | None = Field(
        default=None,
        ge=0,
        description="sessions per year for annualizing volatility (252.0)",
    )
    min_observations: int | None = Field(
        default=None,
        ge=1,
        description="minimum valid returns in the window for VOLATILITY",
    )

    drawdown_type: DrawdownType | None = Field(
        default=None, description="required for DRAWDOWN"
    )

    overlap_policy: OverlapPolicy = OverlapPolicy.WINDOWS_TRACKED
    missing_data_policy: MissingDataPolicy = MissingDataPolicy.EXPLICIT_UNAVAILABLE
    delisting_policy: DelistingPolicy = DelistingPolicy.UNAVAILABLE_WITH_REASON
    corporate_action_policy: CorporateActionPolicy = (
        CorporateActionPolicy.CANONICAL_PHASE3
    )

    formula: str = Field(
        min_length=1,
        description="exact human-readable mathematical definition (the identity "
        "a future researcher needs without reading code)",
    )
    description: str | None = None
    author: str = "orbit-research"

    @model_validator(mode="after")
    def _target_supported(self) -> "LabelContract":
        if self.target_type not in _SUPPORTED_TARGET_TYPES:
            raise ValueError(
                f"target_type {self.target_type.value!r} is not supported by "
                "Phase 5 (supported: forward_return, excess_return, "
                "volatility, drawdown). RISK_ADJUSTED_RETURN combines a "
                "future outcome with a point-in-time trailing-volatility "
                "denominator and is assembled in a later phase."
            )
        return self

    @model_validator(mode="after")
    def _return_convention_required_for_returns(self) -> "LabelContract":
        if self.target_type in (LabelType.FORWARD_RETURN, LabelType.EXCESS_RETURN):
            if self.return_convention is None:
                raise ValueError(
                    "return_convention is required for FORWARD_RETURN and "
                    "EXCESS_RETURN targets"
                )
        elif self.target_type in (LabelType.VOLATILITY, LabelType.DRAWDOWN):
            if self.return_convention == ReturnConvention.SIMPLE_TOTAL_RETURN:
                raise ValueError(
                    "VOLATILITY and DRAWDOWN targets are defined on "
                    "close-to-close PRICE returns only; the current "
                    "estimators/drawdown definitions have no dividend term, "
                    "so a SIMPLE_TOTAL_RETURN convention would misdescribe "
                    "the computed label"
                )
        return self

    @model_validator(mode="after")
    def _excess_requires_benchmark(self) -> "LabelContract":
        if self.target_type == LabelType.EXCESS_RETURN and not self.benchmark:
            raise ValueError("EXCESS_RETURN requires a benchmark instrument_id")
        return self

    @model_validator(mode="after")
    def _volatility_definition(self) -> "LabelContract":
        if self.target_type == LabelType.VOLATILITY:
            if self.volatility_estimator is None:
                raise ValueError(
                    "volatility_estimator is required for VOLATILITY targets"
                )
            if self.annualization is None:
                raise ValueError("annualization is required for VOLATILITY targets")
            if self.annualization <= 0:
                raise ValueError("annualization must be positive")
            if self.min_observations is None:
                raise ValueError(
                    "min_observations is required for VOLATILITY targets"
                )
            if self.min_observations < 2:
                raise ValueError(
                    "min_observations must be >= 2 (a sample std with ddof=1 "
                    "is undefined for fewer than 2 returns)"
                )
            if self.min_observations > self.horizon:
                raise ValueError(
                    "min_observations cannot exceed the horizon: the window "
                    "never yields more returns than H sessions"
                )
        return self

    @model_validator(mode="after")
    def _drawdown_definition(self) -> "LabelContract":
        if self.target_type == LabelType.DRAWDOWN and self.drawdown_type is None:
            raise ValueError("drawdown_type is required for DRAWDOWN targets")
        return self

    # ------------------------------------------------------- identity

    def canonical_json(self) -> str:
        """Deterministic JSON of the full definition (the identity)."""
        return json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        )

    def content_hash(self) -> str:
        """sha256 of the canonical definition: the formula/config identity
        recorded in every label row and experiment metadata. Includes the
        version, so (label_id, version) -> digest is a bijection."""
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    def definition_identity(self) -> str:
        """VERSION-INDEPENDENT definition identity (canonical JSON without
        the version field). Two contracts of the same label_id with the
        same definition always agree; a version bump that changes nothing
        is detectable, so version inflation is refused at registration."""
        data = json.loads(self.canonical_json())
        data.pop("version", None)
        identity = json.dumps(data, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(identity.encode("utf-8")).hexdigest()

    def definition_summary(self) -> dict[str, Any]:
        """The fields a researcher needs to answer 'exactly how was this
        label calculated?' without reading code."""
        return {
            "label_id": self.label_id,
            "version": self.version,
            "content_hash": self.content_hash(),
            "target_type": self.target_type.value,
            "horizon": self.horizon,
            "horizon_semantics": self.horizon_semantics,
            "anchor_mode": self.anchor_mode.value,
            "price_field": self.price_field.value,
            "return_convention": (
                self.return_convention.value if self.return_convention else None
            ),
            "benchmark": self.benchmark,
            "volatility_estimator": (
                self.volatility_estimator.value
                if self.volatility_estimator
                else None
            ),
            "annualization": self.annualization,
            "min_observations": self.min_observations,
            "drawdown_type": (
                self.drawdown_type.value if self.drawdown_type else None
            ),
            "overlap_policy": self.overlap_policy.value,
            "missing_data_policy": self.missing_data_policy.value,
            "delisting_policy": self.delisting_policy.value,
            "corporate_action_policy": self.corporate_action_policy.value,
            "formula": self.formula,
        }


def contract_from_hypothesis_label(
    label: Any,
    label_id: str,
    *,
    anchor_mode: AnchorMode = AnchorMode.DECISION_INSTANT,
    return_convention: ReturnConvention = ReturnConvention.SIMPLE_PRICE_RETURN,
    benchmark: str | None = None,
) -> LabelContract:
    """Bridge a Phase 1 `LabelSpec` (hypothesis-side summary) into the
    operational Phase 5 `LabelContract`.

    The Phase 1 `horizon` enum value is interpreted as TRADING SESSIONS
    (H1->1, H5->5, H21->21, H63->63); the label's definition text becomes the
    contract's formula. `return_convention` and `benchmark` are explicit
    parameters - never inferred silently from prose.
    """
    from orbit.schemas.hypothesis import LabelSpec

    if not isinstance(label, LabelSpec):
        raise TypeError(
            f"expected orbit.schemas.hypothesis.LabelSpec, got {type(label).__name__}"
        )
    horizon = HORIZON_SESSIONS.get(label.horizon)
    if horizon is None:
        raise ValueError(f"unsupported Phase 1 horizon: {label.horizon}")
    return LabelContract(
        label_id=label_id,
        version=label.label_version,
        target_type=label.label_type,
        horizon=horizon,
        anchor_mode=anchor_mode,
        return_convention=(
            return_convention
            if label.label_type in (LabelType.FORWARD_RETURN, LabelType.EXCESS_RETURN)
            else None
        ),
        benchmark=benchmark or label.benchmark,
        formula=label.definition,
        overlap_policy=OverlapPolicy.WINDOWS_TRACKED,
        missing_data_policy=MissingDataPolicy.EXPLICIT_UNAVAILABLE,
        delisting_policy=DelistingPolicy.UNAVAILABLE_WITH_REASON,
        corporate_action_policy=CorporateActionPolicy.CANONICAL_PHASE3,
        description="derived from Phase 1 hypothesis label spec",
    )


__all__ = [
    "AnchorMode",
    "CorporateActionPolicy",
    "DelistingPolicy",
    "DrawdownType",
    "HORIZON_SESSIONS",
    "LabelContract",
    "MissingDataPolicy",
    "OverlapPolicy",
    "PriceField",
    "ReturnConvention",
    "SESSION_ANNUALIZATION",
    "VolatilityEstimator",
    "contract_from_hypothesis_label",
]