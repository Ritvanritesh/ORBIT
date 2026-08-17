"""Seed hypothesis registry (Phase 1: three falsifiable hypotheses).

These are pre-registered BEFORE any feature exploration or data acquisition,
per the charter. Criteria are binding and cannot be moved after results appear.
"""

from orbit.schemas.hypothesis import HypothesisRegistry, HypothesisSpec, LabelSpec
from orbit.schemas.common import (
    EvidenceType,
    Horizon,
    LabelType,
    LeakageClass,
    UniverseScope,
)


def build_seed_registry() -> HypothesisRegistry:
    """Construct the three Phase-1 seed hypotheses (unregistered drafts)."""

    h001 = HypothesisSpec(
        hypothesis_id="H-001",
        title="Cross-sectional 12-1 price momentum in liquid US equities",
        statement=(
            "Ranks of trailing 12-month return excluding the most recent month "
            "in the ORBIT liquid-equity universe predict the cross-section of "
            "5-day forward excess returns after costs."
        ),
        mechanism=(
            "Gradual information diffusion and behavioral underreaction to "
            "fundamental news produce persistent continuation, documented "
            "across decades but sensitive to crowding and turnover."
        ),
        baseline=["equal_weight_top50", "SPY_hold"],
        universe=UniverseScope.LIQUID_EQUITY_50_100,
        label=LabelSpec(
            label_type=LabelType.EXCESS_RETURN,
            horizon=Horizon.H5,
            benchmark="SPY",
            definition=(
                "5-trading-day forward total return minus SPY total return "
                "over the same window, from point-in-time close to close."
            ),
        ),
        feature_families=["momentum"],
        leakage_class=LeakageClass.NONE,
        data_sources=["licensed_market_daily_ohlcv", "corporate_actions"],
        economic_evidence=dict(
            oos_rank_ic=0.03,
            after_cost_annual_excess=0.03,
            min_regimes=3,
            min_walkforward_windows=4,
        ),
        falsification_criteria=(
            "Falsified if: (a) OOS rank IC < 0.03, or (b) after-cost annual "
            "excess vs SPY < 3%, or (c) the effect survives fewer than 3 "
            "distinct regimes or fewer than 4 sequential walk-forward windows, "
            "or (d) momentum adds no OOS lift over the equal-weight baseline."
        ),
        non_goals=[
            "Intraday or order-book momentum",
            "Cryptocurrency or non-US equities",
            "Leveraged or short-only construction in the first evaluation",
        ],
        evidence_type=EvidenceType.ECONOMIC,
    )

    h002 = HypothesisSpec(
        hypothesis_id="H-002",
        title="Low realized volatility anomaly in liquid US equities",
        statement=(
            "Volatility-scaled exposure to low-realized-volatility liquid US "
            "equities produces higher risk-adjusted after-cost returns than an "
            "equal-weight exposure to the same universe."
        ),
        mechanism=(
            "A persistent low-volatility anomaly exists in US equities "
            "(lottery preference and benchmark constraints); volatility "
            "targeting further stabilizes the path."
        ),
        baseline=["equal_weight_top50", "SPY_hold"],
        universe=UniverseScope.LIQUID_EQUITY_50_100,
        label=LabelSpec(
            label_type=LabelType.RISK_ADJUSTED_RETURN,
            horizon=Horizon.H21,
            benchmark="SPY",
            definition=(
                "For each stock: r_i(21) / sigma_i(21) minus r_SPY(21) / "
                "sigma_SPY(21), where r_x(h) is the h-trading-day forward "
                "total return and sigma_x(h) is the annualized realized "
                "volatility of daily returns over the trailing h trading "
                "days ending at the decision close, computed point-in-time."
            ),
        ),
        feature_families=["volatility", "liquidity"],
        leakage_class=LeakageClass.NONE,
        data_sources=["licensed_market_daily_ohlcv", "corporate_actions"],
        economic_evidence=dict(
            oos_rank_ic=0.02,
            after_cost_annual_excess=0.02,
            min_regimes=3,
            min_walkforward_windows=4,
        ),
        falsification_criteria=(
            "Falsified if: (a) risk-adjusted after-cost excess vs the "
            "equal-weight baseline < 2% annualized, or (b) the effect "
            "disappears under volatility-targeted sizing, or (c) it survives "
            "fewer than 3 regimes / 4 walk-forward windows."
        ),
        non_goals=[
            "Options or implied-volatility products",
            "Claims about market-timing skill",
            "High-turnover rebalancing",
        ],
        evidence_type=EvidenceType.ECONOMIC,
    )

    h003 = HypothesisSpec(
        hypothesis_id="H-003",
        title="Post-earnings-announcement drift via point-in-time fundamentals",
        statement=(
            "Standardized earnings surprise computed strictly from "
            "point-in-time SEC EDGAR/XBRL fundamentals predicts the cross-"
            "section of 5-day forward excess returns in liquid US equities."
        ),
        mechanism=(
            "Post-earnings-announcement drift (PEAD) reflects delayed "
            "adjustment to earnings information; using publication-time "
            "fundamental values avoids the revision leakage that inflates "
            "naive PEAD estimates."
        ),
        baseline=["equal_weight_top50", "SPY_hold", "momentum_baseline"],
        universe=UniverseScope.LIQUID_EQUITY_50_100,
        label=LabelSpec(
            label_type=LabelType.EXCESS_RETURN,
            horizon=Horizon.H5,
            benchmark="SPY",
            definition=(
                "5-trading-day forward total return minus SPY total return, "
                "measured from the trading day after the earliest point-in-time "
                "publication timestamp of the filing."
            ),
        ),
        feature_families=["fundamentals"],
        leakage_class=LeakageClass.FUTURE_PUBLICATION,
        data_sources=["sec_edgar_xbrl", "licensed_market_daily_ohlcv"],
        economic_evidence=dict(
            oos_rank_ic=0.02,
            after_cost_annual_excess=0.02,
            min_regimes=2,
            min_walkforward_windows=4,
        ),
        falsification_criteria=(
            "Falsified if: (a) OOS rank IC < 0.02 or after-cost annual excess "
            "< 2%, (b) the signal's OOS lift over the momentum baseline is "
            "not material, or (c) any synthetic future-leak test in the "
            "temporal-truth engine catches the signal using post-publication "
            "or revised fundamentals."
        ),
        non_goals=[
            "Text mining of filings (deferred to Phase 21)",
            "Surprise measured from non-point-in-time vendor revisions",
            "Event-driven trading intraday",
        ],
        evidence_type=EvidenceType.ECONOMIC,
    )

    return HypothesisRegistry(hypotheses=[h001, h002, h003])


def register_seeds() -> HypothesisRegistry:
    """Register the seed hypotheses, freezing them (status REGISTERED)."""
    return build_seed_registry().register_all()