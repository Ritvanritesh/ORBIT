"""Phase 12C — Point-in-Time fundamental feature engine."""
from __future__ import annotations

import hashlib
import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]

# Tag mapping: Phase 12B feature -> list of XBRL tags to try
TAG_MAP = {
    # Valuation (need market price - tracked separately)
    "earnings_yield": ["EarningsPerShareDiluted", "EarningsPerShareBasic"],
    "book_to_market": ["StockholdersEquity", "CommonStockholdersEquity"],
    "sales_to_price": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax"],
    # Profitability
    "roa": [("NetIncomeLoss", "Assets")],
    "roe": [("NetIncomeLoss", "StockholdersEquity")],
    "operating_margin": [("OperatingIncomeLoss", "Revenues")],
    "gross_profitability": [("GrossProfit", "Revenues")],
    # Growth (need prior period)
    "rev_growth_1y": [("Revenues", "Revenues_prior")],
    "earn_growth_1y": [("NetIncomeLoss", "NetIncomeLoss_prior")],
    "cash_growth_1y": [("NetCashProvidedByUsedInOperatingActivities", "NetCashProvidedByUsedInOperatingActivities_prior")],
    # Leverage
    "de_to_equity": [("LongTermDebt", "StockholdersEquity")],
    "de_to_assets": [("LongTermDebt", "Assets")],
    "current_ratio": [("AssetsCurrent", "LiabilitiesCurrent")],
}

# Simple (non-ratio) features
SIMPLE_TAGS = {
    "earnings_yield": ["EarningsPerShareDiluted", "EarningsPerShareBasic"],
    "book_to_market": ["StockholdersEquity", "CommonStockholdersEquity"],
    "sales_to_price": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax"],
    "roa": ["NetIncomeLoss"],
    "roe": ["NetIncomeLoss"],
    "operating_margin": ["OperatingIncomeLoss"],
    "gross_profitability": ["GrossProfit"],
    "rev_growth_1y": ["Revenues"],
    "earn_growth_1y": ["NetIncomeLoss"],
    "cash_growth_1y": ["NetCashProvidedByUsedInOperatingActivities"],
    "de_to_equity": ["LongTermDebt", "LongTermDebtNoncurrent"],
    "de_to_assets": ["LongTermDebt", "LongTermDebtNoncurrent"],
    "current_ratio": ["AssetsCurrent"],
}


def _find_tag(observations: list[dict], tag_list: list[str], period_end: str | None = None) -> dict | None:
    """Find the latest observation matching one of the given tags.

    Uses AS-of join: availability_date <= feature_boundary.
    """
    candidates = []
    for obs in observations:
        if obs["tag"] not in tag_list:
            continue
        if obs["value"] is None:
            continue
        avail = obs.get("availability_date")
        if avail is None:
            continue
        candidates.append(obs)

    if not candidates:
        return None

    # Sort by availability_date descending, take the latest
    candidates.sort(key=lambda x: x.get("availability_date", ""), reverse=True)
    return candidates[0]


def compute_pit_features(
    observations: list[dict[str, Any]],
    as_of_date: date,
    staleness_years: int = 2,
) -> dict[str, Any]:
    """Compute PIT fundamental features for a given as_of_date.

    Uses AS-of join policy:
    - availability_date <= as_of_date
    - age <= staleness_years
    """
    cutoff = as_of_date.isoformat()
    staleness_cutoff = (as_of_date - timedelta(days=365 * staleness_years)).isoformat()

    # Filter to available observations
    available = [
        obs for obs in observations
        if obs.get("availability_date", "") <= cutoff
        and obs.get("availability_date", "") >= staleness_cutoff
        and obs["value"] is not None
    ]

    features = {}

    # Simple features (no ratio computation needed)
    for feat_name, tag_list in SIMPLE_TAGS.items():
        obs = _find_tag(available, tag_list)
        if obs and obs["value"] is not None:
            features[feat_name] = obs["value"]
        else:
            features[feat_name] = None

    # Ratio features
    # ROA = NetIncome / Assets
    ni_obs = _find_tag(available, ["NetIncomeLoss"])
    assets_obs = _find_tag(available, ["Assets"])
    if ni_obs and assets_obs and assets_obs["value"] != 0:
        features["roa"] = ni_obs["value"] / assets_obs["value"]
    elif "roa" not in features:
        features["roa"] = features.get("roa")

    # ROE = NetIncome / Equity
    equity_obs = _find_tag(available, ["StockholdersEquity", "CommonStockholdersEquity"])
    if ni_obs and equity_obs and equity_obs["value"] != 0:
        features["roe"] = ni_obs["value"] / equity_obs["value"]
    elif "roe" not in features:
        features["roe"] = features.get("roe")

    # Operating margin = OperatingIncome / Revenue
    oi_obs = _find_tag(available, ["OperatingIncomeLoss"])
    rev_obs = _find_tag(available, ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax"])
    if oi_obs and rev_obs and rev_obs["value"] != 0:
        features["operating_margin"] = oi_obs["value"] / rev_obs["value"]
    elif "operating_margin" not in features:
        features["operating_margin"] = features.get("operating_margin")

    # Gross profitability = GrossProfit / Revenue
    gp_obs = _find_tag(available, ["GrossProfit"])
    if gp_obs and rev_obs and rev_obs["value"] != 0:
        features["gross_profitability"] = gp_obs["value"] / rev_obs["value"]
    elif "gross_profitability" not in features:
        features["gross_profitability"] = features.get("gross_profitability")

    # Leverage ratios
    debt_obs = _find_tag(available, ["LongTermDebt", "LongTermDebtNoncurrent"])
    if debt_obs and equity_obs and equity_obs["value"] != 0:
        features["de_to_equity"] = debt_obs["value"] / equity_obs["value"]
    elif "de_to_equity" not in features:
        features["de_to_equity"] = features.get("de_to_equity")

    if debt_obs and assets_obs and assets_obs["value"] != 0:
        features["de_to_assets"] = debt_obs["value"] / assets_obs["value"]
    elif "de_to_assets" not in features:
        features["de_to_assets"] = features.get("de_to_assets")

    ca_obs = _find_tag(available, ["AssetsCurrent"])
    cl_obs = _find_tag(available, ["LiabilitiesCurrent"])
    if ca_obs and cl_obs and cl_obs["value"] != 0:
        features["current_ratio"] = ca_obs["value"] / cl_obs["value"]
    elif "current_ratio" not in features:
        features["current_ratio"] = features.get("current_ratio")

    # Growth: need prior year observations
    prior_cutoff = (as_of_date - timedelta(days=365)).isoformat()
    prior_available = [
        obs for obs in observations
        if obs.get("availability_date", "") <= prior_cutoff
        and obs.get("availability_date", "") >= staleness_cutoff
        and obs["value"] is not None
    ]

    for growth_feat, tag_pairs in [
        ("rev_growth_1y", "Revenues"),
        ("earn_growth_1y", "NetIncomeLoss"),
        ("cash_growth_1y", "NetCashProvidedByUsedInOperatingActivities"),
    ]:
        curr = _find_tag(available, [tag_pairs])
        prev = _find_tag(prior_available, [tag_pairs])
        if curr and prev and prev["value"] != 0:
            features[growth_feat] = (curr["value"] - prev["value"]) / abs(prev["value"])
        else:
            features[growth_feat] = None

    features["n_available_observations"] = len(available)
    features["as_of_date"] = as_of_date.isoformat()

    return features
