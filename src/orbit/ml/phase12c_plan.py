"""Phase 12C — Real PIT Fundamental Data Acquisition Plan (LOCKED)."""
from __future__ import annotations

import hashlib
import json
from datetime import date, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]

PHASE12C_VERSION = "1.0.0"
PHASE12C_CREATED = "2026-08-22"

# ---------------------------------------------------------------------------
# STALENESS POLICY (locked before execution)
# ---------------------------------------------------------------------------
PHASE12C_STALENESS = {
    "max_age_years": 2,
    "invalidation_behavior": "null_out_session",
    "record_age": "compute_age_from_availability_timestamp",
    "policy_enforced_before": "model_training",
}

# ---------------------------------------------------------------------------
# AFTER-CLOSE FILING POLICY (locked before execution)
# ---------------------------------------------------------------------------
PHASE12C_AFTER_CLOSE_POLICY = {
    "description": (
        "Filing submitted after the market close (16:00 ET) on date T "
        "becomes available for features starting on the NEXT trading session T+1. "
        "Filing submitted before market open (09:30 ET) on date T "
        "becomes available for features on date T. "
        "Filing submitted during market hours (09:30-16:00 ET) on date T "
        "becomes available for features on the NEXT trading session T+1 "
        "(conservative treatment to avoid intraday lookahead)."
    ),
    "market_open": "09:30 ET",
    "market_close": "16:00 ET",
    "before_open": "same_session",
    "during_market": "next_session",
    "after_close": "next_session",
    "weekend_filing": "next_trading_session",
    "holiday_filing": "next_trading_session",
}

# ---------------------------------------------------------------------------
# AVAILABILITY TIMESTAMP HIERARCHY
# ---------------------------------------------------------------------------
PHASE12C_AVAILABILITY_HIERARCHY = [
    "1. SEC acceptance_timestamp (preferred)",
    "2. filing_date with conservative after-close treatment (fallback)",
    "3. filing_date + 1 business day (last resort, documented)",
]

# ---------------------------------------------------------------------------
# SEC EDGAR DATA SOURCE
# ---------------------------------------------------------------------------
PHASE12C_DATA_SOURCE = {
    "source_id": "SEC-EDGAR-CompanyFacts",
    "source_url": "https://efts.sec.gov/LATEST/search-index?q=%22companyfacts%22",
    "api_base": "https://data.sec.gov/api/xbrl/companyfacts",
    "user_agent": "ORBIT-Research orbit@example.com",
    "rate_limit_seconds": 0.1,
    "data_type": "REAL",
    "license": "Public domain (SEC EDGAR is public)",
    "notes": [
        "SEC EDGAR Company Facts provides structured XBRL data",
        "Filing dates and acceptance timestamps are from SEC records",
        "All data is real SEC filings",
    ],
}

# ---------------------------------------------------------------------------
# XBRL TAG MAPPING
# ---------------------------------------------------------------------------
PHASE12C_XBRL_TAGS = {
    "valuation": {
        "earnings": ["EarningsPerShareDiluted", "EarningsPerShareBasic", "EPS"],
        "book_value": ["StockholdersEquity", "CommonStockholdersEquity", "TotalStockholdersEquity"],
        "revenue": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet"],
    },
    "profitability": {
        "net_income": ["NetIncomeLoss", "ProfitLoss"],
        "operating_income": ["OperatingIncomeLoss", "IncomeLossFromContinuingOperations"],
        "total_assets": ["Assets"],
        "shareholders_equity": ["StockholdersEquity", "CommonStockholdersEquity"],
        "gross_profit": ["GrossProfit"],
    },
    "growth": {
        "revenue": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax"],
        "earnings": ["NetIncomeLoss", "EarningsPerShareDiluted"],
        "operating_cash_flow": ["NetCashProvidedByUsedInOperatingActivities", "OperatingCashFlow"],
    },
    "leverage": {
        "total_debt": ["LongTermDebt", "LongTermDebtNoncurrent", "LongTermDebtAndCapitalLeaseObligations"],
        "total_assets": ["Assets"],
        "current_assets": ["AssetsCurrent"],
        "current_liabilities": ["LiabilitiesCurrent"],
    },
}

# ---------------------------------------------------------------------------
# FORM TYPES TO INCLUDE
# ---------------------------------------------------------------------------
PHASE12C_FORM_TYPES = ["10-K", "10-K/A", "10-Q", "10-Q/A", "20-F", "20-F/A"]

# ---------------------------------------------------------------------------
# ENVIRONMENT DEFINITIONS
# ---------------------------------------------------------------------------
PHASE12C_ENVIRONMENTS = {
    "ENV-12C-050": {
        "description": "50-instrument universe with real SEC fundamentals",
        "dataset_id": "DS-EXP-050",
        "n_instruments_target": 50,
    },
    "ENV-12C-100": {
        "description": "100-instrument universe with real SEC fundamentals",
        "dataset_id": "DS-EXP-100",
        "n_instruments_target": 100,
    },
}


def _compute_plan_digest(plan_dict: dict) -> str:
    """Compute deterministic SHA-256 digest of the plan."""
    plan_str = json.dumps(plan_dict, sort_keys=True, default=str)
    return hashlib.sha256(plan_str.encode()).hexdigest()


def build_phase12c_plan() -> dict[str, Any]:
    """Build and return the locked Phase 12C plan."""
    plan = {
        "phase": "12C",
        "version": PHASE12C_VERSION,
        "created_at": PHASE12C_CREATED,
        "research_question": (
            "Can ORBIT construct a real, point-in-time-safe fundamental "
            "dataset with valid instrument identity, filing availability, "
            "historical lineage, and sufficient coverage to permit valid "
            "fundamental ML experiments?"
        ),
        "data_source": PHASE12C_DATA_SOURCE,
        "staleness": PHASE12C_STALENESS,
        "after_close_policy": PHASE12C_AFTER_CLOSE_POLICY,
        "availability_hierarchy": PHASE12C_AVAILABILITY_HIERARCHY,
        "xbrl_tags": PHASE12C_XBRL_TAGS,
        "form_types": PHASE12C_FORM_TYPES,
        "environments": PHASE12C_ENVIRONMENTS,
    }
    plan["plan_digest"] = _compute_plan_digest(plan)
    return plan


def persist_phase12c_plan(plan: dict[str, Any]) -> Path:
    """Persist the locked plan to benchmarks/."""
    out = REPO_ROOT / "benchmarks" / "phase12c_plan.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(plan, indent=2, default=str), encoding="utf-8")
    return out
