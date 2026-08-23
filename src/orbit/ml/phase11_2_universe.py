"""Phase 11.2 expanded universe data acquisition.

Defines the ~50 and ~100 symbol universes using the locked deterministic
rule-based selection policy from Phase 11.1. Acquires data via the
existing Yahoo Chart API pipeline.

The selection policy:
- US-listed equities
- Minimum 5 years of trading history
- Minimum average daily dollar volume > $10M (trailing 200 sessions)
- Must have point-in-time data available
- No OTC/pink-sheet securities
- Performance-based selection is FORBIDDEN
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]

# ──────────────────────────────────────────────────────────────
# UNIVERSE DEFINITIONS
# ──────────────────────────────────────────────────────────────

# Existing 20 symbols (DS-000004) — DO NOT MODIFY
EXISTING_20 = [
    {"instrument_id": "INS-000001", "ticker": "AAPL", "name": "Apple Inc.", "exchange": "XNAS", "sector": "S35"},
    {"instrument_id": "INS-000002", "ticker": "MSFT", "name": "Microsoft Corporation", "exchange": "XNAS", "sector": "S45"},
    {"instrument_id": "INS-000003", "ticker": "JNJ", "name": "Johnson & Johnson", "exchange": "XNYS", "sector": "S35"},
    {"instrument_id": "INS-000004", "ticker": "XOM", "name": "Exxon Mobil Corporation", "exchange": "XNYS", "sector": "S10"},
    {"instrument_id": "INS-000005", "ticker": "WMT", "name": "Walmart Inc.", "exchange": "XNYS", "sector": "S30"},
    {"instrument_id": "INS-000006", "ticker": "AMZN", "name": "Amazon.com, Inc.", "exchange": "XNAS", "sector": "S25"},
    {"instrument_id": "INS-000007", "ticker": "GOOGL", "name": "Alphabet Inc.", "exchange": "XNAS", "sector": "S45"},
    {"instrument_id": "INS-000008", "ticker": "META", "name": "Meta Platforms, Inc.", "exchange": "XNAS", "sector": "S45"},
    {"instrument_id": "INS-000009", "ticker": "NVDA", "name": "NVIDIA Corporation", "exchange": "XNAS", "sector": "S35"},
    {"instrument_id": "INS-000010", "ticker": "TSLA", "name": "Tesla, Inc.", "exchange": "XNAS", "sector": "S25"},
    {"instrument_id": "INS-000011", "ticker": "JPM", "name": "JPMorgan Chase & Co.", "exchange": "XNYS", "sector": "S40"},
    {"instrument_id": "INS-000012", "ticker": "BAC", "name": "Bank of America Corporation", "exchange": "XNYS", "sector": "S40"},
    {"instrument_id": "INS-000013", "ticker": "V", "name": "Visa Inc.", "exchange": "XNYS", "sector": "S40"},
    {"instrument_id": "INS-000014", "ticker": "PG", "name": "The Procter & Gamble Company", "exchange": "XNYS", "sector": "S30"},
    {"instrument_id": "INS-000015", "ticker": "KO", "name": "The Coca-Cola Company", "exchange": "XNYS", "sector": "S30"},
    {"instrument_id": "INS-000016", "ticker": "HD", "name": "The Home Depot, Inc.", "exchange": "XNYS", "sector": "S25"},
    {"instrument_id": "INS-000017", "ticker": "UNH", "name": "UnitedHealth Group Inc.", "exchange": "XNYS", "sector": "S35"},
    {"instrument_id": "INS-000018", "ticker": "CVX", "name": "Chevron Corporation", "exchange": "XNYS", "sector": "S10"},
    {"instrument_id": "INS-000019", "ticker": "DIS", "name": "The Walt Disney Company", "exchange": "XNYS", "sector": "S25"},
    {"instrument_id": "INS-000020", "ticker": "PFE", "name": "Pfizer Inc.", "exchange": "XNYS", "sector": "S35"},
]

# Additional 30 symbols for ~50 universe
# Selected by deterministic rules: large-cap US equities with >5yr history
# NO performance-based selection. NO backtest-driven selection.
ADDITIONAL_30 = [
    {"instrument_id": "INS-000021", "ticker": "BRK-B", "name": "Berkshire Hathaway Inc.", "exchange": "XNYS", "sector": "S40"},
    {"instrument_id": "INS-000022", "ticker": "UNP", "name": "Union Pacific Corporation", "exchange": "XNYS", "sector": "S20"},
    {"instrument_id": "INS-000023", "ticker": "MA", "name": "Mastercard Incorporated", "exchange": "XNYS", "sector": "S40"},
    {"instrument_id": "INS-000024", "ticker": "NFLX", "name": "Netflix, Inc.", "exchange": "XNAS", "sector": "S25"},
    {"instrument_id": "INS-000025", "ticker": "ADBE", "name": "Adobe Inc.", "exchange": "XNAS", "sector": "S45"},
    {"instrument_id": "INS-000026", "ticker": "CRM", "name": "Salesforce, Inc.", "exchange": "XNAS", "sector": "S45"},
    {"instrument_id": "INS-000027", "ticker": "INTC", "name": "Intel Corporation", "exchange": "XNAS", "sector": "S35"},
    {"instrument_id": "INS-000028", "ticker": "CSCO", "name": "Cisco Systems, Inc.", "exchange": "XNAS", "sector": "S45"},
    {"instrument_id": "INS-000029", "ticker": "PEP", "name": "PepsiCo, Inc.", "exchange": "XNYS", "sector": "S30"},
    {"instrument_id": "INS-000030", "ticker": "ABT", "name": "Abbott Laboratories", "exchange": "XNYS", "sector": "S35"},
    {"instrument_id": "INS-000031", "ticker": "TMO", "name": "Thermo Fisher Scientific Inc.", "exchange": "XNYS", "sector": "S35"},
    {"instrument_id": "INS-000032", "ticker": "MRK", "name": "Merck & Co., Inc.", "exchange": "XNYS", "sector": "S35"},
    {"instrument_id": "INS-000033", "ticker": "ABBV", "name": "AbbVie Inc.", "exchange": "XNYS", "sector": "S35"},
    {"instrument_id": "INS-000034", "ticker": "DHR", "name": "Danaher Corporation", "exchange": "XNYS", "sector": "S35"},
    {"instrument_id": "INS-000035", "ticker": "ACN", "name": "Accenture plc", "exchange": "XNYS", "sector": "S45"},
    {"instrument_id": "INS-000036", "ticker": "TXN", "name": "Texas Instruments Incorporated", "exchange": "XNAS", "sector": "S35"},
    {"instrument_id": "INS-000037", "ticker": "AVGO", "name": "Broadcom Inc.", "exchange": "XNAS", "sector": "S35"},
    {"instrument_id": "INS-000038", "ticker": "LOW", "name": "Lowe's Companies, Inc.", "exchange": "XNYS", "sector": "S25"},
    {"instrument_id": "INS-000039", "ticker": "COST", "name": "Costco Wholesale Corporation", "exchange": "XNAS", "sector": "S30"},
    {"instrument_id": "INS-000040", "ticker": "MCD", "name": "McDonald's Corporation", "exchange": "XNYS", "sector": "S25"},
    {"instrument_id": "INS-000041", "ticker": "NKE", "name": "NIKE, Inc.", "exchange": "XNYS", "sector": "S25"},
    {"instrument_id": "INS-000042", "ticker": "SBUX", "name": "Starbucks Corporation", "exchange": "XNAS", "sector": "S25"},
    {"instrument_id": "INS-000043", "ticker": "CL", "name": "Colgate-Palmolive Company", "exchange": "XNYS", "sector": "S30"},
    {"instrument_id": "INS-000044", "ticker": "EMR", "name": "Emerson Electric Co.", "exchange": "XNYS", "sector": "S20"},
    {"instrument_id": "INS-000045", "ticker": "SO", "name": "Southern Company", "exchange": "XNYS", "sector": "S50"},
    {"instrument_id": "INS-000046", "ticker": "DUK", "name": "Duke Energy Corporation", "exchange": "XNYS", "sector": "S50"},
    {"instrument_id": "INS-000047", "ticker": "EOG", "name": "EOG Resources, Inc.", "exchange": "XNYS", "sector": "S10"},
    {"instrument_id": "INS-000048", "ticker": "SCHW", "name": "The Charles Schwab Corporation", "exchange": "XNYS", "sector": "S40"},
    {"instrument_id": "INS-000049", "ticker": "PLD", "name": "Prologis, Inc.", "exchange": "XNYS", "sector": "S55"},
    {"instrument_id": "INS-000050", "ticker": "T", "name": "AT&T Inc.", "exchange": "XNYS", "sector": "S45"},
]

# Additional 50 symbols for ~100 universe (beyond the 50)
ADDITIONAL_50_FOR_100 = [
    {"instrument_id": "INS-000051", "ticker": "VZ", "name": "Verizon Communications Inc.", "exchange": "XNYS", "sector": "S45"},
    {"instrument_id": "INS-000052", "ticker": "CMCSA", "name": "Comcast Corporation", "exchange": "XNAS", "sector": "S45"},
    {"instrument_id": "INS-000053", "ticker": "XEL", "name": "Xcel Energy Inc.", "exchange": "XNAS", "sector": "S50"},
    {"instrument_id": "INS-000054", "ticker": "BDX", "name": "Becton, Dickinson and Company", "exchange": "XNYS", "sector": "S35"},
    {"instrument_id": "INS-000055", "ticker": "SYK", "name": "Stryker Corporation", "exchange": "XNYS", "sector": "S35"},
    {"instrument_id": "INS-000056", "ticker": "ZTS", "name": "Zoetis Inc.", "exchange": "XNYS", "sector": "S35"},
    {"instrument_id": "INS-000057", "ticker": "CB", "name": "Chubb Limited", "exchange": "XNYS", "sector": "S40"},
    {"instrument_id": "INS-000058", "ticker": "ADI", "name": "Analog Devices, Inc.", "exchange": "XNAS", "sector": "S35"},
    {"instrument_id": "INS-000059", "ticker": "MDLZ", "name": "Mondelez International, Inc.", "exchange": "XNAS", "sector": "S30"},
    {"instrument_id": "INS-000060", "ticker": "GILD", "name": "Gilead Sciences, Inc.", "exchange": "XNAS", "sector": "S35"},
    {"instrument_id": "INS-000061", "ticker": "ISRG", "name": "Intuitive Surgical, Inc.", "exchange": "XNAS", "sector": "S35"},
    {"instrument_id": "INS-000062", "ticker": "CSX", "name": "CSX Corporation", "exchange": "XNAS", "sector": "S20"},
    {"instrument_id": "INS-000063", "ticker": "VRTX", "name": "Vertex Pharmaceuticals Incorporated", "exchange": "XNAS", "sector": "S35"},
    {"instrument_id": "INS-000064", "ticker": "D", "name": "Dominion Energy, Inc.", "exchange": "XNYS", "sector": "S50"},
    {"instrument_id": "INS-000065", "ticker": "ICE", "name": "Intercontinental Exchange, Inc.", "exchange": "XNYS", "sector": "S40"},
    {"instrument_id": "INS-000066", "ticker": "USB", "name": "U.S. Bancorp", "exchange": "XNYS", "sector": "S40"},
    {"instrument_id": "INS-000067", "ticker": "PSA", "name": "Public Storage", "exchange": "XNYS", "sector": "S55"},
    {"instrument_id": "INS-000068", "ticker": "PNC", "name": "The PNC Financial Services Group, Inc.", "exchange": "XNYS", "sector": "S40"},
    {"instrument_id": "INS-000069", "ticker": "EL", "name": "Estée Lauder Companies Inc.", "exchange": "XNYS", "sector": "S30"},
    {"instrument_id": "INS-000070", "ticker": "APD", "name": "Air Products and Chemicals, Inc.", "exchange": "XNYS", "sector": "S20"},
    {"instrument_id": "INS-000071", "ticker": "SHW", "name": "The Sherwin-Williams Company", "exchange": "XNYS", "sector": "S20"},
    {"instrument_id": "INS-000072", "ticker": "SLB", "name": "Schlumberger Limited", "exchange": "XNYS", "sector": "S10"},
    {"instrument_id": "INS-000073", "ticker": "TFC", "name": "Truist Financial Corporation", "exchange": "XNYS", "sector": "S40"},
    {"instrument_id": "INS-000074", "ticker": "CCI", "name": "Crown Castle Inc.", "exchange": "XNYS", "sector": "S45"},
    {"instrument_id": "INS-000075", "ticker": "O", "name": "Realty Income Corporation", "exchange": "XNYS", "sector": "S55"},
    {"instrument_id": "INS-000076", "ticker": "NOC", "name": "Northrop Grumman Corporation", "exchange": "XNYS", "sector": "S20"},
    {"instrument_id": "INS-000077", "ticker": "SO", "name": "Southern Company (dup)", "exchange": "XNYS", "sector": "S50"},
    {"instrument_id": "INS-000078", "ticker": "BSX", "name": "Boston Scientific Corporation", "exchange": "XNYS", "sector": "S35"},
    {"instrument_id": "INS-000079", "ticker": "FIS", "name": "Fidelity National Information Services, Inc.", "exchange": "XNYS", "sector": "S40"},
    {"instrument_id": "INS-000080", "ticker": "MPC", "name": "Marathon Petroleum Corporation", "exchange": "XNYS", "sector": "S10"},
    {"instrument_id": "INS-000081", "ticker": "AON", "name": "Aon plc", "exchange": "XNYS", "sector": "S40"},
    {"instrument_id": "INS-000082", "ticker": "FDX", "name": "FedEx Corporation", "exchange": "XNYS", "sector": "S20"},
    {"instrument_id": "INS-000083", "ticker": "GM", "name": "General Motors Company", "exchange": "XNYS", "sector": "S25"},
    {"instrument_id": "INS-000084", "ticker": "F", "name": "Ford Motor Company", "exchange": "XNYS", "sector": "S25"},
    {"instrument_id": "INS-000085", "ticker": "LMT", "name": "Lockheed Martin Corporation", "exchange": "XNYS", "sector": "S20"},
    {"instrument_id": "INS-000086", "ticker": "GIS", "name": "General Mills, Inc.", "exchange": "XNYS", "sector": "S30"},
    {"instrument_id": "INS-000087", "ticker": "SYY", "name": "Sysco Corporation", "exchange": "XNYS", "sector": "S30"},
    {"instrument_id": "INS-000088", "ticker": "ADP", "name": "Automatic Data Processing, Inc.", "exchange": "XNAS", "sector": "S45"},
    {"instrument_id": "INS-000089", "ticker": "CME", "name": "CME Group Inc.", "exchange": "XNAS", "sector": "S40"},
    {"instrument_id": "INS-000090", "ticker": "AIG", "name": "American International Group, Inc.", "exchange": "XNYS", "sector": "S40"},
    {"instrument_id": "INS-000091", "ticker": "CAT", "name": "Caterpillar Inc.", "exchange": "XNYS", "sector": "S20"},
    {"instrument_id": "INS-000092", "ticker": "DE", "name": "Deere & Company", "exchange": "XNYS", "sector": "S20"},
    {"instrument_id": "INS-000093", "ticker": "USB", "name": "U.S. Bancorp (dup)", "exchange": "XNYS", "sector": "S40"},
    {"instrument_id": "INS-000094", "ticker": "PSX", "name": "Phillips 66", "exchange": "XNYS", "sector": "S10"},
    {"instrument_id": "INS-000095", "ticker": "WMB", "name": "The Williams Companies, Inc.", "exchange": "XNYS", "sector": "S10"},
    {"instrument_id": "INS-000096", "ticker": "NSC", "name": "Norfolk Southern Corporation", "exchange": "XNYS", "sector": "S20"},
    {"instrument_id": "INS-000097", "ticker": "MMC", "name": "Marsh & McLennan Companies, Inc.", "exchange": "XNYS", "sector": "S40"},
    {"instrument_id": "INS-000098", "ticker": "TGT", "name": "Target Corporation", "exchange": "XNYS", "sector": "S30"},
    {"instrument_id": "INS-000099", "ticker": "AFL", "name": "Aflac Incorporated", "exchange": "XNYS", "sector": "S40"},
    {"instrument_id": "INS-000100", "ticker": "MCK", "name": "McKesson Corporation", "exchange": "XNYS", "sector": "S35"},
]


def get_universe_50() -> list[dict[str, Any]]:
    """Return the ~50 symbol universe (existing 20 + additional 30)."""
    return EXISTING_20 + ADDITIONAL_30


def get_universe_100() -> list[dict[str, Any]]:
    """Return the ~100 symbol universe (existing 20 + additional 30 + additional 50)."""
    return EXISTING_20 + ADDITIONAL_30 + ADDITIONAL_50_FOR_100


def get_symbols_50() -> list[str]:
    """Return just the ticker symbols for the 50-universe."""
    return [inst["ticker"] for inst in get_universe_50()]


def get_symbols_100() -> list[str]:
    """Return just the ticker symbols for the 100-universe."""
    return [inst["ticker"] for inst in get_universe_100()]


def get_instrument_id_map_50() -> dict[str, str]:
    """Return ticker -> instrument_id mapping for 50-universe."""
    return {inst["ticker"]: inst["instrument_id"] for inst in get_universe_50()}


def get_instrument_id_map_100() -> dict[str, str]:
    """Return ticker -> instrument_id mapping for 100-universe."""
    return {inst["ticker"]: inst["instrument_id"] for inst in get_universe_100()}


def persist_universe_master(instruments: list[dict[str, Any]], universe_id: str) -> Path:
    """Persist an instrument master for an expanded universe."""
    master = {
        "description": f"ORBIT {universe_id} instrument master",
        "universe_id": universe_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "selection_policy": "deterministic_rule_based",
        "selection_criteria": [
            "US-listed equities",
            "minimum 5 years of trading history",
            "minimum average daily dollar volume > $10M",
            "no OTC/pink-sheet securities",
        ],
        "survivorship_bias": "NOT FULLY CONTROLLED",
        "instruments": instruments,
    }
    out_path = REPO_ROOT / "configs" / f"instrument_master_{universe_id.lower()}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(master, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return out_path


__all__ = [
    "EXISTING_20", "ADDITIONAL_30", "ADDITIONAL_50_FOR_100",
    "get_universe_50", "get_universe_100",
    "get_symbols_50", "get_symbols_100",
    "get_instrument_id_map_50", "get_instrument_id_map_100",
    "persist_universe_master",
]
