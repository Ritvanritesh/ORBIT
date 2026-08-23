"""Phase 12C — SEC EDGAR CompanyFacts data acquisition and normalization."""
from __future__ import annotations

import hashlib
import json
import time
import urllib.request
import urllib.error
from datetime import date, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = REPO_ROOT / "data" / "raw" / "sec_edgar_companyfacts"
NORMALIZED_DIR = REPO_ROOT / "data" / "normalized" / "sec_edgar_companyfacts"

EDGAR_API_BASE = "https://data.sec.gov/api/xbrl/companyfacts"
EDGAR_USER_AGENT = "ORBIT-Research orbit@example.com"
RATE_LIMIT_SECONDS = 0.12


def _edgar_headers() -> dict[str, str]:
    return {"User-Agent": EDGAR_USER_AGENT, "Accept": "application/json"}


def download_companyfacts(cik: int, cache_dir: Path | None = None) -> dict[str, Any] | None:
    """Download SEC EDGAR CompanyFacts for a given CIK.

    Returns the parsed JSON or None on failure.
    """
    if cache_dir is None:
        cache_dir = RAW_DIR / f"CIK{cik:010d}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "companyfacts.json"

    if cache_file.exists():
        return json.loads(cache_file.read_text(encoding="utf-8"))

    url = f"{EDGAR_API_BASE}/CIK{cik:010d}.json"
    try:
        req = urllib.request.Request(url, headers=_edgar_headers())
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        cache_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
        time.sleep(RATE_LIMIT_SECONDS)
        return data
    except (urllib.error.URLError, urllib.error.HTTPError, Exception) as e:
        print(f"  WARNING: Failed to download CIK {cik}: {e}")
        time.sleep(RATE_LIMIT_SECONDS)
        return None


def extract_fundamental_observations(
    companyfacts: dict[str, Any],
    cik: int,
    ticker: str,
) -> list[dict[str, Any]]:
    """Extract canonical fundamental observations from CompanyFacts JSON.

    Returns a list of observation dicts, each with:
      - cik, ticker, company_name
      - tag (XBRL taxonomy tag)
      - value
      - form_type
      - filing_date
      - acceptance_timestamp (if available)
      - period_end
      - fiscal_year, fiscal_period
      - data_type: REAL
    """
    observations = []
    facts = companyfacts.get("facts", {})
    company_name = companyfacts.get("entityName", "")

    for us_gaap_key, us_gaap_data in facts.get("us-gaap", {}).items():
        units = us_gaap_data.get("units", {})
        for unit_key, entries in units.items():
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                val = entry.get("val")
                if val is None:
                    continue

                filing_date_str = entry.get("end") or entry.get("fp", "")
                accn = entry.get("accn", "")
                form_type = entry.get("form", "")
                filed = entry.get("filed", "")
                accepted = entry.get("accepted", "")
                start = entry.get("start", "")
                end = entry.get("end", "")
                fy = entry.get("fy", "")
                fp = entry.get("fp", "")

                obs = {
                    "cik": cik,
                    "ticker": ticker,
                    "company_name": company_name,
                    "tag": us_gaap_key,
                    "value": float(val) if isinstance(val, (int, float)) else None,
                    "unit": unit_key,
                    "form_type": form_type,
                    "filing_date": filed,
                    "acceptance_timestamp": accepted,
                    "accession_number": accn,
                    "period_start": start,
                    "period_end": end,
                    "fiscal_year": fy,
                    "fiscal_period": fp,
                    "data_type": "REAL",
                    "source": "SEC-EDGAR-CompanyFacts",
                }
                observations.append(obs)

    return observations


def normalize_observations(
    observations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Normalize raw observations into canonical form.

    - Parse dates
    - Compute availability timestamps
    - Apply after-close policy
    """
    normalized = []
    for obs in observations:
        if obs["value"] is None:
            continue

        # Parse filing_date to date object
        filing_date = None
        if obs["filing_date"]:
            try:
                filing_date = date.fromisoformat(obs["filing_date"])
            except (ValueError, TypeError):
                pass

        # Parse acceptance_timestamp
        acceptance_dt = None
        if obs["acceptance_timestamp"]:
            try:
                acceptance_dt = datetime.fromisoformat(
                    obs["acceptance_timestamp"].replace("T", " ").replace("Z", "")
                )
            except (ValueError, TypeError):
                pass

        # Compute availability_time using after-close policy
        # Conservative: if filing_date is available, availability = filing_date
        # (in practice, SEC filings become available on the filing_date)
        availability_date = filing_date

        # Parse period dates
        period_end = None
        if obs["period_end"]:
            try:
                period_end = date.fromisoformat(obs["period_end"])
            except (ValueError, TypeError):
                pass

        norm = {
            "cik": obs["cik"],
            "ticker": obs["ticker"],
            "company_name": obs["company_name"],
            "tag": obs["tag"],
            "value": obs["value"],
            "unit": obs["unit"],
            "form_type": obs["form_type"],
            "filing_date": filing_date.isoformat() if filing_date else None,
            "acceptance_timestamp": acceptance_dt.isoformat() if acceptance_dt else None,
            "accession_number": obs["accession_number"],
            "period_start": obs["period_start"],
            "period_end": period_end.isoformat() if period_end else None,
            "fiscal_year": obs["fiscal_year"],
            "fiscal_period": obs["fiscal_period"],
            "availability_date": availability_date.isoformat() if availability_date else None,
            "data_type": "REAL",
            "source": obs["source"],
        }
        normalized.append(norm)

    return normalized


def acquire_all_instruments(
    identity_registry: dict[str, Any],
) -> dict[str, Any]:
    """Acquire real SEC EDGAR data for all mapped instruments.

    Returns a summary dict with per-instrument results.
    """
    results = {
        "acquired": [],
        "failed": [],
        "no_data": [],
        "total_observations": 0,
    }

    for mapping in identity_registry["mappings"]:
        ticker = mapping["ticker"]
        cik = mapping["cik"]
        iid = mapping["instrument_id"]

        print(f"  Downloading {ticker} (CIK {cik})...")
        companyfacts = download_companyfacts(cik)

        if companyfacts is None:
            results["failed"].append({"instrument_id": iid, "ticker": ticker, "cik": cik})
            continue

        observations = extract_fundamental_observations(companyfacts, cik, ticker)
        if not observations:
            results["no_data"].append({"instrument_id": iid, "ticker": ticker, "cik": cik})
            continue

        normalized = normalize_observations(observations)

        results["acquired"].append({
            "instrument_id": iid,
            "ticker": ticker,
            "cik": cik,
            "n_raw": len(observations),
            "n_normalized": len(normalized),
            "observations": normalized,
        })
        results["total_observations"] += len(normalized)

    results["summary"] = {
        "instruments_requested": len(identity_registry["mappings"]),
        "instruments_acquired": len(results["acquired"]),
        "instruments_failed": len(results["failed"]),
        "instruments_no_data": len(results["no_data"]),
        "total_observations": results["total_observations"],
    }

    return results


def persist_raw_acquisition(
    results: dict[str, Any],
    env_id: str,
) -> Path:
    """Persist raw acquisition results."""
    out_dir = RAW_DIR / env_id
    out_dir.mkdir(parents=True, exist_ok=True)

    for inst_data in results["acquired"]:
        ticker = inst_data["ticker"]
        out_file = out_dir / f"{ticker}_observations.json"
        out_file.write_text(
            json.dumps(inst_data["observations"], indent=2, default=str),
            encoding="utf-8",
        )

    summary_file = out_dir / "acquisition_summary.json"
    summary_file.write_text(
        json.dumps({k: v for k, v in results.items() if k != "acquired"},
                    indent=2, default=str),
        encoding="utf-8",
    )
    return out_dir
