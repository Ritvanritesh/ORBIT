"""Phase 12B fundamental data ingestion pipeline with PIT availability."""
from __future__ import annotations
from typing import Any
import json, os
from pathlib import Path
import polars as pl
from datetime import datetime, date
from dateutil import parser as date_parser
from orbit.ml.phase12b_plan import PHASE12B_STALENESS

REPO_ROOT = Path(__file__).resolve().parents[3]


def _safe_int(val) -> int | None:
    """Safely convert to int, returning None for null/NaN."""
    if val is None or (isinstance(val, float) and val != val):  # NaN check
        return None
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


def _safe_float(val) -> float | None:
    """Safely convert to float, returning None for null/NaN."""
    if val is None or (isinstance(val, float) and val != val):  # NaN check
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _safe_date(val) -> date | None:
    """Safely convert to date object."""
    if val is None:
        return None
    try:
        return date_parser.parse(str(val)).date()
    except Exception:
        return None


def load_sec_edgar_companyfacts(snapshot_id: str) -> pl.DataFrame:
    """Load SEC EDGAR companyfacts data for a given snapshot.

    Returns a DataFrame with columns:
    - instrument_id: canonical instrument identifier
    - filing_date: date the filing became publicly available
    - period_end: fiscal period end date
    - eps: earnings per share
    - book_value_per_share: book value per share
    - revenue: trailing 12-month revenue
    - operating_income: operating income
    - net_income: net income
    - total_assets: total assets
    - shareholders_equity: shareholders equity
    - operating_cash_flow: operating cash flow
    - total_debt: total debt
    - current_assets: current assets
    - current_liabilities: current liabilities
    - source_version: version of the data
    - raw_filing_id: SEC CIK/filing identifier
    """
    d = REPO_ROOT / "data" / "normalized" / "fundamentals" / "sec_edgar_companyfacts" / snapshot_id
    if not d.is_dir():
        raise FileNotFoundError(f"SEC EDGAR snapshot not found: {d}")

    records = []
    for filepath in sorted(d.glob('*.json')):
        try:
            with open(filepath, 'rb') as f:
                raw_bytes = f.read()
            # Try different encodings
            for enc in ['utf-8', 'cp1252', 'latin1']:
                try:
                    text = raw_bytes.decode(enc)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                continue  # all encodings failed

            data = json.loads(text)

            # Extract company info
            company = data.get("company", {})
            cik = company.get("cik", None)

            # Extract facts
            facts = data.get("facts", {})
            if not isinstance(facts, dict):
                continue

            # Map SEC fact names to our feature names
            # Based on SEC EDGAR companyfacts structure
            mapping = {
                # Valuation
                "earnings_yield": None,  # computed from EPS/price
                "book_to_market": None,  # computed from book_value/price
                "sales_to_price": None,  # computed from revenue/price

                # Profitability
                "roa": None,  # net_income / total_assets
                "roe": None,  # net_income / shareholders_equity
                "operating_margin": None,  # operating_income / revenue
                "gross_profitability": None,  # gross_profit / revenue

                # Growth
                "revenue_growth": None,  # (rev(D)-rev(D-1))/rev(D-1)
                "earnings_growth": None,  # (EPS(D)-EPS(D-1))/EPS(D-1)
                "cash_flow_growth": None,  # (OCF(D)-OCF(D-1))/OCF(D-1)

                # Leverage
                "debt_to_equity": None,  # total_debt / equity
                "debt_to_assets": None,  # total_debt / total_assets
                "current_ratio": None,  # current_assets / current_liabilities
            }

            # Extract key fundamental values
            # These field names come from SEC EDGAR companyfacts JSON structure
            extracted = {}

            # Try common SEC fact field names
            fact_fields = {
                "eps": ["eps", "earnings_per_share", "basic_eps"],
                "book_value_per_share": ["book_value_per_common_share", "book_value_share"],
                "revenue": ["revenue", "revenues"],
                "operating_income": ["operating_income", "operating_income_loss"],
                "net_income": ["net_income", "net_income_loss"],
                "total_assets": ["total_assets"],
                "shareholders_equity": ["shareholders_equity", "stockholders_equity"],
                "operating_cash_flow": ["operating_cash_flow", "cash_flow_from_operations"],
                "total_debt": ["total_debt"],
                "current_assets": ["current_assets"],
                "current_liabilities": ["current_liabilities"],
            }

            for feat_name, possible_names in fact_fields.items():
                for pname in possible_names:
                    if pname in facts:
                        val = facts[pname]
                        # Handle both numeric and object forms
                        if isinstance(val, (int, float)):
                            extracted[feat_name] = _safe_float(val)
                        elif isinstance(val, dict):
                            # SEC sometimes wraps in {val: ..., unit: ...}
                            extracted[feat_name] = _safe_float(val.get("val"))
                        elif isinstance(val, list) and len(val) > 0:
                            extracted[feat_name] = _safe_float(val[0])
                        # If already the right type, keep it
                        break  # found this feature

            # Also try direct field access for known SEC fact names
            # (these are common in companyfacts)
            for sec_field in ["eps", "revTWelveMonths", "ib", "ni", "at", "seq", "oancf",
                              "dvt", "currassets", "currli"]:
                if sec_field in facts:
                    val = facts[sec_field]
                    if isinstance(val, (int, float)):
                        # Map to our feature names
                        if sec_field == "eps":
                            extracted["eps"] = _safe_float(val)
                        elif sec_field in ("revTWelveMonths", "revenues"):
                            extracted["revenue"] = _safe_float(val)
                        elif sec_field == "ib":
                            extracted["operating_income"] = _safe_float(val)
                        elif sec_field == "ni":
                            extracted["net_income"] = _safe_float(val)
                        elif sec_field == "at":
                            extracted["total_assets"] = _safe_float(val)
                        elif sec_field == "seq":
                            extracted["shareholders_equity"] = _safe_float(val)
                        elif sec_field == "oancf":
                            extracted["operating_cash_flow"] = _safe_float(val)
                        elif sec_field == "dvt":
                            extracted["total_debt"] = _safe_float(val)
                        elif sec_field == "currassets":
                            extracted["current_assets"] = _safe_float(val)
                        elif sec_field == "currli":
                            extracted["current_liabilities"] = _safe_float(val)

            # Compute valuation features from extracted data
            if "eps" in extracted and "revenue" in extracted:
                # earnings_yield = EPS / price - but price is from market data
                # We'll store EPS and let the feature builder compute yield
                pass

            record = {
                "instrument_id": data.get("ticker", data.get("cik", "UNKNOWN")),
                "filing_date": _safe_date(data.get("filing_date")),
                "period_end": _safe_date(data.get("period_end_date")),
                "eps": extracted.get("eps"),
                "book_value_per_share": extracted.get("book_value_per_share"),
                "revenue": extracted.get("revenue"),
                "operating_income": extracted.get("operating_income"),
                "net_income": extracted.get("net_income"),
                "total_assets": extracted.get("total_assets"),
                "shareholders_equity": extracted.get("shareholders_equity"),
                "operating_cash_flow": extracted.get("operating_cash_flow"),
                "total_debt": extracted.get("total_debt"),
                "current_assets": extracted.get("current_assets"),
                "current_liabilities": extracted.get("current_liabilities"),
                "source_version": data.get("version", "unknown"),
                "raw_filing_id": data.get("fy", None),
            }
            records.append(record)
        except Exception as e:
            # Log error but continue with other files
            print(f"Error processing {filepath.name}: {e}")
            continue

    if not records:
        return pl.DataFrame({
            "instrument_id": pl.Series([], dtype=pl.Utf8),
            "filing_date": pl.Series([], dtype=pl.Date),
            "period_end": pl.Series([], dtype=pl.Date),
            "eps": pl.Series([], dtype=pl.Float64),
            "book_value_per_share": pl.Series([], dtype=pl.Float64),
            "revenue": pl.Series([], dtype=pl.Float64),
            "operating_income": pl.Series([], dtype=pl.Float64),
            "net_income": pl.Series([], dtype=pl.Float64),
            "total_assets": pl.Series([], dtype=pl.Float64),
            "shareholders_equity": pl.Series([], dtype=pl.Float64),
            "operating_cash_flow": pl.Series([], dtype=pl.Float64),
            "total_debt": pl.Series([], dtype=pl.Float64),
            "current_assets": pl.Series([], dtype=pl.Float64),
            "current_liabilities": pl.Series([], dtype=pl.Float64),
            "source_version": pl.Series([], dtype=pl.Utf8),
            "raw_filing_id": pl.Series([], dtype=pl.Utf8),
        })

    df = pl.DataFrame(records)
    # Ensure proper types
    df = df.with_columns([
        pl.col("eps").cast(pl.Float64),
        pl.col("book_value_per_share").cast(pl.Float64),
        pl.col("revenue").cast(pl.Float64),
        pl.col("operating_income").cast(pl.Float64),
        pl.col("net_income").cast(pl.Float64),
        pl.col("total_assets").cast(pl.Float64),
        pl.col("shareholders_equity").cast(pl.Float64),
        pl.col("operating_cash_flow").cast(pl.Float64),
        pl.col("total_debt").cast(pl.Float64),
        pl.col("current_assets").cast(pl.Float64),
        pl.col("current_liabilities").cast(pl.Float64),
    ])

    return df


def compute_fundamental_features(
    fundamental_df: pl.DataFrame,
    price_data: pl.DataFrame,
    as_of_date: date,
) -> dict[str, float | None]:
    """Compute fundamental features using AS-of join policy.

    For the given as_of_date, use the latest fundamental observation
    whose filing_date <= as_of_date.

    Parameters
    ----------
    fundamental_df : DataFrame with fundamental data per instrument
    price_data : DataFrame with price data (must have 'price' column)
    as_of_date : date, the session date for which to compute features

    Returns
    -------
    dict of feature_name -> value (or None if not available)
    """
    # Filter fundamentals available by as_of_date
    available = fundamental_df.filter(pl.col("filing_date") <= as_of_date)

    if available.height == 0:
        # No fundamentals available by this date
        # Return all None
        return {
            "earnings_yield": None,
            "book_to_market": None,
            "sales_to_price": None,
            "roa": None,
            "roe": None,
            "operating_margin": None,
            "gross_profitability": None,
            "revenue_growth": None,
            "earnings_growth": None,
            "cash_flow_growth": None,
            "debt_to_equity": None,
            "debt_to_assets": None,
            "current_ratio": None,
        }

    # For each instrument, take the LATEST available fundamental
    # (most recent filing_date <= as_of_date)
    latest = available.group_by("instrument_id").agg(
        pl.col("filing_date").max().alias("max_filing"),
    )
    # Join back to get the full record
    available = available.join(latest, on=["instrument_id", "filing_date"])

    # Now compute features for each instrument
    results = {}
    for row in available.iter_rows(named=True):
        iid = row["instrument_id"]

        # Compute features
        eps = row["eps"]
        bvps = row["book_value_per_share"]
        rev = row["revenue"]
        oi = row["operating_income"]
        ni = row["net_income"]
        ta = row["total_assets"]
        eq = row["shareholders_equity"]
        ocf = row["operating_cash_flow"]
        td = row["total_debt"]
        ca = row["current_assets"]
        cl = row["current_liabilities"]

        # Valuation features (need price - we'll skip these for now
        # or mark as needing market data)
        # earnings_yield = eps / price - need market price
        # book_to_market = bvps / price - need market price
        # sales_to_price = rev / price - need market price

        # Profitability features (no price needed)
        roa = None if ta == 0 or ni is None else ni / ta
        roe = None if eq == 0 or ni is None else ni / eq
        operating_margin = None if rev == 0 or oi is None else oi / rev
        gross_profitability = None if rev == 0 else None  # gross profit not directly extracted

        # Growth features - need prior period data
        # These require comparison to previous fundamental, which is complex
        # We'll mark as None for now; full implementation would track
        # the prior period's values
        revenue_growth = None
        earnings_growth = None
        cash_flow_growth = None

        # Leverage features
        debt_to_equity = None if eq == 0 or td is None else td / eq
        debt_to_assets = None if ta == 0 or td is None else td / ta
        current_ratio = None if cl == 0 or ca is None else ca / cl

        results[iid] = {
            "earnings_yield": None,  # need market price
            "book_to_market": None,  # need market price
            "sales_to_price": None,  # need market price
            "roa": roa,
            "roe": roe,
            "operating_margin": operating_margin,
            "gross_profitability": gross_profitability,
            "revenue_growth": revenue_growth,
            "earnings_growth": earnings_growth,
            "cash_flow_growth": cash_flow_growth,
            "debt_to_equity": debt_to_equity,
            "debt_to_assets": debt_to_assets,
            "current_ratio": current_ratio,
        }

    return results


def validate_pit_compliance(fundamental_df: pl.DataFrame, as_of_date: date) -> dict[str, Any]:
    """Validate point-in-time compliance for a given as_of_date.

    Checks:
    - No future filings used (filing_date <= as_of_date)
    - Staleness check: fundamental not older than max_age_years
    - Availability: at least one fundamental available
    """
    # Check no future filings
    future = fundamental_df.filter(pl.col("filing_date") > as_of_date)
    has_future = future.height > 0

    # Staleness check
    if fundamental_df.height > 0:
        oldest_filing = fundamental_df["filing_date"].min()
        age_years = (as_of_date - oldest_filing).days / 365.25
        too_stale = age_years > PHASE12B_STALENESS["max_age_years"]
    else:
        too_stale = True

    # Availability
    available_by_date = fundamental_df.filter(pl.col("filing_date") <= as_of_date)
    has_available = available_by_date.height > 0

    return {
        "has_future_filings": has_future,
        "too_stale": too_stale,
        "has_available": has_available,
        "compliant": not (has_future or too_stale or not has_available),
    }