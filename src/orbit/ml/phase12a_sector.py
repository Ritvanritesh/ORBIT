"""Phase 12A Family B: Sector Context Features (vectorized)."""
from __future__ import annotations
from typing import Any
import polars as pl


def load_sector_mapping(instruments: list[Any]) -> dict[str, str]:
    """Build instrument_id -> sector_id mapping."""
    mapping = {}
    for inst in instruments:
        inst_id = inst.instrument_id
        sector = getattr(inst, "sector_id", None) or getattr(inst, "sector", None)
        if sector:
            mapping[inst_id] = sector
    return mapping


def compute_sector_features(
    bars: pl.DataFrame,
    sector_map: dict[str, str],
    universe_sessions: pl.DataFrame,
    min_sector_members: int = 2,
) -> pl.DataFrame:
    """Compute sector features using vectorized polars operations."""
    # Add sector column
    sector_df = pl.DataFrame([
        {"instrument_id": k, "sector": v} for k, v in sector_map.items()
    ])
    bars_s = bars.join(sector_df, on="instrument_id", how="inner")

    # Compute daily returns
    bars_s = bars_s.sort(["instrument_id", "trade_date"])
    bars_s = bars_s.with_columns([
        pl.col("close").pct_change().over("instrument_id").alias("daily_return"),
    ])

    # For each session, compute rolling sector aggregates
    # First: per-instrument rolling stats (using shift for PIT)
    inst_stats = bars_s.group_by("instrument_id").agg([
        pl.col("trade_date").alias("_dates"),
        pl.col("daily_return").alias("_returns"),
        pl.col("sector").first().alias("_sector"),
    ])

    # We need per-session features, so iterate over sessions
    # But use vectorized operations within each session
    all_sessions = (
        universe_sessions.select("decision_session")
        .unique()
        .sort("decision_session")
    )

    # Build a lookup: for each instrument, its returns indexed by date
    # Then compute sector aggregates per date
    results = []

    # Pre-compute per-instrument return series
    inst_returns = {}
    inst_sectors = {}
    for row in bars_s.select("instrument_id", "trade_date", "daily_return", "sector").iter_rows(named=True):
        iid = row["instrument_id"]
        if iid not in inst_returns:
            inst_returns[iid] = {}
            inst_sectors[iid] = row["sector"]
        inst_returns[iid][row["trade_date"]] = row["daily_return"]

    # Pre-compute per-session sector aggregates
    session_dates = sorted(set(d for returns in inst_returns.values() for d in returns.keys()))

    for d_row in all_sessions.iter_rows(named=True):
        d = d_row["decision_session"]

        # Get instruments in universe at this session
        session_insts = set(
            universe_sessions.filter(pl.col("decision_session") == d)["instrument_id"].to_list()
        )

        # Collect prior returns for each instrument
        sector_returns = {}  # sector -> list of (inst, ret_20, ret_5, vol_20)
        for iid in session_insts:
            if iid not in inst_returns:
                continue
            dates = sorted(inst_returns[iid].keys())
            prior_dates = [dt for dt in dates if dt < d]
            if len(prior_dates) < 5:
                continue
            last_date = prior_dates[-1]
            rets = [inst_returns[iid][dt] for dt in prior_dates]
            rets_20 = rets[-20:] if len(rets) >= 20 else rets
            rets_5 = rets[-5:]
            sector = inst_sectors.get(iid, "UNKNOWN")

            if sector not in sector_returns:
                sector_returns[sector] = []
            sector_returns[sector].append({
                "instrument_id": iid,
                "ret_20": sum(rets_20) / len(rets_20) if rets_20 else 0,
                "ret_5": sum(rets_5) / len(rets_5) if rets_5 else 0,
                "vol_20": (sum((r - sum(rets_20)/len(rets_20))**2 for r in rets_20) / len(rets_20))**0.5 if len(rets_20) > 1 else 0,
            })

        # Compute sector aggregates
        sector_stats = {}
        for sector, members in sector_returns.items():
            if len(members) < min_sector_members:
                continue
            mean_ret_20 = sum(m["ret_20"] for m in members) / len(members)
            std_ret_20 = (sum((m["ret_20"] - mean_ret_20)**2 for m in members) / len(members))**0.5
            sector_stats[sector] = {
                "sector_ret_20": mean_ret_20,
                "sector_vol_20": std_ret_20,
                "sector_ret_5": sum(m["ret_5"] for m in members) / len(members),
                "sector_dispersion_20": std_ret_20,
            }

        # Emit rows for instruments in sectors with enough members
        for sector, members in sector_returns.items():
            if sector not in sector_stats:
                continue
            ss = sector_stats[sector]
            for m in members:
                results.append({
                    "instrument_id": m["instrument_id"],
                    "decision_session": d,
                    "window_end_session": last_date if prior_dates else d,
                    "sector_ret_20": ss["sector_ret_20"],
                    "sector_vol_20": ss["sector_vol_20"],
                    "sector_ret_5": ss["sector_ret_5"],
                    "sector_trend_5_30": None,
                    "sector_dispersion_20": ss["sector_dispersion_20"],
                })

    if not results:
        return pl.DataFrame()

    return pl.DataFrame(results).sort(["instrument_id", "decision_session"])
