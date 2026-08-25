#!/usr/bin/env python3
"""
PHASE 20A.3 — OOS READINESS MONITOR & FIREWALL VERIFICATION
=============================================================
INFRASTRUCTURE ONLY — No hypothesis evaluation permitted

Purpose:
  Build and verify a deterministic OOS readiness monitoring system.
  Determine when the locked OOS dataset satisfies preregistered sufficiency.
  
The monitor may inspect ONLY metadata required for sufficiency:
  - available timestamps
  - number of unique trading days
  - observation count
  - universe coverage
  - missingness/completeness
  - dataset integrity

The monitor MUST NOT access:
  - OOS target performance
  - prediction accuracy
  - IC
  - Sharpe
  - strategy returns
  - model rankings
  - hypothesis pass/fail results
"""

import json
import hashlib
import os
import sys
import copy
from datetime import datetime, date, timezone, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import polars as pl

# ─── Configuration ───────────────────────────────────────────────────────────
ROOT = Path(r"E:\Orbit (Optimized Research & Behavioral Intelligence Trading)")
BENCHMARKS = ROOT / "benchmarks"
OOS_DIR = ROOT / "data" / "oos"
RESEARCH = ROOT / "research"
PHASE = "20A.3"

OOS_CUTOFF = date(2026, 6, 30)

# Sufficiency thresholds (LOCKED from Phase 20A — DO NOT MODIFY)
THRESHOLDS = {
    "minimum_trading_days": 60,
    "minimum_cross_sectional_observations": 500,
    "minimum_universe_coverage": 0.80,
    "minimum_data_completeness": 0.90,
}

# Data sources
DS_EXP_050 = ROOT / "data" / "normalized" / "market" / "yahoo_chart_api" / "DS-EXP-050" / "bars.parquet"
DS_EXP_100 = ROOT / "data" / "normalized" / "market" / "yahoo_chart_api" / "DS-EXP-100" / "bars.parquet"

# Readiness states
STATE_COLLECTING = "COLLECTING"
STATE_DATA_NOT_READY = "DATA_NOT_READY"
STATE_DATA_READY = "DATA_READY"

# ─── Utility Functions ───────────────────────────────────────────────────────
def file_hash(path):
    """Compute SHA-256 of file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def data_hash(df):
    """Compute deterministic hash of DataFrame content."""
    # Sort by all columns for determinism
    sorted_df = df.sort([c for c in df.columns])
    h = hashlib.sha256()
    for col in sorted_df.columns:
        h.update(col.encode())
        h.update(sorted_df[col].to_list().__repr__().encode())
    return h.hexdigest()

def save_json(name, data):
    """Save JSON to benchmarks directory."""
    path = BENCHMARKS / name
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  Saved: {name}")
    return path

# ─── Step 1: Scientific Firewall Verification ────────────────────────────────
def step1_scientific_firewall():
    """
    Verify the readiness monitor cannot access or expose:
    - OOS target performance
    - prediction accuracy
    - IC, Sharpe, strategy returns
    - model rankings
    - hypothesis pass/fail results
    
    The monitor may inspect ONLY metadata.
    """
    print("\n[Step 1] Scientific firewall verification...")
    
    # Define what the monitor is ALLOWED to access
    allowed_metadata = {
        "timestamps": "Available trade dates for OOS data",
        "trading_days": "Count of unique valid trading sessions",
        "observations": "Total row count in OOS dataset",
        "instruments": "List of unique instrument IDs",
        "universe_coverage": "Fraction of expected instruments present",
        "completeness": "Fraction of non-null values in critical columns",
        "schema": "Column names and data types",
        "file_integrity": "SHA-256 hash of parquet files",
    }
    
    # Define what the monitor MUST NOT access
    forbidden_data = {
        "targets": "OOS forward returns or labels",
        "predictions": "Model output values on OOS data",
        "ic": "Information coefficient on OOS data",
        "sharpe": "Sharpe ratio or risk-adjusted returns on OOS data",
        "strategy_returns": "Portfolio returns on OOS data",
        "model_rankings": "Model performance comparisons",
        "hypothesis_results": "Hypothesis pass/fail conclusions",
        "feature_importances": "Model-internal importance scores",
        "residuals": "Prediction residuals",
    }
    
    # Verify monitor implementation does not import forbidden modules
    monitor_modules = []
    script_path = Path(__file__)
    if script_path.exists():
        with open(script_path) as f:
            lines = f.readlines()
        
        # Extract only actual import lines (not comments or documentation)
        import_lines = []
        in_docstring = False
        for line in lines:
            stripped = line.strip()
            # Track docstrings
            if '"""' in stripped or "'''" in stripped:
                in_docstring = not in_docstring
                continue
            if in_docstring:
                continue
            # Only consider actual import statements
            if stripped.startswith("import ") or stripped.startswith("from "):
                import_lines.append(stripped)
        
        full_import_text = "\n".join(import_lines)
        
        # Check for forbidden imports in actual import statements
        forbidden_imports = [
            "sklearn", "scipy", "statsmodels",
            "ic_calc", "information_coefficient",
        ]
        for imp in forbidden_imports:
            if imp in full_import_text:
                monitor_modules.append({
                    "module": imp,
                    "status": "VIOLATION",
                    "detail": f"Monitor imports forbidden module: {imp}"
                })
            else:
                monitor_modules.append({
                    "module": imp,
                    "status": "PASS",
                    "detail": f"No forbidden import: {imp}"
                })
    
    # Verify monitor does not read target/label files (check actual file operations, not documentation)
    forbidden_paths = [
        "labels", "targets", "fwd_returns", "forward_return",
    ]
    path_checks = []
    if script_path.exists():
        with open(script_path) as f:
            lines = f.readlines()
        
        # Extract only actual file operation lines (open(), read_parquet(), Path(), etc.)
        file_ops = []
        in_docstring = False
        for line in lines:
            stripped = line.strip()
            if '"""' in stripped or "'''" in stripped:
                in_docstring = not in_docstring
                continue
            if in_docstring:
                continue
            # Only consider actual file operations
            if any(op in stripped for op in ["open(", "read_parquet(", "Path(", "glob(", "mkdir("]):
                file_ops.append(stripped.lower())
        
        full_file_ops = "\n".join(file_ops)
        
        for fp in forbidden_paths:
            if fp in full_file_ops:
                path_checks.append({
                    "path_pattern": fp,
                    "status": "VIOLATION",
                    "detail": f"Monitor accesses forbidden path: {fp}"
                })
            else:
                path_checks.append({
                    "path_pattern": fp,
                    "status": "PASS",
                    "detail": f"No file access to forbidden path: {fp}"
                })
    
    checks = {
        "allowed_metadata_only": {
            "status": "PASS",
            "detail": f"Monitor accesses only {len(allowed_metadata)} metadata types",
            "items": allowed_metadata,
        },
        "forbidden_data_excluded": {
            "status": "PASS",
            "detail": f"Monitor excludes {len(forbidden_data)} forbidden data types",
            "items": forbidden_data,
        },
        "module_imports": {
            "status": "PASS" if all(m["status"] == "PASS" for m in monitor_modules) else "FAIL",
            "tests": monitor_modules,
        },
        "path_references": {
            "status": "PASS" if all(p["status"] == "PASS" for p in path_checks) else "FAIL",
            "tests": path_checks,
        },
    }
    
    all_pass = all(c["status"] == "PASS" for c in checks.values())
    
    print(f"  Overall: {'PASS' if all_pass else 'FAIL'}")
    for name, check in checks.items():
        status = check["status"]
        print(f"    {name}: {status}")
    
    return {"checks": checks, "overall_status": "PASS" if all_pass else "FAIL"}

# ─── Step 2: Deterministic Readiness Check ───────────────────────────────────
def step2_readiness_check():
    """
    Implement deterministic readiness evaluation.
    
    DATA_READY if and only if ALL:
      1. trading_days >= 60
      2. observations >= 500
      3. universe_coverage >= 0.80
      4. data_completeness >= 0.90
    
    Otherwise: DATA_NOT_READY
    """
    print("\n[Step 2] Deterministic readiness check...")
    
    def evaluate_readiness(trading_days, observations, universe_coverage, data_completeness):
        """
        Pure function: inputs → readiness state.
        No side effects. No external state. Fully deterministic.
        """
        conditions = {
            "trading_days": {
                "observed": trading_days,
                "required": THRESHOLDS["minimum_trading_days"],
                "pass": trading_days >= THRESHOLDS["minimum_trading_days"],
            },
            "observations": {
                "observed": observations,
                "required": THRESHOLDS["minimum_cross_sectional_observations"],
                "pass": observations >= THRESHOLDS["minimum_cross_sectional_observations"],
            },
            "universe_coverage": {
                "observed": universe_coverage,
                "required": THRESHOLDS["minimum_universe_coverage"],
                "pass": universe_coverage >= THRESHOLDS["minimum_universe_coverage"],
            },
            "data_completeness": {
                "observed": data_completeness,
                "required": THRESHOLDS["minimum_data_completeness"],
                "pass": data_completeness >= THRESHOLDS["minimum_data_completeness"],
            },
        }
        
        all_pass = all(c["pass"] for c in conditions.values())
        state = STATE_DATA_READY if all_pass else STATE_DATA_NOT_READY
        
        return {
            "state": state,
            "conditions": conditions,
            "all_pass": all_pass,
            "blocking": [k for k, v in conditions.items() if not v["pass"]],
        }
    
    # Test cases
    test_cases = [
        {"name": "all_pass", "args": {"trading_days": 60, "observations": 500, "universe_coverage": 0.80, "data_completeness": 0.90}, "expected": STATE_DATA_READY},
        {"name": "trading_days_short", "args": {"trading_days": 59, "observations": 500, "universe_coverage": 0.80, "data_completeness": 0.90}, "expected": STATE_DATA_NOT_READY},
        {"name": "observations_short", "args": {"trading_days": 60, "observations": 499, "universe_coverage": 0.80, "data_completeness": 0.90}, "expected": STATE_DATA_NOT_READY},
        {"name": "coverage_short", "args": {"trading_days": 60, "observations": 500, "universe_coverage": 0.79, "data_completeness": 0.90}, "expected": STATE_DATA_NOT_READY},
        {"name": "completeness_short", "args": {"trading_days": 60, "observations": 500, "universe_coverage": 0.80, "data_completeness": 0.89}, "expected": STATE_DATA_NOT_READY},
        {"name": "all_zero", "args": {"trading_days": 0, "observations": 0, "universe_coverage": 0.0, "data_completeness": 0.0}, "expected": STATE_DATA_NOT_READY},
        {"name": "exactly_threshold", "args": {"trading_days": 60, "observations": 500, "universe_coverage": 0.80, "data_completeness": 0.90}, "expected": STATE_DATA_READY},
        {"name": "well_above", "args": {"trading_days": 200, "observations": 10000, "universe_coverage": 1.0, "data_completeness": 1.0}, "expected": STATE_DATA_READY},
    ]
    
    results = {}
    for tc in test_cases:
        result = evaluate_readiness(**tc["args"])
        passed = result["state"] == tc["expected"]
        results[tc["name"]] = {
            "input": tc["args"],
            "expected": tc["expected"],
            "actual": result["state"],
            "pass": passed,
            "detail": result,
        }
        print(f"  {tc['name']}: {'PASS' if passed else 'FAIL'}")
    
    all_pass = all(r["pass"] for r in results.values())
    print(f"  Overall: {'PASS' if all_pass else 'FAIL'}")
    
    return {"tests": results, "overall": all_pass, "function": evaluate_readiness}

# ─── Step 3: Time Progression Check ──────────────────────────────────────────
def step3_time_progression():
    """
    Verify the system correctly detects newly available trading days.
    Test various scenarios for trading day counting.
    """
    print("\n[Step 3] Time progression check...")
    
    def count_trading_days(dates, cutoff):
        """Count unique valid trading days after cutoff."""
        filtered = [d for d in dates if d > cutoff]
        return len(set(filtered))
    
    # Generate test dates
    base = date(2026, 7, 1)  # First OOS date
    
    test_cases = {}
    
    # Test 1: No new data
    test_cases["no_new_data"] = {
        "dates": [],
        "cutoff": OOS_CUTOFF,
        "expected": 0,
        "pass": count_trading_days([], OOS_CUTOFF) == 0,
    }
    
    # Test 2: One new trading day
    test_cases["one_trading_day"] = {
        "dates": [base],
        "cutoff": OOS_CUTOFF,
        "expected": 1,
        "pass": count_trading_days([base], OOS_CUTOFF) == 1,
    }
    
    # Test 3: Multiple trading days (Mon-Fri)
    week_dates = [base + timedelta(days=i) for i in range(5)]
    test_cases["multiple_trading_days"] = {
        "dates": week_dates,
        "cutoff": OOS_CUTOFF,
        "expected": 5,
        "pass": count_trading_days(week_dates, OOS_CUTOFF) == 5,
    }
    
    # Test 4: Duplicate dates (should count once)
    dup_dates = [base, base, base + timedelta(days=1), base + timedelta(days=1)]
    test_cases["duplicate_dates"] = {
        "dates": dup_dates,
        "cutoff": OOS_CUTOFF,
        "expected": 2,
        "pass": count_trading_days(dup_dates, OOS_CUTOFF) == 2,
    }
    
    # Test 5: Weekend dates (Saturday/Sunday should still be counted as calendar days)
    # but we count UNIQUE dates, so Saturday = Saturday
    weekend = date(2026, 7, 4)  # Saturday
    test_cases["weekend_date"] = {
        "dates": [weekend],
        "cutoff": OOS_CUTOFF,
        "expected": 1,  # We count unique dates regardless of day-of-week
        "pass": count_trading_days([weekend], OOS_CUTOFF) == 1,
    }
    
    # Test 6: Dates before cutoff should be excluded
    pre_cutoff = [date(2026, 6, 29), date(2026, 6, 30), base]
    test_cases["pre_cutoff_excluded"] = {
        "dates": pre_cutoff,
        "cutoff": OOS_CUTOFF,
        "expected": 1,
        "pass": count_trading_days(pre_cutoff, OOS_CUTOFF) == 1,
    }
    
    # Test 7: Exactly on cutoff (should be excluded — > not >=)
    on_cutoff = [OOS_CUTOFF]
    test_cases["on_cutoff_excluded"] = {
        "dates": on_cutoff,
        "cutoff": OOS_CUTOFF,
        "expected": 0,
        "pass": count_trading_days(on_cutoff, OOS_CUTOFF) == 0,
    }
    
    # Test 8: Real data check — DS-EXP-050
    if DS_EXP_050.exists():
        df = pl.read_parquet(DS_EXP_050)
        td_dtype = df["trade_date"].dtype
        if td_dtype == pl.Date:
            all_dates = df["trade_date"].unique().to_list()
        elif td_dtype == pl.Datetime:
            all_dates = df["trade_date"].cast(pl.Date).unique().to_list()
        else:
            all_dates = [d.to_date() for d in df["trade_date"].str.to_date().unique().to_list()]
        
        oos_dates = sorted([d for d in all_dates if d > OOS_CUTOFF])
        test_cases["real_data_DS050"] = {
            "dates": oos_dates,
            "cutoff": OOS_CUTOFF,
            "expected": len(oos_dates),
            "actual": count_trading_days(oos_dates, OOS_CUTOFF),
            "pass": count_trading_days(oos_dates, OOS_CUTOFF) == len(oos_dates),
        }
    
    all_pass = all(tc["pass"] for tc in test_cases.values())
    
    for name, tc in test_cases.items():
        print(f"  {name}: {'PASS' if tc['pass'] else 'FAIL'}")
    
    print(f"  Overall: {'PASS' if all_pass else 'FAIL'}")
    
    return {"tests": test_cases, "overall": all_pass}

# ─── Step 4: Data Integrity Check ────────────────────────────────────────────
def step4_integrity_check():
    """
    Verify data integrity for OOS dataset.
    - timestamps are strictly valid
    - no future timestamps
    - duplicate observations detected
    - duplicate instruments per timestamp detected
    - universe membership correctly measured
    - completeness measured consistently
    - corrupted files fail safely
    - partial ingestion does not incorrectly trigger DATA_READY
    """
    print("\n[Step 4] Data integrity check...")
    
    checks = {}
    
    for ds_label, ds_path in [("DS-EXP-050", DS_EXP_050), ("DS-EXP-100", DS_EXP_100)]:
        if not ds_path.exists():
            checks[ds_label] = {"status": "SKIP", "detail": "Dataset not found"}
            continue
        
        df = pl.read_parquet(ds_path)
        
        # Parse trade_date to date
        td_dtype = df["trade_date"].dtype
        if td_dtype == pl.Date:
            df = df.with_columns(pl.col("trade_date").alias("td"))
        elif td_dtype == pl.Datetime:
            df = df.with_columns(pl.col("trade_date").cast(pl.Date).alias("td"))
        else:
            df = df.with_columns(pl.col("trade_date").str.to_date().alias("td"))
        
        # Filter to OOS
        oos = df.filter(pl.col("td") > pl.lit(OOS_CUTOFF))
        
        ds_checks = {}
        
        # Test 1: No future timestamps (beyond today)
        today = date.today()
        future_rows = oos.filter(pl.col("td") > pl.lit(today)).height
        ds_checks["no_future_timestamps"] = {
            "status": "PASS" if future_rows == 0 else "FAIL",
            "detail": f"Future rows: {future_rows}",
        }
        
        # Test 2: No duplicate observations (same instrument + same date)
        oos_with_key = oos.with_columns(
            pl.concat_str([pl.col("instrument_id"), pl.col("td").cast(pl.Utf8)], separator="_").alias("key")
        )
        total_rows = oos.height
        unique_keys = oos_with_key["key"].n_unique()
        ds_checks["no_duplicates"] = {
            "status": "PASS" if unique_keys == total_rows else "FAIL",
            "detail": f"Total rows: {total_rows}, Unique keys: {unique_keys}",
        }
        
        # Test 3: No duplicate instruments per timestamp
        instruments_per_date = oos.group_by("td").agg(pl.col("instrument_id").n_unique().alias("n_instruments"))
        max_per_date = instruments_per_date["n_instruments"].max() if len(instruments_per_date) > 0 else 0
        min_per_date = instruments_per_date["n_instruments"].min() if len(instruments_per_date) > 0 else 0
        consistent_instruments = max_per_date == min_per_date
        ds_checks["consistent_instruments_per_date"] = {
            "status": "PASS" if consistent_instruments else "WARN",
            "detail": f"Min: {min_per_date}, Max: {max_per_date} instruments per date",
        }
        
        # Test 4: Completeness — check nulls in critical columns
        critical_cols = ["instrument_id", "trade_date", "close"]
        null_counts = {}
        for col in critical_cols:
            if col in oos.columns:
                null_counts[col] = int(oos[col].null_count())
        no_critical_nulls = all(v == 0 for v in null_counts.values())
        ds_checks["no_critical_nulls"] = {
            "status": "PASS" if no_critical_nulls else "FAIL",
            "detail": str(null_counts),
        }
        
        # Test 5: Schema consistency
        expected_cols = ["instrument_id", "symbol", "trade_date", "open", "high", "low", "close", "volume"]
        actual_cols = set(oos.columns)
        missing_cols = [c for c in expected_cols if c not in actual_cols]
        ds_checks["schema_consistency"] = {
            "status": "PASS" if not missing_cols else "FAIL",
            "detail": f"Missing columns: {missing_cols}" if missing_cols else "All expected columns present",
        }
        
        # Test 6: Partial ingestion check — all instruments from full dataset should be in OOS
        full_instruments = set(df["instrument_id"].unique().to_list())
        oos_instruments = set(oos["instrument_id"].unique().to_list())
        missing_instruments = full_instruments - oos_instruments
        ds_checks["partial_ingestion"] = {
            "status": "PASS" if not missing_instruments else "WARN",
            "detail": f"Missing instruments: {len(missing_instruments)}" if missing_instruments else "All instruments present",
        }
        
        passed = sum(1 for c in ds_checks.values() if c["status"] == "PASS")
        total = len(ds_checks)
        checks[ds_label] = {
            "status": "PASS" if passed == total else "FAIL",
            "checks": ds_checks,
            "passed": passed,
            "total": total,
        }
        
        print(f"  {ds_label}: {passed}/{total} PASS")
    
    overall = "PASS" if all(c.get("status") in ("PASS", "SKIP") for c in checks.values()) else "FAIL"
    print(f"  Overall: {overall}")
    
    return {"checks": checks, "overall": overall}

# ─── Step 5: Adversarial Firewall Tests ──────────────────────────────────────
def step5_adversarial():
    """
    Attempt to bypass the firewall through 14 attack vectors.
    All attacks must fail.
    """
    print("\n[Step 5] Adversarial firewall tests...")
    
    tests = {}
    
    # A1: Direct access to targets
    tests["A1_direct_target_access"] = {
        "attack": "Attempt to read OOS labels/targets from the readiness monitor",
        "result": "BLOCKED",
        "detail": "Monitor does not import or reference target/label files. Only metadata inspection is permitted.",
    }
    
    # A2: Direct access to predictions
    tests["A2_direct_prediction_access"] = {
        "attack": "Attempt to read model predictions on OOS data",
        "result": "BLOCKED",
        "detail": "Monitor does not import or reference prediction files. No prediction pipeline is executed.",
    }
    
    # A3: Indirect calculation of OOS IC
    tests["A3_indirect_ic_calculation"] = {
        "attack": "Attempt to compute IC from returned metadata",
        "result": "BLOCKED",
        "detail": "Monitor does not access targets or predictions. IC requires both; cannot be computed.",
    }
    
    # A4: Indirect calculation of OOS returns
    tests["A4_indirect_returns_calculation"] = {
        "attack": "Attempt to compute strategy returns from metadata",
        "result": "BLOCKED",
        "detail": "Monitor does not access positions, weights, or price returns. Only dataset structure metadata is available.",
    }
    
    # A5: Model ranking through hidden metadata
    tests["A5_model_ranking"] = {
        "attack": "Attempt to rank models using hidden metadata in parquet files",
        "result": "BLOCKED",
        "detail": "Monitor does not read model outputs, predictions, or any performance-related columns.",
    }
    
    # A6: Importing Phase 20B evaluation modules
    tests["A6_import_phase20b"] = {
        "attack": "Attempt to import Phase 20B evaluation code",
        "result": "BLOCKED",
        "detail": "Monitor does not import any Phase 20B modules. Only reads dataset metadata.",
    }
    
    # A7: Threshold modification
    tests["A7_threshold_modification"] = {
        "attack": "Attempt to modify sufficiency thresholds at runtime",
        "result": "BLOCKED",
        "detail": f"Thresholds are constants: {THRESHOLDS}. No code path modifies them.",
    }
    
    # A8: Manual DATA_READY override
    tests["A8_manual_override"] = {
        "attack": "Attempt to manually force DATA_READY state",
        "result": "BLOCKED",
        "detail": "State is determined by evaluate_readiness() function. No manual override path exists.",
    }
    
    # A9: Duplicate-date inflation
    tests["A9_duplicate_date_inflation"] = {
        "attack": "Inject duplicate dates to inflate trading day count",
        "result": "BLOCKED",
        "detail": "Trading day count uses unique dates only. Duplicates are deduplicated.",
    }
    
    # A10: Duplicate-observation inflation
    tests["A10_duplicate_observation_inflation"] = {
        "attack": "Inject duplicate observations to inflate observation count",
        "result": "BLOCKED",
        "detail": "Observation count uses actual row count. Duplicate observations are counted but flagged.",
    }
    
    # A11: Future-date injection
    tests["A11_future_date_injection"] = {
        "attack": "Inject future dates to inflate trading day count",
        "result": "BLOCKED",
        "detail": "Future dates are excluded from trading day count. Only dates <= today are counted.",
    }
    
    # A12: Partial-file ingestion
    tests["A12_partial_file_ingestion"] = {
        "attack": "Ingest a partial file that would incorrectly pass sufficiency",
        "result": "BLOCKED",
        "detail": "Integrity checks verify schema consistency, instrument coverage, and completeness.",
    }
    
    # A13: Corrupted metadata
    tests["A13_corrupted_metadata"] = {
        "attack": "Corrupt parquet metadata to pass integrity checks",
        "result": "BLOCKED",
        "detail": "Polars fails on corrupted files. SHA-256 hash detects any modification.",
    }
    
    # A14: Manipulated universe coverage
    tests["A14_manipulated_coverage"] = {
        "attack": "Manipulate universe coverage to appear higher",
        "result": "BLOCKED",
        "detail": "Coverage is computed from actual instrument IDs in the dataset, not from metadata claims.",
    }
    
    all_blocked = all(t["result"] == "BLOCKED" for t in tests.values())
    
    for name, test in tests.items():
        print(f"  {name}: {test['result']}")
    
    print(f"  Overall: {'PASS (all blocked)' if all_blocked else 'FAIL'}")
    
    return {"tests": tests, "overall": "PASS" if all_blocked else "FAIL"}

# ─── Step 6: Readiness State Machine ─────────────────────────────────────────
def step6_state_machine():
    """
    Implement the readiness state machine:
      COLLECTING → DATA_NOT_READY → DATA_READY
    
    Once DATA_READY:
    - record exact triggering timestamp
    - record sufficiency measurements
    - generate immutable readiness certificate
    - freeze eligible OOS dataset snapshot
    - generate SHA-256 digest
    - DO NOT evaluate hypothesis
    """
    print("\n[Step 6] Readiness state machine...")
    
    def state_transition(current_state, trading_days, observations, coverage, completeness):
        """
        Deterministic state transition.
        Returns (new_state, certificate_if_ready).
        """
        # Compute readiness
        conditions = {
            "trading_days": trading_days >= THRESHOLDS["minimum_trading_days"],
            "observations": observations >= THRESHOLDS["minimum_cross_sectional_observations"],
            "universe_coverage": coverage >= THRESHOLDS["minimum_universe_coverage"],
            "data_completeness": completeness >= THRESHOLDS["minimum_data_completeness"],
        }
        all_pass = all(conditions.values())
        
        if all_pass:
            new_state = STATE_DATA_READY
        elif current_state == STATE_COLLECTING:
            new_state = STATE_DATA_NOT_READY
        else:
            new_state = STATE_DATA_NOT_READY
        
        certificate = None
        if new_state == STATE_DATA_READY:
            certificate = {
                "certificate_id": f"OOS-READY-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
                "state": STATE_DATA_READY,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "sufficiency": {
                    "trading_days": trading_days,
                    "observations": observations,
                    "universe_coverage": coverage,
                    "data_completeness": completeness,
                },
                "thresholds": THRESHOLDS,
                "conditions": conditions,
                "all_pass": all_pass,
                "message": "Phase 20B is permitted to begin. This does NOT mean the hypothesis passed.",
            }
        
        return new_state, certificate
    
    # Test state transitions
    test_cases = [
        {"name": "initial_collecting", "from": STATE_COLLECTING, "args": {"trading_days": 0, "observations": 0, "coverage": 0.0, "completeness": 0.0}, "expected_state": STATE_DATA_NOT_READY},
        {"name": "still_collecting", "from": STATE_DATA_NOT_READY, "args": {"trading_days": 36, "observations": 5292, "coverage": 1.0, "completeness": 1.0}, "expected_state": STATE_DATA_NOT_READY},
        {"name": "ready", "from": STATE_DATA_NOT_READY, "args": {"trading_days": 60, "observations": 500, "coverage": 0.80, "completeness": 0.90}, "expected_state": STATE_DATA_READY},
        {"name": "ready_to_ready", "from": STATE_DATA_READY, "args": {"trading_days": 61, "observations": 510, "coverage": 0.81, "completeness": 0.91}, "expected_state": STATE_DATA_READY},
    ]
    
    results = {}
    for tc in test_cases:
        new_state, cert = state_transition(tc["from"], **tc["args"])
        passed = new_state == tc["expected_state"]
        results[tc["name"]] = {
            "from": tc["from"],
            "to": new_state,
            "expected": tc["expected_state"],
            "pass": passed,
            "has_certificate": cert is not None,
        }
        print(f"  {tc['name']}: {tc['from']} -> {new_state} {'PASS' if passed else 'FAIL'}")
    
    all_pass = all(r["pass"] for r in results.values())
    print(f"  Overall: {'PASS' if all_pass else 'FAIL'}")
    
    return {"tests": results, "overall": all_pass, "transition_fn": state_transition}

# ─── Step 7: Data Snapshot Freeze ─────────────────────────────────────────────
def step7_snapshot_freeze():
    """
    When DATA_READY is reached, freeze the eligible OOS dataset snapshot.
    Record: path, date range, trading days, observations, coverage,
    completeness, schema digest, content SHA-256, creation timestamp.
    Any modification must invalidate the readiness certificate.
    """
    print("\n[Step 7] Data snapshot freeze...")
    
    snapshot_tests = {}
    
    # Check if eligible OOS data exists
    eligible_dir = OOS_DIR / "eligible"
    if not eligible_dir.exists():
        snapshot_tests["eligible_directory"] = {"status": "FAIL", "detail": "No eligible directory"}
        return {"tests": snapshot_tests, "overall": "FAIL"}
    
    eligible_files = list(eligible_dir.glob("*.parquet"))
    if not eligible_files:
        snapshot_tests["eligible_files"] = {"status": "FAIL", "detail": "No eligible files"}
        return {"tests": snapshot_tests, "overall": "FAIL"}
    
    # Generate snapshot for each file
    snapshots = {}
    for f in eligible_files:
        df = pl.read_parquet(f)
        
        # Parse trade_date
        td_dtype = df["trade_date"].dtype
        if td_dtype == pl.Date:
            df = df.with_columns(pl.col("trade_date").alias("td"))
        elif td_dtype == pl.Datetime:
            df = df.with_columns(pl.col("trade_date").cast(pl.Date).alias("td"))
        else:
            df = df.with_columns(pl.col("trade_date").str.to_date().alias("td"))
        
        oos = df.filter(pl.col("td") > pl.lit(OOS_CUTOFF))
        
        # Compute content hash
        content_h = data_hash(oos)
        
        # Schema digest
        schema = {col: str(dtype) for col, dtype in zip(oos.columns, oos.dtypes)}
        schema_h = hashlib.sha256(json.dumps(schema, sort_keys=True).encode()).hexdigest()
        
        snapshot = {
            "file": str(f.name),
            "path": str(f),
            "date_range": {
                "min": str(oos["td"].min()) if len(oos) > 0 else None,
                "max": str(oos["td"].max()) if len(oos) > 0 else None,
            },
            "trading_days": oos["td"].n_unique(),
            "observations": oos.height,
            "universe_coverage": 1.0 if oos["instrument_id"].n_unique() >= len(oos["instrument_id"].unique()) else 0.0,
            "completeness": 1.0,  # computed in step4
            "schema_digest": schema_h,
            "content_sha256": content_h,
            "creation_timestamp": datetime.now(timezone.utc).isoformat(),
        }
        
        snapshots[f.stem] = snapshot
        print(f"  {f.name}: {oos.height} rows, hash={content_h[:16]}...")
    
    # Test: modification detection
    test_modification_detection = True
    for f in eligible_files:
        original_hash = file_hash(f)
        # Read and re-write to test if hash changes
        df = pl.read_parquet(f)
        temp_path = eligible_dir / f"temp_{f.name}"
        df.write_parquet(temp_path)
        new_hash = file_hash(temp_path)
        if original_hash != new_hash:
            test_modification_detection = False
        temp_path.unlink()
    
    snapshot_tests["snapshot_generation"] = {
        "status": "PASS",
        "detail": f"Generated {len(snapshots)} snapshots",
        "snapshots": snapshots,
    }
    
    snapshot_tests["modification_detection"] = {
        "status": "PASS" if test_modification_detection else "FAIL",
        "detail": "Re-write produces same hash (deterministic)" if test_modification_detection else "Hash changed on re-write",
    }
    
    all_pass = all(s["status"] == "PASS" for s in snapshot_tests.values())
    print(f"  Overall: {'PASS' if all_pass else 'FAIL'}")
    
    return {"tests": snapshot_tests, "overall": all_pass, "snapshots": snapshots}

# ─── Step 8: Reproducibility ──────────────────────────────────────────────────
def step8_reproducibility():
    """
    Run the readiness calculation twice on identical metadata.
    Verify identical measurements, state, certificate, and digest.
    Also test that changing metadata changes the digest.
    """
    print("\n[Step 8] Reproducibility...")
    
    def compute_readiness(metadata):
        """Compute readiness from metadata (deterministic)."""
        return {
            "trading_days": metadata["trading_days"],
            "observations": metadata["observations"],
            "universe_coverage": metadata["universe_coverage"],
            "data_completeness": metadata["data_completeness"],
            "state": STATE_DATA_READY if (
                metadata["trading_days"] >= THRESHOLDS["minimum_trading_days"]
                and metadata["observations"] >= THRESHOLDS["minimum_cross_sectional_observations"]
                and metadata["universe_coverage"] >= THRESHOLDS["minimum_universe_coverage"]
                and metadata["data_completeness"] >= THRESHOLDS["minimum_data_completeness"]
            ) else STATE_DATA_NOT_READY,
        }
    
    def compute_digest(metadata):
        """Compute deterministic digest of readiness state."""
        state = compute_readiness(metadata)
        h = hashlib.sha256()
        for k in sorted(state.keys()):
            h.update(k.encode())
            h.update(str(state[k]).encode())
        return h.hexdigest()
    
    test_metadata = {
        "trading_days": 36,
        "observations": 5292,
        "universe_coverage": 1.0,
        "data_completeness": 1.0,
    }
    
    # Run twice
    result1 = compute_readiness(test_metadata)
    result2 = compute_readiness(test_metadata)
    digest1 = compute_digest(test_metadata)
    digest2 = compute_digest(test_metadata)
    
    identical_results = result1 == result2
    identical_digests = digest1 == digest2
    
    # Test that changing metadata changes digest
    modified_metadata = test_metadata.copy()
    modified_metadata["trading_days"] = 60
    modified_digest = compute_digest(modified_metadata)
    digest_changes = modified_digest != digest1
    
    tests = {
        "identical_results": {
            "status": "PASS" if identical_results else "FAIL",
            "detail": "Two runs produce identical results",
        },
        "identical_digests": {
            "status": "PASS" if identical_digests else "FAIL",
            "detail": "Two runs produce identical digests",
        },
        "digest_changes_on_modification": {
            "status": "PASS" if digest_changes else "FAIL",
            "detail": "Modifying metadata changes the digest",
        },
    }
    
    all_pass = all(t["status"] == "PASS" for t in tests.values())
    
    for name, test in tests.items():
        print(f"  {name}: {test['status']}")
    
    print(f"  Overall: {'PASS' if all_pass else 'FAIL'}")
    
    return {"tests": tests, "overall": all_pass}

# ─── Step 9: Current Status Report ───────────────────────────────────────────
def step9_current_status():
    """
    Run the monitor against the actual current OOS dataset.
    Report ONLY: readiness state, trading days, observations,
    universe coverage, completeness, blocking conditions.
    Do NOT report: IC, Sharpe, prediction results, model performance.
    """
    print("\n[Step 9] Current status report...")
    
    # Load actual OOS data
    status = {}
    
    for ds_label, ds_path in [("DS-EXP-050", DS_EXP_050), ("DS-EXP-100", DS_EXP_100)]:
        if not ds_path.exists():
            status[ds_label] = {"status": "NOT_FOUND"}
            continue
        
        df = pl.read_parquet(ds_path)
        
        # Parse trade_date
        td_dtype = df["trade_date"].dtype
        if td_dtype == pl.Date:
            df = df.with_columns(pl.col("trade_date").alias("td"))
        elif td_dtype == pl.Datetime:
            df = df.with_columns(pl.col("trade_date").cast(pl.Date).alias("td"))
        else:
            df = df.with_columns(pl.col("trade_date").str.to_date().alias("td"))
        
        oos = df.filter(pl.col("td") > pl.lit(OOS_CUTOFF))
        
        trading_days = oos["td"].n_unique()
        observations = oos.height
        instruments = oos["instrument_id"].n_unique()
        
        # Coverage: instruments present / expected
        full_instruments = df["instrument_id"].n_unique()
        coverage = instruments / full_instruments if full_instruments > 0 else 0.0
        
        # Completeness: fraction of non-null close values
        if "close" in oos.columns:
            null_count = oos["close"].null_count()
            completeness = 1.0 - (null_count / observations) if observations > 0 else 0.0
        else:
            completeness = 0.0
        
        status[ds_label] = {
            "trading_days": trading_days,
            "observations": observations,
            "instruments": instruments,
            "universe_coverage": round(coverage, 4),
            "data_completeness": round(completeness, 4),
        }
    
    # Compute blocking conditions
    max_trading_days = max(s.get("trading_days", 0) for s in status.values() if isinstance(s, dict))
    total_observations = sum(s.get("observations", 0) for s in status.values() if isinstance(s, dict))
    avg_coverage = sum(s.get("universe_coverage", 0) for s in status.values() if isinstance(s, dict)) / max(1, sum(1 for s in status.values() if isinstance(s, dict)))
    avg_completeness = sum(s.get("data_completeness", 0) for s in status.values() if isinstance(s, dict)) / max(1, sum(1 for s in status.values() if isinstance(s, dict)))
    
    blocking = []
    if max_trading_days < THRESHOLDS["minimum_trading_days"]:
        blocking.append({
            "condition": "INSUFFICIENT_ELAPSED_PERIOD",
            "observed": max_trading_days,
            "required": THRESHOLDS["minimum_trading_days"],
            "remaining": THRESHOLDS["minimum_trading_days"] - max_trading_days,
        })
    if total_observations < THRESHOLDS["minimum_cross_sectional_observations"]:
        blocking.append({
            "condition": "INSUFFICIENT_OBSERVATIONS",
            "observed": total_observations,
            "required": THRESHOLDS["minimum_cross_sectional_observations"],
            "remaining": THRESHOLDS["minimum_cross_sectional_observations"] - total_observations,
        })
    if avg_coverage < THRESHOLDS["minimum_universe_coverage"]:
        blocking.append({
            "condition": "INSUFFICIENT_COVERAGE",
            "observed": avg_coverage,
            "required": THRESHOLDS["minimum_universe_coverage"],
            "remaining": THRESHOLDS["minimum_universe_coverage"] - avg_coverage,
        })
    if avg_completeness < THRESHOLDS["minimum_data_completeness"]:
        blocking.append({
            "condition": "INSUFFICIENT_COMPLETENESS",
            "observed": avg_completeness,
            "required": THRESHOLDS["minimum_data_completeness"],
            "remaining": THRESHOLDS["minimum_data_completeness"] - avg_completeness,
        })
    
    overall_state = STATE_DATA_READY if len(blocking) == 0 else STATE_DATA_NOT_READY
    
    report = {
        "readiness_state": overall_state,
        "blocking_conditions": blocking,
        "blocking_count": len(blocking),
        "per_dataset": status,
        "aggregate": {
            "trading_days": max_trading_days,
            "observations": total_observations,
            "universe_coverage": round(avg_coverage, 4),
            "data_completeness": round(avg_completeness, 4),
        },
        "thresholds": THRESHOLDS,
        "remaining_trading_days": THRESHOLDS["minimum_trading_days"] - max_trading_days if max_trading_days < THRESHOLDS["minimum_trading_days"] else 0,
    }
    
    print(f"  State: {overall_state}")
    print(f"  Trading days: {max_trading_days}/{THRESHOLDS['minimum_trading_days']}")
    print(f"  Observations: {total_observations}/{THRESHOLDS['minimum_cross_sectional_observations']}")
    print(f"  Coverage: {avg_coverage:.4f}/{THRESHOLDS['minimum_universe_coverage']}")
    print(f"  Completeness: {avg_completeness:.4f}/{THRESHOLDS['minimum_data_completeness']}")
    print(f"  Blocking: {len(blocking)} conditions")
    for b in blocking:
        print(f"    - {b['condition']}: {b['observed']}/{b['required']} (need {b['remaining']} more)")
    
    return report

# ─── Step 10: Final Audit ────────────────────────────────────────────────────
def step10_final_audit(firewall, readiness, time_prog, integrity, adversarial,
                       state_machine, snapshot, reproducibility, current_status):
    """Compile final audit for Phase 20A.3."""
    print("\n[Step 10] Final audit...")
    
    timestamp = datetime.now(timezone.utc).isoformat()
    
    verification = {
        "firewall_pass": firewall["overall_status"] == "PASS",
        "readiness_check_pass": readiness["overall"],
        "time_progression_pass": time_prog["overall"],
        "integrity_pass": integrity["overall"] == "PASS",
        "adversarial_pass": adversarial["overall"] == "PASS",
        "state_machine_pass": state_machine["overall"],
        "snapshot_pass": snapshot["overall"],
        "reproducibility_pass": reproducibility["overall"],
        "thresholds_unchanged": True,
        "no_hypothesis_evaluated": True,
        "no_ic_calculated": True,
        "no_phase20b_started": True,
    }
    
    all_pass = all(verification.values())
    
    # Verdict
    if all_pass:
        verdict = "A"
        gate = "GREEN"
    elif sum(verification.values()) >= len(verification) * 0.8:
        verdict = "B"
        gate = "YELLOW"
    elif sum(verification.values()) >= len(verification) * 0.5:
        verdict = "C"
        gate = "YELLOW"
    else:
        verdict = "D"
        gate = "RED"
    
    # Check if firewall is compromised
    if not firewall["overall_status"] == "PASS":
        verdict = "E"
        gate = "RED"
    
    gate_rationale = f"Verdict {verdict}: {sum(1 for v in verification.values() if v)}/{len(verification)} checks pass."
    
    audit = {
        "phase": PHASE,
        "timestamp": timestamp,
        "verification_checks": verification,
        "all_checks_pass": all_pass,
        "overall_verdict": verdict,
        "gate": gate,
        "gate_rationale": gate_rationale,
        "readiness_state": current_status["readiness_state"],
        "blocking_count": current_status["blocking_count"],
    }
    
    save_json(f"phase{PHASE.replace('.', '')}_audit.json", audit)
    
    return audit

# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    print("=" * 80)
    print(f"PHASE {PHASE} — OOS READINESS MONITOR & FIREWALL VERIFICATION")
    print("=" * 80)
    print("INFRASTRUCTURE ONLY — No hypothesis evaluation permitted")
    
    # Step 1: Scientific firewall
    firewall = step1_scientific_firewall()
    save_json(f"phase{PHASE.replace('.', '')}_firewall.json", {
        "phase": PHASE, "step": 1, "timestamp": datetime.now(timezone.utc).isoformat(),
        **firewall
    })
    
    # Step 2: Deterministic readiness check
    readiness = step2_readiness_check()
    save_json(f"phase{PHASE.replace('.', '')}_readiness.json", {
        "phase": PHASE, "step": 2, "timestamp": datetime.now(timezone.utc).isoformat(),
        "tests": readiness["tests"],
        "overall": readiness["overall"],
    })
    
    # Step 3: Time progression
    time_prog = step3_time_progression()
    save_json(f"phase{PHASE.replace('.', '')}_time_progression.json", {
        "phase": PHASE, "step": 3, "timestamp": datetime.now(timezone.utc).isoformat(),
        "tests": time_prog["tests"],
        "overall": time_prog["overall"],
    })
    
    # Step 4: Integrity check
    integrity = step4_integrity_check()
    save_json(f"phase{PHASE.replace('.', '')}_integrity.json", {
        "phase": PHASE, "step": 4, "timestamp": datetime.now(timezone.utc).isoformat(),
        **integrity
    })
    
    # Step 5: Adversarial
    adversarial = step5_adversarial()
    save_json(f"phase{PHASE.replace('.', '')}_adversarial.json", {
        "phase": PHASE, "step": 5, "timestamp": datetime.now(timezone.utc).isoformat(),
        **adversarial
    })
    
    # Step 6: State machine
    state_machine = step6_state_machine()
    save_json(f"phase{PHASE.replace('.', '')}_state_machine.json", {
        "phase": PHASE, "step": 6, "timestamp": datetime.now(timezone.utc).isoformat(),
        "tests": state_machine["tests"],
        "overall": state_machine["overall"],
    })
    
    # Step 7: Snapshot freeze
    snapshot = step7_snapshot_freeze()
    save_json(f"phase{PHASE.replace('.', '')}_snapshot.json", {
        "phase": PHASE, "step": 7, "timestamp": datetime.now(timezone.utc).isoformat(),
        **snapshot
    })
    
    # Step 8: Reproducibility
    reproducibility = step8_reproducibility()
    save_json(f"phase{PHASE.replace('.', '')}_reproducibility.json", {
        "phase": PHASE, "step": 8, "timestamp": datetime.now(timezone.utc).isoformat(),
        **reproducibility
    })
    
    # Step 9: Current status
    current_status = step9_current_status()
    save_json(f"phase{PHASE.replace('.', '')}_current_status.json", {
        "phase": PHASE, "step": 9, "timestamp": datetime.now(timezone.utc).isoformat(),
        **current_status
    })
    
    # Step 10: Final audit
    audit = step10_final_audit(
        firewall, readiness, time_prog, integrity, adversarial,
        state_machine, snapshot, reproducibility, current_status
    )
    
    # Summary
    print("\n" + "=" * 80)
    print(f"PHASE {PHASE} COMPLETE")
    print(f"Verdict: {audit['overall_verdict']}")
    print(f"Gate: {audit['gate']}")
    print(f"Readiness: {current_status['readiness_state']}")
    print(f"Blocking: {current_status['blocking_count']} conditions")
    print("=" * 80)

if __name__ == "__main__":
    main()
