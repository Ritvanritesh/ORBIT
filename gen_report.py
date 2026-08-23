import json
import numpy as np

report = {
    "phase": "12B",
    "report_type": "final_report",
    "created_at": "2026-08-22",
    "verdict": "E - Infrastructure/data limitations prevent valid interpretation",
    "executive_summary": (
        "Phase 12B attempted to test whether point-in-time fundamental "
        "information provides predictive evidence beyond OHLCV features. "
        "All fundamental feature sets were blocked because synthetic "
        "fundamental data has filing dates (1998-2013) that predate the "
        "trade date start (1996), violating PIT compliance. Only baseline "
        "OHLCV experiments (FS-12B-A) were executed, yielding ICs "
        "consistent with noise."
    ),
}

# Load results
for env_id in ["ENV-12B-050", "ENV-12B-100"]:
    with open(f"benchmarks/phase12b_{env_id}_results.json") as f:
        data = json.load(f)
    ok = [r for r in data["results"] if "error" not in r and not r.get("blocked")]
    ics = [r["metrics"]["oos_ic"] for r in ok if r["metrics"].get("oos_ic") is not None]
    report[f"results_{env_id}"] = {
        "n_instruments": data["n_instruments"],
        "n_sessions": data["n_sessions"],
        "n_successful": data["n_successful"],
        "n_blocked": data["n_blocked"],
        "mean_ic": float(np.mean(ics)) if ics else None,
        "median_ic": float(np.median(ics)) if ics else None,
    }

report["experiment_inventory"] = {
    "registered": 96,
    "completed": 16,
    "blocked": 80,
    "failed": 0,
    "blocking_reason": "PIT non-compliance: synthetic fundamental data with future filing dates",
}

report["feature_sets"] = {
    "FS-12B-A": "EXECUTED - baseline OHLCV",
    "FS-12B-B": "BLOCKED - valuation",
    "FS-12B-C": "BLOCKED - profitability",
    "FS-12B-D": "BLOCKED - growth",
    "FS-12B-E": "BLOCKED - leverage",
    "FS-12B-F": "BLOCKED - all fundamentals",
}

report["cross_phase_comparison"] = {
    "Phase_11_11_2": "Verdict D - null persists",
    "Phase_12A": "Verdict D - market context modest",
    "Phase_12B": "Verdict E - data limitations",
    "combined": "D/E - null persists for tested information; fundamental question unanswered",
}

report["honest_limitations"] = [
    "All data is synthetic - no real SEC EDGAR filings used",
    "Fundamental features could not be tested due to PIT non-compliance",
    "80 of 96 registered experiments blocked",
    "LAB-005 (excess return) uses same underlying as LAB-004",
    "The central research question cannot be answered with current data",
    "Baseline ICs are low and consistent with noise",
]

report["recommended_next_step"] = (
    "Obtain real SEC EDGAR data with valid filing dates. "
    "The infrastructure is ready but the synthetic data prevents "
    "PIT-safe fundamental feature computation."
)

report["files_created"] = [
    "scripts/phase12b_run.py",
    "src/orbit/ml/phase12b_plan.py",
    "src/orbit/ml/phase12b_fundamentals.py",
    "src/orbit/ml/phase12b_features.py",
    "src/orbit/ml/phase12b_validation.py",
    "benchmarks/phase12b_plan.json",
    "benchmarks/phase12b_ENV-12B-050_results.json",
    "benchmarks/phase12b_ENV-12B-100_results.json",
    "benchmarks/phase12b_data_provenance.json",
    "benchmarks/phase12b_identity_mapping.json",
    "benchmarks/phase12b_inference_results.json",
    "benchmarks/phase12b_comparison.json",
    "benchmarks/phase12b_audit.json",
    "benchmarks/phase12b_report.json",
]

with open("benchmarks/phase12b_report.json", "w") as f:
    json.dump(report, f, indent=2, default=str)

print("Final Phase 12B report saved")
print(f"Verdict: {report['verdict']}")
