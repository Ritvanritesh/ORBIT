import json
import numpy as np

# Cross-phase comparison
comparison = {
    "phase": "12B",
    "report_type": "cross_phase_comparison",
    "created_at": "2026-08-22",
    "phases_compared": ["11/11.2", "12A", "12B"],
    "comparison": {
        "phase_11_11_2": {
            "verdict": "D - Null persists",
            "best_oos_ic": 0.0128,
            "mean_ic": 0.0119,
            "n_universes": 4,
            "universe_sizes": [20, 20, 50, 97],
            "statistical_significance": False,
            "multiple_testing_corrected": True,
            "conclusion": "Expanding from 20 to 97 symbols did not materially improve predictive evidence",
        },
        "phase_12a": {
            "verdict": "D - Market context modest, cross-sectional negative",
            "best_oos_ic": 0.0102,
            "mean_ic_baseline": 0.0082,
            "market_context_ic": 0.0102,
            "market_effect": +0.0019,
            "cross_sectional_effect": -0.0015,
            "statistical_significance": False,
            "conclusion": "Market context produced modest effect; cross-sectional context did not improve results",
        },
        "phase_12b": {
            "verdict": "E - Infrastructure/data limitations",
            "baseline_ic_050": 0.0109,
            "baseline_ic_100": 0.0055,
            "fundamental_tested": False,
            "fundamental_blocked": True,
            "blocking_reason": "PIT non-compliance: synthetic fundamental data",
            "conclusion": "Fundamental features could not be tested due to synthetic data with future filing dates",
        },
    },
    "cumulative_assessment": {
        "overall_trajectory": "Consistent null across all phases",
        "phase_11_baseline": "Null persists with 20-97 symbol universe",
        "phase_12a_addition": "Market context provides no material improvement",
        "phase_12b_addition": "Fundamental features blocked by data limitations",
        "combined_verdict": "D/E - Null persists for tested information; fundamental question unanswered",
        "information_expansion成效": {
            "market_context": "No material improvement",
            "sector_context": "Insufficient data to test",
            "cross_sectional": "Negative effect",
            "fundamental": "Blocked by PIT limitations",
        },
    },
    "best_observed_ic_across_phases": {
        "value": 0.0128,
        "phase": "11/11.2",
        "environment": "ENV-3 (50 symbols)",
        "interpretation": "All observed ICs consistent with noise after multiple-testing correction",
    },
}

with open("benchmarks/phase12b_comparison.json", "w") as f:
    json.dump(comparison, f, indent=2, default=str)

print("Cross-phase comparison saved")
print(json.dumps(comparison["cumulative_assessment"], indent=2))
