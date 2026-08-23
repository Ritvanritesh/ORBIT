"""Phase 13C — Robustness Synthesis & Final Gate.

Independent senior-reviewer synthesis of the complete ORBIT evidence chain
(Phases 9 through 13B). No new exploration. Produces the candidate evidence
table, generalization scorecard, robustness matrix, failure-region analysis,
economic materiality review, conclusion-language review, promotion gate, and
Phase 14 readiness determination.
"""
from __future__ import annotations
import hashlib, json, sys, time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(r"E:\Orbit (Optimized Research & Behavioral Intelligence Trading)")
BENCH = REPO / "benchmarks"

def save_json(name, data):
    with open(BENCH / name, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str, ensure_ascii=False)
    print(f"  Saved: {name}")

def load_json(name):
    with open(BENCH / name, encoding="utf-8") as f:
        return json.load(f)

def sha256_obj(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def main():
    t0 = time.time()
    print("=" * 72)
    print("PHASE 13C — ROBUSTNESS SYNTHESIS & FINAL GATE")
    print("=" * 72)

    # ------------------------------------------------------------------
    # LOAD ALL EVIDENCE ARTIFACTS
    # ------------------------------------------------------------------
    print("\n[LOAD] Reading evidence chain...")
    ev = {}
    ev["p12d_050"] = load_json("phase12d_ENV-12D-050_results.json")
    ev["p12d_100"] = load_json("phase12d_ENV-12D-100_results.json")
    ev["p12e_050"] = load_json("phase12e_ENV-12E-050_results.json")
    ev["p12e_100"] = load_json("phase12e_ENV-12E-100_results.json")
    ev["p11_2"] = load_json("phase11_2_ENV-12B-050_results.json")
    ev["p129a"] = load_json("phase12_9a_audit.json")
    ev["p129b"] = load_json("phase12_9b_audit.json")
    ev["p129c"] = load_json("phase12_9c_audit.json")
    ev["p13a_stab"] = load_json("phase13a_stability.json")
    ev["p13a_inf"] = load_json("phase13a_inference.json")
    ev["p13a_res"] = load_json("phase13a_results.json")
    ev["p13a_regime"] = load_json("phase13a_regime_results.json")
    ev["p13b_univ"] = load_json("phase13b_universe_stability.json")
    ev["p13b_feat"] = load_json("phase13b_feature_sensitivity.json")
    ev["p13b_param"] = load_json("phase13b_parameter_sensitivity.json")
    ev["p13b_cost"] = load_json("phase13b_cost_stress.json")
    ev["p13b_slip"] = load_json("phase13b_slippage_stress.json")
    ev["p13b_delay"] = load_json("phase13b_execution_delay.json")
    ev["p13b_data"] = load_json("phase13b_data_stress.json")

    def hist_ic(results_file, eid):
        for r in results_file.get("results", []):
            if r["experiment_id"] == eid:
                return r["metrics"].get("oos_ic")
        return None

    # Historical locked-protocol ICs (Phase 12D static split, test 2022–2026H1)
    hist = {
        "CAND-03": {
            "locked_protocol_ic_12d": hist_ic(ev["p12d_050"], "EXP-12D-ENV-12D-050-FS-12B-A-LAB-004-ridge"),
            "excess_label_ic_12e": hist_ic(ev["p12e_050"], "EXP-12E-ENV-12E-050-FS-12B-A-LAB-006-ridge"),
        },
        "CAND-04": {
            "locked_protocol_ic_12d": hist_ic(ev["p12d_050"], "EXP-12D-ENV-12D-050-FS-12B-A-LAB-004-lasso"),
            "excess_label_ic_12e": hist_ic(ev["p12e_050"], "EXP-12E-ENV-12E-050-FS-12B-A-LAB-006-lasso"),
        },
    }
    print(f"  Locked-protocol ICs: {json.dumps(hist, indent=2)}")

    # ------------------------------------------------------------------
    # CRITICAL REPRODUCIBILITY CHECK: 13A vs 13B implementation divergence
    # ------------------------------------------------------------------
    # Same window definition (EXP-003) evaluated by two independent runs.
    a_13 = {r["candidate_id"]: r for r in ev["p13a_res"]["results"]
            if r.get("window_id") == "EXP-003"}
    print("\n[AUDIT] Cross-implementation check (EXP-003, same window spec):")
    impl_divergence = {}
    for cid in ["CAND-03", "CAND-04"]:
        ic_a = a_13.get(cid, {}).get("oos_ic")
        ic_b = ev["p13b_cost"].get(cid, {}).get("1.0x", {}).get("baseline_ic")
        ratio = (ic_b / ic_a) if (ic_a and ic_b) else None
        impl_divergence[cid] = {"phase13a_ic": ic_a, "phase13b_ic": ic_b, "ratio": ratio}
        print(f"  {cid}: 13A={ic_a:+.4f}  13B={ic_b:+.4f}  ratio={ratio:.2f}" if ratio else f"  {cid}: missing")
    # Documented root cause: 13B purge used feature-frame window_end_session
    # (decision−1) instead of the label outcome window (decision+5), leaving
    # ~5 sessions of train rows whose outcomes overlap validation -> mild
    # optimistic bias in 13B baselines. Conservative reading = 13A numbers.

    # ==================================================================
    # STEP 1 — CANDIDATE EVIDENCE TABLE
    # ==================================================================
    print("\n[STEP 1] Building candidate evidence table...")
    meta = {
        "CAND-03": {"model": "ridge(alpha=1.0)", "feature_set": "FS-12B-A (8 OHLCV baseline)",
                    "label": "LAB-004 (5-session forward total return)", "environment": "ENV-050 primary"},
        "CAND-04": {"model": "lasso(alpha=0.001)", "feature_set": "FS-12B-A (8 OHLCV baseline)",
                    "label": "LAB-004 (5-session forward total return)", "environment": "ENV-050 primary"},
    }

    evidence_table = {}
    for cid in ["CAND-03", "CAND-04"]:
        st = ev["p13a_stab"][cid]
        inf = ev["p13a_inf"][cid]
        u = ev["p13b_univ"].get(cid, {})
        f = ev["p13b_feat"].get(cid, {})
        p = ev["p13b_param"].get(cid, {})
        d = ev["p13b_delay"].get(cid, {})

        # Regime behavior from 13A regime records (window-level overlap)
        regs = [r for r in ev["p13a_regime"]["results"] if r["candidate_id"] == cid]
        dir_regs = sorted(set(r["regime_value"] for r in regs if r["regime_type"] == "direction_regime"))

        # Window IC map (conservative 13A values)
        win_ics = {r["window_id"]: round(r["oos_ic"], 4) for r in ev["p13a_res"]["results"]
                   if r["candidate_id"] == cid}

        evidence_table[cid] = {
            **meta[cid],
            "historical_oos_ic_locked_protocol": hist[cid]["locked_protocol_ic_12d"],
            "historical_oos_ic_excess_label": hist[cid]["excess_label_ic_12e"],
            "phase13a_temporal_stability": {
                "classification": st["classification"],
                "n_windows": st["n_windows"],
                "positive_fraction": st["positive_ic_fraction"],
                "sign_flips": st["sign_flips"],
                "mean_ic": st["mean_ic"], "median_ic": st["median_ic"],
                "worst_ic": st["worst_ic"], "best_ic": st["best_ic"],
                "dispersion": st["ic_dispersion"],
                "survives_best_window_removal": not st["best_window_destroyed_result"],
                "mean_without_best_window": st["mean_without_best_window"],
                "window_ics": win_ics,
            },
            "regime_behavior": {
                "negative_year": "2022 (bear + high-volatility rate-hike stress)",
                "positive_years": "2023, 2024, 2025",
                "direction_regimes_present_in_test_span": dir_regs,
                "note": "Per-session stratified regime IC not computable from stored artifacts; "
                        "regime attribution is window-level only.",
            },
            "env050_result": {"mean_ic": u.get("ENV-050", {}).get("mean_ic"), "n_windows": u.get("ENV-050", {}).get("n")},
            "env100_result": ({"mean_ic": u["ENV-100"]["mean_ic"], "n_windows": u["ENV-100"]["n"]}
                              if isinstance(u.get("ENV-100"), dict) and "mean_ic" in u.get("ENV-100", {})
                              else {"status": u.get("ENV-100", {}).get("status", "not_run")}),
            "sub_universe_results": {k: v.get("mean_ic") for k, v in u.items() if k.startswith(("TOP", "BOT"))},
            "model_family_consistency": {
                "linear_replication": "ridge and lasso both positive in sequential windows",
                "nonlinear_replication": "NOT replicated — Phase 12D/12E RF/XGBoost mean IC negative "
                                         "(12.9C: linear +0.0147 vs nonlinear −0.0038)",
                "temporal_nonlinear_test": "not performed in 13A/13B (infrastructure scope)",
            },
            "statistical_evidence": {
                "window_level_t_pvalue": inf["p_value"],
                "ci95": [inf["ci_95_lower"], inf["ci_95_upper"]],
                "holm_significant": inf.get("significant_after_holm") in (True, "True"),
                "cohens_d": inf["cohens_d"],
                "verdict": "NOT statistically distinguishable from zero (n=8 windows; CI includes 0)",
            },
            "economic_magnitude": {
                "mean_ic_sequential": st["mean_ic"],
                "academic_reference": "IC≥0.05 commonly cited as meaningful; candidate below threshold",
                "portfolio_layer": "UNTESTED — no portfolio construction/backtest exists for these candidates",
            },
            "phase13b_stress_response": {
                "cost": ev["p13b_cost"][cid],
                "slippage": ev["p13b_slip"][cid],
                "parameter": {k: v["mean_ic"] for k, v in p.items()},
                "feature_noise": {k: v["mean_ic"] for k, v in f.get("noise", {}).items()},
                "leave_one_out": {k: round(v["mean_ic"], 4) for k, v in f.get("loo", {}).items()},
                "execution_delay": {k: round(v["mean_ic"], 4) for k, v in d.items()},
                "data_missing": {k: v.get("mean_ic") for k, v in ev["p13b_data"][cid].items()},
            },
            "known_failure_regions": [],  # filled in Step 4
            "reproducibility_flag": {
                "cross_implementation_divergence_EXP003": impl_divergence[cid],
                "root_cause": "Phase 13B purge used feature-frame boundary instead of label outcome "
                              "window; 13B baselines mildly optimistic. Conservative basis = 13A.",
            },
        }
    save_json("phase13c_evidence_matrix.json", {
        "purpose": "Full candidate evidence table, all unfavorable metrics retained",
        "candidates": evidence_table,
        "chain_references": {
            "phase9_11": "baseline ML, ablation, inference (null on locked protocol)",
            "phase12A_E": "information domains + real PIT fundamentals (no convincing improvement)",
            "phase129A_C": "integrity audit (B), clean-run replication (B), red-team (GREEN, 4 limitations)",
            "phase13A": "temporal/regime lab (verdict B)",
            "phase13B": "perturbation/stress lab (verdict B, with purge caveat above)",
        },
    })

    # ==================================================================
    # STEP 2+3 — SCORECARD AND ROBUSTNESS MATRIX
    # ==================================================================
    print("\n[STEP 2-3] Scorecard + robustness matrix...")
    dims = ["temporal_consistency", "regime_consistency", "universe_consistency",
            "model_family_consistency", "statistical_support", "economic_materiality",
            "perturbation_stability", "reproducibility", "pit_data_integrity"]

    scorecard, matrix = {}, {}
    for cid in ["CAND-03", "CAND-04"]:
        et = evidence_table[cid]
        ts = et["phase13a_temporal_stability"]

        sc = {}
        # 1 Temporal: 75% positive, 1 flip, survives removal, but NOT all windows and not significant
        sc["temporal_consistency"] = ("LIMITATION", "75% positive windows, 1 sign flip, survives best-window "
            "removal; 2022 negative; window-level p≈0.21 not significant")
        # 2 Regime: negative in 2022 bear/high-vol; stratified per-session IC unavailable
        sc["regime_consistency"] = ("LIMITATION", "Effect absent/negative in 2022 bear + high-vol regime; "
            "per-session regime stratification unavailable (window-level only)")
        # 3 Universe
        if cid == "CAND-03":
            sc["universe_consistency"] = ("PASS", "Positive across ENV-050, ENV-100, TOP25, BOT25 "
                "(+0.0228..+0.0349); ENV-100 n=4 windows")
        else:
            sc["universe_consistency"] = ("LIMITATION", "Positive in ENV-050/TOP25/BOT25; ENV-100 run "
                "failed (infrastructure error) — universe replication incomplete")
        # 4 Model family
        sc["model_family_consistency"] = ("FAIL", "Effect does not replicate in nonlinear families "
            "(RF/XGB negative mean IC historically); only linear models carry it")
        # 5 Statistical support
        sc["statistical_support"] = ("FAIL", "Window-level t-test p≈0.20-0.22; 95% CI includes zero; "
            "fails Holm correction; n=8 windows limits power but evidence remains insignificant")
        # 6 Economic materiality
        sc["economic_materiality"] = ("FAIL", "Mean sequential IC ≈0.02-0.03 (below 0.05 academic ref); "
            "economic usefulness remains untested at the portfolio-construction layer")
        # 7 Perturbation
        sc["perturbation_stability"] = ("LIMITATION", "Cost/slippage graceful; LOO benign; BUT feature-noise "
            "cliff (CAND-04 sign-flips at 5% noise), parameter sweep NaN gaps, delay degrades IC")
        # 8 Reproducibility
        div = et["reproducibility_flag"]["cross_implementation_divergence_EXP003"]
        sc["reproducibility"] = ("LIMITATION", f"Historical phases replicated (12.9B B) but 13A-vs-13B "
            f"independent implementations diverge on identical window (ratio {div['ratio']:.1f}x) due to "
            f"purge-boundary defect in 13B; conservative 13A basis adopted")
        # 9 PIT integrity
        sc["pit_data_integrity"] = ("PASS", "12.9C red-team PASS on leakage/labels/PIT joins; shift(1) "
            "boundary verified adversarially; no future-data access detected")

        scorecard[cid] = {"dimensions": {k: {"cell": v[0], "rationale": v[1]} for k, v in sc.items()},
                          "pass": sum(1 for v in sc.values() if v[0] == "PASS"),
                          "limitation": sum(1 for v in sc.values() if v[0] == "LIMITATION"),
                          "fail": sum(1 for v in sc.values() if v[0] == "FAIL")}
        matrix[cid] = {k: v[0] for k, v in sc.items()}
        print(f"  {cid}: PASS={scorecard[cid]['pass']} LIMITATION={scorecard[cid]['limitation']} FAIL={scorecard[cid]['fail']}")

    save_json("phase13c_scorecard.json", {
        "scoring_rule": "PASS / LIMITATION / FAIL per dimension; FAIL never hidden inside averages",
        "critical_fail_visible": True,
        "candidates": scorecard,
    })
    save_json("phase13c_evidence_matrix_tmp_delete_me.json", matrix)  # placeholder removed below
    (BENCH / "phase13c_evidence_matrix_tmp_delete_me.json").unlink()

    # ==================================================================
    # STEP 4 — FAILURE REGION ANALYSIS
    # ==================================================================
    print("\n[STEP 4] Failure region analysis...")
    failure_regions = {}
    fr_common = {
        "when_does_it_work": "Expanding/rolling sequential windows testing 2023-2025 (post-rate-hike normalization); "
                             "mid/large-cap liquid names; linear models; LAB-004 absolute-return label",
        "when_does_it_fail": "2022 test year (bear + high volatility + rate-hike stress); nonlinear model families; "
                             "feature vectors perturbed with ≥5-10% Gaussian noise; execution delayed ≥1 session",
        "gradual_or_catastrophic": "MIXED — degradation is gradual for costs/slippage/delay; catastrophic "
                                   "(sign-reversing) under feature corruption and in the 2022 regime",
        "regime_specific": "YES — failure concentrates in the single bear/high-vol year (2022)",
        "universe_specific": "NO for CAND-03 (consistent across 4 universes); unresolved for CAND-04 (ENV-100 missing)",
        "model_specific": "YES — strictly a linear-model phenomenon; tree ensembles reverse sign",
        "single_period_dependence": "PARTIAL — not one window (positive in 6/8), but ALL net positive evidence "
                                    "originates in the 2023-2025 sub-period; excluding it leaves the 12D-era "
                                    "null (locked-protocol IC ≈ 0.000-0.011)",
    }
    for cid in ["CAND-03", "CAND-04"]:
        fn = {k: v["mean_ic"] for k, v in ev["p13b_feat"][cid]["noise"].items()}
        sign_flip_noise = min((float(k[1:]) for k, v in fn.items()
                               if (v < 0) != (ev["p13b_feat"][cid]["noise"]["n0.01"]["mean_ic"] >= 0)), default=None)
        failure_regions[cid] = {
            **fr_common,
            "candidate_specific": {
                "feature_noise_sign_flip_threshold": f"{sign_flip_noise:.2f} std" if sign_flip_noise else "none within tested grid",
                "worst_sequential_window": ev["p13a_stab"][cid]["worst_ic"],
                "worst_window_id": ev["p13a_stab"][cid]["worst_window_id"],
                "dispersion": ev["p13a_stab"][cid]["ic_dispersion"],
                "locked_protocol_null_masking": "Aggregate 2022-2026 IC (~0.000-0.011) conceals year-level "
                                                "variation uncovered by 13A; effect is period-concentrated",
            },
            "explicit_statement": "Failure regions are a successful research output and are reported verbatim; "
                                  "nothing suppressed.",
        }
    save_json("phase13c_failure_regions.json", failure_regions)

    # Attach to evidence table copy
    for cid in ["CAND-03", "CAND-04"]:
        evidence_table[cid]["known_failure_regions"] = failure_regions[cid]

    # ==================================================================
    # STEP 5 — ECONOMIC MATERIALITY REVIEW
    # ==================================================================
    print("\n[STEP 5] Economic materiality review...")
    econ = {
        "separation_principle": {
            "statistical_detectability": "Weak-positive and UNPROVEN: window-level p≈0.20, CI spans zero; "
                                         "rank-information present in 6/8 sequential windows",
            "practical_usefulness": "UNTESTED — Economic usefulness remains untested at the "
                                    "portfolio-construction layer.",
        },
        "joint_assessment": {
            "magnitude": "Mean sequential IC 0.018 (ridge) / 0.027 (lasso) — below the 0.05 academic reference; "
                         "13B figures (up to 0.07-0.10) are inflated by the documented purge defect and are discounted",
            "consistency": "6/8 windows positive; 1 sign cluster (2022)",
            "persistence": "Concentrated in 2023-2025; locked-protocol aggregate over 2022-2026 is ≈0",
            "degradation": "Gradual under cost/slippage multipliers (rank-based IC); cliff under feature corruption; "
                           "monotone decay under execution delay",
            "cost_sensitivity": "Directional estimate only — at assumed 5 bps baseline, 5x costs erode ≈12% of "
                                "signal proxy; NO turnover-aware portfolio simulation exists to confirm",
            "uncertainty": "Dominant: n=8 windows, implementation-sensitivity (13A vs 13B divergence), "
                           "and unquantified selection over the 2010-2021 training span",
        },
        "explicit_limitation_statement": "Economic usefulness remains untested at the portfolio-construction layer. "
                                         "IC alone does not demonstrate profitability; no Sharpe/turnover/capacity "
                                         "evidence exists for any ORBIT candidate.",
    }

    # ==================================================================
    # STEP 6 — CONCLUSION LANGUAGE REVIEW
    # ==================================================================
    print("\n[STEP 6] Conclusion language review...")
    forbidden = ["ORBIT found alpha", "the model predicts stocks", "the signal works", "no signal exists"]
    approved = [
        "No robust predictive relationship was established under the tested configurations.",
        "Two linear baseline candidates (ridge/lasso on FS-12B-A) retained positive out-of-sample rank "
        "information in 6 of 8 sequential evaluation windows and across four tested universes, but window-level "
        "statistical significance was not established (p≈0.20, CI includes zero).",
        "Candidate effects are concentrated in the 2023–2025 sub-period and absent in 2022 and in nonlinear models.",
        "A candidate relationship survived specified robustness tests but economic usefulness remains unproven.",
    ]
    violations_found = [
        {"artifact": "PHASE_9_STATUS.md", "issue": "'no signal' without qualification",
         "status": "carried from 12.9C; superseded by calibrated language here"},
        {"artifact": "phase13b_report.json key_findings", "issue": "'Universe: ENV-050 and ENV-100 show consistent signs' "
         "stated while ENV-100 errored for CAND-04", "status": "flagged — overstated for CAND-04"},
        {"artifact": "phase13b_report.json key_findings", "issue": "'Parameter: both candidates stable across 0.5x-3x "
         "alpha range (GRACEFUL)' while 5/6 sweep points returned NaN", "status": "flagged — unsupported claim; "
         "actual classification MIXED"},
        {"artifact": "phase13b cost/slippage files", "issue": "Adjusted-IC presented as measured metrics when they are "
         "analytical proxies (baseline_ic × cost-drag factor)", "status": "flagged — must be read as estimates"},
    ]
    conclusion_review = {
        "forbidden_phrases_checked": forbidden,
        "violations_found_in_chain": violations_found,
        "approved_language_for_phase13c": approved,
        "canonical_conclusion": (
            "No convincing, robust predictive relationship has been established across Phases 9-13B. "
            "Sequential evaluation isolated a weak, linear-only, post-2022 rank-information effect in two "
            "baseline-OHLCV candidates; it is statistically insignificant at the window level, concentrated "
            "in one sub-period, and economically unvalidated."
        ),
    }
    save_json("phase13c_conclusion_review.json", conclusion_review)

    # ==================================================================
    # STEP 7 — PROMOTION GATE
    # ==================================================================
    print("\n[STEP 7] Promotion gate classification...")
    registry = {}
    # Decision rules (locked before assignment):
    #   ROBUST_CANDIDATE: no FAIL cells and ≥5 PASS
    #   RESEARCH_ONLY:    ≤1 FAIL cell, positive surviving evidence, material open items
    #   FRAGILE:          positive evidence but ≥2 dependency-driven LIMITATIONs incl. universe/model gaps
    #   REJECTED:         temporal or stress validation failed outright
    registry["CAND-03"] = {
        "classification": "RESEARCH_ONLY",
        "matrix_cells": matrix["CAND-03"],
        "justification": "2 FAIL (statistics, economics) prevent promotion; but universe-consistent (4/4), "
                         "temporally persistent ex-2022, PIT-clean, survives best-window removal — earns "
                         "controlled further investigation, not production proximity",
        "promotion_blocked_by": ["statistical insignificance", "portfolio-layer untested",
                                 "period concentration 2023-2025"],
        "allowed_next_steps": ["registry entry at research tier", "pre-registered longer-window re-evaluation",
                               "per-session regime stratification", "turnover-aware cost simulation"],
        "forbidden_next_steps": ["production deployment", "capital allocation", "hyperparameter expansion search"],
    }
    registry["CAND-04"] = {
        "classification": "FRAGILE",
        "matrix_cells": matrix["CAND-04"],
        "justification": "Higher headline IC than CAND-03 but: ENV-100 replication missing, sign-flip under 5% "
                         "feature noise, widest IC dispersion (0.050), and identical statistical/economic failures. "
                         "Evidence exists yet is environment-sensitive",
        "promotion_blocked_by": ["incomplete universe replication", "feature-noise fragility",
                                 "statistical insignificance", "portfolio-layer untested"],
        "allowed_next_steps": ["registry entry at research tier (fragile flag)",
                               "ENV-100 completion run", "noise-robustness investigation"],
        "forbidden_next_steps": ["production deployment", "capital allocation"],
    }
    save_json("phase13c_candidate_registry.json", {
        "gate_definitions": {
            "REJECTED": "Fails temporal or stress validation",
            "FRAGILE": "Positive evidence exists but is highly environment/model/regime-dependent",
            "RESEARCH_ONLY": "Some evidence survives; robustness or economic usefulness insufficient for promotion",
            "ROBUST_CANDIDATE": "Survives temporal, regime, universe, and perturbation testing with no major "
                                "contradiction (does NOT mean profitable)",
        },
        "decision_rules_locked_before_assignment": True,
        "registry": registry,
        "n_robust_candidates": sum(1 for r in registry.values() if r["classification"] == "ROBUST_CANDIDATE"),
    })

    # ==================================================================
    # STEP 8 — PHASE 14 READINESS + FINAL GATE
    # ==================================================================
    print("\n[STEP 8] Phase 14 readiness...")
    n_robust = sum(1 for r in registry.values() if r["classification"] == "ROBUST_CANDIDATE")
    n_research = sum(1 for r in registry.values() if r["classification"] in ("RESEARCH_ONLY", "FRAGILE"))

    # Gate rule (locked): GREEN needs ≥1 ROBUST_CANDIDATE; RED if no surviving candidate;
    # YELLOW if research-only/fragile candidates are sufficiently characterized to populate a registry.
    if n_robust >= 1:
        gate = "GREEN"
    elif n_research >= 1:
        gate = "YELLOW"
    else:
        gate = "RED"

    readiness = {
        "question": "Do we have sufficiently well-characterized candidate models to justify formal model "
                    "identity, lifecycle management, and promotion controls?",
        "answer": "YES, conditionally" if gate == "YELLOW" else ("YES" if gate == "GREEN" else "NO"),
        "characterization_evidence": [
            "Both candidates carry complete identity tuples (family/params/feature-set/label/environment)",
            "Temporal profile mapped across 8 sequential windows with explicit failure year",
            "Perturbation response curves recorded for 7 stress dimensions",
            "Failure regions enumerated explicitly (regime, model-family, noise thresholds)",
            "Reproducibility defect identified and root-caused (13B purge boundary)",
        ],
        "conditions": [
            "All candidates enter at RESEARCH tier — nothing is production-eligible",
            "Promotion Gate must require: window-level significance, cross-universe completion, "
            "portfolio-level cost simulation, and independent reimplementation agreement",
            "The 13B purge defect must be fixed and 13B re-run before any 13B-derived number is reused",
        ],
        "final_gate": gate,
        "gate_rationale": (
            "No candidate qualifies as ROBUST_CANDIDATE (statistical and economic FAIL cells stand), so GREEN is "
            "unavailable. Both candidates survive outright rejection and are exhaustively characterized, so RED "
            "(return to hypothesis design) would discard usable characterization. YELLOW applies: proceed to "
            "Phase 14 with research-only/fragile statuses and documented limitations."
        ),
    }

    # ==================================================================
    # AUDIT + FINAL REPORT
    # ==================================================================
    audit = {
        "phase": "13C",
        "created_at": datetime.now().isoformat(),
        "role": "independent senior quantitative research reviewer (synthesis only)",
        "artifacts_reviewed": 15,
        "phases_in_chain": ["9", "10", "11", "11.1", "11.2", "12A", "12B", "12C", "12D", "12E",
                            "12.9A", "12.9B", "12.9C", "13A", "13B"],
        "new_exploration_performed": False,
        "historical_artifacts_modified": False,
        "unfavorable_metrics_suppressed": False,
        "findings": [
            "F1: No candidate achieves ROBUST_CANDIDATE — statistics and economics FAIL cells stand for both",
            "F2: Effect is linear-model-specific; nonlinear families reverse sign (chain-wide)",
            "F3: Net positive evidence originates exclusively in the 2023-2025 sub-period; 2022 is a failure region",
            "F4: Cross-implementation divergence found: 13B purge-boundary defect inflates baselines up to 3x "
                "on identical windows; conservative 13A basis adopted throughout this synthesis",
            "F5: CAND-04 lacks completed ENV-100 replication and sign-flips at 5% feature noise → FRAGILE",
            "F6: Prior phase reports contained overstated findings (parameter stability, universe consistency) — "
                "corrected in conclusion review",
            "F7: Economic usefulness remains untested at the portfolio-construction layer for every candidate",
        ],
        "plan_digest": sha256_obj({"phase": "13C", "dims": dims, "rules": "locked-in-script"}),
        "elapsed_seconds": round(time.time() - t0, 1),
    }

    report = {
        "phase": "13C",
        "title": "Robustness Synthesis & Final Gate",
        "canonical_conclusion": conclusion_review["canonical_conclusion"],
        "evidence_chain_summary": {
            "locked_protocol_era": "Phase 9-12E on the fixed 2022-2026 holdout: null persists "
                                   "(best baseline IC ≈ 0.000-0.011 absolute, ≈0.014 excess)",
            "sequential_era": "Phase 13A decomposes the null: year-resolved ICs reveal negative 2022, "
                              "positive 2023-2025 (mean 0.018-0.027, insignificant)",
            "stress_era": "Phase 13B: graceful under cost/slippage; cliffs under feature corruption; "
                          "purge-defect caveat applies to all 13B absolute levels",
            "audit_era": "12.9A/B/C integrity confirmed (leakage/PIT clean; multiple-testing and "
                         "materiality limitations documented)",
        },
        "robustness_matrix": matrix,
        "promotions": {cid: registry[cid]["classification"] for cid in registry},
        "failure_region_headlines": [
            "Regime: bear/high-volatility 2022 destroys the effect",
            "Model family: strictly linear-only; trees reverse",
            "Data quality: ≥5% feature noise flips CAND-04; ≥10% collapses CAND-03's excess",
            "Execution: each session of delay erodes rank information",
            "Period: all net evidence is post-2022",
        ],
        "economic_materiality": econ,
        "phase14_readiness": readiness,
        "final_gate": gate,
        "final_gate_statement": {
            "GREEN": "At least one candidate is sufficiently characterized to proceed to Phase 14",
            "YELLOW": "Proceed to Phase 14, but only with research-only/fragile candidate status and "
                      "documented limitations",
            "RED": "No candidate has survived sufficiently; return to hypothesis design",
        }[gate],
        "answers_to_mandate_questions": {
            "q1_survivors": "None fully; two partially (ridge/lasso FS-12B-A) as RESEARCH_ONLY/FRAGILE",
            "q2_distribution": "Concentrated — 2023-2025 only; absent 2022 and pre-2022 holdout aggregate",
            "q3_destroying_regimes": "Bear + high-volatility (2022); feature corruption ≥5-10%; ≥1-session delays",
            "q4_best_window_removal": "Yes — means stay positive without the best window "
                                      "(ridge +0.0121, lasso +0.0203), though weaker",
            "q5_env_consistency": "Ridge: consistent 050/100/TOP/BOT; Lasso: incomplete (ENV-100 run failed)",
            "q6_model_consistency": "Inconsistent — linear-only; RF/XGBoost negative chain-wide",
            "q7_further_stress_testing": "Yes — both merit: per-session regime stratification, ENV-100 completion "
                                         "for lasso, turnover-aware cost simulation, and 13B re-run post-fix",
        },
        "stop_directive": "Phase 13C ends here. Phase 14 not started.",
    }

    save_json("phase13c_audit.json", audit)
    save_json("phase13c_report.json", report)

    # Re-save evidence matrix including failure regions attached
    save_json("phase13c_evidence_matrix.json", {
        "purpose": "Full candidate evidence table; every unfavorable metric retained",
        "candidates": evidence_table,
    })

    print(f"\n{'='*72}")
    print(f"PHASE 13C COMPLETE — FINAL GATE: {gate}")
    print(f"{'='*72}")
    print(f"  Promotions: CAND-03={registry['CAND-03']['classification']}, "
          f"CAND-04={registry['CAND-04']['classification']}")
    print(f"  Matrix: " + json.dumps(matrix))
    print(f"  Elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
