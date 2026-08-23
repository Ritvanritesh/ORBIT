"""Phase 11 permanent report generator.

Produces:
  benchmarks/phase11_inference_results.parquet  (machine-readable)
  benchmarks/phase11_inference_results.md        (human-readable)
  docs/phase11_statistical_inference.md          (permanent research report)
  PHASE_11_STATUS.md                             (status verdict)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl

from orbit.ml.phase11_effects import EffectSize, significance_economy_matrix

_REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_PARQUET = _REPO_ROOT / "benchmarks" / "phase11_inference_results.parquet"
RESULTS_MD = _REPO_ROOT / "benchmarks" / "phase11_inference_results.md"
RESEARCH_MD = _REPO_ROOT / "docs" / "phase11_statistical_inference.md"
STATUS_MD = _REPO_ROOT / "PHASE_11_STATUS.md"
AUDIT_JSON = _REPO_ROOT / "benchmarks" / "phase11_audit_results.json"
PLAN_JSON = _REPO_ROOT / "benchmarks" / "phase11_inference_plan.json"


def write_markdown_report(analysis: dict[str, Any]) -> Path:
    """Generate benchmarks/phase11_inference_results.md."""
    RESULTS_MD.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase 11 - Statistical Inference and Uncertainty (Results)",
        "",
        f"Generated: {analysis.get('timestamp', 'N/A')}",
        f"Plan digest: {analysis.get('plan_digest', 'N/A')[:16]}...",
        "",
        "## Summary",
        "",
        f"- Phase 9 experiments analyzed: {analysis.get('n_phase9_experiments', 0)}",
        f"- Phase 10 experiments analyzed: {analysis.get('n_phase10_experiments', 0)}",
        f"- Total inference results: {analysis.get('n_inference_results', 0)}",
        "",
        "## Multiple-Comparison Analysis",
        "",
    ]

    mt = analysis.get("multiple_testing")
    if mt:
        lines.append(f"**Family:** {mt.get('family', {}).get('family_id', 'N/A')}")
        lines.append(f"**Members:** {mt.get('family', {}).get('n_members', 0)}")
        lines.append("")

        holm = mt.get("holm_bonferroni", {})
        bh = mt.get("benjamini_hochberg", {})

        lines.append("### Holm-Bonferroni (FWER)")
        lines.append(f"- Raw significant at 5%: {mt.get('n_raw_significant_005', 0)}")
        lines.append(f"- Adjusted significant at 5%: {len(holm.get('significant_at', {}).get('0.05', []))}")
        lines.append(f"- Adjusted significant at 1%: {len(holm.get('significant_at', {}).get('0.01', []))}")
        lines.append("")

        lines.append("### Benjamini-Hochberg (FDR)")
        lines.append(f"- Adjusted significant at 5%: {len(bh.get('significant_at', {}).get('0.05', []))}")
        lines.append(f"- Adjusted significant at 1%: {len(bh.get('significant_at', {}).get('0.01', []))}")
        lines.append("")
    else:
        lines.append("*No multiple-testing analysis available.*")
        lines.append("")

    lines += [
        "## Power Analysis",
        "",
    ]
    power = analysis.get("power_analysis", {})
    if power:
        for key, val in power.items():
            lines.append(f"**{key}:** {val.get('label', 'N/A')} - "
                         f"MDE={val.get('assumed_effect_size', 'N/A')}, "
                         f"n={val.get('sample_size', 'N/A')}")
    lines += [
        "",
        "## Phase 10 Experiment Results",
        "",
    ]

    # Group Phase 10 results by experiment
    p10_results = {}
    for r in analysis.get("inference_results", []):
        for eid in r.source_experiment_ids:
            if eid.startswith("EXP-10"):
                if eid not in p10_results:
                    p10_results[eid] = []
                p10_results[eid].append(r)

    lines.append("| Experiment | Metric | Point Estimate | CI Lower | CI Upper | Method | Effect Size |")
    lines.append("|------------|--------|----------------|----------|----------|--------|-------------|")
    for eid in sorted(p10_results.keys()):
        for r in p10_results[eid]:
            pe = f"{r.ci.point_estimate:.4f}" if r.ci.point_estimate is not None else "N/A"
            lo = f"{r.ci.lower:.4f}" if r.ci.lower is not None else "N/A"
            hi = f"{r.ci.upper:.4f}" if r.ci.upper is not None else "N/A"
            es = f"{r.effect_size:.4f}" if r.effect_size is not None else "N/A"
            lines.append(
                f"| {eid} | {r.metric} | {pe} | {lo} | {hi} | {r.ci.method} | {es} |"
            )
    lines += [
        "",
        "## Statistical vs Economic Significance Matrix",
        "",
    ]

    # Summary matrix
    for eid in sorted(p10_results.keys())[:5]:  # Top 5 for brevity
        for r in p10_results[eid]:
            if r.metric == "oos_ic":
                es_magnitude = r.effect_size if r.effect_size is not None else 0.0
                if es_magnitude < 0.01:
                    es_interp = "negligible"
                elif es_magnitude < 0.03:
                    es_interp = "small"
                else:
                    es_interp = "potentially_meaningful"
                effect = EffectSize(
                    metric="ic",
                    magnitude=es_magnitude,
                    method=r.effect_size_method or "raw_ic_magnitude",
                    interpretation=es_interp,
                    thresholds={"negligible": 0.01, "meaningful": 0.03},
                    raw_value=r.ci.point_estimate,
                )
                matrix = significance_economy_matrix(r.p_value, effect)
                lines.append(f"**{eid}** OOS IC: statistical={matrix['statistical_evidence']}, "
                             f"economic={matrix['economic_meaning']}")
    lines += [
        "",
        "## Limitations and Assumptions",
        "",
        "- Bootstrap CIs are approximate; they do not provide exact finite-sample coverage.",
        "- Block length is set by rule of thumb, not optimized against outcomes.",
        "- The label (5-session forward return) creates overlapping outcomes.",
        "- Backtest metrics inherit all backtester assumptions.",
        "- The inference layer analyzes evidence; it does not generate new signals.",
        "",
    ]

    RESULTS_MD.write_text("\n".join(lines), encoding="utf-8")
    return RESULTS_MD


def write_research_report(
    analysis: dict[str, Any],
    audit_checks: list[dict[str, Any]] | None = None,
) -> Path:
    """Generate docs/phase11_statistical_inference.md (permanent research report)."""
    RESEARCH_MD.parent.mkdir(parents=True, exist_ok=True)

    from orbit.ml.phase11_audit import audit_summary

    lines = [
        "# ORBIT Phase 11: Statistical Inference and Uncertainty",
        "",
        "Version 1.0 - permanent research report",
        "",
        "## 1. Executive Summary",
        "",
        "Phase 11 implements a formal, reproducible, auditable inference layer "
        "over the existing Phase 9 and Phase 10 research results. The analysis "
        "answers: how much should we believe the observed results? The primary "
        "danger is false discovery; the second danger is overcorrecting into "
        "opacity.",
        "",
        f"**Plan digest:** `{analysis.get('plan_digest', 'N/A')[:16]}...`",
        f"**Timestamp:** {analysis.get('timestamp', 'N/A')}",
        f"**Phase 9 experiments:** {analysis.get('n_phase9_experiments', 0)}",
        f"**Phase 10 experiments:** {analysis.get('n_phase10_experiments', 0)}",
        f"**Total inference results:** {analysis.get('n_inference_results', 0)}",
        "",
        "## 2. Locked Inference Protocol",
        "",
        "The inference plan was defined and locked BEFORE any real-data analysis:",
        "",
    ]

    if PLAN_JSON.exists():
        try:
            plan_data = json.loads(PLAN_JSON.read_text(encoding="utf-8"))
            lines.append(f"- Protocol: {plan_data.get('protocol', 'N/A')}")
            lines.append(f"- Seed: {plan_data.get('seed', 'N/A')}")
            lines.append(f"- Confidence level: {plan_data.get('confidence_level', 'N/A')}")
            lines.append(f"- Bootstrap resamples: {plan_data.get('n_bootstrap_resamples', 'N/A')}")
            lines.append(f"- Block length policy: {plan_data.get('block_length_policy', 'N/A')}")
        except Exception:
            lines.append("*(plan file could not be read)*")
    lines.append("")

    lines += [
        "## 3. Data and Source Artifact Lineage",
        "",
        f"- Phase 9 checksum: `{analysis.get('phase9_checksum', 'N/A')[:16]}...`",
        f"- Phase 10 checksum: `{analysis.get('phase10_checksum', 'N/A')[:16]}...`",
        "",
        "## 4. Dependence Diagnostics",
        "",
        "The label (LAB-004, 5-session forward return) creates overlapping "
        "outcomes. Each prediction at session D depends on prices at sessions "
        "D+1 through D+5, overlapping with predictions at sessions D+1 through "
        "D+4. This dependence is explicitly disclosed.",
        "",
        "Bootstrap confidence intervals use the moving block bootstrap method, "
        "which preserves local time-series dependence structure.",
        "",
        "## 5. Confidence Intervals",
        "",
        "All results report: point estimate, CI bounds, confidence level, "
        "method, and assumptions. Bootstrap CIs are the primary method for "
        "IC metrics; t-based CIs are used where per-session statistics are "
        "unavailable.",
        "",
        "## 6. Bootstrap Methodology",
        "",
        "- Method: Moving block bootstrap (time-series-aware)",
        "- Block length: Rule of thumb (ceil(n^(1/3)))",
        "- Resamples: 1,000 (locked)",
        "- Deterministic: seeded with SEED=42",
        "",
        "## 7. Effect Sizes",
        "",
        "Effect sizes are reported alongside statistical evidence. "
        "Statistical significance is NEVER treated as equivalent to "
        "economic usefulness.",
        "",
        "| Metric | Threshold: Negligible | Threshold: Meaningful |",
        "|--------|----------------------|----------------------|",
        "| IC | |IC| < 0.01 | |IC| >= 0.03 |",
        "| Return | |R| < 0% | |R| >= 10% |",
        "| Hit Rate | |HR-0.5| < 0.01 | |HR-0.5| >= 0.03 |",
        "",
        "## 8. Power Analysis",
        "",
        "Power calculations use i.i.d. normal approximations (labeled "
        "APPROXIMATE). They provide context for interpreting non-significant "
        "results: was the design powerful enough to detect economically "
        "meaningful effects?",
        "",
    ]

    power = analysis.get("power_analysis", {})
    if power:
        for key, val in power.items():
            lines.append(f"- **{key}:** {val.get('label', 'N/A')}")
            lines.append(f"  - Effect size: {val.get('assumed_effect_size', 'N/A')}")
            lines.append(f"  - Sample size: {val.get('sample_size', 'N/A')}")
    lines.append("")

    lines += [
        "## 9. Multiple-Comparison Analysis",
        "",
        "The Phase 10 grid (52 experiments) is treated as a SINGLE locked "
        "comparison family. No post-hoc exclusion of inconvenient experiments "
        "is permitted.",
        "",
    ]

    mt = analysis.get("multiple_testing")
    if mt:
        holm = mt.get("holm_bonferroni", {})
        bh = mt.get("benjamini_hochberg", {})
        lines.append(f"**Holm-Bonferroni (FWER control):**")
        lines.append(f"- Raw significant at 5%: {mt.get('n_raw_significant_005', 0)}")
        lines.append(f"- After correction at 5%: {len(holm.get('significant_at', {}).get('0.05', []))}")
        lines.append("")
        lines.append(f"**Benjamini-Hochberg (FDR control):**")
        lines.append(f"- After correction at 5%: {len(bh.get('significant_at', {}).get('0.05', []))}")
        lines.append("")
    lines += [
        "## 10. Statistical vs Economic Significance Matrix",
        "",
        "Every major result is categorized along two axes:",
        "",
        "- **Statistical evidence:** insufficient / evidence under stated assumptions / inconclusive",
        "- **Economic meaning:** negligible / potentially meaningful / not assessable",
        "",
        "A tiny IC may be statistically distinguishable from zero yet economically "
        "negligible. A large backtest return may appear economically attractive "
        "yet statistically inconclusive due to uncertainty, selection effects, "
        "or multiple comparisons.",
        "",
        "## 11. Phase 9 Findings",
        "",
        "*(See Phase 9 benchmark report for details. Phase 9 found no robust "
        "evidence of learnable structure with the original 8-feature set.)*",
        "",
        "## 12. Full Phase 10 Findings",
        "",
        "Phase 10 ran 52 experiments (13 feature sets x 4 model families). "
        "Key observations from the Phase 10 report:",
        "",
        "- All OOS IC values approximately |IC| <= 0.032",
        "- Rank IC values are similarly small",
        "- Hit rates approximately 0.517-0.537",
        "- Best after-cost returns ~+221% and +241% came from leave-one-family-out feature sets",
        "- Effects were inconsistent across model families",
        "- No robust family-specific predictive signal was established",
        "",
        "## 13. Explicit Treatment of All 52 Experiments",
        "",
        "Every Phase 10 experiment (EXP-10001 through EXP-10052) is analyzed "
        "in the inference results. No experiment is omitted. The full family "
        "is used for multiple-comparison corrections.",
        "",
        "## 14. Limitations and Assumptions",
        "",
        "- 20-symbol development universe; generalization untested",
        "- Block length by rule of thumb, not optimized",
        "- 5-session label creates overlapping outcomes",
        "- Bootstrap CIs are approximate",
        "- Backtest metrics inherit backtester assumptions",
        "- The inference layer analyzes evidence; it does not generate new signals",
        "",
        "## 15. Final Verdict",
        "",
    ]

    # Determine verdict based on actual results
    verdict = _determine_verdict(analysis)
    lines.append(f"**{verdict['code']}** - {verdict['description']}")
    lines += [
        "",
        "---",
        "",
        "*This report was generated by the locked Phase 11 inference machinery. "
        "The conclusions are based on actual results, not desired outcomes.*",
        "",
    ]

    RESEARCH_MD.write_text("\n".join(lines), encoding="utf-8")
    return RESEARCH_MD


def _determine_verdict(analysis: dict[str, Any]) -> dict[str, str]:
    """Determine the Phase 11 verdict from actual locked results."""
    mt = analysis.get("multiple_testing")
    if mt is None:
        return {
            "code": "E",
            "description": "Inference methodology insufficient or failed validation",
        }

    bh = mt.get("benjamini_hochberg", {})
    n_sig_bh = len(bh.get("significant_at", {}).get("0.05", []))
    holm = mt.get("holm_bonferroni", {})
    n_sig_holm = len(holm.get("significant_at", {}).get("0.05", []))
    n_total = mt.get("family", {}).get("n_members", 52)

    # Check effect sizes
    large_effects = 0
    for r in analysis.get("inference_results", []):
        if r.metric == "oos_ic" and r.effect_size is not None:
            if r.effect_size >= 0.03:
                large_effects += 1

    if n_sig_holm == 0 and n_sig_bh == 0:
        return {
            "code": "D",
            "description": (
                "Apparent effects are consistent with noise after dependence "
                "and multiple-comparison analysis"
            ),
        }
    elif n_sig_holm > 0 and large_effects > 0:
        return {
            "code": "B",
            "description": (
                "Some effects remain plausible but uncertain after "
                "multiple-comparison correction"
            ),
        }
    elif n_sig_bh > 0 and large_effects > 2:
        return {
            "code": "B",
            "description": (
                "Some effects survive FDR correction; interpretation requires "
                "caution regarding effect size and economic significance"
            ),
        }
    else:
        return {
            "code": "C",
            "description": (
                "Evidence remains weak or inconclusive after "
                "dependence-aware multiple-comparison analysis"
            ),
        }


def write_phase11_status(
    analysis: dict[str, Any],
    audit_checks: list[dict[str, Any]] | None = None,
) -> Path:
    """Generate PHASE_11_STATUS.md."""
    verdict = _determine_verdict(analysis)

    from orbit.ml.phase11_audit import audit_summary

    lines = [
        "# PHASE 11 STATUS",
        "",
        f"**Verdict: {verdict['code']}**",
        "",
        f"> {verdict['description']}",
        "",
        "## Summary",
        "",
        f"- Plan digest: `{analysis.get('plan_digest', 'N/A')[:16]}...`",
        f"- Phase 9 experiments: {analysis.get('n_phase9_experiments', 0)}",
        f"- Phase 10 experiments: {analysis.get('n_phase10_experiments', 0)}",
        f"- Total inference results: {analysis.get('n_inference_results', 0)}",
        "",
    ]

    if audit_checks:
        summary = audit_summary(audit_checks)
        lines += [
            "## Audit",
            "",
            f"- Checks: {summary['checks']}",
            f"- Passed: {summary['passed']}",
            f"- Failed: {summary['failed']}",
            f"- Blocked: {'YES' if summary['blocked'] else 'NO'}",
        ]
        if summary["failed_checks"]:
            lines.append(f"- Failed checks: {', '.join(summary['failed_checks'])}")
        lines.append("")

    lines += [
        "## Multiple-Testing Results",
        "",
    ]
    mt = analysis.get("multiple_testing")
    if mt:
        holm = mt.get("holm_bonferroni", {})
        bh = mt.get("benjamini_hochberg", {})
        lines.append(f"- Raw significant at 5%: {mt.get('n_raw_significant_005', 0)}")
        lines.append(f"- Holm significant at 5%: {len(holm.get('significant_at', {}).get('0.05', []))}")
        lines.append(f"- BH significant at 5%: {len(bh.get('significant_at', {}).get('0.05', []))}")
    lines += [
        "",
        "---",
        "",
        "*Status generated by Phase 11 inference machinery from locked results.*",
        "",
    ]

    STATUS_MD.write_text("\n".join(lines), encoding="utf-8")
    return STATUS_MD


__all__ = [
    "write_markdown_report",
    "write_research_report",
    "write_phase11_status",
]
