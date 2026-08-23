# Phase 14 - Model Registry & Evidence-Gated Promotion

*Generated: 2026-08-23T00:00:00+00:00 | Verdict: **B** | Gate: **GREEN***

> A model being registered is NOT evidence that it works.

## Headline numbers
- Experiments/artifacts inspected: 88 run dirs, 121 result files
- Distinct model versions registered: **11**
- Evidence records: **149**
- Promotion decisions: **13** (PROMOTE=0, BLOCK=10, RETAIN=2, RETIRE=1)
- Status distribution: `{'RESEARCH': 10, 'RETIRED': 1}`
- Adversarial tests: 18/18 passed
- Reproducibility: PASS (double-build identical)
- Replay: IDENTITY_REPLAY 11/11; FULL_NUMERICAL_REPLAY not claimed

## REGISTERED vs VALIDATED vs PAPER
| Tier | Count | Meaning |
|---|---|---|
| REGISTERED | 11 | identity+provenance only |
| VALIDATED | 0 | gates passed; not economic proof |
| PAPER-ELIGIBLE | 0 | none; portfolio gates unsatisfied |

## Candidate statuses (must match Phase 13C)
- CAND-03 (MODEL-00009): RESEARCH - blockers: statistical FAIL, economic FAIL, model-family FAIL, temporal concentration, 13B defect quarantine
- CAND-04 (MODEL-00010): RESEARCH (FRAGILE) - all CAND-03 blockers plus ENV-100 gap, 5%-noise sign-flip, widest dispersion

## Unresolved defects / limitations
1. Phase 13B purge-boundary defect (HIGH): 13B absolute stress levels quarantined.
2. No serialized estimators from Phases 9-13: numerical replay impossible; identity replay only.
3. Portfolio/execution validation: NOT_EVALUATED everywhere.

**Stop directive honored: Phase 15 not started.**