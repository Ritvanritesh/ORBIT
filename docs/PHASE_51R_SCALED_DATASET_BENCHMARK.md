# Phase 51-R: Scaled Dataset Benchmark

## Status
- **Verdict**: C (NO_MEANINGFUL_SCALING_EFFECT)
- **Gate**: YELLOW
- **Completed**: 2026-08-29T08:20:17.003485 UTC
- **Elapsed**: 68.2s

---

## Dataset Comparison

| Metric | V1 | V2 | Change |
|--------|-----|-----|--------|
| Symbols | 97 | 97 | Same universe |
| Raw observations | 680,878 | 680,878 | +0% |
| Effective observations | ~16,821 | ~18,950 | +12.7% |
| Sectors | 12 | 12 | Same |

---

## Data Scaling Effect

| Metric | Value |
|--------|-------|
| Mean deltaIC | +0.0000 |
| Median deltaIC | +0.0000 |
| Variance V1 | 0.000647 |
| Variance V2 | 0.000647 |
| Improved | 0/16 |
| Degraded | 0/16 |
| Classification | **NO_EFFECT** |

---

## Model Comparison

| Model | V1 Mean IC | V2 Mean IC | delta IC | Direction |
|-------|-----------|-----------|------|-----------|

---

## Advanced Model Justification

| Model | Classification | Justification |
|-------|---------------|---------------|
| MLP | READY | 18,950 effective obs > 5,000 threshold |
| TCN | READY | 17,833 effective sequences > 10,000 threshold |
| Transformer | BORDERLINE | 18,950 eff obs (near 20K threshold), 97 symbols (below 150) |

---

## Firewall
- OOS targets accessed: NO
- OOS IC calculated: NO
- Confirmatory tests executed: NO
- Locked registrations modified: NO

## Adversarial
- 38/38 PASS

## Reproducibility
- 12/12 PASS

---

## Next Allowed Step
NO_MEANINGFUL_SCALING_EFFECT

Do NOT automatically begin the next phase. Wait for user approval.
