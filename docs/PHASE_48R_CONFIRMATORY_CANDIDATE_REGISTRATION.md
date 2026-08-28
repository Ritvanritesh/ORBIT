# Phase 48-R: Confirmatory Candidate Registration

**Date:** 2026-08-28T18:55:54.989678+00:00

---

## Summary

| Item | Value |
|---|---|
| **Candidate** | CAND-RIDGE-FS001-001 |
| **Branch** | BR-C3D4E5F6A1B2 |
| **Registration Status** | CONFIRMATORY_REGISTERED |
| **Budget Integrity** | PASS |

---

## Locked Configuration

| Item | Value |
|---|---|
| **Model** | Ridge Regression |
| **Alpha** | 1.0 |
| **Preprocessing** | StandardScaler (fit on train only) |
| **Feature System** | FS-001 |
| **Feature Count** | 4 |
| **Features** | YC_LEVEL, YC_SLOPE, YC_CURVATURE, YC_CHG_10D |
| **Baseline** | 5 price-derived features (RET_5D, RET_10D, RET_20D, VOL_20D, MKT_RET_20D) |
| **Primary Horizon** | H-10 |
| **Secondary Horizon** | H-20 |
| **Universes** | DS-EXP-050, DS-EXP-100 |
| **Experiment Budget** | 6 |
| **Experiment Matrix** | 6 experiments (4 primary H-10 + 2 secondary H-20) |

---

## Primary Success Criterion

Incremental IC > +0.005 AND p < 0.05 (Holm-Bonferroni, family=2) in BOTH universes at H-10.

---

## Mandatory Limitations

1. **TEMPORAL_SIGNAL_DECAY**: Early +0.058 → Middle +0.041 → Late +0.012
2. **YC_LEVEL_UNSTABLE**: Feature distribution shifts across time (Phase 47-R)

---

## Configuration Freeze

PASS — all digests generated and locked.

---

## PIT Integrity

PASS — all features PIT_NATIVE.

---

## Baseline Integrity

PASS — 5 real predictive features, non-degenerate, non-zero variance.

---

## Adversarial

37/37 PASS

---

## Reproducibility

PASS

---

## Firewall

- OOS targets accessed: NO
- OOS IC calculated: NO
- Confirmatory tests executed: NO
- Locked registrations modified: NO

---

## Next Allowed Step

REGISTERED_WAITING_FOR_DATA — Wait for OOS DATA_READY (36/60 days).
