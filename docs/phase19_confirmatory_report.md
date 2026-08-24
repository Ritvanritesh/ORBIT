# Phase 19 — Locked Confirmatory Report

Generated: 2026-08-24T13:41:18.098664+00:00

## Summary

- **Verdict**: B
- **Gate**: YELLOW
- **Holdout Classification**: PSEUDO_CONFIRMATORY
- **Confirmed**: 0
- **Partially Confirmed**: 4
- **Failed**: 0

## Hypothesis Outcomes

### HYP-MOM
- Status: **PARTIALLY_CONFIRMED**
- Mean Val IC: 0.0142
- Exceeds IC Threshold: True
- Exceeds Null: True
- Temporal Stability: True
- Positive Window Fraction: 1.00
- Downgrade Reason: Downgraded from CONFIRMED to PARTIALLY_CONFIRMED because holdout classification is PSEUDO_CONFIRMATORY

### HYP-VOL
- Status: **PARTIALLY_CONFIRMED**
- Mean Val IC: 0.0118
- Exceeds IC Threshold: True
- Exceeds Null: True
- Temporal Stability: True
- Positive Window Fraction: 1.00
- Downgrade Reason: Downgraded from CONFIRMED to PARTIALLY_CONFIRMED because holdout classification is PSEUDO_CONFIRMATORY

### HYP-MAC
- Status: **PARTIALLY_CONFIRMED**
- Mean Val IC: 0.0197
- Exceeds IC Threshold: True
- Exceeds Null: True
- Temporal Stability: True
- Positive Window Fraction: 1.00
- Downgrade Reason: Downgraded from CONFIRMED to PARTIALLY_CONFIRMED because holdout classification is PSEUDO_CONFIRMATORY

### HYP-XSEC
- Status: **PARTIALLY_CONFIRMED**
- Mean Val IC: 0.0264
- Exceeds IC Threshold: True
- Exceeds Null: True
- Temporal Stability: True
- Positive Window Fraction: 1.00
- Downgrade Reason: Downgraded from CONFIRMED to PARTIALLY_CONFIRMED because holdout classification is PSEUDO_CONFIRMATORY

## Critical Limitation

All confirmatory tests are **PSEUDO_CONFIRMATORY** because no genuinely
untouched holdout data exists. Phase 18 consumed all available data through
2026-06-30. Results indicate protocol-locked re-evaluation, not true
out-of-sample confirmation.

## Next Steps

Wait for new data (post 2026-06-30) to perform genuine out-of-sample
confirmation before any deployment consideration.