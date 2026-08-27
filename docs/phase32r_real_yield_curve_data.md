# Phase 32-R: Real Yield Curve Data Acquisition + Point-in-Time Validation

**Date:** 2026-08-27T08:07:11.699586+00:00
**Phase:** 32-R

---

## 1. What Data Was Evaluated

8 Treasury yield series from FRED:
- DGS3MO (3-Month Treasury)
- DGS1 (1-Year Treasury)
- DGS2 (2-Year Treasury)
- DGS5 (5-Year Treasury)
- DGS10 (10-Year Treasury)
- DGS30 (30-Year Treasury)
- T10Y2Y (10Y-2Y Spread)
- T10Y3M (10Y-3M Spread)

## 2. What Was Accepted

All 8 series ACCEPTED:
- All are PIT_NATIVE (published same day at 16:30 ET)
- All have sufficient historical coverage (1982-present minimum)
- All have minor revision risk only

## 3. What Was Rejected

None.

## 4. PIT Limitations

- Publication: Daily at 16:30 ET
- Decision timestamp: Next trading day 09:30 ET
- Availability before decision: YES
- Revision risk: LOW (1-2 days)
- Vintage treatment: NOT required

## 5. Revision Limitations

- Minor revisions possible within 1-2 days
- Impact: Negligible for daily research
- Vintage treatment: Not required

## 6. Yield Curve Infrastructure Status

### Real Data Directory
data/normalized/macro/fred_treasury/

### Features Specified
12 features across 5 categories:
- LEVEL: 2 features (10Y, 2Y)
- SLOPE: 3 features (10Y2Y, 10Y3M, 30Y5Y)
- CURVATURE: 1 feature
- CHANGE: 4 features (5D, 10D, 20D, slope change)
- REGIME: 2 features (z-score, steepener)

## 7. Sector x Macro Infrastructure Status

Deferred to future phase.

## 8. Remaining Data Gaps

None for yield curve data.

## 9. What Phase 33-R Is Allowed to Test

- All 12 yield curve features from REAL FRED data
- Baseline momentum features
- Incremental predictive value

## 10. What Phase 33-R Is NOT Allowed to Test

- OOS targets or predictions
- New unregistered features
- Features from simulated data
- Portfolio construction or backtesting

---

**Verdict:** A
**Gate:** GREEN
**Data Status:** DATA_READY
**Next Step:** Phase 33-R — Yield Curve / Term Structure Re-Exploration Using Real Data (after approval)
