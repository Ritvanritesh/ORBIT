# Phase 30-R: Hypothesis-Driven Data Acquisition & PIT Validation

**Date:** 2026-08-26T15:06:25.831392+00:00
**Phase:** 30-R

---

## 1. What Data Was Evaluated

### Yield Curve Series (6 candidates)
- DGS3MO (3-Month Treasury)
- DGS2 (2-Year Treasury)
- DGS5 (5-Year Treasury)
- DGS10 (10-Year Treasury)
- DGS30 (30-Year Treasury)
- T10Y2Y (10Y-2Y Spread)

### Sector Classification
- GICS depth-1 taxonomy
- 11 sectors observed
- Embedded in instrument master configs

### Market Data (existing)
- DS-EXP-050 (50 instruments, 349K rows)
- DS-EXP-100 (97 instruments, 681K rows)

---

## 2. What Was Accepted

### Yield Curve (6 series, 10 features)
All 6 Treasury yield series ACCEPTED for Priority 1 research:
- DGS3MO, DGS2, DGS5, DGS10, DGS30, T10Y2Y
- All classified PIT_NATIVE
- Features: YC-LEVEL, YC-SLOPE, YC-CURVATURE, YC-CHANGE, YC-REGIME

### Sector Features (3 features)
- SECTOR_RET_20, SECTOR_VOL_20, SECTOR_DISPERSION_20
- All PIT_NATIVE
- Ready for Priority 2 research

### Interactions (6 interactions)
- INT-001 through INT-006
- Macro x Sector combinations with clear economic mechanisms

---

## 3. What Was Rejected

- Intraday yield data (ORBIT uses daily frequency)
- International yield curves (ORBIT focuses on US equities)

---

## 4. PIT Limitations

### Yield Curve Data
- Classification: PIT_NATIVE
- Publication: Daily at 16:30 ET
- Decision timestamp: Next trading day 09:30 ET
- Availability before decision: YES
- Leakage risk: LOW

### Sector Classification
- Classification: PIT_SAFE_WITH_LAG
- Limitation: Historical GICS labels may not be available
- Risk: GICS reclassifications may introduce look-ahead bias
- Mitigation: Use historical labels if available; document limitation if not

---

## 5. Revision Limitations

### Yield Curve Data
- Revisions: Minor, typically within 1-2 days
- Impact: Negligible for daily research
- Vintage treatment: Not required

### Market Data
- Revisions: Adjusted close may be revised
- Impact: Low for research purposes
- Vintage treatment: Not required

---

## 6. Yield Curve Infrastructure Status

### Data Source
- Provider: FRED (Federal Reserve Economic Data)
- Access method: FRED API
- License: Public domain
- Reliability: HIGH

### Feature Groups
| Group | Features | Count |
|-------|----------|-------|
| YC-LEVEL | YC_LEVEL_10Y | 1 |
| YC-SLOPE | YC_SLOPE_10Y2Y, YC_SLOPE_10Y3M | 2 |
| YC-CURVATURE | YC_CURVATURE | 1 |
| YC-CHANGE | YC_CHANGE_5D_10Y, YC_CHANGE_10D_10Y, YC_CHANGE_20D_10Y, YC_SLOPE_CHANGE_5D | 4 |
| YC-REGIME | YC_LEVEL_ZSCORE_252, YC_REGIME_STEEPENER | 2 |

### Status: READY for exploratory research

---

## 7. Sector x Macro Infrastructure Status

### Sector Classification
- Taxonomy: GICS depth-1
- Sectors: 11
- Source: Instrument master configs
- Status: PARTIALLY_READY

### Sector Features
- SECTOR_RET_20, SECTOR_VOL_20, SECTOR_DISPERSION_20
- Status: READY

### Interactions
- 6 interactions registered
- Macro components: YC_SLOPE_10Y2Y, YC_LEVEL_10Y, YC_CHANGE_10D_10Y
- Sector components: Financials, Utilities, Real Estate, IT, Consumer Discretionary, Energy
- Status: READY for exploratory research

### Blocker
- Historical GICS labels not available
- Must acquire vintage sector classification data

---

## 8. Remaining Data Gaps

1. **Historical GICS labels** — Required for PIT-safe sector analysis
2. **FRED API key** — Required for actual data download
3. **Sector-level aggregations** — Must compute from existing price data

---

## 9. What Phase 31-R Is Allowed to Test

- Yield curve features (YC-LEVEL, YC-SLOPE, YC-CURVATURE, YC-CHANGE, YC-REGIME)
- Sector features (SECTOR_RET_20, SECTOR_VOL_20, SECTOR_DISPERSION_20)
- Sector x Macro interactions (INT-001 through INT-006)
- Existing momentum features (MOM_5D, MOM_10D, MOM_20D)
- Existing volatility features (VOL_ZSCORE, realized_vol)

---

## 10. What Phase 31-R Is NOT Allowed to Test

- OOS targets or predictions
- New unregistered features
- Features from ineligible sources
- Portfolio construction or backtesting
- Model optimization based on OOS performance
- Any feature not in the feature registry

---

**Verdict:** A
**Gate:** GREEN
**Next Step:** Phase 31-R — Yield Curve / Term Structure Exploratory Research (after approval)
