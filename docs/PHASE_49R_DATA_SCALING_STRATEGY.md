# Phase 49-R: Data Scaling Strategy

## Completion Status
- **Phase**: 49-R
- **Name**: DATA SCALING STRATEGY
- **Verdict**: A (DATA_SCALING_PLAN_READY)
- **Gate**: GREEN
- **Completed**: 2026-08-29T07:20:13.772580 UTC
- **Elapsed**: 0.4 seconds

---

## Current Dataset

| Metric | DS-EXP-050 | DS-EXP-100 |
|--------|-----------|-----------|
| Rows | 349374 | 680878 |
| Symbols | 50 | 97 |
| Date Range | 1996-08-21 to 2026-08-20 | 1996-08-21 to 2026-08-20 |
| Trading Days | 7547 | 7547 |
| Years | 30.0 | 30.0 |

**FRED Treasury**: 8 series, 113116 total rows

---

## Primary Data Gap

Current dataset lacks survivorship-corrected universe, diversified sector coverage, and critical macro regime indicators (VIX, credit spreads, monetary policy rate, inflation).

---

## Recommended Data Target

| Dimension | Minimum | Target | Ideal |
|-----------|--------:|-------:|------:|
| Symbols | 50 | 150 | 300 |
| Historical years | 10 | 15 | 20 |
| Frequency | Daily | Daily | Daily |
| Raw observations | 126,000 | 567,000 | 1,512,000 |
| Effective observations | ~25,000 | ~85,000 | ~180,000 |

---

## Model Readiness

| Scenario | Ridge | ElasticNet | HGB | LightGBM | MLP | TCN | Transformer |
|----------|-------|-----------|-----|---------|-----|-----|------------|
| Minimum | READY | READY | READY | READY | POSSIBLY | NOT | NOT |
| Practical | READY | READY | READY | READY | READY | POSSIBLY | NOT |
| Advanced | READY | READY | READY | READY | READY | READY | POSSIBLY |
| Ideal | READY | READY | READY | READY | READY | READY | READY |

---

## Acquisition Priority

### PRIORITY 1 (Required)
1. Survivorship-corrected universe (delisted securities)
2. Universe expansion to 150 symbols (sector-diversified)
3. Historical depth to 15 years
4. VIX index (daily, PIT_NATIVE)
5. SP500 index (daily, PIT_NATIVE)
6. Credit spreads BAA-AAA (daily, PIT_NATIVE)
7. FEDFUNDS (monthly, PIT_NATIVE)
8. CPI (monthly, PIT_SAFE_WITH_LAG)
9. T10YIE breakeven inflation (daily, PIT_NATIVE)

### PRIORITY 2 (Useful)
- Sector indices, UNRATE, INDPRO, historical market-cap

### PRIORITY 3 (Future)
- Market breadth, UMCSENT, GDP

### REJECT
- Fundamental data, alternative data, options data, intraday data

---

## Dataset Targets

### ORBIT_DATASET_TARGET_V1
- 150 symbols, 15 years, daily frequency
- 14 macro/market series
- Effective observations: ~85,000
- Supported models: Ridge, ElasticNet, HGB, LightGBM, MLP

### ORBIT_DATASET_TARGET_ADVANCED
- 300 symbols, 20 years, daily frequency
- All P1+P2 macro/market series
- Effective observations: ~180,000
- Supported models: All including TCN and Transformer

---

## Adversarial Testing
- 30/30 tests PASS (including DOCUMENTED_LIMITATION)
- 0 DETECTED, 0 BLOCKED

## Reproducibility
- 10/10 checks PASS

## Firewall
- OOS targets accessed: NO
- Confirmatory tests executed: NO
- Locked registrations modified: NO
- Historical artifacts modified: NO

---

## Next Allowed Step

PHASE 50-R HISTORICAL DATA EXPANSION + PIT / SURVIVORSHIP AUDIT

Do NOT automatically begin the next phase. Wait for user approval.
