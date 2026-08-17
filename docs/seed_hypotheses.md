# Seed Hypotheses

Three falsifiable hypotheses pre-registered in Phase 1, before any feature
exploration or data acquisition. Source of truth: `hypotheses/seeds.py`
(validated against `HypothesisSpec`). The summaries here are informative;
the code is binding.

## H-001: Cross-sectional 12-1 price momentum

**Statement.** Ranks of trailing 12-month return excluding the most recent
month in the ORBIT liquid-equity universe predict the cross-section of 5-day
forward excess returns after costs.

- Universe: liquid equity 50-100
- Label: 5-day forward excess return vs SPY
- Features: momentum family only
- Baselines: equal-weight top 50, SPY hold
- Economic evidence: OOS rank IC >= 0.03, after-cost annual excess >= 3%,
  >= 3 regimes, >= 4 walk-forward windows
- Falsified if: any threshold missed, or no incremental lift over baseline,
  or no momentum family lift over equal-weight.

## H-002: Low realized volatility anomaly

**Statement.** Volatility-scaled exposure to low-realized-volatility liquid US
equities produces higher risk-adjusted after-cost returns than an equal-weight
exposure to the same universe.

- Universe: liquid equity 50-100
- Label: 21-day forward return divided by trailing-21-day realized vol,
  minus the SPY equivalent (risk-adjusted excess, point-in-time)
- Features: volatility, liquidity families
- Baselines: equal-weight top 50, SPY hold
- Economic evidence: OOS rank IC >= 0.02, after-cost risk-adjusted excess
  >= 2%, >= 3 regimes, >= 4 walk-forward windows
- Falsified if: effect vanishes under volatility targeting, or thresholds
  missed.

## H-003: Post-earnings-announcement drift (PEAD) via point-in-time fundamentals

**Statement.** Standardized earnings surprise computed strictly from
point-in-time SEC EDGAR/XBRL fundamentals predicts the cross-section of 5-day
forward excess returns in liquid US equities.

- Universe: liquid equity 50-100
- Label: 5-day forward excess return vs SPY, measured from the trading day
  after the earliest point-in-time publication timestamp
- Features: fundamentals family only
- Baselines: equal-weight top 50, SPY hold, momentum baseline
- Economic evidence: OOS rank IC >= 0.02, after-cost annual excess >= 2%,
  >= 2 regimes, >= 4 walk-forward windows
- Falsified if: thresholds missed, no lift over momentum baseline, or any
  synthetic future-leak test catches use of post-publication/revised data.
- Leakage class: `future_publication` (the entire point of the test is that
  point-in-time discipline removes it).

## Registration status

The three seeds are registered (frozen) by `hypotheses/seeds.py::register_seeds()`.
After registration, falsification criteria and evidence thresholds may not
change without a new hypothesis version.
