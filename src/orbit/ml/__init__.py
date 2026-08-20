"""Phase 9 - Baseline ML Benchmark.

Phase 9 tests whether learnable numeric structure exists beyond the
transparent rules of Phases 1-8. The benchmark follows the ORBIT roadmap:

    FEATURE SNAPSHOT -> MODEL -> SIGNAL -> BACKTEST

Architecture (all canonical):
  - data:      existing Phase 3 snapshot (DS-000004, 20-symbol dev universe)
  - features:  FS-001 v1, 8 documented numeric features computed from
               completed bars only (strict point-in-time boundary)
  - labels:    LAB-004 v1, 5-session forward total return (DECISION_INSTANT),
               registered through the Phase 5 LabelVersionRegistry
  - splits:    strict chronological train / validation / test with
               outcome-window purging at every split boundary
  - models:    Ridge, Lasso, Logistic, Random Forest, XGBoost - all with
               pre-registered small hyperparameter grids and fixed seeds
  - evaluation: OOS IC, rank IC, calibration (validation-only fit), ECE,
               Brier, after-cost backtest via the canonical Phase 7 engine
               with the Phase 8 cost model (CM-001)
  - controls:  Phase 8 baseline strategies executed on the real dataset
               through the identical backtest path
"""

from __future__ import annotations

__version__ = "1.0.0"