"""Corporate-action reconciliation for ingested market data.

Phase 3 rule: flag first, investigate second, never auto-fix. This module
cross-checks normalized bars against the provider's own split/dividend
events and reports:

- splits with no nearby price continuity (suspicious adjustment)
- discontinuities with no corporate action nearby (suspicious data)
- implausible single-day moves or volume spikes (possible corrupt prints)

Findings are warnings; they land in the manifest so every later consumer
can see what was investigated.

Split expectation depends on the adjustment basis recorded in the bars:
a "split_adjusted" series must be CONTINUOUS across the ex-date
(close_after ~= close_before), while an unadjusted series must jump by
the split ratio (close_after ~= close_before * ratio). Asking a
split-adjusted series to jump by the ratio is the classic inverted check
that turns every real split into a false finding.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

import polars as pl

MOVE_THRESHOLD = 0.25        # |overnight move| above this is suspicious
EVENT_LOOKBACK_DAYS = 5      # corporate action accepted within this window
SPLIT_DEVIATION_TOLERANCE = 0.05
VOLUME_SPIKE_FACTOR = 10.0   # volume above this x the series median is suspicious
VOLUME_MEDIAN_WINDOW = 60    # bars used for the volume median
_ADJUSTED_LABELS = frozenset({"split_adjusted", "dividend_and_split_adjusted", "fully_adjusted"})


@dataclass
class Finding:
    instrument_id: str
    code: str
    message: str
    event_date: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "instrument_id": self.instrument_id,
            "code": self.code,
            "message": self.message,
            "event_date": self.event_date,
        }


@dataclass
class ReconciliationReport:
    findings: list[Finding] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "finding_count": len(self.findings),
            "findings": [f.to_dict() for f in self.findings],
        }


def reconcile_market(
    bars: pl.DataFrame, events: pl.DataFrame
) -> ReconciliationReport:
    report = ReconciliationReport()
    if bars.height == 0:
        return report

    if "adjustment" in bars.columns:
        adjusted = bars["adjustment"].drop_nulls().is_in(list(_ADJUSTED_LABELS)).all()
    else:
        adjusted = False

    for inst_id in bars["instrument_id"].unique().to_list():
        one = bars.filter(pl.col("instrument_id") == inst_id).sort("trade_date")
        moves = (
            one.with_columns((pl.col("close") / pl.col("close").shift(1) - 1).alias("overnight"))
            .filter(pl.col("overnight").abs() > MOVE_THRESHOLD)
        )
        _check_volume_spikes(one, report, inst_id)
        ev = (
            events.filter(pl.col("instrument_id") == inst_id)
            if events.height else pl.DataFrame(schema=events.schema)
        )
        for row in moves.iter_rows(named=True):
            d = row["trade_date"]
            nearby = 0
            if ev.height:
                ev_dates = ev.filter(pl.col("ts").dt.date() >= d - timedelta(days=EVENT_LOOKBACK_DAYS))
                nearby = ev_dates.filter(pl.col("ts").dt.date() <= d).height
            code = "explained_by_corporate_action" if nearby else "unexplained_discontinuity"
            explanation = (
                f"near corporate action ({nearby} events)"
                if nearby
                else f"no corporate action within {EVENT_LOOKBACK_DAYS} days"
            )
            report.findings.append(
                Finding(
                    instrument_id=inst_id,
                    code=code,
                    message=(
                        f"overnight move {row['overnight'] * 100:.1f}% on {d.isoformat()} "
                        f"({explanation})"
                    ),
                    event_date=d.isoformat(),
                )
            )

        if ev.height:
            for event in ev.iter_rows(named=True):
                if event["kind"] != "splits":
                    continue
                d = event["ts"].date()
                before = one.filter(pl.col("trade_date") < d).tail(1)
                after = one.filter(pl.col("trade_date") > d).head(1)
                if before.height and after.height:
                    b, a = before["close"][0], after["close"][0]
                    observed_ratio = a / b
                    # adjusted series: continuous (a ~= b); unadjusted: the
                    # price drops by the split ratio (a ~= b / ratio)
                    expected_ratio = 1.0 if adjusted else 1.0 / event["ratio"]
                    deviation = abs(observed_ratio / expected_ratio - 1) if expected_ratio else 1.0
                    if deviation > SPLIT_DEVIATION_TOLERANCE:
                        expected_close = b * expected_ratio
                        report.findings.append(
                            Finding(
                                instrument_id=inst_id,
                                code="split_adjustment_inconsistency",
                                message=(
                                    f"split {event['ratio']:.2f} on {d.isoformat()}: "
                                    f"expected close {expected_close:.4f} "
                                    f"({'continuous' if adjusted else 'ratio-scaled'}), "
                                    f"got {a:.4f} ({deviation * 100:.1f}% off)"
                                ),
                                event_date=d.isoformat(),
                            )
                        )
    return report


def _check_volume_spikes(
    one: pl.DataFrame, report: ReconciliationReport, inst_id: str
) -> None:
    """Flag volume prints far above their own trailing baseline.

    Each bar is compared against the median of the 60 bars BEFORE it, so
    level shifts (e.g. era-wide volume changes) are not misread as spikes.
    Consecutive flagged days are collapsed into a single finding per run,
    so a sustained anomaly yields one warning, not hundreds.

    Known limitation: a basis change that affects the WHOLE series (e.g.
    Yahoo's inflated AAPL prints before 1998) has no clean in-series
    baseline, so it is not detectable from within one series; it shows up
    as discontinuities and stays an open investigation item.
    """
    if "volume" not in one.columns or one.height == 0:
        return
    one = one.sort("trade_date").with_columns(
        pl.col("volume")
        .rolling_median(window_size=VOLUME_MEDIAN_WINDOW, min_samples=30)
        .shift(1)
        .alias("baseline")
    )
    fallback = float(one["volume"].drop_nulls().median())
    rows = [
        {
            "trade_date": r["trade_date"],
            "ratio": (
                float(r["volume"]) / float(r["baseline"])
                if r["baseline"] is not None and r["baseline"] > 0
                else (
                    float(r["volume"]) / fallback
                    if fallback and fallback > 0 and r["volume"] is not None
                    else None
                )
            ),
        }
        for r in one.iter_rows(named=True)
    ]
    rows = [r for r in rows if r["ratio"] is not None]
    runs: list[list[dict]] = []
    for row in rows:
        if row["ratio"] > VOLUME_SPIKE_FACTOR:
            if runs and (row["trade_date"] - runs[-1][-1]["trade_date"]).days <= 3:
                runs[-1].append(row)
            else:
                runs.append([row])
    for run in runs:
        if not run:
            continue
        ratios = [r["ratio"] for r in run]
        first, last = run[0]["trade_date"], run[-1]["trade_date"]
        peak = run[ratios.index(max(ratios))]
        when = (
            f"{first.isoformat()}..{last.isoformat()}"
            if len(run) > 1
            else peak["trade_date"].isoformat()
        )
        report.findings.append(
            Finding(
                instrument_id=inst_id,
                code="volume_spike",
                message=(
                    f"volume up to {max(ratios):.0f}x the trailing "
                    f"{VOLUME_MEDIAN_WINDOW}-day median on {when} "
                    f"({len(run)} consecutive day(s)) - possible corrupt print"
                ),
                event_date=peak["trade_date"].isoformat(),
            )
        )