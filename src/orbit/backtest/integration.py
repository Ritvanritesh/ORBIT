"""Phase 7 integration with the Phase 4/5/6 research infrastructure.

Phase 4 (temporal truth):
  - every signal's `decision_time` must equal `session_close_utc(signal
    session)` - the canonical EOD availability instant. Anything else is
    refused: a backtest that accepts arbitrary timestamps cannot prove it
    respects strict publication-as-of truth.
  - the experiment's `temporal_config.config_digest` (Phase 4 identity) is
    carried into the run manifest, so a run is traceable to the temporal
    contract it ran under.

Phase 5 (predicted / realized / executed are kept strictly apart):
  - `SignalEvent.signal_metric` is the predicted value (a Phase 5
    contract's label definition or a heuristic);
  - `FillEvent.price` is what the simulation executed (with explicit
    spread + slippage cost records);
  - the realized label outcome is computed by the Phase 5 `LabelEngine`
    under the pinned `LabelContract`/version, never by the backtester;
  - `predicted_vs_executed()` / `realized_outcome()` expose the three
    columns side by side - the research comparison, not the simulation.

Phase 6 (experiment lifecycle):
  - `run_backtest_experiment()` drives one experiment through its full
    lifecycle: lineage validation (dataset snapshots, cost model, label
    contract, temporal digest, seed), `mark_running` with the executing
    code/config hashes, the run itself, `complete`, artifact attachment
    (the auditable event stream + manifest), and the single immutable
    result record.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import polars as pl

from orbit.backtest.backtester import Backtester
from orbit.backtest.config import BacktestConfig, backtest_code_hash
from orbit.backtest.clock import MarketEventClock
from orbit.backtest.result import BacktestResult
from orbit.experiments.service import ExperimentService, ResultKind
from orbit.labels.contract import LabelContract
from orbit.labels.engine import LabelEngine
from orbit.temporal.times import session_close_utc


# ------------------------------------------------------------- Phase 4

def validate_signal_temporality(signals: Any) -> list[dict[str, Any]]:
    """Normalize signals and refuse any signal whose decision_time is not
    exactly the session close of its signal session (Phase 4 strict
    publication-as-of semantics)."""
    rows = MarketEventClock.normalize_signals(signals)
    for r in rows:
        expected = session_close_utc(r["signal_session"])
        actual = r["decision_time"]
        if actual != expected:
            raise ValueError(
                f"signal {r['signal_id']} for {r['instrument_id']} on "
                f"{r['signal_session']}: decision_time must be exactly the "
                f"session close ({expected}), got {actual} - a signal "
                "generated at any other instant cannot claim strict "
                "publication-as-of truth"
            )
    return rows


def temporal_digest_to_ref(config_digest: str) -> dict[str, Any]:
    """The Phase 4 temporal identity to carry into the run manifest."""
    return {
        "engine_version": "v1.0.0",
        "config_digest": config_digest,
        "as_of_semantics": "strict_publication_less_than_asof",
        "timezone": "America/New_York",
    }


# ------------------------------------------------------------- Phase 5

def predicted_vs_executed(result: BacktestResult) -> pl.DataFrame:
    """One row per executed signal: predicted metric (signal_metric) next
    to the executed outcome (quantity, notional, fill price, fee). The
    predicted and the executed never share a column."""
    rows = []
    fills_by_signal: dict[str, list] = {}
    for f in result.fills:
        fills_by_signal.setdefault(f.signal_id, []).append(f)
    for s in result.signals:
        fills = fills_by_signal.get(s.signal_id, [])
        executed_quantity = sum(f.filled_quantity for f in fills)
        executed_notional = sum(f.filled_quantity * f.price for f in fills)
        total_fee = sum(f.fee for f in fills)
        vwap = (
            executed_notional / executed_quantity
            if executed_quantity > 0
            else None
        )
        rows.append(
            {
                "signal_id": s.signal_id,
                "instrument_id": s.instrument_id,
                "signal_session": s.signal_session,
                "direction": s.direction.value,
                "target": s.target,
                "predicted_metric": s.signal_metric,
                "executed_quantity": executed_quantity,
                "executed_notional": executed_notional,
                "executed_vwap": vwap,
                "total_fee": total_fee,
            }
        )
    return pl.DataFrame(rows)


def realized_outcome(
    result: BacktestResult,
    label_engine: LabelEngine,
    contract: LabelContract,
    label_version: str | None = None,
) -> pl.DataFrame:
    """The Phase 5 realized label outcome for each executed signal,
    computed by the pinned label engine - never by the backtester.

    Returns one row per signal that has fills, joined with the predicted
    metric and the executed VWAP (the research comparison table).
    """
    predicted = predicted_vs_executed(result)
    executed = predicted.filter(pl.col("executed_quantity") > 0)
    if not executed.height:
        return pl.DataFrame(
            schema={
                "signal_id": pl.Utf8,
                "instrument_id": pl.Utf8,
                "signal_session": pl.Date,
                "predicted_metric": pl.Float64,
                "executed_vwap": pl.Float64,
                "realized_metric": pl.Float64,
                "label_version": pl.Utf8,
            }
        )
    decisions = [
        {
            "instrument_id": r["instrument_id"],
            "session": r["signal_session"],
            "decision_id": r["signal_id"],
            "decision_time": session_close_utc(r["signal_session"]),
        }
        for r in executed.to_dicts()
    ]
    labels = label_engine.compute(contract, decisions)
    realized = labels.rename(
        {"decision_id": "signal_id", "outcome_value": "realized_metric"}
    ).select(["signal_id", "realized_metric"])
    if label_version is not None:
        realized = realized.with_columns(
            pl.lit(label_version).alias("label_version")
        )
    out = executed.join(realized, on="signal_id", how="left")
    return out.select(
        [
            "signal_id",
            "instrument_id",
            "signal_session",
            "predicted_metric",
            "executed_vwap",
            "realized_metric",
        ]
        + (["label_version"] if label_version is not None else [])
    )


# ------------------------------------------------------------- Phase 6

def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def run_backtest_experiment(
    service: ExperimentService,
    experiment_id: str,
    *,
    config: BacktestConfig,
    bars: pl.DataFrame,
    events: pl.DataFrame | None = None,
    volume_basis: str | None = None,
    signals: Any,
    artifacts_dir: str | Path,
    label_engine: LabelEngine | None = None,
    label_contract: LabelContract | None = None,
) -> tuple[BacktestResult, dict[str, Any]]:
    """Run a Phase 7 backtest as one registered Phase 6 experiment.

    Lineage is validated against the registered experiment before anything
    runs (a backtest with mismatched cost assumptions, seed, label contract
    or temporal digest is a different experiment and is refused). The
    experiment transitions DRAFT/RUNNING -> RUNNING (code/config hashes
    pinned) -> COMPLETED, the auditable artifacts are attached, and the
    single immutable result is recorded.

    Returns (BacktestResult, experiment metadata dict).
    """
    exp = service.get(experiment_id)
    if exp is None:
        raise ValueError(f"unknown experiment: {experiment_id}")

    # ---- Phase 4 gate first: every signal must sit exactly at its session
    # ---- close (strict publication-as-of truth). This runs before any
    # ---- lifecycle transition: a temporally invalid input is not an
    # ---- experiment failure, it is an invalid request.
    validate_signal_temporality(signals)

    # ---- lineage validation (Phase 6 identity, before any work)
    if config.seed != exp.seed:
        raise ValueError(
            f"experiment {experiment_id} pins seed {exp.seed}, the backtest "
            f"uses {config.seed} - a different seed is a different experiment"
        )
    if exp.cost_model_id is not None and config.costs != config.costs.from_cost_model(
        exp.cost_model
    ):
        raise ValueError(
            f"experiment {experiment_id} pins cost_model_id "
            f"{exp.cost_model_id} ({exp.cost_model}), the backtest costs "
            f"{config.costs} do not match - a different cost assumption is a "
            "different experiment"
        )
    if not exp.dataset_snapshot_ids:
        raise ValueError(
            f"experiment {experiment_id} has no dataset_snapshot_ids; Phase 6 "
            "requires the exact Phase 3 snapshots consumed (never 'latest "
            "data')"
        )

    manifest_meta: dict[str, Any] = {
        "experiment_id": experiment_id,
        "hypothesis_id": exp.hypothesis_id,
        "dataset_snapshot_ids": exp.dataset_snapshot_ids,
        "feature_refs": [
            {
                "feature_id": r.feature_id,
                "feature_version": r.feature_version,
                "transformation": r.transformation,
            }
            for r in exp.features.feature_refs
        ]
        or [
            {"feature_id": f, "feature_version": exp.features.feature_version}
            for f in exp.features.feature_names
        ],
        "model": exp.model.model_dump(mode="json"),
        "label_id": exp.label_id,
        "label_version": exp.label_version,
        "temporal_config_digest": (
            exp.temporal_config.config_digest
            if exp.temporal_config is not None
            else None
        ),
        "cost_model_id": exp.cost_model_id,
    }

    code_hash = backtest_code_hash()
    config_hash = config.config_hash()
    service.mark_running(
        experiment_id,
        code_hash=code_hash,
        config_hash=config_hash,
        note=f"Phase 7 backtest ({config.sizing.value} sizing, "
        f"execution={config.execution.execution_price.value}/delay="
        f"{config.execution.execution_delay})",
    )

    result: BacktestResult | None = None
    try:
        backtester = Backtester(
            config=config,
            universe=sorted(bars["instrument_id"].unique().to_list()),
            dataset_snapshot_ids=exp.dataset_snapshot_ids,
            code_hash=code_hash,
            experiment_id=experiment_id,
            hypothesis_id=exp.hypothesis_id,
            feature_refs=manifest_meta["feature_refs"],
            model=manifest_meta["model"],
            label_id=exp.label_id,
            label_version=exp.label_version,
            temporal_config_digest=manifest_meta["temporal_config_digest"],
            cost_model_id=exp.cost_model_id,
        )
        result = backtester.run(
            bars,
            signals,
            events_artifact=events,
            volume_basis=volume_basis,
        )
    except Exception as exc:  # noqa: BLE001 - research record, never silent
        service.fail(
            experiment_id,
            note=f"Phase 7 backtest failed: {type(exc).__name__}: {exc}",
        )
        raise

    violations = result.invariant_violations()
    if violations:
        service.fail(
            experiment_id,
            note="Phase 7 backtest completed but the accounting invariants "
            f"were violated: {'; '.join(violations)}",
        )
        raise RuntimeError(
            f"backtest {result.run_id} violated accounting invariants: "
            + "; ".join(violations)
        )

    # ---- artifacts FIRST (the fallible file I/O happens while the
    # ---- experiment is still RUNNING), then complete, then the DB-side
    # ---- attachment and the single immutable result: a failure after
    # ---- `complete` can no longer leave a COMPLETED experiment without
    # ---- its artifacts, and the artifact files never go stale
    out_dir = Path(artifacts_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    events_path = out_dir / f"{result.run_id}_events.jsonl"
    manifest_path = out_dir / f"{result.run_id}_manifest.json"
    summary = result.summary()
    try:
        result.to_jsonl(events_path)
        # the on-disk manifest is byte-deterministic: created_at is part of
        # the in-memory identity bookkeeping, not of the scientific content
        manifest_path.write_text(
            result.manifest.model_dump_json(
                indent=2, exclude={"created_at"}
            ),
            encoding="utf-8",
        )
    except Exception as exc:  # noqa: BLE001 - research record, never silent
        service.fail(
            experiment_id,
            note=f"Phase 7 backtest {result.run_id} failed while writing its "
            f"artifact files: {type(exc).__name__}: {exc}",
        )
        raise

    try:
        service.complete(
            experiment_id,
            note=f"Phase 7 backtest {result.run_id} completed; "
            f"final equity {summary['final_equity']:.2f}",
        )
        service.attach_artifact(
            experiment_id,
            kind="backtest_event_stream_jsonl",
            path=str(events_path),
            checksum=_sha256_file(events_path),
        )
        service.attach_artifact(
            experiment_id,
            kind="backtest_manifest_json",
            path=str(manifest_path),
            checksum=_sha256_file(manifest_path),
        )

        # ---- the single immutable result (violations were checked above,
        # ---- so the only honest kind here is SUPPORTED)
        service.record_result(
            experiment_id,
            kind=ResultKind.SUPPORTED,
            summary=(
                f"Phase 7 backtest {result.run_id}: final equity "
                f"{summary['final_equity']:.2f}, total return "
                f"{summary['total_return']:.4%}, {summary['n_fills']} fills / "
                f"{summary['n_rejects']} rejections over {summary['n_signals']} "
                "signals"
            ),
            metrics={
                "run_id": result.run_id,
                "final_equity": summary["final_equity"],
                "total_return": summary["total_return"],
                "total_pnl": summary["total_pnl"],
                "turnover": summary["turnover"],
                "total_fees": summary["total_fees"],
                "n_signals": summary["n_signals"],
                "n_orders": summary["n_orders"],
                "n_fills": summary["n_fills"],
                "n_rejects": summary["n_rejects"],
                "config_hash": config_hash,
                "code_hash": code_hash,
            },
        )
    except Exception as exc:  # noqa: BLE001 - research record, never silent
        # the experiment is already COMPLETED at this point (the lifecycle
        # forbids completed -> failed); the original error must never be
        # masked by the transition error, so the fail attempt is best-effort
        try:
            service.fail(
                experiment_id,
                note=f"Phase 7 backtest {result.run_id} failed after "
                f"completion (artifacts/result): {type(exc).__name__}: {exc}",
            )
        except Exception:  # noqa: BLE001 - already terminal, never mask
            pass
        raise

    return result, {"experiment_id": experiment_id, "run_id": result.run_id}


__all__ = [
    "predicted_vs_executed",
    "realized_outcome",
    "run_backtest_experiment",
    "temporal_digest_to_ref",
    "validate_signal_temporality",
]