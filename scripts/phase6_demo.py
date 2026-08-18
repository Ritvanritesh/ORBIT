"""Phase 6 end-to-end demo: register, run, decide, reproduce, audit.

Run from the repository root:

    python scripts/phase6_demo.py
"""

from __future__ import annotations

import sys
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from orbit.experiments import (
    Decision,
    ExperimentRegistry,
    ExperimentService,
    ResultKind,
    temporal_config_digest,
)
from orbit.labels.seeds import build_seed_label_registry
from orbit.schemas.experiment import ExperimentSpec, FeatureRef, TemporalConfigRef, WindowSpec
from orbit.temporal.contracts import load_temporal_contract
from hypotheses.seeds import register_seeds


def _now() -> datetime:
    return datetime.now(timezone.utc)


def main() -> None:
    hypotheses = register_seeds()          # Phase 1 registry
    labels = build_seed_label_registry()   # Phase 5 registry
    temporal = load_temporal_contract("configs/temporal.json")  # Phase 4

    workdir = Path(tempfile.mkdtemp(prefix="orbit-phase6-demo-"))
    db = workdir / "experiments.duckdb"
    print(f"ledger: {db}")

    service = ExperimentService(
        registry=ExperimentRegistry(db_path=str(db)),
        hypothesis_registry=hypotheses,
        label_registry=labels,
        temporal_contract=temporal,
    )

    spec = ExperimentSpec(
        experiment_id="EXP-00001",
        hypothesis_id="H-001",
        title="Momentum baseline with pinned label version",
        datasets=["market_daily_v1"],
        dataset_snapshot_ids=["DS-000001", "DS-000002"],
        features={
            "feature_names": ["ret_12m_1m"],
            "feature_version": "v1",
            "feature_refs": [
                FeatureRef(feature_id="FEAT-001", feature_version="v1", transformation="xform-v1")
            ],
        },
        model={"family": "linear", "hyperparameters": {"alpha": 0.1}},
        windows=WindowSpec(
            train_start=date(2020, 1, 1),
            train_end=date(2023, 12, 31),
            val_start=date(2024, 1, 1),
            val_end=date(2024, 12, 31),
            test_start=date(2025, 1, 1),
            test_end=date(2025, 12, 31),
        ),
        label_id="LAB-001",
        label_version="v1",
        cost_model_id="CM-001",
        temporal_config=TemporalConfigRef(
            engine_version=temporal.engine_version,
            config_digest=temporal_config_digest(temporal),
        ),
        seed=42,
        researcher="demo-driver",
        evaluation_protocol="walkforward_v1",
    )

    registered = service.register(spec, registered_at=_now())
    print(f"registered {registered.experiment_id} "
          f"(trial {registered.trial_number}, content {registered.content_hash()[:16]}...)")

    service.mark_running(
        "EXP-00001",
        code_hash="demo-code-v1" + "0" * 48,   # placeholder for a real code digest
        config_hash="demo-config-v1" + "0" * 46,
    )
    service.complete("EXP-00001")

    service.record_result(
        "EXP-00001",
        kind=ResultKind.SUPPORTED,
        summary="OOS rank IC 0.041; after-cost annual excess 4.2%",
        metrics={"oos_rank_ic": 0.041, "after_cost_excess": 0.042},
    )
    service.attach_artifact(
        "EXP-00001",
        kind="metrics",
        path="artifacts/EXP-00001/metrics.json",
        checksum="demo" + "0" * 60,
    )
    service.record_decision(
        "EXP-00001",
        decision=Decision.PROMOTED,
        reason="After-cost annual excess 4.2% exceeds Gate-C threshold 3%.",
        policy_version="Gate-C-v1",
        decision_maker="demo-driver",
    )
    print("promoted via record_decision()")

    spec_obj = service.reproduction_spec("EXP-00001")
    print(f"reproduction digest: {spec_obj.reproduction_digest[:16]}... "
          f"(verified: {spec_obj.verify_digest()})")
    print(f"resolved datasets: {[d['snapshot_id'] for d in spec_obj.datasets]}")
    print(f"resolved label: {spec_obj.label['label_id']} {spec_obj.label['version']}")

    report = service.validate_invariants()
    print(f"invariants: ok={report['ok']} "
          f"experiments={report['experiments']} orphans={report['orphan_counts']}")

    print(f"children of EXP-00001: {[e.experiment_id for e in service.children('EXP-00001')]}")
    print(f"transitions: {[(t['from_status'], t['to_status']) for t in service.transitions('EXP-00001')]}")
    print(f"trial count for H-001: {service.count_trials('H-001')}")


if __name__ == "__main__":
    main()