"""The Temporal Truth Engine.

The engine answers the Phase 4 question:

    "Pretend it is exactly <as_of_time>. Show me everything ORBIT was
     allowed to know at that moment, and why."

Pipeline:

    normalized parquet artifacts
        -> adapters (attach timing facts, never decide)
        -> rules (vectorized availability evaluation)
        -> vintage resolution (latest version released before as_of)
        -> PointInTimeSnapshot (information set + audit trail + digest)

Design rules honored here:

  - the availability gate is publication time, NEVER ingestion time,
    NEVER event/period time alone, NEVER ts_utc for bars;
  - missing publication -> unavailable (no invented availability);
  - revised macro series without vintage history -> unavailable;
  - the engine never silently substitutes any value;
  - evaluation is vectorized polars, but every decision is also available
    per-record (`decide_record`) so tests can interrogate single rows;
  - evaluation is deterministic: same inputs + same as_of -> identical
    decision columns and identical snapshot digest.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import polars as pl

from orbit.ingestion.paths import normalized_dir
from orbit.temporal.adapters import (
    TIMING_SCHEMA,
    empty_timing_frame,
    fred_timing_frame,
    market_timing_frame,
    sec_timing_frame,
)
from orbit.temporal.contracts import TemporalContract, load_temporal_contract
from orbit.temporal.rules import AvailabilityDecision, RuleTrace, decide, trace_rule
from orbit.temporal.snapshot import PointInTimeSnapshot, TemporalSource
from orbit.temporal.times import DecisionCode, Timing, normalize_instant

_EMPTY_DATE = date(1900, 1, 1)


@dataclass(frozen=True)
class SourceInput:
    """One dataset snapshot plus the timing frame derived from it."""

    source: TemporalSource
    timing_frame: pl.DataFrame

    @property
    def snapshot_id(self) -> str:
        return self.source.snapshot_id


@dataclass
class Evaluation:
    """Result of evaluating one timing frame at one as_of."""

    as_of_time: datetime
    frame: pl.DataFrame  # timing columns + decision columns
    allowed: pl.DataFrame
    excluded: pl.DataFrame
    limitations: list[str] = field(default_factory=list)


def _next_day_expr() -> pl.Expr:
    return pl.col("publication_time").dt.date() + pl.duration(days=1)


# Decision columns appended by evaluate(), in append order. The collision
# handling of asof_join must treat them as part of the right-side universe:
# a left frame carrying e.g. its own `allowed` column must never collide
# with the right's decision columns either.
DECISION_COLUMNS: list[tuple[str, pl.DataType]] = [
    ("available_instant", pl.Datetime("us")),
    ("allowed", pl.Boolean),
    ("decision_code", pl.Utf8),
    ("warn_ingested_after_as_of", pl.Boolean),
    ("decision_detail", pl.Utf8),
]


class TemporalTruthEngine:
    """Point-in-time availability engine over the normalized data layer."""

    def __init__(
        self,
        registry: Any | None = None,
        *,
        data_root: Path | None = None,
        contract: TemporalContract | None = None,
        sources: list[TemporalSource] | None = None,
    ):
        self._registry = registry
        self._data_root = data_root
        self._contract = contract or load_temporal_contract()
        self._sources: list[TemporalSource] = sources or []

    # ------------------------------------------------------------ public API

    @property
    def contract(self) -> TemporalContract:
        return self._contract

    @property
    def engine_version(self) -> str:
        return self._contract.engine_version

    def sources(self) -> list[TemporalSource]:
        return list(self._sources)

    def set_sources(self, sources: list[TemporalSource]) -> "TemporalTruthEngine":
        self._sources = list(sources)
        return self

    def decide_record(self, timing: Timing, as_of: datetime | date | str) -> AvailabilityDecision:
        """Single-record availability decision (fixtures, tests, audits)."""
        return decide(timing, as_of)

    def trace_record(self, timing: Timing, as_of: datetime | date | str) -> RuleTrace:
        """Full rule trace for one record: every intermediate value."""
        return trace_rule(timing, as_of)

    def snapshot(
        self,
        as_of: datetime | date | str,
        sources: list[TemporalSource] | None = None,
    ) -> PointInTimeSnapshot:
        """The point-in-time snapshot: everything available at `as_of`."""
        t = normalize_instant(as_of)
        if t is None:
            raise ValueError("as_of_time is required")
        inputs = self._load_sources(sources if sources is not None else self._sources)
        if not inputs:
            raise ValueError("no sources configured; pass sources=[...] or call set_sources()")

        frames = [i.timing_frame for i in inputs]
        combined = pl.concat(frames) if len(frames) > 1 else frames[0]
        combined = combined.sort("record_id")

        evaluated = self.evaluate(combined, t)
        resolved = self._resolve_vintages(evaluated.frame)
        allowed = resolved.filter(pl.col("allowed"))
        excluded = resolved.filter(~pl.col("allowed"))

        limitations = self._limitations(excluded)
        return PointInTimeSnapshot(
            as_of_time=t,
            engine_version=self.engine_version,
            sources=[i.source for i in inputs],
            records=allowed,
            excluded=excluded,
            limitations=limitations,
        )

    def evaluate(self, frame: pl.DataFrame, as_of: datetime | date | str) -> Evaluation:
        """Vectorized evaluation of a timing frame at `as_of`."""
        t = normalize_instant(as_of)
        if t is None:
            raise ValueError("as_of_time is required")
        if frame.height == 0:
            empty = empty_timing_frame()
            empty = empty.with_columns(
                pl.lit(False).alias("allowed"),
                pl.lit("", dtype=pl.Utf8).alias("decision_code"),
                pl.lit("", dtype=pl.Utf8).alias("decision_detail"),
                pl.lit(None, dtype=pl.Datetime("us")).alias("available_instant"),
                pl.lit(False).alias("warn_ingested_after_as_of"),
            )
            return Evaluation(
                as_of_time=t, frame=empty, allowed=empty.filter(pl.col("allowed")),
                excluded=empty.filter(~pl.col("allowed")),
            )

        step1 = frame.with_columns(
            # NULL precision is treated as DATE (most conservative): a
            # record is never available BEFORE the next day when we do not
            # know how precisely its publication time is known. If it were
            # treated as an instant, available_instant could come out NULL
            # and the strict-boundary rejection below would silently
            # disarm (publication after as_of would be allowed).
            pl.when(
                pl.col("publication_precision").is_null()
                | (pl.col("publication_precision") == "date")
            )
            .then(_next_day_expr())
            .otherwise(pl.col("publication_time"))
            .alias("available_instant"),
        )
        # NULL-SAFETY: every operand is a strict boolean. A null comparison
        # (e.g. series_policy is null for market rows) would poison the OR
        # chain with NULL and silently drop rows from BOTH sides - the
        # worst kind of leak, because it looks like a correct rejection.
        reject_pit = (
            (pl.col("series_policy").fill_null("").str.to_lowercase() == "revised")
            & pl.col("vintage_date").is_null()
        )
        reject_missing_pub = pl.col("publication_time").is_null()
        reject_pub_after = (
            pl.col("available_instant").is_not_null()
            & (pl.col("available_instant") >= t)
        )
        reject_no_vintage = (
            pl.col("vintage_date").is_not_null() & (pl.col("vintage_date") >= t.date())
        )
        reject_event_after = (
            pl.col("event_time").is_not_null() & (pl.col("event_time") > t)
        )
        warn_ingested = (
            pl.col("ingestion_time").is_not_null() & (pl.col("ingestion_time") > t)
        )

        decision_code = (
            pl.when(reject_pit).then(pl.lit(DecisionCode.NOT_POINT_IN_TIME.value))
            .when(reject_missing_pub).then(pl.lit(DecisionCode.MISSING_PUBLICATION_TIME.value))
            .when(reject_pub_after).then(pl.lit(DecisionCode.PUBLICATION_AT_OR_AFTER_AS_OF.value))
            .when(reject_no_vintage).then(pl.lit(DecisionCode.NO_VINTAGE_AT_AS_OF.value))
            .when(reject_event_after).then(pl.lit(DecisionCode.EVENT_AFTER_AS_OF.value))
            .when(pl.col("vintage_date").is_not_null())
            .then(pl.lit(DecisionCode.ALLOWED_VINTAGE_RESOLVED.value))
            .when(
                pl.col("publication_precision").is_null()
                | (pl.col("publication_precision") == "date")
            )
            .then(pl.lit(DecisionCode.ALLOWED_DATE_PRECISION.value))
            .otherwise(pl.lit(DecisionCode.ALLOWED_BEFORE_PUBLICATION.value))
        )
        is_allowed = ~(
            reject_pit | reject_missing_pub | reject_pub_after
            | reject_no_vintage | reject_event_after
        )
        step = step1.with_columns(
            is_allowed.alias("allowed"),
            decision_code.alias("decision_code"),
            warn_ingested.alias("warn_ingested_after_as_of"),
        )
        out = step.with_columns(
            pl.when(pl.col("decision_code") == DecisionCode.NOT_POINT_IN_TIME.value)
            .then(pl.lit(
                "series is revised and only the latest vintage is recorded; "
                "the as-of value cannot be established"
            ))
            .when(pl.col("decision_code") == DecisionCode.MISSING_PUBLICATION_TIME.value)
            .then(pl.lit("no publication time recorded; availability is never invented"))
            .when(pl.col("decision_code") == DecisionCode.PUBLICATION_AT_OR_AFTER_AS_OF.value)
            .then(
                pl.lit("available_instant ") + pl.col("available_instant").cast(pl.Utf8)
                + pl.lit(f" is not strictly before as_of={t.isoformat()}")
            )
            .when(pl.col("decision_code") == DecisionCode.NO_VINTAGE_AT_AS_OF.value)
            .then(
                pl.lit("vintage ") + pl.col("vintage_id").fill_null("?")
                + pl.lit(f" released after as_of date {t.date().isoformat()}")
            )
            .when(pl.col("decision_code") == DecisionCode.EVENT_AFTER_AS_OF.value)
            .then(
                pl.lit("event_time ") + pl.col("event_time").cast(pl.Utf8)
                + pl.lit(f" is after as_of={t.isoformat()}")
            )
            .when(pl.col("decision_code").str.starts_with("allowed"))
            .then(
                pl.lit("publication available at ")
                + pl.col("available_instant").cast(pl.Utf8)
                + pl.lit(f" < as_of={t.isoformat()}")
            )
            .otherwise(pl.lit(""))
            .alias("decision_detail"),
        )
        allowed = out.filter(pl.col("allowed"))
        excluded = out.filter(~pl.col("allowed"))
        return Evaluation(
            as_of_time=t, frame=out, allowed=allowed, excluded=excluded,
            limitations=self._limitations(excluded),
        )

    def latest_available(
        self, frame: pl.DataFrame, as_of: datetime | date | str, keys: list[str]
    ) -> pl.DataFrame:
        """For each key, the record with the LATEST available instant < as_of.

        Multiple releases on the same instant are broken deterministically
        (largest record_id wins the tie).
        """
        ev = self.evaluate(frame, as_of)
        return (
            ev.allowed
            .sort(["available_instant", "record_id"])
            .unique(subset=keys, keep="last")
        )

    def historical_vintage(
        self, frame: pl.DataFrame, as_of: datetime | date | str
    ) -> pl.DataFrame:
        """Vintage-resolved point-in-time values for macro observations.

        For each (series, observation) the value of the LATEST version
        released strictly before the as_of date. Observations with no
        released version at as_of are excluded (NO_VINTAGE_AT_AS_OF).
        """
        return self._resolve_vintages(self.evaluate(frame, as_of).frame).filter(
            pl.col("allowed")
        )

    def asof_join(
        self,
        right: pl.DataFrame,
        left: pl.DataFrame,
        left_time_col: str = "decision_time",
    ) -> pl.DataFrame:
        """Temporal join: attach to each left row the most recent right-side
        observation that was available at that row's decision time.

        Semantics (prompt section 16):
          - right records are gated by the FULL availability rules at each
            left row's decision time (publication, vintage, precision,
            event, ingestion warnings);
          - among the available right records for a left row, the one with
            the largest event_time (most recent observation) <= the left
            row's decision time is attached;
          - ties are broken deterministically (largest record_id);
          - left rows with no qualifying right record get a null join.

        Correctness first: this is a grouped cross-product, fine for the
        MVP; optimize only after profiling (prompt section 22).
        """
        if left.height == 0:
            return left
        # a row index keeps duplicate left record_ids distinct (the join
        # picks ONE right record per LEFT ROW, not per record_id); the name
        # must not collide with a column the caller already has
        idx_col = "_join_idx"
        while idx_col in left.columns:
            idx_col = f"_{idx_col}"
        left = left.with_row_index(idx_col)
        # every right column that collides with a left column is renamed so
        # the filter below can never read the LEFT frame's column; column
        # names in the joined output are therefore deterministic too. The
        # right-side schema (raw columns + decision columns appended by
        # evaluate()) is computed ONCE so the joined and null groups always
        # have identical names, order and types - whatever collides.
        rename_map = self._right_rename_map(left.columns, right)
        right_side_schema = self._right_side_schema(rename_map, right)
        if right.height == 0:
            # empty right: every left row gets a null join with the same
            # right-side schema as the (never-materialized) joined groups
            return self._null_right_group(left, right_side_schema).drop(idx_col)
        right_event = rename_map.get("event_time", "event_time")
        right_record = rename_map.get("record_id", "record_id")
        tie_cols = [idx_col, right_event, right_record]

        groups: list[pl.DataFrame] = []
        for t_raw in sorted(
            left[left_time_col].unique().to_list(), key=lambda v: (v is None, v)
        ):
            if t_raw is None:
                # a left row without a decision instant can never join a
                # right record: it has no "moment in time" to be asked about
                groups.append(
                    self._null_right_group(
                        left.filter(pl.col(left_time_col).is_null()),
                        right_side_schema,
                    )
                )
                continue
            group = left.filter(pl.col(left_time_col) == t_raw)
            t = normalize_instant(t_raw)
            evaluated = self._resolve_vintages(self.evaluate(right, t).frame)
            avail = evaluated.filter(pl.col("allowed"))
            if avail.height == 0:
                groups.append(self._null_right_group(group, right_side_schema))
                continue
            if rename_map:
                avail = avail.rename(rename_map)
            candidates = group.join(avail, how="cross")
            candidates = candidates.filter(pl.col(right_event) <= t)
            # keep, per left row, the most recent observation (max event_time);
            # within a tie, the max right record_id for determinism
            joined = (
                candidates
                .sort(tie_cols)
                .unique(subset=[idx_col], keep="last")
            )
            groups.append(joined)
        out = pl.concat(groups) if groups else left
        return out.drop(idx_col)

    @staticmethod
    def _right_rename_map(
        left_cols: list[str], right: pl.DataFrame
    ) -> dict[str, str]:
        """Collision-safe rename targets for the right frame's columns.

        A rename target must not collide with ANY left column (a left
        'right_event_time' must not hijack the renamed right event_time) nor
        with another right-side column (a right frame carrying its own
        'right_source_key' must not collide with the renamed source_key).
        Unused targets are escalated with a counter suffix. The name
        universe is deduped: a decision-column name already present as a raw
        right column (e.g. a chained asof_join output carries 'allowed') is
        renamed once, for both occurrences.
        """
        right_cols = set(right.columns)
        mapping: dict[str, str] = {}
        for c in dict.fromkeys(
            list(right.columns) + [c for c, _ in DECISION_COLUMNS]
        ):
            if c not in left_cols:
                continue
            base = "right_record_id" if c == "record_id" else f"right_{c}"
            name = base
            i = 2
            while name in left_cols or (name != c and name in right_cols):
                name = f"{base}__{i}"
                i += 1
            mapping[c] = name
            left_cols = left_cols + [name]
        return mapping

    @staticmethod
    def _right_side_schema(
        rename_map: dict[str, str], right: pl.DataFrame
    ) -> list[tuple[str, pl.DataType]]:
        """The canonical right-side output schema: raw right columns (with
        decision-named ones REPLACED in place by the engine's decision
        column dtype, exactly as evaluate() overwrites them) followed by the
        decision columns the engine appends. Names/order/types match the
        joined groups exactly, so null groups can always be concatenated."""
        decision_types = dict(DECISION_COLUMNS)
        schema = [
            (
                rename_map.get(c, c),
                decision_types[c] if c in decision_types else right.schema[c],
            )
            for c in right.columns
        ]
        schema += [
            (rename_map.get(c, c), dtype)
            for c, dtype in DECISION_COLUMNS
            if c not in right.columns
        ]
        return schema

    @staticmethod
    def _null_right_group(
        group: pl.DataFrame, right_side_schema: list[tuple[str, pl.DataType]]
    ) -> pl.DataFrame:
        """A left group with no qualifying right record keeps its rows with
        NULL right-side columns (the 'null join'), with a schema identical to
        the joined groups (same names, order and types) so they can always be
        concatenated."""
        nulls = [
            pl.lit(None, dtype=dtype).alias(name)
            for name, dtype in right_side_schema
        ]
        return group.with_columns(nulls)

    # --------------------------------------------------------------- sources

    def _load_sources(self, sources: list[TemporalSource]) -> list[SourceInput]:
        if not sources:
            raise ValueError(
                "no temporal sources configured - pass sources to snapshot() "
                "or build the engine with sources=[...]"
            )
        return [self._load_source(s) for s in sources]

    def _load_source(self, src: TemporalSource) -> SourceInput:
        paths = self._artifact_paths(src)
        if not paths:
            raise FileNotFoundError(
                f"no normalized artifacts found for {src.snapshot_id} "
                f"(domain={src.domain}, provider={src.provider})"
            )
        frames = [self._frame_for(src, p) for p in paths]
        frame = pl.concat(frames) if len(frames) > 1 else frames[0]
        return SourceInput(source=src, timing_frame=frame)

    def _artifact_paths(self, src: TemporalSource) -> list[str]:
        if src.artifact_paths:
            return [p for p in src.artifact_paths if Path(p).exists()]
        if src.domain == "market":
            return [
                str(normalized_dir("market", src.provider, src.snapshot_id) / "bars.parquet")
            ]
        if src.domain in ("sec", "fundamentals"):
            return [
                str(normalized_dir("fundamentals", src.provider, src.snapshot_id) / "facts.parquet")
            ]
        if src.domain == "macro":
            return [
                str(normalized_dir("macro", src.provider, src.snapshot_id) / "series.parquet")
            ]
        return []

    def _frame_for(self, src: TemporalSource, path: str) -> pl.DataFrame:
        df = pl.read_parquet(path)
        if src.domain == "market":
            return market_timing_frame(df, src.snapshot_id, src.ingest_time)
        if src.domain in ("sec", "fundamentals"):
            return sec_timing_frame(df, src.snapshot_id, src.ingest_time)
        if src.domain == "macro":
            return fred_timing_frame(
                df,
                src.snapshot_id,
                src.ingest_time,
                series_policies=self._contract.series_policies,
                default_policy=self._contract.default_series_policy,
            )
        raise ValueError(f"unknown source domain: {src.domain}")

    # ------------------------------------------------------------- internals

    def _resolve_vintages(self, evaluated: pl.DataFrame) -> pl.DataFrame:
        """Keep, per (series, observation), the latest version RELEASED AND
        ALLOWED at as_of. Older allowed versions become VINTAGE_SUPERSEDED
        (audited, not silently dropped). Rejected versions stay rejected:
        a version that was not public at as_of can never be the 'latest'."""
        if evaluated.height == 0:
            return evaluated
        vintage_rows = evaluated.filter(
            pl.col("vintage_date").is_not_null() & pl.col("allowed")
        )
        if vintage_rows.height == 0:
            return evaluated
        resolved = (
            vintage_rows
            .sort(["source_key", "event_time", "vintage_date", "record_id"])
            .unique(subset=["source_key", "event_time"], keep="last")
        )
        superseded = (
            vintage_rows
            .join(
                resolved.select(["record_id"]),
                on="record_id", how="anti",
            )
            .with_columns(
                pl.lit(False).alias("allowed"),
                pl.lit(DecisionCode.VINTAGE_SUPERSEDED.value).alias("decision_code"),
                pl.lit(
                    "an older version of this observation was released before "
                    "as_of; the latest released version supersedes it"
                ).alias("decision_detail"),
            )
        )
        rest = evaluated.filter(pl.col("vintage_date").is_null())
        return pl.concat([rest, resolved, superseded]).sort("record_id")

    def _limitations(
        self, excluded: pl.DataFrame,
    ) -> list[str]:
        if excluded.height == 0:
            return []
        limitations: list[str] = []
        by_code = excluded.group_by("decision_code").agg(pl.len().alias("n"))
        code_counts = {
            r["decision_code"]: r["n"]
            for r in by_code.iter_rows(named=True)
        }
        if code_counts.get(DecisionCode.NOT_POINT_IN_TIME.value):
            series = sorted(
                excluded.filter(pl.col("decision_code") == DecisionCode.NOT_POINT_IN_TIME.value)["source_key"].unique().to_list()
            )
            limitations.append(
                f"revised macro series without vintage history excluded at this as_of "
                f"({code_counts[DecisionCode.NOT_POINT_IN_TIME.value]} observations): {series}. "
                "ALFRED vintage ingestion is required to make these series point-in-time."
            )
        if code_counts.get(DecisionCode.MISSING_PUBLICATION_TIME.value):
            limitations.append(
                f"{code_counts[DecisionCode.MISSING_PUBLICATION_TIME.value]} records have no "
                "publication time and were excluded (availability never invented)."
            )
        if code_counts.get(DecisionCode.NO_VINTAGE_AT_AS_OF.value):
            limitations.append(
                f"{code_counts[DecisionCode.NO_VINTAGE_AT_AS_OF.value]} observations have no "
                "version released before this as_of and were excluded."
            )
        if code_counts.get(DecisionCode.EVENT_AFTER_AS_OF.value):
            limitations.append(
                f"{code_counts[DecisionCode.EVENT_AFTER_AS_OF.value]} forward-dated records "
                "(event after as_of) were excluded from the historical information set."
            )
        return limitations


def build_temporal_source(
    registry: Any,
    snapshot_id: str,
    artifact_paths: list[str] | None = None,
) -> TemporalSource:
    """Build a TemporalSource from a registry snapshot record (provenance)."""
    rec = registry.snapshot(snapshot_id)
    if rec is None:
        raise KeyError(f"unknown snapshot: {snapshot_id}")
    return TemporalSource(
        snapshot_id=snapshot_id,
        domain=rec["domain"],
        provider=rec["provider"],
        checksum=rec["checksum"],
        manifest_path=rec.get("manifest_path"),
        ingest_time=normalize_instant(rec["downloaded_at"]),
        artifact_paths=artifact_paths or [],
    )