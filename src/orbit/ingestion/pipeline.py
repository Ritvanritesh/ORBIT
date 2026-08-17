"""The ingestion pipeline.

Order of operations for one snapshot:

    request -> fingerprint -> (idempotent reuse?)
    -> connector.fetch() -> RawStore.write_once() -> registry.register_snapshot()
    -> parse + validate -> normalize -> parquet + schema sidecar
    -> reconcile (market) -> manifest -> seal raw zone -> DatasetSnapshot

Idempotency: the same request fingerprint produces no second download and
no duplicate records; failed snapshots are never reused (a retry creates a
new snapshot so the failed attempt stays visible in the registry).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl

from orbit.ingestion.checksums import request_fingerprint, sha256_bytes, sha256_file
from orbit.ingestion.manifests import build_manifest
from orbit.ingestion.normalizers.fundamentals import FUNDAMENTALS_SCHEMA_VERSION, normalize_sec_facts
from orbit.ingestion.normalizers.macro import MACRO_SCHEMA_VERSION, normalize_fred_series
from orbit.ingestion.normalizers.market import (
    BAR_COLUMNS,
    MARKET_SCHEMA_VERSION,
    build_corporate_actions,
    normalize_market_bars,
    write_schema_sidecar,
)
from orbit.ingestion.parsing import (
    parse_fred_csv,
    parse_sec_companyfacts,
    parse_stooq_csv,
    parse_yahoo_chart,
)
from orbit.ingestion.paths import normalized_dir, raw_dir, write_json
from orbit.ingestion.providers.base import RawObject
from orbit.ingestion.reconciliation import reconcile_market
from orbit.ingestion.registry import IngestionRegistry
from orbit.ingestion.storage import RawStore
from orbit.ingestion.validators import (
    validate_fred_series,
    validate_market_bars,
    validate_sec_facts,
)

SCHEMA_VERSIONS = {
    "market": MARKET_SCHEMA_VERSION,
    "sec": FUNDAMENTALS_SCHEMA_VERSION,
    "macro": MACRO_SCHEMA_VERSION,
}
# provider -> (connector, parse function) for raw artifact -> parsed tables
MARKET_PARSERS = {
    "yahoo_chart_api": lambda raw: parse_yahoo_chart(raw.body, raw.meta["symbol"]),
    "stooq_csv": lambda raw: parse_stooq_csv(raw.body, raw.meta["symbol"]),
}


@dataclass
class IngestResult:
    snapshot_id: str
    reused: bool
    domain: str
    provider: str
    raw_files: list[str] = field(default_factory=list)
    row_count: int = 0
    validation: dict[str, Any] = field(default_factory=dict)
    normalized_paths: dict[str, str] = field(default_factory=dict)
    snapshot_record: dict[str, Any] | None = None
    reconciliation: dict[str, Any] | None = None

    def summary(self) -> str:
        status = "reused" if self.reused else "ingested"
        return (
            f"[{status}] {self.snapshot_id} {self.domain}/{self.provider} "
            f"files={len(self.raw_files)} rows={self.row_count} "
            f"validation={self.validation.get('status', '?')}"
        )


class IngestionPipeline:
    def __init__(self, registry: IngestionRegistry, raw_store: RawStore):
        self.registry = registry
        self.raw_store = raw_store

    # ----------------------------------------------------------------- market

    def ingest_market(
        self,
        connector: Any,
        symbols: list[str],
        symbol_map: dict[str, str],
        *,
        license_ref: str,
        request_params: dict[str, Any] | None = None,
    ) -> IngestResult:
        provider = connector.provider_name
        params = request_params or {}
        fp = request_fingerprint(
            {"domain": "market", "provider": provider, "symbols": sorted(symbols), "params": params}
        )
        existing = self.registry.snapshot_for_fingerprint(fp)
        if existing and existing["validation_status"] == "ok":
            self._verify_reuse(provider, existing)
            recon = self._refresh_reconciliation(existing["snapshot_id"])
            return IngestResult(
                snapshot_id=existing["snapshot_id"], reused=True, domain="market",
                provider=provider, row_count=existing["row_count"] or 0,
                validation={"status": "ok"}, snapshot_record=existing,
                reconciliation=recon,
            )

        raw_objects = self._load_or_fetch(connector, "market", provider, fp, [
            {"symbol": symbol, **params} for symbol in symbols
        ])
        # Canonical order for byte-identical re-derivation: raw files are
        # re-read from disk in sorted filename order, so the first pass must
        # process in that same order.
        raw_objects.sort(key=lambda r: r.filename)
        if existing is None:
            snapshot_id = self._register_raw(
                domain="market", provider=provider, raw_objects=raw_objects,
                fingerprint=fp, license_ref=license_ref,
            )
        else:
            snapshot_id = existing["snapshot_id"]

        parsed: dict[str, dict[str, pl.DataFrame]] = {}
        issues: list[dict[str, Any]] = []
        for raw in raw_objects:
            symbol = raw.meta["symbol"]
            tables = MARKET_PARSERS[provider](raw)
            parsed[symbol] = {"bars": tables[0], "events": tables[1]}
            date_col = "ts" if provider == "yahoo_chart_api" else "date"
            report = validate_market_bars(tables[0], symbol, date_col=date_col, provider=provider)
            issues.extend(i.to_dict() for i in report.issues)

        status = "ok" if not any(i["level"] == "error" for i in issues) else "failed"
        if status == "failed":
            return self._finish_failed(
                snapshot_id, "market", provider, issues, raw_objects,
            )

        normalized = normalize_market_bars(
            parsed, symbol_map, provider, raw_objects[0].source_uri, snapshot_id
        )
        recon = reconcile_market(normalized["bars"], normalized["events"])
        out = normalized_dir("market", provider, snapshot_id)
        self._write_parquet(out, "bars.parquet", normalized["bars"])
        self._write_parquet(out, "events.parquet", normalized["events"])
        write_schema_sidecar(out, "market_daily_bars", BAR_COLUMNS)
        row_count = normalized["bars"].height
        paths = self._register_artifacts(
            snapshot_id, "market", out, ["bars.parquet", "events.parquet"], row_count
        )
        actions = build_corporate_actions(
            normalized["events"], snapshot_id, f"{provider}"
        )
        write_json(out / "corporate_actions.json", actions)
        self._register_artifacts(
            snapshot_id, "market", out, ["corporate_actions.json"], len(actions)
        )
        earliest = normalized["bars"]["trade_date"].min()
        latest = normalized["bars"]["trade_date"].max()
        manifest = build_manifest(
            snapshot_id=snapshot_id, domain="market", provider=provider,
            source=provider, source_uri=raw_objects[0].source_uri,
            request_fingerprint=fp, checksums=self._checksums_of(raw_objects),
            file_paths={
                r.filename: str(self._raw_path("market", provider, snapshot_id, r.filename))
                for r in raw_objects
            },
            earliest=earliest, latest=latest,
            instruments=list(symbol_map.values()),
            schema_version=MARKET_SCHEMA_VERSION,
            data_version="v1", license_ref=license_ref,
            row_count=row_count, validation_status=status,
            validation_issues=issues,
            meta={
                "symbols": sorted(symbols),
                "adjustment_note": (
                    "yahoo_chart_api OHLC AND volume are split-adjusted to the current "
                    "share basis (volume confirmed continuous across the 7:1 and 4:1 "
                    "splits in the dev sample); stooq_csv Close is split-adjusted but "
                    "volume is raw shares - dollar volume from stooq close*volume is "
                    "discontinuous at splits and must be reconstructed from events"
                ),
                "adjclose_warning": (
                    "adjclose is retroactively dividend+split adjusted by the provider; "
                    "it is a look-ahead if used as a point-in-time price. Use 'close' "
                    "for returns (split-adjusted basis is return-neutral)."
                ),
                "reconciliation": recon.to_dict(),
                "corporate_actions": len(actions),
                "timezone_rule": "see normalized _schema.json",
            },
        )
        manifest_path = manifest.write()
        self.registry.update_manifest_path(snapshot_id, manifest_path)
        self._seal("market", provider, snapshot_id, raw_objects, manifest_path)
        self.registry.update_validation(snapshot_id, status, row_count)
        return IngestResult(
            snapshot_id=snapshot_id, reused=False, domain="market", provider=provider,
            raw_files=[r.filename for r in raw_objects], row_count=row_count,
            validation={"status": status, "issues": issues},
            normalized_paths=paths,
            snapshot_record=self.registry.snapshot(snapshot_id),
            reconciliation=recon.to_dict(),
        )

    def _verify_reuse(self, provider: str, record: dict[str, Any]) -> None:
        """Before reusing an ok snapshot, prove the raw zone is still intact.

        Trust anchor is the manifest (written by ORBIT, outside the raw
        zone): the seal marker inside the raw zone could be tampered
        together with the payloads, so reuse compares on-disk checksums
        against the manifest's recorded checksums.
        """
        import json

        from orbit.ingestion.checksums import sha256_file

        if not record.get("manifest_path") or not Path(record["manifest_path"]).exists():
            raise RuntimeError(
                f"cannot reuse {record['snapshot_id']}: manifest missing - "
                "refusing to serve untrusted raw data"
            )
        manifest = json.loads(Path(record["manifest_path"]).read_text(encoding="utf-8"))
        expected = manifest.get("checksums", {})
        base = self.raw_store.snapshot_dir(record["domain"], provider, record["snapshot_id"])
        actual = {
            p.name: sha256_file(p)
            for p in sorted(base.iterdir())
            if p.is_file() and p.name != "IMMUTABLE.json"
        }
        if actual != expected:
            missing = sorted(set(expected) - set(actual))
            drifted = sorted(
                n for n in expected if n in actual and actual[n] != expected[n]
            )
            raise RuntimeError(
                f"raw zone tampered for {record['snapshot_id']}: "
                f"missing={missing} drifted={drifted} - refusing reuse"
            )

    def _refresh_reconciliation(self, snapshot_id: str) -> dict[str, Any]:
        """Re-run reconciliation against stored normalized bars/events.

        Reconciliation is a review step, not an ingestion artifact: the
        findings must track the current checks even when a snapshot is
        reused, so a rule change is immediately visible without a re-download.
        """
        import json

        record = self.registry.snapshot(snapshot_id)
        out = normalized_dir(record["domain"], record["provider"], snapshot_id)
        bars_path = out / "bars.parquet"
        if not bars_path.exists():
            return {"finding_count": 0, "findings": []}
        events = (
            pl.read_parquet(out / "events.parquet")
            if (out / "events.parquet").exists()
            else pl.DataFrame()
        )
        recon = reconcile_market(pl.read_parquet(bars_path), events).to_dict()
        if record.get("manifest_path"):
            manifest_path = Path(record["manifest_path"])
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest.setdefault("meta", {})["reconciliation"] = recon
                tmp = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
                tmp.write_text(
                    json.dumps(manifest, indent=2, default=str), encoding="utf-8"
                )
                tmp.replace(manifest_path)
        return recon

    # -------------------------------------------------------------- sec/macro

    def ingest_sec(
        self, connector: Any, ciks: list[int], *, license_ref: str, request_params: dict[str, Any] | None = None,
    ) -> IngestResult:
        provider = connector.provider_name
        params = request_params or {}
        fp = request_fingerprint(
            {"domain": "sec", "provider": provider, "ciks": sorted(ciks), "params": params}
        )
        existing = self.registry.snapshot_for_fingerprint(fp)
        if existing and existing["validation_status"] == "ok":
            self._verify_reuse(provider, existing)
            return IngestResult(
                snapshot_id=existing["snapshot_id"], reused=True, domain="sec",
                provider=provider, row_count=existing["row_count"] or 0,
                validation={"status": "ok"}, snapshot_record=existing,
            )

        raw_objects = self._load_or_fetch(connector, "sec", provider, fp, [
            {"cik": cik, **params} for cik in ciks
        ])
        raw_objects.sort(key=lambda r: r.filename)
        if existing is None:
            snapshot_id = self._register_raw(
                domain="sec", provider=provider, raw_objects=raw_objects,
                fingerprint=fp, license_ref=license_ref,
            )
        else:
            snapshot_id = existing["snapshot_id"]

        parsed_frames = []
        issues = []
        for raw in raw_objects:
            facts = parse_sec_companyfacts(raw.body)
            parsed_frames.append(facts)
            report = validate_sec_facts(facts, raw.meta["cik"])
            issues.extend(i.to_dict() for i in report.issues)

        status = "ok" if not any(i["level"] == "error" for i in issues) else "failed"
        if status == "failed":
            return self._finish_failed(
                snapshot_id, "sec", provider, issues, raw_objects,
            )

        facts_all = normalize_sec_facts(pl.concat(parsed_frames), provider, snapshot_id)
        out = normalized_dir("fundamentals", provider, snapshot_id)
        self._write_parquet(out, "facts.parquet", facts_all)
        write_json(
            out / "_schema.json",
            {"kind": "sec_companyfacts", "schema_version": FUNDAMENTALS_SCHEMA_VERSION, "columns": facts_all.columns},
        )
        row_count = facts_all.height
        paths = self._register_artifacts(snapshot_id, "sec", out, ["facts.parquet"], row_count)
        filed_dates = [f for f in facts_all["filed"].drop_nulls().to_list() if isinstance(f, str)]
        if filed_dates:
            filed_parsed = sorted(date.fromisoformat(f) for f in filed_dates)
            earliest, latest = filed_parsed[0], filed_parsed[-1]
        else:
            earliest = latest = None
        manifest = build_manifest(
            snapshot_id=snapshot_id, domain="sec", provider=provider,
            source="SEC EDGAR / XBRL companyfacts", source_uri=raw_objects[0].source_uri,
            request_fingerprint=fp, checksums=self._checksums_of(raw_objects),
            file_paths={
                r.filename: str(self._raw_path("sec", provider, snapshot_id, r.filename))
                for r in raw_objects
            },
            earliest=earliest, latest=latest,
            instruments=[str(c) for c in ciks],
            schema_version=FUNDAMENTALS_SCHEMA_VERSION,
            data_version="v1", license_ref=license_ref,
            row_count=row_count, validation_status=status,
            validation_issues=issues,
            meta={
                "ciks": sorted(ciks),
                "filing_metadata_preserved": True,
                "date_range_semantics": (
                    "earliest/latest are FILING dates (point-in-time availability), "
                    "not reporting-period dates; period coverage lives in start/end "
                    "columns of facts.parquet"
                ),
            },
        )
        manifest_path = manifest.write()
        self.registry.update_manifest_path(snapshot_id, manifest_path)
        self._seal("sec", provider, snapshot_id, raw_objects, manifest_path)
        self.registry.update_validation(snapshot_id, status, row_count)
        return IngestResult(
            snapshot_id=snapshot_id, reused=False, domain="sec", provider=provider,
            raw_files=[r.filename for r in raw_objects], row_count=row_count,
            validation={"status": status, "issues": issues},
            normalized_paths=paths,
            snapshot_record=self.registry.snapshot(snapshot_id),
        )

    def ingest_macro(
        self, connector: Any, series_ids: list[str], *, license_ref: str, request_params: dict[str, Any] | None = None,
    ) -> IngestResult:
        provider = connector.provider_name
        params = request_params or {}
        fp = request_fingerprint(
            {"domain": "macro", "provider": provider, "series": sorted(series_ids), "params": params}
        )
        existing = self.registry.snapshot_for_fingerprint(fp)
        if existing and existing["validation_status"] == "ok":
            self._verify_reuse(provider, existing)
            return IngestResult(
                snapshot_id=existing["snapshot_id"], reused=True, domain="macro",
                provider=provider, row_count=existing["row_count"] or 0,
                validation={"status": "ok"}, snapshot_record=existing,
            )

        raw_objects = self._load_or_fetch(connector, "macro", provider, fp, [
            {"series_id": sid, **params} for sid in series_ids
        ])
        raw_objects.sort(key=lambda r: r.filename)
        if existing is None:
            snapshot_id = self._register_raw(
                domain="macro", provider=provider, raw_objects=raw_objects,
                fingerprint=fp, license_ref=license_ref,
            )
        else:
            snapshot_id = existing["snapshot_id"]

        parsed_frames = []
        issues = []
        for raw in raw_objects:
            series = parse_fred_csv(raw.body, raw.meta["series_id"])
            parsed_frames.append(series)
            report = validate_fred_series(series, raw.meta["series_id"])
            issues.extend(i.to_dict() for i in report.issues)

        status = "ok" if not any(i["level"] == "error" for i in issues) else "failed"
        if status == "failed":
            return self._finish_failed(
                snapshot_id, "macro", provider, issues, raw_objects,
            )

        vintage_requested = params.get("vintage_date")
        series_all = normalize_fred_series(
            pl.concat(parsed_frames),
            vintage_note=(
                "alfred_vintage_requested" if vintage_requested
                else "latest_published_vintage"
            ),
            provider=provider, snapshot_id=snapshot_id,
            vintage_date=vintage_requested,
        )
        out = normalized_dir("macro", provider, snapshot_id)
        self._write_parquet(out, "series.parquet", series_all)
        write_json(
            out / "_schema.json",
            {"kind": "fred_series", "schema_version": MACRO_SCHEMA_VERSION, "columns": series_all.columns},
        )
        row_count = series_all.height
        paths = self._register_artifacts(snapshot_id, "macro", out, ["series.parquet"], row_count)
        dates = series_all["observation_date"].drop_nulls()
        earliest, latest = dates.min(), dates.max()
        manifest = build_manifest(
            snapshot_id=snapshot_id, domain="macro", provider=provider,
            source="FRED / ALFRED", source_uri=raw_objects[0].source_uri,
            request_fingerprint=fp, checksums=self._checksums_of(raw_objects),
            file_paths={
                r.filename: str(self._raw_path("macro", provider, snapshot_id, r.filename))
                for r in raw_objects
            },
            earliest=earliest, latest=latest,
            instruments=sorted(series_ids),
            schema_version=MACRO_SCHEMA_VERSION,
            data_version="v1", license_ref=license_ref,
            row_count=row_count, validation_status=status,
            validation_issues=issues,
            meta={
                "series": sorted(series_ids),
                "vintage_note": (
                    "alfred_vintage_requested" if vintage_requested
                    else "latest_published_vintage"
                ),
                "vintage_date": vintage_requested,
                "vintage_semantics": (
                    "when vintage_date is set this snapshot is ONE ALFRED "
                    "vintage: every observation value is the value as it "
                    "existed on vintage_date (point-in-time). When unset the "
                    "snapshot is the latest published vintage and revised "
                    "series are NOT point-in-time (Phase 4 engine rejects "
                    "them unless ALFRED vintages are ingested)."
                ),
            },
        )
        manifest_path = manifest.write()
        self.registry.update_manifest_path(snapshot_id, manifest_path)
        self._seal("macro", provider, snapshot_id, raw_objects, manifest_path)
        self.registry.update_validation(snapshot_id, status, row_count)
        return IngestResult(
            snapshot_id=snapshot_id, reused=False, domain="macro", provider=provider,
            raw_files=[r.filename for r in raw_objects], row_count=row_count,
            validation={"status": status, "issues": issues},
            normalized_paths=paths,
            snapshot_record=self.registry.snapshot(snapshot_id),
        )

    # ------------------------------------------------------------- internals

    def _load_or_fetch(
        self, connector: Any, domain: str, provider: str, fingerprint: str,
        requests: list[dict[str, Any]],
    ) -> list[RawObject]:
        """Fetch from the provider, unless a prior snapshot's raw files exist.

        Failed or interrupted snapshots are reprocessed from the bytes
        already on disk - the provider is only hit again when nothing was
        stored yet.
        """
        existing = self.registry.snapshot_for_fingerprint(fingerprint)
        if existing:
            from_disk = self._raw_objects_from_disk(domain, provider, existing["snapshot_id"])
            if from_disk:
                import dataclasses

                marker = (
                    self.raw_store.snapshot_dir(domain, provider, existing["snapshot_id"])
                    / "IMMUTABLE.json"
                )
                if marker.exists() and not self.raw_store.verify(
                    domain, provider, existing["snapshot_id"]
                ):
                    raise RuntimeError(
                        f"raw zone tampered for {existing['snapshot_id']} - "
                        "refusing to reprocess it"
                    )
                have = {r.filename for r in from_disk}
                want = {
                    self._request_filename(req, provider) for req in requests
                }
                missing = sorted(want - have)
                if missing:
                    raise RuntimeError(
                        f"raw zone for {existing['snapshot_id']} is INCOMPLETE: "
                        f"missing {missing} - refusing to silently drop data; "
                        "delete the snapshot dir to force a clean re-download"
                    )
                return [
                    dataclasses.replace(obj, source_uri=existing["source_uri"])
                    for obj in from_disk
                ]
        fetched: list[RawObject] = []
        for req in requests:
            fetched.extend(connector.fetch(req))
        return fetched

    @staticmethod
    def _request_filename(request: dict[str, Any], provider: str) -> str:
        if "symbol" in request:
            ext = "csv" if provider == "stooq_csv" else "json"
            return f"{request['symbol']}.{ext}"
        if "cik" in request:
            return f"cik{request['cik']:010d}_companyfacts.json"
        return f"{request['series_id']}.csv"

    def _raw_objects_from_disk(
        self, domain: str, provider: str, snapshot_id: str
    ) -> list[RawObject]:
        base = self.raw_store.snapshot_dir(domain, provider, snapshot_id)
        if not base.exists():
            return []
        objects: list[RawObject] = []
        for path in sorted(base.iterdir()):
            if not path.is_file() or path.name == "IMMUTABLE.json":
                continue
            if domain == "market":
                meta = {"symbol": path.stem}
            elif domain == "sec":
                meta = {"cik": int(path.stem[3:13])}
            else:
                meta = {"series_id": path.stem}
            objects.append(
                RawObject(filename=path.name, body=path.read_bytes(),
                          source_uri="", content_type="", meta=meta)
            )
        return objects

    def _register_raw(
        self, domain: str, provider: str, raw_objects: list[RawObject],
        fingerprint: str, license_ref: str,
    ) -> str:
        record = {
            "domain": domain, "provider": provider,
            "source_uri": raw_objects[0].source_uri,
            "request_fingerprint": fingerprint,
            "checksum": request_fingerprint(
                {r.filename: sha256_bytes(r.body) for r in raw_objects}
            ),
            "file_count": len(raw_objects),
            "downloaded_at": datetime.now(timezone.utc).isoformat(),
            "schema_version": SCHEMA_VERSIONS[domain],
            "license_ref": license_ref,
            "validation_status": "pending",
        }
        snapshot_id = self.registry.register_snapshot(record)
        for raw in raw_objects:
            self.raw_store.write_once(domain, provider, snapshot_id, raw.filename, raw.body)
        # Seal immediately: even a snapshot that later FAILS validation must
        # have a tamper-evident raw zone, so a tampered retry can never
        # reprocess into a "ok" snapshot unnoticed.
        self._seal(domain, provider, snapshot_id, raw_objects, "")
        return snapshot_id

    def _register_artifacts(
        self, snapshot_id: str, domain: str, out_dir: Path, files: list[str], row_count: int,
    ) -> dict[str, str]:
        paths: dict[str, str] = {}
        for f in files:
            path = out_dir / f
            checksum = sha256_file(path)
            artifact_id = self.registry.register_artifact(
                snapshot_id, domain, str(path), checksum, row_count, SCHEMA_VERSIONS[domain]
            )
            paths[f] = str(path)
            write_json(
                out_dir / f"{f}.sha256",
                {"artifact_id": artifact_id, "checksum": checksum, "path": str(path)},
            )
        return paths

    def _write_parquet(self, out_dir: Path, filename: str, df: pl.DataFrame) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        df.write_parquet(out_dir / filename)

    def _seal(
        self, domain: str, provider: str, snapshot_id: str,
        raw_objects: list[RawObject], manifest_path: str,
    ) -> None:
        expected = {r.filename: sha256_bytes(r.body) for r in raw_objects}
        self.raw_store.seal(domain, provider, snapshot_id, expected)

    def _finish_failed(
        self, snapshot_id: str, domain: str, provider: str,
        issues: list[dict[str, Any]], raw_objects: list[RawObject],
    ) -> IngestResult:
        self.registry.update_validation(snapshot_id, "failed", 0)
        return IngestResult(
            snapshot_id=snapshot_id, reused=False, domain=domain, provider=provider,
            raw_files=[r.filename for r in raw_objects], row_count=0,
            validation={"status": "failed", "issues": issues},
            snapshot_record=self.registry.snapshot(snapshot_id),
        )

    def _raw_path(self, domain: str, provider: str, snapshot_id: str, filename: str) -> Path:
        return raw_dir(domain, provider, snapshot_id) / filename

    @staticmethod
    def _checksums_of(raw_objects: list[RawObject]) -> dict[str, str]:
        return {r.filename: sha256_bytes(r.body) for r in raw_objects}