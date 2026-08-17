"""Phase 3 end-to-end run: 5-stock market + SEC + FRED, then verify.

1. Ingests the dev sample (5 stocks across sectors) from the working free
   market source, SEC company facts for the same names, and FRED series.
2. Prints the registry (what ORBIT downloaded, when, checksums, validation).
3. Verifies raw immutability (re-checksums the sealed raw zone).
4. Runs the reproducibility check: process the raw snapshot twice and
   compare normalized outputs byte-for-byte.

Expansion path: 5 -> 20 -> 50 -> 100 is just a longer symbol list in the
config - no pipeline changes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from orbit.ingestion.checksums import sha256_file
from orbit.ingestion.pipeline import IngestionPipeline
from orbit.ingestion.paths import data_root, ensure_layout, load_json, normalized_dir, registry_path
from orbit.ingestion.providers.fred import FredConnector
from orbit.ingestion.providers.sec import SecEdgarConnector
from orbit.ingestion.providers.yahoo import YahooChartConnector
from orbit.ingestion.registry import IngestionRegistry
from orbit.ingestion.snapshot import build_dataset_snapshot
from orbit.ingestion.storage import RawStore

REPO = Path(__file__).resolve().parents[1]

SEC_UA = "ORBIT-Research-Project research@example.com"


def load_dev_config() -> dict:
    dev = load_json(REPO / "configs" / "phase3_dev.json")
    master = load_json(REPO / "configs" / "instrument_master_dev.json")
    sources = load_json(REPO / "configs" / "sources.json")
    return dev, master, sources


def reproducibility_check(registry: IngestionRegistry, raw_store: RawStore,
                         snapshot_id: str, provider: str, domain: str,
                         symbol_map: dict[str, str], source_uri: str) -> bool:
    """Re-derive the normalized layer twice from the sealed raw files.

    Same raw snapshot -> same normalized output, byte for byte. This is the
    Phase 3 headline test: if it fails, stop - do not move to modeling.
    """
    import polars as pl

    from orbit.ingestion.normalizers.market import normalize_market_bars
    from orbit.ingestion.parsing import parse_yahoo_chart
    from orbit.ingestion.validators import validate_market_bars

    src_dir = raw_store.snapshot_dir(domain, provider, snapshot_id)
    if provider == "yahoo_chart_api":
        payloads = sorted(p for p in src_dir.glob("*.json") if p.name != "IMMUTABLE.json")
    else:
        payloads = sorted(src_dir.glob("*.csv"))

    def derive() -> bytes:
        parsed: dict[str, dict[str, pl.DataFrame]] = {}
        for payload in payloads:
            body = payload.read_bytes()
            symbol = payload.stem
            bars, events = parse_yahoo_chart(body, symbol)
            report = validate_market_bars(bars, symbol, date_col="ts", provider=provider)
            if not report.passed:
                raise RuntimeError(f"re-derivation failed validation for {symbol}: {report.errors}")
            parsed[symbol] = {"bars": bars, "events": events}
        normalized = normalize_market_bars(
            parsed, symbol_map, provider, source_uri, snapshot_id
        )
        import io

        buf = io.BytesIO()
        normalized["bars"].write_parquet(buf)
        return buf.getvalue()

    first = derive()
    second = derive()
    registered = registry.artifact_checksum(snapshot_id, str(normalized_dir(domain, provider, snapshot_id) / "bars.parquet"))
    actual = sha256_file(normalized_dir(domain, provider, snapshot_id) / "bars.parquet")

    import hashlib

    digest = hashlib.sha256(first).hexdigest()
    ok = first == second and digest == actual == registered
    print(f"  re-derivation run1==run2: {'PASS' if first == second else 'FAIL'}")
    print(f"  re-derivation == stored artifact: {'PASS' if digest == actual else 'FAIL'}")
    print(f"  artifact == registry checksum: {'PASS' if actual == registered else 'FAIL'}")
    return ok


def reproducibility_check_sec(registry: IngestionRegistry, raw_store: RawStore,
                              snapshot_id: str, provider: str) -> bool:
    """Byte-identical re-derivation for the SEC fundamentals snapshot."""
    import io
    import hashlib

    import polars as pl

    from orbit.ingestion.normalizers.fundamentals import normalize_sec_facts
    from orbit.ingestion.parsing import parse_sec_companyfacts
    from orbit.ingestion.validators import validate_sec_facts

    src_dir = raw_store.snapshot_dir("sec", provider, snapshot_id)
    payloads = sorted(p for p in src_dir.glob("*.json") if p.name != "IMMUTABLE.json")

    def derive() -> bytes:
        frames = []
        for payload in payloads:
            facts = parse_sec_companyfacts(payload.read_bytes())
            cik = int(payload.stem[3:13])
            report = validate_sec_facts(facts, cik)
            if not report.passed:
                raise RuntimeError(f"re-derivation failed SEC validation for {cik}: {report.errors}")
            frames.append(facts)
        out = normalize_sec_facts(pl.concat(frames), provider, snapshot_id)
        buf = io.BytesIO()
        out.write_parquet(buf)
        return buf.getvalue()

    first, second = derive(), derive()
    stored = normalized_dir("fundamentals", provider, snapshot_id) / "facts.parquet"
    actual = sha256_file(stored)
    registered = registry.artifact_checksum(snapshot_id, str(stored))
    digest = hashlib.sha256(first).hexdigest()
    ok = first == second and digest == actual == registered
    print(f"  SEC re-derivation run1==run2: {'PASS' if first == second else 'FAIL'}")
    print(f"  SEC re-derivation == stored artifact: {'PASS' if digest == actual else 'FAIL'}")
    print(f"  SEC artifact == registry checksum: {'PASS' if actual == registered else 'FAIL'}")
    return ok


def reproducibility_check_macro(registry: IngestionRegistry, raw_store: RawStore,
                                snapshot_id: str, provider: str) -> bool:
    """Byte-identical re-derivation for the FRED snapshot."""
    import io
    import hashlib

    import polars as pl

    from orbit.ingestion.normalizers.macro import normalize_fred_series
    from orbit.ingestion.parsing import parse_fred_csv
    from orbit.ingestion.validators import validate_fred_series

    src_dir = raw_store.snapshot_dir("macro", provider, snapshot_id)
    payloads = sorted(p for p in src_dir.glob("*.csv") if p.name != "IMMUTABLE.json")

    def derive() -> bytes:
        frames = []
        for payload in payloads:
            series = parse_fred_csv(payload.read_bytes(), payload.stem)
            report = validate_fred_series(series, payload.stem)
            if not report.passed:
                raise RuntimeError(f"re-derivation failed FRED validation for {payload.stem}")
            frames.append(series)
        out = normalize_fred_series(
            pl.concat(frames), "latest_published_vintage", provider, snapshot_id
        )
        buf = io.BytesIO()
        out.write_parquet(buf)
        return buf.getvalue()

    first, second = derive(), derive()
    stored = normalized_dir("macro", provider, snapshot_id) / "series.parquet"
    actual = sha256_file(stored)
    registered = registry.artifact_checksum(snapshot_id, str(stored))
    digest = hashlib.sha256(first).hexdigest()
    ok = first == second and digest == actual == registered
    print(f"  FRED re-derivation run1==run2: {'PASS' if first == second else 'FAIL'}")
    print(f"  FRED re-derivation == stored artifact: {'PASS' if digest == actual else 'FAIL'}")
    print(f"  FRED artifact == registry checksum: {'PASS' if actual == registered else 'FAIL'}")
    return ok


def main() -> int:
    ensure_layout()
    dev, master, sources = load_dev_config()
    symbols = dev["symbols"]
    symbol_map = {inst["primary_ticker"]: inst["instrument_id"] for inst in master["instruments"]}
    cik_map = {inst["primary_ticker"]: int(inst["cik"]) for inst in master["instruments"]}

    registry = IngestionRegistry(registry_path())
    raw_store = RawStore()
    pipeline = IngestionPipeline(registry, raw_store)

    print("=" * 72)
    print("PHASE 3 RUN: raw ingestion, validation, normalization, provenance")
    print("=" * 72)

    print(f"\n[1] MARKET: {len(symbols)} stocks via {dev['market_provider']}")
    range_cfg = dev.get("date_range")
    request_params = {}
    if range_cfg and len(range_cfg) == 2:
        # Wire the config date_range into explicit epoch params so the
        # config is the single source of truth for the window.
        from datetime import date as _date
        from datetime import datetime as _dt
        from datetime import timezone as _tz

        period1 = int(_dt.combine(_date.fromisoformat(range_cfg[0]), _dt.min.time(), tzinfo=_tz.utc).timestamp())
        period2 = int(_dt.combine(_date.fromisoformat(range_cfg[1]), _dt.min.time(), tzinfo=_tz.utc).timestamp())
        request_params = {"period1": period1, "period2": period2}
    else:
        request_params = {"range": dev.get("range", "30y")}
    market = pipeline.ingest_market(
        YahooChartConnector(),
        symbols,
        symbol_map,
        license_ref="DEVELOPMENT - see configs/sources.json",
        request_params=request_params,
    )
    print("   ", market.summary())
    print(f"    snapshot record: {json.dumps(market.snapshot_record, default=str)[:400]}")
    if market.reconciliation and market.reconciliation.get("finding_count"):
        print(f"    reconciliation: {market.reconciliation['finding_count']} finding(s)")
        for f in market.reconciliation["findings"]:
            print(f"      [{f['code']}] {f['instrument_id']} {f['event_date']}: {f['message'][:110]}")

    print(f"\n[2] SEC EDGAR: {len(symbols)} companies via {dev['fundamentals_provider']}")
    sec = pipeline.ingest_sec(
        SecEdgarConnector(SEC_UA),
        [cik_map[s] for s in symbols],
        license_ref="AUTHORITATIVE - SEC EDGAR public filings",
    )
    print("   ", sec.summary())

    print(f"\n[3] FRED: {len(sources['macro_series'])} series via {dev['macro_provider']}")
    macro = pipeline.ingest_macro(
        FredConnector(),
        sources["macro_series"],
        license_ref="AUTHORITATIVE - FRED public data",
    )
    print("   ", macro.summary())

    print("\n[4] REGISTRY (provenance catalog)")
    for row in registry.dump():
        print(f"    {row['snapshot_id']} {row['domain']}/{row['provider']:28s} "
              f"rows={row['row_count'] or 0:>8d} validation={row['validation_status']}")

    print("\n[5] RAW IMMUTABILITY VERIFY")
    for dom, snap in (("market", market.snapshot_id), ("sec", sec.snapshot_id), ("macro", macro.snapshot_id)):
        ok = raw_store.verify(dom, market.provider if dom == "market" else (sec.provider if dom == "sec" else macro.provider), snap)
        print(f"    {snap} ({dom}): {'PASS' if ok else 'FAIL'}")

    print("\n[6] REPRODUCIBILITY CHECK (same raw snapshot processed twice)")
    ok = reproducibility_check(registry, raw_store, market.snapshot_id, market.provider,
                               "market", symbol_map, market.snapshot_record["source_uri"])
    ok_sec = reproducibility_check_sec(registry, raw_store, sec.snapshot_id, sec.provider)
    ok_macro = reproducibility_check_macro(registry, raw_store, macro.snapshot_id, macro.provider)
    ok = ok and ok_sec and ok_macro
    print(f"    overall: {'PASS' if ok else 'FAIL'}")
    if not ok:
        print("    STOP: do not continue to modeling until this is understood.")

    print("\n[7] DATASET SNAPSHOTS (experiment-pinnable deliveries)")
    for dom, snap, provider, n_inst in (
        ("market", market.snapshot_id, market.provider, len(symbols)),
        ("fundamentals", sec.snapshot_id, sec.provider, len(symbols)),
        ("macro", macro.snapshot_id, macro.provider, len(sources["macro_series"])),
    ):
        ds = build_dataset_snapshot(registry, snap, dom, provider, instrument_count=n_inst)
        print(f"    {ds.snapshot_id} {ds.provider} schema={ds.schema_version} "
              f"range={ds.available_from}..{ds.available_to} rows={ds.row_count} "
              f"validation={ds.validation_status}")

    print(f"\nData root: {data_root()}")
    print("Done.")
    registry.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())