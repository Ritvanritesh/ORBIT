"""ORBIT ingestion: raw immutable data, validation, normalization, manifests.

Phase 3 - provider-independent data ingestion. Connectors are swappable
(Yahoo for development, Stooq key-gated, SEC EDGAR for fundamentals,
FRED/ALFRED for macro); the pipeline, checksums, idempotency, and
reproducibility guarantees stay the same regardless of source.
"""

from orbit.ingestion.checksums import request_fingerprint, sha256_bytes, sha256_file
from orbit.ingestion.manifests import Manifest, build_manifest
from orbit.ingestion.pipeline import IngestResult, IngestionPipeline
from orbit.ingestion.registry import IngestionRegistry
from orbit.ingestion.snapshot import MarketDataAccessor
from orbit.ingestion.storage import RawStore
from orbit.ingestion.validators import Issue, ValidationReport

__all__ = [
    "IngestResult",
    "IngestionPipeline",
    "IngestionRegistry",
    "Issue",
    "Manifest",
    "MarketDataAccessor",
    "RawStore",
    "ValidationReport",
    "build_manifest",
    "request_fingerprint",
    "sha256_bytes",
    "sha256_file",
]