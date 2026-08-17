"""Provider connector contracts.

A provider connector turns one *source request* into raw payloads exactly as
the provider delivered them (bytes + the URL they came from). Connectors know
nothing about storage, validation, or normalization - the ingestion pipeline
owns those layers, so a provider can be swapped without redesigning ORBIT.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


class ProviderUnavailable(RuntimeError):
    """Raised when a source cannot currently be reached or used."""


@dataclass(frozen=True)
class RawObject:
    """One immutable unit of source data as delivered by the provider."""

    filename: str
    body: bytes
    source_uri: str
    content_type: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


class ProviderConnector(Protocol):
    """Fetches raw payloads for a domain (market / sec / macro)."""

    provider_name: str

    def fetch(self, request: dict[str, Any]) -> list[RawObject]:
        """Download everything one request asks for, or raise ProviderUnavailable."""