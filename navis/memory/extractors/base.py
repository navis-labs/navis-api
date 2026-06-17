"""Extractor contract + registry.

An Extractor is a PURE, DETERMINISTIC function from one SignalEvent to zero or
more `MemoryDraft`s. It does no I/O, calls no LLM, reads no clock, and computes
NO intelligence (no health/risk/priority/score). Same signal in → same drafts out,
forever. The service turns drafts into provenanced, deterministically-keyed
MemoryRecords.
"""
from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from navis.memory.envelope import EntityType
from navis.signals.base import SignalEventType, SignalSource


@dataclass(frozen=True, slots=True)
class SignalView:
    """Read-only view of a stored SignalEvent handed to extractors."""

    event_id: uuid.UUID
    org_id: uuid.UUID
    user_id: uuid.UUID | None
    integration_id: uuid.UUID | None
    source: SignalSource
    event_type: SignalEventType
    external_id: str
    external_version: str
    event_timestamp: datetime
    payload: dict[str, Any]


@dataclass(frozen=True, slots=True)
class MemoryDraft:
    """An extractor's output: a fact, before the envelope/provenance is attached."""

    entity_type: EntityType
    entity_natural_key: str  # -> entity_id (stable real-world identity)
    dedup_key: str           # -> memory_id (this specific fact)
    payload: dict[str, Any]
    confidence: float
    schema_version: int


class Extractor(ABC):
    name: str
    version: str
    sources: set[SignalSource]

    @abstractmethod
    def extract(self, signal: SignalView) -> list[MemoryDraft]:
        raise NotImplementedError


# --- Registry ---------------------------------------------------------------
_REGISTRY: list[Extractor] = []


def register(extractor: Extractor) -> Extractor:
    _REGISTRY.append(extractor)
    return extractor


def extractors_for(source: SignalSource) -> list[Extractor]:
    return [e for e in _REGISTRY if source in e.sources]


def run_extractors(signal: SignalView) -> list[MemoryDraft]:
    """Run every extractor registered for this signal's source. Pure + ordered
    (registration order) so the output list is deterministic."""
    drafts: list[MemoryDraft] = []
    for extractor in extractors_for(signal.source):
        drafts.extend(extractor.extract(signal))
    return drafts
