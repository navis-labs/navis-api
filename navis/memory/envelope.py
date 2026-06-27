"""The Memory record envelope + entity vocabulary.

`MemoryRecord` is the immutable, fully-provenanced fact. It is a frozen dataclass
so nothing can mutate it in memory; the ORM row mirrors it exactly. `created_at`
is the only non-deterministic field (ingest time) and is excluded from identity.
"""
from __future__ import annotations

import enum
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


def _canonical_payload(payload: dict[str, Any]) -> str:
    """Canonical, deterministic serialization of a payload for identity comparison.

    `sort_keys=True` recursively orders EVERY level (robust to nested-dict order and to JSONB
    key-order normalization on round-trip — unlike a shallow `sorted(items())`). The result is
    a hashable string (payloads contain lists like `attendees`/`owners`, which would make a
    tuple-of-items unhashable). List ORDER is preserved (it is semantic). `default=str` makes
    any non-JSON value (e.g. datetime) deterministic instead of raising."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


class EntityType(str, enum.Enum):
    PERSON = "person"
    COMPANY = "company"
    CUSTOMER = "customer"
    PROJECT = "project"
    MEETING = "meeting"
    THREAD = "thread"
    RELATIONSHIP = "relationship"
    COMMITMENT = "commitment"
    DECISION = "decision"
    CHANNEL = "channel"      # Slack/chat channel as a first-class entity
    DOCUMENT = "document"    # Google Doc / Drive document


@dataclass(frozen=True, slots=True)
class MemoryRecord:
    memory_id: uuid.UUID
    org_id: uuid.UUID
    entity_type: EntityType
    entity_id: uuid.UUID
    source_signal_event_id: uuid.UUID
    source_integration_id: uuid.UUID | None
    event_timestamp: datetime
    extractor_name: str
    extractor_version: str
    schema_version: int
    confidence: float
    dedup_key: str
    payload: dict[str, Any] = field(default_factory=dict)
    created_at: datetime | None = None  # set by the DB; not part of identity

    def identity_tuple(self) -> tuple:
        """Everything that must be byte-identical across replays (excludes created_at)."""
        return (
            str(self.memory_id),
            str(self.org_id),
            self.entity_type.value,
            str(self.entity_id),
            str(self.source_signal_event_id),
            self.extractor_name,
            self.extractor_version,
            self.schema_version,
            self.dedup_key,
            _canonical_payload(self.payload),
        )
