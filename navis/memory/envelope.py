"""The Memory record envelope + entity vocabulary.

`MemoryRecord` is the immutable, fully-provenanced fact. It is a frozen dataclass
so nothing can mutate it in memory; the ORM row mirrors it exactly. `created_at`
is the only non-deterministic field (ingest time) and is excluded from identity.
"""
from __future__ import annotations

import enum
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


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
            tuple(sorted(self.payload.items())),
        )
