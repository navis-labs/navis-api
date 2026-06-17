"""The SignalEvent contract + connector interface.

A SignalEvent is the ONLY output of the signals domain. It is immutable,
version-aware, globally unique, and idempotent by construction. Signals capture
reality and nothing else — no Memory/State/Reasoning/LLM/business logic here.

Idempotency model
-----------------
Dedup identity is ``(org_id, source, external_id, external_version)``. The
``event_id`` is a deterministic UUIDv5 of that tuple, so re-capturing the same
fact yields the same id — safe to replay, safe to retry.
"""
from __future__ import annotations

import enum
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

# Fixed namespace so event_id is stable across processes/replays.
_EVENT_NAMESPACE = uuid.UUID("6f3c2d1a-9b4e-5c7a-8d2f-1e0a3b6c9d22")


class SignalSource(str, enum.Enum):
    GMAIL = "gmail"
    CALENDAR = "calendar"
    DRIVE = "drive"
    DOCS = "docs"
    MEET = "meet"
    SLACK = "slack"
    NOTION = "notion"
    CRM = "crm"


class SignalEventType(str, enum.Enum):
    """Canonical fact types. Still pure reality capture — the verb that happened,
    not any interpretation of it."""

    # Gmail
    EMAIL_RECEIVED = "email.received"
    # Calendar
    CALENDAR_EVENT_CREATED = "calendar.event.created"
    CALENDAR_EVENT_UPDATED = "calendar.event.updated"
    CALENDAR_EVENT_CANCELLED = "calendar.event.cancelled"
    # Drive
    DRIVE_FILE_CREATED = "drive.file.created"
    DRIVE_FILE_UPDATED = "drive.file.updated"
    DRIVE_FILE_MOVED = "drive.file.moved"
    DRIVE_FILE_SHARED = "drive.file.shared"
    DRIVE_PERMISSION_CHANGED = "drive.permission.changed"
    DRIVE_FILE_TRASHED = "drive.file.trashed"
    # Docs (derived from Drive)
    DOC_CREATED = "doc.created"
    DOC_UPDATED = "doc.updated"
    DOC_MOVED = "doc.moved"
    # Meet (derived from Calendar)
    MEETING_CREATED = "meeting.created"
    MEETING_UPDATED = "meeting.updated"
    MEETING_COMPLETED = "meeting.completed"


class SignalValidationError(ValueError):
    """Raised when a SignalEvent would violate the contract (e.g. tenant-agnostic)."""


@dataclass(frozen=True, slots=True)
class SignalContext:
    """The tenancy triple every signal must carry. No tenant-agnostic signals."""

    org_id: uuid.UUID
    user_id: uuid.UUID
    integration_id: uuid.UUID

    def validate(self) -> None:
        if not self.org_id:
            raise SignalValidationError("signal missing org_id")
        if not self.user_id:
            raise SignalValidationError("signal missing user_id")
        if not self.integration_id:
            raise SignalValidationError("signal missing integration_id")


@dataclass(frozen=True, slots=True)
class SignalEvent:
    """Immutable, append-only fact captured from reality."""

    event_id: uuid.UUID
    org_id: uuid.UUID
    user_id: uuid.UUID
    integration_id: uuid.UUID
    source: SignalSource
    event_type: SignalEventType
    external_id: str
    external_version: str
    timestamp: datetime
    payload: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def make_dedupe_key(
        org_id: uuid.UUID, source: SignalSource, external_id: str, external_version: str
    ) -> str:
        return f"{org_id}:{source.value}:{external_id}:{external_version}"

    @property
    def dedupe_key(self) -> str:
        return self.make_dedupe_key(
            self.org_id, self.source, self.external_id, self.external_version
        )

    @classmethod
    def create(
        cls,
        ctx: SignalContext,
        *,
        source: SignalSource,
        event_type: SignalEventType,
        external_id: str,
        external_version: str,
        timestamp: datetime | None = None,
        payload: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "SignalEvent":
        ctx.validate()
        if not external_id:
            raise SignalValidationError("signal missing external_id")
        if external_version in (None, ""):
            raise SignalValidationError("signal missing external_version (version-aware)")

        dedupe = cls.make_dedupe_key(ctx.org_id, source, external_id, str(external_version))
        return cls(
            event_id=uuid.uuid5(_EVENT_NAMESPACE, dedupe),
            org_id=ctx.org_id,
            user_id=ctx.user_id,
            integration_id=ctx.integration_id,
            source=source,
            event_type=event_type,
            external_id=external_id,
            external_version=str(external_version),
            timestamp=timestamp or datetime.now(timezone.utc),
            payload=payload or {},
            metadata=metadata or {},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": str(self.event_id),
            "org_id": str(self.org_id),
            "user_id": str(self.user_id),
            "integration_id": str(self.integration_id),
            "source": self.source.value,
            "event_type": self.event_type.value,
            "external_id": self.external_id,
            "external_version": self.external_version,
            "timestamp": self.timestamp.isoformat(),
            "payload": self.payload,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SignalEvent":
        """Reconstruct an event from its stored form (e.g. DLQ replay)."""
        return cls(
            event_id=uuid.UUID(data["event_id"]),
            org_id=uuid.UUID(data["org_id"]),
            user_id=uuid.UUID(data["user_id"]),
            integration_id=uuid.UUID(data["integration_id"]),
            source=SignalSource(data["source"]),
            event_type=SignalEventType(data["event_type"]),
            external_id=data["external_id"],
            external_version=data["external_version"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            payload=data.get("payload") or {},
            metadata=data.get("metadata") or {},
        )


@dataclass(slots=True)
class SyncResult:
    """What a connector returns from a sync pass.

    ``events`` are captured facts. ``cursor`` is the new opaque sync state to
    persist on the integration. ``full_resync_required`` signals the caller to
    drop the cursor and run a full sync (e.g. Gmail 404 / Calendar 410).
    """

    events: list[SignalEvent] = field(default_factory=list)
    cursor: dict[str, Any] = field(default_factory=dict)
    full_resync_required: bool = False


class WorkspaceConnector(ABC):
    """A per-source Google connector. Pure capture: given a client + context +
    cursor, produce SignalEvents and the next cursor. Persistence, retries, and
    DLQ are owned by the sync engine — connectors stay deterministic/testable.
    """

    source: SignalSource

    @abstractmethod
    async def full_sync(self, ctx: SignalContext, client: Any) -> SyncResult:
        """Bounded historical capture used at onboarding / cursor recovery."""

    @abstractmethod
    async def incremental_sync(
        self, ctx: SignalContext, client: Any, cursor: dict[str, Any]
    ) -> SyncResult:
        """Capture everything since the stored cursor."""

    async def create_watch(self, ctx: SignalContext, client: Any) -> dict[str, Any]:
        """Create/renew a push channel. Returns watch state to persist. Optional."""
        return {}
