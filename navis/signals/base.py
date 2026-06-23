"""The SignalEvent contract + connector interface.

A SignalEvent is the ONLY output of the signals domain. It is immutable,
version-aware, globally unique, and idempotent by construction. Signals capture
reality and nothing else — no Memory/State/Reasoning/LLM/business logic here.

Idempotency model
-----------------
Dedup identity is ``(org_id, integration_id, source, external_id, external_version)``.
The ``event_id`` is a deterministic UUIDv5 of that tuple, so re-capturing the same
fact yields the same id — safe to replay, safe to retry.

``integration_id`` is in the identity because a provider's ``external_id`` is only
unique WITHIN one connected account, not across an org. Google Drive file ids and
Calendar event ids are global — a file/meeting shared between two colleagues has the
SAME external_id in both accounts — so an org-only identity would collapse the two
observations into one and silently drop the second person's edge from the graph. Keying
on the observing integration makes each account's capture its own fact.
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

# Bump on a NON-additive change to the SignalEvent wire/storage shape. Stamped on every
# event so consumers and from_dict() can branch by version for safe long-horizon replay
# (a 3-year-old DLQ row may predate fields added later). Additive fields don't need a bump.
SIGNAL_SCHEMA_VERSION = 1


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
    # Slack (Events API + backfill — same shape from both paths). Reality only: the verb
    # that happened in the workspace, never an interpretation of it.
    MESSAGE_SENT = "slack.message.sent"
    MESSAGE_EDITED = "slack.message.edited"
    MESSAGE_DELETED = "slack.message.deleted"
    THREAD_REPLY = "slack.thread.reply"
    REACTION_ADDED = "slack.reaction.added"
    REACTION_REMOVED = "slack.reaction.removed"
    CHANNEL_CREATED = "slack.channel.created"
    CHANNEL_ARCHIVED = "slack.channel.archived"
    CHANNEL_UNARCHIVED = "slack.channel.unarchived"
    CHANNEL_RENAMED = "slack.channel.renamed"
    FILE_SHARED = "slack.file.shared"
    USER_JOINED_CHANNEL = "slack.user.joined_channel"
    USER_LEFT_CHANNEL = "slack.user.left_channel"
    APP_MENTION = "slack.app.mention"
    # Identity snapshots (periodic) — replayable, PII-minimized reality of who/what exists,
    # so message signals never have to denormalize names/emails into every row.
    SLACK_USER_SNAPSHOT = "slack.user.snapshot"
    SLACK_CHANNEL_SNAPSHOT = "slack.channel.snapshot"


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
    # When the fact was OBSERVED/captured — distinct from `timestamp` (when it happened)
    # and from the row's created_at (when it was persisted). Defaults to capture time in
    # create(); for legacy events it falls back to `timestamp`. NOT part of the dedupe key.
    captured_at: datetime | None = None
    # Event-schema version (see SIGNAL_SCHEMA_VERSION). NOT part of the dedupe key, so the
    # deterministic event_id is unaffected by schema evolution.
    schema_version: int = SIGNAL_SCHEMA_VERSION

    @staticmethod
    def make_dedupe_key(
        org_id: uuid.UUID,
        integration_id: uuid.UUID,
        source: SignalSource,
        external_id: str,
        external_version: str,
    ) -> str:
        return f"{org_id}:{integration_id}:{source.value}:{external_id}:{external_version}"

    @property
    def dedupe_key(self) -> str:
        return self.make_dedupe_key(
            self.org_id, self.integration_id, self.source, self.external_id, self.external_version
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

        dedupe = cls.make_dedupe_key(
            ctx.org_id, ctx.integration_id, source, external_id, str(external_version)
        )
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
            captured_at=datetime.now(timezone.utc),
            schema_version=SIGNAL_SCHEMA_VERSION,
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
            "captured_at": (self.captured_at or self.timestamp).isoformat(),
            "schema_version": self.schema_version,
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
            # Backward compatible: legacy rows have neither field. captured_at falls back to
            # the occurrence time (closest available); schema_version defaults to v1.
            captured_at=(
                datetime.fromisoformat(data["captured_at"])
                if data.get("captured_at")
                else datetime.fromisoformat(data["timestamp"])
            ),
            schema_version=int(data.get("schema_version", SIGNAL_SCHEMA_VERSION)),
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
