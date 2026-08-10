"""Fakes + fixed-id builders so Memory V2 is testable without DB or Signals."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from navis.memory.envelope import MemoryRecord
from navis.memory.extractors import SignalView
from navis.signals.base import SignalEventType, SignalSource

# FIXED ids so two builds of "the same signal" use identical inputs (determinism).
ORG = uuid.UUID("11111111-1111-1111-1111-111111111111")
USER = uuid.UUID("22222222-2222-2222-2222-222222222222")
INTEG = uuid.UUID("33333333-3333-3333-3333-333333333333")
EVENT = uuid.UUID("44444444-4444-4444-4444-444444444444")
EVENT_TS = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)


def make_view(
    *,
    source: SignalSource,
    event_type: SignalEventType,
    payload: dict[str, Any],
    event_id: uuid.UUID = EVENT,
    external_id: str = "ext-1",
    external_version: str = "1",
    org_id: uuid.UUID = ORG,
    # Overridable so tests can pin WHEN a message was sent. Due-date
    # extraction resolves relative deadlines ("by Friday") against this,
    # not against the wall clock.
    event_timestamp: datetime = EVENT_TS,
) -> SignalView:
    return SignalView(
        event_id=event_id,
        org_id=org_id,
        user_id=USER,
        integration_id=INTEG,
        source=source,
        event_type=event_type,
        external_id=external_id,
        external_version=external_version,
        event_timestamp=event_timestamp,
        payload=payload,
    )


def gmail_view(**overrides) -> SignalView:
    payload = {
        "from": "Alice Founder <alice@acme.com>",
        "to": "Bob Investor <bob@vc.com>",
        "subject": "Re: terms",
        "thread_id": "thread-9",
        "body_text": "Thanks Bob. I'll send you the revised deck by Friday. We have decided to raise.",
    }
    payload.update(overrides.pop("payload", {}))
    return make_view(
        source=SignalSource.GMAIL,
        event_type=SignalEventType.EMAIL_RECEIVED,
        payload=payload,
        external_id="msg-1",
        **overrides,
    )


def calendar_view(**overrides) -> SignalView:
    payload = {
        "summary": "Board sync",
        "status": "confirmed",
        "start": {"dateTime": "2026-06-02T10:00:00Z"},
        "end": {"dateTime": "2026-06-02T11:00:00Z"},
        "attendees": [{"email": "alice@acme.com", "displayName": "Alice"}, {"email": "bob@vc.com"}],
        "hangout_link": "https://meet.google.com/x",
    }
    payload.update(overrides.pop("payload", {}))
    return make_view(
        source=SignalSource.CALENDAR,
        event_type=SignalEventType.CALENDAR_EVENT_CREATED,
        payload=payload,
        external_id="evt-1",
        external_version="etag-7",
        **overrides,
    )


def drive_view(**overrides) -> SignalView:
    payload = {"name": "Q3 Plan", "mime_type": "application/pdf", "parents": ["folder-42"]}
    payload.update(overrides.pop("payload", {}))
    return make_view(
        source=SignalSource.DRIVE,
        event_type=SignalEventType.DRIVE_FILE_UPDATED,
        payload=payload,
        external_id="file-1",
        external_version="v3",
        **overrides,
    )


def slack_message_view(**overrides) -> SignalView:
    payload = {
        "channel_id": "C123", "slack_user_id": "U777", "ts": "1700000000.001",
        "thread_root_ts": None,
        "text": "I'll send the report by Friday. We have decided to ship v2.",
    }
    payload.update(overrides.pop("payload", {}))
    return make_view(
        source=SignalSource.SLACK, event_type=SignalEventType.MESSAGE_SENT,
        payload=payload, external_id="C123:1700000000.001", **overrides,
    )


def slack_user_snapshot_view(**overrides) -> SignalView:
    payload = {"slack_user_id": "U777", "email": "alice@acme.com", "display_name": "Alice", "is_internal": True}
    payload.update(overrides.pop("payload", {}))
    return make_view(
        source=SignalSource.SLACK, event_type=SignalEventType.SLACK_USER_SNAPSHOT,
        payload=payload, external_id="U777", **overrides,
    )


def slack_channel_snapshot_view(**overrides) -> SignalView:
    payload = {"channel_id": "C123", "name": "project-zeus", "visibility": "private"}
    payload.update(overrides.pop("payload", {}))
    return make_view(
        source=SignalSource.SLACK, event_type=SignalEventType.SLACK_CHANNEL_SNAPSHOT,
        payload=payload, external_id="C123", **overrides,
    )


def docs_view(**overrides) -> SignalView:
    payload = {"name": "Strategy Doc", "owners": [{"email": "alice@acme.com"}], "parents": ["folder-9"], "drive_id": "D1"}
    payload.update(overrides.pop("payload", {}))
    return make_view(
        source=SignalSource.DOCS, event_type=SignalEventType.DOC_UPDATED,
        payload=payload, external_id="doc-1", external_version="v2", **overrides,
    )


def meet_view(**overrides) -> SignalView:
    payload = {
        "summary": "Standup", "start": {"dateTime": "2026-06-02T10:00:00Z"},
        "end": {"dateTime": "2026-06-02T10:15:00Z"},
        "attendees": [{"email": "alice@acme.com"}], "conference_id": "conf-xyz",
    }
    payload.update(overrides.pop("payload", {}))
    return make_view(
        source=SignalSource.MEET, event_type=SignalEventType.MEETING_CREATED,
        payload=payload, external_id="evt-1", external_version="100:meeting.created", **overrides,
    )


class FakeMemoryRepo:
    """Append-only with idempotent dedup on memory_id (mimics ON CONFLICT).

    Also captures outbox pointer rows so tests can assert that memory-derived
    domain events would be staged for publish (mirrors the real repo's
    transactional-outbox contract without needing a DB session)."""

    def __init__(self) -> None:
        self.seen: set[uuid.UUID] = set()
        self.records: list[MemoryRecord] = []
        self.outbox: list[dict[str, Any]] = []

    async def append(self, record: MemoryRecord) -> bool:
        if record.memory_id in self.seen:
            return False
        self.seen.add(record.memory_id)
        self.records.append(record)
        return True

    async def append_outbox(self, rows) -> None:
        self.outbox.extend(rows)
