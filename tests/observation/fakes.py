"""In-memory fakes for Observation V2 (append-only outcome log + inbox)."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

from navis.platform.events import DomainEvent, EventType

ORG_A = uuid.UUID("11111111-1111-1111-1111-111111111111")
ORG_B = uuid.UUID("22222222-2222-2222-2222-222222222222")
EXECUTION = uuid.UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
DECISION = uuid.UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")


def _now() -> datetime:
    return datetime.now(timezone.utc)


class FakeOutcomeRepo:
    def __init__(self) -> None:
        self.rows: dict[uuid.UUID, SimpleNamespace] = {}
        self.inbox: set[tuple] = set()

    async def append(self, values: dict[str, Any]):
        oid = values["outcome_id"]
        if oid in self.rows:
            return self.rows[oid], False
        row = SimpleNamespace(created_at=_now(), **values)
        self.rows[oid] = row
        return row, True

    async def inbox_claim(self, consumer: str, event_id: str) -> bool:
        key = (consumer, event_id)
        if key in self.inbox:
            return False
        self.inbox.add(key)
        return True

    async def by_provider_external_id(self, provider, external_id, *, org_id=None):
        matches = [
            r for r in self.rows.values()
            if (r.provider_reference or {}).get("provider") == provider
            and (r.provider_reference or {}).get("external_id") == external_id
        ]
        if org_id is not None:
            matches = [r for r in matches if r.organization_id == org_id]
        matches.sort(key=lambda r: r.occurred_at, reverse=True)
        return matches[0] if matches else None

    async def timeline(self, org_id, *, execution_id=None, decision_id=None):
        rows = [r for r in self.rows.values() if r.organization_id == org_id]
        if execution_id is not None:
            rows = [r for r in rows if r.execution_id == execution_id]
        if decision_id is not None:
            rows = [r for r in rows if r.decision_id == decision_id]
        return sorted(rows, key=lambda r: (r.occurred_at, r.created_at))

    async def for_replay(self, org_id, *, since=None, limit=1000):
        rows = [r for r in self.rows.values() if r.organization_id == org_id]
        if since is not None:
            rows = [r for r in rows if r.occurred_at >= since]
        return sorted(rows, key=lambda r: (r.occurred_at, r.created_at))[:limit]


class FakePublisher:
    def __init__(self) -> None:
        self.events: list[DomainEvent] = []

    async def __call__(self, event: DomainEvent) -> None:
        self.events.append(event)


def action_event(
    event_type: EventType,
    *,
    org_id: uuid.UUID = ORG_A,
    event_id: str = "evt-1",
    execution_id: uuid.UUID = EXECUTION,
    decision_id: uuid.UUID = DECISION,
    signals: list[str] | None = None,
    provider_ref: dict | None = None,
    error: str | None = None,
) -> DomainEvent:
    payload = {
        "event_id": event_id,
        "execution_id": str(execution_id),
        "decision_id": str(decision_id),
        "action_type": "send_email",
        "status": "executed" if event_type == EventType.ACTION_EXECUTED else "failed",
        "reasoning_signal_ids": signals or ["s1", "s2"],
        "reasoning_version": 1,
        "state_watermark": "wm-9",
    }
    if event_type == EventType.ACTION_EXECUTED:
        payload["outcome"] = {"provider_reference": provider_ref or {"provider": "gmail", "external_id": "m-1"}}
    if error:
        payload["error"] = error
    return DomainEvent(type=event_type, org_id=org_id, source="execution", payload=payload)
