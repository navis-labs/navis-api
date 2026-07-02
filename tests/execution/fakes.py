"""In-memory fakes mirroring ExecutionRepository (idempotent insert, outbox, claim,
DLQ, inbox) — for unit tests. Real-DB behavior is exercised in the skip-guarded
Postgres integration suite."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

from navis.execution.status import ExecutionStatus

ORG_A = uuid.UUID("11111111-1111-1111-1111-111111111111")
ORG_B = uuid.UUID("22222222-2222-2222-2222-222222222222")
DECISION = uuid.UUID("33333333-3333-3333-3333-333333333333")
USER = uuid.UUID("44444444-4444-4444-4444-444444444444")


def _now() -> datetime:
    return datetime.now(timezone.utc)


class FakeExecutionRepo:
    def __init__(self) -> None:
        self.rows: dict[uuid.UUID, SimpleNamespace] = {}
        self.by_idem: dict[tuple, uuid.UUID] = {}
        self.events: list[SimpleNamespace] = []
        self.dead_letters: list[SimpleNamespace] = []
        self.log: list[str] = []  # ordering probe: commits (and test-injected markers)

    async def upsert_request(self, values: dict[str, Any]):
        idem = (values["organization_id"], values["idempotency_key"])
        if idem in self.by_idem:
            return self.rows[self.by_idem[idem]], False
        defaults = dict(
            created_at=_now(), updated_at=_now(), approved_at=None, approved_by=None,
            executed_at=None, last_error=None,
        )
        row = SimpleNamespace(**{**defaults, **values})
        self.rows[values["id"]] = row
        self.by_idem[idem] = values["id"]
        return row, True

    async def get(self, execution_id):
        return self.rows.get(execution_id)

    async def add_event(self, execution_id, org_id, event_type, payload):
        self.events.append(
            SimpleNamespace(
                id=uuid.uuid4(), organization_id=org_id, execution_id=execution_id,
                event_type=event_type, payload=payload, published_at=None, created_at=_now(),
            )
        )

    async def flush(self):
        pass

    async def commit(self):
        self.log.append("commit")

    async def claim_executable(self, *, max_attempts, limit=100):
        out = []
        for r in self.rows.values():
            if r.status == ExecutionStatus.APPROVED.value:
                out.append(r)
            elif r.status == ExecutionStatus.FAILED.value and (r.attempts or 0) < max_attempts:
                out.append(r)
        return out[:limit]

    async def claim_stale_executing(self, *, older_than_seconds, limit=100):
        cutoff = _now().timestamp() - older_than_seconds
        return [
            r for r in self.rows.values()
            if r.status == ExecutionStatus.EXECUTING.value and r.updated_at.timestamp() < cutoff
        ][:limit]

    async def add_dead_letter(self, req, *, provider, error):
        self.dead_letters.append(
            SimpleNamespace(
                id=uuid.uuid4(), organization_id=req.organization_id, execution_id=req.id,
                provider=provider, action_type=req.action_type, error=error,
                attempts=req.attempts or 0, payload=req.payload, replayed_at=None, resolved_at=None,
            )
        )

    async def unresolved_dead_letters(self, org_id=None, *, limit=100):
        rows = [d for d in self.dead_letters if d.resolved_at is None]
        if org_id is not None:
            rows = [d for d in rows if d.organization_id == org_id]
        return rows[:limit]

    async def mark_dead_letter_replayed(self, dlq_id):
        for d in self.dead_letters:
            if d.id == dlq_id:
                d.replayed_at = d.resolved_at = _now()

    async def mark_dead_letter_resolved(self, dlq_id):
        for d in self.dead_letters:
            if d.id == dlq_id:
                d.resolved_at = _now()

    async def pending_events(self, *, limit=500):
        return [e for e in self.events if e.published_at is None][:limit]

    async def mark_published(self, ids):
        idset = set(ids)
        for e in self.events:
            if e.id in idset:
                e.published_at = _now()


def event_types(repo: FakeExecutionRepo) -> list[str]:
    return [e.event_type for e in repo.events]
