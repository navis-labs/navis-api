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
        self.by_obligation: dict[tuple, uuid.UUID] = {}
        self.events: list[SimpleNamespace] = []
        self.dead_letters: list[SimpleNamespace] = []
        self.inbox: set[tuple] = set()
        # The worker's pre-provider authorization re-check goes through the
        # repository (see ExecutionRepository.authorize). Assign a real
        # `EntitlementService` over a FakeOrgRepo to exercise the genuine
        # decision function — `test_execution_authorization.py` does exactly
        # that, so the gate is validated against production logic and not a
        # stub. None means "this org is fully entitled", which is the right
        # default for the many tests about worker MECHANICS (retry, rate
        # limiting, telemetry, DLQ, read-after-write) that are not about
        # authorization and would otherwise all need an org seeded.
        self.entitlements = None

    async def authorize(self, org_id, capabilities):
        """Mirrors ExecutionRepository.authorize."""
        if self.entitlements is None:
            return True, [], "active"
        return await self.entitlements.can_use_all(org_id, list(capabilities))

    async def upsert_request(self, values: dict[str, Any]):
        idem = (values["organization_id"], values["idempotency_key"])
        if idem in self.by_idem:
            return self.rows[self.by_idem[idem]], False
        # Mirrors the partial unique index uq_execution_obligation: one row per
        # (org, obligation_key) wherever an obligation identity exists. Without
        # this the fake would happily create the duplicate reply the real
        # database refuses, and unit tests would disagree with production.
        obl = values.get("obligation_key")
        if obl:
            key = (values["organization_id"], obl)
            if key in self.by_obligation:
                return self.rows[self.by_obligation[key]], False
        defaults = dict(
            created_at=_now(), updated_at=_now(), approved_at=None, approved_by=None,
            executed_at=None, last_error=None,
            # Mirrors the column default: an unlabelled row is chat-origin, so
            # it stays out of the proactive Action Center queue.
            origin="chat",
        )
        row = SimpleNamespace(**{**defaults, **values})
        self.rows[values["id"]] = row
        self.by_idem[idem] = values["id"]
        if values.get("obligation_key"):
            self.by_obligation[(values["organization_id"], values["obligation_key"])] = (
                values["id"]
            )
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

    async def claim_executable(self, *, max_attempts, limit=100, exclude=None):
        # Mirrors ExecutionRepository.claim_executable, including `exclude` —
        # the worker sets a rate-limited row aside for the pass and must not
        # re-claim it on the next single-row claim.
        skip = set(exclude or ())
        out = []
        for r in self.rows.values():
            if r.id in skip:
                continue
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
        # Mirrors ExecutionRepository.add_dead_letter: at most one UNRESOLVED
        # dead letter per execution, so the fake cannot report a duplicate-free
        # DLQ that production would not deliver.
        if any(
            d.execution_id == req.id and d.resolved_at is None for d in self.dead_letters
        ):
            return
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

    async def inbox_claim(self, consumer, event_id) -> bool:
        key = (consumer, event_id)
        if key in self.inbox:
            return False
        self.inbox.add(key)
        return True

    async def pending_events(self, *, limit=500):
        return [e for e in self.events if e.published_at is None][:limit]

    async def mark_published(self, ids):
        idset = set(ids)
        for e in self.events:
            if e.id in idset:
                e.published_at = _now()


def event_types(repo: FakeExecutionRepo) -> list[str]:
    return [e.event_type for e in repo.events]
