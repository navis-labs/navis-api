"""ExecutionRepository — the only reader/writer of execution tables.

Idempotent request insert (ON CONFLICT on the deterministic idempotency key) and
worker claim via FOR UPDATE SKIP LOCKED (no global lock, horizontally scalable).
Outbox events are appended in the caller's transaction (outbox pattern).
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Sequence

from sqlalchemy import and_, func, or_, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from navis.execution.models import (
    ExecutionDeadLetter,
    ExecutionInbox,
    ExecutionOutbox,
    ExecutionRequest,
)
from navis.execution.status import ExecutionStatus


class ExecutionRepository:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def upsert_request(self, values: dict[str, Any]) -> tuple[ExecutionRequest, bool]:
        """Insert if absent (idempotent). Returns (row, created)."""
        stmt = (
            pg_insert(ExecutionRequest)
            .values(**values)
            .on_conflict_do_nothing(constraint="uq_execution_idempotency")
        )
        result = await self.db.execute(stmt)
        created = (result.rowcount or 0) == 1
        # id is deterministic (uuid5 of org+key), so the existing row has the same id.
        row = await self.db.get(ExecutionRequest, values["id"])
        return row, created

    async def get(self, execution_id: uuid.UUID) -> ExecutionRequest | None:
        return await self.db.get(ExecutionRequest, execution_id)

    async def add_event(
        self, execution_id: uuid.UUID, org_id: uuid.UUID, event_type: str, payload: dict
    ) -> None:
        self.db.add(
            ExecutionOutbox(
                organization_id=org_id,
                execution_id=execution_id,
                event_type=event_type,
                payload=payload,
            )
        )

    async def flush(self) -> None:
        await self.db.flush()

    async def claim_approved(self, *, limit: int = 100) -> Sequence[ExecutionRequest]:
        """Claim APPROVED requests for execution without global locks."""
        stmt = (
            select(ExecutionRequest)
            .where(ExecutionRequest.status == ExecutionStatus.APPROVED.value)
            .order_by(ExecutionRequest.created_at.asc())
            .limit(limit)
            .with_for_update(skip_locked=True)
        )
        return (await self.db.scalars(stmt)).all()

    async def pending_events(self, *, limit: int = 500) -> Sequence[ExecutionOutbox]:
        stmt = (
            select(ExecutionOutbox)
            .where(ExecutionOutbox.published_at.is_(None))
            .order_by(ExecutionOutbox.created_at.asc())
            .limit(limit)
            .with_for_update(skip_locked=True)
        )
        return (await self.db.scalars(stmt)).all()

    async def mark_published(self, ids: Sequence[uuid.UUID]) -> None:
        if not ids:
            return
        await self.db.execute(
            update(ExecutionOutbox)
            .where(ExecutionOutbox.id.in_(list(ids)))
            .values(published_at=datetime.now(timezone.utc))
        )

    # ---- hardening: executable claim (APPROVED or retryable FAILED) ----

    async def claim_executable(
        self, *, max_attempts: int, limit: int = 100
    ) -> Sequence[ExecutionRequest]:
        """Claim work: APPROVED, or FAILED with attempts remaining (retry)."""
        stmt = (
            select(ExecutionRequest)
            .where(
                or_(
                    ExecutionRequest.status == ExecutionStatus.APPROVED.value,
                    and_(
                        ExecutionRequest.status == ExecutionStatus.FAILED.value,
                        ExecutionRequest.attempts < max_attempts,
                    ),
                )
            )
            .order_by(ExecutionRequest.created_at.asc())
            .limit(limit)
            .with_for_update(skip_locked=True)
        )
        return (await self.db.scalars(stmt)).all()

    # ---- hardening: stale EXECUTING recovery ----

    async def claim_stale_executing(
        self, *, older_than_seconds: int, limit: int = 100
    ) -> Sequence[ExecutionRequest]:
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=older_than_seconds)
        stmt = (
            select(ExecutionRequest)
            .where(
                ExecutionRequest.status == ExecutionStatus.EXECUTING.value,
                ExecutionRequest.updated_at < cutoff,
            )
            .order_by(ExecutionRequest.updated_at.asc())
            .limit(limit)
            .with_for_update(skip_locked=True)
        )
        return (await self.db.scalars(stmt)).all()

    # ---- hardening: dead letters ----

    async def add_dead_letter(
        self, req: ExecutionRequest, *, provider: str | None, error: str
    ) -> None:
        self.db.add(
            ExecutionDeadLetter(
                organization_id=req.organization_id,
                execution_id=req.id,
                provider=provider,
                action_type=req.action_type,
                error=error[:4000],
                attempts=req.attempts or 0,
                payload=req.payload,
            )
        )

    async def unresolved_dead_letters(
        self, org_id: uuid.UUID | None = None, *, limit: int = 100
    ) -> Sequence[ExecutionDeadLetter]:
        stmt = select(ExecutionDeadLetter).where(ExecutionDeadLetter.resolved_at.is_(None))
        if org_id is not None:
            stmt = stmt.where(ExecutionDeadLetter.organization_id == org_id)
        return (await self.db.scalars(stmt.limit(limit))).all()

    async def mark_dead_letter_replayed(self, dlq_id: uuid.UUID) -> None:
        now = datetime.now(timezone.utc)
        await self.db.execute(
            update(ExecutionDeadLetter)
            .where(ExecutionDeadLetter.id == dlq_id)
            .values(replayed_at=now, resolved_at=now)
        )

    # ---- hardening: inbox (idempotent consumer) ----

    async def inbox_claim(self, consumer: str, event_id: uuid.UUID) -> bool:
        """True if this consumer is seeing event_id for the FIRST time."""
        stmt = (
            pg_insert(ExecutionInbox)
            .values(consumer=consumer, event_id=event_id)
            .on_conflict_do_nothing(index_elements=["consumer", "event_id"])
        )
        result = await self.db.execute(stmt)
        return (result.rowcount or 0) == 1

    # ---- hardening: outbox observability ----

    async def outbox_stats(self) -> dict[str, Any]:
        pending = await self.db.scalar(
            select(func.count(ExecutionOutbox.id)).where(ExecutionOutbox.published_at.is_(None))
        )
        oldest = await self.db.scalar(
            select(func.min(ExecutionOutbox.created_at)).where(
                ExecutionOutbox.published_at.is_(None)
            )
        )
        lag = (datetime.now(timezone.utc) - oldest).total_seconds() if oldest else 0.0
        return {"pending": int(pending or 0), "oldest_lag_seconds": round(lag, 1)}
