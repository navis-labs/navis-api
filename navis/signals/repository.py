"""SignalRepository — the only writer/reader of the signal_events log.

Idempotency is enforced at the database with ``INSERT ... ON CONFLICT DO
NOTHING`` on the dedup constraint, so concurrent workers ingesting the same fact
never race or error — at most one row wins, the rest are no-ops.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Sequence

from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from navis.signals.base import SignalEvent, SignalSource
from navis.signals.models import SignalDeadLetter, SignalEventRecord


class SignalRepository:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def insert_idempotent(self, event: SignalEvent) -> bool:
        """Insert one event. Returns True if it was newly written, False if the
        (org, source, external_id, external_version) tuple already existed."""
        stmt = (
            pg_insert(SignalEventRecord)
            .values(
                id=event.event_id,
                organization_id=event.org_id,
                user_id=event.user_id,
                integration_id=event.integration_id,
                source=event.source.value,
                event_type=event.event_type.value,
                external_id=event.external_id,
                external_version=event.external_version,
                occurred_at=event.timestamp,
                payload=event.payload,
                metadata=event.metadata,
            )
            .on_conflict_do_nothing(constraint="uq_signal_dedupe")
        )
        result = await self.db.execute(stmt)
        return (result.rowcount or 0) == 1

    async def get(self, event_id: uuid.UUID) -> SignalEventRecord | None:
        return await self.db.get(SignalEventRecord, event_id)

    async def for_replay(
        self,
        org_id: uuid.UUID,
        *,
        source: SignalSource | None = None,
        since: datetime | None = None,
        limit: int = 1000,
    ) -> Sequence[SignalEventRecord]:
        """Read stored signals for replay, ordered oldest-first for deterministic
        re-emission."""
        stmt = select(SignalEventRecord).where(SignalEventRecord.organization_id == org_id)
        if source is not None:
            stmt = stmt.where(SignalEventRecord.source == source.value)
        if since is not None:
            stmt = stmt.where(SignalEventRecord.occurred_at >= since)
        stmt = stmt.order_by(SignalEventRecord.occurred_at.asc()).limit(limit)
        return (await self.db.scalars(stmt)).all()


class DeadLetterRepository:
    """PostgreSQL-backed dead-letter store — the source of truth for poison signals."""

    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def record(
        self,
        *,
        stage: str,
        error: str,
        org_id: uuid.UUID | None = None,
        integration_id: uuid.UUID | None = None,
        source: str | None = None,
        event_id: uuid.UUID | None = None,
        dedupe_key: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> SignalDeadLetter:
        row = SignalDeadLetter(
            organization_id=org_id,
            integration_id=integration_id,
            source=source,
            stage=stage,
            event_id=event_id,
            dedupe_key=dedupe_key,
            error=error[:8000],
            payload=payload or {},
        )
        self.db.add(row)
        await self.db.flush()
        return row

    async def list_unresolved(
        self, org_id: uuid.UUID, *, limit: int = 100
    ) -> Sequence[SignalDeadLetter]:
        stmt = (
            select(SignalDeadLetter)
            .where(
                SignalDeadLetter.organization_id == org_id,
                SignalDeadLetter.resolved_at.is_(None),
            )
            .order_by(SignalDeadLetter.created_at.asc())
            .limit(limit)
        )
        return (await self.db.scalars(stmt)).all()

    async def mark_replayed(self, ids: Sequence[uuid.UUID], *, resolved: bool) -> None:
        if not ids:
            return
        now = datetime.now(timezone.utc)
        values: dict[str, Any] = {"replayed_at": now}
        if resolved:
            values["resolved_at"] = now
        await self.db.execute(
            update(SignalDeadLetter).where(SignalDeadLetter.id.in_(list(ids))).values(**values)
        )
