"""OutcomeRepository — the only writer/reader of the append-only outcome log.

Append-only: exactly one write op (`append`), idempotent via ON CONFLICT on the
deterministic outcome_id. Plus the inbox for consumer-level dedupe, and lookups
by provider_reference (for webhooks) and by execution/decision (for timelines).
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Sequence

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from navis.observation.models import ObservationInbox, Outcome


class OutcomeRepository:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def append(self, values: dict[str, Any]) -> tuple[Outcome, bool]:
        """Insert one immutable outcome. Returns (row, created)."""
        stmt = (
            pg_insert(Outcome)
            .values(**values)
            .on_conflict_do_nothing(index_elements=["outcome_id"])
        )
        result = await self.db.execute(stmt)
        created = (result.rowcount or 0) == 1
        row = await self.db.get(Outcome, values["outcome_id"])
        return row, created

    async def inbox_claim(self, consumer: str, event_id: str) -> bool:
        stmt = (
            pg_insert(ObservationInbox)
            .values(consumer=consumer, event_id=event_id)
            .on_conflict_do_nothing(index_elements=["consumer", "event_id"])
        )
        result = await self.db.execute(stmt)
        return (result.rowcount or 0) == 1

    async def by_provider_external_id(
        self, provider: str, external_id: str, *, org_id: uuid.UUID | None = None
    ) -> Outcome | None:
        """Find the most recent outcome whose provider_reference matches (webhook lookup)."""
        stmt = select(Outcome).where(
            Outcome.provider_reference["provider"].astext == provider,
            Outcome.provider_reference["external_id"].astext == external_id,
        )
        if org_id is not None:
            stmt = stmt.where(Outcome.organization_id == org_id)
        stmt = stmt.order_by(Outcome.occurred_at.desc()).limit(1)
        return await self.db.scalar(stmt)

    async def timeline(
        self,
        org_id: uuid.UUID,
        *,
        execution_id: uuid.UUID | None = None,
        decision_id: uuid.UUID | None = None,
    ) -> Sequence[Outcome]:
        stmt = select(Outcome).where(Outcome.organization_id == org_id)
        if execution_id is not None:
            stmt = stmt.where(Outcome.execution_id == execution_id)
        if decision_id is not None:
            stmt = stmt.where(Outcome.decision_id == decision_id)
        stmt = stmt.order_by(Outcome.occurred_at.asc(), Outcome.created_at.asc())
        return (await self.db.scalars(stmt)).all()

    async def recent(self, org_id: uuid.UUID, *, limit: int = 200) -> Sequence[Outcome]:
        """This organization's newest outcomes first — the read-surface order.

        NOT `for_replay` with a limit. That method is ascending because replay
        must fold outcomes in the order they occurred, so taking `limit` off it
        returns the OLDEST rows — the opposite of what a feed wants, and a
        difference that only shows up once an org has more history than the page
        size.

        Ordered by `occurred_at` (reality time) with `created_at` breaking ties,
        mirroring `timeline` so the two agree about sequence.
        """
        stmt = (
            select(Outcome)
            .where(Outcome.organization_id == org_id)
            .order_by(Outcome.occurred_at.desc(), Outcome.created_at.desc())
            .limit(limit)
        )
        return (await self.db.scalars(stmt)).all()

    async def for_replay(
        self, org_id: uuid.UUID, *, since: datetime | None = None, limit: int = 1000
    ) -> Sequence[Outcome]:
        stmt = select(Outcome).where(Outcome.organization_id == org_id)
        if since is not None:
            stmt = stmt.where(Outcome.occurred_at >= since)
        stmt = stmt.order_by(Outcome.occurred_at.asc(), Outcome.created_at.asc()).limit(limit)
        return (await self.db.scalars(stmt)).all()
