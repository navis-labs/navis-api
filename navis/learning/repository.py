"""LearningRepository — append-only events + atomic count aggregates.

Idempotent event append (ON CONFLICT on deterministic id). Aggregate increments
are atomic upserts (`successes = successes + N` in SQL — no read-modify-write race).
Rebuild recomputes aggregates from the immutable event log.
"""
from __future__ import annotations

import uuid
from typing import Any, AsyncIterator, Collection, Sequence

from sqlalchemy import and_, delete, or_, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from navis.learning.aggregates import Delta
from navis.learning.models import (
    AggregateDimension,
    LearningAggregate,
    LearningEvent,
    LearningPattern,
    LearningInbox,
)
from navis.learning.patterns import PatternFact


class LearningRepository:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def append_event(self, values: dict[str, Any]) -> tuple[LearningEvent, bool]:
        stmt = pg_insert(LearningEvent).values(**values).on_conflict_do_nothing(
            index_elements=["id"]
        )
        result = await self.db.execute(stmt)
        created = (result.rowcount or 0) == 1
        row = await self.db.get(LearningEvent, values["id"])
        assert row is not None  # just inserted, or the prior row (events are never deleted)
        return row, created

    async def increment_aggregate(
        self, org_id: uuid.UUID, dimension: str, key: str, delta: Delta
    ) -> None:
        stmt = (
            pg_insert(LearningAggregate)
            .values(
                organization_id=org_id, dimension=dimension, key=key,
                successes=delta.successes, failures=delta.failures,
                neutrals=delta.neutrals, total=delta.total,
            )
            .on_conflict_do_update(
                index_elements=["organization_id", "dimension", "key"],
                set_={
                    "successes": LearningAggregate.successes + delta.successes,
                    "failures": LearningAggregate.failures + delta.failures,
                    "neutrals": LearningAggregate.neutrals + delta.neutrals,
                    "total": LearningAggregate.total + delta.total,
                },
            )
        )
        await self.db.execute(stmt)

    async def upsert_pattern(self, fact: PatternFact) -> None:
        stmt = (
            pg_insert(LearningPattern)
            .values(
                id=fact.pattern_id, organization_id=fact.org_id, signal_id=fact.signal_id,
                kind=fact.kind, samples=fact.samples, success_rate=fact.success_rate, detail={},
            )
            .on_conflict_do_update(
                index_elements=["id"],
                set_={"kind": fact.kind, "samples": fact.samples, "success_rate": fact.success_rate},
            )
        )
        await self.db.execute(stmt)

    async def inbox_claim(self, consumer: str, event_id: str) -> bool:
        stmt = (
            pg_insert(LearningInbox)
            .values(consumer=consumer, event_id=event_id)
            .on_conflict_do_nothing(index_elements=["consumer", "event_id"])
        )
        result = await self.db.execute(stmt)
        return (result.rowcount or 0) == 1

    async def stream_events(
        self, org_id: uuid.UUID, *, batch: int = 5000
    ) -> AsyncIterator[LearningEvent]:
        """Keyset-paginated stream of the org's immutable log in (occurred_at, id) order.
        Replay folds this — O(distinct keys) memory, never O(events), and it NEVER truncates
        (the reason all_events, capped at 1M, was retired from the rebuild path). Mirrors
        state.v2 MemoryReader.stream, the established streaming-rebuild pattern."""
        last_ts = None
        last_id = None
        while True:
            stmt = select(LearningEvent).where(LearningEvent.organization_id == org_id)
            if last_ts is not None and last_id is not None:
                # Keyset cursor: (occurred_at, id) strictly after the last row seen.
                stmt = stmt.where(
                    or_(
                        LearningEvent.occurred_at > last_ts,
                        and_(LearningEvent.occurred_at == last_ts, LearningEvent.id > last_id),
                    )
                )
            stmt = stmt.order_by(
                LearningEvent.occurred_at.asc(), LearningEvent.id.asc()
            ).limit(batch)
            rows = (await self.db.scalars(stmt)).all()
            if not rows:
                return
            for row in rows:
                yield row
            if len(rows) < batch:
                return
            last_ts, last_id = rows[-1].occurred_at, rows[-1].id

    async def aggregates(
        self, org_id: uuid.UUID, *, dimension: str | None = None
    ) -> Sequence[LearningAggregate]:
        stmt = select(LearningAggregate).where(LearningAggregate.organization_id == org_id)
        if dimension is not None:
            stmt = stmt.where(LearningAggregate.dimension == dimension)
        return (await self.db.scalars(stmt)).all()

    async def signal_aggregates(self, org_id: uuid.UUID) -> Sequence[LearningAggregate]:
        return await self.aggregates(org_id, dimension=AggregateDimension.SIGNAL.value)

    async def signal_aggregates_for(
        self, org_id: uuid.UUID, keys: Collection[str]
    ) -> Sequence[LearningAggregate]:
        """Only the SIGNAL aggregates for a specific set of signal ids — the per-outcome
        pattern refresh reads just the few signals that outcome touched, never the org's
        entire signal table. Index-covered by the (org, dimension, key) PK prefix + key IN."""
        if not keys:
            return []
        stmt = select(LearningAggregate).where(
            LearningAggregate.organization_id == org_id,
            LearningAggregate.dimension == AggregateDimension.SIGNAL.value,
            LearningAggregate.key.in_(list(keys)),
        )
        return (await self.db.scalars(stmt)).all()

    async def delete_projections(self, org_id: uuid.UUID) -> None:
        await self.db.execute(
            delete(LearningAggregate).where(LearningAggregate.organization_id == org_id)
        )
        await self.db.execute(
            delete(LearningPattern).where(LearningPattern.organization_id == org_id)
        )

    async def flush(self) -> None:
        await self.db.flush()
