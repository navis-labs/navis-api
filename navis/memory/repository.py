"""Append-only MemoryRepository — the ONLY writer of memory_records.

There is exactly one write operation: `append`. No update, no upsert, no delete.
Idempotency is enforced at the database with INSERT ... ON CONFLICT (memory_id)
DO NOTHING, so re-ingestion, at-least-once delivery, and full replay converge to
the same rows without duplicates or races.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Sequence

from sqlalchemy import delete, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from navis.memory.envelope import EntityType, MemoryRecord
from navis.memory.models import MemoryOutbox, MemoryRecord as MemoryRecordRow


class MemoryRepository:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def append(self, record: MemoryRecord) -> bool:
        """Append one immutable fact. Returns True if newly written, False if the
        deterministic memory_id already existed (idempotent)."""
        stmt = (
            pg_insert(MemoryRecordRow)
            .values(
                memory_id=record.memory_id,
                organization_id=record.org_id,
                entity_type=record.entity_type.value,
                entity_id=record.entity_id,
                source_signal_event_id=record.source_signal_event_id,
                source_integration_id=record.source_integration_id,
                extractor_name=record.extractor_name,
                extractor_version=record.extractor_version,
                schema_version=record.schema_version,
                event_timestamp=record.event_timestamp,
                confidence=record.confidence,
                dedup_key=record.dedup_key,
                payload=record.payload,
            )
            .on_conflict_do_nothing(index_elements=["memory_id"])
        )
        result = await self.db.execute(stmt)
        return (result.rowcount or 0) == 1

    async def entity_history(
        self, org_id: uuid.UUID, entity_type: EntityType, entity_id: uuid.UUID
    ) -> Sequence[MemoryRecordRow]:
        """All immutable facts about one entity, oldest-first."""
        stmt = (
            select(MemoryRecordRow)
            .where(
                MemoryRecordRow.organization_id == org_id,
                MemoryRecordRow.entity_type == entity_type.value,
                MemoryRecordRow.entity_id == entity_id,
            )
            .order_by(MemoryRecordRow.event_timestamp.asc())
        )
        return (await self.db.scalars(stmt)).all()

    async def by_signal(
        self, signal_event_id: uuid.UUID
    ) -> Sequence[MemoryRecordRow]:
        stmt = select(MemoryRecordRow).where(
            MemoryRecordRow.source_signal_event_id == signal_event_id
        )
        return (await self.db.scalars(stmt)).all()

    async def list_by_entity_type(
        self,
        org_id: uuid.UUID,
        entity_type: EntityType,
        *,
        limit: int = 500,
    ) -> Sequence[MemoryRecordRow]:
        """Newest-first facts of one entity_type in an org. Backs read APIs that
        surface people / decisions / commitments etc. Uses ix_mem_org_type_time."""
        stmt = (
            select(MemoryRecordRow)
            .where(
                MemoryRecordRow.organization_id == org_id,
                MemoryRecordRow.entity_type == entity_type.value,
            )
            .order_by(MemoryRecordRow.event_timestamp.desc())
            .limit(limit)
        )
        return (await self.db.scalars(stmt)).all()

    # --- governance (GDPR erasure cascade) -------------------------------------------------
    async def scan_org(
        self, org_id: uuid.UUID, *, limit: int = 500, after_id: uuid.UUID | None = None
    ) -> Sequence[MemoryRecordRow]:
        """Keyset-paged scan of an org's memory for the erasure cascade."""
        stmt = (
            select(MemoryRecordRow)
            .where(MemoryRecordRow.organization_id == org_id)
            .order_by(MemoryRecordRow.memory_id)
            .limit(limit)
        )
        if after_id is not None:
            stmt = stmt.where(MemoryRecordRow.memory_id > after_id)
        return (await self.db.scalars(stmt)).all()

    async def scrub_payload(self, memory_id: uuid.UUID, payload: dict[str, Any]) -> None:
        """Governance-only in-place payload rewrite (erasure tombstone). The skeleton
        (memory_id/entity_id/provenance) is preserved so replay + audit stay coherent; the
        sanctioned exception to append-only, exactly like the signal-layer scrub."""
        await self.db.execute(
            update(MemoryRecordRow)
            .where(MemoryRecordRow.memory_id == memory_id)
            .values(payload=payload)
        )

    # ---- transactional outbox (memory-derived domain events) ----
    async def append_outbox(self, rows: Sequence[dict[str, Any]]) -> None:
        """Append pointer rows IN THE CALLER'S TRANSACTION (no commit) — atomic with the
        memory append. Flush so they share the same durable commit."""
        if not rows:
            return
        self.db.add_all([MemoryOutbox(**r) for r in rows])
        await self.db.flush()

    async def pending_outbox(self, *, limit: int = 500) -> Sequence[MemoryOutbox]:
        """Claim unpublished pointers FOR UPDATE SKIP LOCKED so concurrent relay workers
        never publish the same row (oldest-first)."""
        stmt = (
            select(MemoryOutbox)
            .where(MemoryOutbox.published_at.is_(None))
            .order_by(MemoryOutbox.created_at.asc())
            .limit(limit)
            .with_for_update(skip_locked=True)
        )
        return (await self.db.scalars(stmt)).all()

    async def mark_outbox_published(self, ids: Sequence[uuid.UUID]) -> None:
        if not ids:
            return
        await self.db.execute(
            update(MemoryOutbox)
            .where(MemoryOutbox.id.in_(list(ids)))
            .values(published_at=datetime.now(timezone.utc))
        )

    async def purge_published_outbox(self, *, before: datetime) -> int:
        """Delete PUBLISHED outbox rows older than `before` (bounds outbox growth). Unpublished
        rows are never purged. Returns the count removed."""
        result = await self.db.execute(
            delete(MemoryOutbox).where(
                MemoryOutbox.published_at.is_not(None),
                MemoryOutbox.published_at < before,
            )
        )
        return result.rowcount or 0
