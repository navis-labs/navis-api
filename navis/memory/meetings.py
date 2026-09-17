"""Which meetings are cancelled, for read paths that must not cite them.

Memory is append-only, deliberately: a meeting's history is every version of it,
and destroying that history to reflect the present would throw away the audit
trail the whole product is built on. But an append-only store has a consequence
retrieval has to handle explicitly — the `status: confirmed` record written when
a meeting was booked stays exactly as retrievable as the `status: cancelled`
record written when it was called off. Vector search ranks on similarity, not
recency, so the confirmed version (richer text: title, attendees, organiser)
routinely outranks the cancellation stub (almost no text at all).

That is how "cancel the meeting" could succeed on Google and the assistant could
still answer, truthfully citing its own memory, that the meeting exists.

This module does NOT delete or rewrite anything. It answers one question —
"which meeting entities are currently cancelled?" — so a read path can drop
facts about them. The history stays intact for the audit trail and for anything
that legitimately wants to ask what USED to be true.
"""
from __future__ import annotations

import uuid
from collections.abc import Iterable

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from navis.memory.envelope import EntityType
from navis.memory.models import MemoryRecord


async def cancelled_meeting_entity_ids(
    db: AsyncSession, org_id: uuid.UUID, entity_ids: Iterable[uuid.UUID]
) -> set[uuid.UUID]:
    """Of `entity_ids`, those whose LATEST meeting record says cancelled.

    "Latest" is the newest (event_timestamp, memory_id) record for the entity —
    the same fold order State V2 uses — so a meeting that was cancelled and
    later re-created reads as live again, which is correct.

    Scoped to the ids actually retrieved rather than scanning every meeting the
    org has ever had: this runs inside a chat turn, and the candidate set is
    bounded by `k`.
    """
    ids = [e for e in entity_ids if e is not None]
    if not ids:
        return set()

    rows = (
        await db.execute(
            select(
                MemoryRecord.entity_id,
                MemoryRecord.payload,
                MemoryRecord.event_timestamp,
                MemoryRecord.memory_id,
            )
            .where(
                MemoryRecord.organization_id == org_id,
                MemoryRecord.entity_type == EntityType.MEETING.value,
                MemoryRecord.entity_id.in_(ids),
            )
            .order_by(
                MemoryRecord.entity_id,
                MemoryRecord.event_timestamp.desc(),
                MemoryRecord.memory_id.desc(),
            )
        )
    ).all()

    latest_status: dict[uuid.UUID, str | None] = {}
    for entity_id, payload, _ts, _mid in rows:
        if entity_id in latest_status:
            continue  # ordering above guarantees the first row per entity is newest
        status = (payload or {}).get("status")
        latest_status[entity_id] = str(status).lower() if status else None

    return {e for e, status in latest_status.items() if status == "cancelled"}


async def cancelled_meeting_memory_ids(
    db: AsyncSession, org_id: uuid.UUID, memory_ids: Iterable[uuid.UUID]
) -> set[uuid.UUID]:
    """Of `memory_ids`, those that are MEETING facts about a cancelled meeting.

    Retrieval holds memory ids, not entity ids, so this does the join for it:
    resolve ids -> entities, find the cancelled entities, map back. Two bounded
    queries, both keyed on the candidate set.
    """
    ids = [m for m in memory_ids if m is not None]
    if not ids:
        return set()

    rows = (
        await db.execute(
            select(MemoryRecord.memory_id, MemoryRecord.entity_id).where(
                MemoryRecord.organization_id == org_id,
                MemoryRecord.entity_type == EntityType.MEETING.value,
                MemoryRecord.memory_id.in_(ids),
            )
        )
    ).all()
    if not rows:
        return set()

    entity_by_memory = {memory_id: entity_id for memory_id, entity_id in rows}
    cancelled = await cancelled_meeting_entity_ids(
        db, org_id, set(entity_by_memory.values())
    )
    return {m for m, e in entity_by_memory.items() if e in cancelled}
