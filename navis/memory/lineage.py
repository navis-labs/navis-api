"""Read-side lineage: which signal produced a fact, and how important was it?

One email becomes six memory facts — a thread, two people, two relationships, a
company. All six are true, all six are separately retrievable, and when the email
matters they all score alike, so an answer can end up saying the same thing five
times:

    imp=0.9 'Email thread "Accepted: Meeting with mulayam…"'
    imp=0.9 'Person Mulayam <mulayamabmrc@gmail.com>, sender…'
    imp=0.9 'Received email from mulayamabmrc@gmail.com…'

Collapsing them needs to know which event each fact came from, which is exactly
what `memory_records.source_signal_event_id` records. Retrieval asks Memory for
it rather than reading the Signals tables, because the ML layer may not import
Signals (see `tests/ml/test_ml_layer.py`) — and because Memory owns lineage.

Importance rides along in the SAME query on purpose: the lookup is round-trip
bound (~635ms against a remote database, whether 24 ids or 160), so fetching the
two facets separately would double the cost of every search for no benefit.

Read-only. Never fabricates: a fact whose signal has not been scored comes back
with `importance=None`, and callers treat that as neutral rather than as zero.
"""
from __future__ import annotations

import uuid
from collections.abc import Iterable
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from navis.memory.models import MemoryRecord
from navis.signals.importance import IMPORTANCE_VERSION
from navis.signals.models import SignalImportance


@dataclass(frozen=True, slots=True)
class FactLineage:
    """Where a memory fact came from, and what that source was worth."""

    source_signal_event_id: uuid.UUID
    importance: float | None = None


async def lineage_by_memory_id(
    db: AsyncSession,
    memory_ids: Iterable[uuid.UUID],
    *,
    version: int = IMPORTANCE_VERSION,
) -> dict[uuid.UUID, FactLineage]:
    """Map memory_id -> (source signal, importance) in one round trip.

    Uses an OUTER join on importance so a fact is still returned — and therefore
    still de-duplicable — when its signal has not been scored yet.
    """
    ids = list(memory_ids)
    if not ids:
        return {}
    rows = (
        await db.execute(
            select(
                MemoryRecord.memory_id,
                MemoryRecord.source_signal_event_id,
                SignalImportance.score,
            )
            .outerjoin(
                SignalImportance,
                (SignalImportance.signal_event_id == MemoryRecord.source_signal_event_id)
                & (SignalImportance.version == version),
            )
            .where(MemoryRecord.memory_id.in_(ids))
        )
    ).all()
    return {
        memory_id: FactLineage(
            source_signal_event_id=signal_id,
            importance=None if score is None else float(score),
        )
        for memory_id, signal_id, score in rows
    }
