"""Read-side lookup: how important was the signal a memory fact came from?

Memory owns the lineage (`memory_records.source_signal_event_id`); Signals owns
the score (`signal_importance`). Joining them is a Memory read concern, and it
has to live here for a structural reason: the ML layer is forbidden from
importing Signals (see `tests/ml/test_ml_layer.py`), but it may read Memory. So
retrieval asks Memory "how important are these facts?" rather than reaching into
the Signals tables itself.

Read-only. Writes nothing, and never fabricates a score: a fact whose signal has
not been scored is simply absent from the result, and callers treat absence as
neutral rather than as zero — an unscored fact must not be pushed down the
ranking just because a backfill has not reached it yet.
"""
from __future__ import annotations

import uuid
from collections.abc import Iterable

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from navis.memory.models import MemoryRecord
from navis.signals.importance import IMPORTANCE_VERSION
from navis.signals.models import SignalImportance


async def importance_by_memory_id(
    db: AsyncSession,
    memory_ids: Iterable[uuid.UUID],
    *,
    version: int = IMPORTANCE_VERSION,
) -> dict[uuid.UUID, float]:
    """Map memory_id -> importance score, for the memory_ids that have one.

    Scoped to ONE policy version so a re-score at a new version can be computed
    and compared without changing what retrieval sees.
    """
    ids = list(memory_ids)
    if not ids:
        return {}
    rows = (
        await db.execute(
            select(MemoryRecord.memory_id, SignalImportance.score)
            .join(
                SignalImportance,
                SignalImportance.signal_event_id == MemoryRecord.source_signal_event_id,
            )
            .where(
                MemoryRecord.memory_id.in_(ids),
                SignalImportance.version == version,
            )
        )
    ).all()
    return {memory_id: float(score) for memory_id, score in rows}
