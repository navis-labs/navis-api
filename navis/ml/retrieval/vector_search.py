"""Org-scoped vector search over memory_embeddings (brute-force cosine).

Correct and exact at current scale: fetch an org's embeddings (bounded), score by
cosine, return top-k. The scale upgrade is a pgvector ANN index behind the SAME
signature — callers never change. Always tenant-scoped: an org can only search its
own vectors.

We always join `MemoryRecord` to bring back `event_timestamp` alongside each hit
so the LLM can answer date-scoped questions ("meetings this week"). Optional
`since` / `until` half-open bounds narrow the candidate set on the same column.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from navis.memory.models import MemoryRecord
from navis.ml.embeddings.models import MemoryEmbedding
from navis.ml.embeddings.service import EMBEDDING_VERSION
from navis.ml.math import cosine


@dataclass(frozen=True, slots=True)
class Hit:
    memory_id: uuid.UUID
    score: float
    entity_type: str
    source_text: str
    event_timestamp: datetime | None = None


async def search(
    db: AsyncSession,
    org_id: uuid.UUID,
    query_vector: list[float],
    *,
    model: str,
    version: int = EMBEDDING_VERSION,
    k: int = 8,
    candidate_limit: int = 5000,
    since: datetime | None = None,
    until: datetime | None = None,
) -> list[Hit]:
    stmt = (
        select(MemoryEmbedding, MemoryRecord.event_timestamp)
        .join(MemoryRecord, MemoryRecord.memory_id == MemoryEmbedding.memory_id)
        .where(
            MemoryEmbedding.organization_id == org_id,
            MemoryEmbedding.model == model,
            MemoryEmbedding.version == version,
        )
    )
    if since is not None:
        stmt = stmt.where(MemoryRecord.event_timestamp >= since)
    if until is not None:
        stmt = stmt.where(MemoryRecord.event_timestamp < until)

    rows = (await db.execute(stmt.limit(candidate_limit))).all()
    hits = [
        Hit(
            r.MemoryEmbedding.memory_id,
            cosine(query_vector, r.MemoryEmbedding.embedding),
            r.MemoryEmbedding.entity_type,
            r.MemoryEmbedding.source_text,
            r.event_timestamp,
        )
        for r in rows
    ]
    hits.sort(key=lambda h: (-h.score, str(h.memory_id)))  # deterministic tiebreak
    return hits[:k]
