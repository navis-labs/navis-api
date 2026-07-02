"""Org-scoped vector search over memory_embeddings (brute-force cosine).

Fetch an org's embeddings (bounded, ORDERED by memory_id so the candidate set is DETERMINISTIC
under the cap), score by cosine, return top-k with a stable (score desc, memory_id) tiebreak —
deterministic and replay-safe. The scale upgrade is a pgvector ANN index behind the SAME
signature — callers never change. Always tenant-scoped: an org can only search its own vectors.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from navis.ml.embeddings.models import MemoryEmbedding
from navis.ml.embeddings.service import EMBEDDING_VERSION
from navis.ml.math import cosine


@dataclass(frozen=True, slots=True)
class Hit:
    memory_id: uuid.UUID
    score: float
    entity_type: str
    source_text: str


async def search(
    db: AsyncSession,
    org_id: uuid.UUID,
    query_vector: list[float],
    *,
    model: str,
    version: int = EMBEDDING_VERSION,
    k: int = 8,
    candidate_limit: int = 5000,
) -> list[Hit]:
    rows = (
        await db.scalars(
            select(MemoryEmbedding)
            .where(
                MemoryEmbedding.organization_id == org_id,
                MemoryEmbedding.model == model,
                MemoryEmbedding.version == version,
            )
            # Deterministic candidate selection: ORDER BY the PK so that when an org has more
            # than `candidate_limit` embeddings, the SAME candidate set is fetched every run
            # (replay-safe). Without this, `.limit()` returns an unordered/unstable subset →
            # non-deterministic top-k. (pgvector/ANN is the scale upgrade for full recall.)
            .order_by(MemoryEmbedding.memory_id)
            .limit(candidate_limit)
        )
    ).all()
    hits = [
        Hit(r.memory_id, cosine(query_vector, r.embedding), r.entity_type, r.source_text)
        for r in rows
    ]
    hits.sort(key=lambda h: (-h.score, str(h.memory_id)))  # deterministic tiebreak
    return hits[:k]
