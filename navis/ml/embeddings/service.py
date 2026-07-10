"""EmbeddingService — turn immutable Memory facts into versioned vectors.

Reads Memory (allowed: ML may read the truth layers for retrieval) and writes the append-only
memory_embeddings table. Idempotent: the same fact at the same (model, version) collapses via a
deterministic id + ON CONFLICT DO NOTHING, so re-runs and retries never duplicate or corrupt.

Backfill (embed_org) selects ONLY not-yet-embedded records via a NOT EXISTS anti-join, ordered
by memory_id for stable FORWARD PROGRESS — repeated calls drain the backlog and converge (they
never re-scan embedded rows), and provider calls are BATCHED (one round-trip per batch, not per
record) so it scales to millions of facts.
"""
from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import exists, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from navis.memory.models import MemoryRecord
from navis.ml.embeddings.models import MemoryEmbedding, compute_embedding_id
from navis.ml.providers import get_embedding_provider

EMBEDDING_VERSION = 1
# One provider round-trip per batch of texts (embed() is batch-capable). Conservative so a batch
# stays well within provider input/token limits; tune with evidence, never by guessing.
_EMBED_BATCH = 128


def memory_text(record: MemoryRecord) -> str:
    """A stable text representation of a fact for embedding (no model judgement)."""
    payload: dict[str, Any] = record.payload or {}
    parts = [record.entity_type]
    for key in ("title", "summary", "text", "name", "description", "content"):
        val = payload.get(key)
        if isinstance(val, str) and val:
            parts.append(val)
    if len(parts) == 1:  # fall back to the dedup key (always present, human-meaningful)
        parts.append(record.dedup_key)
    return " — ".join(parts)[:2000]


class EmbeddingService:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db
        self.provider = get_embedding_provider()
        self.model = self.provider.info.embed_model

    def _row(self, record: MemoryRecord, vector: list[float], text: str, version: int) -> dict[str, Any]:
        """The single home for a memory_embeddings row shape (used by the single + batch paths)."""
        return {
            "id": compute_embedding_id(record.memory_id, self.model, version),
            "organization_id": record.organization_id,
            "memory_id": record.memory_id,
            "model": self.model,
            "version": version,
            "dim": len(vector),
            "embedding": vector,
            "entity_type": record.entity_type,
            "source_text": text,
        }

    async def embed_record(self, record: MemoryRecord, *, version: int = EMBEDDING_VERSION) -> bool:
        """Embed ONE fact (real-time single-record path). Idempotent; returns True iff a new row
        was written (False if it already existed at this model+version)."""
        text = memory_text(record)
        result = await self.provider.embed([text])
        stmt = pg_insert(MemoryEmbedding).values(
            self._row(record, result.vectors[0], text, version)
        ).on_conflict_do_nothing(index_elements=["id"])
        return (await self.db.execute(stmt)).rowcount == 1

    async def embed_org(
        self, org_id: uuid.UUID, *, version: int = EMBEDDING_VERSION, limit: int = 1000
    ) -> int:
        """Embed up to `limit` NOT-YET-EMBEDDED facts for an org, batched. Selects only records
        with no (model, version) embedding via a NOT EXISTS anti-join, ordered by memory_id, so
        each call makes forward progress and repeated calls converge (drain the backlog) instead
        of re-scanning embedded rows. Returns the number of NEW embeddings written."""
        missing = ~exists().where(
            MemoryEmbedding.memory_id == MemoryRecord.memory_id,
            MemoryEmbedding.model == self.model,
            MemoryEmbedding.version == version,
        )
        records = (
            await self.db.scalars(
                select(MemoryRecord)
                .where(MemoryRecord.organization_id == org_id, missing)
                .order_by(MemoryRecord.memory_id)
                .limit(limit)
            )
        ).all()

        written = 0
        for start in range(0, len(records), _EMBED_BATCH):
            written += await self._embed_batch(list(records[start:start + _EMBED_BATCH]), version)
        await self.db.flush()
        return written

    async def _embed_batch(self, batch: list[MemoryRecord], version: int) -> int:
        if not batch:
            return 0
        texts = [memory_text(r) for r in batch]
        result = await self.provider.embed(texts)  # ONE round-trip for the whole batch
        # Fail-closed: a provider that returns a different vector count than inputs would MISALIGN
        # vectors to records (silent corruption). Refuse to write rather than store wrong vectors.
        if len(result.vectors) != len(batch):
            raise ValueError(
                f"embedding provider returned {len(result.vectors)} vectors for {len(batch)} inputs"
            )
        rows = [self._row(r, vec, text, version) for r, vec, text in zip(batch, result.vectors, texts)]
        stmt = pg_insert(MemoryEmbedding).values(rows).on_conflict_do_nothing(index_elements=["id"])
        return (await self.db.execute(stmt)).rowcount or 0
