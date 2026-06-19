"""EmbeddingService — turn immutable Memory facts into versioned vectors.

Reads Memory (allowed: ML may read the truth layers for retrieval) and writes the
append-only memory_embeddings table. Idempotent: the same fact at the same model+
version collapses via ON CONFLICT DO NOTHING.
"""
from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from navis.memory.models import MemoryRecord
from navis.ml.embeddings.models import MemoryEmbedding, compute_embedding_id
from navis.ml.providers import get_embedding_provider

EMBEDDING_VERSION = 1


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

    async def embed_record(self, record: MemoryRecord, *, version: int = EMBEDDING_VERSION) -> bool:
        text = memory_text(record)
        result = await self.provider.embed([text])
        values = {
            "id": compute_embedding_id(record.memory_id, self.model, version),
            "organization_id": record.organization_id,
            "memory_id": record.memory_id,
            "model": self.model,
            "version": version,
            "dim": result.dim,
            "embedding": result.vectors[0],
            "entity_type": record.entity_type,
            "source_text": text,
        }
        stmt = pg_insert(MemoryEmbedding).values(**values).on_conflict_do_nothing(
            index_elements=["id"]
        )
        return (await self.db.execute(stmt)).rowcount == 1

    async def embed_org(self, org_id: uuid.UUID, *, version: int = EMBEDDING_VERSION, limit: int = 1000) -> int:
        """Embed every not-yet-embedded fact for an org at the current version."""
        embedded_ids = set(
            (
                await self.db.scalars(
                    select(MemoryEmbedding.memory_id).where(
                        MemoryEmbedding.organization_id == org_id,
                        MemoryEmbedding.model == self.model,
                        MemoryEmbedding.version == version,
                    )
                )
            ).all()
        )
        records = (
            await self.db.scalars(
                select(MemoryRecord).where(MemoryRecord.organization_id == org_id).limit(limit)
            )
        ).all()
        count = 0
        for record in records:
            if record.memory_id in embedded_ids:
                continue
            if await self.embed_record(record, version=version):
                count += 1
        await self.db.flush()
        return count
