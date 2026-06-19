"""MemoryRetriever — the read API for semantic recall over Memory.

Answers questions like "What happened with YC?" by embedding the query, searching
the org's vectors, and reranking — returning Memory facts WITH lineage (memory_id),
never generated text. No LLM hallucination: every hit is a real, immutable fact.
"""
from __future__ import annotations

import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from navis.ml.embeddings.service import EMBEDDING_VERSION
from navis.ml.providers import get_embedding_provider
from navis.ml.retrieval.reranker import rerank
from navis.ml.retrieval.vector_search import Hit, search


class MemoryRetriever:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db
        self.provider = get_embedding_provider()
        self.model = self.provider.info.embed_model

    async def search(
        self, org_id: uuid.UUID, query: str, *, k: int = 8, version: int = EMBEDDING_VERSION
    ) -> list[Hit]:
        qvec = (await self.provider.embed([query])).vectors[0]
        hits = await search(self.db, org_id, qvec, model=self.model, version=version, k=k * 3)
        return rerank(query, hits)[:k]
