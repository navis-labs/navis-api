"""Re-embedding jobs — bump the version to rebuild the index with a new model.

Append-only: a re-embed writes new rows at version N+1; old vectors remain for
audit/rollback. Wired as a Celery task; fan-out per org mirrors the brief scheduler.
"""
from __future__ import annotations

import asyncio
import uuid

from navis.ml.embeddings.service import EMBEDDING_VERSION, EmbeddingService
from navis.platform.database import session_scope
from navis.platform.monitoring import get_logger
from navis.platform.queue import celery_app

logger = get_logger("ml.embeddings.jobs")


async def _embed_org(org_id: uuid.UUID, version: int) -> int:
    async with session_scope() as db:
        count = await EmbeddingService(db).embed_org(org_id, version=version)
    logger.info("embedded org=%s version=%d new=%d", org_id, version, count)
    return count


@celery_app.task(name="navis.ml.embeddings.jobs.embed_org")
def embed_org(org_id: str, version: int = EMBEDDING_VERSION) -> dict:
    return {"embedded": asyncio.run(_embed_org(uuid.UUID(org_id), version))}
