"""Embedding jobs — backfill + version-bump rebuilds.

Two Celery tasks live here:

  * `embed_org(org_id)` — the per-org worker. Reads memory_records for the org
    and writes any missing rows into memory_embeddings (idempotent via
    ON CONFLICT). Cheap on re-run: already-embedded records are skipped.

  * `embed_all_orgs()` — fan-out. Enqueues one `embed_org` task per org (keyset-
    paginated so a million orgs don't land in memory at once). Called manually
    for a one-time backfill of pre-existing orgs, and re-callable safely as a
    sweeper — it will only ever re-check missing embeddings, never re-embed
    records that already have a row at the current model+version.

Real-time coverage is provided by `navis.ml.embeddings.subscribers`, which
embeds each new memory record as its pointer event fires. `embed_all_orgs`
exists to (a) backfill records that pre-date the subscriber wiring, and (b)
catch entity types the pointer bus doesn't route today (COMPANY / MEETING /
DOCUMENT / etc. — see _BUS_EVENT_FOR_ENTITY in navis.memory.service).

Append-only: a re-embed with a bumped version writes NEW rows; old vectors
remain for audit/rollback.
"""
from __future__ import annotations

import uuid

from sqlalchemy import select

from navis.ml.embeddings.service import EMBEDDING_VERSION, EmbeddingService
from navis.platform.database import session_scope
from navis.platform.monitoring import get_logger
from navis.platform.queue import celery_app, run_async

logger = get_logger("ml.embeddings.jobs")


async def _embed_org(org_id: uuid.UUID, version: int, max_pages: int) -> int:
    """Advance this org's backfill by up to `max_pages` keyset pages.

    Bounded per invocation so one task can never outrun Celery's time limit on a
    large org; the persisted cursor means the next invocation resumes exactly
    where this one stopped rather than restarting.
    """
    async with session_scope() as db:
        count = await EmbeddingService(db).embed_org(
            org_id, version=version, max_pages=max_pages
        )
    logger.info("embedded org=%s version=%d new=%d", org_id, version, count)
    return count


async def _backfill_status(org_id: uuid.UUID, version: int) -> dict:
    async with session_scope() as db:
        return await EmbeddingService(db).backfill_status(org_id, version=version)


async def _dispatch_all(version: int) -> int:
    """Fan-out: enqueue one embed_org task per organization (keyset-paginated).

    Mirrors navis.orchestration.brief._dispatch_all so a single task never
    sequentially processes every org, and the id list never fully materializes.
    """
    from navis.identity.models import Organization

    dispatched, after, page = 0, None, 1000
    while True:
        async with session_scope() as db:
            stmt = select(Organization.id).order_by(Organization.id).limit(page)
            if after is not None:
                stmt = stmt.where(Organization.id > after)
            ids = (await db.scalars(stmt)).all()
        if not ids:
            break
        for oid in ids:
            embed_org.delay(str(oid), version)
        dispatched += len(ids)
        after = ids[-1]
        if len(ids) < page:
            break
    return dispatched


# Pages per task invocation. A page is settings.embedding_backfill_page_size
# records, so this bounds one task's work well inside the 300s soft time limit
# while the durable cursor carries the rest to the next run.
MAX_PAGES_PER_TASK = 20


@celery_app.task(name="navis.ml.embeddings.jobs.embed_org")
def embed_org(
    org_id: str, version: int = EMBEDDING_VERSION, max_pages: int = MAX_PAGES_PER_TASK
) -> dict:
    return {"embedded": run_async(_embed_org(uuid.UUID(org_id), version, max_pages))}


@celery_app.task(name="navis.ml.embeddings.jobs.backfill_status")
def backfill_status(org_id: str, version: int = EMBEDDING_VERSION) -> dict:
    """Read-only progress for one org — how far the migration has advanced."""
    return run_async(_backfill_status(uuid.UUID(org_id), version))


@celery_app.task(name="navis.ml.embeddings.jobs.embed_all_orgs")
def embed_all_orgs(version: int = EMBEDDING_VERSION) -> dict:
    return {"dispatched": run_async(_dispatch_all(version))}
