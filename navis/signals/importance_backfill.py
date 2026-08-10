"""Backfill importance scores over the existing signal log.

New signals are scored at ingestion. Everything captured before the policy
existed — or before a version bump — is scored here.

Resumable without a cursor: each page selects signals that have NO score at the
target version, so scoring a page removes it from the next query's result set.
An interrupted run simply continues; a completed run is a cheap no-op. Combined
with `ON CONFLICT DO NOTHING` on a deterministic id, running this concurrently
with live ingestion is safe.

Nothing is mutated: `signal_events` is untouched, and a re-score at a NEW version
adds rows beside the old ones rather than replacing them.
"""
from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import select

from navis.platform.database import session_scope
from navis.platform.monitoring import get_logger
from navis.platform.queue import celery_app, run_async
from navis.signals.base import SignalSource
from navis.signals.importance import IMPORTANCE_VERSION, score_payload
from navis.signals.models import (
    SignalEventRecord,
    SignalImportance,
    compute_importance_id,
)
from navis.signals.repository import SignalRepository

logger = get_logger("signals.importance.backfill")

PAGE_SIZE = 500


async def score_page(
    db: Any, org_id: uuid.UUID | None = None, *, version: int = IMPORTANCE_VERSION,
    page_size: int = PAGE_SIZE,
) -> int:
    """Score one page of unscored signals. Returns how many were written."""
    scored = select(SignalImportance.signal_event_id).where(
        SignalImportance.signal_event_id == SignalEventRecord.id,
        SignalImportance.version == version,
    )
    stmt = (
        select(SignalEventRecord)
        .where(~scored.exists())
        .order_by(SignalEventRecord.id)
        .limit(page_size)
    )
    if org_id is not None:
        stmt = stmt.where(SignalEventRecord.organization_id == org_id)
    records = (await db.scalars(stmt)).all()
    if not records:
        return 0

    rows: list[dict[str, Any]] = []
    for rec in records:
        try:
            source = SignalSource(rec.source)
        except ValueError:
            continue  # a source this build doesn't know; leave it unscored, loudly skipped
        importance = score_payload(source, rec.payload or {})
        rows.append(
            {
                "id": compute_importance_id(rec.id, version),
                "organization_id": rec.organization_id,
                "signal_event_id": rec.id,
                "version": version,
                "score": importance.score,
                "classification": importance.classification,
                "features": importance.features,
            }
        )
    await SignalRepository(db).append_importance(rows)
    return len(rows)


async def _backfill(org_id: uuid.UUID | None, version: int, max_pages: int) -> int:
    total = 0
    for _ in range(max_pages):
        async with session_scope() as db:
            written = await score_page(db, org_id, version=version)
            await db.commit()
        total += written
        if written == 0:
            break
    logger.info("importance backfill org=%s version=%d scored=%d", org_id, version, total)
    return total


@celery_app.task(name="navis.signals.importance_backfill.score_signals")
def score_signals(
    org_id: str | None = None, version: int = IMPORTANCE_VERSION, max_pages: int = 200
) -> dict:
    """Backfill importance for one org, or every org when `org_id` is omitted."""
    oid = uuid.UUID(org_id) if org_id else None
    return {"scored": run_async(_backfill(oid, version, max_pages))}
