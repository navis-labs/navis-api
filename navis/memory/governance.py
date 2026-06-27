"""Memory-layer GDPR erasure cascade.

When the Signal layer erases a subject (signals.governance.erase_subject), the EXTRACTED
memory facts about that subject must die too — otherwise PII survives in memory_records,
backups, and replicas. This module registers a subscriber on the signal-layer erasure seam
that, in the SAME transaction, tombstones every memory record referencing the subject.

Replay-safe: the source signal is already tombstoned, so re-extraction on replay yields
empty/erased content anyway; this cascade additionally scrubs records already materialized.
Idempotent: an already-tombstoned record is skipped.
"""
from __future__ import annotations

from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from navis.memory.repository import MemoryRepository
from navis.platform.monitoring import get_logger
from navis.platform.monitoring.metrics import metrics
from navis.signals.governance import SubjectErasureNotice, on_subject_erasure

logger = get_logger("memory.governance")

_TOMBSTONE = "__erased__"
# Memory payload fields that may carry the subject's identity (exact match only — never substring).
_EMAIL_FIELDS = ("email", "person_email", "speaker_email", "sender_email", "organizer")
_SLACK_FIELDS = ("slack_user_id", "speaker_slack_user_id")


def _references_subject(payload: dict[str, Any], *, email: str | None, slack_user_id: str | None) -> bool:
    if email:
        el = email.strip().lower()
        if any((payload.get(f) or "").strip().lower() == el for f in _EMAIL_FIELDS if isinstance(payload.get(f), str)):
            return True
        if any(el == str(e).strip().lower() for e in (payload.get("attendees") or [])):
            return True
        if any(el == str(o).strip().lower() for o in (payload.get("owners") or [])):
            return True
    if slack_user_id and any(payload.get(f) == slack_user_id for f in _SLACK_FIELDS):
        return True
    return False


@on_subject_erasure
async def cascade_erase_memory(notice: SubjectErasureNotice, db: AsyncSession) -> int:
    """Tombstone memory facts about the erased subject. Runs inside the erasure transaction."""
    repo = MemoryRepository(db)
    scanned = erased = 0
    after_id = None
    while True:
        rows = await repo.scan_org(notice.org_id, limit=500, after_id=after_id)
        if not rows:
            break
        for rec in rows:
            after_id = rec.memory_id
            scanned += 1
            payload = rec.payload or {}
            if payload.get("_erased"):
                continue  # idempotent
            if not _references_subject(payload, email=notice.email, slack_user_id=notice.slack_user_id):
                continue
            erased += 1
            # Tombstone the content; keep the row skeleton (entity_id/provenance) for replay+audit.
            await repo.scrub_payload(rec.memory_id, {"_erased": True})
        if len(rows) < 500:
            break
    metrics.incr("memory.erasure.erased", value=erased)
    logger.warning("memory erasure cascade org=%s scanned=%d erased=%d", notice.org_id, scanned, erased)
    return erased
