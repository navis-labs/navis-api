"""Admin endpoints for the current org — destructive maintenance operations
that don't fit any single domain router.

Everything here is org-scoped: a founder can prune / rebuild their own org's
data, never someone else's. No global / cross-tenant primitives live here.
Gate on Permission.MANAGE_INTEGRATIONS — founders and admins have it; view-only
roles do not.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import delete as sa_delete
from sqlalchemy.ext.asyncio import AsyncSession

from navis.decisions.models import DailyBrief
from navis.identity.authz import Principal, require_permission
from navis.identity.dependencies import get_current_context, CurrentContext
from navis.identity.permissions import Permission
from navis.memory.models import MemoryRecord
from navis.platform.database import get_session
from navis.platform.monitoring import get_logger
from navis.platform.monitoring.metrics import metrics
from navis.signals.models import SignalDeadLetter, SignalEventRecord, SignalOutbox
from navis.state.v2 import StateService
from navis.state.v2.repository import SnapshotRepository

logger = get_logger("admin")

router = APIRouter(prefix="/admin", tags=["admin"])


# 7 days is the smallest window that still buys meaningful cleanup — anything
# shorter is more likely a fat-finger than intent. Callers who really want a
# harder purge can lower this constant deliberately in a follow-up.
_MIN_PRUNE_DAYS = 7


class PruneSignalsBody(BaseModel):
    older_than_days: int = Field(
        ...,
        ge=_MIN_PRUNE_DAYS,
        le=3650,
        description=(
            "Delete signal_events + derived memory older than this many days. "
            f"Must be >= {_MIN_PRUNE_DAYS} to guard against accidental full wipes."
        ),
    )
    confirm: bool = Field(
        False,
        description=(
            "Must be true — a second, explicit confirmation that the caller "
            "understands the operation is destructive and cannot be undone from "
            "this endpoint alone (recovery would need a DB backup restore)."
        ),
    )


class PruneSignalsResponse(BaseModel):
    org_id: str
    cutoff: str
    deleted: dict[str, int]
    state_rebuilt: bool
    new_state_counts: dict[str, int] | None = None


@router.post("/signals/prune", response_model=PruneSignalsResponse)
async def prune_old_signals(
    body: PruneSignalsBody,
    ctx: CurrentContext = Depends(get_current_context),
    _p: Principal = Depends(require_permission(Permission.MANAGE_INTEGRATIONS)),
    db: AsyncSession = Depends(get_session),
) -> PruneSignalsResponse:
    """Drop signal_events + derived memory + cached state older than N days
    for the CURRENT ORG, then rebuild state cleanly.

    Delete order matters — memory_records has an ondelete=RESTRICT FK to
    signal_events, so memory must go first. embeddings.memory_id CASCADEs
    on memory_records deletion; identity_link_events.source_signal_event_id
    SET NULLs on signal_events deletion — no manual cleanup needed for either.

    The whole thing runs in one transaction: a mid-fail leaves nothing
    half-purged. The state rebuild happens AFTER the transaction commits so
    the fold reads the post-purge memory_records.
    """
    if not body.confirm:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "confirm=true required — this operation permanently deletes data",
        )

    org_id = ctx.org_id
    cutoff = datetime.now(timezone.utc) - timedelta(days=body.older_than_days)
    deleted: dict[str, int] = {}

    # Order matters — see the docstring. Each delete is scoped by org so we
    # can't accidentally spill across tenants even if the caller's session
    # somehow leaked a stale org_id.

    # 1. memory_records first (RESTRICT FK from memory back to signals).
    #    embeddings.memory_id CASCADEs, so vector rows go with the fact.
    r = await db.execute(
        sa_delete(MemoryRecord).where(
            MemoryRecord.organization_id == org_id,
            MemoryRecord.event_timestamp < cutoff,
        )
    )
    deleted["memory_records"] = r.rowcount or 0

    # 2. signal_events — safe now that dependent memory is gone.
    #    identity_link_events.source_signal_event_id is SET NULL on delete,
    #    so identity aliases persist without their provenance link.
    r = await db.execute(
        sa_delete(SignalEventRecord).where(
            SignalEventRecord.organization_id == org_id,
            SignalEventRecord.occurred_at < cutoff,
        )
    )
    deleted["signal_events"] = r.rowcount or 0

    # 3. signal_outbox — pointer rows for events we just deleted are noise
    #    (their aggregate_id no longer resolves). Purge all old rows for
    #    this org regardless of published state.
    r = await db.execute(
        sa_delete(SignalOutbox).where(
            SignalOutbox.organization_id == org_id,
            SignalOutbox.created_at < cutoff,
        )
    )
    deleted["signal_outbox"] = r.rowcount or 0

    # 4. signal_dead_letters — likewise, poison rows about events past the
    #    cutoff are of no operational value.
    r = await db.execute(
        sa_delete(SignalDeadLetter).where(
            SignalDeadLetter.organization_id == org_id,
            SignalDeadLetter.created_at < cutoff,
        )
    )
    deleted["signal_dead_letters"] = r.rowcount or 0

    # 5. daily_briefs — cached; they reference decisions computed off the
    #    old state. Dropping them means the next /brief/today rebuilds from
    #    the clean state via build_brief.
    r = await db.execute(
        sa_delete(DailyBrief).where(
            DailyBrief.organization_id == org_id,
            DailyBrief.as_of < cutoff,
        )
    )
    deleted["daily_briefs"] = r.rowcount or 0

    # 6. state_snapshots — full dispose for this org across all projection
    #    versions. The next state read (or the explicit rebuild below) will
    #    fold the remaining memory_records from scratch.
    await SnapshotRepository(db).delete(org_id)
    deleted["state_snapshots"] = 1  # boolean-ish; snapshot table has one row per (org, version)

    await db.commit()

    metrics.incr("admin.signals.pruned", value=deleted["signal_events"])
    logger.warning(
        "admin prune org=%s cutoff=%s deleted=%s",
        org_id, cutoff.isoformat(), deleted,
    )

    # 7. Rebuild state OUTSIDE the purge transaction — the fold streams the
    #    remaining memory_records and writes a fresh snapshot. Kept
    #    synchronous so the caller sees the new counts in one round trip;
    #    for very large orgs this could be moved to a Celery task instead.
    new_counts: dict[str, int] | None = None
    state_rebuilt = False
    try:
        state = await StateService(db).rebuild_org(org_id)
        new_counts = state.counts()
        state_rebuilt = True
    except Exception:
        # Rebuild failure is not fatal — the next /context / /brief request
        # will retry it automatically via get_state's rebuild-on-miss path.
        logger.exception("admin prune: state rebuild deferred org=%s", org_id)

    return PruneSignalsResponse(
        org_id=str(org_id),
        cutoff=cutoff.isoformat(),
        deleted=deleted,
        state_rebuilt=state_rebuilt,
        new_state_counts=new_counts,
    )
