"""SignalSyncService — the engine that drives a connector through one sync pass.

Ties together: connector (capture) + ingestion (idempotent persist + enqueue) +
DLQ (poison isolation) + Meet derivation. It does NOT touch identity: it returns
the new cursor + watch state for the caller (orchestration) to persist, and it
expects a ready `client` (token already resolved by the caller's TokenManager).

This separation is deliberate — signals own their log; identity owns tokens and
cursors. The engine is the seam.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from navis.platform.config import settings
from navis.platform.monitoring import get_logger
from navis.signals.base import (
    SignalContext,
    SignalEvent,
    SignalSource,
    SyncResult,
    WorkspaceConnector,
)
from navis.signals.meet.deriver import derive_meeting_events
from navis.signals.reliability.dead_letter import DeadLetterQueue, dead_letter_queue
from navis.signals.service import SignalIngestionService

logger = get_logger("signals.sync")


@dataclass(slots=True)
class SyncOutcome:
    ingested: int = 0
    duplicates: int = 0
    dead_lettered: int = 0
    cursor: dict[str, Any] = field(default_factory=dict)
    full_resync_performed: bool = False
    failed: bool = False


class SignalSyncService:
    def __init__(
        self,
        db: AsyncSession | None = None,
        *,
        ingestion: Any = None,
        dlq: DeadLetterQueue | None = None,
    ) -> None:
        # `ingestion` is injectable for tests; otherwise built from the session. One of the
        # two MUST be provided — a real ingestion service can't exist without a db session.
        if ingestion is not None:
            self.ingestion = ingestion
        elif db is not None:
            self.ingestion = SignalIngestionService(db)
        else:
            raise ValueError("SignalSyncService requires either `ingestion` or a `db` session")
        # Prefer a PostgreSQL-backed DLQ (source of truth) when we have a session.
        if dlq is not None:
            self.dlq = dlq
        elif db is not None:
            from navis.signals.repository import DeadLetterRepository

            self.dlq = DeadLetterQueue(repository=DeadLetterRepository(db))
        else:
            self.dlq = dead_letter_queue

    async def sync(
        self,
        ctx: SignalContext,
        connector: WorkspaceConnector,
        client: Any,
        cursor: dict[str, Any] | None,
    ) -> SyncOutcome:
        """Run incremental if we have a cursor, else full. Recover to full on demand."""
        cursor = cursor or {}

        async def _fetch() -> tuple[SyncResult, bool]:
            if cursor:
                res = await connector.incremental_sync(ctx, client, cursor)
                if res.full_resync_required:
                    logger.info("full resync for %s org=%s", connector.source.value, ctx.org_id)
                    return await connector.full_sync(ctx, client), True
                return res, False
            return await connector.full_sync(ctx, client), True

        try:
            # Bound a hung/slow connector (SIGNAL-004) so it can't pin a worker forever; a
            # timeout surfaces as a terminal fetch failure -> DLQ + cursor unchanged (retried).
            result, full = await asyncio.wait_for(
                _fetch(), timeout=settings.signals_sync_timeout_seconds
            )
        except Exception as exc:  # connector/transport already retried (or timed out) — terminal
            await self.dlq.send(
                stage="fetch",
                error=exc,
                context={
                    "source": connector.source.value,
                    "org_id": str(ctx.org_id),
                    "integration_id": str(ctx.integration_id),
                },
            )
            logger.exception("sync fetch failed source=%s org=%s", connector.source.value, ctx.org_id)
            return SyncOutcome(failed=True, cursor=cursor)

        events = list(result.events)
        # Meet rides on the Calendar pass — derived, never independently polled.
        if connector.source == SignalSource.CALENDAR:
            events.extend(derive_meeting_events(result.events))

        outcome = await self._ingest_all(ctx, connector.source, events)
        outcome.cursor = result.cursor or cursor
        outcome.full_resync_performed = full
        logger.info(
            "sync done source=%s org=%s ingested=%d dup=%d dlq=%d",
            connector.source.value,
            ctx.org_id,
            outcome.ingested,
            outcome.duplicates,
            outcome.dead_lettered,
        )
        return outcome

    async def _ingest_all(
        self, ctx: SignalContext, source: SignalSource, events: list[SignalEvent]
    ) -> SyncOutcome:
        outcome = SyncOutcome()
        # Deterministic total order (occurred_at, event_id) BEFORE persisting — connector
        # events and derived Meet events can share a timestamp; a stable order makes the
        # ingest sequence (and DLQ ordering) reproducible across re-runs (replay reads in
        # the same order). The event_id tiebreak is deterministic but not causal — a true
        # causal sequence would need a per-sync sequence number (future).
        ordered = sorted(events, key=lambda e: (e.timestamp, e.event_id))
        batch_size = max(1, settings.signals_ingest_batch_size)
        for start in range(0, len(ordered), batch_size):
            chunk = ordered[start : start + batch_size]
            try:
                # FAST PATH: one multi-row insert + one commit + publish only the new ones.
                inserted, duplicates = await self.ingestion.ingest_many(chunk)
                outcome.ingested += inserted
                outcome.duplicates += duplicates
            except Exception:
                # A poison row aborts the whole batch — drop to per-event so one bad event
                # can't sink the rest of the chunk (poison isolated to the DLQ).
                logger.warning(
                    "batch ingest failed source=%s org=%s; falling back per-event",
                    source.value, ctx.org_id,
                )
                await self._ingest_one_by_one(ctx, source, chunk, outcome)
        return outcome

    async def _ingest_one_by_one(
        self, ctx: SignalContext, source: SignalSource, chunk: list[SignalEvent], outcome: SyncOutcome
    ) -> None:
        for event in chunk:
            try:
                inserted = await self.ingestion.ingest(event)
                if inserted:
                    outcome.ingested += 1
                else:
                    outcome.duplicates += 1
            except Exception as exc:  # poison item — isolate, keep the batch flowing
                outcome.dead_lettered += 1
                await self.dlq.send(
                    stage="ingest",
                    error=exc,
                    context={"source": source.value, "org_id": str(ctx.org_id)},
                    item=event.to_dict(),
                )
