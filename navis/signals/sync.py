"""SignalSyncService — the engine that drives a connector through one sync pass.

Ties together: connector (capture) + ingestion (idempotent persist + enqueue) +
DLQ (poison isolation) + Meet derivation. It does NOT touch identity: it returns
the new cursor + watch state for the caller (orchestration) to persist, and it
expects a ready `client` (token already resolved by the caller's TokenManager).

This separation is deliberate — signals own their log; identity owns tokens and
cursors. The engine is the seam.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from navis.platform.monitoring import get_logger
from navis.signals.base import (
    SignalContext,
    SignalEvent,
    SignalSource,
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
        # `ingestion` is injectable for tests; otherwise built from the session.
        self.ingestion = ingestion or SignalIngestionService(db)
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
        try:
            if cursor:
                result = await connector.incremental_sync(ctx, client, cursor)
                if result.full_resync_required:
                    logger.info("full resync for %s org=%s", connector.source.value, ctx.org_id)
                    result = await connector.full_sync(ctx, client)
                    full = True
                else:
                    full = False
            else:
                result = await connector.full_sync(ctx, client)
                full = True
        except Exception as exc:  # connector/transport already retried — this is terminal
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
        for event in events:
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
        return outcome
