"""BackfillService — bounded historical capture for an integration.

A thin wrapper over the sync engine forcing the full-sync path (cursor=None).
Used at onboarding and after a cursor-recovery event. Returns the new cursor for
the caller to persist.
"""
from __future__ import annotations

from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from navis.signals.base import SignalContext, WorkspaceConnector
from navis.signals.sync import SignalSyncService, SyncOutcome


class BackfillService:
    def __init__(self, db: AsyncSession) -> None:
        self.engine = SignalSyncService(db)

    async def backfill(
        self, ctx: SignalContext, connector: WorkspaceConnector, client: Any
    ) -> SyncOutcome:
        # cursor=None forces full_sync inside the engine.
        return await self.engine.sync(ctx, connector, client, cursor=None)
