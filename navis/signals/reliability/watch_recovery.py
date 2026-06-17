"""WatchRecoveryService — keep push channels alive.

Google push channels (Gmail watch, Calendar/Drive channels) expire. If a watch
lapses, push stops silently and the mailbox/calendar goes dark. This service
checks expiry and recreates channels before they lapse. It is pure: given a
connector + client + current watch state, it returns the new watch state to
persist (no identity writes here).
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from navis.platform.monitoring import get_logger
from navis.signals.base import SignalContext, WorkspaceConnector
from navis.signals.gmail.watch import is_watch_expiring

logger = get_logger("signals.watch_recovery")


class WatchRecoveryService:
    async def ensure_watch(
        self,
        ctx: SignalContext,
        connector: WorkspaceConnector,
        client: Any,
        current_watch: dict[str, Any] | None,
        *,
        now: datetime | None = None,
    ) -> dict[str, Any] | None:
        """Renew the watch if missing/expiring. Returns new watch state or None
        if no renewal was needed (caller persists when not None)."""
        if not is_watch_expiring(current_watch, now=now):
            return None
        new_state = await connector.create_watch(ctx, client)
        if not new_state:
            logger.warning(
                "watch renewal produced no state source=%s org=%s",
                connector.source.value,
                ctx.org_id,
            )
            return None
        logger.info("watch renewed source=%s org=%s", connector.source.value, ctx.org_id)
        return new_state
