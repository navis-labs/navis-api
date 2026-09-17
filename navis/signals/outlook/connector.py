"""OutlookConnector — production capture for Outlook mail.

    /me/mailFolders/inbox/messages/delta -> pages -> SignalEvents -> deltaLink

Pure capture — no DB writes, no business logic. The sync engine persists cursors
and ingests the returned events, exactly as it does for Gmail.

── Two things worth knowing before changing this ─────────────────────────

**The emitted event type is EMAIL_RECEIVED, the same verb Gmail emits.** The
`source` field says the mail came from Outlook; the event type says what
happened. Everything downstream branches on the verb, so an Outlook message
takes the identical path a Gmail message does with no new code. Adding an
`outlook.email.received` type would fork every consumer in the pipeline and
gain nothing.

**`external_version` is a CONSTANT.** "This email arrived" is a one-time
immutable fact, so exactly one signal should ever exist per message. Graph's
changeKey moves on ANY property write — isRead toggling, a category, a flag —
so versioning by it mints a new event_id every time the founder merely opens or
files a mail, re-entering the same arrival in the append-only log as a duplicate
fact. Gmail hit this exact bug with historyId and fixed it the same way (see
`gmail/connector.py`). Calendar and OneDrive deliberately do the opposite: an
edited meeting or a new file revision IS a distinct fact worth capturing, so
those connectors version by changeKey/eTag.
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from typing import Any

from navis.platform.config import settings
from navis.platform.monitoring import get_logger
from navis.platform.monitoring.metrics import metrics
from navis.signals.base import (
    SignalContext,
    SignalEvent,
    SignalEventType,
    SignalSource,
    SyncResult,
    WorkspaceConnector,
)
from navis.signals.microsoft.delta import drain_delta
from navis.signals.microsoft.graph_time import graph_iso, parse_graph_datetime
from navis.signals.outlook.client import MESSAGE_FIELDS, OutlookClient
from navis.signals.outlook.parser import parse_message

logger = get_logger("signals.outlook")


class OutlookConnector(WorkspaceConnector):
    source = SignalSource.OUTLOOK

    async def full_sync(self, ctx: SignalContext, client: OutlookClient) -> SyncResult:
        """Bounded backfill at onboarding, and the recovery path when a delta
        token expires.

        The lookback window matches Gmail's so a founder who connects Outlook
        and one who connects Gmail see the same horizon on day one. Without it,
        a 3-week-old thread surfaces as a "slipping commitment" the moment the
        mailbox is connected.
        """
        since = datetime.now(timezone.utc) - timedelta(
            days=max(1, settings.microsoft_onboard_lookback_days)
        )
        started = time.perf_counter()
        page = await drain_delta(
            client.transport,
            "/me/mailFolders/inbox/messages/delta",
            params={
                "$select": MESSAGE_FIELDS,
                "$filter": f"receivedDateTime ge {graph_iso(since)}",
            },
            resource="outlook",
            item_cap=max(1, settings.microsoft_onboard_item_cap),
        )
        if page.full_resync_required:
            # A brand-new enumeration cannot have an expired token; if Graph
            # says otherwise something is wrong enough to fail loudly rather
            # than loop between full and incremental forever.
            raise RuntimeError("outlook full_sync received an expired delta token")

        events = self._events(ctx, page.items)
        metrics.observe_ms("outlook.sync.duration", (time.perf_counter() - started) * 1000.0)
        metrics.incr("outlook.sync.events", value=len(events))
        logger.info(
            "outlook full_sync org=%s messages=%d events=%d pages=%d",
            ctx.org_id, len(page.items), len(events), page.pages,
        )
        return SyncResult(events=events, cursor={"delta_link": page.delta_link})

    async def incremental_sync(
        self, ctx: SignalContext, client: OutlookClient, cursor: dict[str, Any]
    ) -> SyncResult:
        delta_link = cursor.get("delta_link")
        if not delta_link:
            return SyncResult(full_resync_required=True)

        started = time.perf_counter()
        page = await drain_delta(
            client.transport, delta_link, resource="outlook"
        )
        if page.full_resync_required:
            # Token aged out. Ask for a full sync rather than persisting a
            # cursor we know Graph has stopped honouring.
            return SyncResult(full_resync_required=True)

        events = self._events(ctx, page.items)
        metrics.observe_ms("outlook.sync.duration", (time.perf_counter() - started) * 1000.0)
        metrics.incr("outlook.sync.events", value=len(events))
        logger.info(
            "outlook incremental org=%s pages=%d captured=%d",
            ctx.org_id, page.pages, len(events),
        )
        return SyncResult(events=events, cursor={"delta_link": page.delta_link})

    # ---- mapping ----

    def _events(
        self, ctx: SignalContext, messages: list[dict[str, Any]]
    ) -> list[SignalEvent]:
        events: list[SignalEvent] = []
        for raw in messages:
            event = self._event_for_message(ctx, raw)
            if event is not None:
                events.append(event)
        return events

    def _event_for_message(
        self, ctx: SignalContext, raw: dict[str, Any]
    ) -> SignalEvent | None:
        message_id = raw.get("id")
        if not message_id:
            return None

        # A delta page marks deletions with @removed. There is no
        # EMAIL_DELETED verb in the ontology and inventing one here would give
        # every downstream consumer a fact it has no handling for, so a
        # deletion is skipped: the signal log is append-only reality capture,
        # and "this message no longer exists" is a state question the state
        # layer answers, not a new fact about the world.
        if "@removed" in raw:
            metrics.incr("outlook.sync.removed_skipped")
            return None

        parsed = parse_message(raw)
        timestamp = parse_graph_datetime(
            raw.get("receivedDateTime") or raw.get("sentDateTime")
        )
        return SignalEvent.create(
            ctx,
            source=SignalSource.OUTLOOK,
            # Same verb Gmail emits — see the module docstring.
            event_type=SignalEventType.EMAIL_RECEIVED,
            external_id=str(message_id),
            # Constant on purpose — see the module docstring. (org, outlook,
            # message_id) is the stable dedupe identity: one signal per email,
            # idempotent across re-syncs and read/flag churn.
            external_version="1",
            timestamp=timestamp,
            payload=parsed,
            metadata={
                "provider": "microsoft",
                "thread_id": raw.get("conversationId"),
                # Security boundary: an email body is authored by an EXTERNAL party, so it is
                # attacker-controllable ("Ignore all previous instructions..."). We stamp
                # provenance + the one hard gate at capture, where origin is known with
                # certainty. content may be REASONED over (that is the product — extracting
                # commitments/decisions), but its embedded text must NEVER be EXECUTED as a
                # command by any future agent. So we forbid execution, not reasoning.
                "source_type": "external_email",
                "content_trust_level": "untrusted",
                "trusted_for_execution": False,
            },
        )

