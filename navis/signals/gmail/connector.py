"""GmailConnector — production capture for Gmail.

Architecture (Google's recommended pattern):

    watch() -> Pub/Sub notification -> historyId -> history.list() -> fetch -> SignalEvents

  * full_sync: bounded backfill at onboarding; records the starting historyId.
  * incremental_sync: history.list paginated from the stored historyId; advances
    the cursor; on 404 (historyId too old / expired) requests a full resync.
  * create_watch: registers the Pub/Sub channel and returns watch state to persist.

Pure capture — no DB writes, no business logic. The sync engine persists cursors
and ingests the returned events.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

from navis.platform.monitoring import get_logger
from navis.signals.base import (
    SignalContext,
    SignalEvent,
    SignalEventType,
    SignalSource,
    SyncResult,
    WorkspaceConnector,
)
from navis.signals.gmail.client import GmailClient
from navis.signals.gmail.parser import parse_message
from navis.signals.gmail.watch import build_watch_state
from navis.signals.google.transport import GoogleAPIError

logger = get_logger("signals.gmail")

# Onboarding backfill cap — keeps first sync under the 60s target.
ONBOARD_MESSAGE_CAP = 500
# Safety cap on history pages per pass (incremental should be small).
MAX_HISTORY_PAGES = 50


class GmailConnector(WorkspaceConnector):
    source = SignalSource.GMAIL

    async def full_sync(self, ctx: SignalContext, client: GmailClient) -> SyncResult:
        profile = await client.get_profile()
        start_history_id = str(profile.get("historyId"))

        message_ids: list[str] = []
        page_token: str | None = None
        while len(message_ids) < ONBOARD_MESSAGE_CAP:
            listing = await client.list_messages(page_token=page_token, max_results=100)
            message_ids.extend(m["id"] for m in listing.get("messages", []))
            page_token = listing.get("nextPageToken")
            if not page_token:
                break
        message_ids = message_ids[:ONBOARD_MESSAGE_CAP]

        events = await self._fetch_and_normalize(ctx, client, message_ids)
        logger.info("gmail full_sync org=%s captured=%d", ctx.org_id, len(events))
        return SyncResult(events=events, cursor={"history_id": start_history_id})

    async def incremental_sync(
        self, ctx: SignalContext, client: GmailClient, cursor: dict[str, Any]
    ) -> SyncResult:
        history_id = cursor.get("history_id")
        if not history_id:
            return SyncResult(full_resync_required=True)

        message_ids: list[str] = []
        latest_history_id = str(history_id)
        page_token: str | None = None
        try:
            for _ in range(MAX_HISTORY_PAGES):
                history = await client.list_history(latest_history_id, page_token=page_token)
                for h in history.get("history", []):
                    for added in h.get("messagesAdded", []):
                        msg = added.get("message", {})
                        if msg.get("id"):
                            message_ids.append(msg["id"])
                if history.get("historyId"):
                    latest_history_id = str(history["historyId"])
                page_token = history.get("nextPageToken")
                if not page_token:
                    break
        except GoogleAPIError as exc:
            if exc.is_not_found:
                # historyId expired — Google requires a full resync.
                logger.warning("gmail history 404 org=%s; full resync required", ctx.org_id)
                return SyncResult(full_resync_required=True)
            raise

        # De-dup ids within the batch before fetching.
        unique_ids = list(dict.fromkeys(message_ids))
        events = await self._fetch_and_normalize(ctx, client, unique_ids)
        logger.info("gmail incremental org=%s captured=%d", ctx.org_id, len(events))
        return SyncResult(events=events, cursor={"history_id": latest_history_id})

    async def create_watch(self, ctx: SignalContext, client: GmailClient) -> dict[str, Any]:
        topic = os.environ.get("GOOGLE_PUBSUB_TOPIC")
        if not topic:
            logger.warning("gmail watch skipped org=%s: no Pub/Sub topic configured", ctx.org_id)
            return {}
        resp = await client.watch(topic)
        return build_watch_state(resp, topic=topic)

    async def _fetch_and_normalize(
        self, ctx: SignalContext, client: GmailClient, message_ids: list[str]
    ) -> list[SignalEvent]:
        if not message_ids:
            return []
        raw_messages = await client.get_messages(message_ids)
        events: list[SignalEvent] = []
        for raw in raw_messages:
            parsed = parse_message(raw)
            if not parsed.get("message_id"):
                continue
            events.append(self._to_event(ctx, parsed))
        return events

    @staticmethod
    def _to_event(ctx: SignalContext, parsed: dict[str, Any]) -> SignalEvent:
        when = datetime.now(timezone.utc)
        if parsed.get("internal_date"):
            try:
                when = datetime.fromtimestamp(int(parsed["internal_date"]) / 1000, tz=timezone.utc)
            except (TypeError, ValueError):
                pass
        elif parsed.get("date"):
            try:
                when = parsedate_to_datetime(parsed["date"])
            except (TypeError, ValueError):
                pass
        # Emails are immutable; version by the message's historyId (fallback "1").
        version = str(parsed.get("history_id") or "1")
        return SignalEvent.create(
            ctx,
            source=SignalSource.GMAIL,
            event_type=SignalEventType.EMAIL_RECEIVED,
            external_id=parsed["message_id"],
            external_version=version,
            timestamp=when,
            payload=parsed,
            metadata={
                "thread_id": parsed.get("thread_id"),
                "label_ids": parsed.get("label_ids", []),
            },
        )
