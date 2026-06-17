"""CalendarConnector — production capture for Google Calendar.

    watch() -> syncToken -> events.list(syncToken) -> SignalEvents

Captures created / updated / cancelled events, attendee + conference changes, and
recurring masters. Version-aware via the event etag, so each edit to the same
event is a distinct, non-deduped signal. On a 410 GONE (expired syncToken) it
requests a full resync.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
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
from navis.signals.calendar.client import CalendarClient
from navis.signals.google.transport import GoogleAPIError

logger = get_logger("signals.calendar")

FULL_SYNC_WINDOW_DAYS = 60
MAX_PAGES = 100


class CalendarConnector(WorkspaceConnector):
    source = SignalSource.CALENDAR

    async def full_sync(self, ctx: SignalContext, client: CalendarClient) -> SyncResult:
        time_min = (
            datetime.now(timezone.utc) - timedelta(days=FULL_SYNC_WINDOW_DAYS)
        ).isoformat()
        return await self._drain(ctx, client, sync_token=None, time_min=time_min)

    async def incremental_sync(
        self, ctx: SignalContext, client: CalendarClient, cursor: dict[str, Any]
    ) -> SyncResult:
        sync_token = cursor.get("sync_token")
        if not sync_token:
            return SyncResult(full_resync_required=True)
        try:
            return await self._drain(ctx, client, sync_token=sync_token, time_min=None)
        except GoogleAPIError as exc:
            if exc.is_gone:  # 410 — syncToken expired
                logger.warning("calendar 410 org=%s; full resync required", ctx.org_id)
                return SyncResult(full_resync_required=True)
            raise

    async def _drain(
        self,
        ctx: SignalContext,
        client: CalendarClient,
        *,
        sync_token: str | None,
        time_min: str | None,
    ) -> SyncResult:
        events: list[SignalEvent] = []
        page_token: str | None = None
        next_sync_token: str | None = sync_token
        for _ in range(MAX_PAGES):
            resp = await client.list_events(
                sync_token=sync_token if page_token is None else None,
                page_token=page_token,
                time_min=time_min,
            )
            for raw in resp.get("items", []):
                events.append(self._to_event(ctx, raw))
            next_sync_token = resp.get("nextSyncToken") or next_sync_token
            page_token = resp.get("nextPageToken")
            if not page_token:
                break
        logger.info("calendar sync org=%s captured=%d", ctx.org_id, len(events))
        return SyncResult(events=events, cursor={"sync_token": next_sync_token})

    @staticmethod
    def _event_type(raw: dict[str, Any]) -> SignalEventType:
        if raw.get("status") == "cancelled":
            return SignalEventType.CALENDAR_EVENT_CANCELLED
        if raw.get("created") and raw.get("created") == raw.get("updated"):
            return SignalEventType.CALENDAR_EVENT_CREATED
        return SignalEventType.CALENDAR_EVENT_UPDATED

    @staticmethod
    def _version(raw: dict[str, Any]) -> str:
        etag = raw.get("etag")
        if etag:
            return etag.strip('"')
        if raw.get("sequence") is not None:
            return str(raw["sequence"])
        return str(raw.get("updated") or "1")

    @classmethod
    def _to_event(cls, ctx: SignalContext, raw: dict[str, Any]) -> SignalEvent:
        when = datetime.now(timezone.utc)
        start = raw.get("start", {})
        ts = start.get("dateTime") or raw.get("updated")
        if ts:
            try:
                when = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except (TypeError, ValueError):
                pass
        conference = raw.get("conferenceData")
        return SignalEvent.create(
            ctx,
            source=SignalSource.CALENDAR,
            event_type=cls._event_type(raw),
            external_id=raw["id"],
            external_version=cls._version(raw),
            timestamp=when,
            payload={
                "summary": raw.get("summary"),
                "status": raw.get("status"),
                "start": raw.get("start"),
                "end": raw.get("end"),
                "attendees": raw.get("attendees", []),
                "organizer": raw.get("organizer"),
                "recurring_event_id": raw.get("recurringEventId"),
                "recurrence": raw.get("recurrence"),
                "conference_data": conference,
                "hangout_link": raw.get("hangoutLink"),
            },
            metadata={
                "is_recurring_instance": bool(raw.get("recurringEventId")),
                "is_recurring_master": bool(raw.get("recurrence")),
                "has_conference": bool(conference or raw.get("hangoutLink")),
            },
        )
