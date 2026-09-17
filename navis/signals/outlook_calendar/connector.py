"""OutlookCalendarConnector — production capture for Outlook Calendar.

    /me/calendarView/delta -> pages -> SignalEvents -> deltaLink

Emits the SAME event types the Google calendar connector emits
(CALENDAR_EVENT_CREATED / _UPDATED / _CANCELLED) for the same reason the
Outlook mail connector emits EMAIL_RECEIVED: the source names the vendor, the
event type names the verb, and every consumer downstream branches on the verb.

Payload is data-minimized on the same terms as the Google connector:
identity-bearing and timing fields are kept because the people graph and the
scheduling surfaces need them; free text and dial-in details are dropped at
capture rather than filtered later. A meeting body routinely carries conference
PINs, candidate names and salary figures, and the append-only signal log is
exactly the wrong place for those to live forever.
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
from navis.signals.microsoft.graph_time import graph_iso, parse_graph_datetime_field
from navis.signals.outlook_calendar.client import EVENT_FIELDS, OutlookCalendarClient

logger = get_logger("signals.outlook_calendar")


class OutlookCalendarConnector(WorkspaceConnector):
    source = SignalSource.OUTLOOK_CALENDAR

    async def full_sync(
        self, ctx: SignalContext, client: OutlookCalendarClient
    ) -> SyncResult:
        """Capture the current window and establish a delta cursor.

        The window straddles now — back `microsoft_calendar_window_days` for the
        recent past that a brief still reasons about, and forward by the same
        amount for what is coming. A future-only window would leave "you missed
        this meeting" unanswerable on day one; a past-only window would make the
        calendar useless for preparation, which is most of what it is for.
        """
        window = timedelta(days=max(1, settings.microsoft_calendar_window_days))
        now = datetime.now(timezone.utc)
        started = time.perf_counter()
        page = await drain_delta(
            client.transport,
            "/me/calendarView/delta",
            params={
                "$select": EVENT_FIELDS,
                "startDateTime": graph_iso(now - window),
                "endDateTime": graph_iso(now + window),
            },
            resource="outlook_calendar",
            item_cap=max(1, settings.microsoft_onboard_item_cap),
        )
        if page.full_resync_required:
            raise RuntimeError(
                "outlook_calendar full_sync received an expired delta token"
            )

        events = self._events(ctx, page.items, capture_mode="full")
        metrics.observe_ms(
            "outlook_calendar.sync.duration", (time.perf_counter() - started) * 1000.0
        )
        metrics.incr("outlook_calendar.sync.events", value=len(events))
        logger.info(
            "outlook_calendar full_sync org=%s items=%d events=%d pages=%d",
            ctx.org_id, len(page.items), len(events), page.pages,
        )
        return SyncResult(events=events, cursor={"delta_link": page.delta_link})

    async def incremental_sync(
        self, ctx: SignalContext, client: OutlookCalendarClient, cursor: dict[str, Any]
    ) -> SyncResult:
        delta_link = cursor.get("delta_link")
        if not delta_link:
            return SyncResult(full_resync_required=True)

        started = time.perf_counter()
        page = await drain_delta(
            client.transport, delta_link, resource="outlook_calendar"
        )
        if page.full_resync_required:
            # Also the path taken when the calendarView window has rolled far
            # enough that Graph retires the token. Expected, not exceptional.
            return SyncResult(full_resync_required=True)

        events = self._events(ctx, page.items, capture_mode="incremental")
        metrics.observe_ms(
            "outlook_calendar.sync.duration", (time.perf_counter() - started) * 1000.0
        )
        metrics.incr("outlook_calendar.sync.events", value=len(events))
        logger.info(
            "outlook_calendar incremental org=%s pages=%d captured=%d",
            ctx.org_id, page.pages, len(events),
        )
        return SyncResult(events=events, cursor={"delta_link": page.delta_link})

    # ---- mapping ----

    def _events(
        self, ctx: SignalContext, items: list[dict[str, Any]], *, capture_mode: str
    ) -> list[SignalEvent]:
        events: list[SignalEvent] = []
        for raw in items:
            event = self._to_event(ctx, raw, capture_mode=capture_mode)
            if event is not None:
                events.append(event)
        return events

    @staticmethod
    def _event_type(raw: dict[str, Any]) -> SignalEventType:
        if raw.get("isCancelled"):
            return SignalEventType.CALENDAR_EVENT_CANCELLED
        created = raw.get("createdDateTime")
        modified = raw.get("lastModifiedDateTime")
        # Graph sets the two within microseconds of each other on creation but
        # not always identically, unlike Google which sets them equal. Treating
        # "modified within a second of creation" as a create keeps the verb
        # honest without a separate lookup.
        if created and modified:
            c = parse_graph_datetime_field({"dateTime": created})
            m = parse_graph_datetime_field({"dateTime": modified})
            if c and m and abs((m - c).total_seconds()) < 1.0:
                return SignalEventType.CALENDAR_EVENT_CREATED
        return SignalEventType.CALENDAR_EVENT_UPDATED

    @staticmethod
    def _min_attendee(attendee: dict[str, Any]) -> dict[str, Any]:
        """Keep the identity (email) plus non-PII response flags; drop the
        display name and any free-text comment. Mirrors the Google connector's
        `_min_attendee` field for field so the two suites produce interchangeable
        attendee records."""
        email_address = attendee.get("emailAddress") or {}
        status = attendee.get("status") or {}
        address = email_address.get("address")
        return {
            "email": address.lower() if isinstance(address, str) else None,
            "response_status": status.get("response"),
            "optional": attendee.get("type") == "optional",
            "resource": attendee.get("type") == "resource",
        }

    @staticmethod
    def _min_conference(raw: dict[str, Any]) -> dict[str, Any] | None:
        """Presence and provider only.

        `onlineMeeting` carries `joinUrl`, `tollNumber` and `conferenceId` —
        a dial-in and its access code. Keeping the flags lets the product say
        "this is a video call"; keeping the rest would put working conference
        credentials in an append-only log and every backup of it.
        """
        if not raw.get("isOnlineMeeting") and not raw.get("onlineMeeting"):
            return None
        return {
            "solution": raw.get("onlineMeetingProvider"),
            "has_video": True,
            "has_phone": bool((raw.get("onlineMeeting") or {}).get("tollNumber")),
        }

    def _to_event(
        self, ctx: SignalContext, raw: dict[str, Any], *, capture_mode: str
    ) -> SignalEvent | None:
        event_id = raw.get("id")
        if not event_id:
            return None
        if "@removed" in raw:
            # Graph reports a deleted event as a removal rather than as a
            # cancellation. There is no DELETED verb; the cancellation verb
            # would be a lie (nobody cancelled it, it was removed from the
            # window or the mailbox), so this is skipped for the same reason
            # the mail connector skips removals.
            metrics.incr("outlook_calendar.sync.removed_skipped")
            return None

        start = raw.get("start")
        when = parse_graph_datetime_field(start) or datetime.now(timezone.utc)
        organizer_address = (
            ((raw.get("organizer") or {}).get("emailAddress") or {}).get("address")
        )
        attendees = raw.get("attendees") or []
        conference = self._min_conference(raw)

        return SignalEvent.create(
            ctx,
            source=SignalSource.OUTLOOK_CALENDAR,
            event_type=self._event_type(raw),
            external_id=str(event_id),
            # changeKey is Graph's per-item version token — see the Outlook mail
            # connector for why a timestamp is not an acceptable substitute.
            external_version=str(
                raw.get("changeKey") or raw.get("lastModifiedDateTime") or "1"
            ),
            timestamp=when,
            payload={
                # Keyed to match the Google calendar payload so downstream
                # consumers read one shape. Graph's `subject` is Google's
                # `summary`; the translation belongs here, not in every reader.
                "summary": raw.get("subject"),
                "status": "cancelled" if raw.get("isCancelled") else "confirmed",
                "start": raw.get("start"),
                "end": raw.get("end"),
                "is_all_day": raw.get("isAllDay"),
                "show_as": raw.get("showAs"),
                "attendees": [self._min_attendee(a) for a in attendees],
                "attendee_count": len(attendees),
                "organizer": (
                    {"email": organizer_address.lower()}
                    if isinstance(organizer_address, str)
                    else None
                ),
                # Graph's series master id is the analogue of Google's
                # recurringEventId; both answer "which series is this from".
                "recurring_event_id": raw.get("seriesMasterId"),
                "recurrence": raw.get("recurrence"),
                "conference_data": conference,
                "web_link": raw.get("webLink"),
            },
            metadata={
                "provider": "microsoft",
                "is_recurring_instance": raw.get("type") in ("occurrence", "exception"),
                "is_recurring_master": raw.get("type") == "seriesMaster",
                "has_conference": bool(conference),
                "capture_mode": capture_mode,
            },
        )
