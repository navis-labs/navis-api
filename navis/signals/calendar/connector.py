"""CalendarConnector — production capture for Google Calendar.

    watch() -> syncToken -> events.list(syncToken) -> SignalEvents

Captures created / updated / cancelled events, attendee + conference changes, and
recurring masters. Version-aware via the event etag, so each edit to the same event is a
distinct, non-deduped signal (complete reality capture; de-noising is a downstream
concern, NOT a signal concern). On a 410 GONE (expired syncToken) it requests a full
resync.

Pagination follows Google's incremental-sync contract: page through nextPageToken until
the FINAL page yields nextSyncToken — intermediate pages have NO syncToken, so the drain
MUST complete to obtain a valid cursor. A safety cap turns a pathological calendar into a
LOUD, retryable failure (DLQ) rather than silently dropping pages and stalling the cursor.

Payload is data-minimized: identity-bearing fields (attendee emails, timing, recurrence,
conference presence) are kept for downstream people/meeting resolution; bulk PII and
sensitive details (attendee names/comments, conference phone numbers / PINs / URIs) are
dropped. event_id is a UUIDv5 of (org, source, external_id, external_version) — independent
of payload — so minimization never affects determinism or replay.
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
from navis.signals.calendar.client import CalendarClient
from navis.signals.google.transport import GoogleAPIError

logger = get_logger("signals.calendar")


class CalendarSyncIncomplete(Exception):
    """A drain hit the page-safety cap before reaching nextSyncToken. Raised (not silently
    swallowed) so the sync fails loud + retryable instead of stalling the cursor."""


class CalendarConnector(WorkspaceConnector):
    source = SignalSource.CALENDAR

    async def full_sync(self, ctx: SignalContext, client: CalendarClient) -> SyncResult:
        time_min = (
            datetime.now(timezone.utc) - timedelta(days=settings.calendar_full_sync_window_days)
        ).isoformat()
        return await self._drain(ctx, client, sync_token=None, time_min=time_min, capture_mode="full")

    async def incremental_sync(
        self, ctx: SignalContext, client: CalendarClient, cursor: dict[str, Any]
    ) -> SyncResult:
        sync_token = cursor.get("sync_token")
        if not sync_token:
            return SyncResult(full_resync_required=True)
        try:
            return await self._drain(
                ctx, client, sync_token=sync_token, time_min=None, capture_mode="incremental"
            )
        except GoogleAPIError as exc:
            if exc.is_gone:  # 410 — syncToken expired
                metrics.incr("calendar.sync.token_expired")
                metrics.incr("calendar.sync.full_resync")
                logger.warning("calendar 410 org=%s; full resync required", ctx.org_id)
                return SyncResult(full_resync_required=True)
            metrics.incr("calendar.sync.errors", labels={"kind": "google_api"})
            raise

    async def _drain(
        self,
        ctx: SignalContext,
        client: CalendarClient,
        *,
        sync_token: str | None,
        time_min: str | None,
        capture_mode: str,
    ) -> SyncResult:
        calendar_id = getattr(client, "calendar_id", "primary")
        events: list[SignalEvent] = []
        page_token: str | None = None
        next_sync_token: str | None = None  # ONLY a real nextSyncToken counts (never the old one)
        pages = 0
        max_pages = max(1, settings.calendar_max_sync_pages)
        started = time.perf_counter()

        while True:
            resp = await client.list_events(
                # syncToken only on the FIRST request; subsequent requests use pageToken.
                sync_token=sync_token if page_token is None else None,
                page_token=page_token,
                time_min=time_min if page_token is None else None,
            )
            for raw in resp.get("items", []):
                events.append(self._to_event(ctx, raw, calendar_id=calendar_id, capture_mode=capture_mode))
            pages += 1
            next_sync_token = resp.get("nextSyncToken") or next_sync_token
            page_token = resp.get("nextPageToken")
            if not page_token:
                break  # reached the final page -> nextSyncToken is now established
            if pages >= max_pages:
                # Cap hit BEFORE the final page. Saving a cursor now would drop pages and
                # stall the token forever -> fail loud + retryable instead.
                metrics.incr("calendar.sync.truncated")
                logger.error(
                    "calendar drain exceeded %d pages without a syncToken org=%s cal=%s",
                    max_pages, ctx.org_id, calendar_id,
                )
                raise CalendarSyncIncomplete(
                    f"calendar sync exceeded {max_pages} pages without a syncToken"
                )

        metrics.observe_ms("calendar.sync.duration", (time.perf_counter() - started) * 1000.0)
        metrics.incr("calendar.sync.pages", value=pages)
        metrics.incr("calendar.sync.events", value=len(events))
        logger.info(
            "calendar sync org=%s cal=%s mode=%s pages=%d captured=%d",
            ctx.org_id, calendar_id, capture_mode, pages, len(events),
        )
        # Prefer the new token; fall back to the old one for incremental safety (the final
        # page always yields nextSyncToken, so this fallback is defensive only).
        return SyncResult(events=events, cursor={"sync_token": next_sync_token or sync_token})

    @staticmethod
    def _event_type(raw: dict[str, Any]) -> SignalEventType:
        if raw.get("status") == "cancelled":
            return SignalEventType.CALENDAR_EVENT_CANCELLED
        if raw.get("created") and raw.get("created") == raw.get("updated"):
            return SignalEventType.CALENDAR_EVENT_CREATED
        return SignalEventType.CALENDAR_EVENT_UPDATED

    @staticmethod
    def _version(raw: dict[str, Any]) -> str:
        # etag changes on ANY modification -> every edit is a distinct fact (complete
        # capture). Choosing a coarser key (sequence) would COLLAPSE edits and lose reality.
        etag = raw.get("etag")
        if etag:
            return etag.strip('"')
        if raw.get("sequence") is not None:
            return str(raw["sequence"])
        return str(raw.get("updated") or "1")

    @staticmethod
    def _min_attendee(a: dict[str, Any]) -> dict[str, Any]:
        # Keep the identity (email) + non-PII signal flags; drop displayName, comment,
        # additionalGuests, photo (bulk PII / free text that must not live forever).
        return {
            "email": a.get("email"),
            "response_status": a.get("responseStatus"),
            "optional": bool(a.get("optional")),
            "resource": bool(a.get("resource")),
        }

    @staticmethod
    def _min_conference(conference: dict[str, Any] | None) -> dict[str, Any] | None:
        if not conference:
            return None
        entry_types = {e.get("entryPointType") for e in conference.get("entryPoints", [])}
        # Keep presence + id; DROP entryPoints (phone numbers, PINs, access codes, URIs).
        return {
            "conference_id": conference.get("conferenceId"),
            "solution": (conference.get("conferenceSolution") or {}).get("key", {}).get("type"),
            "has_video": "video" in entry_types,
            "has_phone": "phone" in entry_types,
        }

    @classmethod
    def _to_event(
        cls, ctx: SignalContext, raw: dict[str, Any], *, calendar_id: str, capture_mode: str
    ) -> SignalEvent:
        when = datetime.now(timezone.utc)
        ts = (raw.get("start") or {}).get("dateTime") or raw.get("updated")
        if ts:
            try:
                when = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except (TypeError, ValueError):
                pass
        conference = raw.get("conferenceData")
        organizer = raw.get("organizer") or {}
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
                # minimized: email + status flags only (downstream resolves people by email)
                "attendees": [cls._min_attendee(a) for a in raw.get("attendees", [])],
                "attendee_count": len(raw.get("attendees", [])),
                "organizer": {"email": organizer.get("email")} if organizer else None,
                "recurring_event_id": raw.get("recurringEventId"),
                "recurrence": raw.get("recurrence"),
                # moved-occurrence lineage: when an instance was relocated from the series time
                "original_start_time": raw.get("originalStartTime"),
                # minimized: presence/id only — NO phone/PIN/URIs; key kept for has_conference
                "conference_data": cls._min_conference(conference),
                "hangout_link": raw.get("hangoutLink"),
            },
            metadata={
                "is_recurring_instance": bool(raw.get("recurringEventId")),
                "is_recurring_master": bool(raw.get("recurrence")),
                "has_conference": bool(conference or raw.get("hangoutLink")),
                # calendar lineage: which calendar + how it was captured (Issue F)
                "calendar_id": calendar_id,
                "capture_mode": capture_mode,
            },
        )
