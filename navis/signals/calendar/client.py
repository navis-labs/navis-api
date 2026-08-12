"""Google Calendar API client over GoogleTransport.

events.list supports incremental sync via syncToken. On a 410 GONE the token is
invalid and a full resync (timeMin window) is required — the connector handles
that; this client just surfaces the error from the transport.
"""
from __future__ import annotations

import hashlib
from typing import Any

from navis.signals.google.transport import GoogleTransport

_BASE = "https://www.googleapis.com/calendar/v3"

# `entryPoints` also carries phone numbers, PINs and access codes. Only the
# video entry is ever read out of a conference — see `conference_info`.
_VIDEO_ENTRY_POINT = "video"


def _conference_request_id(
    event_id: str | None, summary: str, start_iso: str, end_iso: str
) -> str:
    """Stable idempotency key for ONE conference-creation request.

    Google treats `requestId` as the idempotency key for conference creation:
    replaying an insert with the same id returns the SAME conference instead of
    minting a second one. That guarantee is only worth something if the id
    survives a process restart, so this must never use Python's builtin
    `hash()` — string hashing is randomized per process via PYTHONHASHSEED, so
    two workers (or the same worker after a restart) would compute different
    ids for the same meeting and Google would create a second conference.

    When the caller supplies a client-generated `event_id` — the execution
    worker always does, as `navis{execution_id.hex}` — the conference id is
    derived from it, so an event and its conference share ONE deterministic
    identity rooted in the ExecutionRequest. The SHA-256 fallback covers
    callers that let Google assign the event id; it is content-addressed and
    therefore just as stable across processes.
    """
    if event_id:
        return f"{event_id}-conf"
    digest = hashlib.sha256(f"{summary}|{start_iso}|{end_iso}".encode()).hexdigest()
    return f"navisconf-{digest[:32]}"


def conference_info(event: dict[str, Any]) -> dict[str, Any]:
    """Meet join URL + conference status read off an events.insert response.

    Conference creation is ASYNCHRONOUS. Google may return the created event
    with `conferenceData.createRequest.status.statusCode == 'pending'` and no
    join URL yet, and it may later resolve to `failure`. Both are reported
    verbatim rather than smoothed over: an absent Meet link is recoverable, a
    fabricated one sends people to a room that does not exist.

    Only the video entry point is surfaced. Phone numbers, PINs and access
    codes stay out of the execution result, matching the minimization the
    ingest connector already applies to conferences it observes.

    Returns `{}` for an ordinary event with no conference attached.
    """
    conference = event.get("conferenceData") or {}
    status = ((conference.get("createRequest") or {}).get("status") or {}).get(
        "statusCode"
    )

    url = event.get("hangoutLink")
    if not url:
        for entry in conference.get("entryPoints") or []:
            if entry.get("entryPointType") == _VIDEO_ENTRY_POINT and entry.get("uri"):
                url = entry["uri"]
                break

    info: dict[str, Any] = {}
    if url:
        info["meet_url"] = url
    if status:
        # Google's own word for it, even when that word is 'pending'/'failure'.
        info["conference_status"] = status
    elif url:
        # A link with no createRequest block: the conference already exists
        # (e.g. a replayed 409 re-fetch). Having the URL IS the success signal.
        info["conference_status"] = "success"
    return info


class CalendarClient:
    def __init__(self, transport: GoogleTransport, *, calendar_id: str = "primary") -> None:
        self._t = transport
        self._cal = calendar_id

    @property
    def calendar_id(self) -> str:
        return self._cal

    async def list_events(
        self,
        *,
        sync_token: str | None = None,
        page_token: str | None = None,
        time_min: str | None = None,
        max_results: int = 250,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "maxResults": max_results,
            "showDeleted": True,  # required to receive cancellations
            "singleEvents": False,  # keep recurring masters; expand downstream
        }
        if sync_token:
            params["syncToken"] = sync_token
        elif time_min:
            params["timeMin"] = time_min
        if page_token:
            params["pageToken"] = page_token
        return await self._t.request("GET", f"{_BASE}/calendars/{self._cal}/events", params=params)

    async def list_events_expanded(
        self,
        *,
        time_min: str,
        time_max: str,
        max_results: int = 50,
    ) -> dict[str, Any]:
        """List events in a time window, expanded for recurring instances.

        Unlike `list_events` (sync-oriented, keeps recurring MASTER rows),
        this method sets `singleEvents=True` + `orderBy=startTime` so each
        occurrence in the window shows up as its own event with a real
        `start`/`end`. Suitable for the chat "what meetings do I have
        today?" query flow — the caller renders these events verbatim.
        """
        params: dict[str, Any] = {
            "maxResults": max_results,
            "singleEvents": True,
            "orderBy": "startTime",
            "timeMin": time_min,
            "timeMax": time_max,
            "showDeleted": False,
        }
        return await self._t.request(
            "GET", f"{_BASE}/calendars/{self._cal}/events", params=params
        )

    async def watch(self, channel_id: str, address: str, token: str | None = None) -> dict[str, Any]:
        body: dict[str, Any] = {"id": channel_id, "type": "web_hook", "address": address}
        if token:
            body["token"] = token
        return await self._t.request(
            "POST", f"{_BASE}/calendars/{self._cal}/events/watch", json=body
        )

    async def stop_channel(self, channel_id: str, resource_id: str) -> dict[str, Any]:
        return await self._t.request(
            "POST", f"{_BASE}/channels/stop", json={"id": channel_id, "resourceId": resource_id}
        )

    async def freebusy(
        self,
        *,
        time_min: str,
        time_max: str,
        calendar_ids: list[str] | None = None,
        time_zone: str = "UTC",
    ) -> dict[str, Any]:
        """Query blocks of busy time in the given window.

        Returns the raw Google freeBusy response — callers subtract busy blocks from
        the window to derive open slots. `calendar_ids` defaults to the client's
        primary calendar so a single OAuth grant covers the common "one founder,
        one calendar" case; consumers with delegated / shared calendars can pass
        the full list.
        """
        body: dict[str, Any] = {
            "timeMin": time_min,
            "timeMax": time_max,
            "timeZone": time_zone,
            "items": [{"id": cid} for cid in (calendar_ids or [self._cal])],
        }
        return await self._t.request("POST", "https://www.googleapis.com/calendar/v3/freeBusy", json=body)

    async def create_event(
        self,
        *,
        summary: str,
        start_iso: str,
        end_iso: str,
        attendees: list[str] | None = None,
        description: str | None = None,
        location: str | None = None,
        time_zone: str = "UTC",
        send_updates: str = "all",
        conference: bool = False,
        event_id: str | None = None,
    ) -> dict[str, Any]:
        """Create a calendar event on the client's calendar.

        `send_updates='all'` mails every attendee; pass `'none'` for silent creation
        (useful when the caller sends its own confirmation via Gmail so recipients
        don't see two notifications). `conference=True` requests Google Meet by
        binding a conferenceData block — the API returns the Meet link on the
        created event's `hangoutLink` field.

        `event_id` supplies a CLIENT-GENERATED event id. Google enforces
        uniqueness per calendar, so passing a deterministic id turns
        events.insert into an idempotent operation: a replayed insert returns
        409 instead of creating a second event. Callers that want at-most-once
        semantics (the execution worker does) MUST pass one — see
        `navis.execution.providers._senders.calendar_sender`. The id must be
        5-1024 chars from the base32hex alphabet (a-v, 0-9).
        """
        body: dict[str, Any] = {
            "summary": summary,
            "start": {"dateTime": start_iso, "timeZone": time_zone},
            "end": {"dateTime": end_iso, "timeZone": time_zone},
        }
        if event_id:
            body["id"] = event_id
        if description:
            body["description"] = description
        if location:
            body["location"] = location
        if attendees:
            body["attendees"] = [{"email": a} for a in attendees]
        if conference:
            body["conferenceData"] = {
                "createRequest": {
                    "requestId": _conference_request_id(
                        event_id, summary, start_iso, end_iso
                    ),
                    "conferenceSolutionKey": {"type": "hangoutsMeet"},
                }
            }
        params = {"sendUpdates": send_updates}
        if conference:
            params["conferenceDataVersion"] = "1"
        return await self._t.request(
            "POST",
            f"{_BASE}/calendars/{self._cal}/events",
            params=params,
            json=body,
        )

    async def get_event(self, event_id: str) -> dict[str, Any]:
        """Fetch one event by id. Raises GoogleAPIError(404) when absent."""
        return await self._t.request(
            "GET", f"{_BASE}/calendars/{self._cal}/events/{event_id}"
        )

    async def patch_event(
        self,
        event_id: str,
        *,
        start_iso: str | None = None,
        end_iso: str | None = None,
        time_zone: str = "UTC",
        summary: str | None = None,
        description: str | None = None,
        location: str | None = None,
        send_updates: str = "all",
    ) -> dict[str, Any]:
        """Partial-update an existing event (the reschedule path).

        PATCH, not PUT: only the supplied fields change, so a reschedule keeps
        the attendee list, conference data, and any edits the user made in
        Google's own UI. Naturally idempotent — applying the same new start/end
        twice converges on the same event.
        """
        body: dict[str, Any] = {}
        if start_iso:
            body["start"] = {"dateTime": start_iso, "timeZone": time_zone}
        if end_iso:
            body["end"] = {"dateTime": end_iso, "timeZone": time_zone}
        if summary is not None:
            body["summary"] = summary
        if description is not None:
            body["description"] = description
        if location is not None:
            body["location"] = location
        if not body:
            raise ValueError("patch_event called with no fields to change")
        return await self._t.request(
            "PATCH",
            f"{_BASE}/calendars/{self._cal}/events/{event_id}",
            params={"sendUpdates": send_updates},
            json=body,
        )

    async def delete_event(self, event_id: str, *, send_updates: str = "all") -> None:
        """Cancel an event. A 404/410 means it is ALREADY gone — the caller's
        desired end state — so both are swallowed here rather than surfaced as
        an execution failure. Every other status still raises."""
        from navis.signals.google.transport import GoogleAPIError

        try:
            await self._t.request(
                "DELETE",
                f"{_BASE}/calendars/{self._cal}/events/{event_id}",
                params={"sendUpdates": send_updates},
            )
        except GoogleAPIError as exc:
            if exc.is_not_found or exc.is_gone:
                return
            raise
