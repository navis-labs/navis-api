"""OutlookCalendarClient — thin Graph wrapper for calendar capture."""
from __future__ import annotations

from typing import Any

# Everything the connector maps, and nothing else. Notably ABSENT: `body`
# (the meeting description — free text that frequently carries dial-in
# credentials and candidate names), and `attendees[].emailAddress.name`
# beyond what the minimiser keeps. The Google calendar connector minimises the
# same way; capture is deliberately identity-and-timing, not content.
EVENT_FIELDS = ",".join((
    "id",
    "changeKey",
    "subject",
    "start",
    "end",
    "isAllDay",
    "isCancelled",
    "showAs",
    "organizer",
    "attendees",
    "seriesMasterId",
    "type",
    "recurrence",
    "onlineMeeting",
    "onlineMeetingProvider",
    "isOnlineMeeting",
    "webLink",
    "createdDateTime",
    "lastModifiedDateTime",
))


class OutlookCalendarClient:
    def __init__(self, transport) -> None:
        self._transport = transport

    @property
    def transport(self):
        """The underlying GraphTransport — see OutlookClient.transport."""
        return self._transport

    async def calendar_view_delta(
        self,
        *,
        url: str | None = None,
        start_iso: str | None = None,
        end_iso: str | None = None,
    ) -> dict[str, Any]:
        """One page of the calendarView delta enumeration.

        `calendarView` rather than `events` on purpose. The `events` collection
        returns a recurring series as a single master with a recurrence rule and
        leaves expansion to the caller; `calendarView` returns the actual
        OCCURRENCES inside a window, already expanded. Since the product reasons
        about "what is on your calendar on Thursday", expanded occurrences are
        the facts it needs — reimplementing RRULE expansion to get them from
        `events` would be a meaningful source of wrong meeting times.

        The cost is that the window is mandatory and finite, which is why
        `microsoft_calendar_window_days` exists and why there is no Google twin
        for that setting.
        """
        if url:
            return await self._transport.request("GET", url)
        params: dict[str, Any] = {"$select": EVENT_FIELDS}
        if start_iso:
            params["startDateTime"] = start_iso
        if end_iso:
            params["endDateTime"] = end_iso
        return await self._transport.request(
            "GET", "/me/calendarView/delta", params=params
        )
