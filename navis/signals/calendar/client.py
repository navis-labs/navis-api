"""Google Calendar API client over GoogleTransport.

events.list supports incremental sync via syncToken. On a 410 GONE the token is
invalid and a full resync (timeMin window) is required — the connector handles
that; this client just surfaces the error from the transport.
"""
from __future__ import annotations

from typing import Any

from navis.signals.google.transport import GoogleTransport

_BASE = "https://www.googleapis.com/calendar/v3"


class CalendarClient:
    def __init__(self, transport: GoogleTransport, *, calendar_id: str = "primary") -> None:
        self._t = transport
        self._cal = calendar_id

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
