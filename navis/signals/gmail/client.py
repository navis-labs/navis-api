"""Gmail API client over GoogleTransport. Token-in (via transport), JSON-out.

Covers exactly what the connector needs: profile (for the starting historyId),
paginated history.list, message listing, batched message fetch, and watch
create/stop. No business logic — this is a thin, retried REST wrapper.
"""
from __future__ import annotations

import asyncio
from typing import Any

from navis.signals.google.transport import GoogleTransport

_BASE = "https://gmail.googleapis.com/gmail/v1/users/me"

# Bounded concurrency for fan-out message fetches (politeness + quota safety).
_FETCH_CONCURRENCY = 10


class GmailClient:
    def __init__(self, transport: GoogleTransport) -> None:
        self._t = transport

    async def get_profile(self) -> dict[str, Any]:
        return await self._t.request("GET", f"{_BASE}/profile")

    async def list_history(
        self, start_history_id: str, page_token: str | None = None
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "startHistoryId": start_history_id,
            "historyTypes": "messageAdded",
        }
        if page_token:
            params["pageToken"] = page_token
        return await self._t.request("GET", f"{_BASE}/history", params=params)

    async def list_messages(
        self, page_token: str | None = None, max_results: int = 100
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"maxResults": max_results}
        if page_token:
            params["pageToken"] = page_token
        return await self._t.request("GET", f"{_BASE}/messages", params=params)

    async def get_message(self, message_id: str) -> dict[str, Any]:
        return await self._t.request(
            "GET", f"{_BASE}/messages/{message_id}", params={"format": "full"}
        )

    async def get_messages(self, message_ids: list[str]) -> list[dict[str, Any]]:
        """Fetch many messages with bounded concurrency."""
        sem = asyncio.Semaphore(_FETCH_CONCURRENCY)

        async def _one(mid: str) -> dict[str, Any]:
            async with sem:
                return await self.get_message(mid)

        return await asyncio.gather(*(_one(mid) for mid in message_ids))

    async def watch(self, topic_name: str, label_ids: list[str] | None = None) -> dict[str, Any]:
        body: dict[str, Any] = {"topicName": topic_name}
        if label_ids:
            body["labelIds"] = label_ids
        return await self._t.request("POST", f"{_BASE}/watch", json=body)

    async def stop_watch(self) -> dict[str, Any]:
        return await self._t.request("POST", f"{_BASE}/stop")
