"""Gmail API client over GoogleTransport. Token-in (via transport), JSON-out.

Covers exactly what the connector needs: profile (for the starting historyId),
paginated history.list, message listing, batched message fetch, and watch
create/stop. No business logic — this is a thin, retried REST wrapper.
"""
from __future__ import annotations

import asyncio
from typing import Any

from navis.platform.monitoring import get_logger
from navis.platform.monitoring.metrics import metrics
from navis.signals.google.transport import GoogleAPIError, GoogleTransport

logger = get_logger("signals.gmail.client")

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
        self,
        page_token: str | None = None,
        max_results: int = 100,
        query: str | None = None,
    ) -> dict[str, Any]:
        # `query` accepts Gmail search syntax (newer_than:30d, after:YYYY/MM/DD,
        # from:@example.com, ...). Used by full_sync to bound the onboarding
        # backfill by date so we don't ingest months of historical mail.
        params: dict[str, Any] = {"maxResults": max_results}
        if page_token:
            params["pageToken"] = page_token
        if query:
            params["q"] = query
        return await self._t.request("GET", f"{_BASE}/messages", params=params)

    async def get_message(self, message_id: str) -> dict[str, Any]:
        return await self._t.request(
            "GET", f"{_BASE}/messages/{message_id}", params={"format": "full"}
        )

    async def get_messages(self, message_ids: list[str]) -> list[dict[str, Any]]:
        """Fetch many messages with bounded concurrency.

        A message can be deleted between history.list and message.get; that 404 is
        terminal for THAT message and is skipped so it can't sink the whole batch and
        wedge the cursor (the pass would retry the same dead id forever). Any OTHER error
        (transient 5xx, 429, auth) is re-raised so the pass fails + retries with no silent
        loss — the cursor only advances when every live message was actually captured."""
        sem = asyncio.Semaphore(_FETCH_CONCURRENCY)

        async def _one(mid: str) -> dict[str, Any] | None:
            async with sem:
                try:
                    return await self.get_message(mid)
                except GoogleAPIError as exc:
                    if exc.is_not_found:
                        metrics.incr("gmail.fetch.message_gone")
                        logger.info("gmail message %s gone before fetch; skipping", mid)
                        return None
                    raise

        results = await asyncio.gather(*(_one(mid) for mid in message_ids))
        return [r for r in results if r is not None]

    async def watch(self, topic_name: str, label_ids: list[str] | None = None) -> dict[str, Any]:
        body: dict[str, Any] = {"topicName": topic_name}
        if label_ids:
            body["labelIds"] = label_ids
        return await self._t.request("POST", f"{_BASE}/watch", json=body)

    async def stop_watch(self) -> dict[str, Any]:
        return await self._t.request("POST", f"{_BASE}/stop")

    async def send_message(
        self,
        *,
        raw_rfc822_base64url: str,
        thread_id: str | None = None,
    ) -> dict[str, Any]:
        """Send an outbound message via users.messages.send.

        `raw_rfc822_base64url` is a base64url-encoded RFC 2822 message (headers +
        body). Callers build the MIME (see workspace_router._encode_message) and
        pass it in; this method is a pure transport wrapper so quota + retry come
        from GoogleTransport rather than re-implemented here.

        When `thread_id` is provided, Gmail attaches the outbound message to that
        existing thread — used for reply flows so the recipient sees a normal
        reply instead of a brand-new conversation.
        """
        body: dict[str, Any] = {"raw": raw_rfc822_base64url}
        if thread_id:
            body["threadId"] = thread_id
        return await self._t.request("POST", f"{_BASE}/messages/send", json=body)
