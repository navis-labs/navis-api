"""In-memory fakes so the signals domain is fully testable without Google or a DB."""
from __future__ import annotations

import uuid
from types import SimpleNamespace
from typing import Any

from navis.signals.base import SignalContext, SignalEvent
from navis.signals.google.transport import GoogleAPIError


def make_ctx() -> SignalContext:
    return SignalContext(org_id=uuid.uuid4(), user_id=uuid.uuid4(), integration_id=uuid.uuid4())


class FakeIngestion:
    """Mimics SignalIngestionService.ingest with ON CONFLICT semantics."""

    def __init__(self, fail_on: set[str] | None = None) -> None:
        self.seen: set[str] = set()
        self.ingested: list[SignalEvent] = []
        self.fail_on = fail_on or set()  # external_ids that should raise (poison)

    async def ingest(self, event: SignalEvent) -> bool:
        if event.external_id in self.fail_on:
            raise RuntimeError(f"poison {event.external_id}")
        if event.dedupe_key in self.seen:
            return False
        self.seen.add(event.dedupe_key)
        self.ingested.append(event)
        return True

    async def ingest_many(self, events: list[SignalEvent]) -> tuple[int, int]:
        # Mirror an atomic batch insert: a poison row aborts the WHOLE batch (nothing
        # persisted) so the sync engine falls back to per-event isolation.
        if any(e.external_id in self.fail_on for e in events):
            raise RuntimeError("poison in batch")
        inserted = duplicates = 0
        batch_seen: set[str] = set()
        for e in events:
            if e.dedupe_key in self.seen or e.dedupe_key in batch_seen:
                duplicates += 1
                continue
            batch_seen.add(e.dedupe_key)
            self.seen.add(e.dedupe_key)
            self.ingested.append(e)
            inserted += 1
        return inserted, duplicates


class FakeDLQ:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    async def send(self, **kwargs: Any) -> None:
        self.records.append(kwargs)


class FakeGmailClient:
    def __init__(
        self,
        *,
        profile_history_id: str = "1000",
        messages: list[dict] | None = None,
        history_pages: list[dict] | None = None,
        history_raises_404: bool = False,
        page_size: int = 2,
    ) -> None:
        self._profile_history_id = profile_history_id
        self._messages = messages or []
        self._by_id = {m["id"]: m for m in self._messages}
        self._history_pages = history_pages or []
        self._history_raises_404 = history_raises_404
        self._page_size = page_size

    async def get_profile(self) -> dict:
        return {"historyId": self._profile_history_id, "emailAddress": "u@org.com"}

    async def list_messages(self, page_token: str | None = None, max_results: int = 100) -> dict:
        start = int(page_token or 0)
        ids = [{"id": m["id"]} for m in self._messages[start : start + self._page_size]]
        nxt = start + self._page_size
        out: dict[str, Any] = {"messages": ids}
        if nxt < len(self._messages):
            out["nextPageToken"] = str(nxt)
        return out

    async def get_messages(self, message_ids: list[str]) -> list[dict]:
        return [self._by_id[i] for i in message_ids if i in self._by_id]

    async def list_history(self, start_history_id: str, page_token: str | None = None) -> dict:
        if self._history_raises_404:
            raise GoogleAPIError(404, "historyId not found")
        idx = int(page_token or 0)
        if idx >= len(self._history_pages):
            return {"history": [], "historyId": start_history_id}
        return self._history_pages[idx]

    async def watch(self, topic_name: str, label_ids: list[str] | None = None) -> dict:
        return {"historyId": "2000", "expiration": "1700000000000"}


class FakeCalendarClient:
    def __init__(self, *, pages: list[dict] | None = None, raises_410: bool = False) -> None:
        self._pages = pages or []
        self._raises_410 = raises_410

    async def list_events(
        self,
        *,
        sync_token: str | None = None,
        page_token: str | None = None,
        time_min: str | None = None,
        max_results: int = 250,
    ) -> dict:
        if self._raises_410 and sync_token:
            raise GoogleAPIError(410, "sync token gone")
        idx = int(page_token or 0)
        if idx >= len(self._pages):
            return {"items": [], "nextSyncToken": "SYNC-NEXT"}
        return self._pages[idx]


class FakeDriveClient:
    def __init__(
        self,
        *,
        start_token: str = "P100",
        files: list[dict] | None = None,
        change_pages: dict[str, dict] | None = None,
    ) -> None:
        self._start_token = start_token
        self._files = files or []
        # token -> page response. Each page may carry nextPageToken/newStartPageToken.
        self._change_pages = change_pages or {}

    async def get_start_page_token(self) -> dict:
        return {"startPageToken": self._start_token}

    async def list_files(self, page_token: str | None = None, page_size: int = 100) -> dict:
        return {"files": self._files}

    async def list_changes(self, page_token: str, *, page_size: int = 100) -> dict:
        return self._change_pages.get(page_token, {"changes": [], "newStartPageToken": "P200"})


def fake_integration(**overrides: Any) -> SimpleNamespace:
    base = dict(
        id=__import__("uuid").uuid4(),
        access_token="tok",
        refresh_token="refresh",
        token_expires_at=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)
