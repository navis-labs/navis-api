"""DriveConnector — production capture for Google Drive.

    changes.watch -> changes.list(pageToken) -> SignalEvents

Each change yields a Drive signal (created/updated/trashed) and, when the file
is a Google Doc, ALSO a derived Docs signal from the same pass (see drive/docs.py).
Version-aware via modifiedTime so successive edits aren't deduped.

Ownership/permission/parent state is carried in the payload (reality capture).
Higher-level interpretations like "was shared" or "was moved" are diffs across
versions and belong downstream, not in the capture layer.
"""
from __future__ import annotations

from datetime import datetime, timezone
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
from navis.signals.drive.client import DriveClient
from navis.signals.drive.docs import maybe_doc_event

logger = get_logger("signals.drive")

ONBOARD_FILE_CAP = 500
MAX_PAGES = 100


class DriveConnector(WorkspaceConnector):
    source = SignalSource.DRIVE

    async def full_sync(self, ctx: SignalContext, client: DriveClient) -> SyncResult:
        # Establish the change cursor first so nothing is missed during backfill.
        start = await client.get_start_page_token()
        start_token = str(start.get("startPageToken"))

        events: list[SignalEvent] = []
        page_token: str | None = None
        while len(events) < ONBOARD_FILE_CAP:
            resp = await client.list_files(page_token=page_token)
            for file_raw in resp.get("files", []):
                events.extend(self._events_for_file(ctx, file_raw, SignalEventType.DRIVE_FILE_CREATED))
            page_token = resp.get("nextPageToken")
            if not page_token:
                break
        logger.info("drive full_sync org=%s captured=%d", ctx.org_id, len(events))
        return SyncResult(events=events[: ONBOARD_FILE_CAP * 2], cursor={"page_token": start_token})

    async def incremental_sync(
        self, ctx: SignalContext, client: DriveClient, cursor: dict[str, Any]
    ) -> SyncResult:
        page_token = cursor.get("page_token")
        if not page_token:
            return SyncResult(full_resync_required=True)

        events: list[SignalEvent] = []
        new_start_token: str | None = None
        for _ in range(MAX_PAGES):
            resp = await client.list_changes(page_token)
            for change in resp.get("changes", []):
                events.extend(self._events_for_change(ctx, change))
            new_start_token = resp.get("newStartPageToken") or new_start_token
            page_token = resp.get("nextPageToken")
            if not page_token:
                break

        cursor_out = {"page_token": new_start_token or cursor.get("page_token")}
        logger.info("drive incremental org=%s captured=%d", ctx.org_id, len(events))
        return SyncResult(events=events, cursor=cursor_out)

    # ---- mapping ----

    def _events_for_change(self, ctx: SignalContext, change: dict[str, Any]) -> list[SignalEvent]:
        file_raw = change.get("file")
        if change.get("removed") or (file_raw and file_raw.get("trashed")):
            if not file_raw:
                return []  # bare removal with no file metadata; nothing to capture
            return self._events_for_file(ctx, file_raw, SignalEventType.DRIVE_FILE_TRASHED)
        if not file_raw:
            return []
        event_type = (
            SignalEventType.DRIVE_FILE_CREATED
            if file_raw.get("createdTime") == file_raw.get("modifiedTime")
            else SignalEventType.DRIVE_FILE_UPDATED
        )
        return self._events_for_file(ctx, file_raw, event_type)

    @staticmethod
    def _version(file_raw: dict[str, Any]) -> str:
        return str(file_raw.get("modifiedTime") or file_raw.get("createdTime") or "1")

    def _events_for_file(
        self, ctx: SignalContext, file_raw: dict[str, Any], event_type: SignalEventType
    ) -> list[SignalEvent]:
        version = self._version(file_raw)
        when = datetime.now(timezone.utc)
        if file_raw.get("modifiedTime"):
            try:
                when = datetime.fromisoformat(file_raw["modifiedTime"].replace("Z", "+00:00"))
            except (TypeError, ValueError):
                pass

        drive_event = SignalEvent.create(
            ctx,
            source=SignalSource.DRIVE,
            event_type=event_type,
            external_id=file_raw["id"],
            external_version=version,
            timestamp=when,
            payload={
                "name": file_raw.get("name"),
                "mime_type": file_raw.get("mimeType"),
                "parents": file_raw.get("parents", []),
                "owners": file_raw.get("owners", []),
                "shared": file_raw.get("shared"),
                "trashed": file_raw.get("trashed"),
                "web_view_link": file_raw.get("webViewLink"),
            },
            metadata={"shared": bool(file_raw.get("shared"))},
        )
        out = [drive_event]
        # Docs ride on the Drive feed — never polled independently.
        doc_event = maybe_doc_event(ctx, file_raw, event_type, version, when)
        if doc_event is not None:
            out.append(doc_event)
        return out
