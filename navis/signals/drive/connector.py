"""DriveConnector — production capture for Google Drive (incl. Shared Drives).

    changes.watch -> changes.list(pageToken) -> SignalEvents

Each change yields a Drive signal (created/updated/trashed) and, when the file is a Google
Doc, ALSO a derived Docs signal from the same pass (see drive/docs.py). Version-aware via
the Drive `version` field (a monotonic counter — collision-free, unlike a timestamp), so
successive edits are distinct facts and never dedupe-collide.

Ownership/parent state is carried (data-minimized: owner EMAILS only). "Was shared"/"was
moved" are diffs across versions and belong downstream, not in the capture layer.

Pagination follows Google's change-feed contract: page through nextPageToken until the
FINAL page yields newStartPageToken (intermediate pages carry none), so the drain MUST
complete to obtain a valid cursor. A safety cap turns a pathological feed into a LOUD,
retryable failure rather than silently dropping changes and stalling the cursor.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
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
from navis.signals.drive.client import DriveClient
from navis.signals.drive.docs import maybe_doc_event

logger = get_logger("signals.drive")


class DriveSyncIncomplete(Exception):
    """A change drain hit the page-safety cap before reaching newStartPageToken. Raised
    (not swallowed) so the sync fails loud + retryable instead of stalling the cursor."""


class DriveConnector(WorkspaceConnector):
    source = SignalSource.DRIVE

    async def full_sync(self, ctx: SignalContext, client: DriveClient) -> SyncResult:
        # Establish the change cursor FIRST so nothing is missed during backfill.
        start = await client.get_start_page_token()
        start_token = str(start.get("startPageToken"))

        cap = max(1, settings.drive_onboard_file_cap)
        events: list[SignalEvent] = []
        page_token: str | None = None
        files_seen = 0
        started = time.perf_counter()
        while files_seen < cap:
            resp = await client.list_files(page_token=page_token)
            for file_raw in resp.get("files", []):
                events.extend(self._events_for_file(ctx, file_raw, SignalEventType.DRIVE_FILE_CREATED, capture_mode="full"))
                files_seen += 1
                if files_seen >= cap:
                    break
            page_token = resp.get("nextPageToken")
            if not page_token:
                break

        capped = bool(page_token)  # more existing files remain beyond the backfill cap
        if capped:
            # NOT silent: onboarding backfill is bounded; the rest is captured go-forward by
            # the change feed (cursor established above). Operators see this + can re-run a
            # wider backfill if a full historical sweep is required.
            metrics.incr("drive.full_sync.capped")
            logger.warning(
                "drive full_sync capped at %d files org=%s; remainder captured incrementally",
                cap, ctx.org_id,
            )
        metrics.observe_ms("drive.sync.duration", (time.perf_counter() - started) * 1000.0)
        metrics.incr("drive.sync.events", value=len(events))
        logger.info(
            "drive full_sync org=%s files=%d events=%d capped=%s", ctx.org_id, files_seen, len(events), capped
        )
        return SyncResult(events=events, cursor={"page_token": start_token})

    async def incremental_sync(
        self, ctx: SignalContext, client: DriveClient, cursor: dict[str, Any]
    ) -> SyncResult:
        page_token = cursor.get("page_token")
        if not page_token:
            return SyncResult(full_resync_required=True)

        events: list[SignalEvent] = []
        new_start_token: str | None = None
        pages = 0
        max_pages = max(1, settings.drive_max_sync_pages)
        started = time.perf_counter()
        while True:
            resp = await client.list_changes(page_token)
            for change in resp.get("changes", []):
                events.extend(self._events_for_change(ctx, change))
            pages += 1
            new_start_token = resp.get("newStartPageToken") or new_start_token
            next_page = resp.get("nextPageToken")
            if not next_page:
                break  # final page -> newStartPageToken is now established
            if pages >= max_pages:
                metrics.incr("drive.sync.truncated")
                logger.error(
                    "drive change drain exceeded %d pages without newStartPageToken org=%s",
                    max_pages, ctx.org_id,
                )
                raise DriveSyncIncomplete(
                    f"drive sync exceeded {max_pages} pages without newStartPageToken"
                )
            page_token = next_page

        metrics.observe_ms("drive.sync.duration", (time.perf_counter() - started) * 1000.0)
        metrics.incr("drive.sync.pages", value=pages)
        metrics.incr("drive.sync.events", value=len(events))
        logger.info("drive incremental org=%s pages=%d captured=%d", ctx.org_id, pages, len(events))
        return SyncResult(events=events, cursor={"page_token": new_start_token or cursor.get("page_token")})

    # ---- mapping ----

    def _events_for_change(self, ctx: SignalContext, change: dict[str, Any]) -> list[SignalEvent]:
        file_raw = change.get("file")
        if change.get("removed") or (file_raw and file_raw.get("trashed")):
            if not file_raw:
                return []  # bare removal with no file metadata; nothing to capture
            return self._events_for_file(ctx, file_raw, SignalEventType.DRIVE_FILE_TRASHED, capture_mode="incremental")
        if not file_raw:
            return []
        event_type = (
            SignalEventType.DRIVE_FILE_CREATED
            if file_raw.get("createdTime") == file_raw.get("modifiedTime")
            else SignalEventType.DRIVE_FILE_UPDATED
        )
        return self._events_for_file(ctx, file_raw, event_type, capture_mode="incremental")

    @staticmethod
    def _version(file_raw: dict[str, Any]) -> str:
        # Drive `version` is a monotonic counter incremented on EVERY server-side change ->
        # collision-free. A modifiedTime can repeat for two edits in the same millisecond,
        # which would dedupe-collapse (lose) the second edit. Fall back only if absent.
        v = file_raw.get("version")
        if v is not None:
            return str(v)
        return str(file_raw.get("modifiedTime") or file_raw.get("createdTime") or "1")

    def _events_for_file(
        self,
        ctx: SignalContext,
        file_raw: dict[str, Any],
        event_type: SignalEventType,
        *,
        capture_mode: str,
    ) -> list[SignalEvent]:
        version = self._version(file_raw)
        when = datetime.now(timezone.utc)
        if file_raw.get("modifiedTime"):
            try:
                when = datetime.fromisoformat(file_raw["modifiedTime"].replace("Z", "+00:00"))
            except (TypeError, ValueError):
                pass
        drive_id = file_raw.get("driveId")
        # Minimized: owner EMAILS only (identity for future ontology); no displayName.
        owners = [{"email": o.get("emailAddress")} for o in file_raw.get("owners", []) if o.get("emailAddress")]

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
                "owners": owners,
                "shared": file_raw.get("shared"),
                "trashed": file_raw.get("trashed"),
                "drive_id": drive_id,  # shared-drive lineage (My Drive = null)
                # web_view_link DROPPED — sensitive URL, no consumer, re-fetchable by id.
            },
            metadata={
                "shared": bool(file_raw.get("shared")),
                "drive_id": drive_id,
                "capture_mode": capture_mode,
            },
        )
        out = [drive_event]
        # Docs ride on the Drive feed — never polled independently.
        doc_event = maybe_doc_event(ctx, file_raw, event_type, version, when, capture_mode=capture_mode)
        if doc_event is not None:
            out.append(doc_event)
        return out
