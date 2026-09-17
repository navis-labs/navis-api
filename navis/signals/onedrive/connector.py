"""OneDriveConnector — production capture for OneDrive file activity.

    /me/drive/root/delta -> pages -> SignalEvents -> deltaLink

Emits the SAME DRIVE_FILE_* event types the Google Drive connector emits, for
the reason given in the Outlook mail connector: the source names the vendor, the
event type names the verb, and consumers branch on the verb.

Payload is metadata only — name, type, ownership, sharing state, lineage. Never
content, and never a download URL. The Google Drive connector drops
`webViewLink` on the same grounds: a link in an append-only log outlives every
decision anyone made about who should be able to follow it.

Folders are captured alongside files rather than filtered out, because a folder
becoming shared is exactly the kind of ownership change the product is watching
for, and it arrives as a folder item with no `file` facet.
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
from navis.signals.microsoft.delta import drain_delta
from navis.signals.microsoft.graph_time import parse_graph_datetime
from navis.signals.onedrive.client import ITEM_FIELDS, OneDriveClient

logger = get_logger("signals.onedrive")


class OneDriveConnector(WorkspaceConnector):
    source = SignalSource.ONEDRIVE

    async def full_sync(self, ctx: SignalContext, client: OneDriveClient) -> SyncResult:
        started = time.perf_counter()
        page = await drain_delta(
            client.transport,
            "/me/drive/root/delta",
            params={"$select": ITEM_FIELDS},
            resource="onedrive",
            item_cap=max(1, settings.microsoft_onboard_item_cap),
        )
        if page.full_resync_required:
            raise RuntimeError("onedrive full_sync received an expired delta token")

        events = self._events(ctx, page.items, capture_mode="full")
        metrics.observe_ms("onedrive.sync.duration", (time.perf_counter() - started) * 1000.0)
        metrics.incr("onedrive.sync.events", value=len(events))
        logger.info(
            "onedrive full_sync org=%s items=%d events=%d pages=%d",
            ctx.org_id, len(page.items), len(events), page.pages,
        )
        return SyncResult(events=events, cursor={"delta_link": page.delta_link})

    async def incremental_sync(
        self, ctx: SignalContext, client: OneDriveClient, cursor: dict[str, Any]
    ) -> SyncResult:
        delta_link = cursor.get("delta_link")
        if not delta_link:
            return SyncResult(full_resync_required=True)

        started = time.perf_counter()
        page = await drain_delta(client.transport, delta_link, resource="onedrive")
        if page.full_resync_required:
            return SyncResult(full_resync_required=True)

        events = self._events(ctx, page.items, capture_mode="incremental")
        metrics.observe_ms("onedrive.sync.duration", (time.perf_counter() - started) * 1000.0)
        metrics.incr("onedrive.sync.events", value=len(events))
        logger.info(
            "onedrive incremental org=%s pages=%d captured=%d",
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
    def _version(raw: dict[str, Any]) -> str:
        """A per-item version token that changes on every server-side change.

        `cTag` changes when CONTENT changes; `eTag` changes on content OR
        metadata changes, which is the wider and therefore correct choice — a
        file becoming shared is a fact this product cares about and does not
        touch content at all. Falling back to a timestamp is a last resort: two
        edits inside the same second would collide on the dedupe key and the
        second would be silently lost.
        """
        for key in ("eTag", "cTag"):
            value = raw.get(key)
            if value:
                return str(value)
        return str(raw.get("lastModifiedDateTime") or raw.get("createdDateTime") or "1")

    @staticmethod
    def _identity_email(container: Any) -> str | None:
        """Pull the email out of Graph's `identitySet` shape.

        `{user: {displayName, email, id}}`. Only the email is kept — the
        display name is bulk PII with no consumer, exactly as the Google Drive
        connector keeps owner emails and drops displayName.
        """
        if not isinstance(container, dict):
            return None
        user = container.get("user")
        if not isinstance(user, dict):
            return None
        email = user.get("email") or user.get("userPrincipalName")
        return email.lower() if isinstance(email, str) else None

    def _event_type(self, raw: dict[str, Any]) -> SignalEventType:
        if raw.get("deleted"):
            return SignalEventType.DRIVE_FILE_TRASHED
        created = raw.get("createdDateTime")
        modified = raw.get("lastModifiedDateTime")
        if created and modified and created == modified:
            return SignalEventType.DRIVE_FILE_CREATED
        return SignalEventType.DRIVE_FILE_UPDATED

    def _to_event(
        self, ctx: SignalContext, raw: dict[str, Any], *, capture_mode: str
    ) -> SignalEvent | None:
        item_id = raw.get("id")
        if not item_id:
            return None
        # The delta feed's first entry for a drive is the root itself, which is
        # not a document and produces a meaningless "file updated" fact.
        if raw.get("root") is not None:
            return None

        parent = raw.get("parentReference") or {}
        when = (
            parse_graph_datetime(raw.get("lastModifiedDateTime"))
            or parse_graph_datetime(raw.get("createdDateTime"))
            or datetime.now(timezone.utc)
        )
        owner_email = self._identity_email(raw.get("createdBy"))
        modifier_email = self._identity_email(raw.get("lastModifiedBy"))
        is_shared = raw.get("shared") is not None

        return SignalEvent.create(
            ctx,
            source=SignalSource.ONEDRIVE,
            event_type=self._event_type(raw),
            external_id=str(item_id),
            external_version=self._version(raw),
            timestamp=when,
            payload={
                # Keyed to match the Google Drive payload so one reader handles
                # both. Graph splits type across `file`/`folder` facets rather
                # than carrying a single mimeType, so it is reassembled here.
                "name": raw.get("name"),
                "mime_type": (raw.get("file") or {}).get("mimeType"),
                "is_folder": raw.get("folder") is not None,
                "parents": [parent.get("id")] if parent.get("id") else [],
                "owners": [{"email": owner_email}] if owner_email else [],
                "last_modified_by": modifier_email,
                "shared": is_shared,
                "trashed": raw.get("deleted") is not None,
                "size": raw.get("size"),
                # Which drive the item lives in — the analogue of Drive's
                # driveId shared-drive lineage.
                "drive_id": parent.get("driveId"),
            },
            metadata={
                "provider": "microsoft",
                "shared": is_shared,
                "drive_id": parent.get("driveId"),
                "capture_mode": capture_mode,
            },
        )
