"""Docs signals, DERIVED from the Drive change feed. We never poll the Docs API.

A Google Doc is just a Drive file with mimeType
``application/vnd.google-apps.document``. When the Drive connector sees a change
to such a file it ALSO emits a Docs-scoped signal (DOC_CREATED/UPDATED/MOVED),
so Docs intelligence rides for free on the Drive feed.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from navis.signals.base import (
    SignalContext,
    SignalEvent,
    SignalEventType,
    SignalSource,
)

GOOGLE_DOC_MIME = "application/vnd.google-apps.document"

# Map a Drive event type to its Docs equivalent (None = not represented as a Doc signal).
_DRIVE_TO_DOC = {
    SignalEventType.DRIVE_FILE_CREATED: SignalEventType.DOC_CREATED,
    SignalEventType.DRIVE_FILE_UPDATED: SignalEventType.DOC_UPDATED,
    SignalEventType.DRIVE_FILE_MOVED: SignalEventType.DOC_MOVED,
}


def is_google_doc(file_raw: dict[str, Any] | None) -> bool:
    return bool(file_raw) and file_raw.get("mimeType") == GOOGLE_DOC_MIME


def maybe_doc_event(
    ctx: SignalContext,
    file_raw: dict[str, Any],
    drive_event_type: SignalEventType,
    external_version: str,
    timestamp: datetime | None = None,
    *,
    capture_mode: str = "incremental",
) -> SignalEvent | None:
    """Return a Docs SignalEvent if the changed file is a Google Doc, else None.

    A parallel Docs-scoped fact (source=DOCS, so its dedupe key differs from the Drive
    event's — no duplicate reality, no inflation). Payload is data-minimized: owner EMAILS
    only (no displayName), no web_view_link (sensitive URL, re-fetchable by id)."""
    if not is_google_doc(file_raw):
        return None
    doc_type = _DRIVE_TO_DOC.get(drive_event_type)
    if doc_type is None:
        return None
    owners = [{"email": o.get("emailAddress")} for o in file_raw.get("owners", []) if o.get("emailAddress")]
    return SignalEvent.create(
        ctx,
        source=SignalSource.DOCS,
        event_type=doc_type,
        external_id=file_raw["id"],
        external_version=external_version,
        timestamp=timestamp or datetime.now(timezone.utc),
        payload={
            "name": file_raw.get("name"),
            "owners": owners,
            "parents": file_raw.get("parents", []),
            "drive_id": file_raw.get("driveId"),
        },
        metadata={
            "derived_from": "drive",
            "drive_id": file_raw.get("driveId"),
            "capture_mode": capture_mode,
        },
    )
