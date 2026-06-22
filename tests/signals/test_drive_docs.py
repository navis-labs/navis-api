"""Drive connector: pageToken advance, event mapping, and Docs derivation."""
from __future__ import annotations

from navis.signals.base import SignalEventType, SignalSource
from navis.signals.drive.connector import DriveConnector
from navis.signals.drive.docs import GOOGLE_DOC_MIME
from tests.signals.fakes import FakeDriveClient, make_ctx


def _file(fid, *, mime="application/pdf", created="2026-01-01T00:00:00Z", modified=None, trashed=False):
    return {
        "id": fid,
        "name": f"file-{fid}",
        "mimeType": mime,
        "parents": ["root"],
        "createdTime": created,
        "modifiedTime": modified or created,
        "trashed": trashed,
        "owners": [{"emailAddress": "owner@acme.com"}],
        "shared": False,
        "webViewLink": "http://x",
    }


async def test_incremental_maps_changes_and_advances_cursor():
    pages = {
        "P100": {
            "changes": [
                {"changeType": "file", "removed": False, "fileId": "f1", "file": _file("f1")},  # created
                {
                    "changeType": "file",
                    "removed": False,
                    "fileId": "f2",
                    "file": _file("f2", modified="2026-02-01T00:00:00Z"),  # updated
                },
                {"changeType": "file", "removed": True, "fileId": "f3", "file": _file("f3", trashed=True)},
            ],
            "newStartPageToken": "P200",
        }
    }
    client = FakeDriveClient(change_pages=pages)
    res = await DriveConnector().incremental_sync(make_ctx(), client, {"page_token": "P100"})

    by_id = {(e.source, e.external_id): e for e in res.events}
    assert by_id[(SignalSource.DRIVE, "f1")].event_type == SignalEventType.DRIVE_FILE_CREATED
    assert by_id[(SignalSource.DRIVE, "f2")].event_type == SignalEventType.DRIVE_FILE_UPDATED
    assert by_id[(SignalSource.DRIVE, "f3")].event_type == SignalEventType.DRIVE_FILE_TRASHED
    assert res.cursor == {"page_token": "P200"}


async def test_google_doc_emits_derived_docs_signal():
    pages = {
        "P100": {
            "changes": [
                {
                    "changeType": "file",
                    "removed": False,
                    "fileId": "d1",
                    "file": _file("d1", mime=GOOGLE_DOC_MIME),  # created google doc
                }
            ],
            "newStartPageToken": "P200",
        }
    }
    res = await DriveConnector().incremental_sync(
        make_ctx(), FakeDriveClient(change_pages=pages), {"page_token": "P100"}
    )
    sources = {(e.source, e.event_type) for e in res.events}
    assert (SignalSource.DRIVE, SignalEventType.DRIVE_FILE_CREATED) in sources
    assert (SignalSource.DOCS, SignalEventType.DOC_CREATED) in sources  # derived, same pass


async def test_incremental_without_cursor_requests_full_resync():
    res = await DriveConnector().incremental_sync(make_ctx(), FakeDriveClient(), {})
    assert res.full_resync_required is True


# ---- DRIVE-001: multi-page change drain + page-cap safety (never silently drop) ----
import pytest  # noqa: E402
from navis.signals.drive.connector import DriveConnector as _DC, DriveSyncIncomplete  # noqa: E402


async def test_multipage_change_drain_completes_and_advances_cursor():
    pages = {
        "P100": {"changes": [{"changeType": "file", "removed": False, "fileId": "f1", "file": _file("f1")}],
                 "nextPageToken": "P101"},  # intermediate page (NO newStartPageToken)
        "P101": {"changes": [{"changeType": "file", "removed": False, "fileId": "f2",
                              "file": _file("f2", modified="2026-02-01T00:00:00Z")}],
                 "newStartPageToken": "P200"},  # final page yields the token
    }
    res = await DriveConnector().incremental_sync(make_ctx(), FakeDriveClient(change_pages=pages), {"page_token": "P100"})
    drive_ids = {e.external_id for e in res.events if e.source == SignalSource.DRIVE}
    assert drive_ids == {"f1", "f2"}              # BOTH pages captured
    assert res.cursor == {"page_token": "P200"}   # newStartPageToken from the FINAL page


async def test_change_drain_cap_fails_loud(monkeypatch):
    from navis.platform.config import settings

    monkeypatch.setattr(settings, "drive_max_sync_pages", 1)
    pages = {
        "P100": {"changes": [], "nextPageToken": "P101"},   # more pages remain past the cap
        "P101": {"changes": [], "newStartPageToken": "P200"},
    }
    with pytest.raises(DriveSyncIncomplete):  # loud + retryable, never a stalled cursor
        await DriveConnector().incremental_sync(make_ctx(), FakeDriveClient(change_pages=pages), {"page_token": "P100"})


# ---- DRIVE-004: monotonic version (not collidable timestamp) ----

def test_version_uses_monotonic_drive_version():
    raw = {"id": "f", "mimeType": "application/pdf", "parents": ["root"],
           "createdTime": "2026-01-01T00:00:00Z", "modifiedTime": "2026-01-01T00:00:00Z", "version": "42"}
    ev = _DC()._events_for_file(make_ctx(), raw, SignalEventType.DRIVE_FILE_UPDATED, capture_mode="incremental")[0]
    assert ev.external_version == "42"  # the monotonic counter, NOT the timestamp


# ---- DRIVE-003/005/006: minimization + shared-drive lineage ----

def test_payload_minimized_owners_dropped_url_and_lineage():
    raw = {"id": "f", "name": "Q3 plan", "mimeType": "application/pdf", "parents": ["root"],
           "createdTime": "2026-01-01T00:00:00Z", "modifiedTime": "2026-01-01T00:00:00Z",
           "owners": [{"emailAddress": "o@acme.com", "displayName": "Owner Name"}],
           "shared": True, "webViewLink": "https://drive/secret", "driveId": "team-drive-1", "version": "5"}
    ev = _DC()._events_for_file(make_ctx(), raw, SignalEventType.DRIVE_FILE_UPDATED, capture_mode="full")[0]
    p = ev.payload
    assert p["owners"] == [{"email": "o@acme.com"}]          # email only, no displayName (PII)
    assert "web_view_link" not in p and "secret" not in str(p)  # sensitive URL dropped
    assert p["drive_id"] == "team-drive-1"                    # shared-drive lineage
    assert ev.metadata["drive_id"] == "team-drive-1" and ev.metadata["capture_mode"] == "full"
    assert p["parents"] == ["root"] and p["name"] == "Q3 plan"  # downstream contract preserved
