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
