"""Drive connector: pageToken advance, event mapping, and Docs derivation."""
from __future__ import annotations

import inspect

from navis.signals.base import SignalEventType, SignalSource
from navis.signals.drive.client import DriveClient
from navis.signals.drive.connector import DriveConnector
from navis.signals.drive.docs import GOOGLE_DOC_MIME, is_google_doc
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


# ---- Docs MIME discrimination -------------------------------------------
#
# `is_google_doc` is an EXACT match on the Doc MIME. Everything else in the
# `vnd.google-apps.*` family — Sheets, Slides, folders, shortcuts — must stay
# on the Drive path only. A shortcut is the subtle one: it points AT a Doc but
# is not one, so classifying it as a Doc would mint a second Document entity
# for a file that already has its own.

_DOC_LOOKALIKES = {
    "sheet": "application/vnd.google-apps.spreadsheet",
    "slide": "application/vnd.google-apps.presentation",
    "folder": "application/vnd.google-apps.folder",
    "shortcut": "application/vnd.google-apps.shortcut",
    "form": "application/vnd.google-apps.form",
    "drawing": "application/vnd.google-apps.drawing",
    "binary": "application/pdf",
}


@pytest.mark.parametrize("label,mime", sorted(_DOC_LOOKALIKES.items()))
def test_only_the_google_doc_mime_is_treated_as_a_doc(label, mime):
    assert is_google_doc({"mimeType": mime}) is False, label
    assert is_google_doc({"mimeType": GOOGLE_DOC_MIME}) is True


def test_missing_or_null_mime_is_not_a_doc():
    assert is_google_doc({}) is False
    assert is_google_doc({"mimeType": None}) is False
    assert is_google_doc(None) is False


def test_doc_mime_match_is_case_sensitive_and_not_a_prefix_match():
    """Guards against a future loosening to startswith/casefold: Drive's own
    MIME strings are canonical lowercase, so anything else is not a Doc."""
    assert is_google_doc({"mimeType": GOOGLE_DOC_MIME.upper()}) is False
    assert is_google_doc({"mimeType": GOOGLE_DOC_MIME + ".template"}) is False


@pytest.mark.parametrize("mime", sorted(_DOC_LOOKALIKES.values()))
def test_lookalikes_emit_a_drive_signal_but_never_a_docs_signal(mime):
    raw = _file("x1", mime=mime)
    events = _DC()._events_for_file(
        make_ctx(), raw, SignalEventType.DRIVE_FILE_UPDATED, capture_mode="incremental"
    )

    assert [e.source for e in events] == [SignalSource.DRIVE]


# ---- Docs lineage, dedupe and metadata-only contract ---------------------


def test_docs_signal_preserves_file_id_and_stays_metadata_only():
    """The Doc's identity is the Drive file id, and nothing beyond metadata is
    captured — there is no content field because no content is ever fetched."""
    raw = _file("d1", mime=GOOGLE_DOC_MIME)
    raw["driveId"] = "team-drive-1"
    doc = [
        e
        for e in _DC()._events_for_file(
            make_ctx(), raw, SignalEventType.DRIVE_FILE_CREATED, capture_mode="incremental"
        )
        if e.source is SignalSource.DOCS
    ][0]

    assert doc.external_id == "d1"  # the Drive file id IS the doc identity
    assert set(doc.payload) == {"name", "owners", "parents", "drive_id"}
    assert doc.payload["drive_id"] == "team-drive-1"  # shared-drive lineage kept
    assert doc.metadata["derived_from"] == "drive"    # provenance
    assert "web_view_link" not in doc.payload         # sensitive URL still dropped


def test_drive_and_docs_signals_are_distinct_facts_not_duplicates():
    """Same underlying file, two signal sources. The dedupe key includes the
    source, so the pair can never collapse into — or inflate — one another."""
    raw = _file("d1", mime=GOOGLE_DOC_MIME)
    events = _DC()._events_for_file(
        make_ctx(), raw, SignalEventType.DRIVE_FILE_CREATED, capture_mode="incremental"
    )
    drive_ev = next(e for e in events if e.source is SignalSource.DRIVE)
    doc_ev = next(e for e in events if e.source is SignalSource.DOCS)

    assert drive_ev.event_id != doc_ev.event_id
    assert drive_ev.external_id == doc_ev.external_id == "d1"
    assert drive_ev.external_version == doc_ev.external_version  # same edit


def test_docs_signal_versions_by_the_same_monotonic_drive_version():
    """Editing a Doc's CONTENT bumps Drive's `version` even though the metadata
    is unchanged — so successive edits stay distinct facts without ever reading
    the content itself."""
    raw = _file("d1", mime=GOOGLE_DOC_MIME)
    raw["version"] = "7"
    doc = [
        e
        for e in _DC()._events_for_file(
            make_ctx(), raw, SignalEventType.DRIVE_FILE_UPDATED, capture_mode="incremental"
        )
        if e.source is SignalSource.DOCS
    ][0]

    assert doc.external_version == "7"


async def test_trashed_doc_emits_both_drive_and_docs_lifecycle_signals():
    """Trashing follows DRIVE semantics AND gets a Docs-scoped lifecycle fact,
    so the DOCUMENT entity stops reporting a deleted document as live."""
    pages = {
        "P100": {
            "changes": [
                {
                    "changeType": "file",
                    "removed": False,
                    "fileId": "d1",
                    "file": _file("d1", mime=GOOGLE_DOC_MIME, trashed=True),
                }
            ],
            "newStartPageToken": "P200",
        }
    }
    res = await DriveConnector().incremental_sync(
        make_ctx(), FakeDriveClient(change_pages=pages), {"page_token": "P100"}
    )

    assert {(e.source, e.event_type) for e in res.events} == {
        (SignalSource.DRIVE, SignalEventType.DRIVE_FILE_TRASHED),
        (SignalSource.DOCS, SignalEventType.DOC_TRASHED),
    }


def test_docs_signals_are_organization_scoped():
    """The same Google Doc seen by two orgs must never share a signal identity.
    org_id is part of the dedupe key, so the ids diverge by construction."""
    raw = _file("d1", mime=GOOGLE_DOC_MIME)
    a = _DC()._events_for_file(
        make_ctx(), raw, SignalEventType.DRIVE_FILE_CREATED, capture_mode="incremental"
    )
    b = _DC()._events_for_file(
        make_ctx(), raw, SignalEventType.DRIVE_FILE_CREATED, capture_mode="incremental"
    )
    a_doc = next(e for e in a if e.source is SignalSource.DOCS)
    b_doc = next(e for e in b if e.source is SignalSource.DOCS)

    assert a_doc.org_id != b_doc.org_id
    assert a_doc.event_id != b_doc.event_id


def test_drive_client_exposes_no_content_retrieval_surface():
    """Least privilege is structural, not just a scope string: there is no
    export / alt=media / Docs-API call to reach for. Adding one must be a
    deliberate, reviewed change that breaks this test."""
    surface = {m for m in dir(DriveClient) if not m.startswith("_")}

    assert surface == {"get_start_page_token", "list_changes", "list_files", "watch", "stop_channel"}
    source = inspect.getsource(inspect.getmodule(DriveClient))
    for forbidden in ("alt=media", "alt='media'", "/export", "documents.googleapis.com"):
        assert forbidden not in source, forbidden
