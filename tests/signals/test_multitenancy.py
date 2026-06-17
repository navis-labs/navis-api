"""Multi-tenancy: every captured signal carries org + user + integration ids."""
from __future__ import annotations

import base64

from navis.signals.calendar.connector import CalendarConnector
from navis.signals.drive.connector import DriveConnector
from navis.signals.gmail.connector import GmailConnector
from tests.signals.fakes import FakeCalendarClient, FakeDriveClient, FakeGmailClient, make_ctx


def _gmail_raw(mid):
    return {
        "id": mid,
        "threadId": "t",
        "historyId": "1",
        "internalDate": "1700000000000",
        "payload": {
            "mimeType": "text/plain",
            "headers": [{"name": "From", "value": "a@x.com"}],
            "body": {"data": base64.urlsafe_b64encode(b"hi").decode()},
        },
    }


def _assert_tenancy(events, ctx):
    assert events, "expected at least one event"
    for e in events:
        assert e.org_id == ctx.org_id
        assert e.user_id == ctx.user_id
        assert e.integration_id == ctx.integration_id


async def test_gmail_signals_are_tenant_scoped():
    ctx = make_ctx()
    client = FakeGmailClient(messages=[_gmail_raw("m1")], page_size=5)
    res = await GmailConnector().full_sync(ctx, client)
    _assert_tenancy(res.events, ctx)


async def test_calendar_signals_are_tenant_scoped():
    ctx = make_ctx()
    pages = [
        {
            "items": [
                {
                    "id": "e1",
                    "status": "confirmed",
                    "etag": '"1"',
                    "created": "2026-01-01T00:00:00Z",
                    "updated": "2026-01-01T00:00:00Z",
                    "start": {"dateTime": "2026-02-01T10:00:00Z"},
                    "end": {"dateTime": "2026-02-01T11:00:00Z"},
                }
            ],
            "nextSyncToken": "S",
        }
    ]
    res = await CalendarConnector().full_sync(ctx, FakeCalendarClient(pages=pages))
    _assert_tenancy(res.events, ctx)


async def test_drive_signals_are_tenant_scoped():
    ctx = make_ctx()
    files = [
        {
            "id": "f1",
            "name": "doc",
            "mimeType": "application/pdf",
            "parents": ["root"],
            "createdTime": "2026-01-01T00:00:00Z",
            "modifiedTime": "2026-01-01T00:00:00Z",
            "trashed": False,
            "owners": [],
        }
    ]
    res = await DriveConnector().full_sync(ctx, FakeDriveClient(files=files))
    _assert_tenancy(res.events, ctx)
