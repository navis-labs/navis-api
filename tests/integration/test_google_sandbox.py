"""Live Google Workspace sandbox tests for the connectors.

Run ONLY against a disposable sandbox account by providing a valid OAuth access
token (read-only scopes) via env. Each connector test is independently guarded so
you can run just the surface you have creds for.

    GOOGLE_TEST_ACCESS_TOKEN=ya29.xxx pytest tests/integration/test_google_sandbox.py -q

These prove the real payload shapes (which fakes cannot): pagination, cursor
emission, and that every captured event satisfies the contract + tenancy.
"""
from __future__ import annotations

import os
import uuid

import pytest

_TOKEN = os.environ.get("GOOGLE_TEST_ACCESS_TOKEN")

pytestmark = pytest.mark.skipif(
    not _TOKEN, reason="set GOOGLE_TEST_ACCESS_TOKEN to run live Google sandbox tests"
)


def _ctx():
    from navis.signals.base import SignalContext

    return SignalContext(org_id=uuid.uuid4(), user_id=uuid.uuid4(), integration_id=uuid.uuid4())


def _transport():
    from navis.signals.google.transport import GoogleTransport

    async def token() -> str:
        return _TOKEN

    return GoogleTransport(token)


def _assert_contract(events, ctx):
    for e in events:
        assert e.org_id == ctx.org_id and e.user_id == ctx.user_id
        assert e.external_id and e.external_version  # version-aware


async def test_gmail_full_sync_live():
    from navis.signals.gmail import GmailClient, GmailConnector

    ctx = _ctx()
    res = await GmailConnector().full_sync(ctx, GmailClient(_transport()))
    assert "history_id" in res.cursor  # cursor emitted
    _assert_contract(res.events, ctx)


async def test_calendar_full_sync_live():
    from navis.signals.calendar import CalendarClient, CalendarConnector

    ctx = _ctx()
    res = await CalendarConnector().full_sync(ctx, CalendarClient(_transport()))
    assert "sync_token" in res.cursor
    _assert_contract(res.events, ctx)


async def test_drive_full_sync_live():
    from navis.signals.drive import DriveClient, DriveConnector

    ctx = _ctx()
    res = await DriveConnector().full_sync(ctx, DriveClient(_transport()))
    assert "page_token" in res.cursor
    _assert_contract(res.events, ctx)
