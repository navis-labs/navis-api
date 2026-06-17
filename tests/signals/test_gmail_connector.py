"""Gmail connector: pagination, batching, cursor advance, 404 recovery, mapping."""
from __future__ import annotations

import base64

from navis.signals.base import SignalEventType, SignalSource
from navis.signals.gmail.connector import GmailConnector
from tests.signals.fakes import FakeGmailClient, make_ctx


def _raw(mid: str, history_id: str) -> dict:
    body = base64.urlsafe_b64encode(b"hello world").decode()
    return {
        "id": mid,
        "threadId": f"t-{mid}",
        "historyId": history_id,
        "internalDate": "1700000000000",
        "snippet": "hello",
        "labelIds": ["INBOX"],
        "payload": {
            "mimeType": "text/plain",
            "headers": [
                {"name": "From", "value": "alice@acme.com"},
                {"name": "Subject", "value": "hi"},
            ],
            "body": {"data": body},
        },
    }


async def test_full_sync_paginates_and_batches():
    msgs = [_raw("m1", "1001"), _raw("m2", "1002"), _raw("m3", "1003")]
    client = FakeGmailClient(profile_history_id="999", messages=msgs, page_size=2)
    res = await GmailConnector().full_sync(make_ctx(), client)

    assert len(res.events) == 3  # all three across 2 pages
    assert res.cursor == {"history_id": "999"}  # cursor seeded from profile
    assert {e.event_type for e in res.events} == {SignalEventType.EMAIL_RECEIVED}
    assert all(e.source == SignalSource.GMAIL for e in res.events)
    # version-aware: each message versioned by its historyId
    assert {e.external_version for e in res.events} == {"1001", "1002", "1003"}


async def test_incremental_advances_history_cursor():
    msgs = [_raw("m2", "1100")]
    history_pages = [
        {
            "history": [{"messagesAdded": [{"message": {"id": "m2"}}]}],
            "historyId": "1100",
        }
    ]
    client = FakeGmailClient(messages=msgs, history_pages=history_pages)
    res = await GmailConnector().incremental_sync(make_ctx(), client, {"history_id": "1000"})

    assert len(res.events) == 1
    assert res.events[0].external_id == "m2"
    assert res.cursor == {"history_id": "1100"}  # cursor advanced


async def test_incremental_404_requests_full_resync():
    client = FakeGmailClient(history_raises_404=True)
    res = await GmailConnector().incremental_sync(make_ctx(), client, {"history_id": "1000"})
    assert res.full_resync_required is True


async def test_incremental_without_cursor_requests_full_resync():
    res = await GmailConnector().incremental_sync(make_ctx(), FakeGmailClient(), {})
    assert res.full_resync_required is True
