"""Gmail connector: pagination, batching, cursor advance, 404 recovery, mapping; parser
header/body capture for ontology + reality completeness."""
from __future__ import annotations

import base64

import pytest

from navis.signals.base import SignalEventType, SignalSource
from navis.signals.gmail.connector import GmailConnector
from navis.signals.gmail.parser import parse_message
from tests.signals.fakes import FakeGmailClient, make_ctx


def _b64(s: str) -> str:
    return base64.urlsafe_b64encode(s.encode()).decode()


def test_parser_captures_reply_graph_headers():
    raw = {
        "id": "m1", "threadId": "t1", "historyId": "5", "internalDate": "1700000000000",
        "payload": {
            "mimeType": "text/plain",
            "headers": [
                {"name": "Message-ID", "value": "<abc@acme.com>"},
                {"name": "In-Reply-To", "value": "<parent@acme.com>"},
                {"name": "References", "value": "<root@acme.com> <parent@acme.com>"},
                {"name": "From", "value": "a@acme.com"},
            ],
            "body": {"data": _b64("hi")},
        },
    }
    p = parse_message(raw)
    assert p["rfc_message_id"] == "<abc@acme.com>"
    assert p["in_reply_to"] == "<parent@acme.com>"
    assert p["references"] == "<root@acme.com> <parent@acme.com>"
    assert p["body_text"] == "hi" and p["body_mime"] == "text/plain"


def test_parser_falls_back_to_html_body():
    raw = {
        "id": "m2", "threadId": "t2",
        "payload": {
            "mimeType": "multipart/alternative",
            "headers": [{"name": "From", "value": "a@acme.com"}],
            "parts": [
                {"mimeType": "text/html", "body": {"data": _b64("<p>decision: ship it</p>")}},
            ],
        },
    }
    p = parse_message(raw)
    assert "ship it" in p["body_text"]            # HTML-only body is NOT silently dropped
    assert p["body_mime"] == "text/html"


def test_parser_prefers_plain_over_html():
    raw = {
        "id": "m3", "threadId": "t3",
        "payload": {
            "mimeType": "multipart/alternative",
            "headers": [],
            "parts": [
                {"mimeType": "text/html", "body": {"data": _b64("<p>html</p>")}},
                {"mimeType": "text/plain", "body": {"data": _b64("plain")}},
            ],
        },
    }
    p = parse_message(raw)
    assert p["body_text"] == "plain" and p["body_mime"] == "text/plain"


def test_parser_extracts_attachment_metadata_only():
    raw = {
        "id": "m4", "threadId": "t4",
        "payload": {
            "mimeType": "multipart/mixed",
            "headers": [{"name": "From", "value": "a@acme.com"}],
            "parts": [
                {"mimeType": "text/plain", "body": {"data": _b64("see attached")}},
                {"mimeType": "application/pdf", "filename": "Q3.pdf",
                 "body": {"attachmentId": "att-1", "size": 84210}},
            ],
        },
    }
    p = parse_message(raw)
    assert p["body_text"] == "see attached"
    assert p["attachments"] == [
        {"filename": "Q3.pdf", "mime_type": "application/pdf", "size": 84210, "attachment_id": "att-1"}
    ]


async def test_content_encrypted_at_rest_and_decrypted_for_memory(monkeypatch):
    # With keys configured, sensitive content is sealed in the stored payload, and the Memory
    # read seam decrypts it transparently — full round-trip, determinism preserved.
    import navis.memory.service as mem
    import navis.signals.gmail.connector as conn
    from cryptography.fernet import Fernet
    from navis.platform.security.crypto import FieldCipher
    from types import SimpleNamespace

    cipher = FieldCipher([Fernet.generate_key().decode()])
    monkeypatch.setattr(conn, "field_cipher", cipher)
    monkeypatch.setattr(mem, "field_cipher", cipher)

    res = await GmailConnector().full_sync(
        make_ctx(), FakeGmailClient(profile_history_id="1", messages=[_raw("m1", "1")])
    )
    ev = res.events[0]
    # At rest: ciphertext, not the plaintext body. Routing fields stay readable.
    assert ev.payload["body_text"].startswith("fenc:1:")
    assert "hello world" not in str(ev.payload)
    assert ev.payload["_content_fields"] == ["body_text", "snippet"]

    # Memory's read seam round-trips it back to plaintext for extraction.
    rec = SimpleNamespace(
        id=ev.event_id, organization_id=ev.org_id, user_id=ev.user_id,
        integration_id=ev.integration_id, source=ev.source.value,
        event_type=ev.event_type.value, external_id=ev.external_id,
        external_version=ev.external_version, occurred_at=ev.timestamp, payload=ev.payload,
        extra=ev.metadata,
    )
    view = mem.SignalViewFactory.from_record(rec)
    assert view.payload["body_text"] == "hello world"


def test_bcc_list_and_attachment_filenames_sealed_at_rest(monkeypatch):
    import navis.signals.gmail.connector as conn
    from cryptography.fernet import Fernet
    from navis.platform.security.crypto import FieldCipher

    cipher = FieldCipher([Fernet.generate_key().decode()])
    monkeypatch.setattr(conn, "field_cipher", cipher)

    raw = {
        "id": "ma", "threadId": "ta", "internalDate": "1700000000000",
        "payload": {
            "mimeType": "multipart/mixed",
            "headers": [{"name": "From", "value": "a@x.com"}, {"name": "Bcc", "value": "secret@legal.com"}],
            "parts": [
                {"mimeType": "text/plain", "body": {"data": _b64("hi")}},
                {"mimeType": "application/pdf", "filename": "Layoff_Plan.xlsx",
                 "body": {"attachmentId": "z", "size": 9}},
            ],
        },
    }
    ev = GmailConnector._to_event(make_ctx(), parse_message(raw))
    assert ev.payload["bcc_emails"][0].startswith("fenc:1:")        # hidden recipient sealed
    assert "secret@legal.com" not in str(ev.payload)
    assert ev.payload["attachments"][0]["filename"].startswith("fenc:1:")  # filename sealed
    assert "Layoff_Plan" not in str(ev.payload)


def test_content_passthrough_when_no_keys(monkeypatch):
    # No keys configured (dev default): bodies remain plaintext, but the field is still
    # self-described so the reader runs one uniform decrypt path everywhere.
    import navis.signals.gmail.connector as conn
    from navis.platform.security.crypto import FieldCipher

    monkeypatch.setattr(conn, "field_cipher", FieldCipher([]))  # disabled -> passthrough
    ev = GmailConnector._to_event(make_ctx(), parse_message(_raw("m9", "1")))
    assert ev.payload["body_text"] == "hello world"          # passthrough, not sealed
    assert "body_text" in ev.payload["_content_fields"]


def test_parser_normalizes_addresses_and_captures_intelligence_headers():
    raw = {
        "id": "m5", "threadId": "t5",
        "payload": {
            "mimeType": "text/plain",
            "headers": [
                {"name": "From", "value": "John Doe <John@Acme.com>"},
                {"name": "To", "value": "\"Smith, Jane\" <jane@b.com>, bob@c.com"},
                {"name": "Cc", "value": "carol@d.com"},
                {"name": "Bcc", "value": "dan@e.com"},
                {"name": "List-ID", "value": "<news.stripe.com>"},
                {"name": "Auto-Submitted", "value": "auto-generated"},
                {"name": "Precedence", "value": "bulk"},
            ],
            "body": {"data": _b64("hi")},
        },
        "sizeEstimate": 4096,
    }
    p = parse_message(raw)
    assert p["sender_email"] == "john@acme.com" and p["sender_name"] == "John Doe"
    # getaddresses keeps the comma-in-quoted-name address intact (not split into two).
    assert p["to_emails"] == ["jane@b.com", "bob@c.com"]
    assert p["cc_emails"] == ["carol@d.com"] and p["bcc_emails"] == ["dan@e.com"]
    assert p["list_id"] == "<news.stripe.com>"
    assert p["auto_submitted"] == "auto-generated" and p["precedence"] == "bulk"
    assert p["size_estimate"] == 4096


def test_parser_nested_multipart_mixed_alternative():
    raw = {
        "id": "m6", "threadId": "t6",
        "payload": {
            "mimeType": "multipart/mixed",
            "headers": [],
            "parts": [
                {
                    "mimeType": "multipart/alternative",
                    "parts": [
                        {"mimeType": "text/html", "body": {"data": _b64("<p>x</p>")}},
                        {"mimeType": "text/plain", "body": {"data": _b64("nested plain")}},
                    ],
                },
                {"mimeType": "application/pdf", "filename": "a.pdf", "body": {"attachmentId": "z", "size": 1}},
            ],
        },
    }
    p = parse_message(raw)
    assert p["body_text"] == "nested plain" and p["body_mime"] == "text/plain"
    assert [a["filename"] for a in p["attachments"]] == ["a.pdf"]


def test_parser_records_truncation_metadata():
    big = "x" * 25000
    raw = {
        "id": "m7", "threadId": "t7",
        "payload": {"mimeType": "text/plain", "headers": [], "body": {"data": _b64(big)}},
    }
    p = parse_message(raw)
    assert p["body_truncated"] is True
    assert p["body_length"] == 25000
    assert len(p["body_text"]) == 20000


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
    # Email is an immutable fact -> constant version (NOT historyId, which drifts on label
    # changes and would mint a duplicate signal for the same email on a full-resync).
    assert {e.external_version for e in res.events} == {"1"}
    # Every external email body is marked untrusted at capture (prompt-injection boundary):
    # external origin + the hard gate that its text is never executed as a command.
    assert all(e.metadata["content_trust_level"] == "untrusted" for e in res.events)
    assert all(e.metadata["source_type"] == "external_email" for e in res.events)
    assert all(e.metadata["trusted_for_execution"] is False for e in res.events)


async def test_same_email_is_idempotent_across_resync():
    # Re-fetching the same message with a DIFFERENT historyId (e.g. after a label change on
    # a full-resync) must yield the SAME event_id — one email, one signal, forever.
    ctx = make_ctx()
    a = await GmailConnector().full_sync(
        ctx, FakeGmailClient(profile_history_id="1", messages=[_raw("m1", "1001")])
    )
    b = await GmailConnector().full_sync(
        ctx, FakeGmailClient(profile_history_id="2", messages=[_raw("m1", "9999")])
    )
    assert a.events[0].event_id == b.events[0].event_id


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


async def test_history_drain_spans_pages_and_advances_once():
    # Two history pages; the cursor advances to the final response historyId after a
    # COMPLETE drain, capturing messages from every page.
    pages = [
        {"history": [{"messagesAdded": [{"message": {"id": "m1"}}]}], "historyId": "1200", "nextPageToken": "1"},
        {"history": [{"messagesAdded": [{"message": {"id": "m2"}}]}], "historyId": "1200"},
    ]
    client = FakeGmailClient(messages=[_raw("m1", "1"), _raw("m2", "1")], history_pages=pages)
    res = await GmailConnector().incremental_sync(make_ctx(), client, {"history_id": "1000"})
    assert {e.external_id for e in res.events} == {"m1", "m2"}
    assert res.cursor == {"history_id": "1200"}


async def test_get_messages_skips_deleted_but_reraises_transient():
    from navis.signals.gmail.client import GmailClient
    from navis.signals.google.transport import GoogleAPIError

    class _Transport:
        def __init__(self, behavior):
            self.behavior = behavior

        async def request(self, _method, url, **_kw):
            mid = url.rsplit("/", 1)[-1]
            b = self.behavior[mid]
            if isinstance(b, Exception):
                raise b
            return b

    # A 404 (deleted between list and fetch) is skipped; live messages still captured.
    ok = {"id": "m1", "payload": {}}
    t = _Transport({"m1": ok, "m2": GoogleAPIError(404, "gone")})
    got = await GmailClient(t).get_messages(["m1", "m2"])  # type: ignore[arg-type]
    assert [g["id"] for g in got] == ["m1"]

    # A transient 503 must NOT be swallowed — it re-raises so the pass retries (no loss).
    t2 = _Transport({"m1": ok, "m2": GoogleAPIError(503, "unavailable")})
    with pytest.raises(GoogleAPIError):
        await GmailClient(t2).get_messages(["m1", "m2"])  # type: ignore[arg-type]


async def test_history_page_cap_fails_loud_and_does_not_advance_cursor(monkeypatch):
    from navis.platform.config import settings
    from navis.signals.gmail.connector import GmailSyncIncomplete

    monkeypatch.setattr(settings, "gmail_max_history_pages", 1)
    # Page 0 has a nextPageToken -> more pages remain past the cap.
    pages = [
        {"history": [{"messagesAdded": [{"message": {"id": "m1"}}]}], "historyId": "1200", "nextPageToken": "1"},
        {"history": [{"messagesAdded": [{"message": {"id": "m2"}}]}], "historyId": "1300"},
    ]
    client = FakeGmailClient(messages=[_raw("m1", "1"), _raw("m2", "1")], history_pages=pages)
    with pytest.raises(GmailSyncIncomplete):  # loud + retryable; cursor NOT advanced past unread pages
        await GmailConnector().incremental_sync(make_ctx(), client, {"history_id": "1000"})
