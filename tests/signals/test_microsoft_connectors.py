"""Graph delta drain + the three Microsoft connectors.

Focused on the failure modes that are SILENT if they regress — a drain that
restarts every pass, a cursor that advances over undrained pages, a dedupe key
that collapses two distinct facts into one. Each of those produces a system that
looks like it is working while losing a founder's data, which is the only class
of bug worth this much test surface.
"""
from __future__ import annotations

import uuid

import pytest

from navis.signals.base import SignalContext, SignalEventType, SignalSource
from navis.signals.microsoft.delta import DeltaSyncIncomplete, drain_delta
from navis.signals.microsoft.transport import GraphAPIError
from navis.signals.onedrive import OneDriveClient, OneDriveConnector
from navis.signals.outlook import OutlookClient, OutlookConnector
from navis.signals.outlook_calendar import (
    OutlookCalendarClient,
    OutlookCalendarConnector,
)


def _ctx() -> SignalContext:
    return SignalContext(
        org_id=uuid.uuid4(), user_id=uuid.uuid4(), integration_id=uuid.uuid4()
    )


class FakeTransport:
    """Replays a scripted list of Graph responses and records what was requested."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls: list[tuple[str, dict | None]] = []

    async def request(self, method, url, *, params=None, json=None):
        self.calls.append((url, params))
        nxt = self._responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


# ---- delta drain ------------------------------------------------------------


async def test_drain_follows_next_links_verbatim():
    """The nextLink/deltaLink carry an opaque token in the query string. If the
    drain rebuilds a URL instead of following one, Graph restarts the
    enumeration — the sync looks healthy and never advances."""
    transport = FakeTransport([
        {"value": [{"id": "1"}], "@odata.nextLink": "https://graph/next?$skiptoken=ABC"},
        {"value": [{"id": "2"}], "@odata.deltaLink": "https://graph/delta?$deltatoken=XYZ"},
    ])
    page = await drain_delta(transport, "/me/messages/delta", params={"$select": "id"}, resource="t")

    assert [i["id"] for i in page.items] == ["1", "2"]
    assert page.delta_link == "https://graph/delta?$deltatoken=XYZ"
    # Second call used the nextLink EXACTLY, and carried no params of its own.
    assert transport.calls[1] == ("https://graph/next?$skiptoken=ABC", None)


async def test_drain_applies_params_only_to_the_first_request():
    transport = FakeTransport([
        {"value": [], "@odata.nextLink": "https://graph/n2"},
        {"value": [], "@odata.deltaLink": "https://graph/d"},
    ])
    await drain_delta(transport, "/start", params={"$filter": "x"}, resource="t")
    assert transport.calls[0][1] == {"$filter": "x"}
    assert transport.calls[1][1] is None


async def test_expired_delta_token_requests_full_resync():
    """410 Gone is a normal lifecycle event, not a failure. Dead-lettering it
    would retry a cursor Graph has permanently retired."""
    transport = FakeTransport([GraphAPIError(410, "gone")])
    page = await drain_delta(transport, "https://graph/stale", resource="t")
    assert page.full_resync_required is True
    assert page.items == []


async def test_other_errors_propagate():
    transport = FakeTransport([GraphAPIError(403, "forbidden")])
    with pytest.raises(GraphAPIError):
        await drain_delta(transport, "/x", resource="t")


async def test_truncated_drain_raises_rather_than_advancing_the_cursor(monkeypatch):
    """A partial drain has no valid cursor. Persisting one would resume AFTER
    the pages that were never read, losing them permanently — so this fails
    loud and retryable and leaves the old cursor in place."""
    from navis.platform.config import settings

    monkeypatch.setattr(settings, "microsoft_max_sync_pages", 2)
    transport = FakeTransport([
        {"value": [{"id": "1"}], "@odata.nextLink": "https://graph/n1"},
        {"value": [{"id": "2"}], "@odata.nextLink": "https://graph/n2"},
        {"value": [{"id": "3"}], "@odata.nextLink": "https://graph/n3"},
    ])
    with pytest.raises(DeltaSyncIncomplete):
        await drain_delta(transport, "/start", resource="t")


async def test_drain_without_a_delta_link_refuses():
    """Graph always terminates with a deltaLink. Finishing without one means the
    response is not the shape we think, and persisting an empty cursor would
    silently restart the enumeration on every pass."""
    transport = FakeTransport([{"value": [{"id": "1"}]}])
    with pytest.raises(DeltaSyncIncomplete):
        await drain_delta(transport, "/start", resource="t")


# ---- Outlook mail -----------------------------------------------------------


def _message(**over):
    base = {
        "id": "AAMk-immutable",
        "changeKey": "CK1",
        "conversationId": "conv-1",
        "internetMessageId": "<abc@contoso.com>",
        "subject": "Renewal terms",
        "from": {"emailAddress": {"name": "Priya Raman", "address": "Priya@Contoso.com"}},
        "toRecipients": [{"emailAddress": {"name": "You", "address": "you@navis.ai"}}],
        "ccRecipients": [],
        "receivedDateTime": "2026-09-01T10:30:00Z",
        "bodyPreview": "Following up on the renewal",
        "body": {"contentType": "text", "content": "Following up on the renewal terms."},
        "categories": [],
        "isRead": False,
    }
    base.update(over)
    return base


async def test_outlook_incremental_maps_to_the_shared_mail_verb():
    transport = FakeTransport([
        {"value": [_message()], "@odata.deltaLink": "https://graph/d2"},
    ])
    result = await OutlookConnector().incremental_sync(
        _ctx(), OutlookClient(transport), {"delta_link": "https://graph/d1"}
    )
    assert len(result.events) == 1
    event = result.events[0]
    # The source names the vendor; the verb is the SAME one Gmail emits, so no
    # downstream consumer needs an Outlook branch.
    assert event.source is SignalSource.OUTLOOK
    assert event.event_type is SignalEventType.EMAIL_RECEIVED
    assert event.external_id == "AAMk-immutable"
    assert result.cursor == {"delta_link": "https://graph/d2"}


async def test_outlook_version_is_constant():
    """Arrival is a one-time immutable fact, so the version never moves. Gmail
    reaches the same conclusion for the same reason."""
    transport = FakeTransport([
        {"value": [_message(changeKey="CK2")], "@odata.deltaLink": "https://graph/d"},
    ])
    result = await OutlookConnector().incremental_sync(
        _ctx(), OutlookClient(transport), {"delta_link": "https://graph/d0"}
    )
    assert result.events[0].external_version == "1"


async def test_outlook_read_or_flag_churn_is_not_a_new_fact():
    """Graph bumps changeKey on ANY property write — opening a mail, flagging
    it, filing it under a category. Versioning by it would re-emit the same
    arrival as a duplicate EMAIL_RECEIVED every time the founder touches their
    inbox."""
    ctx = _ctx()
    connector = OutlookConnector()
    first = await connector.incremental_sync(
        ctx,
        OutlookClient(FakeTransport([
            {"value": [_message(changeKey="CK1")], "@odata.deltaLink": "https://g/d"},
        ])),
        {"delta_link": "https://g/d0"},
    )
    second = await connector.incremental_sync(
        ctx,
        OutlookClient(FakeTransport([
            {
                "value": [_message(changeKey="CK2", isRead=True, categories=["Blue"])],
                "@odata.deltaLink": "https://g/d",
            },
        ])),
        {"delta_link": "https://g/d0"},
    )
    assert first.events[0].dedupe_key == second.events[0].dedupe_key


async def test_outlook_mail_is_captured_as_untrusted():
    """An email body is attacker-controllable text. The gate that stops a future
    agent EXECUTING instructions found in it is stamped at capture, where the
    external origin is known with certainty — exactly as Gmail does it."""
    transport = FakeTransport([
        {"value": [_message()], "@odata.deltaLink": "https://graph/d"},
    ])
    result = await OutlookConnector().incremental_sync(
        _ctx(), OutlookClient(transport), {"delta_link": "https://graph/d0"}
    )
    metadata = result.events[0].metadata
    assert metadata["source_type"] == "external_email"
    assert metadata["content_trust_level"] == "untrusted"
    assert metadata["trusted_for_execution"] is False


async def test_outlook_missing_delta_link_requests_full_sync():
    result = await OutlookConnector().incremental_sync(
        _ctx(), OutlookClient(FakeTransport([])), {}
    )
    assert result.full_resync_required is True


async def test_outlook_skips_removals():
    transport = FakeTransport([
        {
            "value": [{"id": "gone-1", "@removed": {"reason": "deleted"}}],
            "@odata.deltaLink": "https://graph/d",
        },
    ])
    result = await OutlookConnector().incremental_sync(
        _ctx(), OutlookClient(transport), {"delta_link": "https://graph/d0"}
    )
    assert result.events == []


async def test_outlook_payload_uses_the_canonical_mail_keys():
    """These keys are Gmail's, not Graph's. Every memory extractor and the
    people graph read mail payloads by key; if Outlook emitted `conversationId`
    where Gmail emits `thread_id`, half the org's mail would go unseen by
    consumers nobody remembered to update."""
    from navis.signals.outlook.parser import parse_message

    parsed = parse_message(_message())
    for key in (
        "message_id", "thread_id", "subject", "sender_email", "sender_name",
        "to_emails", "cc_emails", "rfc_message_id", "body_text", "snippet",
        "attachments", "label_ids",
    ):
        assert key in parsed, key
    assert parsed["thread_id"] == "conv-1"
    assert parsed["sender_email"] == "priya@contoso.com"  # normalised
    assert parsed["to_emails"] == ["you@navis.ai"]
    assert parsed["rfc_message_id"] == "<abc@contoso.com>"


async def test_outlook_payload_seals_content_fields():
    """Body and preview are encrypted at rest exactly as Gmail's are, and the
    payload self-describes which fields were sealed so the read seam decrypts
    generically."""
    from navis.signals.outlook.parser import parse_message

    parsed = parse_message(_message())
    assert set(parsed["_content_fields"]) <= {"body_text", "snippet", "bcc"}
    assert "body_text" in parsed["_content_fields"]
    # Routing identity stays plaintext — the people graph needs it.
    assert parsed["sender_email"] == "priya@contoso.com"


# ---- Outlook calendar -------------------------------------------------------


def _event(**over):
    base = {
        "id": "evt-1",
        "changeKey": "CK1",
        "subject": "Board sync",
        "start": {"dateTime": "2026-09-10T09:00:00.0000000", "timeZone": "UTC"},
        "end": {"dateTime": "2026-09-10T10:00:00.0000000", "timeZone": "UTC"},
        "isCancelled": False,
        "organizer": {"emailAddress": {"name": "Chair", "address": "Chair@Contoso.com"}},
        "attendees": [
            {
                "type": "required",
                "emailAddress": {"name": "Priya Raman", "address": "priya@contoso.com"},
                "status": {"response": "accepted"},
            }
        ],
        "type": "singleInstance",
        "createdDateTime": "2026-09-01T08:00:00Z",
        "lastModifiedDateTime": "2026-09-02T08:00:00Z",
        "isOnlineMeeting": True,
        "onlineMeetingProvider": "teamsForBusiness",
        "onlineMeeting": {"joinUrl": "https://teams/join/secret", "tollNumber": "+1-555"},
    }
    base.update(over)
    return base


async def test_calendar_maps_to_the_shared_calendar_verbs():
    transport = FakeTransport([
        {"value": [_event()], "@odata.deltaLink": "https://graph/d"},
    ])
    result = await OutlookCalendarConnector().incremental_sync(
        _ctx(), OutlookCalendarClient(transport), {"delta_link": "https://graph/d0"}
    )
    event = result.events[0]
    assert event.source is SignalSource.OUTLOOK_CALENDAR
    assert event.event_type is SignalEventType.CALENDAR_EVENT_UPDATED
    # Graph's `subject` is Google's `summary` — translated at capture so
    # downstream reads one shape.
    assert event.payload["summary"] == "Board sync"


async def test_calendar_cancellation_uses_the_cancelled_verb():
    transport = FakeTransport([
        {"value": [_event(isCancelled=True)], "@odata.deltaLink": "https://graph/d"},
    ])
    result = await OutlookCalendarConnector().incremental_sync(
        _ctx(), OutlookCalendarClient(transport), {"delta_link": "https://graph/d0"}
    )
    assert result.events[0].event_type is SignalEventType.CALENDAR_EVENT_CANCELLED


async def test_calendar_drops_dial_in_credentials():
    """`onlineMeeting` carries a join URL and a toll number — working
    conference credentials. The flags are kept so the product can say "this is
    a video call"; the credentials are not, because the signal log is
    append-only and outlives every decision about who should have them."""
    transport = FakeTransport([
        {"value": [_event()], "@odata.deltaLink": "https://graph/d"},
    ])
    result = await OutlookCalendarConnector().incremental_sync(
        _ctx(), OutlookCalendarClient(transport), {"delta_link": "https://graph/d0"}
    )
    conference = result.events[0].payload["conference_data"]
    assert conference["has_video"] is True
    assert conference["has_phone"] is True
    serialized = str(result.events[0].payload)
    assert "secret" not in serialized
    assert "+1-555" not in serialized


async def test_calendar_attendees_are_minimized_to_email_and_status():
    transport = FakeTransport([
        {"value": [_event()], "@odata.deltaLink": "https://graph/d"},
    ])
    result = await OutlookCalendarConnector().incremental_sync(
        _ctx(), OutlookCalendarClient(transport), {"delta_link": "https://graph/d0"}
    )
    attendee = result.events[0].payload["attendees"][0]
    assert attendee == {
        "email": "priya@contoso.com",
        "response_status": "accepted",
        "optional": False,
        "resource": False,
    }
    assert "Priya Raman" not in str(result.events[0].payload)


# ---- OneDrive ---------------------------------------------------------------


def _item(**over):
    base = {
        "id": "item-1",
        "eTag": "ET1",
        "cTag": "CT1",
        "name": "Series A model.xlsx",
        "file": {"mimeType": "application/vnd.ms-excel"},
        "size": 1024,
        "createdDateTime": "2026-09-01T08:00:00Z",
        "lastModifiedDateTime": "2026-09-02T08:00:00Z",
        "createdBy": {"user": {"displayName": "Founder", "email": "Founder@Contoso.com"}},
        "lastModifiedBy": {"user": {"displayName": "CFO", "email": "cfo@contoso.com"}},
        "parentReference": {"id": "parent-1", "driveId": "drive-1"},
        "shared": {"scope": "users"},
    }
    base.update(over)
    return base


async def test_onedrive_maps_to_the_shared_file_verbs():
    transport = FakeTransport([
        {"value": [_item()], "@odata.deltaLink": "https://graph/d"},
    ])
    result = await OneDriveConnector().incremental_sync(
        _ctx(), OneDriveClient(transport), {"delta_link": "https://graph/d0"}
    )
    event = result.events[0]
    assert event.source is SignalSource.ONEDRIVE
    assert event.event_type is SignalEventType.DRIVE_FILE_UPDATED
    assert event.external_version == "ET1"  # eTag: content OR metadata changes
    assert event.payload["owners"] == [{"email": "founder@contoso.com"}]
    assert event.payload["shared"] is True


async def test_onedrive_deleted_item_is_trashed_not_dropped():
    transport = FakeTransport([
        {
            "value": [_item(deleted={"state": "deleted"})],
            "@odata.deltaLink": "https://graph/d",
        },
    ])
    result = await OneDriveConnector().incremental_sync(
        _ctx(), OneDriveClient(transport), {"delta_link": "https://graph/d0"}
    )
    assert result.events[0].event_type is SignalEventType.DRIVE_FILE_TRASHED
    assert result.events[0].payload["trashed"] is True


async def test_onedrive_skips_the_drive_root():
    """The delta feed's first entry is the root itself, which is not a document
    and would produce a meaningless "file updated" fact on every full sync."""
    transport = FakeTransport([
        {
            "value": [_item(id="root-1", root={}), _item()],
            "@odata.deltaLink": "https://graph/d",
        },
    ])
    result = await OneDriveConnector().incremental_sync(
        _ctx(), OneDriveClient(transport), {"delta_link": "https://graph/d0"}
    )
    assert [e.external_id for e in result.events] == ["item-1"]


async def test_onedrive_carries_no_download_url():
    transport = FakeTransport([
        {
            "value": [_item(**{"@microsoft.graph.downloadUrl": "https://dl/secret-token"})],
            "@odata.deltaLink": "https://graph/d",
        },
    ])
    result = await OneDriveConnector().incremental_sync(
        _ctx(), OneDriveClient(transport), {"delta_link": "https://graph/d0"}
    )
    assert "secret-token" not in str(result.events[0].payload)
