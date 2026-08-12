"""Google Meet conference creation on Navis-scheduled meetings.

Pins the write path that had zero coverage and therefore shipped a fully
implemented but permanently unreachable conference branch:

  * the scheduling contract must EXPLICITLY request a conference — the sender's
    `payload.get("conference", False)` fallback is what silently produced
    Meet-less events for every booking;
  * the wire request must carry `conferenceData.createRequest` AND
    `conferenceDataVersion=1` — Google ignores the body without the param;
  * `requestId` must be deterministic across processes, so a retry re-uses the
    conference instead of minting a second one;
  * a PENDING conference must never be reported as a usable Meet link.

Non-conference behaviour is pinned alongside, so re-enabling Meet cannot change
what a plain event puts on the wire.
"""
from __future__ import annotations

import uuid

from navis.execution.providers._senders import _google_event_id
from navis.execution.providers.calendar import (
    CalendarProvider,
    CancelMeetingProvider,
    RescheduleMeetingProvider,
)
from navis.execution.providers.gmail import GmailProvider
from navis.execution.providers.slack import SlackProvider
from navis.reasoning.scheduling.contract import (
    MeetingSlot,
    ResolvedAttendee,
    ScheduledProposal,
)
from navis.signals.calendar.client import (
    CalendarClient,
    _conference_request_id,
    conference_info,
)

MEET_URL = "https://meet.google.com/abc-defg-hij"


class _CapturingTransport:
    """Records exactly what CalendarClient puts on the wire."""

    def __init__(self, response: dict | None = None) -> None:
        self.calls: list[dict] = []
        self._response = response if response is not None else {"id": "evt-1"}

    async def request(self, method, url, *, params=None, json=None):
        self.calls.append(
            {"method": method, "url": url, "params": params or {}, "json": json or {}}
        )
        return self._response

    @property
    def last(self) -> dict:
        return self.calls[-1]


def _proposal(title: str = "Sync", start: str = "2026-09-01T15:00:00+00:00"):
    return ScheduledProposal(
        attendee=ResolvedAttendee(email="alice@acme.com", source="direct_email"),
        slot=MeetingSlot(
            start=start,  # type: ignore[arg-type]  # pydantic parses ISO
            end="2026-09-01T15:30:00+00:00",  # type: ignore[arg-type]
            time_zone="UTC",
            duration_minutes=30,
        ),
        title=title,
    )


# ── 1. The producer explicitly requests a conference ───────────────────


def test_scheduling_payload_explicitly_requests_a_conference():
    """The root cause: no producer ever set the key, so the sender's
    `payload.get("conference", False)` default won every single time."""
    payload = _proposal().to_execution_payload()

    assert "conference" in payload, "producer must define the key, not rely on a default"
    assert payload["conference"] is True


def test_conference_is_typed_on_the_contract_not_an_untyped_dict_key():
    assert ScheduledProposal.model_fields["conference"].annotation is bool
    assert _proposal().conference is True


async def test_scheduling_payload_reaches_the_wire_as_conference_data():
    """Joins the two halves of the bug: the seam between the producer's payload
    and the sender's `payload.get("conference", False)` read. Mirrors the exact
    expression in `_senders.calendar_sender`, so a payload-key rename breaks
    this test instead of silently reverting to Meet-less events."""
    payload = _proposal().to_execution_payload()
    transport = _CapturingTransport()

    await CalendarClient(transport).create_event(  # type: ignore[arg-type]
        summary=payload["title"],
        start_iso=payload["start"],
        end_iso=payload["end"],
        conference=bool(payload.get("conference", False)),  # the sender's read
        event_id=_google_event_id(uuid.uuid4()),
    )

    assert "createRequest" in transport.last["json"]["conferenceData"]
    assert transport.last["params"]["conferenceDataVersion"] == "1"


# ── 2-4. The wire request Google actually receives ─────────────────────


async def test_client_receives_conference_true_and_sends_create_request():
    transport = _CapturingTransport()
    await CalendarClient(transport).create_event(  # type: ignore[arg-type]
        summary="Sync",
        start_iso="2026-09-01T15:00:00+00:00",
        end_iso="2026-09-01T15:30:00+00:00",
        conference=True,
        event_id="navisdeadbeef",
    )

    body = transport.last["json"]
    create_request = body["conferenceData"]["createRequest"]
    assert create_request["requestId"]
    assert create_request["conferenceSolutionKey"] == {"type": "hangoutsMeet"}


async def test_conference_data_version_is_set_to_one():
    """Without this param Google silently ignores the conferenceData body."""
    transport = _CapturingTransport()
    await CalendarClient(transport).create_event(  # type: ignore[arg-type]
        summary="Sync",
        start_iso="2026-09-01T15:00:00+00:00",
        end_iso="2026-09-01T15:30:00+00:00",
        conference=True,
        event_id="navisdeadbeef",
    )

    assert transport.last["params"]["conferenceDataVersion"] == "1"


# ── 5-7. requestId determinism ─────────────────────────────────────────


def test_request_id_is_deterministic_for_the_same_execution():
    execution_id = uuid.uuid4()
    event_id = _google_event_id(execution_id)

    first = _conference_request_id(event_id, "Sync", "start", "end")
    second = _conference_request_id(event_id, "Sync", "start", "end")

    assert first == second


def test_request_id_differs_between_executions():
    a = _conference_request_id(_google_event_id(uuid.uuid4()), "Sync", "s", "e")
    b = _conference_request_id(_google_event_id(uuid.uuid4()), "Sync", "s", "e")

    assert a != b


def test_request_id_is_rooted_in_the_execution_event_id():
    """One deterministic identity per ExecutionRequest, not a second scheme."""
    event_id = _google_event_id(uuid.uuid4())

    assert _conference_request_id(event_id, "Sync", "s", "e").startswith(event_id)


def test_request_id_survives_process_hash_randomization():
    """The old implementation used `hash(start_iso)`, which PYTHONHASHSEED
    randomizes per process — two workers computed different ids for the same
    meeting. This recomputes the id in a FRESH interpreter with a different
    seed and asserts the value is unchanged."""
    import subprocess
    import sys

    snippet = (
        "from navis.signals.calendar.client import _conference_request_id;"
        "print(_conference_request_id(None, 'Sync', 'start', 'end'))"
    )

    def run(seed: str) -> str:
        out = subprocess.run(
            [sys.executable, "-c", snippet],
            capture_output=True,
            text=True,
            env={"PYTHONHASHSEED": seed, "PATH": "/usr/bin:/bin"},
            check=True,
        )
        return out.stdout.strip()

    assert run("0") == run("12345") == _conference_request_id(None, "Sync", "start", "end")


# ── 8-10. Meet URL extraction, including the pending case ──────────────


def test_hangout_link_produces_meet_url():
    info = conference_info({"id": "e1", "hangoutLink": MEET_URL})

    assert info["meet_url"] == MEET_URL
    assert info["conference_status"] == "success"


def test_video_entry_point_produces_meet_url_when_hangout_link_absent():
    info = conference_info(
        {
            "id": "e1",
            "conferenceData": {
                "createRequest": {"status": {"statusCode": "success"}},
                "entryPoints": [
                    {"entryPointType": "video", "uri": MEET_URL},
                    {"entryPointType": "phone", "uri": "tel:+1-555-0100", "pin": "1234"},
                ],
            },
        }
    )

    assert info["meet_url"] == MEET_URL
    assert info["conference_status"] == "success"


def test_pending_conference_does_not_fabricate_a_meet_url():
    info = conference_info(
        {"id": "e1", "conferenceData": {"createRequest": {"status": {"statusCode": "pending"}}}}
    )

    assert "meet_url" not in info
    assert info["conference_status"] == "pending"


def test_failed_conference_reports_failure_without_a_url():
    info = conference_info(
        {"id": "e1", "conferenceData": {"createRequest": {"status": {"statusCode": "failure"}}}}
    )

    assert "meet_url" not in info
    assert info["conference_status"] == "failure"


def test_phone_entry_points_are_never_surfaced():
    """Privacy: phone numbers / PINs / access codes stay out of the result,
    matching the minimization the ingest connector already applies."""
    info = conference_info(
        {
            "id": "e1",
            "hangoutLink": MEET_URL,
            "conferenceData": {
                "entryPoints": [
                    {"entryPointType": "phone", "uri": "tel:+1-555-0100", "pin": "1234"},
                    {"entryPointType": "sip", "uri": "sip:x@example.com"},
                ]
            },
        }
    )

    assert info == {"meet_url": MEET_URL, "conference_status": "success"}


async def test_meet_url_reaches_the_provider_reference():
    """End of the chain: the URL the frontend reads off `provider_reference`."""

    async def sender(_payload):
        return "evt-1", conference_info({"id": "evt-1", "hangoutLink": MEET_URL})

    result = await CalendarProvider(sender=sender).execute(
        {"title": "Sync", "start": "2026-09-01T15:00:00+00:00"}
    )

    assert result.ok
    assert result.reference is not None
    assert result.reference.to_dict()["meet_url"] == MEET_URL


# ── 11-14. Nothing else moved ──────────────────────────────────────────


async def test_event_idempotency_id_is_still_sent():
    """Meet support must not disturb the client-generated event id that makes
    duplicate bookings impossible."""
    transport = _CapturingTransport()
    event_id = _google_event_id(uuid.uuid4())
    await CalendarClient(transport).create_event(  # type: ignore[arg-type]
        summary="Sync",
        start_iso="2026-09-01T15:00:00+00:00",
        end_iso="2026-09-01T15:30:00+00:00",
        conference=True,
        event_id=event_id,
    )

    assert transport.last["json"]["id"] == event_id


async def test_non_conference_event_is_unchanged():
    """The default path must put exactly what it always did on the wire."""
    transport = _CapturingTransport()
    await CalendarClient(transport).create_event(  # type: ignore[arg-type]
        summary="Sync",
        start_iso="2026-09-01T15:00:00+00:00",
        end_iso="2026-09-01T15:30:00+00:00",
    )

    assert "conferenceData" not in transport.last["json"]
    assert "conferenceDataVersion" not in transport.last["params"]
    assert transport.last["params"]["sendUpdates"] == "all"


def test_non_conference_event_yields_empty_conference_info():
    assert conference_info({"id": "e1"}) == {}


async def test_send_updates_behaviour_is_untouched_by_conference():
    transport = _CapturingTransport()
    await CalendarClient(transport).create_event(  # type: ignore[arg-type]
        summary="Sync",
        start_iso="2026-09-01T15:00:00+00:00",
        end_iso="2026-09-01T15:30:00+00:00",
        conference=True,
        send_updates="none",
    )

    assert transport.last["params"]["sendUpdates"] == "none"


async def test_dry_run_provider_still_creates_nothing():
    """Approval gate: a provider with no sender must never reach Google."""
    result = await CalendarProvider().execute(
        {"title": "Sync", "start": "2026-09-01T15:00:00+00:00"}
    )

    assert result.reference is not None
    assert result.reference.to_dict()["dry_run"] is True
    assert "meet_url" not in result.reference.to_dict()


async def test_other_senders_keep_the_bare_id_contract():
    """Only the calendar CREATE sender returns a pair. Gmail, Slack and the
    calendar mutation verbs still return a plain id — widening the create
    path must not have leaked into them."""

    async def bare_id_sender(_payload):
        return "external-1"

    cases = [
        (GmailProvider(sender=bare_id_sender), {"to": "a@b.com", "subject": "s", "body": "b"}),
        (SlackProvider(sender=bare_id_sender), {"channel": "C1", "text": "hi"}),
        (
            RescheduleMeetingProvider(sender=bare_id_sender),
            {"event_id": "e1", "start": "2026-09-01T15:00:00+00:00"},
        ),
        (CancelMeetingProvider(sender=bare_id_sender), {"event_id": "e1"}),
    ]

    for provider, payload in cases:
        result = await provider.execute(payload)
        assert result.ok, provider.action_type
        assert result.reference is not None
        assert result.reference.external_id == "external-1", provider.action_type
