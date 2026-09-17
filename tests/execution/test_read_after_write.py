"""Read-after-write: EXECUTED must imply the read model already reflects reality.

The reported failure: cancellation worked on Google, and the assistant kept
answering as if the meeting existed. Nothing was broken in the cancel path —
Navis had simply never been told about its own write. `ACTION_EXECUTED` had one
subscriber (Observation's outcome capture, which writes no signals), and the
only producer of calendar signals was the Celery sync connector.

The contract these tests pin:

    a calendar mutation reaches EXECUTED
      => the affected event has been re-read from Google
      => and written into signal_events

and, just as important, that a failure to do the bookkeeping never rewrites
history about whether the side effect happened.
"""
from __future__ import annotations

import uuid

import pytest

from navis.execution.contract import ProviderReference, ProviderResult
from navis.execution.models import ActionType
from navis.execution.service import ExecutionService
from navis.execution.status import ExecutionStatus
from navis.execution.worker import execute_pass
from navis.signals.calendar.reconcile import (
    CALENDAR_MUTATION_ACTION_TYPES,
    event_id_for,
)
from tests.execution.fakes import DECISION, ORG_A, USER, FakeExecutionRepo


class _Recorder:
    """Stands in for `reconcile_calendar_event`."""

    def __init__(self, *, explode: bool = False) -> None:
        self.calls: list[tuple[uuid.UUID, str]] = []
        self.explode = explode

    async def __call__(self, org_id, event_id):
        self.calls.append((org_id, event_id))
        if self.explode:
            raise RuntimeError("reconcile blew up")
        return True


class _Provider:
    name = "calendar"

    def __init__(self, action_type: str, external_id: str | None = "evt-real") -> None:
        self.action_type = action_type
        self._external_id = external_id

    async def validate(self, payload):
        return None

    async def execute(self, payload):
        return ProviderResult(
            ok=True, reference=ProviderReference("calendar", self._external_id)
        )

    async def status(self, reference):
        return "ok"


async def _approved(repo, action_type: str, payload: dict):
    svc = ExecutionService(repo=repo)
    req, _ = await svc.request(
        ORG_A,
        DECISION,
        action_type,
        payload,
        reasoning_signal_ids=[],
        reasoning_version=None,
        state_watermark=None,
    )
    await svc.approve(req.id, USER)
    return req


# ── Which event did we touch? ──────────────────────────────────────────


def test_event_id_comes_from_the_payload_for_mutations():
    """A reschedule/cancel names an EXISTING event, so the id is in the payload
    and is known before Google is called."""
    assert event_id_for("cancel_meeting", {"event_id": "evt-1"}, None) == "evt-1"
    assert event_id_for("reschedule_meeting", {"event_id": "evt-2"}, "ignored") == "evt-2"


def test_event_id_comes_from_the_provider_for_creations():
    """A new meeting has no id until Google replies, so it comes back on the
    provider reference."""
    assert event_id_for("schedule_meeting", {"title": "Sync"}, "evt-new") == "evt-new"


def test_non_calendar_actions_reconcile_nothing():
    for action_type in ("send_email", "send_slack", "create_task"):
        assert action_type not in CALENDAR_MUTATION_ACTION_TYPES
        assert event_id_for(action_type, {"event_id": "evt-1"}, "x") is None


# ── The ordering guarantee ─────────────────────────────────────────────


@pytest.mark.parametrize(
    "action_type,payload,expected_event",
    [
        (ActionType.CANCEL_MEETING.value, {"event_id": "evt-cancelled"}, "evt-cancelled"),
        (
            ActionType.RESCHEDULE_MEETING.value,
            {"event_id": "evt-moved", "start": "2026-08-09T10:00:00+00:00"},
            "evt-moved",
        ),
        (
            ActionType.SCHEDULE_MEETING.value,
            {"title": "Sync", "start": "2026-08-09T10:00:00+00:00"},
            "evt-real",
        ),
    ],
)
async def test_every_calendar_mutation_reconciles_the_read_model(
    action_type, payload, expected_event
):
    repo = FakeExecutionRepo()
    req = await _approved(repo, action_type, payload)
    recorder = _Recorder()

    result = await execute_pass(
        repo, providers={action_type: _Provider(action_type)}, reconciler=recorder
    )

    assert result["executed"] == 1
    assert req.status == ExecutionStatus.EXECUTED.value
    assert recorder.calls == [(ORG_A, expected_event)]


async def test_reconcile_runs_BEFORE_the_row_reaches_executed():
    """The whole point of the ordering. The frontend polls until EXECUTED, so
    if reconciliation happened after, the next chat turn could still race a
    stale read model — which is the bug we are fixing."""
    repo = FakeExecutionRepo()
    req = await _approved(repo, ActionType.CANCEL_MEETING.value, {"event_id": "evt-1"})
    observed: list[str] = []

    async def _reconciler(org_id, event_id):
        observed.append(req.status)  # status AT reconcile time

    await execute_pass(
        repo,
        providers={ActionType.CANCEL_MEETING.value: _Provider("cancel_meeting")},
        reconciler=_reconciler,
    )

    assert observed == [ExecutionStatus.EXECUTING.value]
    assert req.status == ExecutionStatus.EXECUTED.value


async def test_non_calendar_actions_do_not_reconcile():
    repo = FakeExecutionRepo()
    await _approved(repo, ActionType.SEND_EMAIL.value, {"to": "a@b.com", "subject": "Hi"})
    recorder = _Recorder()

    await execute_pass(
        repo, providers={"send_email": _Provider("send_email")}, reconciler=recorder
    )

    assert recorder.calls == []


# ── Failure must not rewrite history ───────────────────────────────────


async def test_a_failing_reconcile_still_reports_the_execution_as_executed():
    """The meeting IS cancelled on Google. Flipping the row to FAILED because
    bookkeeping broke would both lie about that and re-drive the provider on
    the next pass — attempting to cancel an already-cancelled meeting forever."""
    repo = FakeExecutionRepo()
    req = await _approved(repo, ActionType.CANCEL_MEETING.value, {"event_id": "evt-1"})
    recorder = _Recorder(explode=True)

    result = await execute_pass(
        repo,
        providers={ActionType.CANCEL_MEETING.value: _Provider("cancel_meeting")},
        reconciler=recorder,
    )

    assert result["executed"] == 1
    assert result["failed"] == 0
    assert req.status == ExecutionStatus.EXECUTED.value
    assert recorder.calls == [(ORG_A, "evt-1")]


async def test_a_missing_event_id_is_survivable():
    """A provider that returns no reference on a creation leaves us with
    nothing to reconcile. That is a degraded outcome, not a failed execution."""
    repo = FakeExecutionRepo()
    req = await _approved(
        repo, ActionType.SCHEDULE_MEETING.value, {"title": "Sync", "start": "x"}
    )
    recorder = _Recorder()

    result = await execute_pass(
        repo,
        providers={
            ActionType.SCHEDULE_MEETING.value: _Provider("schedule_meeting", None)
        },
        reconciler=recorder,
    )

    assert result["executed"] == 1
    assert req.status == ExecutionStatus.EXECUTED.value
    assert recorder.calls == []


async def test_a_dry_run_changes_nothing_so_reconciles_nothing():
    """Dry-run providers never touched Google. Reconciling would spend a session
    and an API call to re-confirm a state we never altered."""
    repo = FakeExecutionRepo()
    await _approved(repo, ActionType.CANCEL_MEETING.value, {"event_id": "evt-1"})
    recorder = _Recorder()

    # The default registry is dry-run by design (providers/__init__.py).
    result = await execute_pass(repo, reconciler=recorder)

    assert result["executed"] == 1
    assert recorder.calls == []


# ── The reconcile mapping itself ───────────────────────────────────────


def _integration():
    from types import SimpleNamespace

    return SimpleNamespace(id=uuid.uuid4(), user_id=uuid.uuid4())


def _grant():
    """What reconcile now hands `_to_signal`: a DETACHED credential snapshot.

    Not an ORM row, and that is the point — reconcile used to hold the live
    `Integration` across the Google call with decrypted tokens assigned onto it,
    which the session then committed as plaintext.
    """
    from navis.signals.calendar.reconcile import _CalendarGrant

    return _CalendarGrant(
        integration_id=uuid.uuid4(),
        user_id=uuid.uuid4(),
        access_token="enc:1:stored-ciphertext",
        refresh_token="enc:1:stored-ciphertext",
        token_expires_at=None,
    )


def test_a_live_event_maps_to_the_current_version():
    """Reconciled events must be BYTE-IDENTICAL to synced ones — same event-type
    rules, same payload minimisation, same version key — or the same meeting
    would exist twice in signal_events under two shapes."""
    from navis.signals.base import SignalEventType, SignalSource
    from navis.signals.calendar.reconcile import _to_signal

    raw = {
        "id": "evt-1",
        "etag": '"abc123"',
        "status": "confirmed",
        "summary": "Investor sync",
        "start": {"dateTime": "2026-08-09T14:00:00+00:00"},
        "end": {"dateTime": "2026-08-09T14:30:00+00:00"},
        "attendees": [{"email": "arjun@fund.com", "responseStatus": "accepted"}],
        "organizer": {"email": "founder@acme.com"},
    }
    signal = _to_signal(ORG_A, _grant(), raw)

    assert signal is not None
    assert signal.source is SignalSource.CALENDAR
    assert signal.event_type is SignalEventType.CALENDAR_EVENT_UPDATED
    assert signal.external_id == "evt-1"
    assert signal.external_version == "abc123"  # etag drives versioning
    assert signal.payload["summary"] == "Investor sync"
    assert signal.metadata["capture_mode"] == "reconcile"


def test_a_deleted_event_maps_to_a_cancellation():
    """The success case for a cancel, and the single most important thing to
    record: Google 404s, and we must write CANCELLED rather than nothing."""
    from navis.signals.base import SignalEventType
    from navis.signals.calendar.reconcile import _to_signal

    # Exactly the stub `_reconcile` synthesises on a 404/410 from Google.
    signal = _to_signal(ORG_A, _grant(), {"id": "evt-gone", "status": "cancelled"})

    assert signal is not None
    assert signal.event_type is SignalEventType.CALENDAR_EVENT_CANCELLED
    assert signal.external_id == "evt-gone"


def test_a_malformed_payload_does_not_raise():
    """Bookkeeping must never take down a successful execution."""
    from navis.signals.calendar.reconcile import _to_signal

    assert _to_signal(ORG_A, _grant(), {}) is None  # no id


def test_two_cancellations_of_one_event_do_not_dedupe_into_silence():
    """A deleted event has no etag, so a naive version key would make the
    second cancellation a duplicate and silently drop it."""
    from navis.signals.calendar.reconcile import cancellation_signal

    integration = _integration()
    first = cancellation_signal(ORG_A, integration, "evt-1")
    second = cancellation_signal(ORG_A, integration, "evt-1")
    assert first.external_version != second.external_version
    assert first.event_id != second.event_id
