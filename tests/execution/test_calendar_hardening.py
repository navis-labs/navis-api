"""Regression suite for the Calendar execution flow hardening.

Each test here pins one of the reported production failures so it cannot come
back silently:

  * approve/reject/cancel are idempotent for a REPEATED intent and still 409 on
    a CONFLICTING one (the "409 invalid execution transition executed ->
    approved" report);
  * a replayed execution can never produce a second Google Calendar event —
    covered at both layers that can replay (the worker, and the sender's own
    409-as-success handling);
  * reschedule / cancel providers validate and dispatch through the same
    at-most-once machinery.
"""
from __future__ import annotations

import uuid

import pytest

from navis.execution.contract import ProviderReference, ProviderResult
from navis.execution.models import ActionType
from navis.execution.providers import (
    CancelMeetingProvider,
    RescheduleMeetingProvider,
    get_provider,
)
from navis.execution.providers._senders import _google_event_id
from navis.execution.providers.base import ProviderValidationError
from navis.execution.service import ExecutionService
from navis.execution.status import ExecutionStatus, InvalidTransition
from navis.execution.worker import execute_pass
from navis.platform.events import EventType
from tests.execution.fakes import DECISION, ORG_A, USER, FakeExecutionRepo, event_types


async def _requested(repo, action_type="schedule_meeting", payload=None):
    svc = ExecutionService(repo=repo)
    req, _ = await svc.request(
        ORG_A,
        DECISION,
        action_type,
        payload or {"title": "Sync", "start": "2026-08-09T10:00:00+00:00"},
        reasoning_signal_ids=[],
        reasoning_version=None,
        state_watermark=None,
    )
    return svc, req


# ── Idempotent human verbs ─────────────────────────────────────────────


async def test_approve_after_executed_is_a_noop_not_a_conflict():
    """The exact reported bug: the meeting IS booked, and clicking Approve
    again must NOT surface an error."""
    repo = FakeExecutionRepo()
    svc, req = await _requested(repo)
    await svc.approve(req.id, USER)
    await svc.begin_execution(req)
    await svc.complete_execution(
        req, ProviderResult(ok=True, reference=ProviderReference("calendar", "evt-1"))
    )
    assert req.status == ExecutionStatus.EXECUTED.value

    before = len(repo.events)
    again = await svc.approve(req.id, uuid.uuid4())

    assert again.status == ExecutionStatus.EXECUTED.value
    # No second approval, no second audit row, no second approver recorded.
    assert len(repo.events) == before
    assert again.approved_by == USER


@pytest.mark.parametrize(
    "intermediate", [ExecutionStatus.APPROVED, ExecutionStatus.EXECUTING]
)
async def test_approve_is_idempotent_from_every_in_flight_state(intermediate):
    repo = FakeExecutionRepo()
    svc, req = await _requested(repo)
    await svc.approve(req.id, USER)
    if intermediate is ExecutionStatus.EXECUTING:
        await svc.begin_execution(req)

    before = len(repo.events)
    again = await svc.approve(req.id, USER)

    assert again.status == intermediate.value
    assert len(repo.events) == before


async def test_repeated_reject_and_cancel_are_noops():
    """Re-issuing an intent that already holds is not a conflict."""
    repo = FakeExecutionRepo()
    svc, req = await _requested(repo)
    await svc.reject(req.id, USER)
    before = len(repo.events)
    assert (await svc.reject(req.id, USER)).status == ExecutionStatus.REJECTED.value
    assert len(repo.events) == before

    repo2 = FakeExecutionRepo()
    svc2, req2 = await _requested(repo2)
    await svc2.approve(req2.id, USER)
    await svc2.cancel(req2.id)
    before2 = len(repo2.events)
    assert (await svc2.cancel(req2.id)).status == ExecutionStatus.CANCELLED.value
    assert len(repo2.events) == before2


async def test_conflicting_intent_still_raises():
    """Idempotency must not become permissiveness. Approving something a human
    explicitly declined, or rejecting something already executed, are real
    conflicts and must still fail loudly."""
    repo = FakeExecutionRepo()
    svc, rejected = await _requested(repo)
    await svc.reject(rejected.id, USER)
    with pytest.raises(InvalidTransition):
        await svc.approve(rejected.id, USER)

    repo2 = FakeExecutionRepo()
    svc2, cancelled = await _requested(repo2)
    await svc2.cancel(cancelled.id)
    with pytest.raises(InvalidTransition):
        await svc2.approve(cancelled.id, USER)

    repo3 = FakeExecutionRepo()
    svc3, done = await _requested(repo3)
    await svc3.approve(done.id, USER)
    await svc3.begin_execution(done)
    await svc3.complete_execution(
        done, ProviderResult(ok=True, reference=ProviderReference("calendar", "e"))
    )
    with pytest.raises(InvalidTransition):
        await svc3.reject(done.id, USER)


# ── No duplicate Google events ─────────────────────────────────────────


def test_google_event_id_is_deterministic_and_alphabet_safe():
    """Google requires base32hex (a-v, 0-9), 5-1024 chars. A UUID's hex digits
    plus the 'navis' prefix satisfy that; a violation would 400 at insert."""
    eid = uuid.UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
    first = _google_event_id(eid)
    assert first == _google_event_id(eid)  # replay -> same id -> Google 409s
    assert _google_event_id(uuid.uuid4()) != first
    assert 5 <= len(first) <= 1024
    assert set(first) <= set("abcdefghijklmnopqrstuv0123456789")


class _CountingProvider:
    """Records how many times the provider was actually dispatched."""

    name = "calendar"
    action_type = "schedule_meeting"

    def __init__(self) -> None:
        self.calls = 0

    async def validate(self, payload):
        return None

    async def execute(self, payload):
        self.calls += 1
        return ProviderResult(
            ok=True, reference=ProviderReference("calendar", "evt-real")
        )

    async def status(self, reference):
        return "scheduled"


async def test_worker_never_re_dispatches_a_row_that_already_has_a_reference():
    """The duplicate-event window that idempotency keys alone do NOT close.

    A row can hold a real provider reference and still be re-claimable: the
    stale-EXECUTING recovery pushes an orphaned row to FAILED, and FAILED with
    attempts remaining is claimable work. Without the guard, the second pass
    books a SECOND meeting for the same request.
    """
    repo = FakeExecutionRepo()
    provider = _CountingProvider()
    svc, req = await _requested(repo)
    await svc.approve(req.id, USER)

    first = await execute_pass(repo, providers={"schedule_meeting": provider})
    assert first["executed"] == 1
    assert provider.calls == 1
    assert req.status == ExecutionStatus.EXECUTED.value

    # Simulate the recovery path re-opening the row for retry.
    req.status = ExecutionStatus.FAILED.value
    second = await execute_pass(repo, providers={"schedule_meeting": provider})

    assert provider.calls == 1, "provider must not be dispatched a second time"
    assert second["executed"] == 1
    assert req.status == ExecutionStatus.EXECUTED.value
    assert req.provider_reference["external_id"] == "evt-real"


async def test_dry_run_reference_does_not_block_a_real_send():
    """A dry-run reference records no real side effect, so wiring a real sender
    later must still be able to execute the row."""
    repo = FakeExecutionRepo()
    provider = _CountingProvider()
    svc, req = await _requested(repo)
    await svc.approve(req.id, USER)
    req.provider_reference = {
        "provider": "calendar", "external_id": "dryrun-abc", "dry_run": True
    }

    await execute_pass(repo, providers={"schedule_meeting": provider})

    assert provider.calls == 1
    assert req.provider_reference["external_id"] == "evt-real"


# ── Reschedule / cancel providers ──────────────────────────────────────


def test_new_action_types_are_registered():
    assert get_provider(ActionType.RESCHEDULE_MEETING.value) is not None
    assert get_provider(ActionType.CANCEL_MEETING.value) is not None


async def test_reschedule_provider_requires_an_existing_event():
    p = RescheduleMeetingProvider()
    with pytest.raises(ProviderValidationError):
        await p.validate({"start": "2026-08-09T10:00:00+00:00"})
    with pytest.raises(ProviderValidationError):
        await p.validate({"event_id": "evt-1"})  # no new start
    await p.validate({"event_id": "evt-1", "start": "2026-08-09T10:00:00+00:00"})


async def test_cancel_provider_requires_an_existing_event():
    p = CancelMeetingProvider()
    with pytest.raises(ProviderValidationError):
        await p.validate({"title": "Sync"})
    await p.validate({"event_id": "evt-1"})


async def test_mutations_run_through_the_normal_lifecycle():
    """Reschedule/cancel are ordinary ExecutionRequests — same approval gate,
    same outbox events, same audit trail as a booking."""
    repo = FakeExecutionRepo()
    svc, req = await _requested(
        repo, action_type=ActionType.CANCEL_MEETING.value, payload={"event_id": "evt-9"}
    )
    await svc.approve(req.id, USER)
    result = await execute_pass(repo, max_attempts=3)

    assert result["executed"] == 1
    assert req.status == ExecutionStatus.EXECUTED.value
    assert EventType.ACTION_EXECUTED.value in event_types(repo)
