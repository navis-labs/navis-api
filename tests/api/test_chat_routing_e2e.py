"""End-to-end through the ACTUAL /chat handler, not just the classifier.

`classify()` returning the right enum is necessary but not sufficient — the
reported bug was that the endpoint answered the wrong question, and an endpoint
can route correctly and still compose the wrong response. These tests drive
`chat_router.chat(...)` itself with stubbed agents and assert on the
ChatResponse the frontend would actually receive.

The single assertion that matters most, repeated for every operational verb:
the response must not be the memory fallback.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from navis.api import chat_router as R
from navis.reasoning.calendar_ops.contract import (
    CalendarOpsClarification,
    MeetingMutationProposal,
    TargetEvent,
)
from navis.reasoning.email_ops import EmailClarification, EmailProposal

ORG = uuid.UUID("11111111-1111-1111-1111-111111111111")
USER = uuid.UUID("22222222-2222-2222-2222-222222222222")
NOW = datetime(2026, 8, 7, 9, 0, tzinfo=timezone.utc)

_MEMORY_FALLBACK_MARKERS = ("doesn't have enough memory", "not enough memory")


@dataclass
class _Principal:
    org_id: uuid.UUID = ORG
    subject_id: uuid.UUID = USER


class _DB:
    """Just enough AsyncSession surface for the handler's own calls."""

    async def commit(self):
        return None

    async def scalars(self, _stmt):
        return SimpleNamespace(all=lambda: [])


class _ExecutionService:
    """Records what the endpoint asked the execution layer to create."""

    created: list[dict] = []

    def __init__(self, db):
        self.db = db

    async def request(self, org_id, decision_id, action_type, payload, **kw):
        row = SimpleNamespace(
            id=uuid.uuid4(), status="requested", action_type=action_type
        )
        type(self).created.append(
            {"action_type": action_type, "payload": payload, "decision_id": decision_id}
        )
        return row, True


@pytest.fixture(autouse=True)
def _reset_created():
    _ExecutionService.created = []
    yield


def _body(question: str, **kw) -> R.ChatRequest:
    return R.ChatRequest(question=question, time_zone="UTC", **kw)


def _assert_not_memory_fallback(resp: R.ChatResponse) -> None:
    lowered = resp.content.lower()
    for marker in _MEMORY_FALLBACK_MARKERS:
        assert marker not in lowered, (
            f"operational request answered with the memory fallback: {resp.content!r}"
        )


# ── Reschedule ─────────────────────────────────────────────────────────


async def test_reschedule_creates_a_reschedule_execution_request(monkeypatch):
    target = TargetEvent(
        event_id="evt-1",
        title="Investor sync",
        start=datetime(2026, 8, 7, 14, 0, tzinfo=timezone.utc),
        end=datetime(2026, 8, 7, 14, 30, tzinfo=timezone.utc),
        attendees=["arjun@fund.com"],
    )
    proposal = MeetingMutationProposal(
        operation="reschedule",
        target=target,
        new_start=datetime(2026, 8, 7, 17, 0, tzinfo=timezone.utc),
        new_end=datetime(2026, 8, 7, 18, 0, tzinfo=timezone.utc),
    )

    class _Agent:
        def __init__(self, db, **kw):
            pass

        async def propose(self, ctx):
            assert ctx.operation == "reschedule"
            return proposal

    monkeypatch.setattr(R, "CalendarOpsAgent", _Agent)
    monkeypatch.setattr(R, "ExecutionService", _ExecutionService)

    resp = await R.chat(_body("Reschedule the meeting to 5–6 PM"), _Principal(), _DB())

    _assert_not_memory_fallback(resp)
    assert resp.intent == "command"
    assert _ExecutionService.created[0]["action_type"] == "reschedule_meeting"
    assert _ExecutionService.created[0]["payload"]["event_id"] == "evt-1"
    assert _ExecutionService.created[0]["payload"]["start"] == "2026-08-07T17:00:00+00:00"
    # The card the frontend renders.
    chip = resp.actionProposals[0]
    assert chip.action_type == "reschedule_meeting"
    assert chip.execution_id
    assert chip.structured_payload.duration_minutes == 60
    assert resp.targetEventId == "evt-1"


# ── Cancel ─────────────────────────────────────────────────────────────


async def test_cancel_creates_a_cancel_execution_request(monkeypatch):
    target = TargetEvent(
        event_id="evt-9",
        title="Standup",
        start=datetime(2026, 8, 7, 9, 30, tzinfo=timezone.utc),
        end=datetime(2026, 8, 7, 9, 45, tzinfo=timezone.utc),
        attendees=["team@navis.ai"],
    )

    class _Agent:
        def __init__(self, db, **kw):
            pass

        async def propose(self, ctx):
            assert ctx.operation == "cancel"
            return MeetingMutationProposal(operation="cancel", target=target)

    monkeypatch.setattr(R, "CalendarOpsAgent", _Agent)
    monkeypatch.setattr(R, "ExecutionService", _ExecutionService)

    resp = await R.chat(_body("Cancel the meeting"), _Principal(), _DB())

    _assert_not_memory_fallback(resp)
    assert _ExecutionService.created[0]["action_type"] == "cancel_meeting"
    assert _ExecutionService.created[0]["payload"]["event_id"] == "evt-9"
    # No slot is being moved, so no start is sent.
    assert "start" not in _ExecutionService.created[0]["payload"]
    assert resp.actionProposals[0].channel == "cancel_meeting"


async def test_a_calendar_ops_clarification_creates_nothing(monkeypatch):
    """An open question must never produce an approvable side effect."""

    class _Agent:
        def __init__(self, db, **kw):
            pass

        async def propose(self, ctx):
            return CalendarOpsClarification(
                field="event",
                question="You have 2 meetings that could match. Which one?",
                options=[{"event_id": "a", "label": "A"}, {"event_id": "b", "label": "B"}],
            )

    monkeypatch.setattr(R, "CalendarOpsAgent", _Agent)
    monkeypatch.setattr(R, "ExecutionService", _ExecutionService)

    resp = await R.chat(_body("Cancel the meeting"), _Principal(), _DB())

    assert _ExecutionService.created == []
    assert resp.actionProposals == []
    assert resp.clarification.field == "event"
    assert len(resp.clarification.options) == 2


async def test_nothing_to_act_on_is_prose_not_a_prompt(monkeypatch):
    """`field='none'` is informational — rendering it as a clarification box
    would ask the user to answer a question that isn't one."""

    class _Agent:
        def __init__(self, db, **kw):
            pass

        async def propose(self, ctx):
            return CalendarOpsClarification(
                field="none",
                question="You have no upcoming meetings, so there's nothing to change.",
            )

    monkeypatch.setattr(R, "CalendarOpsAgent", _Agent)
    monkeypatch.setattr(R, "ExecutionService", _ExecutionService)

    resp = await R.chat(_body("Cancel the meeting"), _Principal(), _DB())

    assert resp.clarification is None
    assert "nothing to change" in resp.content
    _assert_not_memory_fallback(resp)


# ── Email ──────────────────────────────────────────────────────────────


async def test_send_email_asks_for_the_body_and_creates_nothing_yet(monkeypatch):
    class _Agent:
        def __init__(self, db):
            pass

        async def propose(self, ctx):
            return EmailClarification(
                field="body",
                question="What should the email to x@y.com say?",
                resolved={"to": "x@y.com", "subject": None, "body": None},
            )

    monkeypatch.setattr(R, "EmailOpsAgent", _Agent)
    monkeypatch.setattr(R, "ExecutionService", _ExecutionService)

    resp = await R.chat(
        _body("Send email to mulayamabmrc@gmail.com"), _Principal(), _DB()
    )

    _assert_not_memory_fallback(resp)
    assert _ExecutionService.created == []
    # The recipient is echoed back so the next turn alone completes the send.
    assert resp.pendingEmail["to"] == "x@y.com"


async def test_a_complete_email_creates_a_send_execution_request(monkeypatch):
    class _Agent:
        def __init__(self, db):
            pass

        async def propose(self, ctx):
            return EmailProposal(
                to="x@y.com", subject="Deck", body="The deck is attached."
            )

    monkeypatch.setattr(R, "EmailOpsAgent", _Agent)
    monkeypatch.setattr(R, "ExecutionService", _ExecutionService)

    resp = await R.chat(
        _body("email x@y.com saying the deck is attached"), _Principal(), _DB()
    )

    _assert_not_memory_fallback(resp)
    created = _ExecutionService.created[0]
    assert created["action_type"] == "send_email"
    assert created["payload"]["to"] == ["x@y.com"]
    assert created["payload"]["subject"] == "Deck"
    assert resp.actionProposals[0].channel == "send_email"


# ── Calendar query ─────────────────────────────────────────────────────


async def test_calendar_query_answers_from_the_calendar(monkeypatch):
    class _Agent:
        def __init__(self, db):
            pass

        async def answer(self, ctx):
            return SimpleNamespace(text="You have 3 meetings this week:\n· Investor sync")

    monkeypatch.setattr(R, "CalendarQueryAgent", _Agent)

    resp = await R.chat(_body("What meetings do I have this week?"), _Principal(), _DB())

    _assert_not_memory_fallback(resp)
    assert "3 meetings this week" in resp.content
    assert resp.intent == "question"


# ── Graceful degradation ───────────────────────────────────────────────


async def test_empty_memory_falls_back_to_workspace_signals(monkeypatch):
    """"What should I do first?" with empty memory but a live calendar."""
    from navis.ml.rag.grounding import Grounding, GroundedContext

    class _EmptyContextBuilder:
        def __init__(self, db):
            pass

        async def build(self, org_id, question, **kw):
            return GroundedContext(org_id=str(org_id), question=question, facts=[])

    async def _fallback(db, org_id, question, *, now=None):
        return GroundedContext(
            org_id=str(org_id),
            question=question,
            facts=[
                Grounding(
                    kind="calendar",
                    source="signal:abc",
                    text="Calendar event: Investor sync at 2026-08-07T14:00:00+00:00",
                )
            ],
        )

    async def _generator(org_id, context):
        # Proves the generator was reached WITH the fallback facts.
        assert context.facts
        return SimpleNamespace(
            text="Start with the investor sync at 2pm.", sources=["signal:abc"]
        )

    monkeypatch.setattr(R, "ContextBuilder", _EmptyContextBuilder)
    monkeypatch.setattr(R, "build_workspace_context", _fallback)
    monkeypatch.setattr(R, "recommendation", _generator)

    resp = await R.chat(_body("What should I do first?"), _Principal(), _DB())

    _assert_not_memory_fallback(resp)
    assert "investor sync" in resp.content.lower()


async def test_genuinely_empty_names_the_connected_sources(monkeypatch):
    """When there really is nothing, say WHICH sources are connected —
    "not enough memory" names an internal concept the founder can't act on."""
    from navis.ml.rag.grounding import GroundedContext

    class _EmptyContextBuilder:
        def __init__(self, db):
            pass

        async def build(self, org_id, question, **kw):
            return GroundedContext(org_id=str(org_id), question=question, facts=[])

    async def _empty_fallback(db, org_id, question, *, now=None):
        return GroundedContext(org_id=str(org_id), question=question, facts=[])

    async def _labels(db, org_id):
        return ["Gmail", "Google Calendar"]

    monkeypatch.setattr(R, "ContextBuilder", _EmptyContextBuilder)
    monkeypatch.setattr(R, "build_workspace_context", _empty_fallback)
    monkeypatch.setattr(R, "connected_source_labels", _labels)

    resp = await R.chat(_body("Who am I keeping waiting?"), _Principal(), _DB())

    assert "Gmail and Google Calendar" in resp.content
    assert "are connected" in resp.content
    _assert_not_memory_fallback(resp)


async def test_no_sources_connected_says_so(monkeypatch):
    from navis.ml.rag.grounding import GroundedContext

    class _EmptyContextBuilder:
        def __init__(self, db):
            pass

        async def build(self, org_id, question, **kw):
            return GroundedContext(org_id=str(org_id), question=question, facts=[])

    async def _empty_fallback(db, org_id, question, *, now=None):
        return GroundedContext(org_id=str(org_id), question=question, facts=[])

    async def _labels(db, org_id):
        return []

    monkeypatch.setattr(R, "ContextBuilder", _EmptyContextBuilder)
    monkeypatch.setattr(R, "build_workspace_context", _empty_fallback)
    monkeypatch.setattr(R, "connected_source_labels", _labels)

    resp = await R.chat(_body("What should I do first?"), _Principal(), _DB())

    assert "no workspace sources are connected" in resp.content
    _assert_not_memory_fallback(resp)


# ── Chat-origin actions stay OUT of the proactive queue ────────────────


async def test_chat_created_actions_are_labelled_chat_origin(monkeypatch):
    """The Action Center is "things Navis noticed". A meeting the founder just
    asked for in conversation is not that, and showing it there put one row on
    two surfaces — each able to approve it."""
    from navis.execution.models import ExecutionOrigin
    from navis.reasoning.scheduling.contract import (
        MeetingSlot,
        ResolvedAttendee,
        ScheduledProposal,
    )

    captured: dict = {}

    class _Svc:
        def __init__(self, db):
            pass

        async def request(self, org_id, decision_id, action_type, payload, **kw):
            captured["origin"] = kw.get("origin")
            return SimpleNamespace(id=uuid.uuid4(), status="requested"), True

    class _Agent:
        def __init__(self, db, **kw):
            pass

        async def propose(self, ctx):
            return ScheduledProposal(
                attendee=ResolvedAttendee(email="chrissy@x.com", source="direct_email"),
                slot=MeetingSlot(
                    start=datetime(2026, 8, 10, 17, 0, tzinfo=timezone.utc),
                    end=datetime(2026, 8, 10, 17, 30, tzinfo=timezone.utc),
                    time_zone="UTC",
                    duration_minutes=30,
                ),
                title="Meeting with Chrissy",
            )

    monkeypatch.setattr(R, "SchedulingAgent", _Agent)
    monkeypatch.setattr(R, "ExecutionService", _Svc)

    await R.chat(
        _body("Schedule a meeting with chrissy@x.com tomorrow at 5 PM"),
        _Principal(),
        _DB(),
    )

    assert captured["origin"] is ExecutionOrigin.CHAT


async def test_the_proactive_bridge_labels_its_actions_proactive():
    """The counterpart: everything the Decision->Execution bridge creates is
    proactive by construction, because that bridge IS the proactive producer."""
    import inspect

    from navis.orchestration import brief

    source = inspect.getsource(brief.propose_actions)
    assert "origin=ExecutionOrigin.PROACTIVE" in source


def test_the_action_center_feed_filters_on_origin():
    """`GET /actions` must not be a firehose of every REQUESTED row."""
    import inspect

    from navis.execution import router as execution_router

    source = inspect.getsource(execution_router.list_pending)
    assert "ExecutionOrigin.PROACTIVE" in source
    assert "ExecutionRequest.origin" in source
