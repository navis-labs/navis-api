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

    def __init__(self):
        self.rollbacks = 0

    async def commit(self):
        return None

    async def rollback(self):
        # Best-effort enrichment runs on the REQUEST's session; when it fails it
        # must hand that session back usable. Counted so the tests can prove it.
        self.rollbacks += 1

    async def scalars(self, _stmt):
        return SimpleNamespace(all=lambda: [])

    # Relevance evidence (P0-2) is read through `execute`. Returning no rows
    # exercises the real production path with UNKNOWN evidence, which is the
    # documented degradation — rather than masking the call behind an
    # AttributeError the handler would swallow.
    async def execute(self, _stmt):
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


@pytest.fixture(autouse=True)
def _entitled(monkeypatch):
    """Default every test to an ENTITLED organization.

    The action routes ask `OrganizationService.can_use(org, EXECUTION)` before
    dispatching, and `_DB` is deliberately too small to answer a real entitlement
    query. Stubbing it to True is what an ACTIVE org on a plan with EXECUTION
    resolves to, so the tests below exercise the path a real customer takes.
    The refusal path has its own test, which overrides this.
    """
    async def _can_use(self, org_id, capability):
        return True

    monkeypatch.setattr(R.OrganizationService, "can_use", _can_use)
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


# ---- Phase 2: State + Reasoning reach the grounded context ----------------
#
# Ask Navis used to read Memory (and raw signals) only — OrganizationalState was
# never loaded and run_reasoning_v2 never ran, so the LLM explained retrieved
# emails with no organizational understanding behind them. These pin the wiring
# through the seam ContextBuilder already exposed (`extra=`), and the graceful
# degradation that keeps chat working when State is cold or absent.

_AS_OF = datetime(2026, 8, 14, 21, 50, tzinfo=timezone.utc)
SIGNAL_ID = uuid.UUID("33333333-3333-3333-3333-333333333333")
# The memory record a folded State reached. Chat publishes it as
# `ChatResponse.stateWatermark`; `/brief/today` publishes the same value.
_WATERMARK = uuid.UUID("44444444-4444-4444-4444-444444444444")


def _fake_signal(org_id=ORG, signal_id=SIGNAL_ID, title="Commitment slipping"):
    # `entity_type` / `entity_id` are REQUIRED fields on the real ReasoningSignal
    # (reasoning/v2/contract.py) — they are the signal's subject, and the chat
    # composition resolves them to a name so four cooling signals don't reach the
    # model as four identical lines. A fake missing them isn't a stand-in for a
    # signal the pipeline can actually receive.
    return SimpleNamespace(
        signal_id=signal_id,
        org_id=org_id,
        entity_type="commitment",
        entity_id="ent-c1",
        title=title,
        why="Promised the deck by Friday; nothing sent.",
        state_as_of=_AS_OF,
    )


def _empty_view():
    from navis.state.v2.scope import ScopedView

    return ScopedView(
        intent_name="morning_brief", as_of=_AS_OF, state_watermark="wm-1",
        policy_version=None, counts={}, entities=[], citations={},
    )


def _stub_state(monkeypatch, *, org_seen: list, signals: list, view=None):
    """Wire a fake State + the two parallel read-side projections, recording the
    org_id State was asked for. `scope_for` is stubbed alongside `run_reasoning_v2`
    because both consume the same OrganizationalState."""
    # `entities` is the canonical raw form on OrganizationalState (models.py) and
    # the composition root reads it to name a signal's subject. Empty here: these
    # tests assert wiring and org-scoping, and an unresolvable subject must leave
    # the rendered conclusion exactly as it was.
    # `watermark.last_memory_id` is a REQUIRED field on the real
    # OrganizationalState (state/v2/models.py) — it is which memory record the
    # fold reached, and chat now returns it as `ChatResponse.stateWatermark` so a
    # client can tell whether an answer and the brief describe the same snapshot.
    # A fake without it isn't a stand-in for a state the pipeline can produce.
    state_obj = SimpleNamespace(
        org_id=ORG, as_of=_AS_OF, entities={},
        watermark=SimpleNamespace(last_memory_id=_WATERMARK),
    )
    monkeypatch.setattr(R, "scope_for", lambda state, intent, policy, *, as_of: view or _empty_view())

    class _StateService:
        def __init__(self, db):
            pass

        async def get_state(self, org_id):
            org_seen.append(org_id)
            return state_obj

    seen_state = []

    def _run_reasoning(state, **kw):
        seen_state.append(state)
        return signals

    monkeypatch.setattr(R, "StateService", _StateService)
    monkeypatch.setattr(R, "run_reasoning_v2", _run_reasoning)
    return state_obj, seen_state


async def test_reasoning_groundings_use_the_authenticated_org_and_reach_the_llm(monkeypatch):
    """State is read with the PRINCIPAL's org_id, reasoning runs on that state,
    and the resulting Grounding arrives through ContextBuilder's `extra=`."""
    from navis.ml.rag.grounding import Grounding, GroundedContext

    org_seen: list = []
    state_obj, seen_state = _stub_state(
        monkeypatch, org_seen=org_seen, signals=[_fake_signal()]
    )
    captured: dict = {}

    class _Builder:
        def __init__(self, db):
            pass

        async def build(self, org_id, question, **kw):
            captured["extra"] = kw.get("extra")
            captured["token_budget"] = kw.get("token_budget")
            # Mimic the real builder: memory hits first, `extra` appended after.
            memory = [Grounding(kind="memory", source="mem-1", text="Ajitesh emailed about pricing.")]
            return GroundedContext(
                org_id=str(org_id), question=question,
                facts=[*memory, *(kw.get("extra") or [])],
            )

    async def _generator(org_id, context):
        captured["kinds"] = [f.kind for f in context.facts]
        return SimpleNamespace(text="Chase the deck.", sources=["mem-1", str(SIGNAL_ID)])

    monkeypatch.setattr(R, "ContextBuilder", _Builder)
    monkeypatch.setattr(R, "recommendation", _generator)

    resp = await R.chat(_body("What should I focus on today?"), _Principal(), _DB())

    # 1. State was asked for using the authenticated org, never a client value.
    assert org_seen == [ORG]
    # 2. run_reasoning_v2 received that exact state object.
    assert seen_state == [state_obj]
    # 3-5. ReasoningSignal became a Grounding with real lineage + fields.
    extra = captured["extra"]
    assert len(extra) == 1
    g = extra[0]
    assert g.kind == "reasoning"
    assert g.source == str(SIGNAL_ID)
    assert "Commitment slipping" in g.text and "deck by Friday" in g.text
    assert g.occurred_at == _AS_OF
    # 6. Memory evidence is NOT displaced, and the budget is unchanged.
    assert captured["kinds"] == ["memory", "reasoning"]
    assert captured["token_budget"] == 1500
    _assert_not_memory_fallback(resp)


async def test_reasoning_grounding_is_scoped_to_one_org(monkeypatch):
    """Org A's reasoning evidence must never enter org B's chat context."""
    other_org = uuid.UUID("99999999-9999-9999-9999-999999999999")
    org_seen: list = []
    _stub_state(monkeypatch, org_seen=org_seen, signals=[_fake_signal(org_id=other_org)])

    extras, watermark = await R._state_groundings(_DB(), ORG)

    # State was fetched for the REQUEST org; nothing lets a caller pass another.
    assert org_seen == [ORG]
    # Sources are the reasoning signal ids of that org's own state — the helper
    # takes a single org_id, so cross-org mixing is not expressible.
    assert [g.source for g in extras] == [str(SIGNAL_ID)]


async def test_zero_reasoning_signals_leaves_chat_unchanged(monkeypatch):
    """An org with no reasoning output still gets the existing memory answer."""
    _stub_state(monkeypatch, org_seen=[], signals=[])

    extras, watermark = await R._state_groundings(_DB(), ORG)

    assert extras == []
    # A state with no signals still knows WHICH snapshot it read.
    assert watermark == str(_WATERMARK)


async def test_state_failure_degrades_instead_of_breaking_chat(monkeypatch):
    """State unavailable is an enrichment miss, never a failed turn."""

    class _BrokenStateService:
        def __init__(self, db):
            pass

        async def get_state(self, org_id):
            raise RuntimeError("state backend down")

    monkeypatch.setattr(R, "StateService", _BrokenStateService)

    db = _DB()
    # No state read => no groundings AND no watermark. Reporting a watermark
    # here would claim a snapshot that was never successfully folded.
    assert await R._state_groundings(db, ORG) == ([], None)
    assert db.rollbacks == 1, "a failed enrichment must not leave the caller a dead session"


async def test_state_timeout_degrades_instead_of_hanging_the_turn(monkeypatch):
    """A cold-cache State fold is a full projection rebuild (~13s measured).
    Chat must not wait for it."""
    import asyncio as _asyncio

    class _SlowStateService:
        def __init__(self, db):
            pass

        async def get_state(self, org_id):
            await _asyncio.sleep(5)
            raise AssertionError("should have been cancelled by the timeout")

    monkeypatch.setattr(R, "StateService", _SlowStateService)
    monkeypatch.setattr(R, "_REASONING_GROUNDING_TIMEOUT_S", 0.05)

    db = _DB()
    assert await R._state_groundings(db, ORG) == ([], None)
    # THE 500. `asyncio.wait_for` cancels the state read mid-transaction;
    # SQLAlchemy then marks it invalid and EVERY later statement on that session
    # raises PendingRollbackError. The turn degraded correctly but handed the
    # caller a poisoned session, and retrieval is the very next thing it does.
    assert db.rollbacks == 1, "a timed-out enrichment must not leave the caller a dead session"


async def test_reasoning_does_not_suppress_the_workspace_fallback(monkeypatch):
    """Reasoning groundings are org-wide and present on nearly every turn.
    Counting them as retrieval would permanently hide raw Calendar/Gmail
    evidence — the fallback decision must look at memory hits only."""
    from navis.ml.rag.grounding import Grounding, GroundedContext

    _stub_state(monkeypatch, org_seen=[], signals=[_fake_signal()])
    called: dict = {}

    class _EmptyMemoryBuilder:
        def __init__(self, db):
            pass

        async def build(self, org_id, question, **kw):
            # No memory hits; only the reasoning `extra` survives.
            return GroundedContext(
                org_id=str(org_id), question=question, facts=list(kw.get("extra") or [])
            )

    async def _fallback(db, org_id, question, *, now=None):
        called["fallback"] = True
        return GroundedContext(
            org_id=str(org_id), question=question,
            facts=[Grounding(kind="calendar", source="signal:abc",
                             text="Calendar event: Investor sync")],
        )

    async def _generator(org_id, context):
        called["kinds"] = [f.kind for f in context.facts]
        return SimpleNamespace(text="Investor sync first.", sources=["signal:abc"])

    monkeypatch.setattr(R, "ContextBuilder", _EmptyMemoryBuilder)
    monkeypatch.setattr(R, "build_workspace_context", _fallback)
    monkeypatch.setattr(R, "recommendation", _generator)

    resp = await R.chat(_body("What should I focus on today?"), _Principal(), _DB())

    assert called.get("fallback") is True, "reasoning suppressed the workspace fallback"
    # Composed: raw workspace evidence AND organizational understanding.
    assert called["kinds"] == ["calendar", "reasoning"]
    _assert_not_memory_fallback(resp)


# ---- Phase 3A: ScopedView (Context) reaches the grounded context ----------
#
# State feeds TWO parallel deterministic read-side projections — scope_for and
# run_reasoning_v2 — and both cross into ML as Grounding. Context does not feed
# Reasoning, and neither ever sees the model.


def _view_with_entities():
    from navis.state.v2.scope import Citation, EntityRef, ScopedView

    return ScopedView(
        intent_name="morning_brief", as_of=_AS_OF, state_watermark="wm-1",
        policy_version=None, counts={"commitment": 1, "person": 1},
        entities=[
            EntityRef(entity_type="commitment", entity_id="ent-c1", rank=0,
                      reason="open, oldest activity",
                      fields={"text": "Send the pricing deck"},
                      first_seen="2026-08-01T09:00:00+00:00",
                      last_seen="2026-08-13T09:00:00+00:00"),
            EntityRef(entity_type="person", entity_id="ent-p1", rank=1,
                      reason="most recent contact",
                      fields={"name": "Ajitesh Kumar", "email": "ajitesh@example.com"},
                      first_seen=None, last_seen=None),
        ],
        citations={
            "ent-c1": Citation(entity_id="ent-c1", memory_ids=["mem-1", "mem-2"],
                               signal_ids=["sig-1"]),
            "ent-p1": Citation(entity_id="ent-p1", memory_ids=["mem-3"], signal_ids=[]),
        },
    )


def test_scoped_view_maps_to_grounding_with_lineage_preserved():
    """1-3: EntityRef → Grounding, and citations survive as `lineage` rather
    than being discarded."""
    groundings = R._scoped_view_to_groundings(_view_with_entities())

    assert [g.kind for g in groundings] == ["state", "state"]
    commitment, person = groundings

    # source is the State entity_id — deterministic and org-scoped.
    assert commitment.source == "ent-c1"
    # The projection's own `reason` is carried, so the LLM sees WHY it was picked.
    assert "Send the pricing deck" in commitment.text
    assert "open, oldest activity" in commitment.text
    assert commitment.occurred_at == datetime(2026, 8, 13, 9, 0, tzinfo=timezone.utc)
    # PROVENANCE: memory_ids and signal_ids both survive, self-describing.
    assert commitment.lineage == ("memory:mem-1", "memory:mem-2", "signal:sig-1")

    # Label falls back through the normalised field names, never a connector field.
    assert "Ajitesh Kumar" in person.text
    assert person.lineage == ("memory:mem-3",)
    assert person.occurred_at is None  # last_seen absent → no invented date


def test_scoped_view_grounding_never_leaks_the_raw_field_dict():
    """The LLM gets a bounded line per entity, not a State record dump."""
    from navis.state.v2.scope import Citation, EntityRef, ScopedView

    view = ScopedView(
        intent_name="morning_brief", as_of=_AS_OF, state_watermark=None,
        policy_version=None, counts={},
        entities=[EntityRef(entity_type="person", entity_id="e1", rank=0,
                            reason="r", fields={"name": "N" * 500, "secret": "do-not-leak"},
                            first_seen=None, last_seen=None)],
        citations={"e1": Citation(entity_id="e1", memory_ids=[], signal_ids=[])},
    )
    g = R._scoped_view_to_groundings(view)[0]

    assert "do-not-leak" not in g.text
    assert len(g.text) < 200  # bounded, not the whole fields dict


def test_empty_scoped_view_yields_no_groundings():
    """7: an org with nothing scoped must not break the chat path."""
    assert R._scoped_view_to_groundings(_empty_view()) == []


# ---- Phase 3E.2.1: State groundings must carry the fields State already holds -
#
# 3E.2 proved the State groundings' problem was not their ORDER. It was that
# "{entity_type}: {label} ({reason})" spent a prompt line on a pointer while the
# fold had already normalised the answer onto `EntityRef.fields`. These pin the
# rendering defect itself, on the shapes the real org actually produces.


def _one_entity(entity_type, fields, *, reason="r", entity_id="e1"):
    from navis.state.v2.scope import Citation, EntityRef, ScopedView

    view = ScopedView(
        intent_name="morning_brief", as_of=_AS_OF, state_watermark=None,
        policy_version=None, counts={},
        entities=[EntityRef(entity_type=entity_type, entity_id=entity_id, rank=0,
                            reason=reason, fields=fields,
                            first_seen=None, last_seen=None)],
        citations={entity_id: Citation(entity_id=entity_id, memory_ids=[], signal_ids=[])},
    )
    return R._scoped_view_to_groundings(view)[0].text


def test_a_commitment_is_labelled_by_its_promise_not_the_email_subject():
    """THE DEFECT. `_LABEL_FIELDS_DEFAULT` reaches `subject` before `text`, so
    the org's one open promise reached the model titled with the mail header it
    was extracted from — and the answer to "what did we promise" was the subject
    line. The subject is still rendered, as `re`."""
    text = _one_entity("commitment", {
        "text": "We'll connect NavisLabs to your workflow",
        "subject": "Invitation from an unknown sender: Rahul and Abhi @ Mon 17 Aug",
    })

    assert text.startswith("commitment: We'll connect NavisLabs to your workflow")
    assert "re Invitation from an unknown sender" in text


def test_a_commitment_carries_its_owner_so_who_owns_this_is_answerable():
    """`speaker_email` is the commitment's owner in State's own terms — folds.py
    builds the `owner` link from exactly this address — and it was dropped."""
    text = _one_entity("commitment", {
        "text": "Send the pricing deck",
        "speaker_name": "Abhi",
        "speaker_email": "abhi@example.com",
        "status": "open",
        "due_at": "2026-08-20T00:00:00+00:00",
    })

    assert "owner Abhi <abhi@example.com>" in text
    assert "status open" in text
    assert "due 2026-08-20T00:00:00+00:00" in text


def test_a_meeting_carries_attendees_and_times():
    """"which people appear across both meetings and email" is a join on
    attendees; the meeting reached the model without any."""
    text = _one_entity("meeting", {
        "title": "Weekly sync",
        "starts_at": "2026-08-13T08:00:00+05:30",
        "attendees": ["a@example.com", "b@example.com"],
    })

    assert "starts 2026-08-13T08:00:00+05:30" in text
    assert "attendees a@example.com, b@example.com" in text


def test_a_thread_carries_its_counterparty_and_direction():
    """`direction == "inbound"` means they spoke last and nobody replied — the
    fact behind every "still open / unanswered" question."""
    text = _one_entity("thread", {
        "subject": "Pricing",
        "from_name": "Ajitesh",
        "from_email": "ajitesh@example.com",
        "direction": "inbound",
        "category": "personal",
        "participants": ["ajitesh@example.com", "founder@navis.in"],
    })

    assert "last message from Ajitesh <ajitesh@example.com>" in text
    assert "direction inbound" in text
    assert "participants ajitesh@example.com, founder@navis.in" in text


def test_only_whitelisted_fields_render_and_absent_ones_invent_nothing():
    """Whitelist, never the dict: `fields` is whatever the connector's payload
    carried. And a field State never established renders as nothing — a thin
    entity stays honestly thin rather than being padded with defaults."""
    text = _one_entity("commitment", {
        "text": "Ship the API",
        "thread_id": "1a00ab8a4f1a3852",       # internal id, not for the prompt
        "rfc_message_id": "<abc@mtasv.net>",   # ditto
    })

    assert text == "commitment: Ship the API (r)"
    for absent in ("owner", "status", "due", "promised", "None", "null"):
        assert absent not in text


def test_list_fields_are_capped_with_the_remainder_counted():
    """Bounded: an 80-attendee meeting costs the same as a 2-attendee one, and
    the count is read off the list rather than estimated."""
    text = _one_entity("meeting", {
        "title": "All hands",
        "attendees": [f"p{i}@example.com" for i in range(10)],
    })

    assert "p3@example.com +6 more" in text
    assert "p4@example.com" not in text


def test_a_rendered_entity_stays_bounded_however_fat_the_fields_are():
    """Deterministic AND bounded — no single entity may become the prompt."""
    text = _one_entity("thread", {
        "subject": "S" * 500,
        "from_name": "N" * 500,
        "from_email": "E" * 500,
        "participants": ["P" * 500] * 50,
    })

    assert len(text) <= R._CONTEXT_TEXT_MAX


def test_rendering_a_scoped_entity_is_deterministic():
    """Same entity in, same line out — no clock, no model, no set iteration."""
    fields = {"text": "Ship it", "speaker_name": "Abhi",
              "speaker_email": "abhi@example.com", "status": "open"}

    assert len({_one_entity("commitment", dict(fields)) for _ in range(5)}) == 1


def test_a_reasoning_signal_names_its_subject():
    """Four RELATIONSHIP_COOLING signals reached the model as four identical
    lines because the subject lived in `entity_id`, a uuid. It is resolved
    through the SAME label function the ScopedView path uses."""
    state = SimpleNamespace(entities={"relationship": {
        "rel-1": {"fields": {"person_email": "quiet@example.com"}},
    }})
    signal = SimpleNamespace(entity_type="relationship", entity_id="rel-1",
                             title="Relationship cooling",
                             why="11 days since the last interaction.")

    text = R._reasoning_text(state, signal)

    assert text == ("Relationship cooling (relationship: quiet@example.com) "
                    "— 11 days since the last interaction.")


def test_a_reasoning_signal_with_no_resolvable_subject_renders_as_before():
    """A subject-less signal, one State no longer holds, and one already spelled
    out in its own title must all degrade to `title — why` — never to a uuid."""
    state = SimpleNamespace(entities={"commitment": {
        "c-1": {"fields": {"text": "Send the deck"}},
    }})

    def _sig(entity_type, entity_id, title="Priority load"):
        return SimpleNamespace(entity_type=entity_type, entity_id=entity_id,
                               title=title, why="Because.")

    assert R._reasoning_text(state, _sig(None, None)) == "Priority load — Because."
    assert R._reasoning_text(state, _sig("commitment", "gone")) == "Priority load — Because."
    # Already inlined by the engine — naming it again would just repeat it.
    already = _sig("commitment", "c-1", title="Commitment slipping: Send the deck")
    assert R._reasoning_text(state, already) == "Commitment slipping: Send the deck — Because."


async def test_context_and_reasoning_groundings_both_reach_the_llm(monkeypatch):
    """4-5: State's two projections arrive together through `extra=`, with
    Context ordered before Reasoning."""
    from navis.ml.rag.grounding import Grounding, GroundedContext

    _stub_state(monkeypatch, org_seen=[], signals=[_fake_signal()],
                view=_view_with_entities())
    captured: dict = {}

    class _Builder:
        def __init__(self, db):
            pass

        async def build(self, org_id, question, **kw):
            captured["extra"] = kw.get("extra")
            memory = [Grounding(kind="memory", source="mem-x", text="An email.")]
            return GroundedContext(org_id=str(org_id), question=question,
                                   facts=[*memory, *(kw.get("extra") or [])])

    async def _generator(org_id, context):
        captured["kinds"] = [f.kind for f in context.facts]
        return SimpleNamespace(text="Focus on the deck.", sources=["ent-c1"])

    monkeypatch.setattr(R, "ContextBuilder", _Builder)
    monkeypatch.setattr(R, "recommendation", _generator)

    resp = await R.chat(_body("What should I focus on today?"), _Principal(), _DB())

    kinds = [g.kind for g in captured["extra"]]
    # CONCLUSIONS FIRST. This previously asserted state-before-reasoning, which
    # put 26 ScopedView entities ahead of the 5 deterministic signals: measured
    # on the real pipeline, "which commitments are slipping?" was answered from
    # email snippets while the `commitment_slipping` signal sat near the end of
    # the context. Reasoning leads, and the State that explains it follows.
    assert kinds == ["reasoning", "state", "state"], "conclusions must precede state"
    # This fake ContextBuilder concatenates rather than ranking; the real one
    # sorts by kind (see test_context_builder_ranks_conclusions_first).
    assert captured["kinds"] == ["memory", "reasoning", "state", "state"]
    _assert_not_memory_fallback(resp)


async def test_state_groundings_do_not_suppress_the_workspace_fallback(monkeypatch):
    """6: `state` is a DERIVED kind — like `reasoning`, it must not be mistaken
    for retrieval, or raw Calendar/Gmail evidence disappears permanently."""
    from navis.ml.rag.grounding import Grounding, GroundedContext

    _stub_state(monkeypatch, org_seen=[], signals=[], view=_view_with_entities())
    called: dict = {}

    class _EmptyMemoryBuilder:
        def __init__(self, db):
            pass

        async def build(self, org_id, question, **kw):
            return GroundedContext(org_id=str(org_id), question=question,
                                   facts=list(kw.get("extra") or []))

    async def _fallback(db, org_id, question, *, now=None):
        called["fallback"] = True
        return GroundedContext(org_id=str(org_id), question=question,
                               facts=[Grounding(kind="calendar", source="signal:abc",
                                                text="Calendar event: Investor sync")])

    async def _generator(org_id, context):
        called["kinds"] = [f.kind for f in context.facts]
        return SimpleNamespace(text="Investor sync first.", sources=["signal:abc"])

    monkeypatch.setattr(R, "ContextBuilder", _EmptyMemoryBuilder)
    monkeypatch.setattr(R, "build_workspace_context", _fallback)
    monkeypatch.setattr(R, "recommendation", _generator)

    await R.chat(_body("What should I focus on today?"), _Principal(), _DB())

    assert called.get("fallback") is True, "state groundings suppressed the fallback"
    assert called["kinds"] == ["calendar", "state", "state"]


def test_state_grounding_cannot_mix_organizations():
    """9: the adapter takes an already-org-scoped ScopedView, so there is no
    parameter through which another tenant's entities could arrive."""
    import inspect

    params = list(inspect.signature(R._scoped_view_to_groundings).parameters)
    assert params == ["view"], "adapter must not take an org_id it could mismatch"
    # Every grounding's source is an entity_id from that one view.
    view = _view_with_entities()
    sources = {g.source for g in R._scoped_view_to_groundings(view)}
    assert sources == {e.entity_id for e in view.entities}


def test_ml_does_not_import_state_or_reasoning():
    """11-12: the boundary Phase 3A must not weaken."""
    import ast
    import pathlib

    root = pathlib.Path(R.__file__).resolve().parents[2] / "navis" / "ml"
    forbidden = {"state", "reasoning", "decisions", "execution"}
    violations = []
    for py in root.rglob("*.py"):
        for node in ast.walk(ast.parse(py.read_text(encoding="utf-8"), filename=str(py))):
            mods = []
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                mods = [node.module]
            for m in mods:
                parts = m.split(".")
                if len(parts) >= 2 and parts[0] == "navis" and parts[1] in forbidden:
                    violations.append(f"ml/{py.name} imports {m}")
    assert not violations, violations


def test_composed_fallback_context_respects_the_token_budget():
    """The workspace fallback builds its context OUTSIDE ContextBuilder, so the
    builder's budget never applied to it. Adding State (22) + Reasoning (10) to
    its ~12 facts pushed the real prompt to ~1777 tokens, past the 1500 the
    retrieval branch enforces. Composition must bound it and report honestly."""
    from navis.ml.rag.context_builder import estimate_tokens
    from navis.ml.rag.grounding import Grounding, GroundedContext

    base = GroundedContext(
        org_id=str(ORG), question="What should I focus on today?",
        facts=[Grounding(kind="gmail", source=f"signal:{i}", text="word " * 60)
               for i in range(12)],
        token_estimate=0,
    )
    extra = [Grounding(kind="state", source=f"ent-{i}", text="word " * 60)
             for i in range(40)]

    composed = R._compose_within_budget(base, extra)

    # The budget now costs what is actually SENT. Both branches account the
    # RENDERED line per fact — index, date, `(kind:source)` citation and all —
    # so the declared estimate tracks the real prompt instead of undercounting
    # it by the ~a-third the citation prefix adds. (This assertion previously
    # pinned that undercount as the contract.)
    rendered = estimate_tokens(composed.render())
    assert composed.token_estimate <= R._CONTEXT_TOKEN_BUDGET
    # Per-line rounding in `estimate_tokens` (floor + max(1,...) per call) makes
    # the sum differ from a whole-string estimate by a few tokens per fact; what
    # matters is that the gap is now noise rather than a systematic shortfall.
    assert abs(composed.token_estimate - rendered) <= len(composed.facts) + 5
    assert len(composed.facts) < len(base.facts) + len(extra), "tail must be dropped"
    assert composed.facts[:12] == base.facts, "retrieved evidence is never dropped first"
    # token_estimate is recomputed, not carried over from `base`.
    assert composed.token_estimate > 0


def test_compose_keeps_everything_when_it_fits():
    from navis.ml.rag.grounding import Grounding, GroundedContext

    base = GroundedContext(org_id=str(ORG), question="q",
                           facts=[Grounding(kind="gmail", source="s1", text="short")])
    extra = [Grounding(kind="state", source="e1", text="also short")]

    composed = R._compose_within_budget(base, extra)

    assert [f.kind for f in composed.facts] == ["gmail", "state"]


# ── Entitlement gate (TASK 6.8) ─────────────────────────────────────────────
#
# `/chat` is the most expensive request in the product and was the only one
# with no capability dependency — authentication only. The capability gate is
# also the sole enforcer of organization LIFECYCLE (see
# `authz.principal_cache_evict_org` and `entitlements._decide`), so a SUSPENDED
# or CHURNED tenant kept full LLM chat after every other surface had stopped.
# These two tests lock the gate and the capability it names; a refactor that
# drops either silently restores the bypass.

def test_chat_is_capability_gated_on_memory():
    import inspect

    from navis.organization.contract import Capability

    dep = inspect.signature(R.chat).parameters["p"].default
    closed = [c.cell_contents for c in (dep.dependency.__closure__ or ())]
    assert any(c is Capability.MEMORY for c in closed), (
        "/chat must be gated on Capability.MEMORY — without a capability "
        "dependency a suspended organization keeps unlimited LLM access"
    )


def test_memory_capability_is_granted_by_every_plan():
    """The gate above must not cost an entitled organization anything.

    MEMORY is in FREE, so adding the dependency denies only suspended/churned
    tenants, an explicit revocation and the `memory` kill switch. If a future
    plan drops MEMORY, chat goes dark for that tier and this test says so.
    """
    from navis.organization.contract import PLAN_CAPABILITIES, Capability

    for plan, caps in PLAN_CAPABILITIES.items():
        assert Capability.MEMORY in caps, f"{plan} would lose /chat"


# ══════════════════════════════════════════════════════════════════════════
# ASK ORCHESTRATION CONTRACT (TASK 7.5)
#
# Ask Navis is the entry point ABOVE the pipeline, not another layer in it. A
# question must be resolved to WHO it is about through the existing identity
# graph, answered from the organization's own persisted conclusions where it has
# them, and returned with lineage a reader can follow back to evidence.
#
# These tests pin the orchestration, not the prose. None of them assert that an
# answer "sounds better" — they assert which layers ran, which ids survived, and
# what the endpoint refuses to invent.
# ══════════════════════════════════════════════════════════════════════════

from navis.identity_resolution.search import PersonMatch, PersonSearchResult  # noqa: E402

_ABHISHEK = PersonMatch(
    canonical_id="c-abhishek-1",
    display_names=("Abhishek Meena",),
    emails=("abhishek@acme.com",),
    alias_ids=("a-1", "a-2"),
)
_ABHISHEK_TWO = PersonMatch(
    canonical_id="c-abhishek-2",
    display_names=("Abhishek Rao",),
    emails=("abhishek.rao@other.com",),
    alias_ids=("a-3",),
)


def _stub_persons(monkeypatch, matches, *, seen: list | None = None):
    """Stand in for the identity graph. Records (org_id, query) it was asked."""

    async def _search(db, org_id, query):
        if seen is not None:
            seen.append((org_id, query))
        return PersonSearchResult(query=query, matches=tuple(matches))

    monkeypatch.setattr(R, "search_persons", _search)


def _capture_builder(monkeypatch, captured: dict, facts=None):
    """A ContextBuilder that records every argument the orchestration passed."""
    from navis.ml.rag.grounding import Grounding, GroundedContext

    default = [Grounding(kind="memory", source="mem-1", text="Pricing thread.")]

    class _Builder:
        def __init__(self, db):
            pass

        async def build(self, org_id, question, **kw):
            captured["org_id"] = org_id
            captured["question"] = question
            captured["retrieval_query"] = kw.get("retrieval_query")
            captured["extra"] = kw.get("extra")
            return GroundedContext(
                org_id=str(org_id), question=question,
                facts=[*(default if facts is None else facts), *(kw.get("extra") or [])],
            )

    monkeypatch.setattr(R, "ContextBuilder", _Builder)


# 1. Person question -> identity + context + grounded answer.
async def test_person_question_resolves_identity_and_scopes_retrieval(monkeypatch):
    """The question's subject goes through the EXISTING identity graph, and the
    resolved aliases are what retrieval searches for.

    The name previously reached exactly one component with any notion of who a
    person is — the embedding model — so "why does X matter" was answered by
    cosine similarity over a corpus that holds X's ADDRESS far more often than
    their display name.
    """
    seen: list = []
    _stub_persons(monkeypatch, [_ABHISHEK], seen=seen)
    _stub_state(monkeypatch, org_seen=[], signals=[])
    captured: dict = {}
    _capture_builder(monkeypatch, captured)

    async def _gen(org_id, context):
        return SimpleNamespace(text="He owns the pricing thread.", sources=["mem-1"])

    monkeypatch.setattr(R, "explanation", _gen)

    resp = await R.chat(
        _body("Why does Abhishek matter to my organization?"), _Principal(), _DB()
    )

    # The identity graph was asked, with the AUTHENTICATED org.
    assert seen and seen[0][0] == ORG
    assert "abhishek" in seen[0][1].lower()
    # The model still sees the founder's own words...
    assert captured["question"] == "Why does Abhishek matter to my organization?"
    # ...while retrieval searches for the resolved person.
    assert "abhishek@acme.com" in captured["retrieval_query"]
    assert "Abhishek Meena" in captured["retrieval_query"]
    # The canonical entity is published so a client can navigate to the person.
    assert resp.entity is not None
    assert resp.entity.canonical_id == "c-abhishek-1"
    assert resp.entity.email == "abhishek@acme.com"


# 2. Ambiguous identity -> ambiguity PRESERVED, never narrowed.
async def test_ambiguous_person_is_asked_about_not_guessed(monkeypatch):
    """Two people share the name: the endpoint must ask, not pick.

    Picking the first would attribute one person's evidence to another — the
    worst thing this endpoint could do — and it would do it invisibly.
    """
    _stub_persons(monkeypatch, [_ABHISHEK, _ABHISHEK_TWO])
    called: list = []

    async def _never(*a, **kw):
        called.append(1)
        raise AssertionError("must not retrieve before the subject is settled")

    monkeypatch.setattr(R, "_grounded_answer", _never)

    resp = await R.chat(_body("Why does Abhishek matter?"), _Principal(), _DB())

    assert called == []
    assert resp.clarification is not None
    assert resp.clarification.field == "person_disambiguation"
    # BOTH candidates are offered, with their canonical ids intact.
    ids = {o["canonical_id"] for o in resp.clarification.options}
    assert ids == {"c-abhishek-1", "c-abhishek-2"}
    # No entity is claimed while the subject is unresolved.
    assert resp.entity is None


# 3. Insufficient evidence -> uncertainty preserved.
async def test_no_evidence_says_so_instead_of_inventing_importance(monkeypatch):
    """No memory, no workspace, no state => an honest refusal, and NO citations.

    "Never manufacture importance" has to hold at the point where there is
    nothing to be important about.
    """
    from navis.ml.rag.grounding import GroundedContext

    _stub_persons(monkeypatch, [])
    _stub_state(monkeypatch, org_seen=[], signals=[])

    class _Empty:
        def __init__(self, db):
            pass

        async def build(self, org_id, question, **kw):
            return GroundedContext(org_id=str(org_id), question=question, facts=[])

    async def _no_workspace(db, org_id, question, *, now=None):
        return GroundedContext(org_id=str(org_id), question=question, facts=[])

    async def _no_sources(db, org_id):
        return []

    monkeypatch.setattr(R, "ContextBuilder", _Empty)
    monkeypatch.setattr(R, "build_workspace_context", _no_workspace)
    monkeypatch.setattr(R, "connected_source_labels", _no_sources)
    monkeypatch.setattr(
        R, "explanation",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("LLM must not be called")),
    )

    resp = await R.chat(_body("Why does Zephyrine matter?"), _Principal(), _DB())

    assert "can't answer that yet" in resp.content
    assert resp.citations == []
    assert resp.entity is None


# 4. Multi-source evidence -> the provider is never fabricated.
async def test_citation_types_follow_the_producing_layer_not_a_guess(monkeypatch):
    """Each citation reports the layer that produced it.

    Reasoning and State are the org's current understanding ("state"); a
    persisted insight and an outcome aggregate are neither recall nor state and
    get their own values. Nothing is stamped with a provider that did not
    produce it — a `gmail`-kind fallback fact stays recall, and a derived
    conclusion is never relabelled as an email.
    """
    from navis.ml.rag.grounding import Grounding, GroundedContext

    _stub_persons(monkeypatch, [])
    _stub_state(monkeypatch, org_seen=[], signals=[])
    facts = [
        Grounding(kind="memory", source="mem-1", text="A remembered email."),
        Grounding(kind="gmail", source="signal:sig-1", text="A raw Gmail signal."),
        Grounding(kind="state", source="ent-1", text="A folded entity.",
                  lineage=("memory:m9", "signal:s9")),
        Grounding(kind="reasoning", source="rs-1", text="A conclusion."),
        Grounding(kind="intelligence", source="insight:i-1", text="An insight.",
                  lineage=("reasoning_signal:rs-1",)),
        Grounding(kind="learning", source="outcome_surface:world", text="An aggregate."),
    ]
    sources = [f.source for f in facts]

    class _Builder:
        def __init__(self, db):
            pass

        async def build(self, org_id, question, **kw):
            return GroundedContext(org_id=str(org_id), question=question, facts=facts)

    async def _gen(org_id, context):
        return SimpleNamespace(text="Answer.", sources=sources)

    monkeypatch.setattr(R, "ContextBuilder", _Builder)
    monkeypatch.setattr(R, "explanation", _gen)

    resp = await R.chat(_body("Where do we stand?"), _Principal(), _DB())

    by_source = {c.source: c.type for c in resp.citations}
    assert by_source["mem-1"] == "episodic"
    assert by_source["signal:sig-1"] == "episodic"
    assert by_source["ent-1"] == "state"
    assert by_source["rs-1"] == "state"
    assert by_source["insight:i-1"] == "intelligence"


# 5. Reasoning / intelligence lineage survives to the wire.
async def test_derived_citations_carry_their_drill_down(monkeypatch):
    """A ScopedView entity is a fold over many records; an insight cites the
    signals behind it. Both were computed and then thrown away, leaving every
    derived citation a dead end for "what did Navis use to reach this?".
    """
    from navis.ml.rag.grounding import Grounding, GroundedContext

    _stub_persons(monkeypatch, [])
    _stub_state(monkeypatch, org_seen=[], signals=[])
    facts = [
        Grounding(kind="state", source="ent-1", text="Folded.",
                  lineage=("memory:m1", "memory:m2", "signal:s1")),
        Grounding(kind="intelligence", source="insight:i-1", text="Concluded.",
                  lineage=("state_watermark:w-7", "reasoning_signal:rs-9")),
        Grounding(kind="memory", source="mem-1", text="Raw."),
    ]

    class _Builder:
        def __init__(self, db):
            pass

        async def build(self, org_id, question, **kw):
            return GroundedContext(org_id=str(org_id), question=question, facts=facts)

    async def _gen(org_id, context):
        return SimpleNamespace(text="A.", sources=["ent-1", "insight:i-1", "mem-1"])

    monkeypatch.setattr(R, "ContextBuilder", _Builder)
    monkeypatch.setattr(R, "explanation", _gen)

    resp = await R.chat(_body("Where do we stand?"), _Principal(), _DB())
    lineage = {c.source: c.lineage for c in resp.citations}

    assert lineage["ent-1"] == ["memory:m1", "memory:m2", "signal:s1"]
    assert lineage["insight:i-1"] == ["state_watermark:w-7", "reasoning_signal:rs-9"]
    # A memory record IS its own lowest level — nothing is synthesised for it.
    assert lineage["mem-1"] == []
    # And the snapshot the whole answer describes is published.
    assert resp.stateWatermark == str(_WATERMARK)


# 6. Execution-capable request still flows through the existing path.
async def test_execution_request_still_goes_through_execution_service(monkeypatch):
    """Entity resolution must not divert an action route. Scheduling still ends
    in ONE ExecutionRequest, still REQUESTED, still awaiting human approval."""
    from navis.reasoning.scheduling.contract import (
        MeetingSlot,
        ResolvedAttendee,
        ScheduledProposal,
    )

    _stub_persons(monkeypatch, [_ABHISHEK])
    monkeypatch.setattr(R, "ExecutionService", _ExecutionService)

    class _Agent:
        def __init__(self, db, **kw):
            pass

        async def propose(self, ctx):
            return ScheduledProposal(
                attendee=ResolvedAttendee(
                    email="abhishek@acme.com", source="person_search_unique"
                ),
                slot=MeetingSlot(
                    start=datetime(2026, 8, 22, 10, 0, tzinfo=timezone.utc),
                    end=datetime(2026, 8, 22, 10, 30, tzinfo=timezone.utc),
                    time_zone="UTC",
                    duration_minutes=30,
                ),
                title="Meeting with Abhishek",
            )

    monkeypatch.setattr(R, "SchedulingAgent", _Agent)

    resp = await R.chat(
        _body("Schedule a meeting with Abhishek tomorrow"), _Principal(), _DB()
    )

    assert len(_ExecutionService.created) == 1
    assert _ExecutionService.created[0]["action_type"] == "schedule_meeting"
    chip = [c for c in resp.actionProposals if c.channel == "schedule_meeting"]
    assert chip and chip[0].execution_status == "requested"


# 7. Tenant isolation across every retrieval the orchestration performs.
async def test_every_ask_lookup_uses_the_authenticated_org(monkeypatch):
    """Identity, State and retrieval all receive the PRINCIPAL's org_id. No
    request field can redirect any of them."""
    seen_persons: list = []
    org_seen: list = []
    _stub_persons(monkeypatch, [_ABHISHEK], seen=seen_persons)
    _stub_state(monkeypatch, org_seen=org_seen, signals=[])
    captured: dict = {}
    _capture_builder(monkeypatch, captured)
    monkeypatch.setattr(
        R, "explanation",
        lambda org_id, ctx: _async_value(SimpleNamespace(text="A.", sources=["mem-1"])),
    )

    # `user_id` is a legacy client field and is ignored; there is no org field
    # on ChatRequest at all, so there is nothing to redirect with.
    await R.chat(
        _body("Why does Abhishek matter?", user_id="not-a-real-user"),
        _Principal(), _DB(),
    )

    assert [o for o, _q in seen_persons] == [ORG]
    assert org_seen == [ORG]
    assert captured["org_id"] == ORG


def _async_value(v):
    async def _c():
        return v
    return _c()


# 8. Entitlement enforcement.
def test_chat_answers_are_gated_on_memory_capability():
    """Reads are gated on MEMORY — the capability gate is the ONLY enforcer of
    organization lifecycle, so this is what makes a suspended org's chat stop."""
    import inspect

    from navis.organization.contract import Capability

    dep = inspect.signature(R.chat).parameters["p"].default
    closed = [c.cell_contents for c in (dep.dependency.__closure__ or ())]
    assert any(c is Capability.MEMORY for c in closed)


async def test_action_routes_refuse_an_org_without_execution(monkeypatch):
    """A FREE/PRO org (no EXECUTION) must not be handed an Approve button that
    can only ever 402, and must not accumulate rows nobody can approve.

    Refused BEFORE the agent runs, so no LLM slot-fill and no Calendar call is
    spent on an action that cannot be offered.
    """
    async def _denied(self, org_id, capability):
        return False

    monkeypatch.setattr(R.OrganizationService, "can_use", _denied)
    monkeypatch.setattr(R, "ExecutionService", _ExecutionService)

    async def _never(*a, **kw):
        raise AssertionError("the scheduling agent must not run")

    monkeypatch.setattr(R, "_scheduling_response", _never)

    resp = await R.chat(
        _body("Schedule a meeting with Abhishek tomorrow"), _Principal(), _DB()
    )

    assert _ExecutionService.created == []
    assert resp.actionProposals == []
    assert "can't act on it" in resp.content


# 9. No fabricated confidence / impact percentages.
def test_intelligence_facts_never_render_a_confidence_percentage():
    """An insight's `confidence` is a real stored float, but a number in a prompt
    is a number the model restates — and a restated confidence reads as a
    measurement of the ANSWER rather than of the insight. Rank is never converted
    into one either. Only counts the engine actually recorded may appear.
    """
    from navis.ml.rag.context_builder import intelligence_facts

    row = SimpleNamespace(
        insight_id="i-1", insight_type="signal_reliability", key="k",
        title="Investor replies slow after Thursday",
        summary="Threads opened late in the week wait longer.",
        sample_size=12, confidence=0.87,
        evidence={"supporting_reasoning_signal_ids": ["rs-1"]},
        supporting_learning_event_ids=["le-1"],
        learning_version=4, state_watermark="w-7",
        created_at=_AS_OF,
    )
    facts = intelligence_facts([row])

    assert len(facts) == 1
    text = facts[0].text
    assert "%" not in text
    assert "0.87" not in text and "87" not in text
    # The count IS a recorded fact and may be stated.
    assert "12 observations" in text
    # Lineage carries the ids instead, so the number stays available at its source.
    assert "reasoning_signal:rs-1" in facts[0].lineage
    assert "state_watermark:w-7" in facts[0].lineage
    assert facts[0].source == "insight:i-1"


def test_intelligence_facts_are_pure_and_deterministic():
    """Same rows in, byte-identical facts out, in the same order — a conclusion
    the model is shown has to be reproducible."""
    from navis.ml.rag.context_builder import intelligence_facts

    def _row(i, t):
        return SimpleNamespace(
            insight_id=f"i-{i}", insight_type=t, key=f"k{i}", title=f"T{i}",
            summary=f"S{i}", sample_size=0, confidence=0.0, evidence={},
            supporting_learning_event_ids=[], learning_version=1,
            state_watermark=None, created_at=None,
        )

    rows = [_row(2, "b"), _row(1, "a"), _row(3, "c")]
    a = intelligence_facts(rows)
    b = intelligence_facts(list(reversed(rows)))
    assert [f.source for f in a] == [f.source for f in b]
    assert [f.text for f in a] == [f.text for f in b]


def test_intelligence_is_a_conclusion_kind_and_leads_the_context():
    """Insights must reach the model AHEAD of the evidence they were drawn from —
    that is the difference between answering from the organization's conclusions
    and asking the LLM to re-derive them from raw memory."""
    from navis.ml.rag.context_builder import _CONCLUSION_KINDS, _compose
    from navis.ml.rag.grounding import Grounding

    assert "intelligence" in _CONCLUSION_KINDS
    ordered = _compose(
        [
            Grounding(kind="memory", source="m", text="raw"),
            Grounding(kind="intelligence", source="insight:i", text="concluded"),
        ],
        "anything",
    )
    assert [g.kind for g in ordered] == ["intelligence", "memory"]


# 10. Existing behaviour remains compatible.
async def test_a_question_naming_nobody_behaves_exactly_as_before(monkeypatch):
    """No resolvable subject => no entity, unscoped retrieval, same answer path.
    Entity resolution must be additive, never a new way for a turn to fail."""
    _stub_persons(monkeypatch, [])
    _stub_state(monkeypatch, org_seen=[], signals=[])
    captured: dict = {}
    _capture_builder(monkeypatch, captured)

    async def _gen(org_id, context):
        return SimpleNamespace(text="Two threads are waiting.", sources=["mem-1"])

    monkeypatch.setattr(R, "explanation", _gen)

    resp = await R.chat(_body("What is waiting on me?"), _Principal(), _DB())

    assert resp.entity is None
    # Retrieval was NOT enriched — the question stands on its own.
    assert captured["retrieval_query"] in (None, "What is waiting on me?")
    assert resp.content == "Two threads are waiting."
    assert resp.intent == "question"


async def test_identity_resolution_failure_degrades_instead_of_failing_the_turn(monkeypatch):
    """Identity resolution is enrichment: losing it costs the answer its entity
    scoping, never the answer."""
    async def _broken(db, org_id, query):
        raise RuntimeError("identity graph down")

    monkeypatch.setattr(R, "search_persons", _broken)
    _stub_state(monkeypatch, org_seen=[], signals=[])
    captured: dict = {}
    _capture_builder(monkeypatch, captured)

    async def _gen(org_id, context):
        return SimpleNamespace(text="Still answered.", sources=["mem-1"])

    monkeypatch.setattr(R, "explanation", _gen)

    resp = await R.chat(_body("Why does Abhishek matter?"), _Principal(), _DB())

    assert resp.content == "Still answered."
    assert resp.entity is None


# ══════════════════════════════════════════════════════════════════════════
# CONTEXT SUFFICIENCY (TASK 7.5.1)
#
# Retrieval finds text that reads like the question. It cannot find a fact whose
# text carries no distinguishing signal — measured on the live org, the ONE
# `relationship` record for the person a question was about reads "Received email
# from <addr>", lexically indistinguishable from 151 others, and scored below
# top-8 for every phrasing tried including one embedding the person's own
# aliases. The gate below notices that and fetches by id instead.
# ══════════════════════════════════════════════════════════════════════════


async def test_expansion_fires_when_retrieval_found_little_about_the_subject(monkeypatch):
    from navis.ml.rag.grounding import Grounding, GroundedContext

    _stub_persons(monkeypatch, [_ABHISHEK])
    _stub_state(monkeypatch, org_seen=[], signals=[])
    seen: dict = {}

    class _Builder:
        def __init__(self, db):
            pass

        async def build(self, org_id, question, **kw):
            seen["subject_terms"] = kw.get("subject_terms")
            # Retrieval found ONE fact naming the subject — below the threshold.
            retrieved = [
                Grounding(kind="memory", source="m1", text="Note about Abhishek Meena."),
                Grounding(kind="memory", source="m2", text="An unrelated invoice."),
            ]
            expand = kw.get("expand")
            extra = await expand(retrieved) if expand else []
            seen["expanded"] = extra
            return GroundedContext(
                org_id=str(org_id), question=question, facts=[*retrieved, *extra]
            )

    async def _records(db, org_id, entity_ids, *, per_entity_limit=4):
        seen["entity_ids"] = list(entity_ids)
        seen["org_id"] = org_id
        return [SimpleNamespace(
            memory_id=uuid.uuid4(), entity_type="relationship",
            entity_id=uuid.uuid4(), payload={"kind": "received", "counterparty": "x"},
            event_timestamp=_AS_OF, source_signal_event_id=SIGNAL_ID,
            source_integration_id=uuid.uuid4(), dedup_key="d",
        )]

    monkeypatch.setattr(R, "ContextBuilder", _Builder)
    monkeypatch.setattr(
        "navis.memory.lineage.records_for_entities", _records
    )
    monkeypatch.setattr(
        "navis.ml.embeddings.service.memory_text",
        lambda rec: "Received email from Abhishek Meena",
    )
    monkeypatch.setattr(
        R, "explanation",
        lambda org_id, ctx: _async_value(SimpleNamespace(text="A.", sources=["m1"])),
    )

    await R.chat(_body("Why does Abhishek matter?"), _Principal(), _DB())

    # The gate fired and fetched by DETERMINISTIC entity id, org-scoped.
    assert seen["org_id"] == ORG
    assert seen["entity_ids"], "expansion must compute entity ids from the aliases"
    assert len(seen["expanded"]) == 1
    # Composition was told who the question is about.
    assert "abhishek@acme.com" in seen["subject_terms"]


async def test_expansion_is_skipped_when_retrieval_already_answered(monkeypatch):
    """A question retrieval already answered costs no extra round trip."""
    from navis.ml.rag.grounding import Grounding, GroundedContext

    _stub_persons(monkeypatch, [_ABHISHEK])
    _stub_state(monkeypatch, org_seen=[], signals=[])
    fired: list = []

    class _Builder:
        def __init__(self, db):
            pass

        async def build(self, org_id, question, **kw):
            retrieved = [
                Grounding(kind="memory", source=f"m{i}",
                          text=f"Abhishek Meena fact {i}")
                for i in range(5)
            ]
            expand = kw.get("expand")
            extra = await expand(retrieved) if expand else []
            return GroundedContext(
                org_id=str(org_id), question=question, facts=[*retrieved, *extra]
            )

    async def _records(db, org_id, entity_ids, *, per_entity_limit=4):
        fired.append(1)
        return []

    monkeypatch.setattr(R, "ContextBuilder", _Builder)
    monkeypatch.setattr("navis.memory.lineage.records_for_entities", _records)
    monkeypatch.setattr(
        R, "explanation",
        lambda org_id, ctx: _async_value(SimpleNamespace(text="A.", sources=["m0"])),
    )

    await R.chat(_body("Why does Abhishek matter?"), _Principal(), _DB())

    assert fired == [], "5 subject facts is sufficient; no second hop"


async def test_expanded_facts_carry_real_provenance_and_no_invented_provider(monkeypatch):
    """An expanded fact is exactly as traceable as a retrieved one: real
    memory_id, real signal id, and an integration ONLY when the record recorded
    one. Nothing is inferred from the text."""
    from navis.ml.rag.grounding import Grounding

    mem_id, integ = uuid.uuid4(), uuid.uuid4()

    async def _records(db, org_id, entity_ids, *, per_entity_limit=4):
        return [
            SimpleNamespace(
                memory_id=mem_id, entity_type="relationship", entity_id=uuid.uuid4(),
                payload={}, event_timestamp=_AS_OF,
                source_signal_event_id=SIGNAL_ID, source_integration_id=integ,
                dedup_key="d1",
            ),
            # Same shape, but the record never recorded an account.
            SimpleNamespace(
                memory_id=uuid.uuid4(), entity_type="person", entity_id=uuid.uuid4(),
                payload={}, event_timestamp=None,
                source_signal_event_id=None, source_integration_id=None,
                dedup_key="d2",
            ),
        ]

    monkeypatch.setattr("navis.memory.lineage.records_for_entities", _records)
    monkeypatch.setattr(
        "navis.ml.embeddings.service.memory_text", lambda rec: f"fact {rec.dedup_key}"
    )

    subject = R._SubjectResolution(match=_ABHISHEK, query="Abhishek")
    out = await R._entity_expansion(_DB(), ORG, subject)

    assert [g.kind for g in out] == ["memory", "memory"]
    assert out[0].source == str(mem_id)
    assert f"signal:{SIGNAL_ID}" in out[0].lineage
    assert f"integration:{integ}" in out[0].lineage
    # No recorded account => NO integration claim. A provider is never guessed.
    assert not any(x.startswith("integration:") for x in out[1].lineage)
    assert not any(x.startswith("signal:") for x in out[1].lineage)


async def test_expansion_degrades_when_memory_is_unavailable(monkeypatch):
    async def _broken(db, org_id, entity_ids, *, per_entity_limit=4):
        raise RuntimeError("memory down")

    monkeypatch.setattr("navis.memory.lineage.records_for_entities", _broken)
    db = _DB()
    out = await R._entity_expansion(db, ORG, R._SubjectResolution(match=_ABHISHEK))
    assert out == []
    assert db.rollbacks == 1


async def test_a_person_with_no_email_alias_expands_to_nothing(monkeypatch):
    """Entity ids are computed from EMAIL natural keys (`memory.ids.entity_id`).
    A person the graph knows only by display name yields no ids, so we fetch
    nothing rather than guess at a key."""
    from navis.identity_resolution.search import PersonMatch

    called: list = []

    async def _records(db, org_id, entity_ids, *, per_entity_limit=4):
        called.append(list(entity_ids))
        return []

    monkeypatch.setattr("navis.memory.lineage.records_for_entities", _records)
    nameless = PersonMatch(
        canonical_id="c-x", display_names=("Someone",), emails=(), alias_ids=("a",)
    )

    out = await R._entity_expansion(
        _DB(), ORG, R._SubjectResolution(match=nameless)
    )

    assert out == []
    assert called == [], "no email => no id to compute => no query at all"


# ══════════════════════════════════════════════════════════════════════════
# EMAIL_SEARCH — explicit mail retrieval (TASK 7.5.4)
#
# Ask Navis owns no Gmail query and no Gmail client. This branch calls the SAME
# `signals.read_router.search_emails` the HTTP route calls, and it can never
# reach ExecutionService.
# ══════════════════════════════════════════════════════════════════════════


def _stub_search_emails(monkeypatch, seen: dict, *, items=None, connected=True):
    async def _search(db, *, org_id, senders=(), text_match=None, limit=10):
        seen["org_id"] = org_id
        seen["senders"] = tuple(senders)
        seen["text_match"] = text_match
        seen["limit"] = limit
        rows = items if items is not None else [
            {
                "event_id": "11111111-2222-3333-4444-555555555555",
                "external_id": "gm-1",
                "thread_id": "t-1",
                "subject": "Term sheet draft",
                "from_email": "abhishek@acme.com",
                "from_name": "Abhishek Meena",
                "to_emails": ["founder@navis.ai"],
                "snippet": "Sending the draft over",
                "received_at": "2026-08-20T09:00:00+00:00",
                "has_attachments": False,
                "label_ids": [],
            }
        ]
        return {"source": "gmail", "connected": connected,
                "count": len(rows), "items": rows}

    monkeypatch.setattr("navis.signals.read_router.search_emails", _search)


async def test_email_search_uses_the_shared_gmail_reader(monkeypatch):
    """The branch delegates; it does not build a query of its own."""
    _stub_persons(monkeypatch, [_ABHISHEK])
    seen: dict = {}
    _stub_search_emails(monkeypatch, seen)

    resp = await R.chat(_body("Find the email from Abhishek"), _Principal(), _DB())

    # Tenant came from the Principal, and the sender filter is the CANONICAL
    # address the identity graph resolved — not the raw word the user typed.
    assert seen["org_id"] == ORG
    assert seen["senders"] == ("abhishek@acme.com",)
    assert "Term sheet draft" in resp.content
    assert resp.entity is not None and resp.entity.canonical_id == "c-abhishek-1"


async def test_email_search_preserves_signal_provenance(monkeypatch):
    """A retrieved email cites the signal_event it actually is."""
    _stub_persons(monkeypatch, [_ABHISHEK])
    _stub_search_emails(monkeypatch, {})

    resp = await R.chat(_body("Find the email from Abhishek"), _Principal(), _DB())

    assert len(resp.citations) == 1
    c = resp.citations[0]
    assert c.source == "signal:11111111-2222-3333-4444-555555555555"
    assert c.type == "episodic"
    assert c.detail == "Term sheet draft"


async def test_email_search_never_leaks_credentials_or_bodies_into_the_answer(monkeypatch):
    """Nothing token-shaped reaches the response, and no LLM is called at all."""
    _stub_persons(monkeypatch, [_ABHISHEK])
    _stub_search_emails(monkeypatch, {})
    monkeypatch.setattr(
        R, "explanation",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("no LLM on a retrieval")),
    )

    resp = await R.chat(_body("Find the email from Abhishek"), _Principal(), _DB())

    blob = resp.model_dump_json().lower()
    for forbidden in ("access_token", "refresh_token", "bearer ", "client_secret"):
        assert forbidden not in blob


async def test_email_search_narrows_by_subject_when_the_user_says_about(monkeypatch):
    _stub_persons(monkeypatch, [])
    seen: dict = {}
    _stub_search_emails(monkeypatch, seen)

    await R.chat(_body("Find emails about the term sheet"), _Principal(), _DB())

    assert seen["text_match"] == "the term sheet"
    assert seen["senders"] == ()


async def test_email_search_respects_the_gmail_entitlement(monkeypatch):
    """Chat is gated on MEMORY, which every plan grants. Reading mail must still
    require GMAIL, or the chat box becomes a way around a revoked capability."""
    _stub_persons(monkeypatch, [_ABHISHEK])
    called: list = []

    async def _search(db, **kw):
        called.append(1)
        return {"source": "gmail", "connected": True, "count": 0, "items": []}

    monkeypatch.setattr("navis.signals.read_router.search_emails", _search)

    from navis.organization.contract import Capability as _Cap

    async def _denied(self, org_id, capability):
        return capability is not _Cap.GMAIL

    monkeypatch.setattr(R.OrganizationService, "can_use", _denied)

    resp = await R.chat(_body("Find the email from Abhishek"), _Principal(), _DB())

    assert called == [], "must not read mail without the capability"
    assert "isn't enabled" in resp.content


async def test_email_search_says_so_when_gmail_is_not_connected(monkeypatch):
    _stub_persons(monkeypatch, [_ABHISHEK])
    _stub_search_emails(monkeypatch, {}, connected=False)

    resp = await R.chat(_body("Find the email from Abhishek"), _Principal(), _DB())

    assert "isn't connected" in resp.content
    assert resp.citations == []


async def test_email_search_empty_result_names_what_was_searched(monkeypatch):
    """"Nothing" must be a fact about the query, never a silent widening."""
    _stub_persons(monkeypatch, [_ABHISHEK])
    _stub_search_emails(monkeypatch, {}, items=[])

    resp = await R.chat(_body("Find the email from Abhishek"), _Principal(), _DB())

    assert "don't have any captured email" in resp.content
    assert "Abhishek Meena" in resp.content
    assert resp.citations == []


async def test_email_search_asks_which_person_when_ambiguous(monkeypatch):
    _stub_persons(monkeypatch, [_ABHISHEK, _ABHISHEK_TWO])
    called: list = []

    async def _search(db, **kw):
        called.append(1)
        return {"source": "gmail", "connected": True, "count": 0, "items": []}

    monkeypatch.setattr("navis.signals.read_router.search_emails", _search)

    resp = await R.chat(_body("Find the email from Abhishek"), _Principal(), _DB())

    assert called == [], "never guess whose mail to read"
    assert resp.clarification is not None
    assert resp.clarification.field == "person_disambiguation"


async def test_email_search_never_reaches_execution(monkeypatch):
    """A read is a read. No ExecutionRequest, ever."""
    _stub_persons(monkeypatch, [_ABHISHEK])
    _stub_search_emails(monkeypatch, {})
    monkeypatch.setattr(R, "ExecutionService", _ExecutionService)

    await R.chat(_body("Find the email from Abhishek"), _Principal(), _DB())

    assert _ExecutionService.created == []


async def test_sending_still_reaches_execution_service(monkeypatch):
    """The new read branch must not have stolen the send path."""
    from navis.reasoning.email_ops import EmailProposal

    _stub_persons(monkeypatch, [_ABHISHEK])
    monkeypatch.setattr(R, "ExecutionService", _ExecutionService)

    class _Agent:
        def __init__(self, db):
            pass

        async def propose(self, ctx):
            return EmailProposal(
                to="abhishek@acme.com", to_name="Abhishek",
                subject="Following up", body="Quick nudge on the deck.",
            )

    monkeypatch.setattr(R, "EmailOpsAgent", _Agent)

    await R.chat(
        _body("Send an email to abhishek@acme.com saying: quick nudge"),
        _Principal(), _DB(),
    )

    assert len(_ExecutionService.created) == 1
    assert _ExecutionService.created[0]["action_type"] == "send_email"


async def test_scheduling_still_reaches_execution_service(monkeypatch):
    from navis.reasoning.scheduling.contract import (
        MeetingSlot,
        ResolvedAttendee,
        ScheduledProposal,
    )

    _stub_persons(monkeypatch, [_ABHISHEK])
    monkeypatch.setattr(R, "ExecutionService", _ExecutionService)

    class _Agent:
        def __init__(self, db, **kw):
            pass

        async def propose(self, ctx):
            return ScheduledProposal(
                attendee=ResolvedAttendee(
                    email="abhishek@acme.com", source="person_search_unique"
                ),
                slot=MeetingSlot(
                    start=datetime(2026, 8, 23, 10, 0, tzinfo=timezone.utc),
                    end=datetime(2026, 8, 23, 10, 30, tzinfo=timezone.utc),
                    time_zone="UTC", duration_minutes=30,
                ),
                title="Meeting with Abhishek",
            )

    monkeypatch.setattr(R, "SchedulingAgent", _Agent)

    await R.chat(
        _body("Schedule a meeting with Abhishek tomorrow"), _Principal(), _DB()
    )

    assert len(_ExecutionService.created) == 1
    assert _ExecutionService.created[0]["action_type"] == "schedule_meeting"


def test_the_accidental_tool_layer_is_gone():
    """`navis/tools/` was duplicate architecture (ActionSpec already is the
    action registry). It must not come back by import."""
    import importlib

    for mod in ("navis.tools", "navis.tools.registry", "navis.tools.data"):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(mod)
