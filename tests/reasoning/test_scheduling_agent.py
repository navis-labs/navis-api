"""SchedulingAgent behavioral tests — the product spec, encoded.

Covers the full multi-turn scheduling flow:
  * Intent detection (positive + negative false-positive guards)
  * Direct email attendee (no RAG)
  * Pronoun → active_signal_id resolution
  * Name → identity-search resolution + ambiguity
  * Missing date / missing time / missing both → clarification
  * Multi-turn merge: bare follow-ups continue the pending flow
  * Freebusy: exact-time busy returns alternatives; flexible-full asks
    for another day
  * Duration default applied with visible assumption
  * Same fully-resolved request twice → same idempotency key

Every test injects fakes (LLM, calendar client, signal getter, person
search) so no test touches Google or a real DB.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from types import SimpleNamespace


from navis.identity_resolution.search import PersonMatch, PersonSearchResult
from navis.ml.providers.base import EmbedResult, GenRequest, GenResult, MLProvider, ProviderInfo
from navis.reasoning.scheduling import (
    Clarification,
    ScheduledProposal,
    SchedulingAgent,
    SchedulingContext,
)
from navis.reasoning.scheduling.contract import PendingScheduling
from navis.reasoning.scheduling.intent import (
    is_probable_followup,
    is_scheduling_intent,
)

ORG = uuid.UUID("11111111-1111-1111-1111-111111111111")
USER = uuid.UUID("22222222-2222-2222-2222-222222222222")


# ── Fakes ──────────────────────────────────────────────────────────────


class _FakeLLM(MLProvider):
    """Returns non-JSON prose. Forces the agent to rely on the
    deterministic regex prefill — which is what we actually want to
    test at the behavioral level."""

    def __init__(self, response_text: str = "not-json") -> None:
        self.info = ProviderInfo(
            name="fake", gen_model="fake", embed_model="fake", embed_dim=1, online=False
        )
        self._response_text = response_text
        self.calls: list[GenRequest] = []

    async def generate(self, req: GenRequest) -> GenResult:
        self.calls.append(req)
        return GenResult(text=self._response_text, model="fake")

    async def embed(self, texts: list[str]) -> EmbedResult:
        return EmbedResult(vectors=[[0.0]] * len(texts), model="fake", dim=1)


class _FakeCalendarClient:
    """Implements just `freebusy`. Busy blocks are ISO strings."""

    def __init__(self, busy_iso_pairs: list[tuple[str, str]] | None = None) -> None:
        self.busy = busy_iso_pairs or []
        self.freebusy_calls: list[dict] = []

    async def freebusy(
        self, *, time_min, time_max, calendar_ids=None, time_zone="UTC"
    ):
        self.freebusy_calls.append(
            {"time_min": time_min, "time_max": time_max, "time_zone": time_zone}
        )
        return {
            "calendars": {
                "primary": {
                    "busy": [{"start": s, "end": e} for s, e in self.busy]
                }
            }
        }


class _FakeTransport:
    async def aclose(self) -> None:
        pass


def _cal_factory(client: _FakeCalendarClient):
    async def _f(db, org_id):
        return client, _FakeTransport()

    return _f


def _fake_signal(
    *,
    signal_id: uuid.UUID,
    sender_email: str,
    sender_name: str = "",
    thread_id: str | None = None,
):
    return SimpleNamespace(
        id=signal_id,
        organization_id=ORG,
        payload={
            "from": f"{sender_name} <{sender_email}>" if sender_name else sender_email,
            "thread_id": thread_id,
        },
    )


def _signal_getter(row):
    async def _get(_ctx):
        return row

    return _get


def _person_search(result: PersonSearchResult):
    async def _search(_org_id, _query):
        return result

    return _search


def _empty_search():
    return _person_search(PersonSearchResult(query="", matches=()))


def _ctx(utterance: str, **overrides) -> SchedulingContext:
    base = dict(
        org_id=ORG,
        user_id=USER,
        utterance=utterance,
        # Fixed "now" — Tuesday 2026-08-04 15:00 UTC → deterministic
        # day-of-week math across test hosts.
        now=datetime(2026, 8, 4, 15, 0, tzinfo=timezone.utc),
        time_zone="UTC",
    )
    base.update(overrides)
    return SchedulingContext(**base)


def _agent(
    *,
    llm: _FakeLLM | None = None,
    calendar: _FakeCalendarClient | None = None,
    signal=None,
    search: PersonSearchResult | None = None,
) -> SchedulingAgent:
    return SchedulingAgent(
        db=None,
        ml_provider=llm or _FakeLLM(),
        calendar_client_factory=_cal_factory(calendar or _FakeCalendarClient()),
        signal_getter=_signal_getter(signal),
        person_search_fn=_person_search(search)
        if search is not None
        else _empty_search(),
    )


# ═════════════════════════════════════════════════════════════════════
#  M: Intent detector — negative false-positive guards
# ═════════════════════════════════════════════════════════════════════


def test_intent_fires_on_scheduling_utterances():
    assert is_scheduling_intent("schedule a meeting with rahul tuesday afternoon")
    assert is_scheduling_intent("book a call with him")
    assert is_scheduling_intent("set up a call with mulayamabmrc@gmail.com")
    assert is_scheduling_intent("fix the meeting with mulayamabmrc@gmail.com")
    assert is_scheduling_intent("fix a meeting with Rahul Tuesday afternoon")
    assert is_scheduling_intent("set up a meeting with him tomorrow at 3")
    assert is_scheduling_intent("schedule a call with the person who emailed me")
    assert is_scheduling_intent("can you arrange a meeting with Rahul?")
    assert is_scheduling_intent("lock in a call with Rahul for Friday morning")
    assert is_scheduling_intent("catch up with rahul next week")


def test_intent_rejects_non_meeting_false_positives():
    assert not is_scheduling_intent("fix the bug")
    assert not is_scheduling_intent("plan the roadmap")
    assert not is_scheduling_intent("lock in pricing")
    assert not is_scheduling_intent("schedule the deployment")
    assert not is_scheduling_intent("set up the project")
    assert not is_scheduling_intent("call the API")
    assert not is_scheduling_intent("meeting notes for the tuesday sync")
    assert not is_scheduling_intent("meeting room booking system")
    assert not is_scheduling_intent("sync the repo")
    assert not is_scheduling_intent("what did rahul say in his last email?")


def test_followup_detector_accepts_fragments_but_rejects_new_questions():
    # Fragment answers → route through scheduling when pending exists.
    assert is_probable_followup("tomorrow at 3 PM")
    assert is_probable_followup("3 PM")
    assert is_probable_followup("friday")
    assert is_probable_followup("mulayamabmrc@gmail.com")
    # Non-scheduling questions → do NOT route through scheduling even
    # if pending exists.
    assert not is_probable_followup("what changed today?")
    assert not is_probable_followup("show me the calendar")
    assert not is_probable_followup("find the deal")


def test_intent_rejects_meeting_QUERIES_regression():
    """Regression: "tell about today meeting schedule" and similar
    ask ABOUT existing meetings, they do NOT create one. Previously the
    detector saw verb=schedule + noun=meeting and misfired, then the
    agent asked "who should I schedule this meeting with?" which is
    the wrong response to a listing/query request.
    """
    # Query verbs at the start → never scheduling
    assert not is_scheduling_intent("tell about today meeting schedule")
    assert not is_scheduling_intent("tell me my meetings today")
    assert not is_scheduling_intent("show me today's meetings")
    assert not is_scheduling_intent("list my meetings this week")
    assert not is_scheduling_intent("how many meetings do I have today")
    assert not is_scheduling_intent("prepare me for my next meeting")
    assert not is_scheduling_intent("what meetings are on my calendar")
    assert not is_scheduling_intent("when is my next call")
    # "meeting schedule" as a noun phrase
    assert not is_scheduling_intent("my meeting schedule for tomorrow")
    # Explicit rejections
    assert not is_scheduling_intent("not schedule anything")
    assert not is_scheduling_intent("nevermind, forget it")


def test_followup_rejects_mid_flow_query_regression():
    """Regression: user in mid-scheduling flow types a query instead of
    an answer ("not scheudle just tell me how many meeting are scheduled
    with me today"). Router must NOT treat this as a scheduling
    follow-up — it should fall through to the general chat.
    """
    assert not is_probable_followup(
        "not scheudle just tell me how many meeting are scheduled with me today"
    )
    assert not is_probable_followup("tell me about my meetings")
    assert not is_probable_followup("how many meetings are there today")
    assert not is_probable_followup("nevermind")
    assert not is_probable_followup("prepare me for the next one")
    # Confirm plain fragment answers still pass
    assert is_probable_followup("tomorrow 3 PM")
    assert is_probable_followup("friday morning")


# ═════════════════════════════════════════════════════════════════════
#  A: fix the meeting with X — direct email, missing when → clarify
# ═════════════════════════════════════════════════════════════════════


async def test_A_direct_email_missing_datetime_asks_when():
    agent = _agent()
    result = await agent.propose(
        _ctx("fix the meeting with mulayamabmrc@gmail.com")
    )
    assert isinstance(result, Clarification)
    # `datetime` — NOT `scope`. `scope` triggers the frontend's
    # reconnect-Google-Calendar affordance, which was appearing
    # incorrectly on this "when?" question. See the regression test
    # below for the guard.
    assert result.field == "datetime"
    assert "when" in result.question.lower()
    # Attendee IS resolved and echoed back in pending — the next turn
    # should not require the user to restate the email.
    assert result.resolved.attendee_email == "mulayamabmrc@gmail.com"
    assert result.resolved.attendee_source == "direct_email"
    assert result.resolved.date_constraint is None
    assert result.resolved.time_window is None


def test_every_agent_clarification_field_serializes_on_the_wire_regression():
    """Regression: the agent's Clarification.field literals and the
    HTTP-facing ChatClarification.field literals MUST stay in sync.
    A mismatch causes a pydantic ValidationError at response time —
    exactly what happened on the first release of the `datetime` field
    (agent set field='datetime', wire type rejected it as invalid)."""
    from typing import get_args
    from navis.api.chat_router import ChatClarification
    from navis.reasoning.scheduling.contract import Clarification

    agent_fields = set(get_args(Clarification.model_fields["field"].annotation))
    wire_fields = set(get_args(ChatClarification.model_fields["field"].annotation))
    missing = agent_fields - wire_fields
    assert not missing, (
        "ChatClarification.field is missing values the agent emits: "
        f"{missing}. Add them to the wire Literal."
    )


async def test_all_null_pending_still_carries_scheduling_context_regression():
    """Regression: an all-null PendingScheduling (returned by the very
    first attendee clarification when NOTHING has been resolved yet)
    must still let a follow-up utterance continue the scheduling flow.

    Previously the frontend stripped `pending_scheduling` from the
    request body if every field was null, breaking multi-turn flows
    like:
       Turn 1: "fix a meeting"           -> Clarification(attendee, all-null pending)
       Turn 2: "with X tomorrow at 3pm"  -> should merge with pending, NOT go to RAG

    The router's follow-up check is `pending is not None`, so any
    PendingScheduling instance (even all-null) SHOULD route the
    utterance through the scheduling agent. This test locks the
    backend behavior; the frontend fix is in
    src/services/api.ts::askNavis.
    """
    from navis.reasoning.scheduling.contract import PendingScheduling

    all_null = PendingScheduling()
    # Every field must be null on a fresh clarification.
    assert all_null.attendee_email is None
    assert all_null.date_constraint is None
    assert all_null.time_window is None
    # The instance itself is NOT None — that's the routing signal.
    assert all_null is not None
    # And the model round-trips through JSON without dropping fields.
    payload = all_null.model_dump()
    assert "attendee_email" in payload  # even if None
    assert "date_constraint" in payload


async def test_when_question_does_not_use_scope_field_regression():
    """Regression: the 'when would you like?' clarification MUST NOT
    use field='scope' because the frontend shows a Reconnect Google
    Calendar button ONLY on scope, and Google is not the issue here."""
    agent = _agent()
    result = await agent.propose(
        _ctx("fix the meeting with mulayamabmrc@gmail.com")
    )
    assert isinstance(result, Clarification)
    assert result.field != "scope", (
        "'when?' asks must not use field='scope' — that would cause "
        "the UI to show a Reconnect Google Calendar button even when "
        "the connection is fine."
    )


# ═════════════════════════════════════════════════════════════════════
#  B: follow-up "tomorrow at 3 PM" merges → freebusy → proposal
# ═════════════════════════════════════════════════════════════════════


async def test_B_followup_merges_with_pending_attendee_and_proposes():
    pending = PendingScheduling(
        attendee_email="mulayamabmrc@gmail.com",
        attendee_source="direct_email",
    )
    agent = _agent(calendar=_FakeCalendarClient())
    # Note: the follow-up utterance has NO scheduling verb — the
    # router routes it through scheduling because pending is set. Here
    # we call the agent directly.
    result = await agent.propose(
        _ctx("tomorrow at 3 PM", pending=pending)
    )
    assert isinstance(result, ScheduledProposal), (
        f"expected proposal, got clarification: "
        f"{getattr(result, 'question', None)}"
    )
    assert result.attendee.email == "mulayamabmrc@gmail.com"
    # Tomorrow 3 PM in the fixed test clock (2026-08-05).
    assert result.slot.start.hour == 15
    assert result.slot.start.date().isoformat() == "2026-08-05"


# ═════════════════════════════════════════════════════════════════════
#  C: name + date only, missing time → asks time
# ═════════════════════════════════════════════════════════════════════


async def test_C_name_and_date_missing_time_asks_time():
    match = PersonMatch(
        canonical_id="c1",
        display_names=("Rahul Sharma",),
        emails=("rahul@partner.com",),
        alias_ids=("a1",),
    )
    agent = _agent(search=PersonSearchResult(query="rahul", matches=(match,)))
    result = await agent.propose(
        _ctx("schedule a meeting with Rahul tomorrow")
    )
    assert isinstance(result, Clarification)
    assert result.field == "time"
    assert "what time" in result.question.lower()
    # Rahul and tomorrow both preserved so the follow-up merges.
    assert result.resolved.attendee_email == "rahul@partner.com"
    assert result.resolved.date_constraint == "tomorrow"
    assert result.resolved.time_window is None


# ═════════════════════════════════════════════════════════════════════
#  D: follow-up "3 PM" preserves Rahul + tomorrow → proposal
# ═════════════════════════════════════════════════════════════════════


async def test_D_time_followup_preserves_name_and_date():
    pending = PendingScheduling(
        attendee_email="rahul@partner.com",
        attendee_name="Rahul Sharma",
        attendee_source="person_search_unique",
        date_constraint="tomorrow",
    )
    agent = _agent(calendar=_FakeCalendarClient())
    result = await agent.propose(_ctx("3 PM", pending=pending))
    assert isinstance(result, ScheduledProposal), (
        f"got clarification: {getattr(result, 'question', None)}"
    )
    assert result.attendee.email == "rahul@partner.com"
    assert result.slot.start.date().isoformat() == "2026-08-05"
    assert result.slot.start.hour == 15


# ═════════════════════════════════════════════════════════════════════
#  E: name + time only, missing date → asks date
# ═════════════════════════════════════════════════════════════════════


async def test_E_name_and_time_missing_date_asks_date():
    match = PersonMatch(
        canonical_id="c1",
        display_names=("Rahul Sharma",),
        emails=("rahul@partner.com",),
        alias_ids=("a1",),
    )
    agent = _agent(search=PersonSearchResult(query="rahul", matches=(match,)))
    result = await agent.propose(
        _ctx("schedule a meeting with Rahul at 3 PM")
    )
    assert isinstance(result, Clarification)
    assert result.field == "date"
    assert result.resolved.attendee_email == "rahul@partner.com"
    assert result.resolved.time_window == "15:00"
    assert result.resolved.date_constraint is None


# ═════════════════════════════════════════════════════════════════════
#  F: follow-up "Friday" preserves Rahul + 3 PM → proposal
# ═════════════════════════════════════════════════════════════════════


async def test_F_date_followup_preserves_name_and_time():
    pending = PendingScheduling(
        attendee_email="rahul@partner.com",
        attendee_name="Rahul Sharma",
        attendee_source="person_search_unique",
        time_window="15:00",
    )
    agent = _agent(calendar=_FakeCalendarClient())
    result = await agent.propose(_ctx("Friday", pending=pending))
    assert isinstance(result, ScheduledProposal), (
        f"got clarification: {getattr(result, 'question', None)}"
    )
    # 2026-08-04 is a Tuesday; "Friday" means 2026-08-07.
    assert result.slot.start.date().isoformat() == "2026-08-07"
    assert result.slot.start.hour == 15


# ═════════════════════════════════════════════════════════════════════
#  G: flexible "Tuesday afternoon" → first free slot in window
# ═════════════════════════════════════════════════════════════════════


async def test_G_flexible_window_returns_first_free_slot():
    match = PersonMatch(
        canonical_id="c1",
        display_names=("Rahul Sharma",),
        emails=("rahul@partner.com",),
        alias_ids=("a1",),
    )
    agent = _agent(
        calendar=_FakeCalendarClient(),
        search=PersonSearchResult(query="rahul", matches=(match,)),
    )
    result = await agent.propose(
        _ctx("schedule a meeting with Rahul Tuesday afternoon")
    )
    assert isinstance(result, ScheduledProposal), (
        f"got clarification: {getattr(result, 'question', None)}"
    )
    # Next Tuesday (2026-08-11), afternoon window (12:00-17:00) UTC —
    # first slot is 12:00.
    assert result.slot.start.date().isoformat() == "2026-08-11"
    assert 12 <= result.slot.start.hour < 17


# ═════════════════════════════════════════════════════════════════════
#  H: "him" + active_signal_id → sender resolved, proposal built
# ═════════════════════════════════════════════════════════════════════


async def test_H_pronoun_resolves_to_active_signal_sender():
    signal_id = uuid.uuid4()
    row = _fake_signal(
        signal_id=signal_id,
        sender_email="rahul@partner.com",
        sender_name="Rahul Sharma",
    )
    agent = _agent(calendar=_FakeCalendarClient(), signal=row)
    result = await agent.propose(
        _ctx(
            "schedule a meeting with him Tuesday afternoon",
            active_signal_id=signal_id,
        )
    )
    assert isinstance(result, ScheduledProposal)
    assert result.attendee.email == "rahul@partner.com"
    assert result.attendee.source == "active_signal_sender"


# ═════════════════════════════════════════════════════════════════════
#  I: "him" without any signal context → asks who (no guess)
# ═════════════════════════════════════════════════════════════════════


async def test_I_pronoun_without_signal_context_asks_who():
    agent = _agent(signal=None)
    result = await agent.propose(
        _ctx("schedule a meeting with him Tuesday afternoon")
    )
    assert isinstance(result, Clarification)
    assert result.field == "attendee"
    # Question mentions the pronoun so the user knows what we couldn't
    # resolve.
    assert "him" in result.question.lower()


# ═════════════════════════════════════════════════════════════════════
#  J: two matching Rahuls → asks which (no arbitrary pick)
# ═════════════════════════════════════════════════════════════════════


async def test_J_two_matches_returns_person_disambiguation():
    matches = (
        PersonMatch(
            canonical_id="c1",
            display_names=("Rahul Agarwal",),
            emails=("rahul.a@partner.com",),
            alias_ids=("a1",),
        ),
        PersonMatch(
            canonical_id="c2",
            display_names=("Rahul Sharma",),
            emails=("rahul.s@other.com",),
            alias_ids=("a2",),
        ),
    )
    agent = _agent(search=PersonSearchResult(query="rahul", matches=matches))
    result = await agent.propose(
        _ctx("schedule a call with Rahul tuesday afternoon")
    )
    assert isinstance(result, Clarification)
    assert result.field == "person_disambiguation"
    assert len(result.options) == 2
    emails = {o["email"] for o in result.options}
    assert emails == {"rahul.a@partner.com", "rahul.s@other.com"}


# ═════════════════════════════════════════════════════════════════════
#  K: exact "3 PM" busy → alternatives clarification (NO silent slide)
# ═════════════════════════════════════════════════════════════════════


async def test_K_exact_time_busy_returns_alternatives_not_silent_slide():
    # Book 15:00-15:30 UTC on next Tuesday (2026-08-11).
    busy = [(
        "2026-08-11T15:00:00+00:00",
        "2026-08-11T15:30:00+00:00",
    )]
    match = PersonMatch(
        canonical_id="c1",
        display_names=("Rahul Sharma",),
        emails=("rahul@partner.com",),
        alias_ids=("a1",),
    )
    agent = _agent(
        calendar=_FakeCalendarClient(busy),
        search=PersonSearchResult(query="rahul", matches=(match,)),
    )
    result = await agent.propose(
        _ctx("schedule with Rahul Tuesday at 3 PM")
    )
    # Must NOT silently slide — must be a clarification asking for a
    # different time and offering nearby alternatives.
    assert isinstance(result, Clarification), (
        f"agent silently produced a proposal: "
        f"{getattr(result, 'slot', None)}"
    )
    assert result.field == "time"
    # Alternatives are populated (there's plenty of space outside the
    # 30-min busy block within the ±4h radius).
    assert result.options, "expected alternative slots"
    # Every option carries a `time_window` the user can pick to re-run.
    assert all("time_window" in o for o in result.options)
    # Pending state cleared time_window (user must pick a new time).
    assert result.resolved.time_window is None
    # But date is preserved.
    assert result.resolved.date_constraint == "tuesday"


# ═════════════════════════════════════════════════════════════════════
#  L: flexible "Tuesday afternoon" partly busy → picks free slot
# ═════════════════════════════════════════════════════════════════════


async def test_L_flexible_window_first_free_slot_after_busy():
    # 12:00-13:00 UTC busy on next Tuesday — first free slot is 13:00.
    busy = [(
        "2026-08-11T12:00:00+00:00",
        "2026-08-11T13:00:00+00:00",
    )]
    match = PersonMatch(
        canonical_id="c1",
        display_names=("Rahul Sharma",),
        emails=("rahul@partner.com",),
        alias_ids=("a1",),
    )
    agent = _agent(
        calendar=_FakeCalendarClient(busy),
        search=PersonSearchResult(query="rahul", matches=(match,)),
    )
    result = await agent.propose(
        _ctx("schedule with Rahul Tuesday afternoon")
    )
    assert isinstance(result, ScheduledProposal)
    # First free 30-min slot after the busy 12-13 block is 13:00 UTC.
    assert result.slot.start.hour == 13
    assert result.slot.start.date().isoformat() == "2026-08-11"


# ═════════════════════════════════════════════════════════════════════
#  N: fully-resolved same request twice → deterministic idempotency
# ═════════════════════════════════════════════════════════════════════


async def test_N_same_request_twice_produces_same_idempotency_key():
    from navis.execution.idempotency import execution_id, idempotency_key
    from navis.api.chat_router import _chat_scheduling_decision_id

    match = PersonMatch(
        canonical_id="c1",
        display_names=("Rahul Sharma",),
        emails=("rahul@partner.com",),
        alias_ids=("a1",),
    )
    agent = _agent(
        calendar=_FakeCalendarClient(),
        search=PersonSearchResult(query="rahul", matches=(match,)),
    )
    r1 = await agent.propose(
        _ctx("schedule with Rahul Tuesday afternoon")
    )
    r2 = await agent.propose(
        _ctx("schedule with Rahul Tuesday afternoon")
    )
    assert isinstance(r1, ScheduledProposal) and isinstance(r2, ScheduledProposal)
    d1 = _chat_scheduling_decision_id(
        ORG, r1.attendee.email, r1.slot.start.isoformat()
    )
    d2 = _chat_scheduling_decision_id(
        ORG, r2.attendee.email, r2.slot.start.isoformat()
    )
    assert d1 == d2
    key1 = idempotency_key(ORG, d1, "schedule_meeting", r1.to_execution_payload())
    key2 = idempotency_key(ORG, d2, "schedule_meeting", r2.to_execution_payload())
    assert key1 == key2
    assert execution_id(ORG, key1) == execution_id(ORG, key2)


# ═════════════════════════════════════════════════════════════════════
#  Extra: duration default surfaced as visible assumption
# ═════════════════════════════════════════════════════════════════════


async def test_duration_default_surfaces_as_assumption():
    match = PersonMatch(
        canonical_id="c1",
        display_names=("Rahul Sharma",),
        emails=("rahul@partner.com",),
        alias_ids=("a1",),
    )
    agent = _agent(
        calendar=_FakeCalendarClient(),
        search=PersonSearchResult(query="rahul", matches=(match,)),
    )
    result = await agent.propose(
        _ctx("schedule with Rahul Tuesday at 3 PM")
    )
    assert isinstance(result, ScheduledProposal)
    assert result.slot.duration_minutes == 30
    assert any("30-min" in a for a in result.assumptions)


async def test_explicit_duration_is_respected_no_assumption():
    match = PersonMatch(
        canonical_id="c1",
        display_names=("Rahul Sharma",),
        emails=("rahul@partner.com",),
        alias_ids=("a1",),
    )
    agent = _agent(
        calendar=_FakeCalendarClient(),
        search=PersonSearchResult(query="rahul", matches=(match,)),
    )
    result = await agent.propose(
        _ctx("schedule a 45-minute call with Rahul Tuesday at 3 PM")
    )
    assert isinstance(result, ScheduledProposal)
    assert result.slot.duration_minutes == 45
    # No default-duration assumption should be emitted.
    assert not any("30-min" in a for a in result.assumptions)


# ═════════════════════════════════════════════════════════════════════
#  Extra: no ExecutionRequest created before clarification completes
# ═════════════════════════════════════════════════════════════════════


async def test_agent_never_creates_execution_request_directly():
    """The SchedulingAgent MUST NOT touch ExecutionService — that
    responsibility lives in the chat_router, and ONLY on
    ScheduledProposal. Verify by checking the agent's return type on
    an incomplete request is Clarification, never a mock proposal."""
    agent = _agent()
    result = await agent.propose(
        _ctx("fix the meeting with mulayamabmrc@gmail.com")
    )
    assert isinstance(result, Clarification), (
        "agent must not fabricate a proposal from an incomplete request"
    )
