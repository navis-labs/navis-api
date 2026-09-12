"""Ask Navis — the ORCHESTRATION entry point above the pipeline.

Ask is not another layer in BUSINESS SYSTEMS → SIGNALS → MEMORY → STATE →
CONTEXT → REASONING/ML → INTELLIGENCE → DECISION → EXECUTION → OBSERVATION →
LEARNING → ADAPTIVE. It sits above it: a question arrives, this module works out
what organizational reality answering it requires, asks the layers that own that
reality, and hands the model an already-grounded bundle to put into words.

  question
    → classify()                     ONE routing decision (reasoning.routing)
    → _resolve_question_subject()    the EXISTING identity graph decides WHO
         ambiguous → clarification, never a guess
    → _state_groundings()            State ├─ scope_for  → Grounding(state)
                                           └─ reasoning  → Grounding(reasoning)
    → ContextBuilder.build(retrieval_query=<resolved aliases>)
         MemoryRetriever (vector + rerank)  → Grounding(memory)
         learning_groundings()              → Grounding(learning)
         intelligence_groundings()          → Grounding(intelligence)
         _compose: conclusions, State, evidence — budgeted to _CONTEXT_TOKEN_BUDGET
    → generation.explanation | recommendation | summary
    → MLProvider (Claude / Azure / Gemini / DeterministicProvider fallback)
    → ChatResponse { content, citations[+lineage], entity, stateWatermark }

THE LLM IS NOT THE SOURCE OF ORGANIZATIONAL TRUTH. It receives a GroundedContext
and nothing else — no raw rows, no tools, no way to reach a side effect — and its
job is to explain what Memory/State/Reasoning/Intelligence already concluded.
Every enrichment above is best-effort: losing one costs the answer some depth,
never the turn.

No canned strings. No fabricated citations. No invented importance: when the org
has nothing on a question we say so rather than manufacture a reason.

Command intents (schedule / reschedule / cancel / email) skip the answer path
entirely and run their own deterministic agents, which end in a REQUESTED
ExecutionRequest a human still has to approve. Chat never executes anything.
"""
from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from typing import Any, Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from navis.api.workspace_fallback import (
    build_workspace_context,
    connected_source_labels,
)
from navis.execution.models import ActionType, ExecutionOrigin
from navis.execution.service import ExecutionService
from navis.identity.authz import Principal
from navis.identity.throttle import rate_limit
from navis.organization.contract import Capability
from navis.organization.guards import require_capability
from navis.organization.service import OrganizationService
from navis.ml.generation.narratives import explanation, recommendation, summary
from navis.ml.rag.grounding import Grounding, GroundedContext
from navis.ml.rag.time_window import parse_time_window
from navis.platform.database import get_session
from navis.platform.monitoring import get_logger
from navis.platform.monitoring.metrics import metrics
from navis.reasoning.calendar_ops import (
    CalendarOpsAgent,
    CalendarOpsClarification,
    CalendarOpsContext,
    MeetingMutationProposal,
)
from navis.reasoning.calendar_query import CalendarQueryAgent, CalendarQueryContext
from navis.reasoning.email_ops import (
    EmailClarification,
    EmailOpsAgent,
    EmailOpsContext,
    EmailProposal,
)
from navis.reasoning.routing import ChatRoute, classify
from navis.reasoning.scheduling import (
    Clarification,
    ScheduledProposal,
    SchedulingAgent,
    SchedulingContext,
)
from navis.reasoning.scheduling.contract import ChatTurn, PendingScheduling

# ── The shared context composer ─────────────────────────────────────────────
#
# Everything below used to be DEFINED in this file. It now lives in
# `orchestration/context.py`, because the Developer boundary needs the same
# assembly and a second copy of it would be a second opinion about what is true
# of an organization. Nothing about the chat turn changed: the names are bound
# here under the spellings this module has always used, so every call site below
# reads exactly as it did.
#
# WHAT MOVED, AND WHAT DID NOT. The composer moved: retrieval, State, Reasoning,
# subject resolution, the sufficiency gate, the budget. Everything that makes
# this a CHAT — intent routing, the scheduling/calendar/email agents,
# clarifications, suggestions, the workspace fallback and the generation step —
# stayed, because none of it is organizational truth.
from navis.orchestration.context import (
    CONTEXT_KIND as _CONTEXT_KIND,
    CONTEXT_TOKEN_BUDGET as _CONTEXT_TOKEN_BUDGET,
    DERIVED_KINDS as _DERIVED_KINDS,
    REASONING_KIND as _REASONING_KIND,
    SubjectResolution as _SubjectResolution,
    build_grounded_context,
    compose_within_budget as _compose_within_budget,
    resolve_question_subject as _resolve_question_subject,
    state_groundings as _state_groundings_result,
)

logger = get_logger("api.chat_router")


async def _state_groundings(
    db: AsyncSession, org_id: uuid.UUID
) -> tuple[list[Grounding], str | None]:
    """The chat turn's view of `orchestration.context.state_groundings`.

    Kept as a two-tuple because that is what this module's callers and its
    characterisation tests have always consumed. `read_only` is deliberately
    NOT passed: the chat path keeps `get_state`, which repairs a corrupt or
    absent snapshot by rebuilding it, exactly as it did before the extraction.
    The Developer boundary is the caller that asks for the read-only variant.
    """
    grounded = await _state_groundings_result(db, org_id)
    return grounded.groundings, grounded.watermark

router = APIRouter(tags=["chat"])


# ── Wire shapes (match src/lib/types.ts NavisChatMessage) ───────────────────


class ChatCitation(BaseModel):
    # "intelligence" and "learning" are additions: a persisted insight and an
    # outcome aggregate were both being reported as "episodic", i.e. as a
    # remembered email. The frontend's `formatMemoryType` falls through to the
    # raw string for an unknown value, so an older client degrades to showing the
    # word rather than breaking — see the handoff note in the task report.
    type: Literal["episodic", "semantic", "state", "decision", "intelligence", "learning"]
    source: str
    detail: str | None = None
    # Drill-down for a DERIVED citation, as the producing layer recorded it:
    # "memory:<uuid>", "signal:<uuid>", "reasoning_signal:<uuid>",
    # "learning_event:<id>", "state_watermark:<id>", "learning_version:<n>".
    # Empty for a fact that is already its own lowest level (a memory record) or
    # that recorded no lineage. Never synthesised.
    lineage: list[str] = Field(default_factory=list)


class ChatEntity(BaseModel):
    """The person this question resolved to, via the identity graph.

    Present only when `identity_resolution` folded the question's subject to
    exactly ONE canonical person. Absent when the question named nobody we know,
    and absent when it named several — an ambiguous subject is reported as a
    `person_disambiguation` clarification instead, never silently narrowed here.

    `canonical_id` is the identity-graph id (the smallest alias uuid in the
    equivalence class), the same one `/identity/persons/lookup` returns, so a
    client can navigate from an answer to the person it is about.
    """

    canonical_id: str
    display_name: str | None = None
    email: str | None = None


class ScheduledMeetingPayload(BaseModel):
    """Structured meeting payload attached to a schedule_meeting proposal.

    Mirrors the fields the ScheduleMeetingProposalCard renders. Kept
    separate from the executor's opaque `payload` so the frontend has a
    typed contract to render against — the executor still receives the
    dict via `structured_payload` -> ExecutionRequest.payload.
    """

    attendee_email: str
    attendee_name: str | None = None
    start: str  # ISO 8601 with offset
    end: str
    time_zone: str
    duration_minutes: int
    title: str
    description: str | None = None
    source_signal_event_id: str | None = None
    thread_id: str | None = None
    assumptions: list[str] = Field(default_factory=list)


class ChatActionProposal(BaseModel):
    id: str
    label: str
    channel: Literal[
        "call",
        "draft_email",
        "send_email",
        "schedule_meeting",
        # Cancelling an existing meeting is its own channel: the governance
        # policy for removing a meeting is not the policy for booking one.
        "cancel_meeting",
        "send_slack",
        "open_decision",
        "open_thread",
        "delegate_to",
        "cancel",
    ]
    primary: bool = False
    estimatedDurationMinutes: int | None = None
    decisionId: str | None = None
    href: str | None = None
    # Real-execution wiring — populated on structured schedule_meeting
    # proposals so the frontend can approve through the canonical
    # /actions/{id}/approve pipeline and poll GET /actions/{id} for
    # status transitions. `execution_status` is a snapshot at proposal
    # time; the frontend polls for authoritative updates.
    execution_id: str | None = None
    execution_status: str | None = None
    action_type: str | None = None
    structured_payload: ScheduledMeetingPayload | None = None


class ChatClarification(BaseModel):
    """Structured follow-up when the scheduling agent needs more input.

    Mirrors `reasoning.scheduling.contract.Clarification` on the wire so
    the frontend renders one clarification component regardless of which
    slot needs disambiguation. `datetime` = both date+time missing;
    `scope` = calendar-connection issue (frontend shows Reconnect CTA
    ONLY on scope, never on datetime/date/time asks).
    """

    field: Literal[
        "attendee",
        "date",
        "datetime",
        "time",
        "duration",
        "person_disambiguation",
        # "which of these existing meetings did you mean?" — the calendar-ops
        # disambiguation. `options` carries {event_id, label, start, end}; the
        # frontend echoes the picked `event_id` back as
        # `pending_calendar_event_id` on the next turn.
        "event",
        "scope",
    ]
    question: str
    options: list[dict] = Field(default_factory=list)
    resolved: dict = Field(default_factory=dict)


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    user_id: str | None = None  # legacy client field; ignored (auth from token)
    # Optional context the scheduling agent uses for pronoun resolution.
    # Frontend sends these opportunistically — every field is optional so
    # existing callers (and the mock layer) keep working unchanged.
    active_signal_id: uuid.UUID | None = None
    thread_id: str | None = Field(None, max_length=256)
    recent_turns: list[ChatTurn] = Field(default_factory=list, max_length=6)
    time_zone: str | None = Field(None, max_length=64)
    # Pending scheduling state echoed back from the LAST turn's
    # `ChatResponse.clarification.resolved`. The scheduling agent merges
    # THIS turn's utterance INTO this pending state so multi-turn flows
    # ("Fix meeting with X" → clarify → "tomorrow at 3 PM") preserve the
    # attendee and any previously-given slots. None on a fresh request.
    pending_scheduling: PendingScheduling | None = None
    # Same echo-back mechanism as `pending_scheduling`, for the other two
    # multi-turn flows. `pending_calendar_event_id` carries the meeting the user
    # picked from a disambiguation list; `pending_email` carries whatever the
    # email agent had resolved (to / subject / body) when it asked for the rest.
    pending_calendar_event_id: str | None = Field(None, max_length=1024)
    pending_email: dict[str, Any] | None = None


class ChatResponse(BaseModel):
    id: str
    role: Literal["assistant"] = "assistant"
    intent: Literal["question", "command", "navigate"]
    content: str
    createdAt: str
    citations: list[ChatCitation] = Field(default_factory=list)
    actionProposals: list[ChatActionProposal] = Field(default_factory=list)
    suggestedActions: list[str] = Field(default_factory=list)
    decisionId: str | None = None
    # Populated ONLY when the scheduling agent needs more information.
    # The frontend renders this inline (attendee/date/time/etc. prompt)
    # and the user's reply feeds back into the next chat turn.
    clarification: ChatClarification | None = None
    # Multi-turn state for the calendar-ops and email flows — the frontend
    # echoes these back on the NEXT request exactly as it already does with
    # `clarification.resolved` for scheduling.
    targetEventId: str | None = None
    pendingEmail: dict[str, Any] | None = None
    # WHO this answer is about, when the identity graph resolved it. See ChatEntity.
    entity: ChatEntity | None = None
    # WHICH snapshot of the organization the answer describes — the memory record
    # State folded to, the same value `/brief/today` publishes as
    # `state_watermark`. Null when State was unavailable for this turn (a cold
    # projection times out at 4s and the answer degrades to memory only), which
    # is the honest report: no snapshot, no watermark. It is the field that lets a
    # client tell "the brief and this answer describe the same organization" from
    # "one of them is stale".
    stateWatermark: str | None = None


# ── Intent detection ────────────────────────────────────────────────────────

_COMMAND_VERBS = re.compile(
    r"\b(schedule|book|reply|draft|send|delegate|cancel|hand[\s-]?over|"
    r"loop\s+in|ask|call|invite|dismiss|snooze)\b",
    re.IGNORECASE,
)
_NAVIGATE_VERBS = re.compile(
    r"\b(show|find|open|view|search|where|list|browse)\b", re.IGNORECASE
)
# Prioritization asks — "what should I do first", "what's most urgent", etc.
# Wire intent stays "question"; internally we route to the `recommendation` generator
# instead of `explanation`, which is contractually not allowed to prioritize.
_PRIORITIZE = re.compile(
    r"\b("
    r"what\s+should\s+i\s+(?:do|tackle|focus\s+on)"
    r"|do\s+first"
    r"|top\s+priorit(?:y|ies)"
    r"|most\s+(?:important|urgent|pressing)"
    r"|prioriti[sz]e"
    r"|biggest\s+(?:thing|priority)"
    r"|where\s+should\s+i\s+start"
    r")\b",
    re.IGNORECASE,
)
# "Summarise this week", "give me a recap", "digest of yesterday" — routes to
# the `summary` generator (structured What happened / Why it matters / What next),
# not `explanation` (which is why unfocused summaries were dumping raw facts).
_SUMMARIZE = re.compile(
    r"\b(summar(?:i[sz]e|y)|recap|digest|tl;?dr|catch\s+me\s+up)\b",
    re.IGNORECASE,
)


def _is_prioritization(q: str) -> bool:
    return bool(_PRIORITIZE.search(q))


def _is_summarize(q: str) -> bool:
    return bool(_SUMMARIZE.search(q))


def _detect_intent(q: str) -> Literal["question", "command", "navigate"]:
    if _COMMAND_VERBS.search(q):
        return "command"
    if _NAVIGATE_VERBS.search(q):
        return "navigate"
    return "question"


def _is_meeting_command(q: str) -> bool:
    # Kept for the fallback (LLM/prose) branch. The scheduling agent
    # uses the richer `is_scheduling_intent` detector from
    # `reasoning.scheduling.intent` so misclassifications here don't
    # accidentally short-circuit real scheduling utterances.
    return bool(
        re.search(r"\b(schedule|book|invite|set\s+up)\b", q, re.IGNORECASE)
        and re.search(r"\b(meet|meeting|call|sync|chat)\b", q, re.IGNORECASE)
    )


# ── Scheduling decision-id (chat-initiated actions) ─────────────────────────
# ExecutionRequest.decision_id is required (nullable=False). Brief-driven
# actions get a real Decision UUID from the daily-brief propose bridge;
# chat-initiated actions have no Decision behind them, so we synthesize a
# deterministic UUIDv5 derived from the stable "who + when" of the
# meeting itself. Same (attendee, slot start) -> same decision -> same
# idempotency key -> retries + double-approve converge to ONE row.
_CHAT_DECISION_NAMESPACE = uuid.UUID("d3c2b1a0-9e8f-5a6b-8c7d-1f2e3d4c5b6a")


def _chat_scheduling_decision_id(
    org_id: uuid.UUID, attendee_email: str, start_iso: str
) -> uuid.UUID:
    key = f"{org_id}|chat_scheduling|{attendee_email.lower()}|{start_iso}"
    return uuid.uuid5(_CHAT_DECISION_NAMESPACE, key)


def _chat_calendar_ops_decision_id(
    org_id: uuid.UUID, operation: str, event_id: str, new_start_iso: str
) -> uuid.UUID:
    """Same determinism as `_chat_scheduling_decision_id`.

    (event, operation, target slot) → one decision → one idempotency key → one
    ExecutionRequest. Asking twice for the same change converges instead of
    queueing two mutations of the same event.
    """
    key = f"{org_id}|chat_calendar_ops|{operation}|{event_id}|{new_start_iso}"
    return uuid.uuid5(_CHAT_DECISION_NAMESPACE, key)


def _chat_email_decision_id(
    org_id: uuid.UUID, to: str, subject: str
) -> uuid.UUID:
    """Deterministic per (recipient, subject).

    Note the payload — not just this id — feeds the execution idempotency key,
    so editing the BODY correctly produces a new request rather than silently
    reusing the old one.
    """
    key = f"{org_id}|chat_email|{to.lower()}|{subject.strip().lower()}"
    return uuid.uuid5(_CHAT_DECISION_NAMESPACE, key)


def _join(items: list[str]) -> str:
    """"a", "a and b", "a, b and c" — used in the honest empty-state message."""
    if len(items) == 1:
        return items[0]
    return f"{', '.join(items[:-1])} and {items[-1]}"


def _is_draft_command(q: str) -> bool:
    return bool(re.search(r"\b(draft|reply|send|compose)\b", q, re.IGNORECASE))


def _is_cancel_command(q: str) -> bool:
    return bool(re.search(r"\bcancel\b", q, re.IGNORECASE))


# ── Response composition ────────────────────────────────────────────────────


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# Which truth layer a `Grounding.kind` belongs to on the wire.
#
# Report WHICH layer each citation came from instead of stamping every one
# "episodic". A ScopedView entity, a ReasoningSignal, an organizational insight
# and a remembered email are four different kinds of claim, and a reader cannot
# judge an answer without knowing which they are looking at.
#
# `reasoning` maps to "state" deliberately: both are the org's current
# understanding as opposed to raw recall. `intelligence` and `learning` get their
# own values because they are neither — an insight is a persisted conclusion with
# its own id and its own page, and an outcome aggregate is a measurement. Calling
# either "episodic" (which is what happened before) told the reader they were
# looking at a remembered email.
#
# Anything else — retrieved memory, and the raw workspace-fallback facts whose
# kind is the connector that produced them (`gmail`, `calendar`) — is recall.
_CITATION_TYPE_BY_KIND: dict[str, str] = {
    _REASONING_KIND: "state",
    _CONTEXT_KIND: "state",
    "intelligence": "intelligence",
    "learning": "learning",
}


def _citation(
    kind: str, source: str, text: str, lineage: tuple[str, ...] = ()
) -> ChatCitation:
    """One grounded fact as a wire citation, carrying its drill-down.

    `lineage` was previously computed and thrown away. A ScopedView entity is a
    fold over many memory records, so its `source` (the entity_id) cannot express
    what it was built from — `_scoped_view_to_groundings` already resolves that
    into `memory:<id>` / `signal:<id>` pointers, and an insight already carries
    its supporting reasoning signals and learning events. Dropping all of it left
    every derived citation a dead end for the frontend, which is exactly the
    question "what did Navis use to reach this answer?" that it could not answer.

    Nothing is invented here: the tuple is passed through as the producing layer
    recorded it, and a fact that recorded no lineage ships an empty list.
    """
    return ChatCitation(
        type=_CITATION_TYPE_BY_KIND.get(kind, "episodic"),
        source=source,
        detail=(text or "")[:200] or None,
        lineage=list(lineage),
    )


# The routes that end in an ExecutionRequest. Named once so the entitlement gate
# in `chat()` cannot drift from the set of branches that actually create rows.
_ACTION_ROUTES = frozenset(
    {
        ChatRoute.SCHEDULE_MEETING,
        ChatRoute.RESCHEDULE_MEETING,
        ChatRoute.CANCEL_MEETING,
        ChatRoute.SEND_EMAIL,
    }
)


async def _grounded_answer(
    db: AsyncSession,
    org_id: uuid.UUID,
    question: str,
    subject: "_SubjectResolution | None" = None,
) -> tuple[str, list[ChatCitation], str | None]:
    """Real LLM answer over org memory. Returns (text, citations, watermark).

    Flow: build the grounded context through the SHARED composer, then run the
    `explanation` generator (versioned prompt -> MLProvider). No caller of this
    function ever sees the raw DB — the whole pipeline enforces the "LLM only
    ever reads GroundedContext" contract from `navis.ml.rag`.

    THE COMPOSITION IS NO LONGER THIS MODULE'S. `build_grounded_context` in
    `orchestration/context.py` performs retrieval, the State/Reasoning join, the
    subject expansion and the token budget — the same call the Developer
    boundary makes, so a founder and an agent are grounded in one organization.
    What remains here is everything that makes this a CHAT: the workspace
    fallback, the generator choice, and turning facts into wire citations.

    `read_only` is NOT passed, deliberately: the chat path keeps `get_state`,
    which repairs an absent or corrupt snapshot by rebuilding it. That is what
    it has always done and a human session is entitled to it.

    Empty-signal state: when retrieval returns zero facts the LLM is NOT called
    (it would fabricate). We surface a truthful "no memory yet" reply so the
    founder knows to wait for ingestion instead of trusting an invented answer.
    """
    # Single wall-clock reading shared by the parser (so "tomorrow" and the
    # time-window filter agree) and by GroundedContext (so the LLM has an anchor
    # for relative-date phrasing instead of guessing from training data).
    now = datetime.now(timezone.utc)
    window = parse_time_window(question, now=now)
    since, until = window if window else (None, None)

    composed = await build_grounded_context(
        db, org_id, question,
        # Already resolved by the caller when it had to decide whether to answer
        # at all; passing it avoids a second identity lookup, never a different
        # answer.
        subject=subject,
        token_budget=_CONTEXT_TOKEN_BUDGET,
        since=since, until=until, now=now,
    )
    context = composed.context
    state_watermark = composed.state_watermark
    # The groundings AS ASSEMBLED, not the ones that survived the budget:
    # the fallback below composes its own bundle and needs what State and
    # Reasoning actually produced.
    extra = composed.derived

    # The fallback decision is about RETRIEVAL, not about enrichment. State and
    # reasoning groundings are org-wide and present on almost every turn, so
    # counting them here would permanently suppress the workspace fallback and
    # silently lose the raw Calendar/Gmail evidence it exists to provide.
    if not [f for f in context.facts if f.kind not in _DERIVED_KINDS]:
        # Memory is empty for this question — but the WORKSPACE may not be.
        # Memory lags ingestion (extractors run behind the sync); raw Calendar
        # and Gmail signals do not. Answering "not enough memory" while three
        # meetings and a dozen threads sit in signal_events was simply wrong.
        workspace = await build_workspace_context(db, org_id, question, now=now)
        if workspace.facts:
            # Compose: raw workspace evidence FIRST, org understanding after it,
            # bounded by the same budget the retrieval branch uses.
            context = _compose_within_budget(workspace, extra)
        elif not extra:
            sources = await connected_source_labels(db, org_id)
            if sources:
                return (
                    f"I don't have anything to go on yet — {_join(sources)} "
                    f"{'is' if len(sources) == 1 else 'are'} connected but "
                    "nothing has synced into this window. Give the first sync "
                    "a few minutes and ask again.",
                    [],
                    state_watermark,
                )
            return (
                "I can't answer that yet — no workspace sources are connected. "
                "Connect Gmail and Google Calendar from the Sources page and "
                "I'll answer from your real data, never from guesswork.",
                [],
                state_watermark,
            )
        # else: no memory, no workspace, but real reasoning signals — `context`
        # already holds them, so answer from organizational understanding alone.
    if _is_prioritization(question):
        generator = recommendation
    elif _is_summarize(question):
        generator = summary
    else:
        generator = explanation
    try:
        narrative = await generator(org_id, context)
    except Exception:
        # LLM provider outage / rate limit / config gap. Fall back to a
        # citations-only response so the user still sees the retrieved
        # facts even when the model can't stitch them into prose.
        logger.exception("chat generation failed org=%s", org_id)
        cited_lines = "\n".join(f"· {f.text}" for f in context.facts[:5])
        return (
            "The reasoning model is temporarily unavailable, but here's "
            f"what your memory has on this:\n\n{cited_lines}",
            [_citation(f.kind, f.source, f.text, f.lineage) for f in context.facts[:5]],
            state_watermark,
        )

    # `detail` carries a human-readable snippet of the cited fact so the frontend
    # can render preview text instead of a bare memory_id. Look up the snippet
    # from the context that was fed to the model — that's the exact fact the
    # citation refers to.
    fact_by_source = {f.source: f for f in context.facts}
    citations = [
        _citation(f.kind, f.source, f.text, f.lineage)
        for src in narrative.sources[:5]
        if (f := fact_by_source.get(src)) is not None
    ]
    return narrative.text, citations, state_watermark


async def _scheduling_response(
    request_id: str,
    body: "ChatRequest",
    p: Principal,
    db: AsyncSession,
    now_utc: datetime,
) -> ChatResponse:
    """Run the scheduling agent. On Clarification return a clarification
    ChatResponse (NO ExecutionRequest — the flow is still open). On
    ScheduledProposal create the canonical ExecutionRequest and return
    the structured meeting card.

    Never falls through to the RAG grounded-answer path — scheduling
    intent owns the request end-to-end. Any recoverable error becomes a
    `scope` clarification.
    """
    ctx = SchedulingContext(
        org_id=p.org_id,
        user_id=p.subject_id,
        utterance=body.question,
        active_signal_id=body.active_signal_id,
        thread_id=body.thread_id,
        recent_turns=body.recent_turns or [],
        time_zone=body.time_zone or "UTC",
        now=now_utc,
        pending=body.pending_scheduling,
    )
    agent = SchedulingAgent(db)
    result = await agent.propose(ctx)

    if isinstance(result, Clarification):
        # Serialize the pending PendingScheduling to a plain dict so the
        # ChatClarification wire type stays simple (`resolved: dict`).
        return ChatResponse(
            id=request_id,
            intent="command",
            content=result.question,
            createdAt=_now_iso(),
            clarification=ChatClarification(
                field=result.field,
                question=result.question,
                options=result.options,
                resolved=result.resolved.model_dump(),
            ),
        )

    assert isinstance(result, ScheduledProposal)  # narrow the union

    # Every slot resolved + a free time found. Create the canonical
    # ExecutionRequest. The decision id is deterministic (same attendee
    # + slot → same decision) so retries / double-approve converge to
    # a single row via the existing execution-idempotency key.
    payload = result.to_execution_payload()
    decision_id = _chat_scheduling_decision_id(
        p.org_id, result.attendee.email, result.slot.start.isoformat()
    )
    # SOURCE-EVENT provenance, not reasoning provenance. Both of these are
    # `signal_events.id` values — `active_signal_id` is looked up as a
    # SignalEventRecord by scheduling/context.py, and `source_signal_event_id`
    # is derived from it — so neither can ever address a row in
    # `reasoning_signals`.
    #
    # They were being written into `reasoning_signal_ids`, which Learning
    # aggregates on and Intelligence cites as `supporting_reasoning_signal_ids`.
    # A signal_event id there is a reference that can never resolve, and it
    # would have keyed a `signal_intelligence` insight to something that is not
    # a reasoning signal. Nothing recomputes it back into the right namespace.
    #
    # `reasoning_signal_ids` stays EMPTY here, and that is the honest answer: a
    # scheduling request the founder typed was not concluded by any persisted
    # ReasoningSignal. The email that prompted it is still recorded — beside it,
    # in its own namespace.
    source_signal_event_ids: list[str] = []
    if body.active_signal_id:
        source_signal_event_ids.append(str(body.active_signal_id))
    elif result.source_signal_event_id:
        # Came from the pending state (attendee resolved on a prior turn from
        # an email context).
        source_signal_event_ids.append(str(result.source_signal_event_id))

    svc = ExecutionService(db)
    req, _created = await svc.request(
        p.org_id,
        decision_id,
        ActionType.SCHEDULE_MEETING.value,
        payload,
        reasoning_signal_ids=[],
        source_signal_event_ids=source_signal_event_ids,
        reasoning_version=None,
        state_watermark=None,
        # Explicit, not defaulted: the founder asked for this in the
        # conversation and is looking at the card right now, so Chat owns
        # reviewing it. Mislabelling it PROACTIVE would put it back on the
        # Action Center board as something "Navis noticed".
        origin=ExecutionOrigin.CHAT,
    )
    await db.commit()

    structured = ScheduledMeetingPayload(
        attendee_email=result.attendee.email,
        attendee_name=result.attendee.name,
        start=result.slot.start.isoformat(),
        end=result.slot.end.isoformat(),
        time_zone=result.slot.time_zone,
        duration_minutes=result.slot.duration_minutes,
        title=result.title,
        description=result.description,
        source_signal_event_id=result.source_signal_event_id,
        thread_id=result.thread_id,
        assumptions=result.assumptions,
    )
    label = f"Schedule with {result.attendee.name or result.attendee.email}"
    proposal_chip = ChatActionProposal(
        id=str(req.id),
        label=label,
        channel="schedule_meeting",
        primary=True,
        estimatedDurationMinutes=result.slot.duration_minutes,
        execution_id=str(req.id),
        execution_status=req.status,
        action_type=ActionType.SCHEDULE_MEETING.value,
        structured_payload=structured,
    )
    cancel_chip = ChatActionProposal(
        id=f"cancel_{req.id}",
        label="Not now",
        channel="cancel",
        primary=False,
    )
    prose = (
        f"Here's the proposed meeting with "
        f"{result.attendee.name or result.attendee.email}. Review the "
        "details below and click Schedule meeting to send the invite."
    )
    return ChatResponse(
        id=request_id,
        intent="command",
        content=prose,
        createdAt=_now_iso(),
        actionProposals=[proposal_chip, cancel_chip],
        decisionId=str(decision_id),
    )


async def _calendar_ops_response(
    request_id: str,
    body: "ChatRequest",
    p: Principal,
    db: AsyncSession,
    now_utc: datetime,
    route: ChatRoute,
) -> ChatResponse:
    """Reschedule / cancel an EXISTING meeting.

    Same shape as `_scheduling_response`: a Clarification returns without
    creating anything; a proposal creates the canonical ExecutionRequest and
    returns a structured card the founder approves through the normal
    /actions/{id}/approve pipeline. Nothing here touches Google.
    """
    operation = "reschedule" if route is ChatRoute.RESCHEDULE_MEETING else "cancel"
    ctx = CalendarOpsContext(
        org_id=p.org_id,
        utterance=body.question,
        time_zone=body.time_zone or "UTC",
        now=now_utc,
        operation=operation,
        target_event_id=body.pending_calendar_event_id,
    )
    result = await CalendarOpsAgent(db).propose(ctx)

    if isinstance(result, CalendarOpsClarification):
        return ChatResponse(
            id=request_id,
            intent="command",
            content=result.question,
            createdAt=_now_iso(),
            # `none` means "nothing to act on" (no matching meeting) —
            # informational, so it ships as plain prose with NO clarification
            # box. Rendering it as a prompt would ask the founder to answer a
            # question that isn't one. Every other field maps 1:1.
            clarification=(
                None
                if result.field == "none"
                else ChatClarification(
                    field=result.field,
                    question=result.question,
                    options=result.options,
                    resolved={},
                )
            ),
            targetEventId=result.target_event_id,
        )

    assert isinstance(result, MeetingMutationProposal)  # narrow the union

    action_type = (
        ActionType.RESCHEDULE_MEETING
        if operation == "reschedule"
        else ActionType.CANCEL_MEETING
    )
    payload = result.to_execution_payload()
    # Deterministic decision id, same rationale as scheduling: the same event +
    # the same target slot converge on ONE ExecutionRequest, so a repeated ask
    # (or a double-submitted turn) can never produce two mutations.
    decision_id = _chat_calendar_ops_decision_id(
        p.org_id, operation, result.target.event_id,
        result.new_start.isoformat() if result.new_start else "",
    )
    svc = ExecutionService(db)
    req, _created = await svc.request(
        p.org_id,
        decision_id,
        action_type.value,
        payload,
        reasoning_signal_ids=[],
        reasoning_version=None,
        state_watermark=None,
        # Explicit, not defaulted: the founder asked for this in the
        # conversation and is looking at the card right now, so Chat owns
        # reviewing it. Mislabelling it PROACTIVE would put it back on the
        # Action Center board as something "Navis noticed".
        origin=ExecutionOrigin.CHAT,
    )
    await db.commit()

    if operation == "reschedule":
        structured = ScheduledMeetingPayload(
            attendee_email=(result.target.attendees or [""])[0],
            attendee_name=None,
            start=result.new_start.isoformat(),  # type: ignore[union-attr]
            end=result.new_end.isoformat(),  # type: ignore[union-attr]
            time_zone=result.target.time_zone,
            duration_minutes=payload["duration_minutes"],
            title=result.target.title,
            description=None,
            assumptions=result.assumptions,
        )
        label = f"Move “{result.target.title}”"
        prose = (
            f"Here's the change to “{result.target.title}”. Review it and "
            "approve to update the event — attendees get the new time."
        )
    else:
        structured = ScheduledMeetingPayload(
            attendee_email=(result.target.attendees or [""])[0],
            attendee_name=None,
            start=result.target.start.isoformat(),
            end=result.target.end.isoformat(),
            time_zone=result.target.time_zone,
            duration_minutes=max(
                int((result.target.end - result.target.start).total_seconds() // 60), 1
            ),
            title=result.target.title,
            description=None,
            assumptions=result.assumptions,
        )
        label = f"Cancel “{result.target.title}”"
        prose = (
            f"Ready to cancel “{result.target.title}”. Approve and I'll remove "
            "it and notify the attendees."
        )

    return ChatResponse(
        id=request_id,
        intent="command",
        content=prose,
        createdAt=_now_iso(),
        actionProposals=[
            ChatActionProposal(
                id=str(req.id),
                label=label,
                channel="schedule_meeting" if operation == "reschedule" else "cancel_meeting",
                primary=True,
                execution_id=str(req.id),
                execution_status=req.status,
                action_type=action_type.value,
                structured_payload=structured,
            ),
            ChatActionProposal(
                id=f"dismiss_{req.id}", label="Not now", channel="cancel", primary=False
            ),
        ],
        decisionId=str(decision_id),
        targetEventId=result.target.event_id,
    )


async def _email_response(
    request_id: str, body: "ChatRequest", p: Principal, db: AsyncSession
) -> ChatResponse:
    """Compose an outbound email into an approvable ExecutionRequest.

    Never invents the message. Missing recipient or body produces a
    clarification whose `resolved` state the frontend echoes back, so the
    follow-up answer alone completes the request.
    """
    pending = body.pending_email or {}
    ctx = EmailOpsContext(
        org_id=p.org_id,
        utterance=body.question,
        pending_to=pending.get("to"),
        pending_subject=pending.get("subject"),
        pending_body=pending.get("body"),
    )
    result = await EmailOpsAgent(db).propose(ctx)

    if isinstance(result, EmailClarification):
        return ChatResponse(
            id=request_id,
            intent="command",
            content=result.question,
            createdAt=_now_iso(),
            clarification=ChatClarification(
                field=(
                    "person_disambiguation"
                    if result.field == "person_disambiguation"
                    else "attendee"
                ),
                question=result.question,
                options=result.options,
                resolved={},
            ),
            pendingEmail=result.resolved,
        )

    assert isinstance(result, EmailProposal)  # narrow the union

    payload = result.to_execution_payload()
    decision_id = _chat_email_decision_id(p.org_id, result.to, result.subject)
    req, _created = await ExecutionService(db).request(
        p.org_id,
        decision_id,
        ActionType.SEND_EMAIL.value,
        payload,
        reasoning_signal_ids=[],
        reasoning_version=None,
        state_watermark=None,
        # Explicit, not defaulted: the founder asked for this in the
        # conversation and is looking at the card right now, so Chat owns
        # reviewing it. Mislabelling it PROACTIVE would put it back on the
        # Action Center board as something "Navis noticed".
        origin=ExecutionOrigin.CHAT,
    )
    await db.commit()

    preview = result.body if len(result.body) <= 400 else result.body[:400] + "…"
    return ChatResponse(
        id=request_id,
        intent="command",
        content=(
            f"Ready to send to {result.to_name or result.to}.\n\n"
            f"Subject: {result.subject}\n\n{preview}"
        ),
        createdAt=_now_iso(),
        actionProposals=[
            ChatActionProposal(
                id=str(req.id),
                label=f"Send to {result.to_name or result.to}",
                channel="send_email",
                primary=True,
                execution_id=str(req.id),
                execution_status=req.status,
                action_type=ActionType.SEND_EMAIL.value,
            ),
            ChatActionProposal(
                id=f"dismiss_{req.id}", label="Not now", channel="cancel", primary=False
            ),
        ],
        decisionId=str(decision_id),
    )


def _scope_error_response(request_id: str, message: str) -> ChatResponse:
    """Uniform failure shape for an operational route that owns its turn.

    Falling through to RAG prose here would answer a DIFFERENT question than the
    one asked, which is how "cancel the meeting" ended up returning a memory
    summary.
    """
    return ChatResponse(
        id=request_id,
        intent="command",
        content=message,
        createdAt=_now_iso(),
        clarification=ChatClarification(
            field="scope", question=message, resolved={}
        ),
    )


def _command_response(
    request_id: str,
    question: str,
    llm_text: str,
    citations: list[ChatCitation],
) -> ChatResponse:
    """Structured proposals for command intents — draft / delegate / cancel.

    Scheduling intents no longer land here — they're routed through
    `_scheduling_response` in the top-level endpoint before this
    function runs. The static "Open meeting scheduler" chip stays as a
    last-resort fallback for the rare case where the scheduling agent
    couldn't run (missing dependencies) — surfaced with a plain link.
    """
    now = _now_iso()

    if _is_meeting_command(question):
        return ChatResponse(
            id=request_id,
            intent="command",
            content=(
                llm_text
                + "\n\nOpen the scheduler to pick from your real calendar availability."
            ),
            createdAt=now,
            actionProposals=[
                ChatActionProposal(
                    id="open_scheduler",
                    label="Open meeting scheduler",
                    channel="schedule_meeting",
                    primary=True,
                    estimatedDurationMinutes=30,
                ),
                ChatActionProposal(
                    id="cancel",
                    label="Cancel",
                    channel="cancel",
                    primary=False,
                ),
            ],
            citations=citations,
        )

    if _is_cancel_command(question):
        return ChatResponse(
            id=request_id,
            intent="command",
            content=(llm_text if llm_text else "Which one should I cancel?"),
            createdAt=now,
            actionProposals=[
                ChatActionProposal(
                    id="browse_calendar",
                    label="Open my calendar",
                    channel="open_thread",
                    primary=True,
                    href="/dashboard",
                ),
                ChatActionProposal(
                    id="cancel",
                    label="Never mind",
                    channel="cancel",
                    primary=False,
                ),
            ],
            citations=citations,
        )

    if _is_draft_command(question):
        return ChatResponse(
            id=request_id,
            intent="command",
            content=(
                llm_text
                + "\n\nOpen the composer to review the draft and confirm before send."
            ),
            createdAt=now,
            actionProposals=[
                ChatActionProposal(
                    id="open_composer",
                    label="Open in composer",
                    channel="draft_email",
                    primary=True,
                ),
                ChatActionProposal(
                    id="cancel",
                    label="Cancel",
                    channel="cancel",
                    primary=False,
                ),
            ],
            citations=citations,
        )

    return ChatResponse(
        id=request_id,
        intent="command",
        content=llm_text,
        createdAt=now,
        citations=citations,
    )


def _derive_suggestions(question: str, citations: list[ChatCitation]) -> list[str]:
    """Pick 3 follow-ups the user is likely to want next.

    Static chips ("What should I do first? / Who am I keeping waiting? /
    Summarise this week") were shown on every turn regardless of context —
    users saw the same three prompts on a calendar question and on a
    commitment question. We derive from the retrieved citations + question
    keywords instead. Deterministic, no LLM: prompts must not fabricate.
    """
    q = question.lower()
    detail = " ".join((c.detail or "").lower() for c in citations)
    picks: list[str] = []

    def add(s: str) -> None:
        if s not in picks:
            picks.append(s)

    if any(w in detail for w in ("meet", "call", "sync", "invit", "calendar")):
        add("What's on my calendar this week?")
    if any(w in detail for w in ("waiting", "owes", "reply", "response", "pending", "unanswered")):
        add("Who am I keeping waiting?")
    if any(w in detail for w in ("commit", "decision", "approve", "invest", "deadline")):
        add("What's my biggest open commitment?")
    if any(w in q for w in ("today", "tomorrow", "tomm", "tmrw", "tmr", "coming", "upcoming", "soon")):
        add("What changed since yesterday?")
    if any(w in q for w in ("summar", "recap", "week", "digest")):
        add("What decisions am I still owed?")

    for fallback in ("What should I do first?", "Summarise this week", "Who am I keeping waiting?"):
        add(fallback)
    return picks[:3]


# ── Explicit mail retrieval ─────────────────────────────────────────────────
#
# "about the term sheet", "mentioning Navis", "regarding the invoice" — the ONE
# free-text narrowing this path accepts, and only when the user marked it with a
# preposition. Anything else would mean guessing which words of a question are a
# search term, and a wrong guess silently returns the wrong mail.
_ABOUT_RE = re.compile(
    r"\b(?:about|mentioning|regarding|re:|concerning|with\s+the\s+subject)\s+"
    r"(?P<term>[A-Za-z0-9][\w'&./-]*(?:\s+[A-Za-z0-9][\w'&./-]*){0,3})",
    re.IGNORECASE,
)
# Stop the preposition from swallowing a trailing question word.
_ABOUT_STOP = frozenset({"me", "us", "it", "this", "that", "them", "him", "her"})

_MAIL_RESULT_LIMIT = 8


def _about_term(question: str) -> str | None:
    m = _ABOUT_RE.search(question)
    if not m:
        return None
    term = " ".join(
        w for w in m.group("term").split() if w.lower() not in _ABOUT_STOP
    ).strip(" ?.!,")
    return term or None


async def _email_search_response(
    request_id: str,
    body: "ChatRequest",
    p: Principal,
    db: AsyncSession,
    subject: "_SubjectResolution",
) -> ChatResponse:
    """Answer an explicit mail-retrieval request from the ORGANIZATION'S OWN
    captured mail — the same rows `GET /signals/gmail/recent` serves.

    NO LLM. The founder asked to see messages, not to be told about them: the
    subjects, senders and dates are read off `signal_events` and listed
    verbatim. A model in this path could only paraphrase facts that are already
    exact, and every paraphrase is a chance to alter one. Memory/State/Reasoning
    are not consulted either — this is source retrieval, not organizational
    understanding, and mixing the two is what made "find the email from X"
    answer with prose about the organization.

    Reads through `signals.read_router.search_emails`, the SAME function the
    HTTP route now calls. Chat owns no Gmail query and no Gmail client.
    """
    from navis.signals.read_router import search_emails

    # The GMAIL entitlement, enforced exactly as `GET /signals/gmail/*` enforces
    # it. Chat is gated on MEMORY, which is granted by every plan — without this
    # an org with Gmail explicitly revoked or kill-switched could read its mail
    # through the chat box after the dashboard had stopped showing it.
    if not await OrganizationService(db).can_use(p.org_id, Capability.GMAIL):
        metrics.incr("chat.email_search.not_entitled")
        return ChatResponse(
            id=request_id, intent="question",
            content=(
                "I can't read your mail — the email capability isn't enabled "
                "for this organization."
            ),
            createdAt=_now_iso(),
        )

    senders = tuple(subject.match.emails) if subject.match else ()
    term = _about_term(body.question)
    result = await search_emails(
        db, org_id=p.org_id, senders=senders, text_match=term,
        limit=_MAIL_RESULT_LIMIT,
    )

    who = (
        subject.match.primary_display_name or subject.match.primary_email
        if subject.match else None
    )
    if not result["connected"]:
        return ChatResponse(
            id=request_id, intent="question",
            content=(
                "Gmail isn't connected yet, so I have no mail to search. "
                "Connect it from the Sources page and ask again."
            ),
            createdAt=_now_iso(),
        )

    items = result["items"]
    if not items:
        # Says what was actually searched for, so "nothing" is a fact about the
        # query rather than a shrug. Never widened silently to "here's some mail".
        narrowing = ", ".join(
            filter(None, [f"from {who}" if who else None, f"about “{term}”" if term else None])
        )
        return ChatResponse(
            id=request_id, intent="question",
            content=(
                f"I don't have any captured email {narrowing}."
                if narrowing else "I don't have any captured email yet."
            ),
            createdAt=_now_iso(),
        )

    lines = []
    for it in items:
        when = (it.get("received_at") or "")[:10]
        sender = it.get("from_name") or it.get("from_email") or "unknown sender"
        lines.append(
            f"· {it.get('subject') or '(no subject)'} — {sender}"
            + (f" · {when}" if when else "")
        )
    headline = f"{len(items)} email{'' if len(items) == 1 else 's'}"
    if who:
        headline += f" from {who}"
    if term:
        headline += f" about “{term}”"

    return ChatResponse(
        id=request_id,
        intent="question",
        content=headline + ":\n\n" + "\n".join(lines),
        createdAt=_now_iso(),
        # Provenance is the SIGNAL EVENT, which is what these rows actually are.
        # `signal:<uuid>` matches the convention `workspace_fallback` already
        # uses, and the provider claim is honest here in a way it is not for a
        # derived fact: these rows came off the Gmail connector.
        citations=[
            ChatCitation(
                type="episodic",
                source=f"signal:{it['event_id']}",
                detail=(it.get("subject") or "")[:200] or None,
            )
            for it in items[:5]
        ],
        entity=(
            ChatEntity(
                canonical_id=subject.match.canonical_id,
                display_name=subject.match.primary_display_name,
                email=subject.match.primary_email,
            )
            if subject.match else None
        ),
    )


def _navigate_response(
    request_id: str, question: str, llm_text: str, citations: list[ChatCitation]
) -> ChatResponse:
    return ChatResponse(
        id=request_id,
        intent="navigate",
        content=llm_text,
        createdAt=_now_iso(),
        citations=citations,
        actionProposals=[
            ChatActionProposal(
                id="open_dashboard",
                label="Open dashboard",
                channel="open_thread",
                primary=True,
                href="/dashboard",
            ),
        ],
        suggestedActions=_derive_suggestions(question, citations),
    )


def _question_response(
    request_id: str, question: str, llm_text: str, citations: list[ChatCitation]
) -> ChatResponse:
    return ChatResponse(
        id=request_id,
        intent="question",
        content=llm_text,
        createdAt=_now_iso(),
        citations=citations,
        suggestedActions=_derive_suggestions(question, citations),
    )


# ── Endpoint ────────────────────────────────────────────────────────────────


@router.post(
    "/chat",
    response_model=ChatResponse,
    # T-23: the most expensive request in the product — embeds a query, runs a
    # full-scan vector search over the org's corpus, folds state and calls an LLM.
    # Counted per authenticated organization so one tenant cannot starve the rest.
    dependencies=[Depends(rate_limit("chat", "rate_limit_chat_per_window"))],
)
async def chat(
    body: ChatRequest,
    # CAPABILITY-GATED ON MEMORY, like every other product surface.
    #
    # This was `Depends(get_principal)` — authentication only — and it made
    # `/chat` the single expensive endpoint in the product with no entitlement
    # gate. Every sibling has one: the signal reads on GMAIL/CALENDAR/DRIVE,
    # `/decision-memory` and `/memory/people` on DECISION, `/brief/*` and
    # `/context` on DECISION, `/intelligence/*` on INTELLIGENCE, `/actions/*`
    # on EXECUTION.
    #
    # That gate is also the ONLY place organization LIFECYCLE is enforced —
    # `authz.principal_cache_evict_org` says so outright ("the capability gate
    # reads lifecycle from the database on every request"), and
    # `entitlements._decide` refuses anything that is not TRIAL or ACTIVE
    # before it looks at plan or kill switch. So a SUSPENDED or CHURNED
    # organization kept full access to the request this router's own comment
    # calls "the most expensive request in the product": embed the query,
    # full-scan the org's vectors, fold state, call the LLM. Suspension stopped
    # the brief, the signal feeds and the action queue, and left the one
    # surface that actually spends money running.
    #
    # MEMORY is the capability chat consumes — it answers out of the org's
    # Memory — and it is granted by EVERY plan including FREE
    # (`PLAN_CAPABILITIES`), so no entitled organization changes behaviour:
    # this denies suspended/churned tenants, an explicit revocation, and the
    # `memory` kill switch. Nothing else.
    #
    # `require_capability` depends on `get_principal` and FastAPI caches a
    # dependency per request, so the principal is still resolved exactly once.
    p: Principal = Depends(require_capability(Capability.MEMORY)),
    db: AsyncSession = Depends(get_session),
) -> ChatResponse:
    """Ask Navis — grounded LLM chat over the org's Memory.

    Scheduling utterances (fresh OR mid-clarification follow-ups) are
    routed to the reasoning-side scheduling agent BEFORE the general
    grounded-answer path. Once we're in a scheduling flow we OWN the
    turn — errors become a `scope` clarification, not a fall-through
    to RAG prose with confusing episodic/relationship citations.
    """
    request_id = f"chat_{uuid.uuid4().hex[:12]}"
    now_utc = datetime.now(timezone.utc)

    # ONE routing decision, taken in one place. See `reasoning.routing` for why
    # the previous three-independent-predicates arrangement dropped whole verb
    # classes ("reschedule", "cancel the meeting", "send email to …") into the
    # memory-QA path and answered them with "not enough memory".
    route = classify(
        body.question,
        has_pending_scheduling=body.pending_scheduling is not None,
    )
    logger.info("chat route=%s org=%s", route.value, p.org_id)

    # ── EXECUTION ENTITLEMENT, asked ONCE before any action route ─────────
    #
    # These four routes all end in `ExecutionService.request`, and that service
    # deliberately checks only `require_spec` — it is the intake for the router,
    # chat and the proactive proposer alike, so it does not know whose entitlement
    # to ask about. `POST /actions` carries `require_capability(EXECUTION)` for
    # exactly this reason; chat reached the same service without it.
    #
    # The consequence was not a security hole (approve/reject/cancel are each
    # gated, and the worker re-checks before every provider call, so nothing could
    # ever RUN) but a promise the product could not keep: a FREE or PRO
    # organization — neither plan grants EXECUTION — got a fully rendered
    # "Schedule meeting" card whose only button returns 402. Rows accumulated in
    # `execution_requests` that no one in that org could ever approve.
    #
    # Asked BEFORE dispatch, not inside the branches: the agents run an LLM
    # slot-fill and hit Google Calendar, and there is no reason to spend either on
    # an action that cannot be offered. Returned as prose rather than raised as
    # 402 because each branch below wraps itself in `except Exception` — an
    # HTTPException here would be swallowed and reported as "something went
    # wrong", which is the opposite of what happened.
    if route in _ACTION_ROUTES and not await OrganizationService(db).can_use(
        p.org_id, Capability.EXECUTION
    ):
        metrics.incr("chat.action.not_entitled")
        logger.info("chat action route refused, no EXECUTION org=%s route=%s",
                    p.org_id, route.value)
        return ChatResponse(
            id=request_id,
            intent="command",
            content=(
                "I can find this for you, but I can't act on it — taking actions "
                "isn't enabled for this organization. Ask an admin to check the "
                "plan, then try again."
            ),
            createdAt=_now_iso(),
        )

    if route is ChatRoute.SCHEDULE_MEETING:
        try:
            return await _scheduling_response(
                request_id, body, p, db, now_utc
            )
        except Exception:  # noqa: BLE001 — surface as clarification
            logger.exception("scheduling handoff failed org=%s", p.org_id)
            # Scheduling owns this turn. Surface a scope clarification
            # so the founder sees a real error, not RAG prose that
            # doesn't answer them.
            return ChatResponse(
                id=request_id,
                intent="command",
                content=(
                    "Something went wrong while scheduling. Try again in "
                    "a moment, or reconnect Google Calendar if this keeps "
                    "happening."
                ),
                createdAt=_now_iso(),
                clarification=ChatClarification(
                    field="scope",
                    question=(
                        "Something went wrong while scheduling. Try again "
                        "in a moment."
                    ),
                    resolved=(
                        body.pending_scheduling.model_dump()
                        if body.pending_scheduling is not None
                        else {}
                    ),
                ),
            )

    if route in (ChatRoute.RESCHEDULE_MEETING, ChatRoute.CANCEL_MEETING):
        try:
            return await _calendar_ops_response(request_id, body, p, db, now_utc, route)
        except Exception:  # noqa: BLE001
            logger.exception("calendar-ops handoff failed org=%s", p.org_id)
            return _scope_error_response(
                request_id,
                "I couldn't reach your calendar to make that change. Try again "
                "in a moment.",
            )

    if route is ChatRoute.SEND_EMAIL:
        try:
            return await _email_response(request_id, body, p, db)
        except Exception:  # noqa: BLE001
            logger.exception("email handoff failed org=%s", p.org_id)
            return _scope_error_response(
                request_id,
                "I couldn't prepare that email just now. Try again in a moment.",
            )

    # Calendar-query intent — questions ABOUT existing meetings
    # ("what's on my calendar today?", "prepare me for my next
    # meeting", "how many meetings this week?"). We answer these by
    # reading Google Calendar directly instead of going through RAG,
    # which returns "not enough memory" for calendar queries because
    # events aren't reliably indexed in memory_records.
    if route is ChatRoute.CALENDAR_QUERY:
        try:
            calendar_ctx = CalendarQueryContext(
                org_id=p.org_id,
                utterance=body.question,
                time_zone=body.time_zone or "UTC",
                now=now_utc,
            )
            result = await CalendarQueryAgent(db).answer(calendar_ctx)
            return ChatResponse(
                id=request_id,
                intent="question",
                content=result.text,
                createdAt=_now_iso(),
                suggestedActions=_derive_suggestions(body.question, []),
            )
        except Exception:  # noqa: BLE001 — never crash the endpoint
            logger.exception("calendar-query handoff failed org=%s", p.org_id)
            # Fall through to the grounded answer, which now degrades through
            # Calendar/Gmail signals rather than claiming empty memory.

    # ── WHO is this question about? ──────────────────────────────────────
    #
    # Resolved BEFORE the mail branch as well as the answer path: "find the
    # email from Ajeet" needs the identity graph to know which addresses Ajeet
    # actually sends from, and an ambiguous name must be asked about rather than
    # narrowed to whichever person the mail happens to match.
    # Asked BEFORE retrieval, through the existing identity graph, because the
    # answer changes what we retrieve (see `_retrieval_query`). An ambiguous
    # subject is a real answer and is returned as one: picking the first of two
    # people called "Priya" would attribute one person's evidence to another,
    # which is the single worst thing this endpoint could do.
    subject = await _resolve_question_subject(db, p.org_id, body.question)
    if subject.ambiguous:
        metrics.incr("chat.subject.ambiguous")
        logger.info(
            "chat subject ambiguous org=%s candidates=%d", p.org_id,
            len(subject.candidates),
        )
        return ChatResponse(
            id=request_id,
            intent="question",
            content=(
                f"There are {len(subject.candidates)} people matching "
                f"“{subject.query}”. Which one do you mean?"
            ),
            createdAt=_now_iso(),
            clarification=ChatClarification(
                field="person_disambiguation",
                question=(
                    f"Which “{subject.query}” do you mean?"
                ),
                # The identity graph's own rendering — canonical_id, every known
                # alias, every observed display name. Nothing added.
                options=[m.to_dict() for m in subject.candidates],
            ),
        )

    # ── EXPLICIT MAIL RETRIEVAL ───────────────────────────────────────────
    # Dispatched here, AFTER identity resolution and its ambiguity check, so a
    # sender filter is built from the canonical addresses the graph resolved and
    # two people with one name are asked about rather than merged.
    #
    # Read-only and it stops here: no Memory, no State, no Reasoning, no LLM,
    # and no path from this branch to ExecutionService. Sending is a different
    # route (`SEND_EMAIL`), decided earlier and unaffected.
    if route is ChatRoute.EMAIL_SEARCH:
        try:
            return await _email_search_response(request_id, body, p, db, subject)
        except Exception:  # noqa: BLE001 — this route owns its turn
            logger.exception("email search failed org=%s", p.org_id)
            return ChatResponse(
                id=request_id, intent="question",
                content=(
                    "I couldn't reach your mail just now. Try again in a moment."
                ),
                createdAt=_now_iso(),
            )

    intent = _detect_intent(body.question)
    llm_text, citations, state_watermark = await _grounded_answer(
        db, p.org_id, body.question, subject
    )

    if intent == "command":
        response = _command_response(request_id, body.question, llm_text, citations)
    elif intent == "navigate":
        response = _navigate_response(request_id, body.question, llm_text, citations)
    else:
        response = _question_response(request_id, body.question, llm_text, citations)

    # Stamp the lineage the three builders above don't take: WHO the answer is
    # about and WHICH snapshot it describes. Set here rather than threaded
    # through each builder because it is the same answer to the same two
    # questions regardless of which prose shape the intent selected.
    response.stateWatermark = state_watermark
    if subject.match is not None:
        response.entity = ChatEntity(
            canonical_id=subject.match.canonical_id,
            display_name=subject.match.primary_display_name,
            email=subject.match.primary_email,
        )

    # ── One structured trace line per Ask ─────────────────────────────────
    # IDS AND COUNTS ONLY — never a question, never a fact, never an address.
    # This is what makes "what did Navis use to reach this answer?" answerable
    # from the logs without storing anything sensitive: the entity it resolved,
    # the snapshot it read, and how many citations of each layer survived into
    # the answer. `request_id` is the same value the response carries.
    by_type: dict[str, int] = {}
    for c in citations:
        by_type[c.type] = by_type.get(c.type, 0) + 1
    logger.info(
        "ask request_id=%s org=%s route=%s intent=%s entity=%s watermark=%s "
        "citations=%d by_type=%s",
        request_id, p.org_id, route.value, intent,
        subject.match.canonical_id if subject.match else None,
        state_watermark, len(citations), by_type,
    )
    return response
