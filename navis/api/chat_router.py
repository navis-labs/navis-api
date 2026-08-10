"""Ask Navis chat endpoint — grounded LLM answers over org Memory.

Every response goes through the canonical ML stack (`navis.ml`):

  question → ContextBuilder.build(org_id, question)
    → MemoryRetriever (vector search + rerank) → GroundedContext (cited facts)
    → generation.explanation(org_id, context)
    → MLProvider (Claude / Azure / Gemini / DeterministicProvider fallback)
    → Narrative { text, sources, model }

No canned strings. No fabricated citations. When the org has no memory
yet, we return an honest "not enough signal to answer" reply — never
invent an answer with no grounding.

Command intents (schedule / draft / cancel) still emit structured
action proposals the AskConsole renders as one-click chips, because
those flows are UX affordances that don't need the LLM to author.
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
from navis.identity.authz import Principal, get_principal
from navis.ml.generation.narratives import explanation, recommendation, summary
from navis.ml.rag.context_builder import ContextBuilder
from navis.ml.rag.time_window import parse_time_window
from navis.platform.database import get_session
from navis.platform.monitoring import get_logger
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

logger = get_logger("api.chat_router")

router = APIRouter(tags=["chat"])


# ── Wire shapes (match src/lib/types.ts NavisChatMessage) ───────────────────


class ChatCitation(BaseModel):
    type: Literal["episodic", "semantic", "state", "decision"]
    source: str
    detail: str | None = None


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


async def _grounded_answer(
    db: AsyncSession,
    org_id: uuid.UUID,
    question: str,
) -> tuple[str, list[ChatCitation]]:
    """Real LLM answer over org memory. Returns (text, citations).

    Flow: build the grounded context via vector-search retrieval → run the
    `explanation` generator (versioned prompt → MLProvider). No caller of
    this function ever sees the raw DB — the whole pipeline enforces the
    "LLM only ever reads GroundedContext" contract from `navis.ml.rag`.

    Empty-signal state: when the retriever returns zero facts the LLM is
    NOT called (would fabricate). We surface a truthful "no memory yet"
    reply so the founder knows to wait for ingestion instead of trusting
    an invented answer.
    """
    # Single wall-clock reading shared by the parser (so "tomorrow" and the
    # time-window filter agree) and by GroundedContext (so the LLM has an anchor
    # for relative-date phrasing instead of guessing from training data).
    now = datetime.now(timezone.utc)
    window = parse_time_window(question, now=now)
    since, until = window if window else (None, None)
    context = await ContextBuilder(db).build(
        org_id, question, k=8, token_budget=1500, since=since, until=until, now=now
    )
    if not context.facts:
        # Memory is empty for this question — but the WORKSPACE may not be.
        # Memory lags ingestion (extractors run behind the sync); raw Calendar
        # and Gmail signals do not. Answering "not enough memory" while three
        # meetings and a dozen threads sit in signal_events was simply wrong.
        context = await build_workspace_context(db, org_id, question, now=now)
        if not context.facts:
            sources = await connected_source_labels(db, org_id)
            if sources:
                return (
                    f"I don't have anything to go on yet — {_join(sources)} "
                    f"{'is' if len(sources) == 1 else 'are'} connected but "
                    "nothing has synced into this window. Give the first sync "
                    "a few minutes and ask again.",
                    [],
                )
            return (
                "I can't answer that yet — no workspace sources are connected. "
                "Connect Gmail and Google Calendar from the Sources page and "
                "I'll answer from your real data, never from guesswork.",
                [],
            )
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
            [
                ChatCitation(type="episodic", source=f.source, detail=f.text[:200])
                for f in context.facts[:5]
            ],
        )

    # `detail` carries a human-readable snippet of the cited fact so the frontend
    # can render preview text instead of a bare memory_id. Look up the snippet
    # from the context that was fed to the model — that's the exact fact the
    # citation refers to.
    fact_text_by_source = {f.source: f.text for f in context.facts}
    citations = [
        ChatCitation(
            type="episodic",
            source=src,
            detail=(fact_text_by_source.get(src) or "")[:200] or None,
        )
        for src in narrative.sources[:5]
    ]
    return narrative.text, citations


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
    reasoning_signal_ids: list[str] = []
    if body.active_signal_id:
        reasoning_signal_ids.append(str(body.active_signal_id))
    elif result.source_signal_event_id:
        # Signal came from the pending state (attendee resolved on a
        # prior turn from an email context).
        reasoning_signal_ids.append(result.source_signal_event_id)

    svc = ExecutionService(db)
    req, _created = await svc.request(
        p.org_id,
        decision_id,
        ActionType.SCHEDULE_MEETING.value,
        payload,
        reasoning_signal_ids=reasoning_signal_ids,
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


@router.post("/chat", response_model=ChatResponse)
async def chat(
    body: ChatRequest,
    p: Principal = Depends(get_principal),
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

    intent = _detect_intent(body.question)
    llm_text, citations = await _grounded_answer(db, p.org_id, body.question)

    if intent == "command":
        return _command_response(request_id, body.question, llm_text, citations)
    if intent == "navigate":
        return _navigate_response(request_id, body.question, llm_text, citations)
    return _question_response(request_id, body.question, llm_text, citations)
