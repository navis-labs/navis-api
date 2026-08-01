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

from navis.identity.authz import Principal, get_principal
from navis.ml.generation.narratives import explanation, recommendation, summary
from navis.ml.rag.context_builder import ContextBuilder
from navis.ml.rag.time_window import parse_time_window
from navis.platform.database import get_session
from navis.platform.monitoring import get_logger

logger = get_logger("api.chat_router")

router = APIRouter(tags=["chat"])


# ── Wire shapes (match src/lib/types.ts NavisChatMessage) ───────────────────


class ChatCitation(BaseModel):
    type: Literal["episodic", "semantic", "state", "decision"]
    source: str
    detail: str | None = None


class ChatActionProposal(BaseModel):
    id: str
    label: str
    channel: Literal[
        "call",
        "draft_email",
        "send_email",
        "schedule_meeting",
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


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    user_id: str | None = None  # legacy client field; ignored (auth from token)


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
    return bool(
        re.search(r"\b(schedule|book|invite|set\s+up)\b", q, re.IGNORECASE)
        and re.search(r"\b(meet|meeting|call|sync|chat)\b", q, re.IGNORECASE)
    )


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
        return (
            "Navis doesn't have enough memory to answer this yet. "
            "Once more signals ingest, ask again — every answer here is "
            "cited from your real workspace, never fabricated.",
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


def _command_response(
    request_id: str,
    question: str,
    llm_text: str,
    citations: list[ChatCitation],
) -> ChatResponse:
    """Structured proposals for command intents — draft / schedule / delegate.

    The LLM answer supplies context ("here's what I know about this thread")
    while the action proposals stay deterministic — those are UX affordances
    the frontend renders as one-click chips.
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
    """Ask Navis — grounded LLM chat over the org's Memory."""
    request_id = f"chat_{uuid.uuid4().hex[:12]}"
    intent = _detect_intent(body.question)
    llm_text, citations = await _grounded_answer(db, p.org_id, body.question)

    if intent == "command":
        return _command_response(request_id, body.question, llm_text, citations)
    if intent == "navigate":
        return _navigate_response(request_id, body.question, llm_text, citations)
    return _question_response(request_id, body.question, llm_text, citations)
