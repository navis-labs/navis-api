"""EmailOpsAgent — turn "send email to X about Y" into an approvable send.

"Send email to mulayamabmrc@gmail.com" used to reach the memory-QA path and come
back as "Navis doesn't have enough memory to answer this yet", which is a
non-sequitur: the user issued a command, not a question, and the command needs
no memory at all.

Contract matches the other reasoning agents — `propose()` returns exactly one of
`EmailClarification` | `EmailProposal` and never sends anything. Only the
execution worker talks to Gmail.

The agent will NOT invent an email body. If the utterance carries no message,
it asks. A Chief of Staff that silently makes up what you said to an investor
is worse than one that asks a question.
"""
from __future__ import annotations

import re
import uuid
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.ext.asyncio import AsyncSession

from navis.identity_resolution.search import search_persons
from navis.platform.monitoring import get_logger
from navis.reasoning.routing import normalise_utterance

logger = get_logger("reasoning.email_ops")

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")

# "... saying <body>", "... that <body>", "... about <topic>", "... telling them
# <body>". The first group that matches wins; everything after it is the body.
_BODY_RE = re.compile(
    r"\b(?:saying|to\s+say|that\s+says|telling\s+(?:them|him|her)|"
    r"letting\s+(?:them|him|her)\s+know)\b[:,]?\s*(?P<body>.+)$",
    re.IGNORECASE,
)
_TOPIC_RE = re.compile(
    r"\b(?:about|re|regarding|asking\s+about|to\s+confirm)\b[:,]?\s*(?P<topic>.+)$",
    re.IGNORECASE,
)
# "to Priya", "to Priya Sharma" — a name recipient rather than an address.
_TO_NAME_RE = re.compile(
    r"\bto\s+(?P<name>[A-Z][a-zA-Z'\-]+(?:\s+[A-Z][a-zA-Z'\-]+)?)\b"
)
_NON_NAME_AFTER_TO = {
    "Today", "Tomorrow", "Monday", "Tuesday", "Wednesday", "Thursday",
    "Friday", "Saturday", "Sunday", "Morning", "Afternoon", "Evening",
    "The", "My", "Our", "It", "Them", "Him", "Her", "Everyone", "All",
}


class EmailOpsContext(BaseModel):
    model_config = ConfigDict(extra="ignore")

    org_id: uuid.UUID
    utterance: str = Field(..., min_length=1, max_length=2000)
    # Carried across turns exactly like PendingScheduling: the frontend echoes
    # back whatever the last clarification resolved so "say we'll ship Friday"
    # completes a request whose recipient was already established.
    pending_to: str | None = Field(None, max_length=320)
    pending_subject: str | None = Field(None, max_length=300)
    pending_body: str | None = Field(None, max_length=8000)


class EmailClarification(BaseModel):
    model_config = ConfigDict(extra="ignore")

    kind: Literal["clarification"] = "clarification"
    field: Literal["recipient", "body", "person_disambiguation"]
    question: str = Field(..., max_length=500)
    options: list[dict] = Field(default_factory=list)
    # Everything resolved so far, echoed back on the next turn.
    resolved: dict = Field(default_factory=dict)


class EmailProposal(BaseModel):
    model_config = ConfigDict(extra="ignore")

    kind: Literal["proposal"] = "proposal"
    to: str
    to_name: str | None = None
    subject: str
    body: str
    assumptions: list[str] = Field(default_factory=list)

    def to_execution_payload(self) -> dict:
        """Matches `GmailProvider.validate` + `_senders.gmail_sender`."""
        return {
            "to": [self.to],
            "subject": self.subject,
            "body": self.body,
            "assumptions": self.assumptions,
        }


EmailOpsResult = EmailClarification | EmailProposal


class EmailOpsAgent:
    """Resolve recipient → subject → body, or ask for whichever is missing."""

    def __init__(self, db: AsyncSession | None) -> None:
        self.db = db

    async def propose(self, ctx: EmailOpsContext) -> EmailOpsResult:
        utterance = normalise_utterance(ctx.utterance)

        # ── WHO ───────────────────────────────────────────────────────
        to = ctx.pending_to
        to_name: str | None = None
        match = _EMAIL_RE.search(utterance)
        if match:
            to = match.group(0).strip().lower()
        elif to is None:
            name = _extract_recipient_name(utterance)
            if name and self.db is not None:
                try:
                    result = await search_persons(self.db, ctx.org_id, name)
                except Exception:  # noqa: BLE001 — identity graph is best-effort
                    logger.exception("email_ops person search failed org=%s", ctx.org_id)
                    result = None
                if result is not None and result.single is not None:
                    to = result.single.primary_email
                    to_name = result.single.primary_display_name
                elif result is not None and result.matches:
                    return EmailClarification(
                        field="person_disambiguation",
                        question=(
                            f'I found {len(result.matches)} people matching '
                            f'"{name}". Which one should I email?'
                        ),
                        options=[
                            {
                                "canonical_id": m.canonical_id,
                                "display_name": m.primary_display_name,
                                "email": m.primary_email,
                            }
                            for m in result.matches
                        ],
                        resolved=_resolved(None, ctx.pending_subject, ctx.pending_body),
                    )
            if not to and name:
                return EmailClarification(
                    field="recipient",
                    question=(
                        f'I couldn\'t find an email address for "{name}". '
                        "What address should I use?"
                    ),
                    resolved=_resolved(None, ctx.pending_subject, ctx.pending_body),
                )
        if not to:
            return EmailClarification(
                field="recipient",
                question="Who should I send this to? An email address works best.",
                resolved=_resolved(None, ctx.pending_subject, ctx.pending_body),
            )

        # ── WHAT ──────────────────────────────────────────────────────
        body = _extract_body(utterance) or ctx.pending_body
        topic = _extract_topic(utterance)
        if not body:
            # Never fabricate the message. Ask — and carry the recipient so the
            # answer alone completes the request.
            hint = f" about {topic}" if topic else ""
            return EmailClarification(
                field="body",
                question=(
                    f"What should the email to {to_name or to} say{hint}? "
                    "I'll draft it exactly as you describe and show it before sending."
                ),
                resolved=_resolved(to, ctx.pending_subject or topic, None),
            )

        assumptions: list[str] = []
        subject = ctx.pending_subject or topic
        if not subject:
            subject = _subject_from_body(body)
            assumptions.append(
                f'Used "{subject}" as the subject — edit it before approving if '
                "you'd like something else."
            )
        return EmailProposal(
            to=to,
            to_name=to_name,
            subject=subject[:200],
            body=body,
            assumptions=assumptions,
        )


# ── Extraction helpers ─────────────────────────────────────────────────


def _resolved(to: str | None, subject: str | None, body: str | None) -> dict:
    return {"to": to, "subject": subject, "body": body}


def _extract_recipient_name(utterance: str) -> str | None:
    m = _TO_NAME_RE.search(utterance)
    if not m:
        return None
    name = m.group("name").strip()
    if name.split()[0] in _NON_NAME_AFTER_TO:
        return None
    return name


def _extract_body(utterance: str) -> str | None:
    m = _BODY_RE.search(utterance)
    if not m:
        return None
    body = m.group("body").strip().strip('"').strip("'").strip()
    return body or None


def _extract_topic(utterance: str) -> str | None:
    m = _TOPIC_RE.search(utterance)
    if not m:
        return None
    topic = m.group("topic").strip().strip('"').strip("'").strip()
    # "about" can also introduce a body clause; keep the topic short so it reads
    # as a subject line rather than a paragraph.
    return topic[:120] or None


def _subject_from_body(body: str) -> str:
    """First sentence (or first 60 chars) of the body, title-cased at the front.

    Derived, not invented — the founder can always see and change it on the card
    before approving, and the assumption is surfaced explicitly.
    """
    first = re.split(r"(?<=[.!?])\s", body.strip(), maxsplit=1)[0]
    first = first.strip().rstrip(".")
    if len(first) > 60:
        first = first[:57].rsplit(" ", 1)[0] + "…"
    return first[:1].upper() + first[1:] if first else "Quick note"
