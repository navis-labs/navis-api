"""The real Gmail thread behind a proactive candidate.

Item 4 of the phase spec: drafting must use the ACTUAL thread as its primary
context, not the candidate's one-line title. A draft written from "Reply to
John" alone can only be generic; a draft written from the thread knows who to
address, what the subject is, what they actually asked, and which message it
must reply to.

Read from `signal_events` — Navis's own read model — rather than by calling
Gmail. The messages are already there (that is how the candidate came to exist),
it costs no quota, and it means drafting works identically in tests, in replay,
and offline. The reconcile path added in the calendar work keeps that read model
current for calendar; for Gmail the sync is the writer.

Read-only. Nothing here mutates state or reaches the network.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from email.utils import parseaddr

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from navis.platform.monitoring import get_logger
from navis.signals.base import SignalEventType, SignalSource
from navis.signals.models import SignalEventRecord

logger = get_logger("execution.drafts.thread_context")

# Enough of a conversation to address it correctly. Drafting needs the latest
# inbound message and whether the founder has spoken since — not the archive.
_MAX_MESSAGES = 25


@dataclass(frozen=True, slots=True)
class ThreadMessage:
    """One message in the thread, normalised for drafting."""

    message_id: str
    rfc_message_id: str | None
    sent_at: datetime | None
    direction: str  # "inbound" | "outbound"
    from_email: str | None
    from_name: str | None
    subject: str | None
    snippet: str | None

    @property
    def is_inbound(self) -> bool:
        return self.direction != "outbound"


@dataclass(frozen=True, slots=True)
class ThreadContext:
    """What a draft needs to know about the conversation it is joining."""

    thread_id: str
    subject: str | None
    messages: tuple[ThreadMessage, ...]  # oldest → newest

    @property
    def latest(self) -> ThreadMessage | None:
        return self.messages[-1] if self.messages else None

    @property
    def latest_inbound(self) -> ThreadMessage | None:
        """The message a reply is answering."""
        for m in reversed(self.messages):
            if m.is_inbound:
                return m
        return None

    def latest_inbound_from(self, email: str | None) -> ThreadMessage | None:
        """The newest inbound message from a SPECIFIC person.

        A follow-up to Chrissy must quote Chrissy's promise, not whatever the
        last person to speak happened to say.
        """
        if not email:
            return self.latest_inbound
        target = email.strip().lower()
        for m in reversed(self.messages):
            if m.is_inbound and (m.from_email or "").lower() == target:
                return m
        return None

    def watermark(self) -> str | None:
        """Opaque snapshot of "where this conversation was".

        Compared later to decide whether a draft went stale. Message id plus
        timestamp: the id alone would miss an edit-in-place, and the timestamp
        alone collides when two messages land in the same second.
        """
        last = self.latest
        if last is None:
            return None
        when = last.sent_at.isoformat() if last.sent_at else ""
        return f"{last.message_id}|{when}"

    def reply_subject(self) -> str:
        """"Re: <subject>", without stacking a second Re: on a reply."""
        base = (self.subject or "").strip()
        if not base:
            return "Re:"
        if base.lower().startswith("re:"):
            return base
        return f"Re: {base}"


async def load_thread(
    db: AsyncSession, org_id: uuid.UUID, thread_id: str
) -> ThreadContext | None:
    """Every captured message in one Gmail thread, oldest first.

    Returns None when the thread is unknown — drafting then refuses rather than
    writing a reply into a conversation it cannot see.
    """
    if not thread_id:
        return None
    rows = (
        await db.scalars(
            select(SignalEventRecord)
            .where(
                SignalEventRecord.organization_id == org_id,
                SignalEventRecord.source == SignalSource.GMAIL.value,
                SignalEventRecord.event_type == SignalEventType.EMAIL_RECEIVED.value,
                SignalEventRecord.channel_id.is_(None),
            )
            .order_by(SignalEventRecord.occurred_at.asc())
            .limit(500)
        )
    ).all()

    messages: list[ThreadMessage] = []
    subject: str | None = None
    for rec in rows:
        payload = _decrypted(rec)
        if str(payload.get("thread_id") or "") != thread_id:
            continue
        subject = subject or payload.get("subject")
        messages.append(_to_message(rec, payload))

    if not messages:
        return None
    messages.sort(key=lambda m: (m.sent_at is None, m.sent_at))
    return ThreadContext(
        thread_id=thread_id,
        subject=subject,
        messages=tuple(messages[-_MAX_MESSAGES:]),
    )


def _decrypted(rec: SignalEventRecord) -> dict:
    """Gmail bodies are sealed at rest; the read seam decrypts them."""
    from navis.signals.read_router import _decrypt_content

    try:
        return _decrypt_content(rec.payload or {})
    except Exception:  # noqa: BLE001 — a body we cannot read is not a crash
        logger.exception("draft thread context: decrypt failed for signal %s", rec.id)
        return rec.payload or {}


def _to_message(rec: SignalEventRecord, payload: dict) -> ThreadMessage:
    labels = payload.get("label_ids") or (rec.extra or {}).get("label_ids") or []
    direction = (
        "outbound"
        if isinstance(labels, (list, tuple)) and "SENT" in labels
        else "inbound"
    )
    name, email = parseaddr(str(payload.get("from") or ""))
    return ThreadMessage(
        message_id=rec.external_id,
        rfc_message_id=payload.get("rfc_message_id"),
        sent_at=rec.occurred_at,
        direction=direction,
        from_email=(payload.get("sender_email") or email or "").strip().lower() or None,
        from_name=payload.get("sender_name") or name or None,
        subject=payload.get("subject"),
        snippet=(payload.get("snippet") or payload.get("body_text") or "")[:400] or None,
    )
