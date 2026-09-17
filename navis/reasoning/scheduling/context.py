"""Attendee resolution — pronoun → sender email, name → identity graph.

Deterministic priority (first hit wins):

  1. Explicit email address in the current utterance
     ("schedule with alice@x.com") → `direct_email`.
  2. Explicit `active_signal_id` — the email the user is currently
     looking at. Pronoun ("him", "her") uses that signal's sender.
  3. Explicit `thread_id` — the newest inbound signal in the thread's
     sender. Only used when active_signal_id is absent.
  4. Explicit person name matched against `identity_aliases` — must
     resolve to a SINGLE canonical class. Multiple matches → the
     agent surfaces a person-picker clarification.
  5. Otherwise → attendee clarification. NEVER falls back to
     "the most recent email overall" for a pronoun — that would
     silently pick the wrong Rahul when the user is mid-context on
     another conversation.

Never guesses. Never fabricates an email.

The DB-touching pieces are wrapped in small callables (`SignalGetter`,
`PersonSearchFn`) so tests can inject fakes without spinning up a real
SQLAlchemy engine.
"""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from email.utils import parseaddr
from typing import Awaitable, Callable, Optional

from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from navis.identity_resolution.search import PersonMatch, PersonSearchResult, search_persons
from navis.reasoning.scheduling.contract import (
    MeetingConstraints,
    ResolvedAttendee,
    SchedulingContext,
)
from navis.signals.base import SignalSource
from navis.signals.models import SignalEventRecord

# RFC-5322-ish email extractor. Kept conservative — we'd rather miss an
# oddly-formatted address than pull an accidental substring out of prose.
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")


@dataclass(frozen=True, slots=True)
class AttendeeResolution:
    """Outcome of one attendee-resolution pass.

    Exactly one of these is set:
      * attendee — we know who the meeting is with.
      * candidates — >1 person matched; caller emits a clarification.
      * None (both empty) — nothing found; caller asks a clarification.
    """

    attendee: ResolvedAttendee | None = None
    candidates: tuple[PersonMatch, ...] = ()

    @property
    def is_ambiguous(self) -> bool:
        return self.attendee is None and len(self.candidates) > 1

    @property
    def is_missing(self) -> bool:
        return self.attendee is None and not self.candidates


SignalGetter = Callable[[SchedulingContext], Awaitable[Optional[SignalEventRecord]]]
PersonSearchFn = Callable[[uuid.UUID, str], Awaitable[PersonSearchResult]]


async def resolve_attendee(
    db: AsyncSession,
    ctx: SchedulingContext,
    constraints: MeetingConstraints,
    *,
    signal_getter: SignalGetter | None = None,
    person_search_fn: PersonSearchFn | None = None,
) -> AttendeeResolution:
    """Return the attendee, an ambiguous set, or nothing (=> clarify).

    `signal_getter` and `person_search_fn` are dependency-injected so
    tests can substitute in-memory fakes for the DB round-trips. The
    production code path uses the defaults at the bottom of this module,
    which are the same async-session queries the agent used before.
    """
    # 1. Direct email in the utterance always wins.
    match = _EMAIL_RE.search(ctx.utterance)
    if match:
        email = match.group(0).strip().lower()
        return AttendeeResolution(
            attendee=ResolvedAttendee(email=email, source="direct_email")
        )

    getter = signal_getter or _default_signal_getter(db)
    searcher = person_search_fn or _default_person_search(db)

    # 2. Pronoun resolution — anchor to the signal the user is looking at.
    if constraints.pronoun:
        pronoun_attendee = await _resolve_by_signal_context(ctx, getter)
        if pronoun_attendee is not None:
            return AttendeeResolution(attendee=pronoun_attendee)
        # Fall through only if no signal context exists — the agent will
        # emit a "who did you mean by 'him'?" clarification.
        return AttendeeResolution()

    # 3. Name search over the identity graph.
    hint = (constraints.attendee_hint or "").strip()
    if not hint:
        return AttendeeResolution()
    result = await searcher(ctx.org_id, hint)
    if result.single is not None:
        m = result.single
        email = m.primary_email
        if not email:
            # Rare: matched on display_name only; no email alias yet.
            # Treat as "no attendee" so the agent asks for the email.
            return AttendeeResolution()
        return AttendeeResolution(
            attendee=ResolvedAttendee(
                email=email,
                name=m.primary_display_name,
                canonical_id=m.canonical_id,
                source="person_search_unique",
            )
        )
    if result.matches:
        return AttendeeResolution(candidates=result.matches)
    return AttendeeResolution()


async def _resolve_by_signal_context(
    ctx: SchedulingContext, getter: SignalGetter
) -> ResolvedAttendee | None:
    """Pronoun → signal sender.

    Only resolves from an EXPLICIT active_signal_id or thread_id; no
    "most-recent-email-in-the-org" fallback. If neither reference
    exists, the agent asks the user WHO they meant instead of guessing.
    """
    row = await getter(ctx)
    if row is None:
        return None
    if row.organization_id != ctx.org_id:
        return None
    return _attendee_from_signal_row(row, source="active_signal_sender")


# ── Default async-session implementations ──────────────────────────────


def _default_signal_getter(db: AsyncSession) -> SignalGetter:
    async def _fetch(ctx: SchedulingContext) -> SignalEventRecord | None:
        if ctx.active_signal_id is not None:
            row = await db.get(SignalEventRecord, ctx.active_signal_id)
            if row is not None and row.organization_id == ctx.org_id:
                return row
        if ctx.thread_id:
            thread_row = await _latest_thread_signal(db, ctx.org_id, ctx.thread_id)
            if thread_row is not None:
                return thread_row
        # No "most recent email overall" fallback — a pronoun with no
        # explicit reference means we ask WHO, never guess.
        return None

    return _fetch


def _default_person_search(db: AsyncSession) -> PersonSearchFn:
    async def _search(org_id: uuid.UUID, query: str) -> PersonSearchResult:
        return await search_persons(db, org_id, query)

    return _search


def _attendee_from_signal_row(
    row: SignalEventRecord, *, source: str
) -> ResolvedAttendee | None:
    """Extract (name, email) from a Gmail-shaped signal payload."""
    payload = row.payload or {}
    raw_from = payload.get("from") or ""
    name, email = parseaddr(str(raw_from))
    email = (email or "").strip().lower()
    if not email:
        return None
    return ResolvedAttendee(
        email=email,
        name=(name or None),
        canonical_id=None,
        source=source,  # type: ignore[arg-type]
    )


async def _latest_thread_signal(
    db: AsyncSession, org_id: uuid.UUID, thread_id: str
) -> SignalEventRecord | None:
    """Newest Gmail signal in the thread. Uses JSONB path for the thread key."""
    from sqlalchemy import func

    stmt = (
        select(SignalEventRecord)
        .where(
            SignalEventRecord.organization_id == org_id,
            SignalEventRecord.source == SignalSource.GMAIL.value,
            # payload->>'thread_id' — matches the parser's field name.
            func.jsonb_extract_path_text(SignalEventRecord.payload, "thread_id")
            == thread_id,
        )
        .order_by(desc(SignalEventRecord.occurred_at))
        .limit(1)
    )
    return await db.scalar(stmt)


