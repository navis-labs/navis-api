"""Calendar + Gmail facts for questions Memory can't answer yet.

"Navis doesn't have enough memory to answer this yet" was being returned for
questions the workspace could answer perfectly well:

  "What should I do first?"     — three meetings today and six unanswered
                                  threads is a real answer to that.
  "Who am I keeping waiting?"   — literally derivable from inbound mail with no
                                  outbound reply in the thread.
  "Prepare me for my 3pm"       — the event, its attendees, and the last mail
                                  from each of them.

Memory (the extracted, embedded, semantic layer) lags ingestion by design: it
only fills as the extractors run. The RAW signal tables do not. So when the
retriever comes back empty we build a second GroundedContext straight from
`signal_events` — the same rows `/signals/calendar/upcoming` and
`/signals/gmail/important` serve to the dashboard — and answer from that.

Two rules keep this honest:
  * These are FACTS with real lineage (`signal:<uuid>`), not summaries. They go
    through the same `GroundedContext` the model already only-ever-sees, so the
    no-fabrication contract is unchanged.
  * We only claim to have nothing when Calendar AND Gmail are also empty. Then
    "not enough signal" is true, and we say so — naming which sources are
    connected rather than blaming "memory".

LAYERING — this lives in `api`, and that is forced, not incidental. It reads
Signals and produces an `ml.rag` GroundedContext, and the two architecture
invariants in tests/ml/test_ml_layer.py forbid that combination anywhere else:

  * a truth-pipeline domain (`reasoning`, `signals`, …) may not import
    `navis.ml`  → so it cannot live in `reasoning`;
  * `navis.ml` may not import a pipeline domain (`signals`, …)
                → so it cannot live in `ml.rag` beside ContextBuilder.

`api` is the composition root that is allowed to depend on both — which is
exactly what "compose an answer out of memory OR raw signals" is.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from navis.identity.models import Integration, IntegrationProvider
from navis.ml.rag.grounding import Grounding, GroundedContext
from navis.platform.monitoring import get_logger
from navis.signals.base import SignalSource
from navis.signals.models import SignalEventRecord

logger = get_logger("api.workspace_fallback")

# Bounded on purpose. This path exists to answer "what's going on right now",
# not to dump the workspace into a prompt.
_CALENDAR_LOOKAHEAD_DAYS = 7
_GMAIL_LOOKBACK_DAYS = 7
# How far back to look for cancellations that kill an upcoming meeting. Must
# comfortably exceed the lookahead: a meeting 7 days out can be cancelled at any
# point before it, and a cancellation we miss re-animates a deleted meeting.
_CANCELLATION_LOOKBACK_DAYS = 90
_MAX_EVENTS = 10
_MAX_EMAILS = 12


async def build_workspace_context(
    db: AsyncSession,
    org_id: uuid.UUID,
    question: str,
    *,
    now: datetime | None = None,
) -> GroundedContext:
    """A GroundedContext built from raw Calendar + Gmail signals.

    Returns a context with `facts == []` when the org genuinely has nothing —
    the caller uses that to decide between "here's what I know" and an honest
    "no signal yet".
    """
    now = now or datetime.now(timezone.utc)
    facts: list[Grounding] = []
    try:
        facts.extend(await _calendar_facts(db, org_id, now))
    except Exception:  # noqa: BLE001 — a fallback that raises is worse than a thin one
        logger.exception("workspace fallback: calendar facts failed org=%s", org_id)
    try:
        facts.extend(await _gmail_facts(db, org_id, now))
    except Exception:  # noqa: BLE001
        logger.exception("workspace fallback: gmail facts failed org=%s", org_id)

    return GroundedContext(
        org_id=str(org_id),
        question=question,
        facts=facts,
        token_estimate=sum(len(f.text) for f in facts) // 4,
        as_of=now,
    )


async def _calendar_facts(
    db: AsyncSession, org_id: uuid.UUID, now: datetime
) -> list[Grounding]:
    """Upcoming meetings, newest-version-per-event, cancelled ones removed."""
    cancelled = await _cancelled_event_ids(db, org_id, now)
    rows = (
        await db.scalars(
            select(SignalEventRecord)
            .where(
                SignalEventRecord.organization_id == org_id,
                SignalEventRecord.source == SignalSource.CALENDAR.value,
                SignalEventRecord.occurred_at >= now,
                SignalEventRecord.occurred_at
                <= now + timedelta(days=_CALENDAR_LOOKAHEAD_DAYS),
            )
            .order_by(SignalEventRecord.occurred_at.asc())
            .limit(_MAX_EVENTS * 3)  # over-fetch: several versions per event
        )
    ).all()

    # Newest INGESTED version per event. Ordering by occurred_at (the meeting's
    # start time) cannot distinguish versions of the same meeting at all — every
    # version shares a start — so "first seen wins" was picking an arbitrary
    # edit. `created_at` is when we learned the fact, which is the only ordering
    # that means "most recent truth".
    rows = sorted(
        rows,
        key=lambda r: (getattr(r, "created_at", None) or r.occurred_at),
        reverse=True,
    )

    seen: set[str] = set()
    out: list[Grounding] = []
    for rec in rows:
        key = rec.external_id or str(rec.id)
        # Filtering `event_type != CANCELLED` on the ROW is not enough, and was
        # the bug: a cancelled meeting still has its original CREATED row, whose
        # occurred_at is the (future) meeting start, so it sails through the
        # window filter untouched. The cancellation must be matched to the
        # EVENT, not to a row.
        if key in cancelled:
            continue
        if key in seen:
            continue
        seen.add(key)
        payload = rec.payload or {}
        summary = str(payload.get("summary") or "(untitled meeting)")
        organizer = (payload.get("organizer") or {}).get("email")
        attendee_count = payload.get("attendee_count") or 0
        when = rec.occurred_at.isoformat() if rec.occurred_at else "unscheduled"
        detail = f"Calendar event: {summary} at {when}"
        if organizer:
            detail += f", organised by {organizer}"
        if attendee_count:
            detail += f", {attendee_count} attendee(s)"
        out.append(
            Grounding(
                kind="calendar",
                source=f"signal:{rec.id}",
                text=detail,
                occurred_at=rec.occurred_at,
            )
        )
        if len(out) >= _MAX_EVENTS:
            break
    # Restore chronological order — the newest-first sort above was only a
    # means of picking versions, not how a founder wants their day listed.
    out.sort(key=lambda g: (g.occurred_at is None, g.occurred_at))
    return out


async def _cancelled_event_ids(
    db: AsyncSession, org_id: uuid.UUID, now: datetime
) -> set[str]:
    """External ids of calendar events cancelled recently.

    Scoped to a bounded lookback rather than all history: a meeting cancelled
    last year cannot also be in the next 7 days, and an unbounded scan of
    signal_events is not something a chat turn should pay for. The window is
    generous relative to the lookahead so a cancellation recorded before the
    meeting's own start is always caught.
    """
    from navis.signals.base import SignalEventType

    rows = (
        await db.scalars(
            select(SignalEventRecord.external_id).where(
                SignalEventRecord.organization_id == org_id,
                SignalEventRecord.source == SignalSource.CALENDAR.value,
                SignalEventRecord.event_type
                == SignalEventType.CALENDAR_EVENT_CANCELLED.value,
                SignalEventRecord.created_at
                >= now - timedelta(days=_CANCELLATION_LOOKBACK_DAYS),
            )
        )
    ).all()
    return {r for r in rows if r}


async def _gmail_facts(
    db: AsyncSession, org_id: uuid.UUID, now: datetime
) -> list[Grounding]:
    """Recent inbound mail — the raw material for "who am I keeping waiting"."""
    from navis.signals.base import SignalEventType
    from navis.signals.read_router import _decrypt_content

    rows = (
        await db.scalars(
            select(SignalEventRecord)
            .where(
                SignalEventRecord.organization_id == org_id,
                SignalEventRecord.source == SignalSource.GMAIL.value,
                SignalEventRecord.event_type
                == SignalEventType.EMAIL_RECEIVED.value,
                SignalEventRecord.occurred_at
                >= now - timedelta(days=_GMAIL_LOOKBACK_DAYS),
            )
            .order_by(SignalEventRecord.occurred_at.desc())
            .limit(_MAX_EMAILS)
        )
    ).all()

    out: list[Grounding] = []
    for rec in rows:
        payload = _decrypt_content(rec.payload or {})
        subject = str(payload.get("subject") or "(no subject)")
        sender = payload.get("sender_email") or payload.get("sender_name") or "unknown"
        snippet = str(payload.get("snippet") or "").strip()
        text = f"Email from {sender}: {subject}"
        if snippet:
            text += f" — {snippet[:240]}"
        out.append(
            Grounding(
                kind="gmail",
                source=f"signal:{rec.id}",
                text=text,
                occurred_at=rec.occurred_at,
            )
        )
    return out


#: Provider -> the name a person recognises. One mapping, used by both readers
#: below, so a label can never differ between "what is connected" and "how
#: healthy is it".
_SOURCE_LABELS = {
    IntegrationProvider.GMAIL: "Gmail",
    IntegrationProvider.GOOGLE_CALENDAR: "Google Calendar",
    IntegrationProvider.DRIVE: "Google Drive",
    IntegrationProvider.SLACK: "Slack",
    IntegrationProvider.NOTION: "Notion",
}


@dataclass(frozen=True, slots=True)
class SourceHealth:
    """One integration's intent, auth health and ingestion recency.

    THE THREE FIELDS ARE THE THREE QUESTIONS, and the reason this type exists
    rather than a bare label list. `connected` is what the user wants,
    `auth_state` is whether the credential works, `last_success_at` is whether
    data is actually arriving — and a reader needs all three to tell these apart:

        connected=True,  auth_state=ok,                 recent success   HEALTHY
        connected=True,  auth_state=ok,                 old/no success   STALE
        connected=True,  auth_state=reconsent_required                   BROKEN
        connected=False                                                  DISCONNECTED

    Before this, every one of those rendered as "connected".
    """

    source: str
    label: str
    connected: bool
    auth_state: str
    last_attempt_at: datetime | None
    last_success_at: datetime | None
    last_failure_reason: str | None


async def source_health(db: AsyncSession, org_id: uuid.UUID) -> list[SourceHealth]:
    """Every integration this org has, with its health. THE single reader.

    Returns DISCONNECTED rows too, deliberately: "you disconnected Slack" and
    "you never connected Slack" are different answers to "why is there nothing
    from Slack?", and only the row can tell them apart.

    Ordered by label so the output is stable across calls — an unordered list
    that reshuffles between requests makes a diff in a UI or a test meaningless.
    """
    rows = (
        await db.scalars(
            select(Integration).where(Integration.organization_id == org_id)
        )
    ).all()
    out: list[SourceHealth] = []
    for r in rows:
        label = _SOURCE_LABELS.get(r.provider)
        if not label:
            continue  # a provider with no display name is not one we surface
        provider_value = r.provider.value if hasattr(r.provider, "value") else str(r.provider)
        out.append(
            SourceHealth(
                source=provider_value,
                label=label,
                connected=bool(r.is_active),
                auth_state=r.auth_state,
                last_attempt_at=r.last_attempt_at,
                last_success_at=r.last_success_at,
                last_failure_reason=r.last_failure_reason,
            )
        )
    out.sort(key=lambda s: s.label)
    return out


async def connected_source_labels(
    db: AsyncSession, org_id: uuid.UUID
) -> list[str]:
    """Human-readable names of the org's active integrations.

    Used to make the genuinely-empty message specific ("Gmail is connected but
    hasn't synced yet") instead of the blanket "not enough memory", which told
    the founder nothing about what to do next.

    UNCHANGED IN MEANING — still "which sources did the user connect", still
    filtered on `is_active` alone, still deduplicated by label. Every existing
    caller keeps the answer it had. It is now DERIVED from `source_health`
    rather than running its own query, so the two can never disagree about which
    sources exist; callers that need to know whether a connected source is
    actually WORKING call `source_health` directly.

    DEDUPLICATION IS LOAD-BEARING and is why this is not simply a list
    comprehension over `source_health`. One organization legitimately holds
    several rows for one provider — the Google-family clone writes a Gmail,
    Calendar and Drive row from a single grant, and a second Google account adds
    another set — so the row list repeats providers. This function names SYSTEMS
    to a person ("Gmail is connected but hasn't synced yet"), and "Gmail, Gmail"
    is not a sentence anybody wants to read. `source_health` deliberately does
    NOT dedupe: there, each row is a distinct connection with its own credential
    and its own health, and collapsing two accounts into one would hide a broken
    one behind a working one.
    """
    out: list[str] = []
    for s in await source_health(db, org_id):
        if s.connected and s.label not in out:
            out.append(s.label)
    return out
