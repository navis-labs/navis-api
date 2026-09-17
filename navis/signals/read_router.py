"""Signals read API — surfaces the newest captured signals to the frontend.

  GET  /signals/gmail/recent      latest emails from the org's Gmail signal stream
  GET  /signals/gmail/important   top-N ranked emails over a rolling window
  GET  /signals/calendar/upcoming meetings scheduled after now, latest version per event
  GET  /signals/drive/recent      recent Drive file activity, latest version per file

Post-connect UX: the moment Gmail / Calendar / Drive OAuth completes and the
sync pass writes rows into `signal_events`, the dashboard has to show "here's
what we're seeing". The sync pipeline already persists real Google data
(gmail/connector.py, calendar/connector.py, drive/connector.py); these routes
are the thin read seam over that persisted stream.

Read-only + tenant-scoped by design. Sealed payload fields (Gmail body/snippet
via `_content_fields`) are decrypted transparently on the way out so the
frontend never needs to know the storage-at-rest scheme. Any field that fails
decryption degrades to an empty string — one rotated-out key must never brick
the whole list.

Not a raw dump: fields are curated per source (subject/from/snippet for Gmail,
summary/times/attendees for Calendar, name/mime/event_type for Drive). Callers
that need extracted commitments / decisions use `/brief` or `/context`; this
router intentionally exposes the underlying raw signals only.
"""
from __future__ import annotations

import re
import uuid
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone
from typing import Any, TypeAlias

from fastapi import APIRouter, Depends, Query
from sqlalchemy import DateTime, cast, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from navis.identity.dependencies import CurrentContext, get_current_context
from navis.identity.models import Integration, IntegrationProvider
# THE identity function, not a copy of it. A meeting's entity id must be the
# same uuid5 Memory minted and `/context` publishes, or the two feeds describing
# one meeting cannot be joined — re-deriving the formula here would be a second
# identity space that agrees only by luck. `memory.ids` imports nothing but
# `uuid`, so this costs the signals layer no dependency; `signals/router.py`
# already reaches into `navis.memory` for the same kind of reason.
from navis.memory.envelope import EntityType
from navis.memory.ids import entity_id as entity_id_for
from navis.organization.contract import Capability
from navis.organization.guards import require_capability
from navis.platform.database import get_session
from navis.platform.monitoring.metrics import metrics
from navis.platform.security import FieldDecryptError, field_cipher
from navis.signals.base import SignalEventType, SignalSource
from navis.signals.models import SignalEventRecord

router = APIRouter(prefix="/signals", tags=["signals"])

# Result-size ceilings picked to keep a single response bounded (memory, JSON
# size, decrypt cost) without capping the frontend's typical "show me my
# inbox / calendar" query, which asks for ~20-50 rows at a time.
_MAX_LIMIT = 100
_DEFAULT_LIMIT = 10
# Upper bound on how far ahead /calendar/upcoming will scan. Two weeks covers
# the visible horizon for the dashboard's "upcoming meetings" card without
# pulling a whole recurring series' worth of instances into the response.
_MAX_DAYS_AHEAD = 30
_DEFAULT_DAYS_AHEAD = 14

# Window for /gmail/important. Anything older than a month is stale for a
# "what matters right now" list; the ceiling caps decrypt cost on the fetch.
_MAX_DAYS_BACK = 90
_DEFAULT_DAYS_BACK = 30
# Sampling cap for the important-ranking pass. We score in Python (label /
# regex / heuristics), so we bound how many rows we pull to keep the endpoint
# under ~200ms even on a very chatty inbox. 500 covers > 15 emails/day for a
# 30-day window — more than a normal executive inbox surfaces as "candidate
# important".
_IMPORTANT_CANDIDATE_CAP = 500

# Free/consumer domains — a "@gmail.com" sender is much less likely to be a
# real business counterparty than "@acme.com". Duplicated from the extractor
# (memory/extractors/email.py) so this scorer stays self-contained: the read
# path must not depend on extractor internals rotating out from under it.
_FREE_SENDER_DOMAINS = frozenset({
    "gmail.com", "googlemail.com", "yahoo.com", "outlook.com", "hotmail.com",
    "icloud.com", "proton.me", "protonmail.com", "aol.com", "live.com",
})

# Gmail categories that are almost never "important business mail". PROMOTIONS
# and SOCIAL are hard-skipped (100% junk for our use case). UPDATES / FORUMS
# get a heavy penalty but stay eligible — a shipping receipt or a critical
# vendor notification occasionally hides there.
_HARD_SKIP_LABELS = frozenset({"CATEGORY_PROMOTIONS", "CATEGORY_SOCIAL"})
_SOFT_PENALTY_LABELS = frozenset({"CATEGORY_UPDATES", "CATEGORY_FORUMS"})

# Commitment / decision language — same intent as the memory extractor's
# regex, kept local to insulate this file from extractor evolution. If the
# sender promised something ("I'll send…", "by EOD") or a decision was made
# ("we decided", "approved"), the email is far more likely to be actionable.
_COMMITMENT_RE = re.compile(
    r"\b(i'?ll|we'?ll|i will|we will|follow(?:ing)? up|get back to you|"
    r"send (?:you|over|across)|circle back|"
    r"by (?:eod|end of day|tomorrow|monday|tuesday|wednesday|thursday|friday|next week))\b",
    re.IGNORECASE,
)
_DECISION_RE = re.compile(
    r"\b(we'?ve decided|we have decided|we decided|decision:|"
    r"we'?ve agreed|approved|we will go with|let'?s go with)\b",
    re.IGNORECASE,
)


async def _is_connected(
    db: AsyncSession, *, org_id: uuid.UUID, provider: IntegrationProvider
) -> bool:
    """True iff an active Integration exists for (org, provider). The frontend
    uses this to distinguish "not connected yet" from "connected, no data" —
    both return an empty items list, but the UX copy differs."""
    row = await db.scalar(
        select(Integration.id).where(
            Integration.organization_id == org_id,
            Integration.provider == provider,
            Integration.is_active.is_(True),
        )
    )
    return row is not None


def _decrypt_content(payload: dict[str, Any]) -> dict[str, Any]:
    """Decrypt the fields sealed at capture time (self-described via
    `_content_fields`). Mirrors memory.service._decrypt_content so this route
    and the extractor share one decryption contract — a field that fails to
    decrypt degrades to empty rather than 500'ing the whole list."""
    fields = payload.get("_content_fields")
    if not fields:
        return payload
    out = dict(payload)
    for f in fields:
        val = out.get(f)
        if isinstance(val, str):
            try:
                out[f] = field_cipher.decrypt(val)
            except FieldDecryptError:
                metrics.incr("signals.read.content_decrypt_failed")
                out[f] = ""
    return out


# UTC-aware epoch used as the "unknown timestamp" fallback in the sort key.
# `datetime.min` is naive; comparing a naive value against the tz-aware
# `occurred_at` from the DB raises TypeError inside `sort`.
_EPOCH_UTC = datetime.min.replace(tzinfo=timezone.utc)


def _important_sort_key(
    entry: tuple[int, _SignalRow, dict[str, Any]],
) -> tuple[datetime, int]:
    """Ordering for /gmail/important results.

    Sort by recency FIRST, score SECOND (both descending). Score-first
    ordering was structurally biased toward older content — Gmail's
    IMPORTANT label + user stars accumulate over time, so a 4-week-old
    thread reliably out-scored a fresh mid-day arrival and the feed jumped
    4wk → 1d → 2wk → 4d instead of walking backwards in time. Score
    stays as a tiebreaker so two emails from the same instant (batched
    ingest, unlikely but possible) render deterministically.
    """
    score, rec, _ = entry
    return (rec.occurred_at or _EPOCH_UTC, score)


def _score_email(payload: dict[str, Any]) -> int | None:
    """Rank an email for the "important" list. Returns None to hard-skip
    (junk we never want on the dashboard); returns an int score otherwise.

    Pure function over the DECRYPTED payload shape produced by
    `_decrypt_content` + gmail/parser.py. Every signal used here is already
    present at capture time, so the endpoint stays a pure read path — no
    LLM, no I/O. The weights are heuristic and will be tuned once we track
    dismissal rate (see the roadmap for the LLM-backed follow-up).
    """
    labels = set(payload.get("label_ids") or [])
    # Hard-skip promotions/social — they poison "top 10 important" completely.
    if labels & _HARD_SKIP_LABELS:
        return None
    # Mailing-list / auto-generated headers: newsletters, digests, CI bots.
    # These fill an inbox but almost never demand attention.
    if payload.get("list_id"):
        return None
    if str(payload.get("auto_submitted") or "").lower() not in ("", "no"):
        return None
    if str(payload.get("precedence") or "").lower() in ("bulk", "list", "junk"):
        return None

    score = 0
    # Gmail's own IMPORTANT / STARRED bits are the strongest single signal —
    # they encode Google's inbox ML plus the user's own explicit action.
    if "IMPORTANT" in labels:
        score += 10
    if "STARRED" in labels:
        score += 8
    if labels & _SOFT_PENALTY_LABELS:
        score -= 4

    # Content-based signals — need the decrypted body/snippet. Missing body
    # (e.g. body_length == 0) still lets us score off labels + metadata.
    body = payload.get("body_text") or ""
    snippet = payload.get("snippet") or ""
    text = f"{body}\n{snippet}"
    if _DECISION_RE.search(text):
        score += 6
    if _COMMITMENT_RE.search(text):
        score += 5

    # Attachments correlate strongly with real work (contracts, decks, specs)
    # vs. chatter. One is enough; more doesn't linearly mean more important.
    if payload.get("attachments"):
        score += 3

    # Mid-thread reply — the conversation is live, someone is waiting on a
    # response. `in_reply_to` is the RFC header the parser captures.
    if payload.get("in_reply_to"):
        score += 3

    # Business domain sender > free-mail sender. Consumer-domain senders
    # can still be important (a customer emailing from gmail.com) so this is
    # a nudge, not a gate.
    sender = str(payload.get("sender_email") or "").lower()
    if "@" in sender:
        domain = sender.rsplit("@", 1)[1]
        if domain and domain not in _FREE_SENDER_DOMAINS:
            score += 2

    # Direct-addressed (short TO, no CC) usually means "for you specifically",
    # not a broadcast. A CC of 20 people is almost always FYI.
    to_count = len(payload.get("to_emails") or [])
    cc_count = len(payload.get("cc_emails") or [])
    if 1 <= to_count <= 3 and cc_count <= 2:
        score += 2
    if to_count + cc_count >= 15:
        score -= 3

    return score


# What the `_shape_*` helpers accept. Two things get shaped: real ORM rows
# (Gmail's straight select) and `_RowShim`s over a DISTINCT ON subquery mapping
# (Calendar/Drive). Annotating them with the concrete `SignalEventRecord` alone
# made every DISTINCT ON call site a type error even though it is the intended
# design — see `_RowShim` at the bottom of this module.
#
# A structural Protocol looks like the tidier answer here and does NOT work:
#   * the ORM declares `id: Mapped[uuid.UUID]`, and protocol matching compares
#     that declared type against `uuid.UUID` rather than the descriptor's value
#     type, so SignalEventRecord fails to match; and
#   * `_RowShim` serves its attributes through `__getattr__`, which type
#     checkers do not use to satisfy protocol members, so it fails too.
# A plain union matches both nominally, which is all this needs.
#
# `from __future__ import annotations` makes annotations lazy strings, so the
# alias may name `_RowShim` before its definition without a runtime NameError.
_SignalRow: TypeAlias = "SignalEventRecord | _RowShim"


def _shape_gmail_from_decrypted(rec: _SignalRow, p: dict[str, Any]) -> dict[str, Any]:
    """Shape a Gmail record whose payload is ALREADY decrypted. Used by
    /gmail/important, which decrypts once for scoring and reuses the same
    dict for shaping — decrypt cost is real on large windows."""
    return {
        "event_id":        str(rec.id),
        "external_id":     rec.external_id,
        "thread_id":       p.get("thread_id"),
        "subject":         p.get("subject"),
        "from_email":      p.get("sender_email"),
        "from_name":       p.get("sender_name"),
        "to_emails":       p.get("to_emails") or [],
        "snippet":         p.get("snippet"),
        "received_at":     rec.occurred_at.isoformat() if rec.occurred_at else None,
        "has_attachments": bool(p.get("attachments")),
        "label_ids":       p.get("label_ids") or [],
    }


def _shape_gmail(rec: _SignalRow) -> dict[str, Any]:
    return _shape_gmail_from_decrypted(rec, _decrypt_content(rec.payload or {}))


# ── The provider-neutral envelope ───────────────────────────────────────────
#
# Every item a surface consumes should describe WHAT KIND OF THING it is and
# WHICH SYSTEM produced it, as two separate facts. `source_type` is the
# organizational concept ("calendar", and later "message", "issue", "document");
# `provider` is the system that observed it ("google_calendar", and later
# "outlook_calendar", "slack", "jira"). A consumer branches on the CONCEPT and
# never on the provider — which is what stops a future Jira or Slack feed from
# needing its own renderer, and stops today's calendar card from being a Google
# card wearing a generic name.
#
# The distinction is not cosmetic here: `SignalSource.CALENDAR` is already the
# concept and `IntegrationProvider.GOOGLE_CALENDAR` is already the system, so
# both values are read from the vocabularies the pipeline already keeps rather
# than introduced by this module.
_CALENDAR_SOURCE_TYPE = SignalSource.CALENDAR.value
_CALENDAR_PROVIDER = IntegrationProvider.GOOGLE_CALENDAR.value


def _timed(slot: Any) -> str | None:
    """The ISO-8601 instant a Google start/end slot names, or None if all-day.

    Google expresses a timed event as `{"dateTime": "...", "timeZone": "..."}`
    and an all-day event as `{"date": "YYYY-MM-DD"}`. None here means ALL-DAY,
    not missing — `all_day` in the shaped item is what says which, so a consumer
    never has to infer it from a null.

    Deliberately identical to `memory.extractors.calendar.MeetingExtractor`,
    which reads `(start or {}).get("dateTime")` into `starts_at`. That is the
    point: the same meeting must not have one set of field names in Memory and
    another on the signals read path.
    """
    if not isinstance(slot, dict):
        return None
    value = slot.get("dateTime")
    return value if isinstance(value, str) else None


def _is_all_day(slot: Any) -> bool:
    return isinstance(slot, dict) and not slot.get("dateTime") and bool(slot.get("date"))


def _shape_calendar(rec: _SignalRow, *, org_id: uuid.UUID) -> dict[str, Any]:
    """One upcoming meeting, in the names the rest of the pipeline already uses.

    WHY THE CANONICAL NAMES ARE HERE. This endpoint used to publish Google's own
    shape — `summary`, and `start`/`end` as raw `{dateTime|date, timeZone}`
    objects — while Memory, State and every reasoning engine call the same three
    things `title`, `starts_at` and `ends_at`. One meeting therefore had two
    representations inside one backend, and the consumer was the one made to
    reconcile them: a client reading this feed had to know Google's slot format
    AND guess which of `start`, `start_time`, `starts_at` or `scheduled_at` a
    given surface meant. The canonical names now come from the backend, once.

    STRICTLY ADDITIVE. `summary`, `start`, `end` and `hangout_link` are still
    published unchanged — an all-day event's date lives in `start.date`, which
    the canonical fields deliberately do not flatten into a timestamp — so no
    existing consumer changes. New consumers read the canonical names.

    `entity_id` IS THE JOIN, and it is the field whose absence made "no meeting
    context available yet" unanswerable. Memory keys a meeting entity as
    uuid5(org, "meeting", <calendar event id>) — `memory.ids.entity_id`, the one
    identity function in the system — and `GET /context` publishes exactly that
    id for every MEETING entity State holds, with its commitments, decisions and
    citations. Without it here, the two feeds described the same meeting and
    could not be joined, so a client had no way to distinguish "Navis knows
    nothing about this meeting" from "Navis knows something and this feed cannot
    reach it". It is COMPUTED, not stored: a pure function of (org, event id),
    so it is correct even for a meeting Memory has not extracted yet — in which
    case `/context` simply holds no such entity, which is the honest absence.

    NO DEEP LINK IS INVENTED. Google's `htmlLink` is not captured by the
    connector, so this feed cannot produce one; `meeting_url` is the conference
    link and is null when the meeting has none. A surface that wants "open in
    the provider" needs the connector to capture it first.
    """
    p = rec.payload or {}
    organizer = p.get("organizer") or {}
    conference = p.get("conference_data") or {}
    start, end = p.get("start"), p.get("end")
    return {
        "event_id":       str(rec.id),
        "external_id":    rec.external_id,
        # ── provider-neutral envelope (see above) ──────────────────────────
        "source_type":    _CALENDAR_SOURCE_TYPE,
        "provider":       _CALENDAR_PROVIDER,
        "event_type":     rec.event_type,
        # The organizational entity this signal is about, in Memory's id space.
        "entity_type":    EntityType.MEETING.value,
        "entity_id":      str(entity_id_for(org_id, EntityType.MEETING.value, rec.external_id)),
        # ── canonical meeting fields (same names as Memory/State) ──────────
        "title":          p.get("summary"),
        "starts_at":      _timed(start),
        "ends_at":        _timed(end),
        # True when the event is a whole-day block, which is why `starts_at` /
        # `ends_at` are null. Without this a consumer cannot tell an all-day
        # meeting from one whose times were never captured.
        "all_day":        _is_all_day(start),
        "meeting_url":    p.get("hangout_link"),
        # WHEN THE THING HAPPENS vs WHEN WE SAW IT. `occurred_at` for a calendar
        # signal is the meeting's start (see `_latest_per_external_id_stmt`);
        # `observed_at` is when this version was ingested. A surface that treats
        # ingestion time as event time reorders the day.
        "observed_at":    rec.created_at.isoformat() if rec.created_at else None,
        # ── unchanged, provider-shaped, still published ────────────────────
        "summary":        p.get("summary"),
        "status":         p.get("status"),
        "start":          start,
        "end":            end,
        "attendees_count": p.get("attendee_count") or 0,
        # ADDITIVE: the emails behind the count. A meeting card that can only say
        # "2 attendees" cannot name who is in the room, which is the one thing the
        # founder needs before joining. Emails only — the connector already drops
        # attendee displayName as bulk PII, and nothing here re-adds it.
        "attendees":      [a.get("email") for a in (p.get("attendees") or []) if a.get("email")],
        "organizer_email": organizer.get("email"),
        "has_conference": bool(conference) or bool(p.get("hangout_link")),
        "hangout_link":   p.get("hangout_link"),
        "occurred_at":    rec.occurred_at.isoformat() if rec.occurred_at else None,
    }


def _shape_drive(rec: _SignalRow) -> dict[str, Any]:
    p = rec.payload or {}
    return {
        "event_id":    str(rec.id),
        "external_id": rec.external_id,
        "name":        p.get("name"),
        "mime_type":   p.get("mime_type"),
        "event_type":  rec.event_type,
        "owners":      [o.get("email") for o in (p.get("owners") or []) if o.get("email")],
        "shared":      bool(p.get("shared")),
        "trashed":     bool(p.get("trashed")),
        "modified_at": rec.occurred_at.isoformat() if rec.occurred_at else None,
    }


def _latest_per_external_id_stmt(*, org_id: uuid.UUID, source: SignalSource):
    """PostgreSQL DISTINCT ON: newest row per external_id. Calendar and Drive
    are version-aware (etag / Drive `version`), so every edit creates a new
    signal_events row for the same external_id. The frontend list wants one
    entry per meeting / file — the latest one — so we collapse versions here
    rather than pushing that logic onto every consumer.

    "Latest" is INGESTION order (`created_at`), not `occurred_at`. That
    distinction is load-bearing: for a calendar signal `occurred_at` is the
    MEETING'S START TIME (see CalendarConnector._to_event), not when we learned
    about it. Cancelling a meeting three weeks out writes a cancellation row
    stamped `now`, which is EARLIER than the original row's start — so ordering
    by occurred_at picked the stale CREATED row, and the caller's
    `event_type != CALENDAR_EVENT_CANCELLED` filter then had nothing to catch.
    The cancelled meeting stayed on the dashboard permanently, even after a
    full sync. Rescheduling a meeting to an earlier slot had the same failure.

    `id` breaks ties: signal ids are deterministic UUIDv5, so two rows written
    in the same transaction still collapse to a stable, replay-identical pick
    instead of whichever the planner happened to emit first.
    """
    return (
        select(SignalEventRecord)
        .where(
            SignalEventRecord.organization_id == org_id,
            SignalEventRecord.source == source.value,
        )
        .distinct(SignalEventRecord.external_id)
        .order_by(
            SignalEventRecord.external_id,
            SignalEventRecord.created_at.desc(),
            SignalEventRecord.id.desc(),
        )
    ).subquery()


async def search_emails(
    db: AsyncSession,
    *,
    org_id: uuid.UUID,
    senders: Sequence[str] = (),
    text_match: str | None = None,
    limit: int = _DEFAULT_LIMIT,
) -> dict[str, Any]:
    """THE org's captured mail, newest first, optionally narrowed. ONE reader.

    Extracted from `gmail_recent` so the HTTP route and Ask Navis answer "find
    the email from X" out of the SAME implementation. The route below is now a
    thin adapter over this; nothing about its response changed.

    WHY THE FILTERS ARE SQL PREDICATES AND WHAT THEY CAN SEE. Only `body_text`
    and `snippet` are sealed at capture (`_content_fields`); `sender_email`,
    `sender_name` and `subject` are stored as plaintext JSONB because identity
    resolution and the importance scorer both read them. So sender and subject
    narrow in the database, over the whole corpus, with no decryption — while
    BODY TEXT IS DELIBERATELY NOT SEARCHED. Matching it would mean decrypting
    every candidate row in Python to filter, which turns a bounded query into a
    full scan of sealed content; a caller that needs body relevance should use
    Memory retrieval, which is what the embeddings exist for.

    `senders` are matched case-insensitively against BOTH the address and the
    display name, OR-ed together — the caller passes every alias the identity
    graph resolved for one person, so a person who mails from two addresses is
    still one person here. Empty means "no sender filter", never "match nobody".

    Org-scoped in the statement, like every other read in this module.
    """
    connected = await _is_connected(db, org_id=org_id, provider=IntegrationProvider.GMAIL)
    stmt = select(SignalEventRecord).where(
        SignalEventRecord.organization_id == org_id,
        SignalEventRecord.source == SignalSource.GMAIL.value,
        SignalEventRecord.event_type == SignalEventType.EMAIL_RECEIVED.value,
    )
    if senders:
        stmt = stmt.where(
            or_(
                *[
                    cond
                    for s in senders
                    if s
                    for cond in (
                        SignalEventRecord.payload["sender_email"].astext.ilike(
                            f"%{_like_escape(s)}%", escape="\\"
                        ),
                        SignalEventRecord.payload["sender_name"].astext.ilike(
                            f"%{_like_escape(s)}%", escape="\\"
                        ),
                    )
                ]
            )
        )
    if text_match:
        stmt = stmt.where(
            SignalEventRecord.payload["subject"].astext.ilike(
                f"%{_like_escape(text_match)}%", escape="\\"
            )
        )
    rows = (
        await db.scalars(
            stmt.order_by(SignalEventRecord.occurred_at.desc()).limit(limit)
        )
    ).all()
    items = [_shape_gmail(r) for r in rows]
    return {"source": "gmail", "connected": connected, "count": len(items), "items": items}


def _like_escape(value: str) -> str:
    """Neutralise LIKE metacharacters so a user's `%` cannot become a wildcard.

    Same escaping `identity_resolution.search` already applies for the same
    reason; `\\` is Postgres's default LIKE escape and is declared explicitly.
    """
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


@router.get(
    "/gmail/recent",
    dependencies=[Depends(require_capability(Capability.GMAIL))],
)
async def gmail_recent(
    limit: int = Query(_DEFAULT_LIMIT, ge=1, le=_MAX_LIMIT),
    ctx: CurrentContext = Depends(get_current_context),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """Return the org's most recent captured Gmail messages (newest first).

    Read straight off `signal_events`. Gmail's external_version is constant
    ("1" — an email is an immutable fact) so external_id is already unique
    per (org, integration); no DISTINCT ON needed. Bodies/snippets are
    encrypted at rest and decrypted on the way out for the caller.

    An HTTP adapter over `search_emails` — the tenant comes from the
    authenticated context and the unfiltered call is the original behaviour.
    """
    return await search_emails(db, org_id=ctx.org_id, limit=limit)


@router.get(
    "/gmail/important",
    dependencies=[Depends(require_capability(Capability.GMAIL))],
)
async def gmail_important(
    limit: int = Query(_DEFAULT_LIMIT, ge=1, le=_MAX_LIMIT),
    days_back: int = Query(_DEFAULT_DAYS_BACK, ge=1, le=_MAX_DAYS_BACK),
    ctx: CurrentContext = Depends(get_current_context),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """Top-`limit` emails from the last `days_back` days, ranked by
    heuristic importance (Gmail labels, sender, thread state, content
    signals). Skips promotions/social/mailing-list traffic entirely.

    Ranking is deterministic and versioned by `_score_email` — no LLM in
    this path. Sealed body/snippet fields are decrypted first so the scorer
    can inspect commitment/decision language before shaping the response.
    """
    connected = await _is_connected(db, org_id=ctx.org_id, provider=IntegrationProvider.GMAIL)
    since = datetime.now(timezone.utc) - timedelta(days=days_back)
    # Pull a bounded candidate window and rank in Python. Doing the sort in
    # SQL would require the scorer as a stored function — the heuristics
    # will churn, so we keep them in Python behind the cap.
    rows = (
        await db.scalars(
            select(SignalEventRecord)
            .where(
                SignalEventRecord.organization_id == ctx.org_id,
                SignalEventRecord.source == SignalSource.GMAIL.value,
                SignalEventRecord.event_type == SignalEventType.EMAIL_RECEIVED.value,
                SignalEventRecord.occurred_at >= since,
            )
            .order_by(SignalEventRecord.occurred_at.desc())
            .limit(_IMPORTANT_CANDIDATE_CAP)
        )
    ).all()

    scored: list[tuple[int, SignalEventRecord, dict[str, Any]]] = []
    for r in rows:
        decrypted = _decrypt_content(r.payload or {})
        score = _score_email(decrypted)
        if score is None or score <= 0:
            # score == 0 means "no positive signal at all" — leaving those
            # out is deliberate: better empty than misleading.
            continue
        scored.append((score, r, decrypted))

    scored.sort(key=_important_sort_key, reverse=True)
    top = scored[:limit]

    items: list[dict[str, Any]] = []
    for score, rec, decrypted in top:
        shaped = _shape_gmail_from_decrypted(rec, decrypted)
        shaped["importance_score"] = score
        items.append(shaped)
    return {
        "source": "gmail",
        "connected": connected,
        "window_days": days_back,
        "candidates_scanned": len(rows),
        "count": len(items),
        "items": items,
    }


@router.get(
    "/calendar/upcoming",
    dependencies=[Depends(require_capability(Capability.CALENDAR))],
)
async def calendar_upcoming(
    limit: int = Query(_DEFAULT_LIMIT, ge=1, le=_MAX_LIMIT),
    days_ahead: int = Query(_DEFAULT_DAYS_AHEAD, ge=1, le=_MAX_DAYS_AHEAD),
    ctx: CurrentContext = Depends(get_current_context),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """Upcoming meetings scheduled in the next `days_ahead` days.

    The signal_events table stores one row per event version (etag-driven),
    so we DISTINCT ON external_id to collapse a series of edits to the
    latest snapshot. Cancelled events are excluded — the UI card is for
    "what's on the calendar", not an audit trail.

    A meeting that has already STARTED but not yet ENDED is still returned.
    `occurred_at` is the meeting's start (see `_latest_per_external_id_stmt`),
    so gating on `occurred_at >= now` deleted a meeting from the dashboard the
    instant it began — exactly when the founder opens Today to join it. The
    window is therefore closed on the END time, falling back to the start when
    an event carries no parseable end (all-day events use `date`, not
    `dateTime`).
    """
    connected = await _is_connected(
        db, org_id=ctx.org_id, provider=IntegrationProvider.GOOGLE_CALENDAR
    )
    now = datetime.now(timezone.utc)
    horizon = now + timedelta(days=days_ahead)

    latest = _latest_per_external_id_stmt(org_id=ctx.org_id, source=SignalSource.CALENDAR)
    # payload->'end'->>'dateTime' is an ISO-8601 string with offset; Postgres
    # casts it to timestamptz directly. COALESCE keeps all-day / malformed
    # events behaving exactly as before rather than dropping them.
    ends_at = func.coalesce(
        cast(latest.c.payload["end"]["dateTime"].astext, DateTime(timezone=True)),
        latest.c.occurred_at,
    )
    stmt = (
        select(latest)
        .where(
            latest.c.event_type != SignalEventType.CALENDAR_EVENT_CANCELLED.value,
            ends_at >= now,
            latest.c.occurred_at <= horizon,
        )
        .order_by(latest.c.occurred_at.asc())
        .limit(limit)
    )
    result = await db.execute(stmt)
    items = [
        _shape_calendar(_RowShim(row), org_id=ctx.org_id)
        for row in result.mappings().all()
    ]
    return {
        "source": "calendar",
        # The same two facts every item carries, hoisted so a consumer can route
        # the whole feed without inspecting a row — and so an EMPTY feed still
        # says what kind of thing it would have held.
        "source_type": _CALENDAR_SOURCE_TYPE,
        "provider": _CALENDAR_PROVIDER,
        "connected": connected,
        "count": len(items),
        "items": items,
    }


@router.get(
    "/drive/recent",
    dependencies=[Depends(require_capability(Capability.DRIVE))],
)
async def drive_recent(
    limit: int = Query(_DEFAULT_LIMIT, ge=1, le=_MAX_LIMIT),
    ctx: CurrentContext = Depends(get_current_context),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """Most recent Drive file activity (latest version per file).

    Files are versioned by Drive's monotonic `version` counter, so multiple
    rows exist per file over its lifetime. DISTINCT ON collapses to the
    newest observation. Trashed files are surfaced so the UI can render a
    "removed" state; callers can filter client-side if they only want live
    files."""
    connected = await _is_connected(db, org_id=ctx.org_id, provider=IntegrationProvider.DRIVE)

    latest = _latest_per_external_id_stmt(org_id=ctx.org_id, source=SignalSource.DRIVE)
    stmt = (
        select(latest)
        .order_by(latest.c.occurred_at.desc())
        .limit(limit)
    )
    result = await db.execute(stmt)
    items = [_shape_drive(_RowShim(row)) for row in result.mappings().all()]
    return {"source": "drive", "connected": connected, "count": len(items), "items": items}


class _RowShim:
    """Minimal attribute shim over a SQLAlchemy row mapping so the `_shape_*`
    helpers can be reused for both ORM instances (Gmail's straight select) and
    subquery rows (Calendar/Drive's DISTINCT ON path). Only exposes the
    attributes the shapers read — nothing else, so a caller can't accidentally
    treat it as a full ORM object."""

    __slots__ = ("_row",)

    def __init__(self, row: Mapping[Any, Any]) -> None:
        self._row = row

    def __getattr__(self, name: str) -> Any:
        try:
            return self._row[name]
        except KeyError as exc:
            raise AttributeError(name) from exc
