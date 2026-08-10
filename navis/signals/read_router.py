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
from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from navis.identity.dependencies import CurrentContext, get_current_context
from navis.identity.models import Integration, IntegrationProvider
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
    entry: tuple[int, "SignalEventRecord", dict[str, Any]],
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


def _shape_gmail_from_decrypted(rec: SignalEventRecord, p: dict[str, Any]) -> dict[str, Any]:
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


def _shape_gmail(rec: SignalEventRecord) -> dict[str, Any]:
    return _shape_gmail_from_decrypted(rec, _decrypt_content(rec.payload or {}))


def _shape_calendar(rec: SignalEventRecord) -> dict[str, Any]:
    p = rec.payload or {}
    organizer = p.get("organizer") or {}
    conference = p.get("conference_data") or {}
    return {
        "event_id":       str(rec.id),
        "external_id":    rec.external_id,
        "summary":        p.get("summary"),
        "status":         p.get("status"),
        "start":          p.get("start"),
        "end":            p.get("end"),
        "attendees_count": p.get("attendee_count") or 0,
        "organizer_email": organizer.get("email"),
        "has_conference": bool(conference) or bool(p.get("hangout_link")),
        "hangout_link":   p.get("hangout_link"),
        "occurred_at":    rec.occurred_at.isoformat() if rec.occurred_at else None,
    }


def _shape_drive(rec: SignalEventRecord) -> dict[str, Any]:
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
    """
    connected = await _is_connected(db, org_id=ctx.org_id, provider=IntegrationProvider.GMAIL)
    rows = (
        await db.scalars(
            select(SignalEventRecord)
            .where(
                SignalEventRecord.organization_id == ctx.org_id,
                SignalEventRecord.source == SignalSource.GMAIL.value,
                SignalEventRecord.event_type == SignalEventType.EMAIL_RECEIVED.value,
            )
            .order_by(SignalEventRecord.occurred_at.desc())
            .limit(limit)
        )
    ).all()
    items = [_shape_gmail(r) for r in rows]
    return {"source": "gmail", "connected": connected, "count": len(items), "items": items}


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
    """
    connected = await _is_connected(
        db, org_id=ctx.org_id, provider=IntegrationProvider.GOOGLE_CALENDAR
    )
    now = datetime.now(timezone.utc)
    horizon = now + timedelta(days=days_ahead)

    latest = _latest_per_external_id_stmt(org_id=ctx.org_id, source=SignalSource.CALENDAR)
    stmt = (
        select(latest)
        .where(
            latest.c.event_type != SignalEventType.CALENDAR_EVENT_CANCELLED.value,
            latest.c.occurred_at >= now,
            latest.c.occurred_at <= horizon,
        )
        .order_by(latest.c.occurred_at.asc())
        .limit(limit)
    )
    result = await db.execute(stmt)
    items = [_shape_calendar(_RowShim(row)) for row in result.mappings().all()]
    return {"source": "calendar", "connected": connected, "count": len(items), "items": items}


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

    def __init__(self, row: dict[str, Any]) -> None:
        self._row = row

    def __getattr__(self, name: str) -> Any:
        try:
            return self._row[name]
        except KeyError as exc:
            raise AttributeError(name) from exc
