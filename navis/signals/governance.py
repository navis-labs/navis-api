"""Signal-layer data governance: retention enforcement (M-1) + per-subject erasure (CR-2).

Both reduce to ONE primitive — tombstoning the self-described content fields of a signal —
so retention (time-based) and erasure (subject-based) share it. Tombstoning (not row-delete)
is the event-sourcing-safe form of forgetting: the event SKELETON (id, occurred_at, source,
type, lineage) survives so replay/audit stay intact, while the sensitive payload is gone.

Erasure architecture (recommended HYBRID):
  * LIVE data — physical content tombstone (this module): deterministic, replay-safe,
    idempotent, audited. Removes the personal data from primary storage + replicas now.
  * BACKUPS / cold storage — cryptographic erasure (follow-up): per-subject content keys so
    deleting the key renders historical ciphertext (incl. old backups) permanently unreadable
    WITHOUT mutating immutable archives. Requires per-subject key management — flagged, not
    built here, because it is new key infrastructure.

Pure transforms below are unit-tested without a DB; the DB orchestration reuses the existing
SignalRepository, AuditService, metrics, and (for retention) Celery beat.
"""
from __future__ import annotations

import hashlib
import hmac
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from sqlalchemy.ext.asyncio import AsyncSession

from navis.identity.audit import AuditService
from navis.platform.config import settings
from navis.platform.monitoring import get_logger
from navis.platform.monitoring.metrics import metrics
from navis.signals.repository import SignalRepository

logger = get_logger("signals.governance")


# ---- Erasure cascade contract (P1) -----------------------------------------------------
# The DURABLE cascade contract is the audit row (replay-safe, queryable). This is an
# additional in-process emission SEAM so future Memory/State layers can subscribe and erase
# their own projections — NO queue, NO bus, NO coupling; subscribers are optional + isolated.
@dataclass(frozen=True, slots=True)
class SubjectErasureNotice:
    """SIGNAL_SUBJECT_ERASURE — a subject's data was tombstoned in the signal log. The
    `email`/`slack_user_id` are carried ONLY to drive the in-process cascade (so downstream
    layers can erase their projections in the SAME transaction); they are never persisted or
    logged — the durable record uses `subject_handle` (the keyed, non-reversible hash)."""

    org_id: uuid.UUID
    subject_handle: str
    erased: int
    email: str | None = None
    slack_user_id: str | None = None


# Async subscribers get the notice + the live db session so they erase atomically with signals.
ErasureSubscriber = Callable[["SubjectErasureNotice", AsyncSession], Any]
_ERASURE_SUBSCRIBERS: list[ErasureSubscriber] = []


def on_subject_erasure(fn: ErasureSubscriber) -> ErasureSubscriber:
    """Register a subscriber (e.g. Memory erasure). Usable as a decorator."""
    _ERASURE_SUBSCRIBERS.append(fn)
    return fn


async def _notify_erasure(notice: SubjectErasureNotice, db: AsyncSession) -> None:
    for fn in _ERASURE_SUBSCRIBERS:
        try:
            await fn(notice, db)
        except Exception:  # pragma: no cover - a subscriber must never break the erasure
            logger.exception("subject-erasure subscriber failed")

# A scrubbed field is overwritten with this — distinguishable from a missing/None value, so
# replay + Memory can tell "this was erased" from "this was never present".
TOMBSTONE = "__erased__"

# ---- retention policy: how long CONTENT (free-text / PII body) may live, by retention_class,
# with a per-source fallback for signals that predate retention tagging (e.g. Gmail). None =
# no content to expire (metadata-only events) or managed by erasure, not TTL. Metadata + the
# event skeleton are retained far longer (the row itself is not deleted here).
_CONTENT_TTL_DAYS_BY_CLASS: dict[str, int | None] = {
    "slack_message": 90,
    "slack_file": 90,
    "slack_reaction": 30,          # high volume, low value -> shortest window
    "slack_channel_event": None,   # no free-text content
    "slack_identity": None,        # snapshots: identity, governed by erasure not TTL
}
# Per-source content TTL. EVERY known source is listed explicitly — no silent default.
#   gmail / slack : 90d — carry free-text bodies/messages (sensitive PII content).
#   calendar / drive / docs / meet : None — capture METADATA only (titles/filenames already
#       minimized + encrypted at capture); there is no free-text body to age out, and their
#       identity data is governed by erasure, not a content TTL.
_CONTENT_TTL_DAYS_BY_SOURCE: dict[str, int | None] = {
    "gmail": 90, "slack": 90, "calendar": None, "drive": None, "docs": None, "meet": None,
}


def content_retention_days(source: str, retention_class: str | None) -> int | None:
    """Days after which a signal's CONTENT should be tombstoned (None = keep).

    An UNKNOWN source never silently returns None: it is loud (metric + warning) and
    conservative (keep, don't delete unclassified content) so a new connector gets a policy
    before its data ages out — fail-closed toward preservation + visibility."""
    if retention_class in _CONTENT_TTL_DAYS_BY_CLASS:
        return _CONTENT_TTL_DAYS_BY_CLASS[retention_class]
    if source in _CONTENT_TTL_DAYS_BY_SOURCE:
        return _CONTENT_TTL_DAYS_BY_SOURCE[source]
    metrics.incr("signals.retention.unknown_source", labels={"source": str(source)})
    logger.warning("retention: source %r has no content-TTL policy; keeping (classify it)", source)
    return None


def scrub_content_fields(payload: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Tombstone the self-described encrypted content fields. Idempotent: a row already marked
    `_scrubbed` is a no-op (so replay/retries never double-apply). Returns (new_payload,
    changed)."""
    fields = payload.get("_content_fields")
    if not fields or payload.get("_scrubbed"):
        return payload, False
    out = dict(payload)
    for f in fields:
        if f in out:
            out[f] = TOMBSTONE
    out.pop("_content_fields", None)
    out["_scrubbed"] = True
    return out, True


# ---- per-subject erasure (CR-2) ----
# EXACT normalized-address fields are the ONLY erasure-match source. Raw RFC headers
# (from/to/cc) are kept for audit but NEVER matched: substring matching on a raw header is a
# GDPR-critical false-positive risk ("alice@x.com" ⊂ "alice@x.com.evil.com" / "bob+alice@x.com").
_EMAIL_LIST_FIELDS = ("to_emails", "cc_emails", "bcc_emails")


def subject_matches(payload: dict[str, Any], *, email: str | None, slack_user_id: str | None) -> bool:
    """Does this signal contain the subject? Matches ONLY on exact, normalized plaintext
    identifiers (author slack_user_id; parsed sender/recipient emails) — never substring, never
    raw headers. Encrypted fields (e.g. bcc_emails) simply never equal a plaintext address, so
    they can't produce a false positive (a BCC-only subject is found via crypto-shred, future)."""
    if slack_user_id and payload.get("slack_user_id") == slack_user_id:
        return True
    if not email:
        return False
    el = email.strip().lower()
    if (payload.get("sender_email") or "").strip().lower() == el:
        return True
    return any(
        el == str(e).strip().lower()
        for k in _EMAIL_LIST_FIELDS
        for e in (payload.get(k) or [])
    )


def scrub_subject(payload: dict[str, Any], *, email: str | None, slack_user_id: str | None) -> tuple[dict[str, Any], bool]:
    """Erase one subject from a signal: tombstone the content AND the subject's own identity
    references (their email in sender/recipient fields, their slack_user_id). Other people's
    data on a shared message is preserved (pseudonymize the one subject, not the row)."""
    # Content fields (incl. Gmail's raw `bcc`, which is a self-described content field) are
    # tombstoned wholesale here — that covers the raw-header BCC-completeness requirement.
    out, changed = scrub_content_fields(payload)
    el = email.strip().lower() if email else None
    if el and (out.get("sender_email") or "").strip().lower() == el:
        out["sender_email"] = TOMBSTONE
        changed = True
    if el:
        for k in _EMAIL_LIST_FIELDS:
            vals = out.get(k)
            if vals:
                new = [TOMBSTONE if str(e).strip().lower() == el else e for e in vals]
                if new != vals:
                    out[k] = new
                    changed = True
    if slack_user_id and out.get("slack_user_id") == slack_user_id:
        out["slack_user_id"] = TOMBSTONE
        changed = True
    return out, changed


def _min_content_ttl_days() -> int:
    vals = [d for d in {**_CONTENT_TTL_DAYS_BY_CLASS, **_CONTENT_TTL_DAYS_BY_SOURCE}.values() if d]
    return min(vals) if vals else 3650


# ---- DB orchestration (reuses SignalRepository + AuditService + metrics) ----

async def run_retention_purge(
    db: AsyncSession, *, now: datetime | None = None, dry_run: bool = False, max_rows: int = 50_000
) -> dict[str, int]:
    """Enforce content retention: tombstone CONTENT of signals past their per-class TTL. Keeps
    the event skeleton (replay-safe). Idempotent (skips already-scrubbed). dry_run counts only."""
    now = now or datetime.now(timezone.utc)
    repo = SignalRepository(db)
    cutoff = now - timedelta(days=_min_content_ttl_days())  # only rows old enough to POSSIBLY expire
    scanned = expired = 0
    after_id: uuid.UUID | None = None
    while scanned < max_rows:
        rows = await repo.scan_expired_content(before=cutoff, limit=500, after_id=after_id)
        if not rows:
            break
        for rec in rows:
            after_id = rec.id
            scanned += 1
            ttl = content_retention_days(rec.source, (rec.extra or {}).get("retention_class"))
            if ttl is None or rec.occurred_at >= now - timedelta(days=ttl):
                continue
            new_payload, changed = scrub_content_fields(rec.payload or {})
            if not changed:
                continue
            expired += 1
            if not dry_run:
                await repo.scrub_payload(rec.id, new_payload)
        if len(rows) < 500:
            break
    # Metrics ALWAYS emitted (incl. dry-run) for observability of scope vs effect.
    metrics.incr("signals.retention.scanned", value=scanned, labels={"dry_run": str(dry_run)})
    metrics.incr("signals.retention.expired", value=expired, labels={"dry_run": str(dry_run)})
    if expired and not dry_run:
        await AuditService(db).record(
            "signal_retention_purge", actor_kind="system",
            metadata={"content_tombstoned": expired, "scanned": scanned},
        )
    logger.info("retention purge scanned=%d expired=%d dry_run=%s", scanned, expired, dry_run)
    return {"scanned": scanned, "expired": expired}


async def erase_subject(
    db: AsyncSession, org_id: uuid.UUID, *, email: str | None = None, slack_user_id: str | None = None,
    actor_id: uuid.UUID | None = None, dry_run: bool = False, max_rows: int = 100_000,
) -> dict[str, Any]:
    """GDPR right-to-erasure for one subject across an org's signals (messages, emails,
    snapshots). Physical content+identity tombstone — deterministic, idempotent (re-running
    re-matches already-tombstoned rows and changes nothing), audited. Emits an audit row that
    future Memory/State consume to erase their projections."""
    if not email and not slack_user_id:
        raise ValueError("erase_subject requires an email or slack_user_id")
    repo = SignalRepository(db)
    scanned = matched = erased = 0
    after_id: uuid.UUID | None = None
    while scanned < max_rows:
        rows = await repo.scan_org_signals(org_id, limit=500, after_id=after_id)
        if not rows:
            break
        for rec in rows:
            after_id = rec.id
            scanned += 1
            payload = rec.payload or {}
            if not subject_matches(payload, email=email, slack_user_id=slack_user_id):
                continue
            matched += 1
            new_payload, changed = scrub_subject(payload, email=email, slack_user_id=slack_user_id)
            if changed:
                erased += 1
                if not dry_run:
                    await repo.scrub_payload(rec.id, new_payload)
        if len(rows) < 500:
            break
    # Metrics ALWAYS emitted (incl. dry-run).
    metrics.incr("signals.erasure.scanned", value=scanned, labels={"dry_run": str(dry_run)})
    metrics.incr("signals.erasure.matched", value=matched, labels={"dry_run": str(dry_run)})
    metrics.incr("signals.erasure.erased", value=erased, labels={"dry_run": str(dry_run)})
    result = {"scanned": scanned, "matched": matched, "erased": erased, "dry_run": dry_run}
    handle = _subject_handle(email, slack_user_id)
    if not dry_run:
        # The auditable proof a valid deletion ran — subject identifiers are NOT stored in the
        # audit metadata (that would re-introduce the PII we just erased); a hashed handle is.
        await AuditService(db).record(
            "signal_subject_erasure", org_id=org_id, actor_id=actor_id, actor_kind="system",
            target_id=handle, metadata=result,
        )
        # Cascade seam: notify Memory/State to erase their projections in the SAME transaction.
        await _notify_erasure(
            SubjectErasureNotice(org_id=org_id, subject_handle=handle, erased=erased,
                                 email=email, slack_user_id=slack_user_id),
            db,
        )
    logger.warning("subject erasure org=%s %s", org_id, result)
    return result


def _subject_handle(email: str | None, slack_user_id: str | None) -> str:
    """A non-reversible handle for the audit trail (so the proof-of-erasure doesn't re-store
    the erased identity). KEYED (HMAC with the server secret), not a bare SHA-256: emails are a
    low-entropy, enumerable space, so an unkeyed digest is dictionary-/rainbow-attackable. The
    secret makes the handle non-reversible AND non-correlatable without server access."""
    raw = f"{(email or '').strip().lower()}|{slack_user_id or ''}".encode()
    key = (settings.secret_key or "navis-governance").encode()
    return "subj:" + hmac.new(key, raw, hashlib.sha256).hexdigest()[:32]
