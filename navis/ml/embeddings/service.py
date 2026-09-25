"""EmbeddingService — turn immutable Memory facts into versioned vectors.

Reads Memory (allowed: ML may read the truth layers for retrieval) and writes the
append-only memory_embeddings table. Idempotent: the same fact at the same model+
version collapses via ON CONFLICT DO NOTHING.

Writes always target the WRITE pin; retrieval reads the READ pin. See
`navis.ml.embeddings.pins` — holding them apart is what lets a full re-embed run
to completion behind a live, complete index.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select, update
from sqlalchemy import text as sa_text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from navis.memory.models import MemoryRecord
from navis.ml.embeddings.admission import admits
from navis.ml.embeddings.models import (
    EmbeddingBackfillProgress,
    MemoryEmbedding,
    compute_embedding_id,
    compute_progress_id,
)
from navis.ml.embeddings.pins import EMBEDDING_VERSION, EmbeddingPin, write_pin
# Width of the typed vector column retrieval reads; defined where retrieval
# owns it so writer and reader cannot disagree about the index shape.
from navis.ml.retrieval.vector_search import ANN_DIM
from navis.platform.config import settings
from navis.platform.monitoring import get_logger
from navis.platform.monitoring.metrics import metrics

logger = get_logger("ml.embeddings")


# --- Embedding text ----------------------------------------------------------
#
# DESIGN. The vector for a fact is only as good as the string we hand the embedder.
# The original renderer scanned a generic key list (title/summary/text/name/...)
# and fell back to `dedup_key` when none matched. For four of the highest-volume
# entity types that fallback WAS the output, so ~94% of the corpus embedded
# opaque identifiers:
#
#     thread       — 19f98866cbc10958            (dedup_key = thread_id)
#     person       — someone@example.com         (dedup_key = email)
#     relationship — inbound:someone@example.com (dedup_key)
#     company      — example.com                 (dedup_key = domain)
#
# `thread.subject` was already stored and simply never read. Each type now has an
# explicit renderer that answers "what is this fact ABOUT?" in natural language.
#
# Rules every renderer obeys:
#   * never emit an opaque id (thread_id, message_id, folder_id, slack_user_id…) —
#     they carry no semantics and poison similarity with hash noise;
#   * never emit raw HTML — snippets come from the extractor's cleaned text;
#   * never emit a whole email body — bodies average ~12KB and would swamp
#     everything else in the corpus; bounded fields only;
#   * stay concise and deterministic — no clock, no model, no randomness, so the
#     same record always yields the same vector.
#
# Renderers read payload keys that may be absent on older records (the additive
# keys land only on newly-extracted facts); every access degrades gracefully.

_MAX_TEXT = 2000
_MAX_LIST = 5


def _clean(value: Any) -> str:
    """Collapse whitespace; drop non-strings and blanks."""
    if not isinstance(value, str):
        return ""
    return " ".join(value.split()).strip()


def _people(values: Any, limit: int = _MAX_LIST) -> str:
    if not isinstance(values, (list, tuple)):
        return ""
    seen = [_clean(v) for v in values if _clean(v)]
    if not seen:
        return ""
    head = ", ".join(seen[:limit])
    extra = len(seen) - limit
    return f"{head} and {extra} others" if extra > 0 else head


def _domain_of(email: str) -> str:
    return email.split("@", 1)[1] if "@" in email else ""


# Raw role tokens ("from"/"to") read as noise next to a name; the natural-language
# form embeds closer to how a founder would ask ("who sent me…", "who was cc'd").
#
# The PHRASES stay vendor-neutral even where the token is not. `slack_member` is
# the vocabulary Memory's Slack extractor writes, so ML has to decode it — but it
# renders as "channel member", because a Teams or Discord membership is the same
# organizational fact and embedding one vendor's name would make the vector claim
# something the entity does not say. Any future `*_member` token maps here too.
_ROLE_WORDS = {
    "from": "sender",
    "to": "recipient",
    "cc": "copied recipient",
    "attendee": "meeting attendee",
    "slack_member": "channel member",
}


def _person_text(p: dict) -> str:
    name, email = _clean(p.get("name")), _clean(p.get("email"))
    role = _clean(p.get("role"))
    who = f"{name} <{email}>" if name and email else (name or email)
    if not who:
        return ""
    bits = [f"Person {who}"]
    readable_role = _ROLE_WORDS.get(role, role)
    if readable_role:
        bits.append(readable_role)
    domain = _domain_of(email)
    if domain:
        bits.append(f"at {domain}")
    return ", ".join(bits)


def _relationship_text(p: dict) -> str:
    email = _clean(p.get("person_email"))
    via = _clean(p.get("via")) or "email"
    direction = _clean(p.get("direction"))
    if not email:
        return ""
    phrasing = {
        "inbound": f"Received {via} from {email}",
        "outbound": f"Sent {via} to {email}",
        "active_in": f"Active with {email} on {via}",
    }.get(direction, f"{via} contact with {email}")
    domain = _domain_of(email)
    return f"{phrasing}{f' at {domain}' if domain else ''}"


def _company_text(p: dict) -> str:
    domain, name = _clean(p.get("domain")), _clean(p.get("name"))
    if not (domain or name):
        return ""
    # Domains are the only identifier we have, but they ARE semantic (unlike a
    # hex id): "stripe.com" embeds near "stripe". Add the readable stem so the
    # vector also carries the bare company word.
    label = name or domain.rsplit(".", 2)[0].replace("-", " ")
    return f"Company {label} ({domain})" if domain else f"Company {label}"


def _thread_head(p: dict) -> str:
    """A thread WITHOUT its body — subject, participants, category.

    Split out so there is ONE definition of the derived part and the body is a
    separate decision. `_thread_text` still returns the identical string it
    always did, so every stored embedding and the chat path are untouched; the
    Developer boundary renders this head alone (see `memory_text_minimal`).

    Splitting the full string on ": " would NOT be equivalent and must not be
    used as a shortcut: a real subject contains colons ("Re: Invitation from an
    unknown sender: ...") and the split would cut inside it.
    """
    subject = _clean(p.get("subject"))
    participants = _people(p.get("participants"))
    category = _clean(p.get("category"))
    bits = [f'Email thread "{subject}"' if subject else "Email thread"]
    if participants:
        bits.append(f"with {participants}")
    if category and category != "personal":
        bits.append(f"({category})")
    return " ".join(bits)


def _thread_text(p: dict) -> str:
    subject = _clean(p.get("subject"))
    participants = _people(p.get("participants"))
    snippet = _clean(p.get("snippet"))
    if not (subject or participants or snippet):
        return ""
    head = _thread_head(p)
    return f"{head}: {snippet}" if snippet else head


def _commitment_text(p: dict) -> str:
    text = _clean(p.get("text"))
    if not text:
        return ""
    speaker = _clean(p.get("speaker_email"))
    subject = _clean(p.get("subject"))
    lead = f"Commitment by {speaker}" if speaker else "Commitment"
    if subject:
        lead += f' in "{subject}"'
    return f"{lead}: {text}"


def _decision_text(p: dict) -> str:
    text = _clean(p.get("text"))
    if not text:
        return ""
    subject = _clean(p.get("subject"))
    return f'Decision in "{subject}": {text}' if subject else f"Decision: {text}"


def _meeting_text(p: dict) -> str:
    title = _clean(p.get("title"))
    attendees = _people(p.get("attendees"))
    starts = _clean(p.get("starts_at"))[:16].replace("T", " ")
    if not (title or attendees):
        return ""
    bits = [f'Meeting "{title}"' if title else "Meeting"]
    if starts:
        bits.append(f"on {starts}")
    if attendees:
        bits.append(f"with {attendees}")
    organizer = _clean(p.get("organizer"))
    if organizer:
        bits.append(f"organised by {organizer}")
    return " ".join(bits)


def _document_text(p: dict) -> str:
    title = _clean(p.get("title"))
    if not title:
        return ""
    owners = _people(p.get("owners"))
    return f'Document "{title}" owned by {owners}' if owners else f'Document "{title}"'


def _project_text(p: dict) -> str:
    # `name` here is the FILE observed inside the folder that stands in for a
    # project — label it honestly rather than implying a project called "x.pdf".
    name = _clean(p.get("name"))
    if not name:
        return ""
    owners = _people(p.get("owners"))
    text = f'Project file "{name}"'
    return f"{text} owned by {owners}" if owners else text


def _channel_text(p: dict) -> str:
    # NOT "Slack channel": CHANNEL is a normalized entity, and the payload
    # (channel_id/name/visibility/workspace_id) carries no producer. A Teams or
    # Discord channel folds into the same entity type, so naming one vendor here
    # would embed a false claim about every other. Vendor identity belongs to the
    # Signal, below State — ML only ever sees the normalized fact.
    name, visibility = _clean(p.get("name")), _clean(p.get("visibility"))
    if not name:
        return ""
    return f"Channel #{name}" + (f" ({visibility})" if visibility else "")


def _customer_text(p: dict) -> str:
    name = _clean(p.get("name"))
    return f"Customer {name}" if name else ""


_RENDERERS = {
    "person": _person_text,
    "relationship": _relationship_text,
    "company": _company_text,
    "thread": _thread_text,
    "commitment": _commitment_text,
    "decision": _decision_text,
    "meeting": _meeting_text,
    "document": _document_text,
    "project": _project_text,
    "channel": _channel_text,
    "customer": _customer_text,
}


def _generic_text(payload: dict) -> str:
    """Unknown entity type: the old generic scan, minus the dedup_key fallback."""
    for key in ("title", "summary", "text", "name", "description", "content"):
        value = _clean(payload.get(key))
        if value:
            return value
    return ""


def memory_text(record: MemoryRecord) -> str:
    """A stable, semantically meaningful text representation of a fact.

    Deterministic and model-free: same record in, same string out, forever — so a
    re-embed is reproducible and a stored vector always matches its source_text.
    """
    payload: dict[str, Any] = record.payload or {}
    renderer = _RENDERERS.get(record.entity_type)
    body = renderer(payload) if renderer else _generic_text(payload)
    if not body:
        # Nothing semantic to say. Keep the entity type + dedup key so the row is
        # still searchable by exact identifier, and so this case stays visible in
        # the corpus instead of silently embedding an empty string.
        body = f"{record.entity_type} {record.dedup_key}"
    return body[:_MAX_TEXT]


#: Entity types whose rendered fact embeds RAW SOURCE CONTENT, and what it is.
#:
#: Audited across all eleven renderers. The other eight are metadata only —
#: names, emails, domains, titles, owners, times, attendees — and carry nothing
#: from a message body.
#:
#:   thread      the email BODY (`snippet`), uncapped by `_clean` and bounded
#:               only by `_MAX_TEXT`, so up to ~2000 characters of message text.
#:   commitment  the promise as spoken (`text`).
#:   decision    the decision as recorded (`text`).
#:
#: `commitment` and `decision` are DELIBERATELY NOT minimized. Their quoted text
#: IS the organizational fact — "somebody committed to something" with the
#: something removed is not a smaller fact, it is a useless one — and a caller
#: reaching this bundle already holds `view_memory` on this organization. The
#: thread snippet is different in kind: the subject line and participants are
#: the organizational fact, and the body is incidental content that happens to
#: ride along because similarity search wanted it.
RAW_CONTENT_ENTITY_TYPES: frozenset[str] = frozenset({"thread", "commitment", "decision"})


def memory_text_minimal(entity_type: str, payload: dict | None) -> str | None:
    """The derived organizational fact, with raw BODY content left out.

    Returns None when there is nothing to substitute — an unknown entity type,
    an absent payload, or a render that comes back empty — and the caller then
    keeps what it already had rather than replacing a real fact with a blank.
    Minimization must never be able to erase a fact.

    Only `thread` currently differs from `memory_text`; see
    `RAW_CONTENT_ENTITY_TYPES` for why commitment and decision do not.
    """
    if not isinstance(payload, dict):
        return None
    if entity_type == "thread":
        return _thread_head(payload) or None
    renderer = _RENDERERS.get(entity_type)
    if renderer is None:
        return None
    return renderer(payload) or None


@dataclass(frozen=True, slots=True)
class BackfillPage:
    """Outcome of one keyset page — the unit of progress and of resumption."""

    embedded: int
    skipped: int
    scanned: int
    cursor: uuid.UUID | None
    done: bool


class EmbeddingService:
    def __init__(self, db: AsyncSession, *, pin: EmbeddingPin | None = None) -> None:
        self.db = db
        self.pin = pin or write_pin()
        # Retained so existing callers/tests that read .provider/.model keep working.
        self.provider = self.pin.provider
        self.model = self.pin.model

    # ---- single record (real-time path) ------------------------------------

    async def embed_record(
        self, record: MemoryRecord, *, version: int = EMBEDDING_VERSION
    ) -> bool:
        """Embed one fact. Returns True if a NEW row was written.

        Admission runs here too, so the real-time subscriber and the backfill
        agree on what belongs in the index.
        """
        if not admits(record):
            metrics.incr("ml.embeddings.rejected", labels={"entity_type": record.entity_type})
            return False
        return await self._write_batch([record], version=version) == 1

    # ---- batched write -----------------------------------------------------

    async def _write_batch(self, records: list[MemoryRecord], *, version: int) -> int:
        """Embed and insert a batch in ONE provider call and ONE statement.

        Batching matters for a hosted embedder: the previous one-call-per-record
        shape meant ~4.5k HTTPS round-trips for a full re-embed. The provider API
        already accepted a list; only the caller had to change.
        """
        if not records:
            return 0
        texts = [memory_text(r) for r in records]
        vectors = await self.pin.embed(texts)
        if len(vectors) != len(records):
            # Never silently mis-pair a vector with the wrong fact.
            raise RuntimeError(
                f"embedding provider returned {len(vectors)} vectors for {len(records)} inputs"
            )
        rows = [
            {
                "id": compute_embedding_id(r.memory_id, self.pin.model, version),
                "organization_id": r.organization_id,
                "memory_id": r.memory_id,
                "model": self.pin.model,
                "version": version,
                "dim": len(vec),
                "embedding": vec,
                "entity_type": r.entity_type,
                "source_text": text,
            }
            for r, text, vec in zip(records, texts, vectors)
        ]
        stmt = pg_insert(MemoryEmbedding).values(rows).on_conflict_do_nothing(
            index_elements=["id"]
        )
        written = (await self.db.execute(stmt)).rowcount or 0
        await self._fill_typed_vectors([r["id"] for r in rows])
        return written

    async def _fill_typed_vectors(self, ids: list[uuid.UUID]) -> None:
        """Mirror the JSONB vector into the typed `embedding_vec` column.

        Retrieval's fast path filters on `embedding_vec IS NOT NULL` (see
        `ml/retrieval/vector_search.py`), so a row written without it would be
        silently invisible to search while looking perfectly healthy in the
        table — the worst kind of miss.

        Kept as a follow-up statement rather than an insert column so the ORM
        model stays free of a pgvector type dependency and a database without
        migration 0038 still works. Idempotent (`IS NULL` guard) and scoped to
        the rows just written; failure degrades to the JSONB scan rather than
        failing the embed.
        """
        if not ids:
            return
        try:
            await self.db.execute(
                sa_text(
                    "UPDATE memory_embeddings "
                    "   SET embedding_vec = (embedding::text)::vector "
                    " WHERE id = ANY(:ids) AND embedding_vec IS NULL "
                    f"  AND dim = {ANN_DIM}"
                ),
                {"ids": ids},
            )
        except Exception:  # noqa: BLE001 — typed column is an optimisation, not a requirement
            logger.exception("could not populate embedding_vec; search falls back to JSONB")
            metrics.incr("ml.embeddings.typed_vector_failed")

    # ---- resumable backfill ------------------------------------------------

    async def _progress(self, org_id: uuid.UUID, version: int) -> EmbeddingBackfillProgress:
        """Get-or-create the progress row for this org at the write pin's index."""
        pid = compute_progress_id(org_id, self.pin.model, version)
        await self.db.execute(
            pg_insert(EmbeddingBackfillProgress)
            .values(
                id=pid,
                organization_id=org_id,
                model=self.pin.model,
                version=version,
                cursor_memory_id=None,
                embedded_count=0,
                skipped_count=0,
                failed_count=0,
            )
            .on_conflict_do_nothing(index_elements=["id"])
        )
        return await self.db.get_one(EmbeddingBackfillProgress, pid)

    async def backfill_page(
        self,
        org_id: uuid.UUID,
        *,
        version: int = EMBEDDING_VERSION,
        page_size: int | None = None,
        batch_size: int | None = None,
    ) -> BackfillPage:
        """Advance one keyset page for `org_id`. Safe to call repeatedly.

        Ordered by memory_id and resumed from a persisted cursor, so every record
        is visited exactly once and an interrupted run continues where it stopped.
        The previous implementation used `LIMIT n` with no ORDER BY and no offset:
        with 3,078 records and a limit of 1,000 it re-scanned the same 1,000 rows
        on every call, and the other 2,078 could never be embedded at all.

        ONLY UNEMBEDDED RECORDS ARE SELECTED, and the cursor RESETS on completion.
        Both are required for this to work as a recurring sweeper rather than a
        one-shot backfill, because `memory_id` is a CONTENT-derived UUIDv5 (see
        `memory.ids`) and is therefore NOT time-ordered: a record ingested today
        is equally likely to sort below the cursor as above it. With a
        monotonically advancing cursor and no reset, a completed backfill parked
        the cursor at the maximum id and every subsequently created record whose
        id sorted below it could never be visited again — permanently invisible
        to retrieval. That is why entity types outside the real-time subscriber's
        five bus events had zero vectors.

        The anti-join is what makes the reset cheap: a re-scan filters
        already-embedded rows in SQL, so a sweep with nothing pending costs one
        empty query and zero provider calls. It is also scoped to THIS pin's
        (model, version), so a re-embed onto a new index correctly sees every
        record as pending without touching the old index.
        """
        page_size = page_size or settings.embedding_backfill_page_size
        batch_size = batch_size or settings.embedding_backfill_batch_size
        progress = await self._progress(org_id, version)

        already_embedded = (
            select(MemoryEmbedding.id)
            .where(
                MemoryEmbedding.memory_id == MemoryRecord.memory_id,
                MemoryEmbedding.model == self.pin.model,
                MemoryEmbedding.version == version,
            )
            .exists()
        )
        stmt = select(MemoryRecord).where(
            MemoryRecord.organization_id == org_id,
            ~already_embedded,
        )
        if progress.cursor_memory_id is not None:
            stmt = stmt.where(MemoryRecord.memory_id > progress.cursor_memory_id)
        records = (
            await self.db.scalars(stmt.order_by(MemoryRecord.memory_id).limit(page_size))
        ).all()

        if not records:
            await self._mark_complete(progress)
            return BackfillPage(0, 0, 0, progress.cursor_memory_id, done=True)

        eligible = [r for r in records if admits(r)]
        skipped = len(records) - len(eligible)
        embedded = 0
        for start in range(0, len(eligible), batch_size):
            embedded += await self._write_batch(
                eligible[start : start + batch_size], version=version
            )

        cursor = records[-1].memory_id
        done = len(records) < page_size
        await self._advance(progress, cursor, embedded, skipped, done=done)
        metrics.incr("ml.embeddings.backfilled", value=embedded)
        if skipped:
            metrics.incr("ml.embeddings.rejected", value=skipped)
        logger.info(
            "backfill org=%s model=%s v=%d scanned=%d embedded=%d skipped=%d done=%s",
            org_id, self.pin.model, version, len(records), embedded, skipped, done,
        )
        return BackfillPage(embedded, skipped, len(records), cursor, done=done)

    async def _advance(
        self,
        progress: EmbeddingBackfillProgress,
        cursor: uuid.UUID,
        embedded: int,
        skipped: int,
        *,
        done: bool,
    ) -> None:
        """Move the cursor and counters forward in the caller's transaction.

        The cursor commits together with the vectors it accounts for, so a crash
        can only ever replay a page — never skip one.

        On completion the cursor RESETS to NULL so the next sweep starts from the
        beginning and can see records created below the previous high-water mark
        (memory_id is content-derived, not time-ordered — see `backfill_page`).
        `completed_at` is still stamped, so "has the initial backfill finished?"
        remains answerable; it is a milestone, not a stop flag.
        """
        values: dict[str, Any] = {
            "cursor_memory_id": None if done else cursor,
            "embedded_count": EmbeddingBackfillProgress.embedded_count + embedded,
            "skipped_count": EmbeddingBackfillProgress.skipped_count + skipped,
        }
        if done:
            values["completed_at"] = datetime.now(timezone.utc)
        await self.db.execute(
            update(EmbeddingBackfillProgress)
            .where(EmbeddingBackfillProgress.id == progress.id)
            .values(**values)
        )

    async def _mark_complete(self, progress: EmbeddingBackfillProgress) -> None:
        """Nothing left to embed at this pin. Reset the cursor so the next sweep
        re-scans from the start (cheaply — the anti-join returns nothing until a
        new record lands), and stamp `completed_at` the first time only."""
        values: dict[str, Any] = {"cursor_memory_id": None}
        if progress.completed_at is None:
            values["completed_at"] = datetime.now(timezone.utc)
        await self.db.execute(
            update(EmbeddingBackfillProgress)
            .where(EmbeddingBackfillProgress.id == progress.id)
            .values(**values)
        )

    async def embed_org(
        self,
        org_id: uuid.UUID,
        *,
        version: int = EMBEDDING_VERSION,
        max_pages: int = 1000,
    ) -> int:
        """Drive one org's backfill to completion (bounded by max_pages).

        Returns the count of NEW vectors written by this invocation. Re-running
        after completion is a cheap no-op — the cursor is already at the end.
        """
        total = 0
        for _ in range(max_pages):
            page = await self.backfill_page(org_id, version=version)
            total += page.embedded
            if page.done:
                break
        await self.db.flush()
        return total

    async def backfill_status(
        self, org_id: uuid.UUID, *, version: int = EMBEDDING_VERSION
    ) -> dict[str, Any]:
        """Observable migration state for one org: how far along, and finished?"""
        total = len(
            (
                await self.db.scalars(
                    select(MemoryRecord.memory_id).where(
                        MemoryRecord.organization_id == org_id
                    )
                )
            ).all()
        )
        indexed = len(
            (
                await self.db.scalars(
                    select(MemoryEmbedding.memory_id).where(
                        MemoryEmbedding.organization_id == org_id,
                        MemoryEmbedding.model == self.pin.model,
                        MemoryEmbedding.version == version,
                    )
                )
            ).all()
        )
        row = await self.db.get(
            EmbeddingBackfillProgress, compute_progress_id(org_id, self.pin.model, version)
        )
        return {
            "org_id": str(org_id),
            "model": self.pin.model,
            "version": version,
            "memory_records": total,
            "indexed": indexed,
            "embedded_count": row.embedded_count if row else 0,
            "skipped_count": row.skipped_count if row else 0,
            "cursor_memory_id": str(row.cursor_memory_id) if row and row.cursor_memory_id else None,
            "completed_at": row.completed_at.isoformat() if row and row.completed_at else None,
            "complete": bool(row and row.completed_at),
        }
