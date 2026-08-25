"""Durable ReasoningSignal provenance — the record a downstream id resolves to.

THE GAP THIS CLOSES. `run_reasoning_v2` produced signals in-process, Decision
copied the few that became decisions into `daily_briefs.decisions`, and every
other signal was discarded. Meanwhile Execution, Observation, Learning and
Intelligence all persist `reasoning_signal_ids` — permanent references to rows
that never existed.

That was survivable only while the ids could be recomputed. They cannot:
`compute_signal_id` hashes `reasoning_version` into the UUIDv5, so bumping the
version (2 -> 3 in Phase 3B) permanently changes every id. The one reasoning
reference in production today was written at v2 and can never be re-derived by
today's code. Provenance that only exists while the process is running is not
provenance.

WHY HERE. `navis/reasoning/v2/**` is AST-pinned to import no sqlalchemy (see
tests/reasoning/test_reasoning_v2.py) because the engines must stay a pure
function of State. This module sits OUTSIDE v2 for exactly that reason: the same
domain owns its contract and its storage, while the engines keep their purity.
Writes are driven from the orchestration layer, which already composes State ->
Reasoning -> Decision; Reasoning never calls a database itself.

IMMUTABLE AND REPRODUCIBLE. `signal_id` is already a deterministic UUIDv5 of the
reasoning identity tuple, so it is the primary key, the insert is append-only
with ON CONFLICT DO NOTHING, and re-running an unchanged brief writes nothing.
Replaying the same State at the same reasoning_version reproduces the same rows
byte for byte. Nothing here is ever updated or deleted.

TWO TABLES, BECAUSE A SIGNAL IS TWO THINGS
==========================================

`compute_signal_id` hashes (org, reasoning_version, engine, engine_version, type,
entity, dedup_key) and deliberately NOT the state watermark, so the same slipping
commitment keeps one id at every snapshot. That stability is load-bearing:
`execution_requests`, `learning_events`, observation payloads and
`intelligence_insights` all persist `reasoning_signal_ids` as long-lived
references, and `compute_signal_id` hashing `reasoning_version` means they cannot
be recomputed once a version is bumped. Not one `dedup_key` in the entire engine
set carries time — every one is an entity_id or a constant — so the identity is,
by construction, "this engine's conclusion about this thing".

But every OTHER field a signal carries is a function of the snapshot. Confidence,
severity, `why` and `evidence` are all derived from `state.as_of`, so the same
conclusion legitimately looks different at every fold. Storing those on the
identity row under ON CONFLICT DO NOTHING meant the FIRST detection won forever:

    live, commitment f33778d8 was stored at 7.16 days old / confidence 0.7386,
    while the decision citing that very row said 12 days / 0.8951. Both were
    computed; only the first was ever written.

So identity and observation are now separate rows. `reasoning_signals` is the
stable conclusion, written once. `reasoning_signal_observations` is what that
conclusion looked like at one state snapshot, appended per pass and never
overwritten. Old references keep resolving to the identity; anything that needs
to know what was true AT a snapshot reads the observation for it.

WHY NOT JUST PUT THE WATERMARK IN THE ID. It would orphan every stored reference
(52 of production's 122 decision references already resolve to nothing), fork
insight provenance, break the entity dedupe `execution.obligation` exists to
provide, and multiply rows with sync frequency rather than with organizational
activity. The identity is not the thing that changes — the observation is.
"""
from __future__ import annotations

import uuid
from collections.abc import Iterable, Sequence
from datetime import datetime
from typing import Any

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
    select,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column

from navis.platform.database.base import Base
from navis.reasoning.v2.contract import ReasoningSignal

# Deterministic namespace for observation ids — disjoint from every other UUIDv5
# space in the app (reasoning signals, decisions, memory records, dismissals).
_OBSERVATION_NAMESPACE = uuid.UUID("5f3e1d92-7b04-5a68-9c31-2e8d40b7f6a5")


def compute_observation_id(
    signal_id: uuid.UUID, state_watermark: str | None
) -> uuid.UUID:
    """Deterministic id for ONE conclusion seen at ONE snapshot.

    THIS IS THE (signal_id, state_watermark) UNIQUENESS CONSTRAINT, expressed the
    way every other identity in this codebase is: as a UUIDv5 primary key over the
    natural key. It is not merely a stylistic match — a plain UNIQUE index could
    not do the job, because `state_watermark` is nullable and Postgres treats NULLs
    as distinct, so an organization with no watermark yet would accumulate a fresh
    duplicate observation on every single pass. Folding the null into a sentinel
    inside the hash closes that hole, and makes re-running one reasoning pass a
    no-op at the database rather than a race the application has to win.

    PRESENCE IS ENCODED, not collapsed. `compute_brief_id` and `compute_decision_id`
    both spell this `state_watermark or '-'`, which cannot tell "no watermark" from
    a watermark whose value is the string "-". Unreachable today — a watermark is
    always `str(memory_id)` — but this is the key that has to survive a null, so it
    is worth one extra prefix to make the two cases provably distinct rather than
    accidentally distinct.
    """
    key = "-" if state_watermark is None else f"wm:{state_watermark}"
    return uuid.uuid5(_OBSERVATION_NAMESPACE, f"{signal_id}|{key}")


class ReasoningSignalRecord(Base):
    """One emitted ReasoningSignal, exactly as the engines produced it."""

    __tablename__ = "reasoning_signals"

    # The deterministic id downstream layers already store. Same signal at the
    # same reasoning_version => same row, so replay converges instead of forking.
    signal_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    organization_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    signal_type: Mapped[str] = mapped_column(String(40), nullable=False, index=True)
    entity_type: Mapped[str | None] = mapped_column(String(40), nullable=True)
    entity_id: Mapped[str | None] = mapped_column(String(128), nullable=True)

    title: Mapped[str] = mapped_column(String(255), nullable=False)
    why: Mapped[str] = mapped_column(Text, nullable=False)
    # The engine's own evidence dict — carries whatever lineage the engine had
    # (person_email, thread_id, relevance_evidence, ...). Never re-derived here.
    evidence: Mapped[dict] = mapped_column(JSONB, default=dict)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    severity: Mapped[float] = mapped_column(Float, default=0.0)

    engine_name: Mapped[str] = mapped_column(String(64), nullable=False)
    engine_version: Mapped[str] = mapped_column(String(16), nullable=False)
    # THE SNAPSHOT THIS CONCLUSION WAS FIRST SEEN AT — first-write-wins, and that
    # is now its documented meaning rather than an accident. What the conclusion
    # looks like NOW lives in `reasoning_signal_observations`; these columns say
    # when Navis first reached it. Reading them as current is the defect this
    # table was split to remove, so every snapshot-scoped consumer must go
    # through `observations_at` / `latest_observations`.
    state_as_of: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    state_watermark: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    reasoning_version: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    # Logical time (= state_as_of), never wall clock — same discipline as
    # ReasoningSignal itself, so a replayed row is byte-identical.
    created_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class ReasoningSignalObservation(Base):
    """What ONE stable conclusion looked like at ONE state snapshot.

    Append-only and never updated, exactly like `reasoning_signals` — but keyed by
    (signal, snapshot) instead of by signal alone, which is the whole point. A
    commitment that is 7 days old at one fold and 12 at the next produces one
    signal row and two observation rows, so both the identity and the history
    survive. Neither was possible before: the identity row could hold only one set
    of values, and it kept the oldest.

    Nothing here is a second source of truth. The engine's output is copied
    verbatim — no re-derivation, no recomputation, no interpretation — and the
    identity, entity and engine all stay on `reasoning_signals` where they belong.
    """

    __tablename__ = "reasoning_signal_observations"
    __table_args__ = (
        # The snapshot-scoped read: "which conclusions were observed at watermark
        # W for this org?" — backs `observations_at` and the obligation counts.
        Index(
            "ix_reasoning_obs_org_watermark",
            "organization_id", "state_watermark",
        ),
        # The history of one conclusion, newest last.
        Index("ix_reasoning_obs_signal_time", "signal_id", "state_as_of"),
    )

    # uuid5(signal_id, state_watermark) — see `compute_observation_id`. The PK IS
    # the uniqueness requirement, and it makes re-running a pass a no-op.
    observation_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    signal_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("reasoning_signals.signal_id", ondelete="CASCADE"),
        nullable=False,
    )
    # Denormalized from the parent so every read can be org-scoped in its own
    # WHERE clause rather than through a join. Same rule `resolve_signals`
    # already enforces: an id alone must never address another tenant's row.
    organization_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
    )

    # WHICH snapshot. Nullable because an org with no folded memory has no
    # watermark, and that is a real state rather than a missing value.
    state_watermark: Mapped[str | None] = mapped_column(String(64), nullable=True)
    state_as_of: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # WHAT the conclusion looked like there — the engine's own values, verbatim.
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    severity: Mapped[float] = mapped_column(Float, default=0.0)
    why: Mapped[str] = mapped_column(Text, nullable=False)
    evidence: Mapped[dict] = mapped_column(JSONB, default=dict)

    # Logical time (= state_as_of), never wall clock — so a replayed row is
    # byte-identical to the original.
    created_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    # Wall-clock write time. Operational metadata only (how long a snapshot took
    # to reach the store); never read by the projection and never part of identity.
    recorded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


def _row(signal: ReasoningSignal) -> dict[str, Any]:
    return {
        "signal_id": signal.signal_id,
        "organization_id": signal.org_id,
        "signal_type": signal.signal_type.value,
        "entity_type": signal.entity_type,
        "entity_id": signal.entity_id,
        "title": signal.title,
        "why": signal.why,
        "evidence": signal.evidence or {},
        "confidence": signal.confidence,
        "severity": signal.severity,
        "engine_name": signal.engine_name,
        "engine_version": signal.engine_version,
        "state_as_of": signal.state_as_of,
        "state_watermark": signal.state_watermark,
        "reasoning_version": signal.reasoning_version,
        "created_at": signal.created_at,
    }


def _uuids(raw_ids: Iterable[uuid.UUID | str]) -> list[uuid.UUID]:
    """Coerce stored references to UUIDs, dropping any that cannot address a row.

    A non-uuid reference is skipped rather than raising: these ids arrive from
    persisted JSONB payloads written by older code, and one malformed entry must
    not take down a whole resolution.
    """
    out: list[uuid.UUID] = []
    for raw in raw_ids:
        try:
            out.append(raw if isinstance(raw, uuid.UUID) else uuid.UUID(str(raw)))
        except (TypeError, ValueError):
            continue
    return out


def _observation_row(signal: ReasoningSignal) -> dict[str, Any]:
    return {
        "observation_id": compute_observation_id(signal.signal_id, signal.state_watermark),
        "signal_id": signal.signal_id,
        "organization_id": signal.org_id,
        "state_watermark": signal.state_watermark,
        "state_as_of": signal.state_as_of,
        "confidence": signal.confidence,
        "severity": signal.severity,
        "why": signal.why,
        "evidence": signal.evidence or {},
        "created_at": signal.created_at,
    }


async def persist_signals(db: AsyncSession, signals: Iterable[ReasoningSignal]) -> int:
    """Append every signal of a reasoning pass. Returns OBSERVATIONS newly written.

    TWO WRITES, TWO DIFFERENT CONFLICT RULES, and the difference is the fix:

      * the stable conclusion is inserted ON CONFLICT DO NOTHING, because its
        identity and its engine lineage genuinely never change;
      * the observation is inserted per (signal, snapshot), so a conclusion the
        engines reached again at a NEWER snapshot is recorded instead of
        discarded. That write used to be swallowed by the identity row's
        conflict rule, which is why a decision could cite a signal whose stored
        confidence contradicted the decision's own.

    Neither is an UPDATE. History is appended, never overwritten: the 7.16-day
    observation and the 11.85-day observation both survive, and each stays
    attributable to the snapshot it was true of.

    The return value counts observations rather than identities on purpose. "How
    many signals did this pass record?" is a question about the pass, and after
    the first fold the identity insert legitimately writes nothing at all — so
    counting identities would report a healthy pass as having done nothing.

    Persists ALL signals, not only those that became decisions: an
    interpretation-only signal (OPPORTUNITY, CONTEXT_SUMMARY, ...) never reaches
    a brief, yet it is exactly the kind of thing an insight needs to explain why
    the organization was told something. Storing only the decision-bearing ones
    would leave the same hole in a smaller place.

    Idempotent end to end: both statements are ON CONFLICT DO NOTHING over
    deterministic primary keys, so re-running one pass — a retry, an at-least-once
    redelivery, a replayed brief — converges rather than duplicating.
    """
    signals = list(signals)
    if not signals:
        return 0
    await db.execute(
        pg_insert(ReasoningSignalRecord)
        .values([_row(s) for s in signals])
        .on_conflict_do_nothing(index_elements=["signal_id"])
    )
    # Ordered after the identity insert, in the SAME transaction: the observation
    # carries an FK to `reasoning_signals`, so the parent must exist first. One
    # statement each — a pass is hundreds of signals and this is on the brief's
    # critical path.
    written = (
        await db.execute(
            pg_insert(ReasoningSignalObservation)
            .values([_observation_row(s) for s in signals])
            .on_conflict_do_nothing(index_elements=["observation_id"])
        )
    ).rowcount or 0
    return written


async def signal_type_counts_at(
    db: AsyncSession,
    org_id: uuid.UUID,
    *,
    reasoning_version: int,
    state_watermark: str | None,
) -> dict[str, int]:
    """How many signals of each type Reasoning emitted for ONE snapshot.

    THE QUESTION THIS ANSWERS. A brief shows the top `brief_max_items` decisions
    — seven by default — out of everything Reasoning concluded. So "N people are
    waiting on you" cannot be counted from the brief: the brief is a ranked,
    deduped, TRUNCATED selection, and counting it reports the size of a display
    window as if it were the size of the obligation.

    The full set is not lost, it is right here: `persist_signals` appends EVERY
    signal of a pass, and `(organization_id, reasoning_version, state_watermark)`
    is exactly the snapshot a brief is keyed on
    (`decisions.base.compute_brief_id`). So the same brief a client is reading
    resolves to the same rows, deterministically, for as long as they exist.

    Returns EVERY type present, keyed by the raw stored string. It deliberately
    does NOT classify, filter by confidence, or dedupe by entity: those are
    Decision's ranking policy, and reproducing them here would be a second
    ranker that could drift from the first. `decisions.base.obligation_counts`
    owns which types are obligations and applies that itself.

    Aggregated in SQL rather than in Python: an active organization's pass is
    hundreds of signals and the caller only ever wants the tallies, so there is
    no reason to move the rows.

    AN EMPTY RESULT MEANS "NOTHING IS RECORDED FOR THIS SNAPSHOT", never "this
    organization has no obligations". Briefs built before this table existed,
    and briefs whose reasoning pass was never persisted, both land here — so a
    caller must report absence rather than zero. In practice the two are
    distinguishable: `engines/meta.py` emits CONTEXT_SUMMARY for any org with a
    single entity, so a persisted pass over a non-empty org is never empty.

    IT COUNTS OBSERVATIONS, AND THAT IS THE WHOLE CORRECTION. It used to filter
    `reasoning_signals.state_watermark`, which stores the snapshot a conclusion was
    FIRST seen at — so an obligation that had been outstanding since an earlier fold
    was invisible here while still being ranked into the brief. Live, that produced
    `shown = 4, total = 2`, with the two highest-ranked decisions
    (both COMMITMENT_SLIPPING, first detected days earlier) missing from the count
    entirely. The blind spot grew with obligation age, which is precisely backwards.

    An observation exists for every snapshot at which the engines actually reached
    the conclusion, so joining through it asks the intended question: not "which
    conclusions were BORN here?" but "which conclusions were TRUE here?".

    DISTINCT signal identities, not observation rows: one conclusion observed at
    this snapshot is one obligation. The PK already makes a second observation of
    the same (signal, snapshot) impossible, so this is belt-and-braces — but the
    count means "how many things does the founder owe", and that must never be
    inflatable by how many times a pass ran.

    Org-scoped in the statement, like every read in this module — on the
    observation's own column, not through the join, so the filter is indexed.
    """
    rows = (
        await db.execute(
            select(
                ReasoningSignalRecord.signal_type,
                func.count(func.distinct(ReasoningSignalObservation.signal_id)),
            )
            .join(
                ReasoningSignalRecord,
                ReasoningSignalRecord.signal_id == ReasoningSignalObservation.signal_id,
            )
            .where(
                ReasoningSignalObservation.organization_id == org_id,
                ReasoningSignalObservation.state_watermark == state_watermark,
                ReasoningSignalRecord.reasoning_version == reasoning_version,
            )
            .group_by(ReasoningSignalRecord.signal_type)
        )
    ).all()
    return {signal_type: int(count) for signal_type, count in rows}


async def observations_at(
    db: AsyncSession,
    org_id: uuid.UUID,
    signal_ids: Iterable[uuid.UUID | str],
    *,
    state_watermark: str | None,
) -> dict[str, ReasoningSignalObservation]:
    """Resolve stored signal ids -> what they looked like AT ONE SNAPSHOT.

    THE SNAPSHOT-SPECIFIC READ. `resolve_signals` answers "what conclusion is this
    id?"; this answers "what was that conclusion at watermark W?". A consumer that
    needs the second and calls the first gets the first-detection values, which is
    how a decision came to report 11.85 days while its cited row said 7.16.

    FAILS CLOSED. An id with no observation at this watermark is simply ABSENT from
    the result — never silently substituted with an older observation, never with
    the latest one, never with the identity row's own columns. "We did not record
    this conclusion at that snapshot" is a real answer and the caller must be able
    to see it; guessing would rebuild the defect one layer up.

    Org-scoped for the same reason `resolve_signals` is: these ids travel through
    Execution/Observation/Learning payloads and must never resolve cross-tenant.
    """
    ids = _uuids(signal_ids)
    if not ids:
        return {}
    rows = (
        await db.scalars(
            select(ReasoningSignalObservation).where(
                ReasoningSignalObservation.organization_id == org_id,
                ReasoningSignalObservation.signal_id.in_(ids),
                ReasoningSignalObservation.state_watermark == state_watermark,
            )
        )
    ).all()
    return {str(r.signal_id): r for r in rows}


async def observation_history(
    db: AsyncSession, org_id: uuid.UUID, signal_id: uuid.UUID | str
) -> Sequence[ReasoningSignalObservation]:
    """Every recorded observation of ONE conclusion, oldest first.

    How a conclusion AGED — the commitment at 7.16 days and the same commitment at
    11.85 — which is the question the old single-row store could not answer at all.
    Ordered by logical time with the id as a total-order tiebreak, so two
    observations sharing a `state_as_of` still sort identically on every query.
    """
    ids = _uuids([signal_id])
    if not ids:
        return []
    return (
        await db.scalars(
            select(ReasoningSignalObservation)
            .where(
                ReasoningSignalObservation.organization_id == org_id,
                ReasoningSignalObservation.signal_id == ids[0],
            )
            .order_by(
                ReasoningSignalObservation.state_as_of.asc().nulls_first(),
                ReasoningSignalObservation.observation_id.asc(),
            )
        )
    ).all()


async def resolve_signals(
    db: AsyncSession, org_id: uuid.UUID, signal_ids: Iterable[uuid.UUID | str]
) -> dict[str, ReasoningSignalRecord]:
    """Resolve stored reasoning ids -> records, scoped to ONE organization.

    `org_id` is a required filter, not a convenience: these ids travel through
    Execution/Observation/Learning payloads, and a lookup keyed on id alone would
    resolve another tenant's signal if an id were ever replayed across orgs.
    Ids that do not resolve are simply absent — a caller learns "no record",
    never a fabricated one.

    IDENTITY ONLY. The row's `confidence`/`severity`/`why` are its FIRST
    observation, not its current one. A consumer asking "what was true at the
    snapshot this decision was built from?" must call `observations_at`; this
    answers "which conclusion is this id?" and nothing more.
    """
    ids = _uuids(signal_ids)
    if not ids:
        return {}
    rows = (
        await db.scalars(
            select(ReasoningSignalRecord).where(
                ReasoningSignalRecord.organization_id == org_id,
                ReasoningSignalRecord.signal_id.in_(ids),
            )
        )
    ).all()
    return {str(r.signal_id): r for r in rows}
