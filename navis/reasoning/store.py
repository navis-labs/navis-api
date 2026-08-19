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
"""
from __future__ import annotations

import uuid
from collections.abc import Iterable
from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text, select
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column

from navis.platform.database.base import Base
from navis.reasoning.v2.contract import ReasoningSignal


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
    # Which State snapshot this was true of, and under which reasoning policy.
    state_as_of: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    state_watermark: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    reasoning_version: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    # Logical time (= state_as_of), never wall clock — same discipline as
    # ReasoningSignal itself, so a replayed row is byte-identical.
    created_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


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


async def persist_signals(db: AsyncSession, signals: Iterable[ReasoningSignal]) -> int:
    """Append every signal of a reasoning pass. Returns rows newly written.

    Persists ALL signals, not only those that became decisions: an
    interpretation-only signal (OPPORTUNITY, CONTEXT_SUMMARY, ...) never reaches
    a brief, yet it is exactly the kind of thing an insight needs to explain why
    the organization was told something. Storing only the decision-bearing ones
    would leave the same hole in a smaller place.
    """
    rows = [_row(s) for s in signals]
    if not rows:
        return 0
    stmt = pg_insert(ReasoningSignalRecord).values(rows).on_conflict_do_nothing(
        index_elements=["signal_id"]
    )
    return (await db.execute(stmt)).rowcount or 0


async def resolve_signals(
    db: AsyncSession, org_id: uuid.UUID, signal_ids: Iterable[uuid.UUID | str]
) -> dict[str, ReasoningSignalRecord]:
    """Resolve stored reasoning ids -> records, scoped to ONE organization.

    `org_id` is a required filter, not a convenience: these ids travel through
    Execution/Observation/Learning payloads, and a lookup keyed on id alone would
    resolve another tenant's signal if an id were ever replayed across orgs.
    Ids that do not resolve are simply absent — a caller learns "no record",
    never a fabricated one.
    """
    ids: list[uuid.UUID] = []
    for raw in signal_ids:
        try:
            ids.append(raw if isinstance(raw, uuid.UUID) else uuid.UUID(str(raw)))
        except (TypeError, ValueError):
            continue  # a non-uuid reference cannot address a signal; skip it
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
