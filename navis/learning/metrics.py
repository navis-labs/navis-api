"""THE LEARNING READ CONTRACT — the only supported way to consume Learning.

Learning's tables are Learning's business. Everything downstream reads this
module instead, and gets back frozen dataclasses of primitives: no ORM rows, no
session semantics, no column names, nothing that pins a consumer to a schema it
does not own.

WHY THIS EXISTS. `IntelligenceRepository.build_view` used to SELECT
`learning_events`, `learning_aggregates` and `learning_patterns` directly — nine
queries' worth of knowledge about Learning's storage living in another layer —
and `AdaptiveRepository.build_view` had grown its own near-identical copy of the
same reads. Two consumers reimplementing one layer's storage is how the meaning
of `failures` drifts apart, and it is precisely what the OUTCOME_OBSERVED
contract prevents on the way IN.

WHAT IT DOES NOT DO. No scoring, no recommendation, no interpretation beyond the
counts Learning already committed to. `success_rate` is derived on read from
stored integers; it is not a stored opinion.

This module previously held one orphaned `success_rates()` helper with no caller
anywhere in navis/ or tests/. It is now the boundary it was described as being.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime

from sqlalchemy import func, select

from navis.learning.aggregates import success_rate
from navis.learning.models import (
    AggregateDimension,
    LearningAggregate,
    LearningEvent,
    LearningPattern,
)

# Bounded lineage: how many supporting ids one key may carry, and how many
# events a single lineage pass will scan. Full counts always come from the
# aggregates, so this bound costs precision in the citation list only.
_MAX_LINEAGE = 50
_LINEAGE_SCAN_LIMIT = 20000


@dataclass(frozen=True, slots=True)
class AggregateFact:
    """One (dimension, key) counter. Integers Learning committed to."""

    dimension: str
    key: str
    successes: int
    failures: int
    neutrals: int
    total: int

    @property
    def decisive(self) -> int:
        """Samples that carried a verdict. Neutrals are attempts, not verdicts."""
        return self.successes + self.failures

    @property
    def rate(self) -> float | None:
        """successes / decisive, or None when nothing decisive has happened."""
        return success_rate(self.successes, self.failures)


@dataclass(frozen=True, slots=True)
class SignalPattern:
    """A derived organizational pattern: "this signal kind is reliable"."""

    signal_id: str
    kind: str            # reliable | unreliable
    success_rate: float
    samples: int


@dataclass(frozen=True, slots=True)
class LearningSnapshot:
    """Everything Learning is willing to say about one organization, right now.

    Org-scoped by construction: every query below filters on `org_id` and the
    snapshot carries it, so a consumer cannot accidentally hold two orgs' facts
    in one value.
    """

    org_id: uuid.UUID
    # How much has been learned. Monotonic count of learning events; consumers
    # use it as a version key, which is why it is a count and not a timestamp.
    learning_version: int
    # Logical time: the newest OCCURRED_AT in the log, never wall clock and
    # never an ingestion time.
    as_of: datetime | None
    # The State snapshot the newest learning was true against.
    state_watermark: str | None
    # Which reasoning produced the evidence this snapshot summarises. Pairs with
    # `state_watermark`: together they say "this is what the org learned, from
    # reasoning vN, against that world". A consumer citing a learned fact cannot
    # otherwise say which reasoning generation stands behind it.
    reasoning_version: int | None = None
    aggregates: tuple[AggregateFact, ...] = ()
    patterns: tuple[SignalPattern, ...] = ()
    # key -> supporting learning_event ids (what evidence supports this key)
    lineage: dict[str, list[str]] = field(default_factory=dict)
    # key -> ReasoningSignal ids (why the org was told about it at all)
    signal_lineage: dict[str, list[str]] = field(default_factory=dict)

    def by_dimension(self, dimension: AggregateDimension | str) -> list[AggregateFact]:
        """This dimension's counters, in stable key order."""
        wanted = dimension.value if isinstance(dimension, AggregateDimension) else dimension
        return sorted(
            (a for a in self.aggregates if a.dimension == wanted), key=lambda a: a.key
        )

    def outcomes_by_surface(self) -> dict[str, list[AggregateFact]]:
        """OUTCOME counters grouped into delivery / judgement / world.

        THE MAPPING STAYS LEARNING'S. The OUTCOME dimension is keyed on
        `outcome_type`, so a consumer that wants "how did the world respond?"
        would otherwise have to hold its own list of which types are world types
        — a second copy of Learning's vocabulary, in another layer, free to drift.
        That is the duplication the read contract exists to prevent, so the
        grouping is computed here with `classify_surface`, the same function the
        classifier uses on the way in.

        Only surfaces with at least one recorded outcome appear: an org that has
        never had a proposal rejected gets no `judgement` key, rather than a key
        holding a fabricated zero.
        """
        from navis.learning.attribution import classify_surface

        grouped: dict[str, list[AggregateFact]] = {}
        for agg in self.by_dimension(AggregateDimension.OUTCOME):
            grouped.setdefault(classify_surface(agg.key).value, []).append(agg)
        return grouped


async def learning_snapshot(db, org_id: uuid.UUID) -> LearningSnapshot:
    """Read one organization's complete learning state.

    Every statement below is scoped to `org_id`. There is no unscoped read in
    this module, which is what makes "Org A's learning cannot influence Org B"
    a property of the boundary rather than of each consumer's diligence.
    """
    learning_version = int(
        await db.scalar(
            select(func.count(LearningEvent.id)).where(
                LearningEvent.organization_id == org_id
            )
        )
        or 0
    )
    as_of = await db.scalar(
        select(func.max(LearningEvent.occurred_at)).where(
            LearningEvent.organization_id == org_id
        )
    )
    # The watermark and reasoning version of the NEWEST learning event. Ordered
    # by (occurred_at, id) so the choice stays deterministic when two events
    # share a timestamp. Read in ONE row rather than two queries — they describe
    # the same event and picking them separately could straddle two.
    newest = (
        await db.execute(
            select(LearningEvent.state_watermark, LearningEvent.reasoning_version)
            .where(
                LearningEvent.organization_id == org_id,
                LearningEvent.state_watermark.is_not(None),
            )
            .order_by(LearningEvent.occurred_at.desc(), LearningEvent.id.desc())
            .limit(1)
        )
    ).first()
    state_watermark = newest[0] if newest else None
    reasoning_version = newest[1] if newest else None

    agg_rows = (
        await db.scalars(
            select(LearningAggregate).where(LearningAggregate.organization_id == org_id)
        )
    ).all()
    pattern_rows = (
        await db.scalars(
            select(LearningPattern).where(LearningPattern.organization_id == org_id)
        )
    ).all()

    lineage, signal_lineage = await _lineage(db, org_id)
    return LearningSnapshot(
        org_id=org_id,
        learning_version=learning_version,
        as_of=as_of,
        state_watermark=state_watermark,
        reasoning_version=reasoning_version,
        aggregates=tuple(
            AggregateFact(r.dimension, r.key, r.successes, r.failures, r.neutrals, r.total)
            for r in agg_rows
        ),
        patterns=tuple(
            SignalPattern(p.signal_id, p.kind, p.success_rate, p.samples)
            for p in pattern_rows
        ),
        lineage=lineage,
        signal_lineage=signal_lineage,
    )


async def _lineage(db, org_id: uuid.UUID) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Bounded supporting lineage per signal/decision key.

    Two maps over the SAME bounded scan (one query, not two):
      * learning-event ids  — what evidence supports this key;
      * ReasoningSignal ids — which signals' outcomes produced that evidence.

    A decision key's signal lineage is the union of the reasoning signals on the
    events that decided it, so a decision can be traced back through Learning to
    the Reasoning that triggered it.
    """
    events = (
        await db.scalars(
            select(LearningEvent)
            .where(LearningEvent.organization_id == org_id)
            .order_by(LearningEvent.id.asc())
            .limit(_LINEAGE_SCAN_LIMIT)
        )
    ).all()
    lineage: dict[str, list[str]] = {}
    signal_lineage: dict[str, list[str]] = {}

    def _add(bucket: dict[str, list[str]], key: str, value: str) -> None:
        slot = bucket.setdefault(key, [])
        if value not in slot and len(slot) < _MAX_LINEAGE:
            slot.append(value)

    for e in events:
        eid = str(e.id)
        signal_ids = [str(s) for s in (e.reasoning_signal_ids or [])]
        for sid in signal_ids:
            _add(lineage, sid, eid)
            # A signal insight's key IS its ReasoningSignal id; recording it
            # keeps the map total, so every key answers the same question.
            _add(signal_lineage, sid, sid)
        if e.decision_id is not None:
            did = str(e.decision_id)
            _add(lineage, did, eid)
            for sid in signal_ids:
                _add(signal_lineage, did, sid)
    return lineage, signal_lineage
