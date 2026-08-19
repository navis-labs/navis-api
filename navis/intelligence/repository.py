"""IntelligenceRepository — reads ONLY Learning, writes ONLY intelligence_insights.

Learning is the single allowed upstream dependency (per the Core Law), and it is
consumed through Learning's own read contract (`navis.learning.metrics`) rather
than by querying its tables. This module used to SELECT `learning_events`,
`learning_aggregates` and `learning_patterns` itself, which meant Intelligence
carried a second, independently-maintained model of how Learning stores things —
the exact coupling the OUTCOME_OBSERVED event prevents on the way into Learning.
"""
from __future__ import annotations

import uuid
from typing import Any, Sequence

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from navis.intelligence.insights import (
    REALITY_INSIGHT_TYPES,
    AggRow,
    LearningView,
    PatternRow,
)
from navis.intelligence.models import (
    PRE_VERSIONED_REALITY,
    IntelligenceInsight,
    recover_reasoning_version,
)
from navis.learning.metrics import learning_snapshot
from navis.learning.models import AggregateDimension

_REALITY_TYPES = sorted(t.value for t in REALITY_INSIGHT_TYPES)
_REALITY_TYPE_SET = frozenset(_REALITY_TYPES)
# A reality row whose id is reproducible under no known version key. Ranked below
# `PRE_VERSIONED_REALITY` so an unknown provenance never outranks a known one,
# and never silently hides rows: if NOTHING in a generation resolves, they all
# share this rank and all survive.
_UNRECOVERABLE = PRE_VERSIONED_REALITY - 1


# ── Which projection is CURRENT ───────────────────────────────────────────────
#
# `intelligence_insights` is append-only, so every projection an org has ever run
# is still in the table and a naive read serves all of them at once. Live, that
# meant reasoning v3's "4 of 20 tracked relationships are cooling" was returned
# beside v4's corrected "0 of 20", both stamped with the same watermark, the same
# learning_version and the same logical time.
#
# The two insight families supersede each other differently, because they are
# versioned against different clocks:
#
#   Learning-derived — an accumulating log. Each (insight_type, key) is restated
#     at every `learning_version`, so the current answer is the newest row PER
#     KEY and older versions of that key are history.
#
#   Reality-derived  — a SNAPSHOT. Every pass emits a complete description of the
#     organization right now, so the current answer is the whole newest pass and
#     nothing else. A key the newest pass did not emit is not missing data, it is
#     a fact that stopped being true: `risk_intelligence/relationship_cooling`
#     disappeared under v4 precisely because those four risks were newsletter
#     noise, and a per-key rule would have kept serving the v3 row forever.
#     Safe because a pass is all-or-nothing — `session_scope` commits once at the
#     end, so a partial generation is never visible.
#
# Nothing here deletes, mutates or reorders stored rows. The READ decides what is
# current; `get_insights(include_superseded=True)` still serves the full log.


def _at(row) -> tuple[bool, Any]:
    """`created_at` as a sortable key — an undated row is older than any dated one.

    A tuple rather than a coalesced default so a NULL is never compared against a
    datetime and no fabricated timestamp enters the ordering.
    """
    return (row.created_at is not None, row.created_at)


def _newest_per_key(rows: list, rank) -> list:
    """One row per (insight_type, key): the highest-ranked. Rank must be total."""
    best: dict[tuple[str, str], Any] = {}
    for row in rows:
        identity = (row.insight_type, row.key)
        if identity not in best or rank(row) > rank(best[identity]):
            best[identity] = row
    return list(best.values())


def _learning_rank(row) -> tuple:
    return (row.learning_version, _at(row), str(row.insight_id))


def _current_generation(rows: list) -> list:
    """The newest reality projection: one State snapshot, one reasoning version.

    The snapshot is chosen by logical time (`created_at`, = `state.as_of`) and the
    watermark string breaks a tie — both are stored columns. The reasoning version
    is recovered from each row's own identity, because two versions of the same
    snapshot are identical in every column and differ only in the id.
    """
    if not rows:
        return []
    version = {
        row.insight_id: (
            _UNRECOVERABLE if (v := recover_reasoning_version(row)) is None else v
        )
        for row in rows
    }
    groups: dict[str | None, list] = {}
    for row in rows:
        groups.setdefault(row.state_watermark, []).append(row)
    watermark = max(
        groups,
        key=lambda wm: (
            max(_at(r) for r in groups[wm]),
            (wm is not None, wm or ""),
        ),
    )
    generation = groups[watermark]
    newest = max(version[r.insight_id] for r in generation)
    return _newest_per_key(
        [r for r in generation if version[r.insight_id] == newest],
        lambda r: (version[r.insight_id], _at(r), str(r.insight_id)),
    )


def current_insights(rows) -> list:
    """The rows a production read should serve, given any set of stored rows.

    Pure and total: correct over an org's ENTIRE history, which is what lets the
    repository narrow in SQL for scale without changing the answer.
    """
    rows = list(rows)
    learning = [r for r in rows if r.insight_type not in _REALITY_TYPE_SET]
    reality = [r for r in rows if r.insight_type in _REALITY_TYPE_SET]
    current = _newest_per_key(learning, _learning_rank) + _current_generation(reality)
    return sorted(current, key=lambda r: (r.insight_type, r.key))


class IntelligenceRepository:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def build_view(self, org_id: uuid.UUID) -> LearningView:
        """Learning's snapshot, mapped into the shape the generators fold.

        The mapping is all this method does. `LearningView` stays Intelligence's
        own type — the generators are written against it and nothing about their
        input shape changes — but every VALUE in it now arrives from Learning's
        contract, so this layer no longer knows that `learning_aggregates` is a
        table or that `learning_version` is a COUNT of anything.
        """
        snap = await learning_snapshot(self.db, org_id)

        def aggs(dimension: AggregateDimension) -> list[AggRow]:
            return [
                AggRow(a.key, a.successes, a.failures, a.neutrals, a.total)
                for a in snap.by_dimension(dimension)
            ]

        return LearningView(
            org_id=org_id,
            learning_version=snap.learning_version,
            as_of=snap.as_of,
            signal_aggs=aggs(AggregateDimension.SIGNAL),
            decision_aggs=aggs(AggregateDimension.DECISION),
            outcome_aggs=aggs(AggregateDimension.OUTCOME),
            patterns=[
                PatternRow(p.signal_id, p.kind, p.success_rate, p.samples)
                for p in snap.patterns
            ],
            lineage=snap.lineage,
            state_watermark=snap.state_watermark,
            signal_lineage=snap.signal_lineage,
        )

    async def append_insight(self, values: dict[str, Any]) -> bool:
        stmt = (
            pg_insert(IntelligenceInsight)
            .values(**values)
            .on_conflict_do_nothing(index_elements=["insight_id"])
        )
        result = await self.db.execute(stmt)
        return (result.rowcount or 0) == 1

    async def get_insights(
        self, org_id: uuid.UUID, *, include_superseded: bool = False
    ) -> Sequence[IntelligenceInsight]:
        """This organization's CURRENT insights — one per logical insight.

        `include_superseded=True` is the explicit historical path: the full
        append-only log, every projection this org has ever run, exactly as this
        method returned before. Nothing is ever deleted; the default read simply
        stops serving superseded projections beside current ones.

        The two queries below are a narrowing for scale, not the rule: they hand
        `current_insights` a superset of what it would have chosen from the whole
        table, so the answer is the same whether or not they run.
        """
        if include_superseded:
            return (
                await self.db.scalars(
                    select(IntelligenceInsight)
                    .where(IntelligenceInsight.organization_id == org_id)
                    .order_by(
                        IntelligenceInsight.insight_type.asc(),
                        IntelligenceInsight.key.asc(),
                    )
                )
            ).all()
        return current_insights(
            [*await self._learning_rows(org_id), *await self._reality_rows(org_id)]
        )

    async def _learning_rows(self, org_id: uuid.UUID) -> Sequence[IntelligenceInsight]:
        """Newest row per (insight_type, key) — DISTINCT ON does the per-key pick."""
        return (
            await self.db.scalars(
                select(IntelligenceInsight)
                .where(
                    IntelligenceInsight.organization_id == org_id,
                    IntelligenceInsight.insight_type.not_in(_REALITY_TYPES),
                )
                .distinct(IntelligenceInsight.insight_type, IntelligenceInsight.key)
                .order_by(
                    IntelligenceInsight.insight_type.asc(),
                    IntelligenceInsight.key.asc(),
                    IntelligenceInsight.learning_version.desc(),
                    IntelligenceInsight.created_at.desc().nulls_last(),
                    IntelligenceInsight.insight_id.desc(),
                )
            )
        ).all()

    async def _reality_rows(self, org_id: uuid.UUID) -> Sequence[IntelligenceInsight]:
        """Every row of the newest State snapshot — ALL its reasoning versions.

        The version is not a column (it lives in the id), so the query cannot
        narrow past the watermark; `current_insights` picks the version. Two
        passes over one snapshot is the worst case, so this stays a handful of
        rows however long the org's history grows.
        """
        watermarked = (
            IntelligenceInsight.organization_id == org_id,
            IntelligenceInsight.insight_type.in_(_REALITY_TYPES),
        )
        watermark = await self.db.scalar(
            select(IntelligenceInsight.state_watermark)
            .where(*watermarked)
            .group_by(IntelligenceInsight.state_watermark)
            .order_by(
                func.max(IntelligenceInsight.created_at).desc().nulls_last(),
                IntelligenceInsight.state_watermark.desc().nulls_last(),
            )
            .limit(1)
        )
        return (
            await self.db.scalars(
                select(IntelligenceInsight)
                .where(
                    *watermarked,
                    IntelligenceInsight.state_watermark.is_(None)
                    if watermark is None
                    else IntelligenceInsight.state_watermark == watermark,
                )
                .order_by(
                    IntelligenceInsight.insight_type.asc(),
                    IntelligenceInsight.key.asc(),
                )
            )
        ).all()

    async def delete_org_insights(self, org_id: uuid.UUID) -> None:
        from sqlalchemy import delete

        await self.db.execute(
            delete(IntelligenceInsight).where(IntelligenceInsight.organization_id == org_id)
        )

    async def flush(self) -> None:
        await self.db.flush()
