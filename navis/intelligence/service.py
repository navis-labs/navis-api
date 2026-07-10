"""IntelligenceService — project Learning into append-only organizational insights.

Deterministic and append-only: same Learning => same insight_ids => same rows.
created_at is logical time (max occurred_at of the consumed learning), never wall clock.
Intelligence consumes Learning ONLY. No State, Memory, Signals, Reasoning, etc.
"""
from __future__ import annotations

import uuid
from typing import Any

from navis.intelligence.insights import InsightDraft, LearningView
from navis.intelligence.insights import project as project_view
from navis.intelligence.models import compute_insight_id


def draft_to_values(view: LearningView, draft: InsightDraft) -> dict[str, Any]:
    """Pure mapping from a draft + view to an append-only insight row."""
    insight_id = compute_insight_id(
        view.org_id, draft.insight_type.value, draft.key, view.learning_version
    )
    return {
        "insight_id": insight_id,
        "organization_id": view.org_id,
        "insight_type": draft.insight_type.value,
        "key": draft.key,
        "title": draft.title,
        "summary": draft.summary,
        "evidence": draft.evidence,
        "sample_size": draft.sample_size,
        "confidence": draft.confidence,
        "supporting_learning_event_ids": sorted(draft.supporting)[:50],
        "learning_version": view.learning_version,
        "created_at": view.as_of,
    }


def project_values(view: LearningView) -> list[dict[str, Any]]:
    """The full deterministic projection as plain row dicts (pure, no I/O)."""
    return [draft_to_values(view, d) for d in project_view(view)]


class IntelligenceService:
    def __init__(self, repo) -> None:
        self.repo = repo

    async def project(self, org_id: uuid.UUID) -> list[dict[str, Any]]:
        view = await self.repo.build_view(org_id)
        if view.learning_version == 0:
            return []  # nothing learned yet => no insights
        rows = project_values(view)
        for values in rows:
            await self.repo.append_insight(values)
        await self.repo.flush()
        return rows

    async def get_insights(self, org_id: uuid.UUID):
        return await self.repo.get_insights(org_id)

    async def get_influence_graph(self, org_id: uuid.UUID):
        """The Learning-scoped knowledge graph: signal/decision -> outcome-type influence.
        Deterministic, derived-on-read (no table); the ORG ENTITY graph is State's, not
        Intelligence's — see navis.intelligence.graph for the boundary rationale."""
        from navis.intelligence.graph import build_influence_graph

        events = await self.repo.influence_events(org_id)
        return build_influence_graph(events)

    async def get_trends(self, org_id: uuid.UUID):
        """The temporal dimension: per-signal/decision outcome trends (improving/degrading/
        stable), recent vs prior window. Deterministic, derived-on-read from Learning only."""
        from navis.intelligence.trends import build_trends

        events = await self.repo.recent_events(org_id)
        return build_trends(events)
