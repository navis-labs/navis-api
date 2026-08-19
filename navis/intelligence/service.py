"""IntelligenceService — project Learning into append-only organizational insights.

Deterministic and append-only: same Learning => same insight_ids => same rows.
created_at is logical time (max occurred_at of the consumed learning), never wall clock.
Intelligence consumes Learning ONLY. No State, Memory, Signals, Reasoning, etc.
"""
from __future__ import annotations

import uuid
from typing import Any

from navis.intelligence.insights import (
    REALITY_INSIGHT_TYPES,
    InsightDraft,
    LearningView,
    OrganizationalRealityView,
)
from navis.intelligence.insights import project as project_view
from navis.intelligence.models import compute_insight_id


def _authoritative_watermark(
    view: LearningView, reality: OrganizationalRealityView | None
) -> str | None:
    """The State snapshot this projection describes.

    Reality wins when it is supplied, and it is not a preference: it IS the
    watermark of the State that was folded, read straight off
    `OrganizationalState.watermark`. The Learning value is whatever an execution
    happened to record on its way past — live, both calendar executions recorded
    NULL, so `LearningView.state_watermark` (the newest non-null across the org)
    can stamp an insight with an unrelated action's watermark.

    Falls back to Learning's value so a Learning-only projection — the
    LEARNING_UPDATED subscriber and POST /intelligence/project both call with
    `reality=None` — behaves exactly as it did.
    """
    if reality is not None and reality.state_watermark:
        return reality.state_watermark
    return view.state_watermark or None


def _version_key(
    view: LearningView,
    draft: InsightDraft,
    reality: OrganizationalRealityView | None,
) -> str:
    """Which clock this insight's identity is versioned against.

    A Learning fold changes when an outcome is recorded; a reality fold changes
    when State moves. Keying both on `learning_version` would freeze every
    reality insight at whatever was true the first time it was projected —
    the table is append-only with ON CONFLICT DO NOTHING, so the row would never
    be revisited. Membership in `REALITY_INSIGHT_TYPES` is the only test.
    """
    if draft.insight_type in REALITY_INSIGHT_TYPES:
        # Guaranteed present: `insights.project` refuses to run the reality
        # generators without a watermark, so a reality draft cannot exist here
        # without one.
        assert reality is not None and reality.state_watermark
        # BOTH clocks. A reality insight is a function of the State snapshot AND
        # of the reasoning that interpreted it; keying on the watermark alone let
        # a reasoning-version change collide with the conclusions it replaced,
        # and ON CONFLICT DO NOTHING then kept the superseded row forever. See
        # OrganizationalRealityView.reasoning_version.
        return f"{reality.state_watermark}|r{reality.reasoning_version}"
    return str(view.learning_version)


def _provenance(
    view: LearningView,
    draft: InsightDraft,
    reality: OrganizationalRealityView | None = None,
) -> dict[str, Any]:
    """The evidence chain for one insight, as far as its inputs can carry it.

    Rides in `evidence` (already JSONB) rather than in new columns, so the
    contract widens without a migration and older rows stay readable.

    Deliberately omits keys it cannot fill: an insight for which Learning
    recorded no watermark or no reasoning signals says nothing, rather than
    saying `null`. Absence is honest; a null would read as "we looked and there
    was none".

    NOT AVAILABLE HERE: memory_ids / signal_event_ids. LearningEvent records
    outcome, decision, execution and reasoning-signal ids but no Memory lineage,
    so the chain stops at the ReasoningSignal. A reality view does not close that
    gap either — it carries each signal's own `evidence` verbatim and nothing
    more. Reaching further would mean Intelligence querying Memory directly,
    which the layer boundary forbids; it belongs on LearningEvent, upstream.
    """
    out: dict[str, Any] = {}
    watermark = _authoritative_watermark(view, reality)
    if watermark:
        out["state_watermark"] = watermark
    # A reality draft names the signals it folded; a Learning draft's signal
    # lineage is looked up from the learning events that supported it. Never
    # both, and never a fabricated empty list — absence stays absence.
    signal_ids = list(draft.supporting_signals) or (view.signal_lineage.get(draft.key) or [])
    if signal_ids:
        out["supporting_reasoning_signal_ids"] = sorted(signal_ids)
    return out


def draft_to_values(
    view: LearningView,
    draft: InsightDraft,
    reality: OrganizationalRealityView | None = None,
) -> dict[str, Any]:
    """Pure mapping from a draft + view to an append-only insight row.

    `version_key` is `learning_version` for every draft the six generators
    produce, because all six are folds over the Learning log — reality does not
    change what they conclude and must not change their ids. A reality-derived
    generator (a later phase) keys on the watermark instead; see
    `compute_insight_id`.
    """
    is_reality = draft.insight_type in REALITY_INSIGHT_TYPES
    insight_id = compute_insight_id(
        view.org_id,
        draft.insight_type.value,
        draft.key,
        _version_key(view, draft, reality),
    )
    return {
        "insight_id": insight_id,
        "organization_id": view.org_id,
        "insight_type": draft.insight_type.value,
        "key": draft.key,
        "title": draft.title,
        "summary": draft.summary,
        # Engine-derived facts first; provenance cannot overwrite a measurement.
        "evidence": {**draft.evidence, **_provenance(view, draft, reality)},
        "sample_size": draft.sample_size,
        "confidence": draft.confidence,
        "supporting_learning_event_ids": sorted(draft.supporting)[:50],
        # Carried on a reality insight too, and deliberately: it records the
        # Learning state this reality was projected alongside, and keeps
        # Adaptive's `adaptive_version` (a max over this column) reading exactly
        # as it did before reality insights existed.
        "learning_version": view.learning_version,
        "state_watermark": _authoritative_watermark(view, reality),
        "engine_name": draft.engine_name,
        "engine_version": draft.engine_version,
        # Logical time: State's clock for a reality fold, Learning's for a
        # Learning fold. Never wall clock, either way.
        "created_at": (reality.as_of if is_reality and reality else view.as_of),
    }


def project_values(
    view: LearningView, reality: OrganizationalRealityView | None = None
) -> list[dict[str, Any]]:
    """The full deterministic projection as plain row dicts (pure, no I/O)."""
    return [draft_to_values(view, d, reality) for d in project_view(view, reality)]


class IntelligenceService:
    def __init__(self, repo) -> None:
        self.repo = repo

    async def project(
        self,
        org_id: uuid.UUID,
        *,
        reality: OrganizationalRealityView | None = None,
    ) -> list[dict[str, Any]]:
        """Project this org's insights from Learning, and — when supplied — the
        current organizational reality.

        `reality` is keyword-only with a `None` default so both existing callers
        (the LEARNING_UPDATED subscriber and POST /intelligence/project) keep
        working untouched and keep producing byte-identical rows. It is built by
        `navis.orchestration.brief`, the composition root; Intelligence never
        reads State or Reasoning itself.

        In this phase reality changes exactly one thing: the `state_watermark`
        an insight is stamped with becomes the one State actually folded to,
        rather than one inherited from whichever LearningEvent happened to record
        it. No generator consumes reality yet.
        """
        if reality is not None and reality.org_id != org_id:
            # Fail loud rather than project one org's reality into another's
            # insights. Every id in this table hashes org_id, so a mismatch
            # would not collide — it would silently attribute Org A's state to
            # Org B's row, which is worse than a crash.
            raise ValueError(
                f"reality view is for org {reality.org_id}, not {org_id}"
            )
        view = await self.repo.build_view(org_id)
        if view.learning_version == 0 and reality is None:
            # Nothing learned and no reality supplied => nothing to project.
            #
            # This used to short-circuit on `learning_version == 0` alone, which
            # was right while Learning was the only input and became wrong the
            # moment reality became a second one: an organization that has
            # tracked State and reached reasoning conclusions but has not yet
            # EXECUTED anything has plenty that is true of it, and would have
            # been given no insights at all until its first outcome landed.
            # The six Learning generators still emit nothing on an empty view
            # (every one of them gates on decisive samples), so allowing the
            # pass to continue costs nothing and changes no Learning output.
            return []
        rows = project_values(view, reality)
        for values in rows:
            await self.repo.append_insight(values)
        await self.repo.flush()
        return rows

    async def get_insights(self, org_id: uuid.UUID, *, include_superseded: bool = False):
        """The org's current insights; `include_superseded` serves the full log."""
        return await self.repo.get_insights(
            org_id, include_superseded=include_superseded
        )
