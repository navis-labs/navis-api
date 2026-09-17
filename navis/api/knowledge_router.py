"""Knowledge edge — canonical organizational entities, straight from State V2.

WHAT THIS REPLACES. `/memory/people` and `/decision-memory` read raw
`memory_records` and re-derive, in the API layer, things State V2 already holds:
who exists, what they are connected to, when they were first and last seen, and
where each fact came from. Worse, they derive things nothing holds — a
`responseLikelihood` bucket from an inbound/outbound ratio, an `escalationRisk`
from a commitment count, an `impact` score from a subject regex. The frontend
then rebuilt "Knowledge" on top by concatenating unrelated feeds and joining them
with `String.includes()`. Every one of those steps is a second organizational
model competing with the canonical one.

THE DEPENDENCY ARROW: API → State (+ Memory lineage). Same shape as
`context_router`, and the same composition-root pattern `orchestration.brief`
uses for Reasoning: read the snapshot, read the evidence, hand both to a PURE
projection (`state.v2.knowledge.project_knowledge`). No business logic here.

WHY THE EVIDENCE READ IS HERE AND NOT IN THE PROJECTION. `classifications_by_entity`
is I/O, and `navis/state/**` performs none. This is the layer permitted to touch
both — exactly as `build_brief` reads the same helper and hands it to
`run_reasoning_v2`. It is the SAME helper and the SAME predicates, so Knowledge
and Reasoning cannot disagree about who is a real counterparty.

READ-ONLY. Nothing here writes. `incremental_update` folds new memory into the
snapshot cache (the same call `/context` already makes) and is idempotent.
"""
from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from navis.identity.authz import Principal
from navis.identity.throttle import rate_limit
from navis.memory.envelope import EntityType
from navis.organization.contract import Capability
from navis.organization.guards import require_capability
from navis.platform.database import get_session
from navis.platform.monitoring import get_logger
from navis.platform.monitoring.metrics import timed
from navis.state.v2 import StateService
from navis.state.v2.knowledge import KnowledgeEntity, project_knowledge

logger = get_logger("knowledge")

router = APIRouter(prefix="/knowledge", tags=["knowledge"])

# Which entity types carry an identity verdict, and therefore need their lineage
# evidence loaded. Reading it for every type would be ten pointless joins: a
# meeting is a meeting whatever classification its mail carried.
_EVIDENCE_TYPES = (EntityType.PERSON, EntityType.RELATIONSHIP)

# How many entities of one type `/home` publishes. The org in production holds 86
# threads and 37 documents; a home surface wants a readable page, not the whole
# fold. `counts` stays exact and `gating` still reports the difference, so the
# bound is visible rather than silent.
_HOME_PER_TYPE = 50


def _identity(
    evidence: dict[str, dict[str, Any]],
) -> dict[str, dict[str, str]]:
    """Evidence -> the canonical identity verdict, decided AT THIS SEAM.

    WHY HERE AND NOT IN THE PROJECTION. The two predicates live in
    `reasoning.v2.base`, and `navis/state/**` may not import Reasoning — State is
    upstream of it, and `tests/state/test_scope_projection.py` pins that with an
    AST check. So the verdict is computed by the composition root and handed down
    as a plain string, exactly as `orchestration.brief` reads the same evidence
    and hands it to `run_reasoning_v2`.

    ONE vocabulary, one pair of predicates, two consumers. Reasoning uses them to
    decide whether a cooling relationship is worth raising; Knowledge uses them
    to decide whether an address is a person. Reproducing either test would be a
    second notion of "is this a real person" — the defect this endpoint removes.
    """
    from navis.reasoning.v2.base import (
        has_human_bearing_evidence,
        is_observed_non_human,
    )
    from navis.state.v2.knowledge import (
        IDENTITY_HUMAN,
        IDENTITY_NON_HUMAN,
        IDENTITY_UNKNOWN,
    )

    out: dict[str, dict[str, str]] = {}
    for entity_type, by_entity in evidence.items():
        verdicts: dict[str, str] = {}
        for entity_id, observed in by_entity.items():
            if has_human_bearing_evidence(observed):
                verdicts[entity_id] = IDENTITY_HUMAN
            elif is_observed_non_human(observed):
                verdicts[entity_id] = IDENTITY_NON_HUMAN
            else:
                verdicts[entity_id] = IDENTITY_UNKNOWN
        out[entity_type] = verdicts
    return out


async def _evidence(db: AsyncSession, org_id: uuid.UUID) -> dict[str, dict[str, Any]]:
    """{entity_type: {entity_id: frozenset[classification]}} for the gated types.

    THE CANONICAL IDENTITY READ, and the whole reason this endpoint exists. It is
    `memory.lineage.classifications_by_entity` — a set union over the append-only
    importance log — which is what Reasoning V2 already uses to decide whether a
    human is on the other end of a relationship.

    Measured on live production data it separates all five real correspondents
    from all thirty-three automated senders, including the nine
    `ghsa-*@noreply.github.com` reply-routing addresses that `_is_automated_email`
    misses because they arrive as `To:` recipients and its regex only reads the
    string. Evidence beats pattern-matching, and it is already here.
    """
    from navis.memory.lineage import classifications_by_entity

    out: dict[str, dict[str, Any]] = {}
    for entity_type in _EVIDENCE_TYPES:
        out[entity_type.value] = await classifications_by_entity(
            db, org_id, entity_type=entity_type.value
        )
    return out


def _matches(entity: KnowledgeEntity, needle: str) -> bool:
    """Whether this entity is a SEARCH hit for `needle`.

    Substring matching is legitimate HERE and nowhere else in this file: the user
    typed a fragment of a name and wants entities whose name contains it. That is
    what search is. It is emphatically NOT how entities are connected to one
    another — every relationship in this API comes from a persisted
    `EntityLink`, never from comparing two labels.

    Matches the canonical label and the identifying property values, never the
    whole `properties` blob: a commitment's body text would otherwise make every
    query match every commitment.
    """
    if needle in entity.canonical_label.lower():
        return True
    if needle in entity.entity_id.lower():
        return True
    for key in ("email", "person_email", "domain", "subject", "title", "name"):
        value = entity.properties.get(key)
        if isinstance(value, str) and needle in value.lower():
            return True
    return False


@router.get(
    "/home",
    dependencies=[Depends(rate_limit("read", "rate_limit_read_per_window"))],
)
async def knowledge_home(
    q: str | None = None,
    p: Principal = Depends(require_capability(Capability.DECISION)),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """Everything Navis currently knows about this organization.

    Returns canonical State entities grouped by canonical type, each carrying its
    persisted links, its provenance, both timestamps and its identity verdict.
    Nothing is constructed, classified or joined here — see
    `state.v2.knowledge.project_knowledge`.

    `q` filters the SAME canonical entities rather than searching a different
    surface. The old search read `/memory/people` and `/decision-memory` only, so
    a company, a commitment or a meeting could not be found at all.
    """
    with timed("knowledge.state"):
        state = await StateService(db).incremental_update(p.org_id)
    with timed("knowledge.evidence"):
        evidence = await _evidence(db, p.org_id)
    with timed("knowledge.project"):
        view = project_knowledge(
            state,
            evidence=evidence,
            identity=_identity(evidence),
            # Unbounded when searching: a query for a specific person must not
            # miss them because they fell outside the first page of their type.
            per_type_limit=None if q else _HOME_PER_TYPE,
        )

    body = view.to_dict()
    if q:
        needle = q.strip().lower()
        body["entities"] = {
            entity_type: [
                e.to_dict() for e in rows if _matches(e, needle)
            ]
            for entity_type, rows in view.entities.items()
        }
        body["entities"] = {k: v for k, v in body["entities"].items() if v}
        body["query"] = q
    logger.info(
        "knowledge home org=%s watermark=%s types=%d gated=%s q=%s",
        p.org_id, view.state_watermark, len(view.entities),
        {k: v["published"] for k, v in view.gating.items()}, bool(q),
    )
    return body


@router.get(
    "/entity/{entity_type}/{entity_id}",
    dependencies=[Depends(rate_limit("read", "rate_limit_read_per_window"))],
)
async def knowledge_entity(
    entity_type: str,
    entity_id: str,
    p: Principal = Depends(require_capability(Capability.DECISION)),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """One entity, its persisted neighbours, and what the pipeline has said about it.

    THE GRAPH IS TRAVERSED, NOT RECONSTRUCTED. `neighbours` resolves each
    `EntityLink` to the entity it addresses, so every connection shown is one
    State persisted. There is no text search anywhere on this path.

    `decisions` and `insights` are matched on canonical `entity_type` +
    `entity_id`, the same keys Decision and Intelligence already store — never on
    a title or a label. Two decisions can legitimately share a title (production
    holds exactly that case: one commitment text repeated in two different
    emails, two different entity ids), so matching on display text would both
    over-match and hide a real decision.
    """
    state = await StateService(db).incremental_update(p.org_id)
    if entity_type not in state.entities or entity_id not in (
        state.entities.get(entity_type) or {}
    ):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "entity not found")

    evidence = await _evidence(db, p.org_id)
    view = project_knowledge(
        state, evidence=evidence, identity=_identity(evidence)
    )
    # The projection gates `person`/`relationship` on evidence, so a withheld
    # entity is absent from the grouped lists. Detail must still serve it: a
    # founder following a link to Calendly deserves the honest page that says
    # what Navis observed, not a 404 that looks like data loss.
    subject = next(
        (e for e in view.entities.get(entity_type, ()) if e.entity_id == entity_id),
        None,
    )
    if subject is None:
        from navis.state.v2.knowledge import _entity

        typed = EntityType(entity_type)
        raw = next(
            (v for v in state._by_type(typed) if v.entity_id == entity_id), None
        )
        if raw is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "entity not found")
        subject = _entity(
            entity_type,
            raw,
            evidence.get(entity_type, {}).get(entity_id),
            _identity(evidence).get(entity_type, {}).get(entity_id, "unknown"),
        )

    # ── Neighbours: resolve each persisted edge to the entity it names ───────
    index = {
        (e.entity_type, e.entity_id): e
        for rows in view.entities.values()
        for e in rows
    }
    neighbours = []
    for link in subject.relationships:
        found = index.get((link.entity_type, link.entity_id))
        if found is None:
            # The edge exists but its target was gated or is a type this build
            # does not publish. Report the edge with what State does say rather
            # than dropping it — a missing neighbour is information.
            neighbours.append({
                "role": link.role,
                "entity_type": link.entity_type,
                "entity_id": link.entity_id,
                "canonical_label": None,
                "resolved": False,
            })
            continue
        neighbours.append({
            "role": link.role,
            "entity_type": found.entity_type,
            "entity_id": found.entity_id,
            "canonical_label": found.canonical_label,
            "identity": found.identity,
            "last_seen": found.last_seen,
            "resolved": True,
        })

    decisions, insights = await _pipeline_mentions(db, p.org_id, entity_type, entity_id)
    return {
        "as_of": view.as_of.isoformat() if view.as_of else None,
        "state_watermark": view.state_watermark,
        "entity": subject.to_dict(),
        "neighbours": neighbours,
        "decisions": decisions,
        "insights": insights,
    }


async def _pipeline_mentions(
    db: AsyncSession, org_id: uuid.UUID, entity_type: str, entity_id: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """What Decision and Intelligence have said about this exact entity.

    KNOWLEDGE READS THESE, IT DOES NOT COMPUTE THEM. The boundary in the
    architecture note is the point: Knowledge says "Ajitesh has an unresolved
    commitment"; Intelligence says what that MEANS alongside a cooling
    relationship; Decision says what to do. So this quotes their stored
    conclusions verbatim and adds no interpretation of its own — no risk score,
    no importance, no causality.

    Matched on canonical ids only. The latest brief is the current projection;
    older briefs describe superseded snapshots and would resurrect decisions the
    founder has already moved past.
    """
    from navis.decisions.service import DecisionService
    from navis.intelligence.repository import IntelligenceRepository

    decisions: list[dict[str, Any]] = []
    brief = await DecisionService(db).latest_brief(org_id)
    for d in (brief.decisions or []) if brief else []:
        if d.get("entity_type") == entity_type and str(d.get("entity_id")) == entity_id:
            decisions.append({
                # `decision_id` is the identity. Two decisions may share a title.
                "decision_id": d.get("decision_id"),
                "signal_type": d.get("signal_type"),
                "title": d.get("title"),
                "why": d.get("why"),
                "category": d.get("category"),
                "state_watermark": d.get("state_watermark"),
                "reasoning_signal_ids": d.get("reasoning_signal_ids") or [],
                "supporting_insight_ids": d.get("supporting_insight_ids") or [],
            })

    insights: list[dict[str, Any]] = []
    for row in await IntelligenceRepository(db).get_insights(org_id):
        named = (row.evidence or {}).get("entities") or ()
        if any(
            isinstance(x, dict)
            and x.get("entity_type") == entity_type
            and str(x.get("entity_id")) == entity_id
            for x in named
        ):
            insights.append({
                "insight_id": str(row.insight_id),
                "insight_type": row.insight_type,
                "title": row.title,
                "state_watermark": row.state_watermark,
            })
    return decisions, insights
