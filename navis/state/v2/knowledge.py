"""Knowledge — the human-readable projection of canonical organizational State.

WHAT THIS IS FOR. "What does Navis currently know about this organization?" The
answer already exists in exactly one place: the State V2 snapshot. Every entity,
every edge, every citation and both timestamps are already folded there, keyed by
an id that is stable across renames. Knowledge does not compute any of it — it
SHAPES it, the same way `scope.py` shapes State for an intent.

WHY IT LIVES HERE, next to `scope.py` and not in a new package. Scoping's own
header states the rule: a read-side concern on State is not a new layer. Knowledge
is a second read-side shape over the same snapshot, so it belongs beside the
first. A `navis/knowledge/**` package would be a second organizational model —
the exact thing the architecture forbids.

WHAT IT IS NOT
--------------
Not Intelligence. It reports what is true and what is connected; it never says
what that MEANS. No risk, no churn, no opportunity, no importance, no causality,
no baseline, no behavioural pattern. Those are Reasoning's and Intelligence's
words and they are earned from evidence this projection does not read.

Not a second identity system. WHICH ENTITIES ARE PEOPLE is decided by
`memory.lineage.classifications_by_entity` + `reasoning.v2.base`'s predicates —
the same canonical test Reasoning V2 already applies in production, handed in as
primitives exactly like `run_reasoning_v2(state, evidence=...)` receives it. There
is no regex here and there must never be one: `_is_automated_email` in
`api/memory_router.py` is a five-layer regex that misses nine GHSA advisory
addresses and six `noreply@` senders on live data, because it guesses from the
string instead of reading what was observed.

Not a source of truth. Every value is copied from State. A field State does not
hold is ABSENT, never defaulted — `UNKNOWN` and `UNAVAILABLE` are not zero.

PURE. No clock, no I/O, no randomness: same (state, evidence) always produces the
same view, so a replayed snapshot is byte-identical.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from navis.memory.envelope import EntityType
from navis.state.v2.models import OrganizationalState, _BaseView
from navis.state.v2.scope import EntityLink, _links

# ── Identity ────────────────────────────────────────────────────────────────

# The three answers the canonical evidence can give. Deliberately TRI-STATE, and
# `UNKNOWN` is deliberately not folded into either side: an entity Navis has
# observed nothing about is not thereby a person, and is not thereby a robot.
#
# THE VERDICT IS COMPUTED BY THE COMPOSITION ROOT, not here. Deciding it would
# mean importing `reasoning.v2.base`'s predicates, and `navis/state/**` may not
# depend on Reasoning — State is upstream of it, and an AST test in
# `tests/state/test_scope_projection.py` pins exactly that. So the verdict
# arrives as a plain string, the same way `run_reasoning_v2(state, evidence=...)`
# receives its evidence: read at the seam, flattened to primitives, handed in.
# See `api/knowledge_router._identity`.
IDENTITY_HUMAN = "human"
IDENTITY_NON_HUMAN = "observed_non_human"
IDENTITY_UNKNOWN = "unknown"


# Entity types whose membership is decided by observed evidence rather than by
# existence. A PERSON row is written for every address that ever appeared on a
# message — including the nine `ghsa-*@noreply.github.com` reply-routing
# addresses GitHub puts in `To:` — so "a person entity exists" is an observation,
# never a claim that a human is on the other end. Every other type means what it
# says: a meeting is a meeting.
_EVIDENCE_GATED = frozenset({EntityType.PERSON.value, EntityType.RELATIONSHIP.value})


# ── Output shape ────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class KnowledgeProvenance:
    """Where this entity's current state was learned. Pointers, never bodies.

    `Citation` in `scope.py` makes the same promise for the same reason: a view
    that inlined the raw email would stop being a projection of State and start
    being a second copy of Memory.
    """

    memory_ids: tuple[str, ...] = ()
    signal_event_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class KnowledgeEntity:
    """One canonical organizational entity, as Knowledge publishes it.

    `entity_id` is State's own id — `uuid5(ns, "{org}|{type}|{natural_key}")`
    where the natural key is the EMAIL for a person, the DOMAIN for a company,
    the thread id for a thread. A display name is never part of it, so renaming
    somebody does not mint a new entity. Display strings are never identity.

    `relationships` are `EntityLink`s read from State's persisted graph. They are
    never derived, never string-matched, and never inferred.
    """

    entity_type: str
    entity_id: str
    canonical_label: str
    properties: dict[str, Any] = field(default_factory=dict)
    relationships: tuple[EntityLink, ...] = ()
    provenance: KnowledgeProvenance = field(default_factory=KnowledgeProvenance)
    first_seen: str | None = None
    last_seen: str | None = None
    # The classifications observed across this entity's whole lineage, and the
    # verdict they support. Carried on every entity so a surface can SHOW why an
    # address is or is not treated as a person instead of trusting a filter it
    # cannot see.
    observed_evidence: tuple[str, ...] = ()
    identity: str = IDENTITY_UNKNOWN

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_type": self.entity_type,
            "entity_id": self.entity_id,
            "canonical_label": self.canonical_label,
            "properties": self.properties,
            "relationships": [
                {"role": r.role, "entity_type": r.entity_type, "entity_id": r.entity_id}
                for r in self.relationships
            ],
            "provenance": {
                "memory_ids": list(self.provenance.memory_ids),
                "signal_event_ids": list(self.provenance.signal_event_ids),
            },
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "observed_evidence": list(self.observed_evidence),
            "identity": self.identity,
        }


@dataclass(frozen=True, slots=True)
class KnowledgeView:
    """Everything Navis currently knows, grouped by canonical entity type.

    `counts` are State's OWN counts — `state.counts()` — not the length of the
    published lists. The two differ whenever a type is bounded or gated, and the
    difference is exactly what a surface must show rather than hide: "5 of 38
    person entities carry human evidence" is the honest sentence, and
    `len(people)` alone cannot say it.
    """

    org_id: str
    as_of: datetime | None
    state_watermark: str | None
    projection_version: int
    entities: dict[str, tuple[KnowledgeEntity, ...]]
    counts: dict[str, int]
    # Per gated type: how many entities exist, how many were published, and why
    # the rest were not. Never a silent filter.
    gating: dict[str, dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "org_id": self.org_id,
            "as_of": self.as_of.isoformat() if self.as_of else None,
            "state_watermark": self.state_watermark,
            "projection_version": self.projection_version,
            "entities": {
                key: [e.to_dict() for e in rows] for key, rows in self.entities.items()
            },
            "counts": self.counts,
            "gating": self.gating,
        }


# ── Canonical labels ────────────────────────────────────────────────────────


def _first_str(view: _BaseView, *keys: str) -> str | None:
    for key in keys:
        value = view.fields.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


# WHICH FIELD NAMES AN ENTITY, per type, in preference order. Read straight off
# the typed accessors in `state.v2.models` — `PersonState.name/.email`,
# `CompanyState.domain`, `MeetingState.title`, `ThreadState.subject`,
# `RelationshipState.person_email` — so this cannot drift into inventing a name.
# The entity id is the last resort and is a real identifier, never a placeholder
# that would read as a name.
_LABEL_FIELDS: dict[str, tuple[str, ...]] = {
    EntityType.PERSON.value: ("name", "email"),
    EntityType.COMPANY.value: ("domain", "name"),
    EntityType.RELATIONSHIP.value: ("person_email",),
    EntityType.COMMITMENT.value: ("text", "subject"),
    EntityType.MEETING.value: ("title",),
    EntityType.THREAD.value: ("subject",),
    EntityType.DOCUMENT.value: ("title", "name"),
    EntityType.PROJECT.value: ("name", "folder_id"),
    EntityType.CUSTOMER.value: ("name", "external_ref"),
    EntityType.DECISION.value: ("text", "subject"),
    EntityType.CHANNEL.value: ("name", "channel_id"),
}


def canonical_label(entity_type: str, view: _BaseView) -> str:
    """A name a founder can read, taken from State and never constructed."""
    return _first_str(view, *_LABEL_FIELDS.get(entity_type, ())) or view.entity_id


# ── Projection ──────────────────────────────────────────────────────────────


def _entity(
    entity_type: str,
    view: _BaseView,
    observed: frozenset[str] | None,
    identity: str = IDENTITY_UNKNOWN,
) -> KnowledgeEntity:
    prov = view.provenance or {}
    return KnowledgeEntity(
        entity_type=entity_type,
        entity_id=view.entity_id,
        canonical_label=canonical_label(entity_type, view),
        # State's current field dict, copied so a consumer cannot mutate the
        # snapshot through it. Nothing is added, removed or renamed.
        properties=dict(view.fields or {}),
        # The persisted graph, typed by `scope._links` — one role map, shared
        # with Context, so Knowledge and Context can never disagree about what
        # an edge points at.
        relationships=_links(view),
        provenance=KnowledgeProvenance(
            memory_ids=tuple(prov.get("memory_ids", [])),
            signal_event_ids=tuple(prov.get("signal_ids", [])),
        ),
        first_seen=view.first_seen,
        last_seen=view.last_seen,
        observed_evidence=tuple(sorted(observed or ())),
        identity=identity,
    )


def project_knowledge(
    state: OrganizationalState,
    *,
    evidence: dict[str, dict[str, frozenset[str]]] | None = None,
    identity: dict[str, dict[str, str]] | None = None,
    per_type_limit: int | None = None,
) -> KnowledgeView:
    """Shape a State snapshot into Knowledge. PURE and deterministic.

    `evidence` is `{entity_type: {entity_id: frozenset[classification]}}`, built
    by the composition root from `memory.lineage.classifications_by_entity`, and
    `identity` is `{entity_type: {entity_id: verdict}}` — that same evidence run
    through Reasoning's two predicates at the seam. Both are handed in rather
    than fetched or decided here, for the same two reasons Reasoning's evidence
    is: this stays a pure function of a snapshot, and `navis/state/**` may
    neither perform I/O nor import Reasoning.

    An entity absent from `identity` is UNKNOWN — never "automated", never
    "human". A caller that supplies no `identity` at all therefore publishes no
    gated entities, which is the correct fail-closed answer to "we did not
    establish who these are".

    GATING, AND WHY IT IS REPORTED RATHER THAN APPLIED SILENTLY. A `person` row
    exists for every address ever seen on a message, so publishing them all puts
    Calendly, GoDaddy and nine GitHub advisory ids on a page headed "People".
    Publishing only the human-bearing ones is correct, but doing it quietly would
    make 33 entities vanish with no way to ask where they went. So both numbers
    travel: `counts` is what State holds, the list is what carried evidence, and
    `gating` names the difference and the rule that produced it.

    Ordering is by (identity, canonical_label, entity_id) — a total order that
    does not depend on dict iteration, so one snapshot always projects
    identically. `per_type_limit` truncates the LIST only; `counts` stays exact.
    """
    evidence = evidence or {}
    identity = identity or {}
    entities: dict[str, tuple[KnowledgeEntity, ...]] = {}
    gating: dict[str, dict[str, Any]] = {}

    for entity_type in sorted(state.entities):
        bucket = state.entities.get(entity_type) or {}
        if not bucket:
            continue
        try:
            typed = EntityType(entity_type)
        except ValueError:
            # A snapshot holding a type this build does not know is reported by
            # `counts` and not published — guessing its shape would be inventing.
            continue
        by_entity = evidence.get(entity_type, {})
        verdicts = identity.get(entity_type, {})
        rows = [
            _entity(
                entity_type,
                view,
                by_entity.get(view.entity_id),
                verdicts.get(view.entity_id, IDENTITY_UNKNOWN),
            )
            for view in state._by_type(typed)
        ]
        if entity_type in _EVIDENCE_GATED:
            published = [r for r in rows if r.identity == IDENTITY_HUMAN]
            gating[entity_type] = {
                "rule": "has_human_bearing_evidence",
                "total": len(rows),
                "published": len(published),
                "withheld": len(rows) - len(published),
                # The withheld rows split by WHY, so "we observed these are not
                # people" stays distinguishable from "we have observed nothing
                # about these" — which is the difference between a filter and a
                # gap in the data.
                "withheld_observed_non_human": sum(
                    1 for r in rows if r.identity == IDENTITY_NON_HUMAN
                ),
                "withheld_unknown": sum(
                    1 for r in rows if r.identity == IDENTITY_UNKNOWN
                ),
            }
            rows = published
        rows.sort(key=lambda r: (r.identity, r.canonical_label.lower(), r.entity_id))
        if per_type_limit is not None:
            rows = rows[:per_type_limit]
        entities[entity_type] = tuple(rows)

    watermark = state.watermark.last_memory_id
    return KnowledgeView(
        org_id=str(state.org_id),
        # State's logical clock, never wall clock — so "as of" is the moment of
        # the newest fact folded, not the moment the page was requested.
        as_of=state.as_of,
        state_watermark=str(watermark) if watermark else None,
        projection_version=state.projection_version,
        entities=entities,
        # State's own counts for EVERY type it models, including the ones with
        # zero rows. A type that is genuinely empty reports 0; a type this build
        # cannot publish still reports its real count above.
        counts=state.counts(),
        gating=gating,
    )
