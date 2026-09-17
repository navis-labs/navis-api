"""Intent-scoped view over the current OrganizationalState.

State V2 is source-of-truth ("what is currently the case" for the whole org).
Consumers usually want a shaped slice — e.g. "the founder's morning brief" picks
a small top-N per entity type, ranked by relevance to that intent. This module
is that shaping function.

Design decisions this file locks in (see "Context Layer — Sanity Check"):

  * Pure function of (state, intent, policy, as_of). Same inputs → same output.
    ``as_of`` is REQUIRED even when a given intent doesn't consume it, so no
    caller ever leaks wall-clock time into a scoped view. Matches State's own
    determinism law.
  * Read-only over State. Never writes memory/state, never reaches into
    Reasoning or Decision internals. If it ever needs to, the design is wrong.
  * Ephemeral. Not persisted, not cached. State's snapshot is already the cache.
    (Assembly of a morning-brief view is O(entities) — cheap compared to the
    fold that produced the state.)
  * Lives inside state/v2/ because scoping is a read-side concern on State,
    not a new layer. Extraction to its own package is warranted only when a
    2nd intent proves the pattern generalises cleanly.

Adding a new intent = add a variant to ``Intent`` + a match arm in
``scope_for``. Each intent's implementation should be a small, testable
function that declares its own citation strategy.
"""
from __future__ import annotations

import itertools
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Union

from navis.memory.envelope import EntityType
from navis.state.v2.models import OrganizationalState, _BaseView


# ── Intents ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class MorningBrief:
    """The founder's daily scan. Top-N per entity type, ordered so what needs
    attention surfaces first. Deterministic given (state, policy, as_of)."""


# The union widens as new intents are added. Callers should ``match`` on it so
# adding a variant is a compile-time reminder to update every dispatch.
Intent = Union[MorningBrief]


# ── Output shape ────────────────────────────────────────────────────────────


# WHICH ENTITY TYPE EACH LINK ROLE POINTS AT.
#
# State stores links as ``{role: [entity_id, ...]}`` and the entity ids alone do
# not say what they address — so a consumer holding ``{"owner": ["455812d6"]}``
# cannot resolve it without knowing that an owner is a person. That knowledge
# exists in exactly one place, `folds.compute_links`, and this is its read-side
# transcription: every ``_link(...)`` call there appears here and nothing else
# does.
#
# CLOSED, and deliberately so. A role this map does not name is DROPPED rather
# than guessed — see `_links`. Inventing a type for an unrecognised role is the
# one way this module could manufacture a relationship, which is precisely what
# it must never do. Adding a link in `compute_links` means adding it here; the
# pinning test (`tests/state/test_scope_projection.py`) fails otherwise.
_LINK_TARGET_TYPE: dict[str, str] = {
    "relationships": EntityType.RELATIONSHIP.value,  # person   -> relationship
    "commitments":   EntityType.COMMITMENT.value,    # relationship -> commitment
    "meetings":      EntityType.MEETING.value,       # relationship -> meeting
    "owner":         EntityType.PERSON.value,        # commitment -> person
    "owners":        EntityType.PERSON.value,        # document -> person
    "people":        EntityType.PERSON.value,        # company/channel -> person
    "channels":      EntityType.CHANNEL.value,       # person -> channel
    "documents":     EntityType.DOCUMENT.value,      # person/project -> document
    "projects":      EntityType.PROJECT.value,       # document -> project
}


@dataclass(frozen=True, slots=True)
class EntityLink:
    """One persisted edge from an entity in scope to another entity.

    THE THING CONTEXT USED TO DROP. State computes a real entity graph in
    `folds.compute_links` — a commitment names its owner, a relationship names
    its commitments and meetings — and `EntityRef` carried `fields` but not
    `links`, so every consumer downstream of Context received a ranked FLAT LIST
    and had to treat each entity as unrelated to every other. Live, that is why
    one slipping commitment and the cooling relationship of the very same person
    reached the brief as two independent cards.

    ``role`` is State's own link key, verbatim ("owner", "commitments",
    "meetings"). ``entity_type`` is resolved from `_LINK_TARGET_TYPE`, never
    inferred from the id.

    DETERMINISTIC AND PERSISTED. Nothing here is derived, scored, or guessed:
    the edge either exists in `state.links` or it does not appear.
    """
    role: str
    entity_type: str
    entity_id: str


@dataclass(frozen=True, slots=True)
class EntityRef:
    """One entity in scope, with its rank and why it made the cut.

    ``fields`` is the entity's current-state field dict (from State V2's raw
    fold — same shape typed views expose). Consumers pick what they need.

    ``links`` is the entity's persisted edges (see `EntityLink`). Empty for an
    entity State recorded none for, which is honest absence rather than a gap.
    """
    entity_type: str
    entity_id: str
    rank: int
    reason: str
    fields: dict[str, Any]
    first_seen: str | None
    last_seen: str | None
    # Defaulted so every existing construction site — tests and fakes included —
    # stays valid and keeps producing the view it did before.
    links: tuple[EntityLink, ...] = ()


@dataclass(frozen=True, slots=True)
class Citation:
    """Provenance chain back to Memory / Signals for one focused entity.

    Consumers that need the raw email / meeting / doc drill from memory_id to
    the source signal_event. Never inline the raw payload here — that would
    make ScopedView a body-carrying object; it stays a pointer-carrying one.
    """
    entity_id: str
    memory_ids: list[str]
    signal_ids: list[str]


@dataclass(frozen=True, slots=True)
class ScopedView:
    """Purpose-shaped read over State.

    Not truth — always cite through to Memory. Callers that want a raw fact
    resolve ``citations[entity_id].memory_ids`` back to memory_records.

    ``intent_name`` / ``as_of`` / ``state_watermark`` / ``policy_version`` are
    the reproducibility keys: same tuple → same view, forever.
    """
    intent_name: str
    as_of: datetime | None
    state_watermark: str | None
    policy_version: int | None
    counts: dict[str, int]
    entities: list[EntityRef]
    citations: dict[str, Citation]

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent_name,
            "as_of": self.as_of.isoformat() if self.as_of else None,
            "state_watermark": self.state_watermark,
            "policy_version": self.policy_version,
            "counts": self.counts,
            "entities": [
                {
                    "entity_type": e.entity_type,
                    "entity_id": e.entity_id,
                    "rank": e.rank,
                    "reason": e.reason,
                    "fields": e.fields,
                    "first_seen": e.first_seen,
                    "last_seen": e.last_seen,
                    # ADDITIVE. Every key above is untouched and in the same
                    # place; a consumer that does not read `links` sees exactly
                    # the view it saw before. Always present (an empty list when
                    # State recorded no edge) so a client can tell "this entity
                    # has no links" from "this build does not report them".
                    "links": [
                        {
                            "role": link.role,
                            "entity_type": link.entity_type,
                            "entity_id": link.entity_id,
                        }
                        for link in e.links
                    ],
                }
                for e in self.entities
            ],
            "citations": {
                eid: {"memory_ids": c.memory_ids, "signal_ids": c.signal_ids}
                for eid, c in self.citations.items()
            },
        }


# ── Dispatch ────────────────────────────────────────────────────────────────


def scope_for(
    state: OrganizationalState,
    intent: Intent,
    policy: dict | None,
    *,
    as_of: datetime | None,
) -> ScopedView:
    """Shape a state snapshot for an intent. Pure, deterministic.

    ``as_of`` is REQUIRED (keyword-only). Callers normally pass ``state.as_of``.
    Intents that rank by "age" use it; intents that only sort by last_seen
    don't, but the parameter is mandatory anyway so no callsite forgets when
    a future intent adds age-based ranking.
    """
    if isinstance(intent, MorningBrief):
        return _scope_morning_brief(state, policy, as_of=as_of)
    raise ValueError(f"unknown intent: {intent!r}")


# ── MorningBrief intent ─────────────────────────────────────────────────────


# Sensible per-type caps. The Adaptive policy can tighten via `brief_max_items`
# but never loosen past these — the intent is a "scan," not a data dump.
_DEFAULT_TOP_N = 5
_MAX_TOP_N = 25


def _cap(policy: dict | None) -> int:
    if not policy:
        return _DEFAULT_TOP_N
    raw = policy.get("brief_max_items", _DEFAULT_TOP_N)
    try:
        n = int(raw)
    except (TypeError, ValueError):
        return _DEFAULT_TOP_N
    return max(1, min(n, _MAX_TOP_N))


def _cite(view: _BaseView) -> Citation:
    prov = view.provenance or {}
    return Citation(
        entity_id=view.entity_id,
        memory_ids=list(prov.get("memory_ids", []))[-10:],  # bounded, newest-most
        signal_ids=list(prov.get("signal_ids", []))[-10:],
    )


def _links(view: _BaseView) -> tuple[EntityLink, ...]:
    """The entity's persisted edges, typed and deterministically ordered.

    Reads `state.links` — computed by `folds.compute_links` from joins on data
    the extractors recorded (an email address, a meeting attendee list, a folder
    id) — and attaches the target entity type from `_LINK_TARGET_TYPE`.

    FAILS CLOSED IN BOTH DIRECTIONS. A role the map does not name is skipped, and
    so is a falsy id: an unrecognised edge contributes nothing rather than an
    entity reference whose type is a guess. Sorted by (role, entity_id) so one
    snapshot always flattens identically, whatever order the fold produced.
    """
    out: list[EntityLink] = []
    for role, ids in (view.links or {}).items():
        target = _LINK_TARGET_TYPE.get(role)
        if target is None:
            continue
        for entity_id in ids or ():
            if entity_id:
                out.append(
                    EntityLink(role=role, entity_type=target, entity_id=str(entity_id))
                )
    return tuple(sorted(out, key=lambda link: (link.role, link.entity_id)))


def _ref(entity_type: str, view: _BaseView, rank: int, reason: str) -> EntityRef:
    return EntityRef(
        entity_type=entity_type,
        entity_id=view.entity_id,
        rank=rank,
        reason=reason,
        fields=dict(view.fields or {}),
        first_seen=view.first_seen,
        last_seen=view.last_seen,
        links=_links(view),
    )


def _sort_desc_by_last_seen(views: Sequence[_BaseView]) -> list[_BaseView]:
    # last_seen is ISO-8601 (see builders.py); lexical sort == chronological.
    # entity_id tiebreaker so two entities with identical last_seen produce a
    # stable order across runs.
    return sorted(views, key=lambda v: (v.last_seen or "", v.entity_id), reverse=True)


def _sort_asc_by_last_seen(views: Sequence[_BaseView]) -> list[_BaseView]:
    return sorted(views, key=lambda v: (v.last_seen or "￿", v.entity_id))


def _open_only(views: Sequence[_BaseView]) -> list[_BaseView]:
    """Entities whose fact-derived `status` has not closed out. Absent == open,
    matching CommitmentState/DecisionState's own default."""
    return [v for v in views if (v.fields or {}).get("status", "open") == "open"]


# ── Per-type selection policy ───────────────────────────────────────────────
#
# THE DEFECT THIS REPLACES. `_scope_morning_brief` used to hardcode five
# `state.<type>()` calls, so State could hold meetings, documents, decisions,
# projects, customers and channels and Context would silently drop every one of
# them — a real organizational entity could never reach Reasoning or ML. Worse,
# the failure was invisible and recurring: every entity type added to State
# since the function was written was dropped by default.
#
# The projection is now keyed by EntityType and asserted EXHAUSTIVE by
# `tests/state/test_scope_projection.py`, so a new entity type cannot be added
# without deciding how it is scoped. Complete by construction, not by vigilance.
#
# Every rule ranks on NORMALIZED State fields only — `last_seen`, `status`,
# link counts. There is deliberately no branch on source/connector anywhere in
# this module: a Google Calendar meeting and a Teams meeting are both a
# MeetingState, and a Drive doc and a Notion page are both a DocumentState.


@dataclass(frozen=True, slots=True)
class _TypePolicy:
    """How one entity type is selected and ranked, and why it made the cut."""

    reason: str
    select: Callable[[OrganizationalState], list[_BaseView]]


_MORNING_BRIEF: dict[EntityType, _TypePolicy] = {
    # "What's slipping" — open promises, least recently touched first.
    EntityType.COMMITMENT: _TypePolicy(
        "open, oldest activity",
        lambda s: _sort_asc_by_last_seen(_open_only(s.commitments())),
    ),
    # "Who's going cold" — stalest counterparty first.
    EntityType.RELATIONSHIP: _TypePolicy(
        "stalest interaction",
        lambda s: _sort_asc_by_last_seen(s.relationships()),
    ),
    # "What's active" — freshest conversation first.
    EntityType.THREAD: _TypePolicy(
        "most recently active",
        lambda s: _sort_desc_by_last_seen(s.threads()),
    ),
    EntityType.PERSON: _TypePolicy(
        "most recently seen",
        lambda s: _sort_desc_by_last_seen(s.people()),
    ),
    # "Which orgs matter" — most connected by known people.
    EntityType.COMPANY: _TypePolicy(
        "most connected",
        lambda s: sorted(
            s.companies(), key=lambda c: (-len(c.links.get("people", [])), c.entity_id)
        ),
    ),
    # Ranked by last_seen like every other type, NOT by start time. Ordering by
    # `starts_at` would make a meeting outrank live organizational pressure
    # purely for being on the calendar; `fields` still carries starts_at/ends_at/
    # attendees/status/conference_id, so downstream reasoning decides what the
    # timing means. Selection here is not a judgement about importance.
    EntityType.MEETING: _TypePolicy(
        "most recently updated",
        lambda s: _sort_desc_by_last_seen(s.meetings()),
    ),
    # Title/owners/status only — never contents. Context carries pointers, not
    # document bodies; a body belongs to retrieval, not to a state projection.
    EntityType.DOCUMENT: _TypePolicy(
        "most recently updated",
        lambda s: _sort_desc_by_last_seen(s.documents()),
    ),
    # Decisions recorded AND decisions owed share this type (discriminated by
    # payload `kind`), so both compete here on the same footing.
    EntityType.DECISION: _TypePolicy(
        "open, oldest activity",
        lambda s: _sort_asc_by_last_seen(_open_only(s.decisions())),
    ),
    EntityType.PROJECT: _TypePolicy(
        "most recently active",
        lambda s: _sort_desc_by_last_seen(s.projects()),
    ),
    EntityType.CUSTOMER: _TypePolicy(
        "most recently active",
        lambda s: _sort_desc_by_last_seen(s.customers()),
    ),
    EntityType.CHANNEL: _TypePolicy(
        "most recently active",
        lambda s: _sort_desc_by_last_seen(s.channels()),
    ),
}

# Tie-break order WITHIN one round-robin tier (see `_scope_morning_brief`).
# The first five preserve the ordering this intent has always had; the rest
# follow. This decides nothing but which of two equally-ranked entities is
# listed first — no type can starve another, because selection is round-robin.
_TIER_ORDER: tuple[EntityType, ...] = (
    EntityType.COMMITMENT,
    EntityType.RELATIONSHIP,
    EntityType.THREAD,
    EntityType.PERSON,
    EntityType.COMPANY,
    EntityType.MEETING,
    EntityType.DOCUMENT,
    EntityType.DECISION,
    EntityType.PROJECT,
    EntityType.CUSTOMER,
    EntityType.CHANNEL,
)


def _scope_morning_brief(
    state: OrganizationalState,
    policy: dict | None,
    *,
    as_of: datetime | None,
) -> ScopedView:
    """Top-N per entity type shaped for the daily brief, across EVERY entity
    type State can hold (see `_MORNING_BRIEF`).

    EMISSION ORDER IS ROUND-ROBIN BY RANK, not grouped by type: the best
    commitment, then the best relationship, ... then the SECOND commitment, and
    so on. Consumers downstream truncate this list against a token budget
    (`api/chat_router.py`), so a type-grouped order would mean whichever type
    was concatenated last got dropped first — a meeting or a document could
    never survive a full inbox. Interleaving makes the types compete on rank
    instead, and guarantees each contributes its single most relevant entity
    before any type contributes a second.

    Note: this is pure entity SELECTION and RANKING. Any judgement about
    "slipping" / "cooling" (with thresholds, severities, why-strings) lives
    in Reasoning V2 — this module does not reproduce it, only surfaces the
    entities Reasoning is likely to talk about so the UI can drill through.
    """
    cap = _cap(policy)
    entities: list[EntityRef] = []
    citations: dict[str, Citation] = {}

    def emit(view: _BaseView, entity_type: str, reason: str) -> None:
        entities.append(_ref(entity_type, view, rank=len(entities), reason=reason))
        # Lineage rides along for EVERY type, from the entity's own provenance —
        # never re-derived, never connector ids. A type with no provenance keeps
        # an honestly empty citation rather than a fabricated one.
        citations[view.entity_id] = _cite(view)

    # Each type independently selects its own top-N by its own rule ...
    per_type = [
        [
            (et.value, view, _MORNING_BRIEF[et].reason)
            for view in _MORNING_BRIEF[et].select(state)[:cap]
        ]
        for et in _TIER_ORDER
    ]
    # ... then they interleave, tier by tier. `zip_longest` pads exhausted types
    # with None, so a type with fewer entities (or none at all) simply stops
    # contributing instead of shifting anyone else's position.
    for tier in itertools.zip_longest(*per_type):
        for item in tier:
            if item is not None:
                entity_type, view, reason = item
                emit(view, entity_type, reason)

    policy_version = None
    if policy:
        raw_v = policy.get("policy_version") or policy.get("version")
        if isinstance(raw_v, int):
            policy_version = raw_v

    return ScopedView(
        intent_name="morning_brief",
        as_of=as_of,
        state_watermark=str(state.watermark.last_memory_id) if state.watermark.last_memory_id else None,
        policy_version=policy_version,
        counts=state.counts(),
        entities=entities,
        citations=citations,
    )
