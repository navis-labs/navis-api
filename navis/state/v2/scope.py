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

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Union

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


@dataclass(frozen=True, slots=True)
class EntityRef:
    """One entity in scope, with its rank and why it made the cut.

    ``fields`` is the entity's current-state field dict (from State V2's raw
    fold — same shape typed views expose). Consumers pick what they need.
    """
    entity_type: str
    entity_id: str
    rank: int
    reason: str
    fields: dict[str, Any]
    first_seen: str | None
    last_seen: str | None


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


def _ref(entity_type: str, view: _BaseView, rank: int, reason: str) -> EntityRef:
    return EntityRef(
        entity_type=entity_type,
        entity_id=view.entity_id,
        rank=rank,
        reason=reason,
        fields=dict(view.fields or {}),
        first_seen=view.first_seen,
        last_seen=view.last_seen,
    )


def _sort_desc_by_last_seen(views: list[_BaseView]) -> list[_BaseView]:
    # last_seen is ISO-8601 (see builders.py); lexical sort == chronological.
    # entity_id tiebreaker so two entities with identical last_seen produce a
    # stable order across runs.
    return sorted(views, key=lambda v: (v.last_seen or "", v.entity_id), reverse=True)


def _sort_asc_by_last_seen(views: list[_BaseView]) -> list[_BaseView]:
    return sorted(views, key=lambda v: (v.last_seen or "￿", v.entity_id))


def _scope_morning_brief(
    state: OrganizationalState,
    policy: dict | None,
    *,
    as_of: datetime | None,
) -> ScopedView:
    """Top-N per entity type shaped for the daily brief.

    Ranking:
      * commitments (open, oldest first)  → "what's slipping"
      * relationships (stalest first)     → "who's going cold"
      * threads (freshest first)          → "what's active"
      * people (most recently active)     → "who's in the mix"
      * companies (most connected)        → "which orgs matter"

    Note: this is pure entity SELECTION and RANKING. Any judgement about
    "slipping" / "cooling" (with thresholds, severities, why-strings) lives
    in Reasoning V2 — this module does not reproduce it, only surfaces the
    entities Reasoning is likely to talk about so the UI can drill through.
    """
    cap = _cap(policy)
    entities: list[EntityRef] = []
    citations: dict[str, Citation] = {}

    def emit(bucket: list[EntityRef], view: _BaseView, entity_type: str, reason: str) -> None:
        bucket.append(_ref(entity_type, view, rank=len(bucket), reason=reason))
        citations[view.entity_id] = _cite(view)

    # Commitments — open, oldest last_seen first (most likely to have slipped).
    open_commits = [c for c in state.commitments() if c.fields.get("status", "open") == "open"]
    for c in _sort_asc_by_last_seen(open_commits)[:cap]:
        emit(entities, c, "commitment", reason="open, oldest activity")

    # Relationships — stalest first (highest chance of "cooling").
    for r in _sort_asc_by_last_seen(state.relationships())[:cap]:
        emit(entities, r, "relationship", reason="stalest interaction")

    # Threads — freshest first (what's active in the inbox).
    for t in _sort_desc_by_last_seen(state.threads())[:cap]:
        emit(entities, t, "thread", reason="most recently active")

    # People — most recently active (who's in the mix).
    for p in _sort_desc_by_last_seen(state.people())[:cap]:
        emit(entities, p, "person", reason="most recently seen")

    # Companies — ranked by connected people count (most represented in org).
    companies_by_size = sorted(
        state.companies(),
        key=lambda c: (-len(c.links.get("people", [])), c.entity_id),
    )
    for co in companies_by_size[:cap]:
        emit(entities, co, "company", reason="most connected")

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
