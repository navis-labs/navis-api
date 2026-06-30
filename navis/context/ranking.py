"""Deterministic STRUCTURAL ranking — order, never judgement.

Ranks scope entries so the budgeter admits the most structurally-central entities first.
Signals are purely structural and replay-stable: entity-type priority, recency
(as_of - last_seen, the State clock — never now()), and link degree. There is NO risk,
priority, health, or score here — those are Reasoning's exclusively. Ties break on
entity_id for a total, reproducible order.
"""
from __future__ import annotations

from datetime import datetime

from navis.context.models import ContextScope, ScopeEntry
from navis.memory.envelope import EntityType
from navis.state.v2.models import OrganizationalState

# Lower number = ranked earlier. Structural centrality only (people/orgs anchor a graph).
_TYPE_PRIORITY: dict[str, int] = {
    EntityType.PERSON.value: 0,
    EntityType.COMPANY.value: 1,
    EntityType.RELATIONSHIP.value: 2,
    EntityType.PROJECT.value: 3,
    EntityType.MEETING.value: 4,
    EntityType.COMMITMENT.value: 5,
    EntityType.DECISION.value: 6,
    EntityType.THREAD.value: 7,
    EntityType.CHANNEL.value: 8,
    EntityType.DOCUMENT.value: 9,
    EntityType.CUSTOMER.value: 10,
}
_UNKNOWN_TYPE = 99
_NO_RECENCY = float("inf")  # entities with no last_seen rank after dated ones


def _recency_days(as_of: datetime | None, last_seen_iso: str | None) -> float:
    if as_of is None or not last_seen_iso:
        return _NO_RECENCY
    try:
        last = datetime.fromisoformat(last_seen_iso)
    except (TypeError, ValueError):
        return _NO_RECENCY
    return (as_of - last).total_seconds() / 86400.0


class StructuralRanker:
    """The default Ranker: pure function of (state, scope)."""

    def rank(self, state: OrganizationalState, scope: ContextScope) -> list[ScopeEntry]:
        as_of = state.as_of

        def key(entry: ScopeEntry) -> tuple:
            raw = state.entities.get(entry.entity_type, {}).get(entry.entity_id, {})
            link_degree = sum(len(v) for v in (raw.get("links") or {}).values())
            return (
                _TYPE_PRIORITY.get(entry.entity_type, _UNKNOWN_TYPE),  # type first
                _recency_days(as_of, raw.get("last_seen")),           # fresher first
                -link_degree,                                          # more connected first
                entry.entity_id,                                      # total-order tiebreak
            )

        return sorted(scope.entries, key=key)
