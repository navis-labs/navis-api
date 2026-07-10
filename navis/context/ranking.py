"""Deterministic STRUCTURAL ranking — order, never judgement.

Ranks scope entries so the budgeter admits the most structurally-central entities first
under a tight budget. Signals are purely structural and replay-stable: entity-type
centrality, recency (as_of - last_seen, the State clock — never now()), and link degree
(from State's existing link sets). Ties break on entity_id for a total, reproducible order.

ARCHITECTURAL CONTRACT (enforced by review, not the compiler — DO NOT VIOLATE):
  Ranking exists ONLY to produce a deterministic structural ORDERING of a ContextScope. It
  is NOT an intelligence layer. It may use ONLY structural signals. Future developers MUST
  NOT introduce — here — any of:
        risk · urgency · priority · health · importance · business value · recommendations
        LLM output · ML output · embeddings · vector similarity · sentiment · prediction
  That logic belongs EXCLUSIVELY to Reasoning. `_TYPE_PRIORITY` is a TYPE-centrality preference
  (people/orgs anchor a graph), NOT a judgement of entity importance. If a deployment needs a
  different ORDERING it injects a deterministic table (StructuralRanker(type_priority=...)) or
  a whole custom Ranker — it never adds interpretation to this file.

Fail-safe by construction: ranking NEVER raises. Any entity whose recency cannot be computed
(missing/invalid/naive/overflowing timestamp) sorts LAST among its type; a malformed link
structure contributes degree 0; an unknown entity type sorts after all known types. Pure,
stateless, replay-safe: same (State, Scope) ⇒ same order, forever, on any platform.
"""
from __future__ import annotations

from datetime import datetime

from navis.context.models import ContextScope, ScopeEntry
from navis.memory.envelope import EntityType
from navis.state.v2.models import OrganizationalState

# Structural centrality ordering (lower = ranked earlier). People/orgs anchor the entity
# graph; relations and work artefacts follow. This is the DEFAULT ordering preference — a
# deployment may inject an alternative deterministic table (see StructuralRanker). It is an
# ORDERING preference only, never a judgement of importance/risk/priority.
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
_UNKNOWN_TYPE = 99           # truly-unknown type strings sort after all known types
_NO_RECENCY = float("inf")   # entities with no computable recency sort after dated ones


def _assert_priority_complete(table: dict[str, int]) -> None:
    """Fail-fast (at import): every EntityType must have a centrality entry in the DEFAULT
    table, so adding a new Memory entity type without ranking it is caught at startup, not as
    a silent rank-last in production."""
    missing = {e.value for e in EntityType} - set(table)
    if missing:
        raise RuntimeError(f"_TYPE_PRIORITY is missing entries for: {sorted(missing)}")


_assert_priority_complete(_TYPE_PRIORITY)


def _recency_days(as_of: datetime | None, last_seen_iso: str | None) -> float:
    """Age in days using the projection clock (NEVER now()). FAIL-SAFE: returns +inf (sorts
    last) for any of — no as_of, missing/empty/invalid ISO, a naive-vs-aware mismatch, or an
    arithmetic overflow. The subtraction is INSIDE the guard precisely because a naive
    last_seen minus an aware as_of raises TypeError. A future last_seen (clock skew or a
    scheduled event) yields a negative value and sorts as freshest — structurally correct for
    a forward-dated meeting; we deliberately do not clamp it (clamping would hide a real,
    relevant future-dated entity). Deterministic and platform-independent (IEEE-754)."""
    if as_of is None or not last_seen_iso:
        return _NO_RECENCY
    try:
        last = datetime.fromisoformat(last_seen_iso)
        return (as_of - last).total_seconds() / 86400.0
    except (TypeError, ValueError, OverflowError):
        return _NO_RECENCY


def _link_degree(raw: dict) -> int:
    """Structural connectedness = total size of an entity's link lists. FAIL-SAFE: a
    malformed `links` (not a dict) or a non-sized link value contributes 0 rather than raising
    — ranking must never throw. Recomputed from State's existing link sets (see below)."""
    links = raw.get("links")
    if not isinstance(links, dict):
        return 0
    total = 0
    for value in links.values():
        try:
            total += len(value)
        except TypeError:
            continue  # a non-sized link value → contributes 0
    return total


class StructuralRanker:
    """The default Ranker: pure function of (state, scope). No clock, no randomness, no
    mutable state, no I/O, no platform dependence — same (state, scope) ⇒ same order, forever.

    The type-centrality ordering is injectable as a deterministic provider: pass
    `type_priority` to override the default table without reimplementing rank() (or inject a
    different Ranker entirely at the service seam). An injected table may be partial — types
    absent from it sort last (via _UNKNOWN_TYPE)."""

    def __init__(self, type_priority: dict[str, int] | None = None) -> None:
        if type_priority is not None and not all(
            isinstance(v, int) and not isinstance(v, bool) for v in type_priority.values()
        ):
            raise ValueError("StructuralRanker.type_priority values must be ints")
        self._priority = type_priority if type_priority is not None else _TYPE_PRIORITY

    def rank(self, state: OrganizationalState, scope: ContextScope) -> list[ScopeEntry]:
        as_of = state.as_of
        entities = state.entities
        priority = self._priority

        def key(entry: ScopeEntry) -> tuple[int, float, int, str]:
            raw = entities.get(entry.entity_type, {}).get(entry.entity_id, {})
            return (
                priority.get(entry.entity_type, _UNKNOWN_TYPE),  # type centrality first
                _recency_days(as_of, raw.get("last_seen")),      # fresher first
                -_link_degree(raw),                              # more connected first
                entry.entity_id,                                 # total-order tiebreak
            )

        # Python computes `key` once per element (decorate-sort) — no repeated parse/lookup.
        return sorted(scope.entries, key=key)
