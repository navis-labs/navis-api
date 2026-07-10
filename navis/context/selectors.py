"""Deterministic entity selection — task spec over State, no interpretation.

Picks the entity types a task needs, then (optionally) pulls in their ONE-HOP neighbours by
traversing State's EXISTING link sets (folds.py). It never builds a graph, never recurses,
never scores, never reads Memory. Pure and stateless ⇒ reentrant and parallel-safe.

Scalability contract (must hold at 100M+ entities):
  * A SCOPED task is O(selected entities + their link fan-out) — it never scans or indexes
    the whole org. Link targets are resolved by O(1) bucket membership over the few
    non-selected types, not via a full reverse index.
  * A FULL_ORG task (entity_types is None) selects every entity directly, so link expansion
    can add NOTHING and is skipped entirely (no wasted index/scan). FULL_ORG is inherently
    O(org) — that is the whole-org semantics, bounded by State itself.

Determinism: the selector does no sorting of its own — ContextScope canonicalizes
(sorts + dedups) entity_types and entries on construction, so the output is identical for
the same (State, TaskSpec) regardless of dict/set iteration order. No clock, no uuid, no
randomness, no mutable state.
"""
from __future__ import annotations

from navis.context.models import ContextScope, ScopeEntry, SelectionReason
from navis.context.tasks import TaskSpec
from navis.state.v2.models import OrganizationalState


class DeterministicSelector:
    """The default Selector: pure function of (state, spec)."""

    def select(self, state: OrganizationalState, spec: TaskSpec) -> ContextScope:
        entities = state.entities

        # 1) Direct selection: every entity of a selected type. (None = every type present.)
        if spec.entity_types is None:
            types: tuple[str, ...] = tuple(entities.keys())
        else:
            types = tuple(t for t in spec.entity_types if t in entities)

        chosen: dict[str, ScopeEntry] = {}
        for et in types:
            for eid in entities.get(et, {}):
                chosen[eid] = ScopeEntry(eid, et, SelectionReason.TASK_RELEVANT.value)

        # 2) One-hop, NON-RECURSIVE link expansion. Skipped when all types are selected
        #    (entity_types is None) — every target is already chosen, so there is nothing to
        #    add and no reason to resolve types. For a scoped task we resolve ONLY the
        #    unchosen targets (bounded by the selection's link fan-out) against the few
        #    NON-selected buckets via O(1) membership — never a full-org reverse index. A
        #    neighbour's own links are NOT followed (one hop), so the scope cannot blow up.
        if spec.expand_links and spec.entity_types is not None:
            selected = set(types)
            pending: set[str] = set()
            for et in types:
                for raw in entities.get(et, {}).values():
                    for targets in (raw.get("links") or {}).values():
                        for tid in targets or ():
                            # Malformed/dangling-safe: ignore non-string/empty ids and any id
                            # already selected (so we never duplicate an entity).
                            if isinstance(tid, str) and tid and tid not in chosen:
                                pending.add(tid)
            for tid in pending:
                for et, bucket in entities.items():
                    if et in selected:
                        continue  # selected-type ids are already chosen → not pending
                    if tid in bucket:
                        chosen[tid] = ScopeEntry(tid, et, SelectionReason.LINK_EXPANSION.value)
                        break
                # not found in any bucket → dangling link, safely skipped (never fabricated)

        # ContextScope canonicalizes (sorts + dedup-checks) — no redundant sort here.
        return ContextScope(entity_types=types, entries=tuple(chosen.values()))
