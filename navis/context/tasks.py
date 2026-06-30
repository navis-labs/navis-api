"""ContextTask vocabulary + declarative selection specs.

A task says WHAT a reasoning job needs, never HOW reasoning works — there is NO execution
logic in this layer, only data. Specs are a declarative table (not an if/elif chain) so a
new task plugs in by adding one entry, without touching the selector. Entity types reference
the shared Memory/State vocabulary (EntityType) — a value enum, not a data dependency.

The registry is validated at import (fail-fast): every ContextTask MUST have exactly one
TaskSpec, and every TaskSpec must reference only known EntityType values. An orphan task or a
typo'd entity type breaks app startup, not a single request at 2am.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass

from navis.memory.envelope import EntityType

# The authoritative entity-type vocabulary. Any TaskSpec type outside this set is a typo or
# a stale reference and is rejected at construction.
_VALID_ENTITY_TYPES = frozenset(e.value for e in EntityType)


class ContextTask(str, enum.Enum):
    """The reasoning task a bundle is assembled for. Pure vocabulary — no behaviour."""

    FULL_ORG = "full_org"            # whole-org scope (parity with pre-Context reasoning)
    RELATIONSHIPS = "relationships"  # people/relationships/companies/meetings
    COMMITMENTS = "commitments"      # commitments + their owners/relationships


@dataclass(frozen=True, slots=True)
class TaskSpec:
    """Declarative selection rule for a task — data only, no logic.

    entity_types:
        None  -> every entity type present in State.
        frozenset[str] -> exactly these types (must be a NON-EMPTY set of known EntityType
        values; use None for "all", never an empty set). Order-independent: selection sorts
        for a deterministic result.
    expand_links:
        True  -> after selecting the typed entities, pull in their ONE-HOP neighbours via
                 State's existing link sets (folds.py). This is a SINGLE hop and is
                 explicitly NON-RECURSIVE / non-transitive: a neighbour's own links are NOT
                 followed, so the scope can never blow up or loop. There is no depth knob.
        False -> no expansion; only the directly-selected types.
    """

    entity_types: frozenset[str] | None
    expand_links: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.expand_links, bool):
            raise ValueError("TaskSpec.expand_links must be a bool")
        if self.entity_types is None:
            return
        types = frozenset(self.entity_types)  # coerce/dedup any iterable input
        if not types:
            raise ValueError("TaskSpec.entity_types must be None (all types) or a NON-EMPTY set")
        unknown = sorted(t for t in types if t not in _VALID_ENTITY_TYPES)
        if unknown:
            raise ValueError(f"TaskSpec.entity_types references unknown EntityType(s): {unknown}")
        object.__setattr__(self, "entity_types", types)


_REL = frozenset({
    EntityType.PERSON.value, EntityType.RELATIONSHIP.value,
    EntityType.COMPANY.value, EntityType.MEETING.value,
})
_COMMIT = frozenset({
    EntityType.COMMITMENT.value, EntityType.PERSON.value, EntityType.RELATIONSHIP.value,
})

# The single source of truth: ContextTask -> TaskSpec. Dict keys are unique by construction
# (no duplicate tasks possible) and typed as ContextTask (no extraneous keys).
TASK_SPECS: dict[ContextTask, TaskSpec] = {
    ContextTask.FULL_ORG: TaskSpec(entity_types=None, expand_links=True),
    ContextTask.RELATIONSHIPS: TaskSpec(entity_types=_REL, expand_links=True),
    ContextTask.COMMITMENTS: TaskSpec(entity_types=_COMMIT, expand_links=True),
}


def _assert_complete_registry() -> None:
    """Fail-fast invariant (runs at import): every ContextTask has exactly one TaskSpec.
    An orphan task breaks startup instead of a single request later."""
    missing = set(ContextTask) - set(TASK_SPECS)
    if missing:
        raise RuntimeError(
            f"ContextTask(s) without a TaskSpec (orphan task): {sorted(t.value for t in missing)}"
        )


_assert_complete_registry()
