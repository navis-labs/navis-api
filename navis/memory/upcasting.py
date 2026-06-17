"""Schema upcasting — let v1/v2/v3 records coexist forever.

A stored record keeps the `schema_version` it was written with (never rewritten —
that would violate append-only and break replay). On READ, consumers (State) call
`upcast` to bring a payload up to the current shape by applying a chain of
registered upcasters. Adding a v2:

    register_upcaster(EntityType.PERSON, 1, lambda p: {**p, "display_name": p.get("name")})
    CURRENT_SCHEMA_VERSION[EntityType.PERSON] = 2

Old v1 rows then upcast to v2 on read; nothing is migrated in place.
"""
from __future__ import annotations

from typing import Callable

from navis.memory.envelope import EntityType
from navis.memory.schemas import current_version

Upcaster = Callable[[dict], dict]

# (entity_type, from_version) -> function producing the (from_version+1) payload.
_UPCASTERS: dict[tuple[EntityType, int], Upcaster] = {}


def register_upcaster(entity_type: EntityType, from_version: int, fn: Upcaster) -> None:
    _UPCASTERS[(entity_type, from_version)] = fn


def upcast(entity_type: EntityType, payload: dict, version: int) -> tuple[dict, int]:
    """Apply the upcaster chain until the current version (or until a gap)."""
    target = current_version(entity_type)
    current = dict(payload)
    while version < target:
        fn = _UPCASTERS.get((entity_type, version))
        if fn is None:
            break  # no path forward; return what we have (caller sees version < target)
        current = fn(current)
        version += 1
    return current, version
