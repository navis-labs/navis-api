"""Deterministic UUIDv5 keys for identity aliases and link events.

Identity Resolution uses a namespace disjoint from Memory's (`MEMORY_NAMESPACE`) so
alias ids can never collide with memory_ids or entity_ids in the same org. Replay is
byte-identical: same (org, provider, kind, value) → same alias_id, forever.
"""
from __future__ import annotations

import uuid

# FIXED namespace — must never change. Rotating it re-keys every alias and forks the
# identity graph away from all existing link events. If a breaking evolution is needed,
# introduce a new alias table alongside this one; never mutate this constant.
IDENTITY_NAMESPACE = uuid.UUID("2c9f6e05-4b6a-5a12-9d1c-31f8e07b6a4c")


def alias_id(org_id: uuid.UUID, provider: str, kind: str, value: str) -> uuid.UUID:
    """Deterministic PK for one (provider, kind, value) alias within an org.

    `value` MUST be pre-normalized by `resolver.normalize_alias_value` — mixing raw
    and normalized values would fork the alias into two rows for the same person."""
    key = f"{org_id}|{provider}|{kind}|{value}"
    return uuid.uuid5(IDENTITY_NAMESPACE, key)


def link_event_id(
    org_id: uuid.UUID,
    source_signal_event_id: uuid.UUID,
    alias_a: uuid.UUID,
    alias_b: uuid.UUID,
) -> uuid.UUID:
    """Deterministic PK for one link edge.

    Aliases are sorted so (A, B) and (B, A) collapse to one row — a link is symmetric,
    and the source signal is the same. Same source + same pair → same row, always."""
    lo, hi = sorted([str(alias_a), str(alias_b)])
    key = f"{org_id}|{source_signal_event_id}|{lo}|{hi}"
    return uuid.uuid5(IDENTITY_NAMESPACE, key)
