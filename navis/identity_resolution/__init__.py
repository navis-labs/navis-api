"""Identity Resolution — provider-agnostic organizational identity graph.

Sits ABOVE Memory. Memory stores immutable, per-provider facts (Gmail person keyed by
email; Slack person keyed by slack_user_id; Calendar attendee keyed by email; and so on
for GitHub, Notion, HubSpot). Identity Resolution folds those provider identities into
a canonical organizational identity graph — WITHOUT modifying MemoryRecord or the
Memory schema.

Two append-only tables carry the entire ontology:

  * identity_aliases      — one row per (provider, kind, value) alias per org
  * identity_link_events  — append-only edges: alias_a and alias_b refer to the same person

Both use deterministic UUIDv5 primary keys, so replay converges to identical rows.
Canonical resolution is a BFS over link events computed at query time — no destructive
merge, no mutable equivalence class, so a wrong link is a recoverable append (never
data corruption).

Future Knowledge Graph projections read these two tables verbatim: aliases are the
nodes, link events are the edges. No schema migration required to introduce a KG
projector on top.
"""
from navis.identity_resolution.resolver import (
    Alias,
    AliasKind,
    AliasProvider,
    canonical_alias_ids,
    normalize_alias_value,
)

__all__ = [
    "Alias",
    "AliasKind",
    "AliasProvider",
    "canonical_alias_ids",
    "normalize_alias_value",
]
