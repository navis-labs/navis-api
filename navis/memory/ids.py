"""Deterministic identity for Memory V2.

Two UUIDv5s, both pure functions of their inputs — replay always reproduces them:

  * memory_id  — identity of THIS fact/record. Encodes the dedup tuple, so the
                 same signal + extractor + dedup_key always yields the same row.
  * entity_id  — stable identity of the real-world ENTITY the fact is about
                 (e.g. a person by email). Many records share one entity_id over
                 time, forming that entity's append-only history.
"""
from __future__ import annotations

import uuid

# Fixed namespace — must NEVER change, or all ids change and replay diverges.
MEMORY_NAMESPACE = uuid.UUID("b1e6d2a4-7c83-5f19-9a0e-2d4c6b8f1037")


def memory_record_id(
    org_id: uuid.UUID,
    signal_event_id: uuid.UUID,
    extractor_name: str,
    entity_type: str,
    dedup_key: str,
) -> uuid.UUID:
    """Deterministic PK for one memory record.

    EVOLUTION LAW (so replay never produces stale or duplicate histories):
      * `extractor_version` is deliberately NOT in this key. A logic change that keeps the
        same (name, entity_type, dedup_key) yields the SAME memory_id -> ON CONFLICT DO
        NOTHING -> existing rows are preserved (not overwritten, not duplicated).
      * A BREAKING change that SHOULD produce new facts must bump the `extractor_name`
        (e.g. "email_commitment" -> "email_commitment_v2"): the new name is in this key, so
        it yields new rows while old history is preserved, and a clean replay converges.
        NEVER add extractor_version to this key — that would re-key all 10B rows and
        duplicate every history.
      * `dedup_key` must be CONTENT-derived for text extractors (a stable hash of the fact),
        not positional (a sentence index), which shifts when text processing changes and
        creates duplicate histories.
    """
    key = f"{org_id}|{signal_event_id}|{extractor_name}|{entity_type}|{dedup_key}"
    return uuid.uuid5(MEMORY_NAMESPACE, key)


def normalize_natural_key(entity_type: str, natural_key: str) -> str:
    """Canonicalize a natural key before it becomes an entity_id. REPLAY-DETERMINISTIC.

    Emails and domains are case-INSENSITIVE → lowercased. Opaque provider ids (Slack U…/C…,
    Google Drive file ids, Calendar event ids) are case-SENSITIVE and are preserved VERBATIM:
    `"1AbC" != "1abc"` are two different Drive files, so a blanket `.lower()` could merge two
    distinct entities into one history — a Sev-0 identity collision at scale. The `@` test
    detects an email without a provider lookup; opaque ids never contain `@`.

    Conservative by design: this NEVER aggressively merges (e.g. no Gmail dot/plus folding —
    that risks merging distinct people at providers where dots are significant). Cross-identity
    resolution (slack_user_id ↔ email) is a downstream concern (State/edges), never baked into
    immutable identity, because a wrong merge is unrecoverable corruption while fragmentation
    is recoverable.
    """
    k = natural_key.strip()
    if "@" in k or entity_type == "company":  # email or domain -> case-insensitive
        return k.lower()
    return k


def entity_id(org_id: uuid.UUID, entity_type: str, natural_key: str) -> uuid.UUID:
    """Deterministic identity of a real-world entity within an org."""
    key = f"{org_id}|{entity_type}|{normalize_natural_key(entity_type, natural_key)}"
    return uuid.uuid5(MEMORY_NAMESPACE, key)
