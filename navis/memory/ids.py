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
    """Deterministic PK for one memory record."""
    key = f"{org_id}|{signal_event_id}|{extractor_name}|{entity_type}|{dedup_key}"
    return uuid.uuid5(MEMORY_NAMESPACE, key)


def entity_id(org_id: uuid.UUID, entity_type: str, natural_key: str) -> uuid.UUID:
    """Deterministic identity of a real-world entity within an org."""
    key = f"{org_id}|{entity_type}|{natural_key.strip().lower()}"
    return uuid.uuid5(MEMORY_NAMESPACE, key)
