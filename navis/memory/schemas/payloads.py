"""Per-entity payload schema versions.

Each entity_type's `payload` has a `schema_version`. When a payload shape changes,
bump the version here and register an upcaster (see `memory.upcasting`) — old
records keep their stored version and are upcast on read, so v1/v2/v3 coexist and
replay never breaks.

The payload field documentation below is the contract State will read against.
"""
from __future__ import annotations

from navis.memory.envelope import EntityType

# The version a NEWLY extracted record of each type is written with.
CURRENT_SCHEMA_VERSION: dict[EntityType, int] = {
    EntityType.PERSON: 1,
    EntityType.COMPANY: 1,
    EntityType.CUSTOMER: 1,
    EntityType.PROJECT: 1,
    EntityType.MEETING: 1,
    EntityType.THREAD: 1,
    EntityType.RELATIONSHIP: 1,
    EntityType.COMMITMENT: 1,
    EntityType.DECISION: 1,
}

# Human-readable payload contracts (v1). Kept here so State has one place to look.
PAYLOAD_FIELDS_V1: dict[EntityType, tuple[str, ...]] = {
    EntityType.PERSON: ("email", "name", "role"),
    EntityType.COMPANY: ("domain", "name"),
    EntityType.CUSTOMER: ("external_ref", "name"),  # populated from a CRM signal source
    EntityType.PROJECT: ("folder_id", "name"),
    EntityType.MEETING: ("title", "starts_at", "ends_at", "attendees", "status", "has_conference"),
    EntityType.THREAD: ("thread_id", "subject", "message_id"),
    EntityType.RELATIONSHIP: ("person_email", "direction", "via"),
    EntityType.COMMITMENT: ("text", "speaker_email", "subject", "thread_id"),
    EntityType.DECISION: ("text", "subject"),
}


def current_version(entity_type: EntityType) -> int:
    return CURRENT_SCHEMA_VERSION[entity_type]
