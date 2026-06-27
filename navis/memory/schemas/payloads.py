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
    EntityType.CHANNEL: 1,
    EntityType.DOCUMENT: 1,
}

# Human-readable payload contracts. Fields beyond the original v1 set are ADDITIVE optional
# keys (backward compatible — old records simply lack them; State reads via .get()), so they
# do NOT bump the schema version. Breaking shape changes still bump + register an upcaster.
PAYLOAD_FIELDS_V1: dict[EntityType, tuple[str, ...]] = {
    EntityType.PERSON: ("email", "name", "role", "source", "slack_user_id"),
    EntityType.COMPANY: ("domain", "name"),
    EntityType.CUSTOMER: ("external_ref", "name"),  # populated from a CRM signal source
    EntityType.PROJECT: ("folder_id", "name", "owners", "shared", "mime_type", "drive_id"),
    EntityType.MEETING: ("title", "starts_at", "ends_at", "attendees", "status",
                         "has_conference", "organizer", "is_recurring", "conference_id"),
    EntityType.THREAD: ("thread_id", "subject", "message_id",
                        "rfc_message_id", "in_reply_to", "references", "channel_id"),
    EntityType.RELATIONSHIP: ("person_email", "direction", "via", "slack_user_id", "channel_id"),
    EntityType.COMMITMENT: ("text", "speaker_email", "speaker_slack_user_id", "subject",
                            "thread_id", "channel_id", "via"),
    EntityType.DECISION: ("text", "subject", "channel_id", "via"),
    EntityType.CHANNEL: ("channel_id", "name", "visibility", "workspace_id"),
    EntityType.DOCUMENT: ("document_id", "title", "owners", "parents", "drive_id", "source"),
}


def current_version(entity_type: EntityType) -> int:
    return CURRENT_SCHEMA_VERSION[entity_type]
