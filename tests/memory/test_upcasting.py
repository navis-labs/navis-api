"""Schema evolution: v1 records upcast to the current version on read."""
from __future__ import annotations

from navis.memory.envelope import EntityType
from navis.memory.schemas import payloads
from navis.memory.upcasting import register_upcaster, upcast


def test_upcast_applies_registered_chain():
    original = payloads.CURRENT_SCHEMA_VERSION[EntityType.PERSON]
    try:
        # Simulate shipping a v2 person schema with a new field.
        payloads.CURRENT_SCHEMA_VERSION[EntityType.PERSON] = 2
        register_upcaster(
            EntityType.PERSON, 1, lambda p: {**p, "display_name": p.get("name")}
        )

        v1_payload = {"email": "a@acme.com", "name": "Alice", "role": "from"}
        upgraded, version = upcast(EntityType.PERSON, v1_payload, 1)

        assert version == 2
        assert upgraded["display_name"] == "Alice"
        assert upgraded["email"] == "a@acme.com"  # original fields preserved
    finally:
        payloads.CURRENT_SCHEMA_VERSION[EntityType.PERSON] = original


def test_current_version_records_need_no_upcast():
    payload = {"email": "a@acme.com"}
    out, version = upcast(EntityType.PERSON, payload, 1)
    assert version == 1 and out == payload  # already current → unchanged
