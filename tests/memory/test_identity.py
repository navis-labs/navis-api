"""Identity laws: replay stability, entity stability, extractor evolution, natural-key
normalization (opaque ids stay case-sensitive — Sev-0 guard), content-hash dedup stability."""
from __future__ import annotations

import uuid

from navis.memory.ids import entity_id, memory_record_id, normalize_natural_key
from navis.memory.extractors import run_extractors
from navis.memory.envelope import EntityType
from tests.memory.fakes import gmail_view, slack_message_view

ORG = uuid.UUID("11111111-1111-1111-1111-111111111111")
SIG = uuid.UUID("44444444-4444-4444-4444-444444444444")


# 1. Replay stability — same inputs → same memory_id.
def test_replay_memory_id_stable():
    a = memory_record_id(ORG, SIG, "email_person", "person", "alice@x.com")
    b = memory_record_id(ORG, SIG, "email_person", "person", "alice@x.com")
    assert a == b


# 2. Entity stability — same natural key → same entity_id.
def test_entity_id_stable():
    assert entity_id(ORG, "person", "alice@x.com") == entity_id(ORG, "person", "alice@x.com")


# 3/4. Extractor evolution — version is NOT in the id; only the NAME is.
def test_version_change_same_id_but_name_change_new_id():
    base = memory_record_id(ORG, SIG, "email_commitment", "commitment", "k")
    # a version bump is invisible to the id (id excludes version) -> same id
    assert memory_record_id(ORG, SIG, "email_commitment", "commitment", "k") == base
    # a NAME bump (breaking change) -> new id, old history preserved
    assert memory_record_id(ORG, SIG, "email_commitment_v2", "commitment", "k") != base


# 5. Natural-key normalization.
def test_email_case_and_whitespace_normalize_to_same_entity():
    assert entity_id(ORG, "person", " Alice@X.COM ") == entity_id(ORG, "person", "alice@x.com")


def test_opaque_ids_are_case_sensitive_sev0_guard():
    # Slack/Drive/Doc ids are case-sensitive: "C123" and "c123" are DIFFERENT entities.
    # A blanket .lower() would merge them — the Sev-0 we are guarding against.
    assert entity_id(ORG, "channel", "C123") != entity_id(ORG, "channel", "c123")
    assert entity_id(ORG, "document", "1AbC") != entity_id(ORG, "document", "1abc")
    # ...but case-insensitive emails/domains still fold.
    assert entity_id(ORG, "company", "Acme.COM") == entity_id(ORG, "company", "acme.com")
    assert normalize_natural_key("channel", "C123") == "C123"      # preserved
    assert normalize_natural_key("person", "A@X.com") == "a@x.com"  # email lowered


# 6. Dedup-key stability — content hash is invariant to sentence re-ordering / index shift.
def test_commitment_dedup_is_content_stable_not_positional():
    base = gmail_view(payload={"body_text": "I'll send the deck by Friday. We have decided to ship."})
    # Same commitment, different surrounding order -> the COMMITMENT memory_id is identical.
    reordered = gmail_view(payload={"body_text": "We have decided to ship. I'll send the deck by Friday."})
    def commit_ids(view):
        from navis.memory.service import build_memory_records
        return {r.memory_id for r in build_memory_records(view) if r.entity_type == EntityType.COMMITMENT}
    assert commit_ids(base) == commit_ids(reordered)


def test_slack_commitment_content_keyed():
    drafts = run_extractors(slack_message_view())
    commits = [d for d in drafts if d.entity_type == EntityType.COMMITMENT]
    assert commits and len(commits[0].dedup_key) == 16  # content hash, not an index
