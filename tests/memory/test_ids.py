"""Deterministic ids: replay-stable, dedup-correct, tenant-isolated."""
from __future__ import annotations

import uuid

from navis.memory.ids import entity_id, memory_record_id

ORG_A = uuid.UUID("11111111-1111-1111-1111-111111111111")
ORG_B = uuid.UUID("99999999-9999-9999-9999-999999999999")
SIG = uuid.UUID("44444444-4444-4444-4444-444444444444")


def test_memory_id_is_deterministic():
    a = memory_record_id(ORG_A, SIG, "email_person", "person", "alice@acme.com")
    b = memory_record_id(ORG_A, SIG, "email_person", "person", "alice@acme.com")
    assert a == b  # replay reproduces the same id


def test_memory_id_varies_by_dedup_key():
    a = memory_record_id(ORG_A, SIG, "email_person", "person", "alice@acme.com")
    b = memory_record_id(ORG_A, SIG, "email_person", "person", "bob@vc.com")
    assert a != b


def test_memory_id_varies_by_signal():
    other_sig = uuid.UUID("55555555-5555-5555-5555-555555555555")
    a = memory_record_id(ORG_A, SIG, "email_person", "person", "alice@acme.com")
    b = memory_record_id(ORG_A, other_sig, "email_person", "person", "alice@acme.com")
    assert a != b  # different source signal = different fact


def test_ids_are_tenant_isolated():
    a = memory_record_id(ORG_A, SIG, "email_person", "person", "alice@acme.com")
    b = memory_record_id(ORG_B, SIG, "email_person", "person", "alice@acme.com")
    assert a != b  # same email, different org → different id

    ea = entity_id(ORG_A, "person", "alice@acme.com")
    eb = entity_id(ORG_B, "person", "alice@acme.com")
    assert ea != eb


def test_entity_id_groups_facts_about_one_entity():
    # Same person across two signals shares one entity_id (their history).
    e1 = entity_id(ORG_A, "person", "Alice@Acme.com")
    e2 = entity_id(ORG_A, "person", "alice@acme.com ")  # case + whitespace normalized
    assert e1 == e2
