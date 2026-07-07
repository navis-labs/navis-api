"""Pure resolver invariants: deterministic ids, replay-safe extraction, correct
same-record co-observation. No DB — resolver.extract_aliases and the id helpers
are pure functions and must remain so."""
from __future__ import annotations

import uuid

from navis.identity_resolution.ids import alias_id, link_event_id
from navis.identity_resolution.resolver import (
    Alias,
    AliasKind,
    AliasProvider,
    extract_aliases,
    normalize_alias_value,
)
from navis.signals.base import SignalSource

ORG_A = uuid.UUID("11111111-1111-1111-1111-111111111111")
ORG_B = uuid.UUID("99999999-9999-9999-9999-999999999999")
SIG = uuid.UUID("44444444-4444-4444-4444-444444444444")


# --- ids ---------------------------------------------------------------------

def test_alias_id_is_deterministic():
    a = alias_id(ORG_A, "gmail", "email", "alice@acme.com")
    b = alias_id(ORG_A, "gmail", "email", "alice@acme.com")
    assert a == b


def test_alias_id_is_tenant_isolated():
    a = alias_id(ORG_A, "gmail", "email", "alice@acme.com")
    b = alias_id(ORG_B, "gmail", "email", "alice@acme.com")
    assert a != b


def test_alias_id_varies_by_provider():
    """Same (kind, value) under two providers = two aliases. The link log
    is what asserts they refer to the same person, not the id."""
    g = alias_id(ORG_A, "gmail", "email", "alice@acme.com")
    h = alias_id(ORG_A, "hubspot", "email", "alice@acme.com")
    assert g != h


def test_alias_id_varies_by_kind():
    email = alias_id(ORG_A, "slack", "email", "alice@acme.com")
    slack = alias_id(ORG_A, "slack", "slack_user_id", "U123")
    assert email != slack


def test_link_event_id_is_symmetric():
    """(A,B) and (B,A) collapse to one row — a link is symmetric."""
    a = uuid.UUID("00000000-0000-0000-0000-000000000001")
    b = uuid.UUID("00000000-0000-0000-0000-000000000002")
    ab = link_event_id(ORG_A, SIG, a, b)
    ba = link_event_id(ORG_A, SIG, b, a)
    assert ab == ba


# --- normalization -----------------------------------------------------------

def test_email_normalization_is_case_insensitive():
    assert normalize_alias_value(AliasKind.EMAIL, "Alice@Acme.COM") == "alice@acme.com"
    assert normalize_alias_value(AliasKind.EMAIL, "  bob@x.com ") == "bob@x.com"


def test_opaque_ids_are_case_sensitive():
    """Slack U... and GitHub logins are case-significant at the source; case-folding
    them can merge two distinct users."""
    assert normalize_alias_value(AliasKind.SLACK_USER_ID, "U01Ab") == "U01Ab"
    assert normalize_alias_value(AliasKind.GITHUB_LOGIN, "OctoCat") == "OctoCat"


# --- gmail extractor ---------------------------------------------------------

def test_gmail_extractor_collects_from_to_cc():
    aliases, pairs = extract_aliases(
        SignalSource.GMAIL,
        {
            "from": "Alice <alice@acme.com>",
            "to": "Bob <bob@acme.com>, carol@acme.com",
            "cc": "Dave <dave@acme.com>",
        },
    )
    assert not pairs  # Gmail carries no cross-provider co-observations
    values = sorted(a.value for a in aliases)
    assert values == ["alice@acme.com", "bob@acme.com", "carol@acme.com", "dave@acme.com"]
    # Display name captured when RFC 5322 provided one.
    by_value = {a.value: a for a in aliases}
    assert by_value["alice@acme.com"].display_name == "Alice"
    assert by_value["carol@acme.com"].display_name is None


def test_gmail_extractor_dedupes_within_signal():
    """Same address appearing in both to and cc must produce one alias, not two."""
    aliases, _ = extract_aliases(
        SignalSource.GMAIL,
        {"from": "alice@acme.com", "to": "bob@acme.com", "cc": "Bob <bob@acme.com>"},
    )
    assert sorted(a.value for a in aliases) == ["alice@acme.com", "bob@acme.com"]


def test_gmail_extraction_is_replay_stable():
    payload = {"from": "alice@acme.com", "to": "bob@acme.com"}
    a1, p1 = extract_aliases(SignalSource.GMAIL, payload)
    a2, p2 = extract_aliases(SignalSource.GMAIL, payload)
    assert [(a.provider, a.kind, a.value) for a in a1] \
        == [(a.provider, a.kind, a.value) for a in a2]
    assert p1 == p2


# --- slack extractor ---------------------------------------------------------

def test_slack_profile_links_slack_id_and_email():
    """A Slack profile that self-declares an email produces a structural link
    edge between the slack_user_id and email aliases — the SAME record binds
    the two, so we can assert equivalence with high confidence."""
    aliases, pairs = extract_aliases(
        SignalSource.SLACK,
        {"slack_user_id": "U01ABC", "email": "alice@acme.com", "display_name": "Alice"},
    )
    kinds = sorted((a.kind.value, a.value) for a in aliases)
    assert kinds == [("email", "alice@acme.com"), ("slack_user_id", "U01ABC")]
    assert len(pairs) == 1
    a, b = pairs[0]
    assert {a.kind, b.kind} == {AliasKind.SLACK_USER_ID, AliasKind.EMAIL}


def test_slack_message_without_email_only_records_slack_id():
    aliases, pairs = extract_aliases(
        SignalSource.SLACK, {"slack_user_id": "U01ABC"}
    )
    assert len(aliases) == 1 and aliases[0].kind == AliasKind.SLACK_USER_ID
    assert not pairs  # nothing to link


# --- calendar extractor ------------------------------------------------------

def test_calendar_extractor_reads_attendees():
    aliases, pairs = extract_aliases(
        SignalSource.CALENDAR,
        {"attendees": [
            {"email": "alice@acme.com", "displayName": "Alice"},
            {"email": "Bob@Acme.com"},   # normalized
            "carol@acme.com",             # tolerant of plain-string attendee
        ]},
    )
    assert not pairs
    values = sorted(a.value for a in aliases)
    assert values == ["alice@acme.com", "bob@acme.com", "carol@acme.com"]


# --- unknown source ----------------------------------------------------------

def test_unknown_source_is_safe_noop():
    aliases, pairs = extract_aliases(SignalSource.DRIVE, {"anything": True})
    assert aliases == []
    assert pairs == []


# --- Alias.compute_id --------------------------------------------------------

def test_alias_compute_id_matches_helper():
    a = Alias(provider=AliasProvider.GMAIL, kind=AliasKind.EMAIL, value="alice@acme.com")
    assert a.compute_id(ORG_A) == alias_id(ORG_A, "gmail", "email", "alice@acme.com")
