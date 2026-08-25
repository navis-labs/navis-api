"""OpportunityEngine relevance — P0-2.

Every address on every email becomes a RELATIONSHIP, so the engine used to emit
"Active inbound engagement" for newsletters, marketing blasts and billing robots.

The obvious gate — Memory's `is_automated` — is the wrong instrument, and P0-1A
proved it against live data: Google Calendar sends an RSVP ON BEHALF of a human
attendee, stamping `Auto-Submitted: auto-generated` while keeping that person's
address in `From:`. The flag is a per-MESSAGE property folded latest-wins onto a
long-lived address, so a real counterparty is marked automated or not depending
on which message happened to arrive last.

Relevance is therefore decided from the SET of `signal_importance.classification`
values observed across the whole relationship, which is monotone: evidence
accumulates and is never retracted. These tests pin that behaviour against the
production engine — no mocks of the gate itself.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest

from navis.memory.envelope import EntityType
from navis.reasoning.v2 import run_reasoning_v2
from navis.reasoning.v2.contract import SignalType
from navis.reasoning.v2.base import HUMAN_BEARING_EVIDENCE as _HUMAN_BEARING_EVIDENCE
from navis.state.v2.builders import MemoryRecordView, build_projection

NS = uuid.UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
ORG_A = uuid.UUID("11111111-1111-1111-1111-111111111111")
ORG_B = uuid.UUID("22222222-2222-2222-2222-222222222222")
NOW = datetime(2026, 8, 16, 9, 0, tzinfo=timezone.utc)


def _rel(
    email: str,
    *,
    org_id: uuid.UUID = ORG_A,
    days_ago: float = 1.0,
    direction: str = "inbound",
    is_automated: bool | None = None,
    category: str | None = None,
    tag: str = "r",
) -> MemoryRecordView:
    """One RELATIONSHIP memory record, exactly as email.py writes it."""
    payload: dict = {"person_email": email, "direction": direction, "via": "email"}
    if is_automated is not None:
        payload["is_automated"] = is_automated
    if category is not None:
        payload["category"] = category
    return MemoryRecordView(
        memory_id=uuid.uuid5(NS, f"mem:{org_id}:{email}:{tag}"),
        org_id=org_id,
        entity_type=EntityType.RELATIONSHIP.value,
        entity_id=uuid.uuid5(NS, f"ent:{org_id}:relationship:{email}"),
        event_timestamp=NOW - timedelta(days=days_ago),
        payload=payload,
        source_signal_event_id=uuid.uuid5(NS, f"sig:{org_id}:{email}:{tag}"),
        schema_version=1,
        confidence=0.9,
        # Observed the moment it happened — this corpus models a live inbox, not
        # a backfill, so the two clocks coincide. The field is what gives the
        # projection an observation cursor, which is the provenance
        # `state_watermark` carries.
        observed_at=NOW - timedelta(days=days_ago),
    )


def _entity_id(email: str, org_id: uuid.UUID = ORG_A) -> str:
    return str(uuid.uuid5(NS, f"ent:{org_id}:relationship:{email}"))


def _opportunities(records, evidence=None, *, org_id: uuid.UUID = ORG_A):
    """Run the REAL reasoning pass and keep the opportunity signals."""
    state = build_projection(org_id, list(records), as_of=NOW)
    return [
        s for s in run_reasoning_v2(state, evidence=evidence or {})
        if s.signal_type is SignalType.OPPORTUNITY
    ]


# ── The vocabulary is bound to its source of truth ─────────────────────
#
# `navis/reasoning/v2/**` may not import `navis.signals` (AST invariant in
# test_reasoning_v2.py), so the policy set is written as literals. This test is
# what stops the two sides drifting apart silently.


def test_human_bearing_classes_are_all_real_importance_classes():
    from navis.signals.importance import ImportanceClass

    known = {
        v for k, v in vars(ImportanceClass).items()
        if not k.startswith("_") and isinstance(v, str)
    }
    unknown = _HUMAN_BEARING_EVIDENCE - known
    assert not unknown, (
        f"{unknown} are not ImportanceClass values — the reasoning policy and the "
        "signals taxonomy have drifted apart"
    )


def test_non_human_classes_are_deliberately_excluded_not_forgotten():
    """These stay first-class shared evidence for other engines; this engine
    simply does not read them as evidence of a human relationship."""
    for cls in ("automated_other", "marketing", "newsletter", "invoice", "otp",
                "social", "spam"):
        assert cls not in _HUMAN_BEARING_EVIDENCE


# ── 1. Human meeting evidence survives automated evidence ──────────────


def test_a_relationship_with_meeting_evidence_survives_automated_evidence():
    """The live case: a calendar RSVP from a real human. `automated_other` is
    present, but `meeting` is human-bearing, so the relationship stands."""
    records = [_rel("human@partner.com")]
    signals = _opportunities(
        records, {_entity_id("human@partner.com"): frozenset({"automated_other", "meeting"})}
    )
    assert len(signals) == 1
    assert signals[0].evidence["person_email"] == "human@partner.com"
    assert signals[0].evidence["relevance_evidence"] == ["automated_other", "meeting"]


# ── 2/3. Non-human-only evidence is suppressed ─────────────────────────


def test_a_purely_automated_relationship_is_suppressed():
    signals = _opportunities(
        [_rel("robot@vendor.com")],
        {_entity_id("robot@vendor.com"): frozenset({"automated_other"})},
    )
    assert signals == []


def test_a_marketing_relationship_is_suppressed():
    signals = _opportunities(
        [_rel("news@substack.com")],
        {_entity_id("news@substack.com"): frozenset({"marketing", "newsletter"})},
    )
    assert signals == []


# ── 4/5. Each human-bearing class on its own is sufficient ─────────────


@pytest.mark.parametrize("cls", sorted(_HUMAN_BEARING_EVIDENCE))
def test_every_human_bearing_class_alone_preserves_the_opportunity(cls):
    signals = _opportunities(
        [_rel("alice@acme.com")], {_entity_id("alice@acme.com"): frozenset({cls})}
    )
    assert len(signals) == 1, f"{cls} must preserve the relationship"


# ── 6. Mixed history is monotone ───────────────────────────────────────


def test_old_marketing_plus_new_meeting_survives():
    """Evidence accumulates. An address that has EVER been observed in a
    human-bearing exchange keeps that evidence — unlike latest-wins folding,
    where the newest message decides and the answer flips with arrival order."""
    records = [
        _rel("mixed@partner.com", days_ago=2.0, tag="old"),
        _rel("mixed@partner.com", days_ago=0.5, tag="new"),
    ]
    evidence = {_entity_id("mixed@partner.com"): frozenset({"marketing", "meeting"})}
    assert len(_opportunities(records, evidence)) == 1


def test_evidence_order_cannot_change_the_outcome():
    """A set has no order, so the same two classifications decide the same way
    however they arrived. This is the property `is_automated` lacked."""
    eid = _entity_id("mixed@partner.com")
    records = [_rel("mixed@partner.com")]
    a = _opportunities(records, {eid: frozenset({"marketing", "meeting"})})
    b = _opportunities(records, {eid: frozenset({"meeting", "marketing"})})
    assert [s.signal_id for s in a] == [s.signal_id for s in b]
    assert len(a) == 1


# ── 7. THE CRITICAL CASE: is_automated conflicts with the evidence ─────


def test_is_automated_TRUE_does_not_suppress_a_human_meeting():
    """The exact live failure. Memory says automated (correct about the RSVP
    message); the accumulated evidence says `meeting` (correct about the
    person). The evidence wins — the flag is not consulted at all."""
    records = [_rel("rsvp@partner.com", is_automated=True, category="automated")]
    signals = _opportunities(
        records, {_entity_id("rsvp@partner.com"): frozenset({"meeting"})}
    )
    assert len(signals) == 1
    assert signals[0].evidence["relevance_evidence"] == ["meeting"]


def test_is_automated_FALSE_does_not_rescue_a_vendor():
    """The mirror: the flag cannot admit what the evidence rejects, either."""
    records = [_rel("blast@vendor.com", is_automated=False, category="personal")]
    signals = _opportunities(
        records, {_entity_id("blast@vendor.com"): frozenset({"marketing"})}
    )
    assert signals == []


# ── 8. UNKNOWN stays UNKNOWN ───────────────────────────────────────────


def test_no_evidence_preserves_prior_behaviour_and_fabricates_nothing():
    """Missing evidence must never silently mute a real relationship, and must
    never be reported as human relevance that was not observed."""
    signals = _opportunities([_rel("unknown@x.com", is_automated=None)], {})
    assert len(signals) == 1
    assert signals[0].evidence["relevance_evidence"] == []


def test_an_empty_evidence_set_is_also_unknown():
    """A relationship present in the map with no classifications is the same
    absence of information as one missing from it."""
    signals = _opportunities(
        [_rel("unknown@x.com")], {_entity_id("unknown@x.com"): frozenset()}
    )
    assert len(signals) == 1


# ── 9. Organization isolation ──────────────────────────────────────────


def test_org_A_evidence_cannot_influence_org_B():
    """Entity ids are org-scoped, so B's identically-addressed relationship
    cannot be reached by A's evidence — it stays UNKNOWN, not suppressed."""
    evidence_a = {_entity_id("shared@vendor.com", ORG_A): frozenset({"marketing"})}

    assert _opportunities([_rel("shared@vendor.com", org_id=ORG_A)], evidence_a) == []

    in_b = _opportunities(
        [_rel("shared@vendor.com", org_id=ORG_B)], evidence_a, org_id=ORG_B
    )
    assert len(in_b) == 1
    assert in_b[0].org_id == ORG_B
    assert in_b[0].evidence["relevance_evidence"] == []


# ── 10. Provenance ─────────────────────────────────────────────────────


def test_the_signal_keeps_full_provenance_and_cites_its_relevance_evidence():
    from navis.reasoning.v2.contract import REASONING_VERSION

    records = [_rel("alice@acme.com")]
    state = build_projection(ORG_A, records, as_of=NOW)
    (s,) = [
        x for x in run_reasoning_v2(
            state, evidence={_entity_id("alice@acme.com"): frozenset({"commitment"})}
        )
        if x.signal_type is SignalType.OPPORTUNITY
    ]
    assert s.org_id == ORG_A
    assert s.entity_type == "relationship"
    assert s.entity_id == _entity_id("alice@acme.com")
    assert s.state_as_of == NOW
    assert s.state_watermark == str(state.watermark.last_memory_id)
    assert s.reasoning_version == REASONING_VERSION
    assert s.engine_name == "opportunity"
    # The relevance call is auditable back to the signals it was derived from.
    assert s.evidence["relevance_evidence"] == ["commitment"]


def test_reasoning_stays_deterministic_with_evidence():
    """Same (state, evidence) in → byte-identical signals out, ids included."""
    records = [_rel("alice@acme.com")]
    ev = {_entity_id("alice@acme.com"): frozenset({"meeting", "marketing"})}
    a = _opportunities(records, ev)
    b = _opportunities(records, ev)
    assert [x.to_dict() for x in a] == [x.to_dict() for x in b]


# ══════════════════════════════════════════════════════════════════════════════
# Phase 4.1.1 — RELATIONSHIP_COOLING relevance
# ══════════════════════════════════════════════════════════════════════════════
#
# The same vocabulary, now shared from base.py and applied by
# RelationshipReasoningEngine. Live, that engine had NO gate and reported
# "relationship cooling" for microsoft-noreply@microsoft.com,
# openscout+on-the-side@substack.com, mail@mail.adobe.com and
# founders@mail.slashy.com — which Intelligence then aggregated into
# organizational RISK insights.

_COOLING_DAYS = 12.0  # inside [RELATIONSHIP_COOLING_DAYS, MAX_SURFACE_AGE_DAYS]


def _cooling(records, evidence=None, *, org_id: uuid.UUID = ORG_A):
    """Run the REAL reasoning pass and keep the cooling signals."""
    state = build_projection(org_id, list(records), as_of=NOW)
    return [
        s for s in run_reasoning_v2(state, evidence=evidence or {})
        if s.signal_type is SignalType.RELATIONSHIP_COOLING
    ]


def test_one_shared_relevance_definition_not_two():
    """Both engines must read the SAME set — a second copy would drift."""
    from navis.reasoning.v2.base import HUMAN_BEARING_EVIDENCE
    from navis.reasoning.v2.engines import meta

    assert meta._HUMAN_BEARING_EVIDENCE is HUMAN_BEARING_EVIDENCE


# ── 1/2. Real human and business relationships still cool ──────────────


@pytest.mark.parametrize("cls", sorted(_HUMAN_BEARING_EVIDENCE))
def test_a_human_bearing_relationship_still_cools(cls):
    signals = _cooling(
        [_rel("alice@acme.com", days_ago=_COOLING_DAYS)],
        {_entity_id("alice@acme.com"): frozenset({cls})},
    )
    assert len(signals) == 1, f"{cls} must keep the cooling conclusion"
    assert signals[0].evidence["person_email"] == "alice@acme.com"
    assert signals[0].evidence["relevance_evidence"] == [cls]


# ── 3/4. The exact production offenders ────────────────────────────────


@pytest.mark.parametrize(
    "email,observed",
    [
        # Live evidence, read from signal_importance on 2026-08-17.
        ("microsoft-noreply@microsoft.com", {"invoice"}),
        ("openscout+on-the-side@substack.com", {"newsletter"}),
        ("mail@mail.adobe.com", {"marketing"}),
        ("founders@mail.slashy.com", {"automated_other"}),
        ("noreply@github.com", {"automated_other"}),
        ("no-reply@spotify.com", {"automated_other"}),
        ("info@join.netflix.com", {"marketing"}),
    ],
)
def test_the_production_offenders_no_longer_cool(email, observed):
    assert _cooling(
        [_rel(email, days_ago=_COOLING_DAYS)],
        {_entity_id(email): frozenset(observed)},
    ) == []


# ── 5. Mixed history: human evidence qualifies ─────────────────────────


def test_a_relationship_with_any_human_evidence_survives_automation():
    """astravision.global@gmail.com live: newsletter + marketing + invoice +
    automated_other AND commitment + meeting. The human evidence decides."""
    observed = frozenset({
        "automated_other", "commitment", "invoice",
        "marketing", "meeting", "newsletter",
    })
    signals = _cooling(
        [_rel("astravision.global@gmail.com", days_ago=_COOLING_DAYS)],
        {_entity_id("astravision.global@gmail.com"): observed},
    )
    assert len(signals) == 1
    assert signals[0].evidence["relevance_evidence"] == sorted(observed)


# ── UNKNOWN: the deliberate asymmetry between the two engines ──────────


def test_unknown_yields_no_cooling_conclusion():
    """"Absence of evidence = no relationship conclusion." A cooling claim that
    cannot establish a counterparty is exactly the fabrication this forbids."""
    assert _cooling([_rel("unknown@x.com", days_ago=_COOLING_DAYS)], {}) == []
    assert _cooling(
        [_rel("unknown@x.com", days_ago=_COOLING_DAYS)],
        {_entity_id("unknown@x.com"): frozenset()},
    ) == []


def test_unknown_still_yields_an_opportunity():
    """The asymmetry, pinned. OPPORTUNITY keeps UNKNOWN (a wrong warm window
    costs a glance); COOLING drops it (a wrong risk corrupts the org's picture
    of itself). Both are correct, for different reasons."""
    assert len(_opportunities([_rel("unknown@x.com")], {})) == 1


# ── 6/7. Determinism and organization isolation ────────────────────────


def test_cooling_is_deterministic_with_evidence():
    records = [_rel("alice@acme.com", days_ago=_COOLING_DAYS)]
    ev = {_entity_id("alice@acme.com"): frozenset({"meeting", "marketing"})}
    a, b = _cooling(records, ev), _cooling(records, ev)
    assert [x.to_dict() for x in a] == [x.to_dict() for x in b]


def test_cooling_evidence_is_org_scoped():
    """A's evidence cannot qualify B's identically-addressed relationship."""
    ev_a = {_entity_id("shared@acme.com", ORG_A): frozenset({"meeting"})}
    in_a = _cooling([_rel("shared@acme.com", days_ago=_COOLING_DAYS)], ev_a)
    assert len(in_a) == 1 and in_a[0].org_id == ORG_A
    # In B the same address is UNKNOWN, so no conclusion is drawn.
    assert _cooling(
        [_rel("shared@acme.com", org_id=ORG_B, days_ago=_COOLING_DAYS)],
        ev_a, org_id=ORG_B,
    ) == []


# ── 9/10. Neighbouring engines are untouched by the gate ───────────────


def test_the_gate_does_not_reach_commitments_or_other_risk_engines():
    """The gate is relationship-scoped. A commitment from an address with only
    automated evidence is still an unmet promise."""
    import datetime as _dt

    from tests.state.fakes import corpus

    records = corpus(ORG_A)
    as_of = max(r.event_timestamp for r in records) + _dt.timedelta(days=10)
    state = build_projection(ORG_A, records, as_of=as_of)
    # Every relationship observed as purely automated => no cooling at all...
    ev = {r.entity_id: frozenset({"automated_other"}) for r in state.relationships()}
    types = {s.signal_type for s in run_reasoning_v2(state, evidence=ev)}
    assert SignalType.RELATIONSHIP_COOLING not in types
    # ...while the obligation and decay engines keep their own conclusions.
    assert SignalType.COMMITMENT_SLIPPING in types
    assert SignalType.CONTEXT_SUMMARY in types


def test_priority_load_counts_cooling_under_the_same_gate():
    """PRIORITY_LOAD recounts from State, so without the gate it would report
    cooling relationships that produced no signal — and organizational_status
    surfaces both numbers side by side."""
    import datetime as _dt

    from tests.state.fakes import corpus

    records = corpus(ORG_A)
    as_of = max(r.event_timestamp for r in records) + _dt.timedelta(days=10)
    state = build_projection(ORG_A, records, as_of=as_of)

    automated = {r.entity_id: frozenset({"marketing"}) for r in state.relationships()}
    load = [s for s in run_reasoning_v2(state, evidence=automated)
            if s.signal_type is SignalType.PRIORITY_LOAD]
    assert load and load[0].evidence["cooling"] == 0

    human = {r.entity_id: frozenset({"meeting"}) for r in state.relationships()}
    load_h = [s for s in run_reasoning_v2(state, evidence=human)
              if s.signal_type is SignalType.PRIORITY_LOAD]
    cooling_signals = [s for s in run_reasoning_v2(state, evidence=human)
                       if s.signal_type is SignalType.RELATIONSHIP_COOLING]
    assert load_h[0].evidence["cooling"] == len(cooling_signals) > 0
