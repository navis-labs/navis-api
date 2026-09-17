"""Knowledge — the canonical projection, and the rules that keep it canonical.

The defect these pin is architectural rather than cosmetic. Knowledge used to be
assembled in the browser from `/memory/people`, `/decision-memory`, the attention
feed and the blocker list, with entities invented per feed
(`proj-${slug(name)}`), companies derived from an email domain, and connections
made with `String.includes()`. Underneath, `/memory/people` re-derived from raw
`memory_records` what State V2 already folds, and filtered people with a
five-layer regex that — measured on live data — let nine `ghsa-*@noreply.github.com`
advisory addresses and six `noreply@` senders through as People.

Every test here asserts the same thing from a different angle: there is ONE
organizational model, it is State V2, and Knowledge only reshapes it.
"""
from __future__ import annotations

from navis.api.knowledge_router import _identity
from navis.memory.envelope import EntityType
from navis.state.v2.builders import build_projection
from navis.state.v2.knowledge import (
    IDENTITY_HUMAN,
    IDENTITY_NON_HUMAN,
    IDENTITY_UNKNOWN,
    canonical_label,
    project_knowledge,
)
from tests.state.fakes import ORG_A, ORG_B, rec


def classify_identity(observed):
    """One entity's verdict, via the real composition-root classifier.

    The verdict is computed at the API seam rather than inside the projection,
    because `navis/state/**` may not import Reasoning (AST-pinned in
    `test_scope_projection.py`) and the two predicates live in
    `reasoning.v2.base`. Going through `_identity` here means these tests
    exercise the SAME code path the endpoint does.
    """
    return _identity({"person": {"e": observed}})["person"]["e"]

# ── A production-shaped corpus ──────────────────────────────────────────────
#
# Three addresses that behave exactly as the live org's do:
#   * a real correspondent, evidenced by a commitment and a meeting;
#   * an automated sender, evidenced only by newsletter/marketing mail;
#   * a reply-routing address that arrives as a `To:` recipient and is never
#     observed as a sender — the GHSA case, and the one the regex missed.

HUMAN = "ajitesh@example.com"
ROBOT = "notifications@calendly.com"
ROUTING = "ghsa-28wg-ghj8-5hjv@noreply.github.com"


def _corpus(org_id=ORG_A):
    return [
        rec(EntityType.PERSON, HUMAN, {"email": HUMAN, "name": "Ajitesh"},
            offset=0, tag="p-h", org_id=org_id),
        rec(EntityType.RELATIONSHIP, HUMAN,
            {"person_email": HUMAN, "direction": "outbound"},
            offset=1, tag="r-h", org_id=org_id),
        rec(EntityType.COMMITMENT, "c-1",
            {"text": "I'll come back to you on it.", "speaker_email": HUMAN,
             "promised_at": "2026-06-01T12:00:00+00:00"},
            offset=2, tag="c-1", org_id=org_id),
        rec(EntityType.MEETING, "m-1",
            {"title": "Intro call", "attendees": [HUMAN]},
            offset=3, tag="m-1", org_id=org_id),
        # Automated sender: State records is_automated on its own facts.
        rec(EntityType.PERSON, ROBOT,
            {"email": ROBOT, "name": "Calendly", "is_automated": True,
             "category": "update"},
            offset=4, tag="p-r", org_id=org_id),
        # Reply-routing recipient: NO is_automated, because it was never a
        # sender. This is the address the old regex let through.
        rec(EntityType.PERSON, ROUTING, {"email": ROUTING, "role": "to"},
            offset=5, tag="p-g", org_id=org_id),
        rec(EntityType.COMPANY, "example.com", {"domain": "example.com"},
            offset=6, tag="co", org_id=org_id),
    ]


def _ids(state, entity_type: str, *keys: str) -> dict[str, str]:
    """natural-key -> the canonical State entity_id, read from State itself.

    Reads whichever field carries the natural key for the type — `email` on a
    person, `person_email` on a relationship, `domain` on a company — rather than
    recomputing the uuid5, so the test resolves ids the same way a consumer would.
    """
    out = {}
    for view in state.entities.get(entity_type, {}).values():
        fields = view.get("fields", {})
        key = fields.get("email") or fields.get("person_email") or fields.get("domain")
        if key in keys:
            out[key] = view["entity_id"]
    return out


def _evidence(state):
    """The canonical lineage evidence, shaped as the composition root hands it.

    Mirrors `memory.lineage.classifications_by_entity` output: entity_id ->
    observed classifications. The human carries human-bearing evidence; the two
    automated addresses carry only newsletter/marketing.

    RELATIONSHIP evidence is supplied too, and that is faithful rather than
    convenient: it is the SAME dict `orchestration.brief` reads and hands to
    `run_reasoning_v2`, and production's cooling signal for this counterparty
    publishes `relevance_evidence: ["commitment", "meeting"]` — that dict,
    verbatim. A relationship gated without it would be an artefact of the
    fixture, not of the org.
    """
    people = _ids(state, "person", HUMAN, ROBOT, ROUTING)
    rels = _ids(state, "relationship", HUMAN)
    return {
        "person": {
            people[HUMAN]: frozenset({"commitment", "meeting"}),
            people[ROBOT]: frozenset({"automated_other", "marketing"}),
            people[ROUTING]: frozenset({"newsletter"}),
        },
        "relationship": {
            rels[HUMAN]: frozenset({"commitment", "meeting"}),
        },
    }


def _view(org_id=ORG_A):
    state = build_projection(org_id, _corpus(org_id))
    ev = _evidence(state)
    return state, project_knowledge(state, evidence=ev, identity=_identity(ev))


# ── 1. Identity: canonical evidence, never a regex ──────────────────────────


def test_the_identity_verdict_is_tri_state():
    assert classify_identity(frozenset({"commitment"})) == IDENTITY_HUMAN
    assert classify_identity(frozenset({"newsletter"})) == IDENTITY_NON_HUMAN
    # ABSENT evidence is UNKNOWN — never "automated", never "human".
    assert classify_identity(None) == IDENTITY_UNKNOWN
    assert classify_identity(frozenset()) == IDENTITY_UNKNOWN


def test_no_automated_sender_is_published_as_a_person():
    """ACCEPTANCE 1. Calendly is automated by State's own flag; the GHSA routing
    address is not — it was never a sender — and the old regex missed it. Both
    must be withheld, and for the same canonical reason."""
    _state, view = _view()
    emails = {e.properties.get("email") for e in view.entities["person"]}
    assert emails == {HUMAN}
    assert ROBOT not in emails
    assert ROUTING not in emails


def test_the_human_is_published_with_the_evidence_that_earned_it():
    _state, view = _view()
    [person] = view.entities["person"]
    assert person.identity == IDENTITY_HUMAN
    assert person.observed_evidence == ("commitment", "meeting")


def test_an_entity_with_no_evidence_is_unknown_and_never_silently_a_person():
    """UNKNOWN ≠ ZERO and UNKNOWN ≠ PERSON. An org whose importance log is empty
    must not have its whole People list asserted into existence."""
    state = build_projection(ORG_A, _corpus())
    view = project_knowledge(state, evidence={}, identity={})
    assert view.entities["person"] == ()
    gate = view.gating["person"]
    assert gate["withheld_unknown"] == 3
    assert gate["withheld_observed_non_human"] == 0
    # The entities still EXIST and the count still says so.
    assert view.counts["person"] == 3


def test_gating_reports_the_difference_rather_than_hiding_it():
    """A silent filter is indistinguishable from data loss. Both numbers travel."""
    _state, view = _view()
    gate = view.gating["person"]
    assert gate["rule"] == "has_human_bearing_evidence"
    assert (gate["total"], gate["published"], gate["withheld"]) == (3, 1, 2)
    assert gate["withheld_observed_non_human"] == 2
    assert view.counts["person"] == 3, "counts are State's, not the list length"


def test_only_person_and_relationship_are_evidence_gated():
    """A meeting is a meeting whatever classification its mail carried."""
    _state, view = _view()
    assert set(view.gating) <= {"person", "relationship"}
    assert len(view.entities["meeting"]) == 1
    assert len(view.entities["company"]) == 1
    assert len(view.entities["commitment"]) == 1


# ── 2. Identity is stable; display strings are never identity ───────────────


def test_the_entity_id_is_states_own_id():
    state, view = _view()
    for entity_type, rows in view.entities.items():
        for entity in rows:
            assert entity.entity_id in state.entities[entity_type]


def test_renaming_a_person_does_not_change_their_entity_id():
    """ACCEPTANCE 3. The natural key is the email; the display name is not part
    of the id. `proj-${slug(name)}` in the old composer had the opposite
    property — renaming a project minted a new entity."""
    before = build_projection(ORG_A, _corpus())
    renamed = _corpus()
    renamed[0] = rec(EntityType.PERSON, HUMAN,
                     {"email": HUMAN, "name": "Ajitesh Kumar (CEO)"},
                     offset=0, tag="p-h")
    after = build_projection(ORG_A, renamed)
    assert set(before.entities["person"]) == set(after.entities["person"])


def test_the_canonical_label_is_read_from_state_never_constructed():
    state, view = _view()
    [person] = view.entities["person"]
    assert person.canonical_label == "Ajitesh"
    [company] = view.entities["company"]
    assert company.canonical_label == "example.com"


def test_an_entity_with_no_nameable_field_falls_back_to_its_real_id():
    """A real identifier, never a placeholder that would read as a name."""
    class _Raw:
        entity_id = "abc-123"
        fields: dict = {}

    assert canonical_label("person", _Raw()) == "abc-123"


# ── 3. Relationships come from persisted links only ─────────────────────────


def test_relationships_are_the_persisted_state_edges():
    """ACCEPTANCE 4/5. Never `label.includes(...)` — these are the same
    `EntityLink`s `/context` publishes, from `folds.compute_links`."""
    state, view = _view()
    edges = {
        (e.entity_type, link.role, link.entity_type)
        for rows in view.entities.values()
        for e in rows
        for link in e.relationships
    }
    assert ("commitment", "owner", "person") in edges
    assert ("relationship", "commitments", "commitment") in edges
    assert ("relationship", "meetings", "meeting") in edges


def test_every_published_edge_exists_in_state():
    state, view = _view()
    for rows in view.entities.values():
        for entity in rows:
            raw = state.entities[entity.entity_type][entity.entity_id]
            for link in entity.relationships:
                assert link.entity_id in raw["links"][link.role]


def test_knowledge_never_string_matches_to_connect_entities():
    """A source-level pin. The composer this replaces made 13 such joins."""
    import inspect

    from navis.state.v2 import knowledge

    src = inspect.getsource(knowledge)
    for forbidden in (".includes(", "in label", "label.lower() in", "startswith(label"):
        assert forbidden not in src


# ── 4. Provenance and time survive ──────────────────────────────────────────


def test_provenance_reaches_the_wire():
    """ACCEPTANCE 6. `/memory/people` returned no memory ids at all, so the chain
    from a card back to the captured email was unwalkable."""
    _state, view = _view()
    [person] = view.entities["person"]
    assert person.provenance.memory_ids
    assert person.provenance.signal_event_ids
    wire = person.to_dict()
    assert wire["provenance"]["memory_ids"] == list(person.provenance.memory_ids)


def test_provenance_is_never_fabricated():
    state, view = _view()
    for rows in view.entities.values():
        for entity in rows:
            raw = state.entities[entity.entity_type][entity.entity_id]
            assert list(entity.provenance.memory_ids) == raw["provenance"]["memory_ids"]


def test_both_timestamps_survive_and_the_clock_is_logical():
    """ACCEPTANCE 7. `/memory/people` stamped `asOf: datetime.now()` — wall clock
    on a projection of folded history."""
    state, view = _view()
    assert view.as_of == state.as_of
    assert view.state_watermark == str(state.watermark.last_memory_id)
    for rows in view.entities.values():
        for entity in rows:
            assert entity.first_seen and entity.last_seen


def test_the_projection_is_pure_and_deterministic():
    state = build_projection(ORG_A, _corpus())
    ev, ident = _evidence(state), None
    ident = _identity(ev)
    assert project_knowledge(state, evidence=ev, identity=ident).to_dict() == \
           project_knowledge(state, evidence=ev, identity=ident).to_dict()


# ── 5. Knowledge is not Intelligence ────────────────────────────────────────


def test_knowledge_publishes_no_interpretation():
    """ACCEPTANCE 9. `_derive_person` shipped `escalationRisk` and
    `responseLikelihood` computed from an inbound/outbound ratio; `_derive_decision`
    shipped an `impact` score from a subject regex. None of it was evidence."""
    _state, view = _view()
    body = view.to_dict()
    forbidden = {
        "risk", "churn", "opportunity", "importance", "impact", "escalation",
        "escalationRisk", "responseLikelihood", "severity", "score", "baseline",
        "likelihood", "prediction", "recommendation",
    }
    for rows in body["entities"].values():
        for entity in rows:
            assert forbidden.isdisjoint(entity)
            # `properties` is State's own field dict; it must not have gained
            # a derived key either.
            assert forbidden.isdisjoint(entity["properties"])


def test_counts_cover_every_type_state_models_including_empty_ones():
    """UNAVAILABLE ≠ ZERO, but a genuinely empty type must still report 0 so a
    surface can tell them apart."""
    _state, view = _view()
    assert set(view.counts) == {t.value for t in EntityType}
    assert view.counts["project"] == 0
    assert view.counts["customer"] == 0


# ── 6. Tenancy ──────────────────────────────────────────────────────────────


def test_no_entity_or_edge_can_cross_a_tenant():
    _state_a, view_a = _view(ORG_A)
    state_b = build_projection(ORG_B, _corpus(ORG_B))
    b_ids = {eid for bucket in state_b.entities.values() for eid in bucket}
    published = {e.entity_id for rows in view_a.entities.values() for e in rows}
    targets = {
        link.entity_id
        for rows in view_a.entities.values()
        for e in rows
        for link in e.relationships
    }
    assert published.isdisjoint(b_ids)
    assert targets.isdisjoint(b_ids)
    assert view_a.org_id == str(ORG_A)
