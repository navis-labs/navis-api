"""Context projection completeness — Phase 3C.

`_scope_morning_brief` used to hardcode five `state.<type>()` calls, so State
could hold meetings, documents, decisions, projects, customers and channels and
Context dropped every one of them before Reasoning or ML ever saw them. The
failure was invisible and recurring: every entity type added to State AFTER that
function was written was dropped by default.

These pin the fix at the root — the projection is keyed by EntityType and
asserted exhaustive, so a new entity type cannot be added without deciding how
it is scoped.
"""
from __future__ import annotations



import pytest

from navis.memory.envelope import EntityType
from navis.state.v2 import MorningBrief, scope_for
from navis.state.v2.builders import build_projection
from navis.state.v2.scope import _MORNING_BRIEF, _TIER_ORDER
from tests.state.fakes import ORG_A, ORG_B, corpus, rec


def _view(records, org_id=ORG_A, policy=None):
    state = build_projection(org_id, list(records))
    return state, scope_for(state, MorningBrief(), policy, as_of=state.as_of)


def _types(view) -> set[str]:
    return {e.entity_type for e in view.entities}


def _of_type(view, entity_type: str) -> list:
    return [e for e in view.entities if e.entity_type == entity_type]


# One record per entity type, with the normalized fields State actually models.
_ONE_OF_EACH = [
    rec(EntityType.PERSON, "alice@acme.com",
        {"email": "alice@acme.com", "name": "Alice", "role": "from"}, offset=0, tag="p"),
    rec(EntityType.COMPANY, "acme.com", {"domain": "acme.com"}, offset=1, tag="co"),
    rec(EntityType.RELATIONSHIP, "alice@acme.com",
        {"person_email": "alice@acme.com", "direction": "inbound"}, offset=2, tag="r"),
    rec(EntityType.THREAD, "t-1",
        {"thread_id": "t-1", "subject": "Q3 pricing", "message_id": "m1"}, offset=3, tag="t"),
    rec(EntityType.COMMITMENT, "c-1",
        {"text": "I'll send the deck", "speaker_email": "alice@acme.com"}, offset=4, tag="cm"),
    rec(EntityType.MEETING, "evt-1",
        {"title": "Nick and Abhi", "starts_at": "2026-08-13T10:30:00+00:00",
         "ends_at": "2026-08-13T11:15:00+00:00", "status": "confirmed",
         "attendees": ["alice@acme.com", "nick@x.com"], "has_conference": True,
         "conference_id": "conf-xyz"}, offset=5, tag="mt"),
    rec(EntityType.DOCUMENT, "doc-1",
        {"document_id": "doc-1", "title": "Product Spec", "owners": ["alice@acme.com"],
         "status": "active"}, offset=6, tag="d"),
    rec(EntityType.DECISION, "dec-1",
        {"text": "We decided to raise", "kind": "record"}, offset=7, tag="dec"),
    rec(EntityType.PROJECT, "folder-1", {"folder_id": "folder-1", "name": "Zeus"},
        offset=8, tag="pr"),
    rec(EntityType.CUSTOMER, "cust-1", {"external_ref": "cust-1", "name": "Acme Inc"},
        offset=9, tag="cu"),
    rec(EntityType.CHANNEL, "C1", {"channel_id": "C1", "name": "general"},
        offset=10, tag="ch"),
]


# ── The root fix: completeness is structural, not remembered ───────────


def test_every_state_entity_type_has_a_scope_policy():
    """The guard that stops this defect recurring. A new EntityType with no
    policy fails HERE, not silently in production."""
    assert set(_MORNING_BRIEF) == set(EntityType)
    assert set(_TIER_ORDER) == set(EntityType)
    assert len(_TIER_ORDER) == len(EntityType)  # no duplicates


def test_no_state_entity_type_is_dropped_by_the_projection():
    state, view = _view(_ONE_OF_EACH)
    present = {t for t, n in state.counts().items() if n}
    assert present == {e.value for e in EntityType}, "fixture must cover every type"
    assert _types(view) == present, (
        f"Context dropped {present - _types(view)} — the exact Phase 3C defect"
    )


# ── 1/2. Meetings and documents reach Context ──────────────────────────


def test_a_meeting_in_state_appears_in_the_scoped_view_with_its_fields():
    _state, view = _view(_ONE_OF_EACH)
    (meeting,) = _of_type(view, "meeting")
    assert meeting.fields["title"] == "Nick and Abhi"
    assert meeting.fields["starts_at"] == "2026-08-13T10:30:00+00:00"
    assert meeting.fields["ends_at"] == "2026-08-13T11:15:00+00:00"
    assert meeting.fields["attendees"] == ["alice@acme.com", "nick@x.com"]
    assert meeting.fields["status"] == "confirmed"
    # Virtual-meeting info rides along as normalized state, not as provider logic.
    assert meeting.fields["conference_id"] == "conf-xyz"
    assert meeting.last_seen and meeting.reason


def test_a_document_in_state_appears_in_the_scoped_view_with_its_fields():
    _state, view = _view(_ONE_OF_EACH)
    (doc,) = _of_type(view, "document")
    assert doc.fields["title"] == "Product Spec"
    assert doc.fields["owners"] == ["alice@acme.com"]
    assert doc.fields["status"] == "active"


def test_context_carries_document_pointers_never_document_bodies():
    """Context is a state projection, not a RAG dump. Whatever State models is
    what appears — and State models no body, so none can leak."""
    _state, view = _view(_ONE_OF_EACH)
    (doc,) = _of_type(view, "document")
    for leaky in ("body", "content", "text", "html", "snippet"):
        assert leaky not in doc.fields


# ── 3/4. Existing and remaining types behave as specified ──────────────


def test_the_original_five_types_keep_their_selection_rules():
    _state, view = _view(_ONE_OF_EACH)
    reasons = {e.entity_type: e.reason for e in view.entities}
    assert reasons["commitment"] == "open, oldest activity"
    assert reasons["relationship"] == "stalest interaction"
    assert reasons["thread"] == "most recently active"
    assert reasons["person"] == "most recently seen"
    assert reasons["company"] == "most connected"


def test_closed_commitments_and_decisions_are_excluded_open_ones_are_not():
    """`status` is fact-derived; absent means open. Same rule for both types."""
    records = [
        rec(EntityType.COMMITMENT, "c-open", {"text": "open one"}, offset=0, tag="c1"),
        rec(EntityType.COMMITMENT, "c-done", {"text": "done one", "status": "fulfilled"},
            offset=1, tag="c2"),
        rec(EntityType.DECISION, "d-open", {"text": "open call"}, offset=2, tag="d1"),
        rec(EntityType.DECISION, "d-done", {"text": "closed call", "status": "resolved"},
            offset=3, tag="d2"),
    ]
    _state, view = _view(records)
    assert [e.fields["text"] for e in _of_type(view, "commitment")] == ["open one"]
    assert [e.fields["text"] for e in _of_type(view, "decision")] == ["open call"]


def test_decisions_recorded_and_requested_both_compete():
    """One entity type, discriminated by `kind` — neither is filtered out here."""
    records = [
        rec(EntityType.DECISION, "d-rec", {"text": "we decided", "kind": "record"},
            offset=0, tag="d1"),
        rec(EntityType.DECISION, "d-req", {"text": "can you approve?", "kind": "request"},
            offset=1, tag="d2"),
    ]
    _state, view = _view(records)
    assert {e.fields["kind"] for e in _of_type(view, "decision")} == {"record", "request"}


# ── 5/6. Lineage is preserved for every type ───────────────────────────


def test_every_projected_entity_carries_its_existing_lineage():
    _state, view = _view(_ONE_OF_EACH)
    assert len(view.citations) == len(view.entities)
    for e in view.entities:
        cite = view.citations[e.entity_id]
        assert cite.entity_id == e.entity_id
        assert cite.memory_ids, f"{e.entity_type} lost its memory lineage"
        assert cite.signal_ids, f"{e.entity_type} lost its signal lineage"


@pytest.mark.parametrize("entity_type", ["meeting", "document", "project", "customer",
                                         "decision", "channel"])
def test_the_newly_projected_types_cite_real_upstream_ids(entity_type):
    """Lineage comes from the entity's own provenance — not re-derived, and not
    connector ids."""
    state, view = _view(_ONE_OF_EACH)
    (ref,) = _of_type(view, entity_type)
    raw = state.entities[entity_type][ref.entity_id]
    assert view.citations[ref.entity_id].memory_ids == raw["provenance"]["memory_ids"]
    assert view.citations[ref.entity_id].signal_ids == raw["provenance"]["signal_ids"]


# ── 7/8. Normalized fields only; no connector branching ────────────────


_CONNECTORS = ("gmail", "slack", "calendar", "drive", "google", "notion", "jira",
               "hubspot", "salesforce", "outlook", "sharepoint", "teams", "zoom")


def test_scope_module_contains_no_connector_specific_branching():
    """HARD REQUIREMENT: connector differences belong below State. A Teams
    meeting and a Calendar meeting are both a MeetingState here.

    Checked over the AST, not the text: prose explaining WHY the layer is
    source-agnostic is not a branch on source. What would be a violation is an
    executable string literal — `if source == "gmail"`, a provider lookup key —
    so this scans every non-docstring string constant.
    """
    import ast
    import pathlib

    path = pathlib.Path(__file__).resolve().parents[2] / "navis" / "state" / "v2" / "scope.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = getattr(node, "body", None)
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                docstrings.add(id(body[0].value))

    for node in ast.walk(tree):
        if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                and id(node) not in docstrings):
            lowered = node.value.lower()
            for connector in _CONNECTORS:
                assert connector not in lowered, (
                    f"scope.py has an executable {connector!r} literal at line "
                    f"{node.lineno} — connector semantics must live below State"
                )


def test_the_same_meeting_projects_identically_whatever_produced_it():
    """Provider lives in signal metadata, never in the semantic entity."""
    def one(conference_id):
        return _view([rec(EntityType.MEETING, "evt-1",
                          {"title": "Sync", "starts_at": "2026-08-13T10:30:00+00:00",
                           "conference_id": conference_id}, offset=0, tag="m")])[1]

    a, b = one("google-meet-abc"), one("teams-xyz")
    assert [e.entity_type for e in a.entities] == [e.entity_type for e in b.entities]
    assert [e.reason for e in a.entities] == [e.reason for e in b.entities]


# ── 9. Ranking is deterministic, interleaved, and starves nobody ───────


def test_projection_is_deterministic():
    _s1, a = _view(_ONE_OF_EACH)
    _s2, b = _view(_ONE_OF_EACH)
    assert a.to_dict() == b.to_dict()


def test_types_interleave_by_rank_instead_of_being_concatenated():
    """The ordering fix. Downstream truncates against a token budget, so a
    type-grouped list would mean whichever type came last got dropped first —
    a meeting could never survive a full inbox."""
    _state, view = _view(_ONE_OF_EACH)
    first_tier = [e.entity_type for e in view.entities[:len(EntityType)]]
    assert len(set(first_tier)) == len(EntityType), (
        "every type must contribute its best entity before any type contributes a second"
    )


def test_a_busy_inbox_cannot_starve_meetings_or_documents():
    """20 threads must not push the single meeting out of the head of the list."""
    noisy = list(_ONE_OF_EACH) + [
        rec(EntityType.THREAD, f"t-{i}", {"thread_id": f"t-{i}", "subject": f"s{i}"},
            offset=100 + i, tag=f"tn{i}")
        for i in range(20)
    ]
    _state, view = _view(noisy)
    head = [e.entity_type for e in view.entities[:len(EntityType)]]
    assert "meeting" in head and "document" in head


def test_rank_is_a_dense_global_sequence():
    _state, view = _view(_ONE_OF_EACH)
    assert [e.rank for e in view.entities] == list(range(len(view.entities)))


# ── 10. Budget: per-type cap still bounds the projection ───────────────


def test_the_per_type_cap_still_bounds_every_type():
    many = [
        rec(EntityType.MEETING, f"evt-{i}", {"title": f"m{i}"}, offset=i, tag=f"m{i}")
        for i in range(30)
    ] + [
        rec(EntityType.DOCUMENT, f"doc-{i}", {"title": f"d{i}"}, offset=50 + i, tag=f"d{i}")
        for i in range(30)
    ]
    _state, view = _view(many)
    assert len(_of_type(view, "meeting")) == 5      # _DEFAULT_TOP_N
    assert len(_of_type(view, "document")) == 5


def test_policy_can_tighten_the_cap_for_every_type():
    _state, view = _view(_ONE_OF_EACH, policy={"brief_max_items": 1})
    for entity_type in _types(view):
        assert len(_of_type(view, entity_type)) <= 1


# ── 11. Sparse / empty state ───────────────────────────────────────────


def test_an_empty_state_projects_an_empty_view_without_error():
    state, view = _view([])
    assert view.entities == [] and view.citations == {}
    assert view.counts == state.counts()


def test_types_with_no_entities_do_not_shift_anyone_elses_position():
    """zip_longest pads exhausted types with None; a type that contributes
    nothing must be invisible, not a gap."""
    _state, view = _view([
        rec(EntityType.MEETING, "evt-1", {"title": "Solo"}, offset=0, tag="m"),
    ])
    assert [e.entity_type for e in view.entities] == ["meeting"]
    assert [e.rank for e in view.entities] == [0]


# ── 7 (labels) — downstream label resolution uses normalized fields ────


def test_grounding_labels_resolve_from_normalized_state_fields():
    """The chat composition turns EntityRef.fields into a human label. Every
    projected type must yield something better than a bare uuid."""
    from navis.api.chat_router import _entity_label

    _state, view = _view(_ONE_OF_EACH)
    for e in view.entities:
        label = _entity_label(e)
        assert label and label != e.entity_id, f"{e.entity_type} has no readable label"


# ── 12. Organization isolation ─────────────────────────────────────────


def test_no_cross_org_entity_can_enter_the_scoped_view():
    """scope_for reads ONE already-scoped OrganizationalState; there is no seam
    through which another tenant's entity could arrive."""
    state_a = build_projection(ORG_A, [
        rec(EntityType.MEETING, "evt-1", {"title": "A meeting"}, offset=0, tag="ma",
            org_id=ORG_A),
    ])
    state_b = build_projection(ORG_B, [
        rec(EntityType.MEETING, "evt-1", {"title": "B meeting"}, offset=0, tag="mb",
            org_id=ORG_B),
    ])
    view_a = scope_for(state_a, MorningBrief(), None, as_of=state_a.as_of)
    view_b = scope_for(state_b, MorningBrief(), None, as_of=state_b.as_of)

    assert [e.fields["title"] for e in view_a.entities] == ["A meeting"]
    assert [e.fields["title"] for e in view_b.entities] == ["B meeting"]
    # Same natural key, different org → different entity id, no overlap anywhere.
    ids_a = {e.entity_id for e in view_a.entities}
    ids_b = {e.entity_id for e in view_b.entities}
    assert ids_a.isdisjoint(ids_b)
    assert set(view_a.citations).isdisjoint(set(view_b.citations))


# ── Temporal contract ──────────────────────────────────────────────────


def test_the_view_carries_the_state_snapshots_clock_and_watermark():
    state, view = _view(_ONE_OF_EACH)
    assert view.as_of == state.as_of
    assert view.state_watermark == str(state.watermark.last_memory_id)
    assert view.counts == state.counts()
    for e in view.entities:
        assert e.first_seen and e.last_seen  # per-entity temporal fields survive


# ── 14/15. Layer boundaries ────────────────────────────────────────────


def test_state_does_not_import_ml_or_reasoning():
    """Context is a read-side projection of State. It may not reach downstream."""
    import ast
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[2] / "navis" / "state"
    for py in root.rglob("*.py"):
        tree = ast.parse(py.read_text(encoding="utf-8"), filename=str(py))
        for node in ast.walk(tree):
            mods = []
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                mods = [node.module]
            for m in mods:
                assert not m.startswith(("navis.ml", "navis.reasoning", "navis.decisions")), (
                    f"{py.name} imports {m}"
                )


def test_scope_performs_no_io():
    """State -> scope_for -> ScopedView. No database, no connectors, no RAG."""
    import ast
    import pathlib

    path = pathlib.Path(__file__).resolve().parents[2] / "navis" / "state" / "v2" / "scope.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        mods = []
        if isinstance(node, ast.Import):
            mods = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods = [node.module]
        for m in mods:
            assert m.split(".")[0] not in {"sqlalchemy", "httpx", "redis", "asyncpg", "openai"}


# ══════════════════════════════════════════════════════════════════════════════
# CONTEXT AS A PIPELINE STAGE
# ══════════════════════════════════════════════════════════════════════════════
#
# `scope_for` had two callers — the `/context` endpoint and Ask Navis — and none
# between State and Intelligence, so Context was an API helper rather than a
# stage. `orchestration.brief` now builds it on the brief path and hands its
# citations to Intelligence. These pin what that stage is allowed to be.


def test_context_reads_state_and_never_a_provider():
    """Context SELECTS from what State already folded. It may not reach Gmail,
    Calendar, Slack or any connector — a stage that could fetch would be a second
    ingestion path with no Memory record behind it."""
    import ast
    import pathlib

    scope = pathlib.Path(__file__).resolve().parents[2] / "navis" / "state" / "v2" / "scope.py"
    tree = ast.parse(scope.read_text(encoding="utf-8"), filename=str(scope))

    forbidden_roots = {
        "navis.signals", "navis.ml", "navis.reasoning", "navis.decisions",
        "navis.intelligence", "navis.execution", "navis.learning", "navis.adaptive",
    }
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    leaks = {m for m in imported if ".".join(m.split(".")[:2]) in forbidden_roots}
    assert not leaks, f"Context reached outside State: {sorted(leaks)}"

    # No session, no client, no clock. Selection is a pure function of a snapshot.
    source = scope.read_text(encoding="utf-8")
    for banned in ("sqlalchemy", "AsyncSession", "httpx", "requests", "datetime.now("):
        assert banned not in source, f"Context gained `{banned}`"


def test_context_invents_no_entity_that_state_does_not_hold():
    """Every entity Context names must exist in the snapshot it was built from.
    A selection that could add an entity would be manufacturing organizational
    facts one layer below Intelligence."""
    state = build_projection(ORG_A, corpus(ORG_A))
    view = scope_for(state, MorningBrief(), None, as_of=state.as_of)

    in_state = {
        (entity_type, entity_id)
        for entity_type, entities in state.entities.items()
        for entity_id in entities
    }
    assert view.entities
    for ref in view.entities:
        assert (ref.entity_type, ref.entity_id) in in_state


def test_context_citations_are_stable_and_come_from_the_entity_itself():
    """Same snapshot in, byte-identical citations out — Intelligence versions an
    insight on the watermark, so a citation that moved between two projections of
    one snapshot would make the row unreproducible."""
    state = build_projection(ORG_A, corpus(ORG_A))
    a = scope_for(state, MorningBrief(), None, as_of=state.as_of)
    b = scope_for(state, MorningBrief(), None, as_of=state.as_of)
    assert a.to_dict() == b.to_dict()

    cited = 0
    for ref in a.entities:
        citation = a.citations.get(ref.entity_id)
        if citation is None:
            continue
        raw = state.entities[ref.entity_type][ref.entity_id].get("provenance", {})
        # A SUBSET of the entity's own provenance — bounded, never added to.
        assert set(citation.memory_ids) <= {str(m) for m in raw.get("memory_ids", [])}
        assert set(citation.signal_ids) <= {str(s) for s in raw.get("signal_ids", [])}
        cited += bool(citation.memory_ids)
    assert cited, "no entity carried a memory citation"


def test_reasoning_still_reads_the_full_state_not_the_context_selection():
    """CONTEXT IS A SELECTION, NOT A SUBSTITUTE. Reasoning is an exhaustive
    detector: feeding it a top-N slice would silently stop it finding
    obligations, which is the opposite of what a selection is for. Context feeds
    interpretation (Intelligence); State feeds detection (Reasoning)."""
    import inspect

    from navis.reasoning.v2 import run_reasoning_v2

    params = inspect.signature(run_reasoning_v2).parameters
    assert set(params) == {"state", "prior_state", "evidence"}, (
        "run_reasoning_v2 gained an input; a ScopedView must never be one"
    )

    # And the selection really is narrower than the snapshot it came from.
    state = build_projection(ORG_A, corpus(ORG_A))
    capped = scope_for(state, MorningBrief(), {"brief_max_items": 1}, as_of=state.as_of)
    assert len(capped.entities) < sum(len(e) for e in state.entities.values())


# ══════════════════════════════════════════════════════════════════════════════
# 7.5.7 — Context preserves State's entity graph
#
# Context ranked entities and dropped every EDGE between them, so the layers
# above it received a flat list and had to treat each entity as unrelated to
# every other. Live, that is why one slipping commitment and the cooling
# relationship OF THE SAME PERSON reached the brief as two independent cards
# while State had recorded `relationship --commitments--> commitment` all along.
# ══════════════════════════════════════════════════════════════════════════════


# The AJITESH case from the production audit, reduced to the records that
# produce its edges. `compute_links` joins on the values the extractors recorded
# — the commitment's `speaker_email` matching the relationship's `person_email`,
# and the meeting's attendee list — so these are the real join keys, not fixtures
# shaped to pass.
def _linked_counterparty(org_id=ORG_A):
    """Entity ids are org-scoped (`fakes.rec` hashes org_id), so this must be
    built per org — a shared module-level list would give both tenants ORG_A's
    ids and make an isolation assertion vacuous."""
    return [
        rec(EntityType.PERSON, "ajitesh@example.com",
            {"email": "ajitesh@example.com", "name": "Ajitesh"},
            offset=0, tag="p-aj", org_id=org_id),
        rec(EntityType.RELATIONSHIP, "ajitesh@example.com",
            {"person_email": "ajitesh@example.com", "direction": "outbound"},
            offset=1, tag="r-aj", org_id=org_id),
        rec(EntityType.COMMITMENT, "c-aj",
            {"text": "I'll come back to you on it.", "speaker_email": "ajitesh@example.com",
             "promised_at": "2026-06-01T12:00:00+00:00"},
            offset=2, tag="c-aj", org_id=org_id),
        rec(EntityType.MEETING, "m-aj",
            {"title": "Intro", "attendees": ["ajitesh@example.com"]},
            offset=3, tag="m-aj", org_id=org_id),
    ]


_LINKED_COUNTERPARTY = _linked_counterparty()


def test_context_carries_the_edges_state_persisted():
    """The four edges the audit found in production must survive the projection."""
    state, view = _view(_LINKED_COUNTERPARTY)
    edges = {
        (e.entity_type, link.role, link.entity_type)
        for e in view.entities
        for link in e.links
    }
    assert ("commitment", "owner", "person") in edges
    assert ("relationship", "commitments", "commitment") in edges
    assert ("relationship", "meetings", "meeting") in edges
    assert ("person", "relationships", "relationship") in edges


def test_every_link_names_both_a_type_and_an_id():
    """A link a consumer cannot resolve is not a link. `entity_id` alone does not
    say what it addresses, which is why the role map exists."""
    _state, view = _view(_LINKED_COUNTERPARTY)
    links = [link for e in view.entities for link in e.links]
    assert links, "fixture must produce edges"
    for link in links:
        assert link.entity_type and link.entity_id and link.role


def test_links_come_from_state_and_are_never_invented():
    """Every edge Context publishes must exist in the State entity it came from."""
    state, view = _view(_LINKED_COUNTERPARTY)
    for e in view.entities:
        raw = state.entities[e.entity_type][e.entity_id]
        for link in e.links:
            assert link.entity_id in raw["links"][link.role]


def test_an_entity_with_no_edges_reports_an_empty_tuple_not_a_guess():
    _state, view = _view([
        rec(EntityType.PERSON, "solo@example.com",
            {"email": "solo@example.com", "name": "Solo"}, offset=0, tag="solo"),
    ])
    assert [e.links for e in view.entities] == [()]


def test_the_link_role_map_covers_every_role_state_can_write():
    """THE PIN THAT KEEPS THE MAP CLOSED AND HONEST.

    `_LINK_TARGET_TYPE` is a read-side transcription of `folds.compute_links`,
    and an unrecognised role is DROPPED rather than guessed — so a link added to
    the fold without a matching entry here would silently vanish from Context
    instead of failing. Reading the roles out of the fold's own source is what
    makes that impossible: adding `_link(x, "teams", y)` breaks this test.
    """
    import inspect
    import re

    from navis.state.v2 import folds
    from navis.state.v2.scope import _LINK_TARGET_TYPE

    written = set(re.findall(r'_link\([^,]+,\s*"([^"]+)"', inspect.getsource(folds)))
    assert written, "could not read link roles from folds.compute_links"
    assert written <= set(_LINK_TARGET_TYPE), (
        f"roles written by compute_links with no target type: "
        f"{sorted(written - set(_LINK_TARGET_TYPE))}"
    )


def test_an_unknown_role_is_dropped_rather_than_typed_by_guess():
    from navis.state.v2.scope import _links

    class _Raw:
        links = {"teams": ["t-1"], "owner": ["p-1"]}

    roles = {link.role for link in _links(_Raw())}
    assert roles == {"owner"}


def test_links_are_deterministic_across_repeated_projections():
    state, first = _view(_LINKED_COUNTERPARTY)
    second = scope_for(state, MorningBrief(), None, as_of=state.as_of)
    assert [
        [(link.role, link.entity_type, link.entity_id) for link in e.links]
        for e in first.entities
    ] == [
        [(link.role, link.entity_type, link.entity_id) for link in e.links]
        for e in second.entities
    ]


def test_links_are_additive_to_the_wire_shape():
    """Every field the ScopedView published before is still there, in place."""
    _state, view = _view(_LINKED_COUNTERPARTY)
    entity = view.to_dict()["entities"][0]
    assert {
        "entity_type", "entity_id", "rank", "reason",
        "fields", "first_seen", "last_seen",
    } <= set(entity)
    assert isinstance(entity["links"], list)


def test_links_cannot_address_another_tenant():
    """Links are computed within one org's fold, so an id from another tenant can
    never appear — `compute_links` only ever joins entities of the state it was
    given."""
    _state_a, view_a = _view(_linked_counterparty(ORG_A), org_id=ORG_A)
    state_b, _view_b = _view(_linked_counterparty(ORG_B), org_id=ORG_B)
    b_ids = {eid for bucket in state_b.entities.values() for eid in bucket}
    a_link_targets = {link.entity_id for e in view_a.entities for link in e.links}
    assert a_link_targets, "fixture must produce edges"
    assert a_link_targets.isdisjoint(b_ids)
