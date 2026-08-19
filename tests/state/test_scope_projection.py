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
from tests.state.fakes import ORG_A, ORG_B, rec


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
    tree = ast.parse(path.read_text(), filename=str(path))

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
        tree = ast.parse(py.read_text(), filename=str(py))
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
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        mods = []
        if isinstance(node, ast.Import):
            mods = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods = [node.module]
        for m in mods:
            assert m.split(".")[0] not in {"sqlalchemy", "httpx", "redis", "asyncpg", "openai"}
