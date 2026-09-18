"""Memory minimization FAILS CLOSED — P0-3E.

WHAT IS AT STAKE. A memory fact arrives at this function carrying
`memory_embeddings.source_text`, which for an email thread is the whole
rendered fact INCLUDING THE MESSAGE BODY. `minimize_memory_facts` is the only
thing between that string and `POST /developer/context`, which serves a model
running on someone else's machine. So every branch of it is a disclosure
decision, and a branch that passes the incoming text through unexamined is a
decision to publish an inbox.

THE BUG THESE PIN. Three paths used to do exactly that, all of them written as
availability trade-offs rather than as authorization:

    lookup raised            -> return the bundle unchanged  (raw body)
    memory_id returned no row -> keep the fact's text         (raw body)
    no minimal rendering      -> keep the fact's text         (raw body)

Each one is reached precisely when Navis CANNOT ESTABLISH what the string
contains, which is the one situation in which it must not be published. The
tests below are grouped by the four cases P0-3E names — allowed, denied, lookup
error, cross-organization — plus the two properties that make the fix safe to
ship: the bundle survives item-level withholding, and nothing leaks through
provenance or the logs.

NO SECOND AUTHORIZATION SYSTEM IS TESTED HERE, because none was added. The
authority is unchanged and lives above this function:

    credential -> ctx.organization_id -> view_memory -> layers -> org filtering

This function asks the existing org-scoped query what a fact may say, and
withholds when it cannot get an answer.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest

from navis.memory.envelope import EntityType
from navis.memory.lineage import FactDisclosure
from navis.ml.embeddings.service import (
    RAW_CONTENT_ENTITY_TYPES,
    memory_text,
    memory_text_minimal,
)
from navis.ml.rag.grounding import GroundedContext, Grounding
from navis.orchestration.context import WITHHELD_FACT_TEXT, minimize_memory_facts

ORG = uuid.UUID("dddd0000-0000-4000-8000-00000000000e")
OTHER_ORG = uuid.UUID("eeee0000-0000-4000-8000-00000000000f")
OCCURRED = datetime(2026, 9, 2, 8, 30, tzinfo=timezone.utc)

#: The string that must never reach a caller. A real email body, in the shape
#: `_thread_text` embeds it: the head, then `: `, then the snippet.
BODY = "Wire the $2.4M bridge to account 0117-882 before Friday, do not tell the board yet."
THREAD_PAYLOAD = {
    "subject": "Bridge financing",
    "participants": ["cfo@acme.com", "founder@acme.com"],
    "snippet": BODY,
}
THREAD_ID = uuid.UUID("11110000-0000-4000-8000-000000000001")
PERSON_ID = uuid.UUID("22220000-0000-4000-8000-000000000002")

#: What retrieval actually puts on a thread fact — computed by the SAME renderer
#: that produced every `source_text` in the corpus, so the fixture cannot drift
#: from production by being hand-written.
THREAD_SOURCE_TEXT = memory_text(
    type("R", (), {"payload": THREAD_PAYLOAD, "entity_type": "thread", "dedup_key": "t1"})()
)


class _DB:
    """A session the minimizer must not query except through the lookup."""

    async def execute(self, *a, **k):  # pragma: no cover
        raise AssertionError("minimize_memory_facts issued its own query")


def _thread_fact(memory_id: uuid.UUID = THREAD_ID) -> Grounding:
    """A memory fact as RETRIEVAL builds it: text IS the embedded source_text."""
    return Grounding(
        kind="memory", source=str(memory_id), text=THREAD_SOURCE_TEXT, occurred_at=OCCURRED
    )


def _bundle(*facts: Grounding) -> GroundedContext:
    return GroundedContext(
        org_id=str(ORG), question="what should I know about the bridge?",
        facts=list(facts), token_estimate=999, as_of=OCCURRED,
    )


def _disclosures(monkeypatch, answer):
    """Stand in for the org-scoped lookup. `answer` is a mapping or an exception.

    Patches the module attribute rather than the function's own reference:
    `minimize_memory_facts` imports it locally on every call, so this is the
    real seam and not a copy of one.
    """
    import navis.memory.lineage as lineage

    async def _lookup(db, org_id, memory_ids):
        assert org_id == ORG, "the lookup must be scoped to the caller's organization"
        if isinstance(answer, BaseException):
            raise answer
        return dict(answer)

    monkeypatch.setattr(lineage, "disclosure_by_memory_id", _lookup)


def _telemetry(monkeypatch) -> dict[str, float]:
    """Counters recorded during the call, independent of `metrics_enabled`."""
    from navis.platform.monitoring.metrics import metrics

    seen: dict[str, float] = {}

    def _incr(name, value=1.0, labels=None):
        seen[name] = seen.get(name, 0.0) + value

    monkeypatch.setattr(metrics, "incr", _incr)
    return seen


def _rendered(context: GroundedContext) -> str:
    """Everything a caller receives, as one string — the honest place to look.

    Asserting on `fact.text` alone would miss a body that escaped through
    `lineage`, `subject_id` or the question echo. `render()` is the exact bundle
    the Developer route serializes from.
    """
    return context.render() + repr([f.lineage for f in context.facts])


# ── A. ALLOWED MEMORY -> minimized content returned ─────────────────────────


async def test_an_allowed_thread_returns_the_organizational_fact_without_the_body(
    monkeypatch,
):
    """The subject line and participants ARE the organizational fact. The body
    is content that rode along because similarity search wanted it."""
    _telemetry(monkeypatch)
    _disclosures(monkeypatch, {
        THREAD_ID: FactDisclosure("thread", THREAD_ID, THREAD_PAYLOAD)
    })

    out = await minimize_memory_facts(_DB(), ORG, _bundle(_thread_fact()))

    fact = out.facts[0]
    assert fact.text == 'Email thread "Bridge financing" with cfo@acme.com, founder@acme.com'
    assert BODY not in _rendered(out), "the message body reached the caller"
    assert fact.text != WITHHELD_FACT_TEXT, "an authorized fact must not be withheld"


async def test_an_allowed_fact_gains_its_canonical_subject_read_from_the_row(
    monkeypatch,
):
    """P0-3C's subject, filled from the same round trip — READ, never inferred."""
    _telemetry(monkeypatch)
    _disclosures(monkeypatch, {
        THREAD_ID: FactDisclosure("thread", THREAD_ID, THREAD_PAYLOAD)
    })

    out = await minimize_memory_facts(_DB(), ORG, _bundle(_thread_fact()))

    assert out.facts[0].subject_type == "thread"
    assert out.facts[0].subject_id == str(THREAD_ID)


async def test_a_metadata_only_fact_is_unchanged_and_still_attributed(monkeypatch):
    """Eight of the eleven renderers carry no body, so minimization is a no-op on
    them. It must stay a no-op: withholding a person's name would be the
    over-correction, and the fix is not allowed to cost the bundle facts it is
    authorized to carry."""
    _telemetry(monkeypatch)
    payload = {"email": "cfo@acme.com", "name": "Dana Reed"}
    text = memory_text(
        type("R", (), {"payload": payload, "entity_type": "person", "dedup_key": "p"})()
    )
    _disclosures(monkeypatch, {PERSON_ID: FactDisclosure("person", PERSON_ID, payload)})

    out = await minimize_memory_facts(
        _DB(), ORG, _bundle(Grounding("memory", str(PERSON_ID), text))
    )

    assert out.facts[0].text == text
    assert out.facts[0].subject_id == str(PERSON_ID)


async def test_a_fact_with_no_recorded_entity_stays_honestly_unattributed(monkeypatch):
    """`subject_id=None` is a real answer. Inventing one from the payload would be
    the fabricated attribution the subject field exists to remove."""
    _telemetry(monkeypatch)
    _disclosures(monkeypatch, {
        THREAD_ID: FactDisclosure("thread", None, THREAD_PAYLOAD)
    })

    out = await minimize_memory_facts(_DB(), ORG, _bundle(_thread_fact()))

    assert out.facts[0].subject_type == "thread"
    assert out.facts[0].subject_id is None


@pytest.mark.parametrize("entity_type", sorted(RAW_CONTENT_ENTITY_TYPES - {"thread"}))
async def test_commitment_and_decision_text_follows_THE_EXISTING_POLICY(
    monkeypatch, entity_type
):
    """NOT minimized, and that is the existing decision rather than an oversight.

    "Somebody committed to something" with the something removed is not a
    smaller fact, it is a useless one — and a caller holding `view_memory` on
    this organization is authorized to read it. P0-3E was told not to create a
    second authorization system, so it does not re-decide this here.
    """
    _telemetry(monkeypatch)
    payload = {"text": "We will refund Northwind in full.", "speaker_email": "cfo@acme.com"}
    text = memory_text(
        type("R", (), {"payload": payload, "entity_type": entity_type, "dedup_key": "c"})()
    )
    mid = uuid.uuid4()
    _disclosures(monkeypatch, {mid: FactDisclosure(entity_type, mid, payload)})

    out = await minimize_memory_facts(
        _DB(), ORG, _bundle(Grounding("memory", str(mid), text))
    )

    assert "refund Northwind in full" in out.facts[0].text
    assert out.facts[0].text != WITHHELD_FACT_TEXT


# ── B. DENIED MEMORY -> content withheld ────────────────────────────────────


async def test_a_memory_id_the_org_scoped_lookup_did_not_return_is_withheld(
    monkeypatch,
):
    """NO ROW IS A DENIAL, and it used to read as a pass.

    The lookup filters on `organization_id`, so a memory_id it does not return
    is one this organization may not read — another tenant's row, or one deleted
    between the vector search and this read. Nothing here is authorized to say
    what its text is, so nothing here says it.
    """
    seen = _telemetry(monkeypatch)
    _disclosures(monkeypatch, {})  # the query ran; it returned no row for this id

    out = await minimize_memory_facts(_DB(), ORG, _bundle(_thread_fact()))

    assert out.facts[0].text == WITHHELD_FACT_TEXT
    assert BODY not in _rendered(out)
    assert seen["context.minimize_memory.withheld"] == 1


async def test_a_withheld_fact_keeps_the_provenance_the_caller_already_holds(
    monkeypatch,
):
    """Withheld is not deleted. The caller can still see that Navis holds a fact
    here, cite it, and fetch it through a route that authorizes it — which is the
    difference between a disclosure decision and a blank organization."""
    _telemetry(monkeypatch)
    fact = Grounding(
        kind="memory", source=str(THREAD_ID), text=THREAD_SOURCE_TEXT,
        occurred_at=OCCURRED, lineage=("signal:abc", "integration:def"),
    )
    _disclosures(monkeypatch, {})

    out = await minimize_memory_facts(_DB(), ORG, _bundle(fact))

    kept = out.facts[0]
    assert kept.kind == "memory"
    assert kept.source == str(THREAD_ID)
    assert kept.occurred_at == OCCURRED
    assert kept.lineage == ("signal:abc", "integration:def")
    assert len(out.facts) == 1, "the fact must not be dropped"


async def test_a_withheld_fact_does_not_acquire_a_subject_it_could_not_read(
    monkeypatch,
):
    """`subject_type`/`subject_id` are READ FROM the row the lookup failed to
    return. Filling them from anywhere else would be fabricated authorization."""
    _telemetry(monkeypatch)
    _disclosures(monkeypatch, {})

    out = await minimize_memory_facts(_DB(), ORG, _bundle(_thread_fact()))

    assert out.facts[0].subject_type is None
    assert out.facts[0].subject_id is None


async def test_a_source_that_is_not_a_memory_id_is_withheld(monkeypatch):
    """Unlookupable is unverifiable. This used to skip the id and then keep the
    text, so a fact with a malformed source was published verbatim."""
    seen = _telemetry(monkeypatch)
    _disclosures(monkeypatch, {})

    out = await minimize_memory_facts(
        _DB(), ORG, _bundle(Grounding("memory", "not-a-uuid", THREAD_SOURCE_TEXT))
    )

    assert out.facts[0].text == WITHHELD_FACT_TEXT
    assert BODY not in _rendered(out)
    assert seen["context.minimize_memory.withheld"] == 1


def test_every_entity_type_has_a_minimal_renderer():
    """So failing closed on an UNKNOWN type costs this build nothing today.

    An entity type absent from `_RENDERERS` is withheld, because its
    `source_text` came from `_generic_text` — the payload scan that reads
    `text`, `content`, `description` and `summary`, any of which may be a raw
    body. Withholding is the right default for a type nobody has audited.

    Asserting the set is currently EMPTY is what makes the safe default also
    the free one: every type Memory can produce has an audited renderer, so no
    fact in this corpus loses its text to the new branch. It fails the build
    when a twelfth entity type is added without one — at which point that
    type's facts lose their text instead of silently publishing it, and this
    line says which type and why.
    """
    from navis.ml.embeddings.service import _RENDERERS

    missing = sorted({e.value for e in EntityType} - set(_RENDERERS))
    assert missing == [], (
        f"these entity types have no minimal renderer, so their facts will be "
        f"withheld from the Developer bundle: {missing} — add one to _RENDERERS "
        f"(and audit it for raw body content) or accept the withholding"
    )


#: One payload per entity type, keyed as the EXTRACTOR that produces it writes
#: it — `person_email` for a relationship, not `counterparty_email`. Written out
#: per type rather than as one union dict on purpose: a union hides a renderer
#: reading a key nothing ever writes, which is the failure the test below is for.
_RENDERABLE_PAYLOADS: dict[str, dict] = {
    "person": {"name": "Dana Reed", "email": "cfo@acme.com", "role": "from"},
    "relationship": {"person_email": "cfo@acme.com", "direction": "inbound", "via": "email"},
    "company": {"domain": "acme.com", "name": "Acme"},
    "thread": THREAD_PAYLOAD,
    "commitment": {"text": "We will refund Northwind.", "speaker_email": "cfo@acme.com"},
    "decision": {"text": "We are refunding Northwind.", "subject": "Northwind"},
    "meeting": {"title": "Bridge review", "attendees": ["cfo@acme.com"],
                "starts_at": "2026-09-02T08:30:00", "organizer": "founder@acme.com"},
    "document": {"title": "Q3 plan", "owners": ["founder@acme.com"]},
    "project": {"name": "bridge.pdf", "owners": ["founder@acme.com"]},
    "channel": {"name": "finance", "visibility": "private"},
    "customer": {"name": "Northwind"},
}


def test_the_renderable_payload_fixtures_cover_every_entity_type():
    """So the parametrization below cannot quietly stop covering a new type."""
    assert sorted(_RENDERABLE_PAYLOADS) == sorted(e.value for e in EntityType)


@pytest.mark.parametrize("entity_type", sorted(_RENDERABLE_PAYLOADS))
def test_a_renderable_payload_minimizes_rather_than_withholding(entity_type):
    """The other half of the invariant above: a renderer that EXISTS is also
    SATISFIABLE by the payload its extractor writes.

    A renderer present but never satisfiable would withhold every fact of its
    type — failing closed, so safe, but silently useless — and the membership
    assertion above would still pass.
    """
    assert memory_text_minimal(entity_type, _RENDERABLE_PAYLOADS[entity_type]) is not None


# ── C. DISCLOSURE LOOKUP ERROR -> content withheld, no raw body ─────────────


async def test_a_failed_lookup_withholds_every_memory_text(monkeypatch):
    """THE FAIL-OPEN THIS PHASE CLOSED.

    The lookup is what tells this function which facts carry a body. Without it
    the answer is unknown for every memory fact, so every memory fact is
    withheld. Previously this path returned the bundle UNCHANGED — the embedded
    text, bodies included — and recorded a counter about it.
    """
    seen = _telemetry(monkeypatch)
    _disclosures(monkeypatch, RuntimeError("connection reset by peer"))

    out = await minimize_memory_facts(
        _DB(), ORG,
        _bundle(_thread_fact(), _thread_fact(uuid.uuid4())),
    )

    assert [f.text for f in out.facts] == [WITHHELD_FACT_TEXT, WITHHELD_FACT_TEXT]
    assert BODY not in _rendered(out), "a lookup failure published the message body"
    assert seen["context.minimize_memory.failed"] == 1
    assert seen["context.minimize_memory.withheld"] == 2


async def test_a_failed_lookup_does_not_blank_the_rest_of_the_bundle(monkeypatch):
    """Item-level, as P0-3E requires. State, Reasoning, Intelligence and Learning
    are DERIVED claims, not embedded source, so they are not this function's
    business and are passed through untouched. A caller that lost the memory
    layer still has an organization to reason about."""
    _telemetry(monkeypatch)
    _disclosures(monkeypatch, RuntimeError("statement timeout"))
    derived = [
        Grounding("state", "e1", "Dana Reed — relationship cooling", lineage=("memory:m9",)),
        Grounding("reasoning", "s1", "priority_load is high"),
        Grounding("learning", "l1", "replies within 2 days 71% of the time"),
        Grounding("intelligence", "i1", "renewal risk: Northwind"),
    ]

    out = await minimize_memory_facts(
        _DB(), ORG, _bundle(_thread_fact(), *derived)
    )

    assert len(out.facts) == 5, "the bundle was blanked instead of the item"
    assert out.facts[1:] == derived, "a derived layer was altered by memory minimization"
    assert out.facts[0].text == WITHHELD_FACT_TEXT


async def test_one_unresolvable_fact_does_not_withhold_its_neighbours(monkeypatch):
    """Per item, in both directions: the fact that resolved keeps its minimized
    text while the one that did not is withheld."""
    seen = _telemetry(monkeypatch)
    other = uuid.UUID("33330000-0000-4000-8000-000000000003")
    _disclosures(monkeypatch, {
        THREAD_ID: FactDisclosure("thread", THREAD_ID, THREAD_PAYLOAD)
    })

    out = await minimize_memory_facts(
        _DB(), ORG, _bundle(_thread_fact(), _thread_fact(other))
    )

    assert out.facts[0].text == 'Email thread "Bridge financing" with cfo@acme.com, founder@acme.com'
    assert out.facts[1].text == WITHHELD_FACT_TEXT
    assert seen["context.minimize_memory.withheld"] == 1
    assert BODY not in _rendered(out)


async def test_an_unrenderable_payload_is_withheld_rather_than_passed_through(
    monkeypatch,
):
    """The row was read, and it cannot be re-rendered — a payload that is absent
    or not an object. The embedded text is still whatever was embedded, so it is
    still unverified, so it is still withheld."""
    seen = _telemetry(monkeypatch)
    _disclosures(monkeypatch, {THREAD_ID: FactDisclosure("thread", THREAD_ID, None)})

    out = await minimize_memory_facts(_DB(), ORG, _bundle(_thread_fact()))

    assert out.facts[0].text == WITHHELD_FACT_TEXT
    assert BODY not in _rendered(out)
    assert seen["context.minimize_memory.withheld"] == 1


async def test_the_logs_do_not_print_the_protected_content(monkeypatch, caplog):
    """A log line that names the withheld item is one edit away from carrying it,
    so the warning carries a COUNT and the organization id."""
    import logging

    _telemetry(monkeypatch)
    _disclosures(monkeypatch, RuntimeError("connection reset by peer"))

    with caplog.at_level(logging.DEBUG, logger="navis.orchestration.context"):
        await minimize_memory_facts(_DB(), ORG, _bundle(_thread_fact()))

    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert BODY not in logged, "the withheld body was written to the log"
    assert "Bridge financing" not in logged, "the subject line was written to the log"
    assert "withheld 1/1 memory fact texts" in logged
    assert str(ORG) in logged


async def test_the_token_estimate_is_recomputed_after_withholding(monkeypatch):
    """The facts changed length, so the figure has to change with them — it is
    the RENDERED cost of what actually goes out, and a stale one would let the
    Developer boundary report a budget it is not spending."""
    _telemetry(monkeypatch)
    _disclosures(monkeypatch, RuntimeError("connection reset by peer"))
    before = _bundle(_thread_fact())

    out = await minimize_memory_facts(_DB(), ORG, before)

    assert out.token_estimate != before.token_estimate
    assert out.token_estimate > 0


# ── D. CROSS-ORGANIZATION MEMORY -> 0 rows ──────────────────────────────────


async def test_the_disclosure_query_filters_on_organization_id():
    """ISOLATION IS A PROPERTY OF THE STATEMENT, not of the caller.

    Reads the compiled SQL of the REAL lookup rather than trusting a stub, so
    deleting the org predicate fails here. That predicate is what makes the
    test below true: a memory_id belonging to another tenant matches no row, the
    minimizer sees no disclosure, and the fact is withheld.
    """
    from sqlalchemy.dialects import postgresql

    from navis.memory.lineage import disclosure_by_memory_id

    captured: dict = {}

    class _Recording:
        async def execute(self, statement):
            captured["sql"] = str(
                statement.compile(
                    dialect=postgresql.dialect(),
                    compile_kwargs={"literal_binds": True},
                )
            )

            class _R:
                @staticmethod
                def all():
                    return []  # exactly what Postgres returns for another tenant's id

            return _R()

    out = await disclosure_by_memory_id(_Recording(), ORG, [THREAD_ID])

    assert out == {}, "a cross-organization id must resolve to no disclosure"
    sql = captured["sql"]
    assert "memory_records.organization_id = " in sql
    assert str(ORG) in sql, "the caller's organization must be bound into the query"
    assert str(OTHER_ORG) not in sql


async def test_another_tenants_memory_id_is_withheld_not_rendered(monkeypatch):
    """The end-to-end consequence of the filter above, at the boundary that
    matters: 0 rows -> no disclosure -> no text."""
    seen = _telemetry(monkeypatch)

    class _Empty:
        """A session that answers as Postgres would for another tenant's id.

        NOT stubbing the lookup: `disclosure_by_memory_id` runs for real here,
        builds its own org-scoped statement and interprets the empty result. So
        this covers the caller's handling of "no row", which is the half the
        compiled-SQL test above cannot reach.
        """

        async def execute(self, statement):
            class _R:
                @staticmethod
                def all():
                    return []

            return _R()

    out = await minimize_memory_facts(_Empty(), ORG, _bundle(_thread_fact()))

    assert out.facts[0].text == WITHHELD_FACT_TEXT
    assert BODY not in _rendered(out)
    assert seen["context.minimize_memory.withheld"] == 1


# ── The asymmetry stays deliberate ──────────────────────────────────────────


async def test_nothing_runs_when_there_is_no_memory_layer(monkeypatch):
    """A bundle of derived facts needs no lookup, so it pays for none."""
    calls: list[int] = []

    import navis.memory.lineage as lineage

    async def _lookup(db, org_id, memory_ids):  # pragma: no cover
        calls.append(1)
        return {}

    monkeypatch.setattr(lineage, "disclosure_by_memory_id", _lookup)
    bundle = _bundle(Grounding("state", "e1", "Dana Reed — relationship cooling"))

    out = await minimize_memory_facts(_DB(), ORG, bundle)

    assert out is bundle
    assert calls == []


def test_the_application_path_still_composes_without_minimization():
    """`/chat` MUST be unaffected. The flag is off by default and `/chat` does not
    set it — a founder reading their own organization in the Navis UI already has
    the mail, and the asymmetry is the point."""
    import ast
    import inspect
    import textwrap

    import navis.api.chat_router as chat
    from navis.orchestration.context import build_grounded_context

    assert (
        inspect.signature(build_grounded_context).parameters["minimize_memory"].default
        is False
    )
    tree = ast.parse(textwrap.dedent(inspect.getsource(chat._grounded_answer)))
    calls = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "build_grounded_context"
    ]
    assert len(calls) == 1
    assert "minimize_memory" not in {kw.arg for kw in calls[0].keywords}, (
        "the Application path acquired the Developer boundary's minimization"
    )
