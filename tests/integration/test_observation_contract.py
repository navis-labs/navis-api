"""INTEGRATION — Execution -> Observation -> Learning -> Intelligence / ML, on real PostgreSQL.

Observation is the only durable record that an action happened. The execution row
says what Navis intended and what state it reached; the outcome log is what
Learning and any audit actually read. So the questions that matter here are not
"does it write a row" but:

    can an outcome be LOST?                 (relay delivery semantics)
    can it be DOUBLE-COUNTED?               (dedupe under replay + retry)
    can it be recorded BEFORE the fact?     (ordering / durability)
    does it say WHY, or only that it failed? (outcome semantics)
    does a human's judgement reach it?      (feedback boundary)
    can org A's outcome land on org B?      (isolation)
    does it carry a credential?             (secret safety)

And, one hop further down, the questions Learning answers with them:

    does the outcome become a LEARNING EVENT, with its lineage intact?
    is a human's rejection learned as a decisive failure, or shrugged off?
    does a CHANNEL fault get blamed on the reasoning that asked for the send?
    does replaying the same outcome inflate what the org believes?
    can org A's learning move org B's aggregates?

And the last hop, which is what makes the learning usable rather than merely
stored:

    can the ML layer READ what was learned, through Learning's contract?
    is a delivery fault ever presented to a model as a reasoning failure?
    can one org's learned feature reach another org's context?
    does a replayed event change what the model would be told?

"Observation recorded it", "Learning learned the right thing from it" and "a
model can be grounded on it" are three different claims, and only the third
closes the loop.

Everything below is the real implementation. Nothing about Execution,
Observation or Learning is stubbed except, in one test, a deliberately failing
Observation handler — which is the failure this file exists to characterise.
"""
from __future__ import annotations

import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("NAVIS_TEST_DATABASE_URL"),
    reason="set NAVIS_TEST_DATABASE_URL (disposable Postgres) to run these",
)

ORG_A = uuid.UUID("cccccccc-0000-4000-8000-00000000000a")
ORG_B = uuid.UUID("cccccccc-0000-4000-8000-00000000000b")
NOW = datetime(2026, 8, 19, 10, 0, tzinfo=timezone.utc)

# The name a database must carry before this file is allowed to drop it.
_DISPOSABLE_PREFIX = "navis_test"


def _assert_disposable_database(url: str) -> None:
    """Refuse any target that is not an explicitly disposable database.

    This checks the DATABASE NAME, not the host. The fixture below calls
    `Base.metadata.drop_all`, and what makes that safe is that the target holds
    nothing except this run's fixtures — which is a property of the database,
    not of where it is hosted.

    The previous guard was `"neon.tech" not in url`, which got that backwards in
    both directions: it refused a throwaway database that happens to live on a
    managed host, while happily dropping a local `navis` or `postgres` that may
    be someone's working copy. A name allowlist refuses BOTH — including the
    live `neondb` this project actually uses.
    """
    from urllib.parse import urlsplit

    name = urlsplit(url).path.lstrip("/")
    assert name.startswith(_DISPOSABLE_PREFIX), (
        f"refusing to drop_all on database {name!r}: point "
        f"NAVIS_TEST_DATABASE_URL at a disposable database named "
        f"{_DISPOSABLE_PREFIX}*"
    )


# ── Infrastructure ─────────────────────────────────────────────────────


@pytest.fixture()
async def engine():
    """A disposable engine on an EMPTY schema.

    Every test still starts against empty tables — that has not changed. What
    changed is how they are emptied: `drop_all` + `create_all` per test issued
    roughly five hundred DDL round trips (51 tables, twice, plus indexes), which
    against a hosted database cost over four minutes PER TEST and made this file
    a two-hour run nobody would sit through.

    `create_all` is checkfirst, so it is real work exactly once; after that a
    single `TRUNCATE ... CASCADE` over every table resets the whole schema in one
    statement. CASCADE because these tables are a foreign-key graph and
    truncating them one at a time in the wrong order is what makes people reach
    for drop_all in the first place.
    """
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine

    import navis.registry  # noqa: F401 — registers models AND every subscriber
    from navis.platform.database import Base

    url = os.environ["NAVIS_TEST_DATABASE_URL"]
    _assert_disposable_database(url)
    eng = create_async_engine(url)
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        # `memory_embeddings.embedding_vec` exists ONLY in migration 0038 — the
        # ORM model stores vectors as JSONB and the typed pgvector column is
        # added by raw DDL — so `create_all` cannot produce it and the retrieval
        # path fails with "column e.embedding_vec does not exist". That is a
        # divergence between the fixture schema and the real one, not a fact
        # about the code under test, so the fixture mirrors the migration rather
        # than the tests avoiding retrieval.
        from navis.ml.retrieval.vector_search import ANN_DIM

        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.execute(
            text(
                f"ALTER TABLE memory_embeddings "
                f"ADD COLUMN IF NOT EXISTS embedding_vec vector({ANN_DIM})"
            )
        )
        tables = ", ".join(f'"{t.name}"' for t in Base.metadata.sorted_tables)
        await conn.execute(text(f"TRUNCATE {tables} RESTART IDENTITY CASCADE"))
    yield eng
    await eng.dispose()


@pytest.fixture()
async def Session(engine, monkeypatch):
    from sqlalchemy.ext.asyncio import async_sessionmaker

    maker = async_sessionmaker(engine, expire_on_commit=False)

    @asynccontextmanager
    async def _scope():
        async with maker() as s:
            try:
                yield s
                await s.commit()
            except Exception:
                await s.rollback()
                raise

    # EVERY HOP OF THE SYNCHRONOUS CHAIN, because relaying one execution event
    # reaches four layers deep and each opens its OWN session:
    #
    #   relay_pass  --ACTION_EXECUTED-->  observation.on_action_executed
    #               --OUTCOME_OBSERVED->  learning.on_outcome_observed
    #               --LEARNING_UPDATED->  intelligence.on_learning_updated
    #
    # Each of these imported `session_scope` BY VALUE at module import, so
    # patching `navis.platform.database.session_scope` alone misses them — and a
    # missed hop silently opens a connection to whatever the settings point at,
    # which is the production database. Patching the source module too is belt
    # and braces for anything resolving it lazily.
    for target in (
        "navis.observation.subscribers.session_scope",
        "navis.learning.subscribers.session_scope",
        "navis.intelligence.subscribers.session_scope",
        "navis.ml.embeddings.subscribers.session_scope",
        "navis.platform.database.session_scope",
    ):
        monkeypatch.setattr(target, _scope, raising=False)

    # The Redis mirror is a documented best-effort, non-authoritative copy
    # ("losing the tail of this stream loses no truth"). Redis is not running
    # here, and letting each nested publish wait on it would measure the sandbox
    # rather than the contract.
    from navis.platform.events import event_bus

    monkeypatch.setattr(event_bus, "_mirror_to_redis", False, raising=False)
    return maker


@pytest.fixture()
async def orgs(Session):
    from navis.identity.models import Organization, User
    from navis.organization.contract import Plan
    from navis.organization.service import OrganizationService

    async with Session() as s:
        for oid, name in ((ORG_A, "Alpha"), (ORG_B, "Beta")):
            s.add(Organization(id=oid, name=name, slug=f"{name.lower()}-{uuid.uuid4().hex[:6]}"))
        user = User(email="founder@alpha.test")
        s.add(user)
        await s.flush()
        for oid in (ORG_A, ORG_B):
            await OrganizationService(s).bootstrap(oid, plan=Plan.ENTERPRISE)
        await s.commit()
        return user.id


class _Provider:
    """A provider that succeeds. Not a stub of Execution — a connector."""

    name = "gmail"
    action_type = "send_email"

    def __init__(self) -> None:
        self._sender = self._send

    async def _send(self, payload):
        return "gmail-1"

    async def validate(self, payload):
        return None

    async def execute(self, payload):
        from navis.execution.contract import ProviderReference, ProviderResult

        return ProviderResult(ok=True, reference=ProviderReference("gmail", "gmail-1"))

    async def status(self, reference):
        return "sent"


async def _approved(Session, org_id, user_id, payload=None):
    from navis.execution.service import ExecutionService

    async with Session() as s:
        req, _ = await ExecutionService(s).request(
            org_id, uuid.uuid4(), "send_email",
            payload or {"to": "a@b.com", "subject": "s", "body": "b"},
            reasoning_signal_ids=["sig-1", "sig-2"], reasoning_version=7,
            state_watermark="w-42",
        )
        await ExecutionService(s).approve(req.id, user_id)
        await s.commit()
        return req.id, req.decision_id


_DEFAULT = object()


async def _drain(Session, providers=_DEFAULT, **kw):
    """`providers={}` means "nothing is wired", which is a REAL state.

    Sentinel rather than `providers or default`, because an empty registry is
    falsy and would silently fall back to a working provider — the exact trap
    `get_provider` documents ("`is None`, NOT truthiness").
    """
    from navis.execution.repository import ExecutionRepository
    from navis.execution.worker import execute_pass

    @asynccontextmanager
    async def unit():
        async with Session() as s:
            yield ExecutionRepository(s)
            await s.commit()

    registry = {"send_email": _Provider()} if providers is _DEFAULT else providers
    return await execute_pass(unit=unit, providers=registry, limit=10, **kw)


async def _relay(Session):
    from navis.execution.repository import ExecutionRepository
    from navis.execution.worker import relay_pass

    async with Session() as s:
        published = await relay_pass(ExecutionRepository(s))
        await s.commit()
    return published


async def _unpublished_event_types(Session, execution_id) -> list[str]:
    """Which of this execution's outbox rows are still awaiting delivery."""
    from navis.execution.models import ExecutionOutbox
    from sqlalchemy import select

    async with Session() as s:
        rows = (await s.scalars(
            select(ExecutionOutbox).where(
                ExecutionOutbox.execution_id == execution_id,
                ExecutionOutbox.published_at.is_(None),
            )
        )).all()
    return [r.event_type for r in rows]


async def _outcomes(Session, org_id=None, execution_id=None):
    from navis.observation.models import Outcome
    from sqlalchemy import select

    stmt = select(Outcome)
    if org_id is not None:
        stmt = stmt.where(Outcome.organization_id == org_id)
    if execution_id is not None:
        stmt = stmt.where(Outcome.execution_id == execution_id)
    async with Session() as s:
        return (await s.scalars(stmt.order_by(Outcome.created_at.asc()))).all()


async def _learning_events(Session, org_id):
    from navis.learning.models import LearningEvent
    from sqlalchemy import select

    async with Session() as s:
        return (await s.scalars(
            select(LearningEvent)
            .where(LearningEvent.organization_id == org_id)
            .order_by(LearningEvent.occurred_at.asc(), LearningEvent.id.asc())
        )).all()


async def _snapshot(Session, org_id):
    """Learning read through its OWN contract — the same call Intelligence makes."""
    from navis.learning.metrics import learning_snapshot

    async with Session() as s:
        return await learning_snapshot(s, org_id)


async def _agg(Session, org_id, dimension, key):
    """One aggregate counter, or a clear failure.

    A missing counter is itself a defect worth naming — an absent row and a row
    of zeroes mean different things — so this asserts rather than returning None
    and letting the caller trip over an AttributeError.
    """
    from navis.learning.models import AggregateDimension

    snap = await _snapshot(Session, org_id)
    wanted = dimension.value if isinstance(dimension, AggregateDimension) else dimension
    for a in snap.aggregates:
        if a.dimension == wanted and a.key == key:
            return a
    raise AssertionError(
        f"no {wanted} aggregate for {key!r} in org {org_id}; "
        f"present: {sorted((a.dimension, a.key) for a in snap.aggregates)}"
    )


# ══ LINEAGE ════════════════════════════════════════════════════════════


async def test_an_executed_action_reaches_observation_with_complete_lineage(
    Session, orgs
):
    """Every provenance question must be answerable from the outcome row alone.

    Observation is what Learning reads, and Learning is forbidden from reading
    any other module's tables — so a field missing here is a question that can
    never be answered downstream.
    """
    from navis.observation.models import OutcomeType

    user_id = orgs
    execution_id, decision_id = await _approved(Session, ORG_A, user_id)
    await _drain(Session)
    await _relay(Session)

    rows = await _outcomes(Session, execution_id=execution_id)
    assert len(rows) == 1
    o = rows[0]
    assert o.outcome_type == OutcomeType.ACTION_EXECUTED.value  # what happened
    assert o.organization_id == ORG_A                            # whose
    assert o.execution_id == execution_id                        # which action
    assert o.decision_id == decision_id                          # what was decided
    assert o.reasoning_signal_ids == ["sig-1", "sig-2"]           # why
    assert o.reasoning_version == 7
    assert o.state_watermark == "w-42"                           # against what world
    assert o.provider_reference == {"provider": "gmail", "external_id": "gmail-1"}
    assert o.payload["action_type"] == "send_email"
    assert o.occurred_at is not None and o.recorded_at is not None
    assert o.source_event_id, "no provenance for the event that produced this"


# ══ THE RELAY CAN NO LONGER LOSE AN OUTCOME ════════════════════════════


async def test_a_failing_observation_handler_leaves_the_event_unpublished(
    Session, orgs, monkeypatch
):
    """THE DEFECT THIS FILE EXISTS FOR.

    `relay_pass` ignored the return value of `publish()` and marked every row
    published unconditionally. `Delivery` exists precisely so a relay can tell
    "every handler ran" from "publish() was called", and both sibling relays
    (signals, memory) already used it — this one was missed.

    So a transient Observation failure — a database timeout, a dropped
    connection — marked the ACTION_EXECUTED row published and the outcome was
    gone forever, with only a best-effort Redis dead letter behind it. The
    execution row still said EXECUTED, so nothing surfaced the loss.

    What must hold: the row stays unpublished, and the NEXT pass records it.
    """
    import navis.observation.subscribers as obs

    user_id = orgs
    execution_id, _ = await _approved(Session, ORG_A, user_id)
    await _drain(Session)

    real_capture = obs._capture
    calls = {"n": 0}

    async def _flaky(event, outcome_type):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("observation database timeout")
        await real_capture(event, outcome_type)

    monkeypatch.setattr(obs, "_capture", _flaky)

    # One execution emits several events (REQUESTED, APPROVED, EXECUTING,
    # EXECUTED); only the EXECUTED one has a failing handler here, so the others
    # publish normally. The claim under test is about THAT row.
    await _relay(Session)
    assert await _unpublished_event_types(Session, execution_id) == ["ActionExecuted"], (
        "the row whose handler failed was marked published anyway"
    )
    assert await _outcomes(Session, execution_id=execution_id) == []

    # The execution itself is untouched — Observation failing must never rewrite
    # history about whether the side effect happened.
    from navis.execution.models import ExecutionRequest
    from navis.execution.status import ExecutionStatus

    async with Session() as s:
        row = await s.get(ExecutionRequest, execution_id)
    assert row.status == ExecutionStatus.EXECUTED.value
    assert row.provider_reference == {"provider": "gmail", "external_id": "gmail-1"}

    # The retry succeeds, and produces exactly ONE outcome.
    published = await _relay(Session)
    assert published >= 1
    rows = await _outcomes(Session, execution_id=execution_id)
    assert len(rows) == 1, f"retry produced {len(rows)} outcomes"


async def test_a_repeated_relay_never_double_counts_an_outcome(Session, orgs):
    """At-least-once delivery, exactly-once effect.

    Two independent guards must hold together: the deterministic
    `outcome_id = uuid5(source_event_id, outcome_type)` collapses the insert, and
    the observation inbox claims the event id once per consumer. Relaying the
    same event repeatedly must converge on one row.
    """
    from navis.execution.models import ExecutionOutbox
    from sqlalchemy import update

    user_id = orgs
    execution_id, _ = await _approved(Session, ORG_A, user_id)
    await _drain(Session)
    await _relay(Session)

    for _ in range(3):
        # Force re-delivery the way a crash between publish and mark would.
        async with Session() as s:
            await s.execute(
                update(ExecutionOutbox)
                .where(ExecutionOutbox.execution_id == execution_id)
                .values(published_at=None)
            )
            await s.commit()
        await _relay(Session)

    rows = await _outcomes(Session, execution_id=execution_id)
    executed = [r for r in rows if r.outcome_type == "action_executed"]
    assert len(executed) == 1, f"{len(executed)} outcomes for one execution"


# ══ ORDERING / DURABILITY ══════════════════════════════════════════════


async def test_no_outcome_exists_before_the_execution_result_is_committed(
    Session, orgs
):
    """Observation must never claim something happened that the database has not
    durably recorded.

    The outbox pattern is what guarantees it: the event is written in the SAME
    transaction as the status change, and the relay only reads COMMITTED rows.
    So after execution but before relay there is a durable execution result and
    no outcome — never the reverse.
    """
    from navis.execution.models import ExecutionRequest
    from navis.execution.status import ExecutionStatus

    user_id = orgs
    execution_id, _ = await _approved(Session, ORG_A, user_id)
    await _drain(Session)

    async with Session() as s:
        row = await s.get(ExecutionRequest, execution_id)
    assert row.status == ExecutionStatus.EXECUTED.value, "result must be durable first"
    assert await _outcomes(Session, execution_id=execution_id) == [], (
        "an outcome existed before the relay ran"
    )

    await _relay(Session)
    assert len(await _outcomes(Session, execution_id=execution_id)) == 1


# ══ OUTCOME SEMANTICS — failures are not all the same fact ═════════════


async def test_an_authorization_denial_is_distinguishable_from_any_other_failure(
    Session, orgs
):
    """`action_failed` alone cannot tell an operator anything.

    "The organization lost its entitlement", "nobody connected Google" and
    "Gmail was down five times" are three different facts, and only some need a
    human. They were previously separated only by prose inside `error`.
    """
    from navis.execution.contract import FailureKind
    from navis.organization.contract import Capability
    from navis.organization.entitlements import EntitlementService
    from navis.organization.repository import OrganizationRepository

    user_id = orgs
    execution_id, _ = await _approved(Session, ORG_A, user_id)

    async with Session() as s:
        await EntitlementService(OrganizationRepository(s)).set(
            ORG_A, Capability.GMAIL, False
        )
        await s.commit()

    await _drain(Session)
    await _relay(Session)

    rows = await _outcomes(Session, execution_id=execution_id)
    failed = [r for r in rows if r.outcome_type == "action_failed"]
    assert len(failed) == 1
    assert failed[0].payload["failure_kind"] == FailureKind.AUTHORIZATION_DENIED.value
    assert not failed[0].provider_reference, "nothing was executed"


async def test_an_unroutable_action_reports_its_own_failure_kind(Session, orgs):
    """A different cause must produce a different classification, or the field is
    decoration rather than information."""
    from navis.execution.contract import FailureKind

    user_id = orgs
    execution_id, _ = await _approved(Session, ORG_A, user_id)
    await _drain(Session, providers={})  # no provider wired for this action
    await _relay(Session)

    rows = await _outcomes(Session, execution_id=execution_id)
    failed = [r for r in rows if r.outcome_type == "action_failed"]
    assert failed and failed[0].payload["failure_kind"] == FailureKind.UNROUTABLE.value


# ══ HUMAN FEEDBACK — the boundary that was missing entirely ════════════


async def test_a_rejected_proposal_reaches_observation_with_the_humans_reason(
    Session, orgs
):
    """A founder pressing Reject is the clearest negative signal the product can
    get: a direct verdict on something Navis proposed, arriving before any side
    effect.

    Execution has always EMITTED `ACTION_REJECTED` — written to the outbox in the
    same transaction as the status change and relayed like everything else.
    Observation had no subscriber, so it was published and dropped.
    """
    from navis.execution.service import ExecutionService
    from navis.observation.models import OutcomeType

    user_id = orgs
    async with Session() as s:
        req, _ = await ExecutionService(s).request(
            ORG_A, uuid.uuid4(), "send_email",
            {"to": "a@b.com", "subject": "s", "body": "b"},
            reasoning_signal_ids=["sig-9"], reasoning_version=3, state_watermark="w-3",
        )
        await ExecutionService(s).reject(req.id, user_id, reason="tone is wrong")
        await s.commit()
        execution_id, decision_id = req.id, req.decision_id

    await _relay(Session)

    rows = await _outcomes(Session, execution_id=execution_id)
    assert len(rows) == 1, "the human's judgement never reached Observation"
    o = rows[0]
    assert o.outcome_type == OutcomeType.ACTION_REJECTED.value
    assert o.payload["reason"] == "tone is wrong", "the reason was discarded"
    # Full lineage, so the rejection is attributable to what was proposed and why.
    assert o.decision_id == decision_id
    assert o.reasoning_signal_ids == ["sig-9"]
    assert not o.provider_reference, "nothing was executed, so there is no receipt"


async def test_a_cancelled_action_is_recorded_separately_from_a_rejection(
    Session, orgs
):
    """Rejection is a verdict on the proposal; cancellation is usually a
    withdrawal overtaken by events. Collapsing them would teach a later Learning
    pass that they are the same evidence."""
    from navis.execution.service import ExecutionService
    from navis.observation.models import OutcomeType

    async with Session() as s:
        req, _ = await ExecutionService(s).request(
            ORG_A, uuid.uuid4(), "send_email",
            {"to": "a@b.com", "subject": "s", "body": "b"},
            reasoning_signal_ids=[], reasoning_version=1, state_watermark="w",
        )
        await ExecutionService(s).cancel(req.id)
        await s.commit()
        execution_id = req.id

    await _relay(Session)
    rows = await _outcomes(Session, execution_id=execution_id)
    assert len(rows) == 1
    assert rows[0].outcome_type == OutcomeType.ACTION_CANCELLED.value


# ══ ISOLATION ══════════════════════════════════════════════════════════


async def test_an_outcome_is_owned_by_the_persisted_execution_not_the_payload(
    Session, orgs
):
    """Cross-org lineage attack.

    The execution belongs to ORG_B; its payload forges a context naming ORG_A.
    Lineage must come from the persisted row, so the outcome must land on ORG_B
    and ORG_A's log must stay empty.
    """
    user_id = orgs
    execution_id, _ = await _approved(
        Session, ORG_B, user_id,
        payload={
            "to": "a@b.com", "subject": "s", "body": "b",
            "__ctx__": {"org_id": str(ORG_A), "execution_id": str(uuid.uuid4()),
                        "decision_id": str(uuid.uuid4())},
        },
    )
    await _drain(Session)
    await _relay(Session)

    b_rows = await _outcomes(Session, org_id=ORG_B)
    a_rows = await _outcomes(Session, org_id=ORG_A)
    assert len(b_rows) == 1 and b_rows[0].execution_id == execution_id
    assert a_rows == [], "a forged payload attached an outcome to another org"


# ══ SECRET SAFETY ══════════════════════════════════════════════════════


async def test_no_credential_material_reaches_the_outcome_log(Session, orgs):
    """Outcomes are immutable and feed Learning. A secret written here cannot be
    redacted later — it can only be deleted with the fact it belongs to."""
    from navis.observation.models import Outcome
    from sqlalchemy import select

    user_id = orgs
    secret = "ya29.OBSERVATION_SHOULD_NEVER_SEE_THIS"

    class _LeakyProvider(_Provider):
        async def execute(self, payload):
            from navis.execution.contract import ProviderReference, ProviderResult

            # A connector that carelessly puts a token in its reference `extra`.
            return ProviderResult(
                ok=True,
                reference=ProviderReference("gmail", "gmail-2", {"note": "sent"}),
            )

    execution_id, _ = await _approved(
        Session, ORG_A, user_id,
        payload={"to": "a@b.com", "subject": "s", "body": "b",
                 "__ctx__": {"org_id": str(ORG_A), "execution_id": str(uuid.uuid4()),
                             "decision_id": str(uuid.uuid4())}},
    )
    await _drain(Session, providers={"send_email": _LeakyProvider()})
    await _relay(Session)

    async with Session() as s:
        rows = (await s.scalars(select(Outcome))).all()
    blob = str([(r.provider_reference, r.payload, r.source_event_id) for r in rows])
    for needle in (secret, "ya29.", "Bearer ", "refresh_token", "access_token"):
        assert needle not in blob, f"{needle!r} reached the outcome log"


# ══ THE CONTRACT LEARNING CONSUMES ═════════════════════════════════════


async def test_the_observation_event_carries_no_judgement_only_fact(Session, orgs):
    """Observation records WHAT happened, never what it was WORTH.

    The emitted event used to carry a `success` score (0.0/0.5/1.0) computed from
    a table inside Observation — a second opinion that had already drifted from
    Learning's own classifier, which calls ACTION_EXECUTED neutral rather than
    0.5. The outcome TABLE is asserted elsewhere to hold no score column; this
    asserts the same law on the way out.
    """
    from navis.observation.service import ObservationService
    from navis.observation.models import OutcomeType

    captured: list = []

    async def _publisher(event):
        captured.append(event)

    async with Session() as s:
        svc = ObservationService(s, publisher=_publisher)
        await svc.record(
            org_id=ORG_A, source_event_id="evt-contract",
            outcome_type=OutcomeType.ACTION_EXECUTED,
            execution_id=uuid.uuid4(), decision_id=uuid.uuid4(),
            reasoning_signal_ids=["s1"], reasoning_version=2,
            state_watermark="w", provider_reference={"provider": "gmail"},
            payload={"action_type": "send_email"},
        )
        await s.commit()

    assert len(captured) == 1
    payload = captured[0].payload
    for judgement in ("success", "confidence", "score", "prediction", "recommendation"):
        assert judgement not in payload, f"Observation emitted a judgement: {judgement}"
    # And every field Learning needs is present without reading another table.
    for required in (
        "outcome_id", "org_id", "outcome_type", "execution_id", "decision_id",
        "reasoning_signal_ids", "reasoning_version", "state_watermark",
        "provider_reference", "occurred_at", "payload",
    ):
        assert required in payload, f"the Learning contract is missing {required}"


# ══ OBSERVATION -> LEARNING ════════════════════════════════════════════
#
# Everything above stops at the `outcomes` table. These carry the same real
# events one hop further — through the OUTCOME_OBSERVED subscriber, into
# `learning_events`, its aggregates, and out through Learning's read contract —
# because "Observation records it" and "Learning learned the right thing from
# it" are different claims and only the second one closes the loop.


async def test_an_executed_action_becomes_a_learning_event_with_its_lineage_intact(
    Session, orgs
):
    """The full chain on real infrastructure, and every lineage field survives it.

    `provider_reference` and the cause fields are the point: Observation had
    always published them and Learning dropped them on the floor, so a learning
    event could not say which message it was about.
    """
    from navis.learning.models import OutcomeSurface

    user_id = orgs
    execution_id, decision_id = await _approved(Session, ORG_A, user_id)
    await _drain(Session)
    await _relay(Session)

    events = await _learning_events(Session, ORG_A)
    assert len(events) == 1, "the outcome never reached Learning"
    e = events[0]
    assert e.execution_id == execution_id
    assert e.decision_id == decision_id
    assert e.reasoning_signal_ids == ["sig-1", "sig-2"]
    assert e.reasoning_version == 7
    assert e.state_watermark == "w-42"
    assert e.provider_reference == {"provider": "gmail", "external_id": "gmail-1"}
    assert e.outcome_type == "action_executed"
    assert e.outcome_surface == OutcomeSurface.DELIVERY.value
    assert e.occurred_at is not None
    # Business time, not ingestion time: Learning takes occurred_at off the event
    # and has no now() fallback, so the two must not be the same instant by
    # accident of speed.
    assert e.occurred_at == (await _outcomes(Session, execution_id=execution_id))[0].occurred_at


async def test_a_human_rejection_is_learned_as_a_decisive_failure(Session, orgs):
    """Execution -> Observation -> Learning for the product's clearest negative.

    Every hop of this already existed; Learning simply had no entry for
    `action_rejected`, so it fell through to NEUTRAL and a founder's "no" taught
    the system nothing.
    """
    from navis.execution.service import ExecutionService
    from navis.learning.models import AggregateDimension, LearningResult, OutcomeSurface

    user_id = orgs
    async with Session() as s:
        req, _ = await ExecutionService(s).request(
            ORG_A, uuid.uuid4(), "send_email",
            {"to": "a@b.com", "subject": "s", "body": "b"},
            reasoning_signal_ids=["sig-9"], reasoning_version=3, state_watermark="w-1",
        )
        await ExecutionService(s).reject(req.id, user_id, reason="tone is wrong")
        await s.commit()

    await _relay(Session)

    events = await _learning_events(Session, ORG_A)
    assert len(events) == 1
    assert events[0].outcome_type == "action_rejected"
    assert events[0].outcome_surface == OutcomeSurface.JUDGEMENT.value
    assert events[0].result == LearningResult.FAILURE.value

    signal = await _agg(Session, ORG_A, AggregateDimension.SIGNAL, "sig-9")
    assert (signal.successes, signal.failures) == (0, 1), "the rejection taught nothing"
    assert signal.rate == 0.0


async def test_a_channel_failure_does_not_poison_the_signals_success_rate(
    Session, orgs
):
    """A delivery fault is a fact about the channel, not about the reasoning.

    Nothing is wired for `send_email` here, so Execution fails the action with
    FailureKind.UNROUTABLE — a real infrastructure failure produced by the real
    worker, not a synthesised event.
    """
    from navis.learning.models import AggregateDimension, OutcomeSurface

    user_id = orgs
    execution_id, _ = await _approved(Session, ORG_A, user_id)
    await _drain(Session, providers={})       # no provider => genuine failure
    await _relay(Session)

    from navis.execution.contract import FailureKind

    failed = [e for e in await _learning_events(Session, ORG_A)
              if e.outcome_type == "action_failed"]
    assert len(failed) == 1, "the failure never reached Learning"
    assert failed[0].outcome_surface == OutcomeSurface.DELIVERY.value
    # The cause survived the trip, so "unroutable" stays distinguishable from
    # "the customer said no" without re-reading Observation.
    assert failed[0].failure_kind == FailureKind.UNROUTABLE.value

    signal = await _agg(Session, ORG_A, AggregateDimension.SIGNAL, "sig-1")
    assert (signal.successes, signal.failures) == (0, 0), (
        "a channel fault was counted against the reasoning that asked for the send"
    )
    assert signal.neutrals == 1, "the attempt was dropped rather than recorded"
    # And it is fully measurable where it belongs.
    outcome = await _agg(Session, ORG_A, AggregateDimension.OUTCOME, "action_failed")
    assert (outcome.failures, outcome.total) == (1, 1)


async def test_replaying_the_same_outcome_does_not_inflate_learning(Session, orgs):
    """At-least-once delivery is the norm; double-counting must be impossible.

    Re-publishes the SAME persisted outcome through the real subscriber, which is
    exactly what `ObservationReplayService` does after an incident.
    """
    from navis.learning.models import AggregateDimension
    from navis.observation.service import ObservationService

    user_id = orgs
    execution_id, _ = await _approved(Session, ORG_A, user_id)
    await _drain(Session)
    await _relay(Session)

    before = await _agg(Session, ORG_A, AggregateDimension.OUTCOME, "action_executed")
    outcome = (await _outcomes(Session, execution_id=execution_id))[0]

    async with Session() as s:
        await ObservationService(s).republish(outcome)
        await s.commit()

    assert len(await _learning_events(Session, ORG_A)) == 1, "replay forged a second fact"
    after = await _agg(Session, ORG_A, AggregateDimension.OUTCOME, "action_executed")
    assert (after.successes, after.failures, after.neutrals, after.total) == (
        before.successes, before.failures, before.neutrals, before.total
    ), "replay inflated the aggregates"


async def test_one_orgs_learning_never_reaches_another(Session, orgs):
    """Org isolation across the whole loop, not just at the outcome table.

    Both orgs execute actions attributed to the SAME signal key, which is the
    case a per-key aggregate could get wrong.
    """
    from navis.learning.models import AggregateDimension

    user_id = orgs
    await _approved(Session, ORG_A, user_id)
    await _approved(Session, ORG_B, user_id)
    await _drain(Session)
    await _relay(Session)

    a_events = await _learning_events(Session, ORG_A)
    b_events = await _learning_events(Session, ORG_B)
    assert len(a_events) == 1 and len(b_events) == 1
    assert {e.organization_id for e in a_events} == {ORG_A}
    assert {e.organization_id for e in b_events} == {ORG_B}

    # The shared signal key must have its own counter per org.
    a_agg = await _agg(Session, ORG_A, AggregateDimension.SIGNAL, "sig-1")
    b_agg = await _agg(Session, ORG_B, AggregateDimension.SIGNAL, "sig-1")
    assert a_agg.total == 1 and b_agg.total == 1, "one org's counts leaked into the other"

    # And the contract each consumer reads is org-scoped end to end.
    snap_a = await _snapshot(Session, ORG_A)
    snap_b = await _snapshot(Session, ORG_B)
    assert snap_a.org_id == ORG_A and snap_a.learning_version == 1
    assert snap_b.org_id == ORG_B and snap_b.learning_version == 1
    # The KEYS legitimately repeat (both orgs used "sig-1"); the EVIDENCE must not.
    for key, event_ids in snap_a.lineage.items():
        assert set(event_ids).isdisjoint(
            {str(e.id) for e in b_events}
        ), f"org B's evidence supported org A's key {key}"


async def test_learning_reaches_intelligence_only_through_the_contract(Session, orgs):
    """LEARNING_UPDATED -> Intelligence, and Intelligence sees one org's facts.

    `build_view` no longer queries learning_events / learning_aggregates /
    learning_patterns; it calls `navis.learning.metrics.learning_snapshot`. This
    proves the rewired path still produces a real view over real rows.
    """
    from navis.intelligence.repository import IntelligenceRepository

    user_id = orgs
    await _approved(Session, ORG_A, user_id)
    await _approved(Session, ORG_B, user_id)
    await _drain(Session)
    await _relay(Session)

    async with Session() as s:
        view_a = await IntelligenceRepository(s).build_view(ORG_A)
        view_b = await IntelligenceRepository(s).build_view(ORG_B)

    assert view_a.org_id == ORG_A and view_b.org_id == ORG_B
    assert view_a.learning_version == 1 and view_b.learning_version == 1
    # Logical time is the learning's own occurred_at, never wall clock.
    assert view_a.as_of == (await _learning_events(Session, ORG_A))[0].occurred_at
    # The watermark travelled Execution -> Observation -> Learning -> Intelligence.
    assert view_a.state_watermark == "w-42"
    assert [a.key for a in view_a.outcome_aggs] == ["action_executed"]


async def test_learning_is_deterministic_under_rebuild(Session, orgs):
    """Same events + same logic => same aggregates. The replayability proof, on
    real Postgres: delete every projection and fold the immutable log again."""
    from navis.learning.replay import LearningReplayService

    user_id = orgs
    await _approved(Session, ORG_A, user_id)
    await _drain(Session)
    await _relay(Session)

    live = {
        (a.dimension, a.key): (a.successes, a.failures, a.neutrals, a.total)
        for a in (await _snapshot(Session, ORG_A)).aggregates
    }
    assert live, "nothing was learned, so the rebuild would prove nothing"

    async with Session() as s:
        await LearningReplayService(s).rebuild(ORG_A)
        await s.commit()

    rebuilt = {
        (a.dimension, a.key): (a.successes, a.failures, a.neutrals, a.total)
        for a in (await _snapshot(Session, ORG_A)).aggregates
    }
    assert rebuilt == live


async def test_no_credential_material_reaches_the_learning_log(Session, orgs):
    """Learning events are immutable too, and they now carry `provider_reference`
    verbatim — so the secret-safety law has to hold one layer further down than
    it used to."""
    user_id = orgs
    await _approved(Session, ORG_A, user_id)
    await _drain(Session)
    await _relay(Session)

    events = await _learning_events(Session, ORG_A)
    blob = str([
        (e.provider_reference, e.failure_kind, e.source_event_id, e.state_watermark)
        for e in events
    ])
    for needle in ("ya29.", "Bearer ", "refresh_token", "access_token", "password"):
        assert needle not in blob, f"{needle!r} reached the learning log"


# ══ LEARNING -> ML / FEATURE CONSUMPTION ═══════════════════════════════
#
# The last hop. Everything above proves the organization LEARNS from what it
# did; these prove that learning is actually READABLE by the ML layer, through
# Learning's contract, as cited facts a model can be grounded on.
#
# ML pulls at query time rather than subscribing to LEARNING_UPDATED — see
# `ContextBuilder.build`. That is why nothing below drives an event into ML: the
# feature is computed when someone asks a question, not on the outcome path.


async def _learning_ml_facts(Session, org_id):
    """The ML layer's own view of what this org has learned."""
    from navis.ml.rag.context_builder import learning_groundings

    async with Session() as s:
        return await learning_groundings(s, org_id)


async def test_an_outcome_becomes_an_ml_feature_end_to_end(Session, orgs):
    """THE PROOF: synthetic organizational outcome -> Observation -> Learning ->
    ML feature consumer -> feature representation -> grounded context.

    Every hop is the real implementation on real PostgreSQL. Nothing is stubbed.
    """
    from navis.ml.rag.context_builder import LEARNING_KIND, ContextBuilder

    user_id = orgs
    execution_id, _ = await _approved(Session, ORG_A, user_id)
    await _drain(Session)
    await _relay(Session)

    # Learning recorded it (proved above); ML can now read it as a fact.
    facts = await _learning_ml_facts(Session, ORG_A)
    assert facts, "ML sees nothing after a real outcome reached Learning"
    assert {f.kind for f in facts} == {LEARNING_KIND}

    delivery = next(f for f in facts if f.source == "outcome_surface:delivery")
    assert "action_executed" in delivery.text
    # Reproducibility: the fact names the learning version it was derived at.
    assert any(l.startswith("learning_version:") for l in delivery.lineage)

    # And it survives into the context an LLM is actually grounded on.
    async with Session() as s:
        context = await ContextBuilder(s).build(ORG_A, "how are we doing?")
    learned = [f for f in context.facts if f.kind == LEARNING_KIND]
    assert learned, "the learned fact never reached the grounded context"
    assert "action_executed" in context.render()


async def test_one_orgs_learned_feature_is_invisible_to_another(Session, orgs):
    """Isolation at the ML boundary, not just in the tables.

    ORG_A executes; ORG_B does nothing. ORG_B must have no learned facts at all —
    not an empty-but-present feature, nothing.
    """
    user_id = orgs
    await _approved(Session, ORG_A, user_id)
    await _drain(Session)
    await _relay(Session)

    assert await _learning_ml_facts(Session, ORG_A), "org A learned nothing"
    assert await _learning_ml_facts(Session, ORG_B) == [], (
        "org B was handed a feature derived from org A's outcomes"
    )

    # And the rendered context proves it, since that is what reaches the model.
    from navis.ml.rag.context_builder import LEARNING_KIND, ContextBuilder

    async with Session() as s:
        ctx_b = await ContextBuilder(s).build(ORG_B, "how are we doing?")
    assert [f for f in ctx_b.facts if f.kind == LEARNING_KIND] == []


async def test_a_delivery_failure_is_never_presented_as_a_reasoning_failure(
    Session, orgs
):
    """The separation, carried all the way to the prompt.

    No provider is wired, so Execution fails the action for real. The model must
    be told this was a DELIVERY fault, and the signal's reliability must not be
    damaged by it.
    """
    from navis.learning.models import AggregateDimension

    user_id = orgs
    await _approved(Session, ORG_A, user_id)
    await _drain(Session, providers={})   # genuine unroutable failure
    await _relay(Session)

    facts = await _learning_ml_facts(Session, ORG_A)
    delivery = next(f for f in facts if f.source == "outcome_surface:delivery")
    assert "action_failed" in delivery.text
    assert "NOT a judgement" in delivery.text, "the model was not told this was delivery"

    # No world or judgement fact was invented from a channel fault.
    sources = {f.source for f in facts}
    assert "outcome_surface:world" not in sources
    assert "outcome_surface:judgement" not in sources
    # And the reasoning that asked for the send is untouched.
    signal = await _agg(Session, ORG_A, AggregateDimension.SIGNAL, "sig-1")
    assert (signal.successes, signal.failures) == (0, 0)


async def test_a_human_rejection_is_represented_as_judgement(Session, orgs):
    from navis.execution.service import ExecutionService

    user_id = orgs
    async with Session() as s:
        req, _ = await ExecutionService(s).request(
            ORG_A, uuid.uuid4(), "send_email",
            {"to": "a@b.com", "subject": "s", "body": "b"},
            reasoning_signal_ids=["sig-9"], reasoning_version=3, state_watermark="w-1",
        )
        await ExecutionService(s).reject(req.id, user_id, reason="tone is wrong")
        await s.commit()
    await _relay(Session)

    facts = await _learning_ml_facts(Session, ORG_A)
    judgement = next(f for f in facts if f.source == "outcome_surface:judgement")
    assert "action_rejected" in judgement.text
    assert "verdict" in judgement.text, "a human's rejection was not framed as judgement"
    # The founder's private reason is NOT rendered into the prompt.
    assert "tone is wrong" not in " ".join(f.text for f in facts)


async def test_a_world_outcome_is_represented_as_world(Session, orgs):
    """A provider webhook — the only surface that answers "did it work?"."""
    from navis.observation.models import OutcomeType
    from navis.observation.service import ObservationService

    user_id = orgs
    await _approved(Session, ORG_A, user_id)
    await _drain(Session)
    await _relay(Session)

    # The reply arrives against the same provider reference the send recorded.
    class _Webhook:
        provider = "gmail"
        external_id = "gmail-1"
        source_event_id = "hook-reply-1"
        outcome_type = OutcomeType.REPLY_RECEIVED
        occurred_at = NOW
        raw = {"thread": "t-1"}
        # The connected mailbox the notification named. The HTTP edge resolves
        # this to an owning Integration and passes the organization in; this test
        # exercises the service directly, so it supplies the resolved tenant.
        external_account = "founder@org-a.test"

    async with Session() as s:
        assert (
            await ObservationService(s).record_from_webhook(_Webhook(), org_id=ORG_A)
            is not None
        )
        await s.commit()

    facts = await _learning_ml_facts(Session, ORG_A)
    world = next(f for f in facts if f.source == "outcome_surface:world")
    assert "reply_received" in world.text
    assert "actually did" in world.text
    # Delivery and world remain two separate facts about the same action.
    assert "reply_received" not in next(
        f for f in facts if f.source == "outcome_surface:delivery"
    ).text


async def test_an_organization_with_no_outcomes_gets_no_learned_facts(Session, orgs):
    """Missing outcome -> no fabricated feature. Nothing has executed at all."""
    assert await _learning_ml_facts(Session, ORG_A) == []
    assert await _learning_ml_facts(Session, ORG_B) == []


async def test_a_duplicate_event_does_not_duplicate_the_ml_feature(Session, orgs):
    """Replay safety, measured at the feature rather than at the table.

    Learning's idempotency is proved above; this proves the property a consumer
    actually depends on — that re-delivery cannot inflate what the model is told.
    """
    from navis.observation.service import ObservationService

    user_id = orgs
    execution_id, _ = await _approved(Session, ORG_A, user_id)
    await _drain(Session)
    await _relay(Session)

    before = [(f.source, f.text) for f in await _learning_ml_facts(Session, ORG_A)]
    assert before

    outcome = (await _outcomes(Session, execution_id=execution_id))[0]
    async with Session() as s:
        await ObservationService(s).republish(outcome)
        await s.commit()

    after = [(f.source, f.text) for f in await _learning_ml_facts(Session, ORG_A)]
    assert after == before, "replay changed what the model would be told"


async def test_a_learning_outage_degrades_the_context_instead_of_failing_it(
    Session, orgs, monkeypatch
):
    """Requirement: an ML-side failure must never take down the answer.

    Learning is enrichment. If reading it raises, the context must still be
    built from retrieval — exactly as it was before this seam existed.
    """
    import navis.ml.rag.context_builder as mod
    from navis.ml.rag.context_builder import LEARNING_KIND, ContextBuilder

    user_id = orgs
    await _approved(Session, ORG_A, user_id)
    await _drain(Session)
    await _relay(Session)

    async def _boom(db, org_id):
        raise RuntimeError("learning is down")

    monkeypatch.setattr(mod, "learning_groundings", _boom)

    async with Session() as s:
        context = await ContextBuilder(s).build(ORG_A, "how are we doing?")
    assert [f for f in context.facts if f.kind == LEARNING_KIND] == []
    assert context.org_id == str(ORG_A)  # the turn still produced a context


# ══ THE LEARNING READ PATH, HARDENED ═══════════════════════════════════
#
# `learning_snapshot` folded three separate learning_events statements into one
# and stopped hydrating whole ORM entities for the lineage scan. Neither is
# allowed to change a single value it returns, so these pin the semantics that
# the collapse could plausibly have broken.


async def test_snapshot_stats_come_from_one_consistent_newest_event(Session, orgs):
    """count, as_of, watermark and reasoning_version, folded into one query.

    The watermark and the reasoning version must come from the SAME event — the
    newest one that recorded a watermark. Picking them with separate statements
    could straddle two events; a correlated subquery cannot.
    """
    user_id = orgs
    await _approved(Session, ORG_A, user_id)      # watermark "w-42", reasoning 7
    await _drain(Session)
    await _relay(Session)

    snap = await _snapshot(Session, ORG_A)
    events = await _learning_events(Session, ORG_A)
    assert snap.learning_version == len(events) == 1
    assert snap.as_of == max(e.occurred_at for e in events)
    assert snap.state_watermark == "w-42"
    assert snap.reasoning_version == 7


async def test_an_org_with_no_learning_reports_zero_not_a_crash(Session, orgs):
    """The aggregate select always returns a row; the subqueries return NULL.

    An org that has learned nothing must read as (0, None, None, None) rather
    than raising or inventing a version.
    """
    snap = await _snapshot(Session, ORG_A)
    assert snap.learning_version == 0
    assert snap.as_of is None
    assert snap.state_watermark is None
    assert snap.reasoning_version is None
    assert snap.aggregates == () and snap.patterns == ()


async def test_events_without_a_watermark_still_count_toward_the_version(
    Session, orgs
):
    """The watermark subquery filters on `state_watermark IS NOT NULL`; the
    count must NOT. An execution that recorded no watermark is still learning
    that happened, and `learning_version` is an identity key for Intelligence —
    under-counting it would collide two different projections."""
    from navis.execution.service import ExecutionService

    user_id = orgs
    async with Session() as s:
        req, _ = await ExecutionService(s).request(
            ORG_A, uuid.uuid4(), "send_email",
            {"to": "a@b.com", "subject": "s", "body": "b"},
            reasoning_signal_ids=["sig-nw"], reasoning_version=2,
            state_watermark=None,           # nothing recorded a watermark
        )
        await ExecutionService(s).reject(req.id, user_id, reason="no")
        await s.commit()
    await _relay(Session)

    snap = await _snapshot(Session, ORG_A)
    assert snap.learning_version == 1, "an unwatermarked event vanished from the count"
    assert snap.as_of is not None
    assert snap.state_watermark is None    # honestly absent, not fabricated


async def test_the_projected_lineage_scan_returns_the_same_maps(Session, orgs):
    """The scan selects three columns instead of whole entities. It must consult
    the same events, in the same order, and build byte-identical maps."""
    from sqlalchemy import select

    from navis.learning.metrics import _LINEAGE_SCAN_LIMIT, _MAX_LINEAGE
    from navis.learning.models import LearningEvent

    user_id = orgs
    for _ in range(3):
        await _approved(Session, ORG_A, user_id)
    await _drain(Session)
    await _relay(Session)

    snap = await _snapshot(Session, ORG_A)

    # Rebuild the maps the OLD way — full entities — and compare.
    async with Session() as s:
        events = (await s.scalars(
            select(LearningEvent)
            .where(LearningEvent.organization_id == ORG_A)
            .order_by(LearningEvent.id.asc())
            .limit(_LINEAGE_SCAN_LIMIT)
        )).all()

    lineage: dict[str, list[str]] = {}
    signal_lineage: dict[str, list[str]] = {}

    def _add(bucket, key, value):
        slot = bucket.setdefault(key, [])
        if value not in slot and len(slot) < _MAX_LINEAGE:
            slot.append(value)

    for e in events:
        eid = str(e.id)
        # KEYED BY OUTCOME TYPE TOO, and in this position, because that is what
        # `_lineage` does. This recomputation exists to prove the projected
        # select returns the same maps as the entity scan, so it has to model
        # the same rule — omitting this made the comparison assert that the
        # projection had DROPPED a key it is supposed to produce.
        if e.outcome_type:
            _add(lineage, e.outcome_type, eid)
        sigs = [str(x) for x in (e.reasoning_signal_ids or [])]
        for sid in sigs:
            _add(lineage, sid, eid)
            _add(signal_lineage, sid, sid)
        if e.decision_id is not None:
            did = str(e.decision_id)
            _add(lineage, did, eid)
            for sid in sigs:
                _add(signal_lineage, did, sid)

    assert snap.lineage == lineage
    assert snap.signal_lineage == signal_lineage
    assert snap.lineage, "the comparison proved nothing — both maps were empty"


# ══ THE BOUNDED AGGREGATE READ ═════════════════════════════════════════
#
# `learning_aggregates` holds one DECISION row per decision the org ever made,
# and the snapshot used to return all of them — a read that grows with history
# rather than with evidence. The snapshot now applies Learning's OWN
# `patterns.MIN_SAMPLES` to that dimension in SQL. These prove the filter drops
# exactly what its only reader was already discarding, and nothing else.


async def _seed_decision_aggregates(Session, org, *, low: int, qualifying: int):
    """Decision counters written directly, so the shape is exact.

    `low` rows sit below MIN_SAMPLES decisive; `qualifying` sit at or above it.
    Neutrals are added to the low rows on purpose: a row can have a large `total`
    and still be no evidence at all, and the filter must key on decisive samples
    rather than on volume.
    """
    from sqlalchemy import text

    async with Session() as s:
        await s.execute(
            text("""
            INSERT INTO learning_aggregates
              (organization_id, dimension, key, successes, failures, neutrals, total)
            SELECT :org, 'decision', 'low-' || g, 1, 1, 40, 42
              FROM generate_series(1, :n) g
            """),
            {"org": org, "n": low},
        )
        await s.execute(
            text("""
            INSERT INTO learning_aggregates
              (organization_id, dimension, key, successes, failures, neutrals, total)
            SELECT :org, 'decision', 'ok-' || g, 4, 1, 0, 5
              FROM generate_series(1, :n) g
            """),
            {"org": org, "n": qualifying},
        )
        await s.commit()


async def test_only_decision_aggregates_with_enough_evidence_reach_the_snapshot(
    Session, orgs
):
    """THE REGRESSION PROOF. 100 low-sample rows, 20 qualifying rows."""
    from navis.learning.models import AggregateDimension
    from navis.learning.patterns import MIN_SAMPLES

    await _seed_decision_aggregates(Session, ORG_A, low=100, qualifying=20)

    snap = await _snapshot(Session, ORG_A)
    decisions = snap.by_dimension(AggregateDimension.DECISION)

    assert len(decisions) == 20, f"expected the 20 qualifying rows, got {len(decisions)}"
    assert all(a.key.startswith("ok-") for a in decisions)
    assert all(a.decisive >= MIN_SAMPLES for a in decisions)
    # A high `total` built entirely from neutrals is not evidence and must not
    # sneak through: every low row had total=42.
    assert not any(a.key.startswith("low-") for a in decisions)


async def test_the_low_sample_aggregates_are_still_stored(Session, orgs):
    """Filtered from the READ, never deleted. They keep accumulating and appear
    the moment they cross the threshold — this is a projection rule, not a
    retention policy."""
    from sqlalchemy import func, select

    from navis.learning.models import LearningAggregate

    await _seed_decision_aggregates(Session, ORG_A, low=100, qualifying=20)

    async with Session() as s:
        stored = await s.scalar(
            select(func.count())
            .select_from(LearningAggregate)
            .where(
                LearningAggregate.organization_id == ORG_A,
                LearningAggregate.dimension == "decision",
            )
        )
    assert stored == 120, "the filter deleted learning data"


async def test_a_low_sample_decision_appears_once_it_earns_its_place(Session, orgs):
    """The boundary, from both sides. Nothing is lost — only deferred."""
    from sqlalchemy import text

    from navis.learning.models import AggregateDimension

    async with Session() as s:
        await s.execute(
            text("""
            INSERT INTO learning_aggregates
              (organization_id, dimension, key, successes, failures, neutrals, total)
            VALUES (:org, 'decision', 'borderline', 2, 2, 9, 13)
            """),
            {"org": ORG_A},
        )
        await s.commit()

    snap = await _snapshot(Session, ORG_A)
    assert [a.key for a in snap.by_dimension(AggregateDimension.DECISION)] == []

    async with Session() as s:  # one more decisive outcome -> 5 decisive
        await s.execute(
            text("UPDATE learning_aggregates SET successes = 3, total = 14 "
                 "WHERE organization_id = :org AND key = 'borderline'"),
            {"org": ORG_A},
        )
        await s.commit()

    snap = await _snapshot(Session, ORG_A)
    rows = snap.by_dimension(AggregateDimension.DECISION)
    assert [a.key for a in rows] == ["borderline"]
    assert rows[0].decisive == 5


async def test_signal_and_outcome_aggregates_are_untouched_by_the_filter(
    Session, orgs
):
    """Only DECISION is bounded.

    A single executed action produces a SIGNAL row with 0 decisive samples and an
    OUTCOME row with 0 decisive samples. Both must survive: ML reports raw
    delivery/judgement/world counts with no threshold, and filtering them would
    delete the surface evidence rather than trim it.
    """
    from navis.learning.models import AggregateDimension

    user_id = orgs
    await _approved(Session, ORG_A, user_id)
    await _drain(Session)
    await _relay(Session)

    snap = await _snapshot(Session, ORG_A)
    signals = snap.by_dimension(AggregateDimension.SIGNAL)
    outcomes = snap.by_dimension(AggregateDimension.OUTCOME)

    assert {a.key for a in signals} == {"sig-1", "sig-2"}
    assert all(a.decisive == 0 for a in signals), "the fixture stopped proving the point"
    assert [a.key for a in outcomes] == ["action_executed"]
    # And the surface grouping ML renders still sees it.
    assert "delivery" in snap.outcomes_by_surface()


async def test_intelligence_output_is_unchanged_by_the_bounded_read(Session, orgs):
    """Equivalence, not similarity: the projected rows must be identical.

    Builds the view the production way (filtered), then rebuilds `LearningView`
    from the UNFILTERED aggregates and projects both through the same generators.
    """
    from sqlalchemy import select

    from navis.intelligence.insights import AggRow, project as project_view
    from navis.intelligence.repository import IntelligenceRepository
    from navis.intelligence.service import project_values
    from navis.learning.models import LearningAggregate

    await _seed_decision_aggregates(Session, ORG_A, low=100, qualifying=20)
    user_id = orgs
    await _approved(Session, ORG_A, user_id)
    await _drain(Session)
    await _relay(Session)

    async with Session() as s:
        filtered_view = await IntelligenceRepository(s).build_view(ORG_A)
        unfiltered_rows = (await s.scalars(
            select(LearningAggregate).where(
                LearningAggregate.organization_id == ORG_A,
                LearningAggregate.dimension == "decision",
            )
        )).all()

    # 120 seeded + 1 from the real executed action above. That real one is
    # `action_executed` — a DELIVERY attempt, neutral, 0 decisive — so it is
    # exactly the kind of row the filter is meant to hold back, and it makes the
    # comparison below stronger rather than weaker.
    assert len(unfiltered_rows) == 121, "the unfiltered arm proved nothing"

    # Same view, but with every decision row the database holds.
    import dataclasses

    unfiltered_view = dataclasses.replace(
        filtered_view,
        decision_aggs=[
            AggRow(r.key, r.successes, r.failures, r.neutrals, r.total)
            for r in unfiltered_rows
        ],
    )

    assert len(filtered_view.decision_aggs) == 20
    assert project_values(filtered_view) == project_values(unfiltered_view)
    # And the decision insights that DO get emitted are the qualifying ones.
    keys = {d.key for d in project_view(filtered_view)
            if d.insight_type.value == "decision_intelligence"}
    assert keys == {f"ok-{i}" for i in range(1, 21)}
