"""A brief decision must say what has already been done about it.

Before this, `GET /brief/today` served raw reasoning output and `GET /actions`
returned REQUESTED rows only — so a reply that had ALREADY BEEN SENT was
invisible to every frontend surface. The drawer kept offering "Take Action" for
work that was finished, and clicking it called propose() for a completed
obligation.

The enrichment resolves by the P0 STABLE OBLIGATION KEY, never by decision_id.
That is the whole point: the same unanswered email carries a different
decision_id in every brief, so lineage matching would lose the execution the
moment a brief was rebuilt.

Runs ONLY when `NAVIS_TEST_DATABASE_URL` points at a DISPOSABLE async Postgres —
the fixture drops every table on teardown.

    NAVIS_TEST_DATABASE_URL=postgresql+asyncpg://... pytest tests/integration -q

Real Postgres because the lookup is served off the partial unique index
uq_execution_obligation, and because "one query, not N" is only meaningful
against a real driver. Nothing here sends mail or approves anything.
"""
from __future__ import annotations

import os
import uuid

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("NAVIS_TEST_DATABASE_URL"),
    reason="set NAVIS_TEST_DATABASE_URL to run brief-enrichment Postgres tests",
)

THREAD = "19fdb57e89581d43"
MESSAGE = "<calendar-b7706463-b4ea-4492-b1aa-89c6dfe8@google.com>"
OTHER_MESSAGE = "<CACoxYfgQQVZ1K=2OQOXvYjZ6-ms9Gp4Fxowp=s9N@mail.gmail.com>"
GMAIL_ID = "19feb69777d46809"


@pytest.fixture()
async def engine():
    from sqlalchemy.ext.asyncio import create_async_engine

    import navis.registry  # noqa: F401  register all models
    from navis.platform.database import Base

    eng = create_async_engine(os.environ["NAVIS_TEST_DATABASE_URL"])
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield eng
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await eng.dispose()


async def _org(engine) -> uuid.UUID:
    from sqlalchemy import text as sql

    org_id = uuid.uuid4()
    async with engine.begin() as conn:
        await conn.execute(
            sql("INSERT INTO organizations (id, name, slug) VALUES (:i, :n, :s)"),
            {"i": org_id, "n": "enrich-test", "s": f"enr-{org_id.hex[:12]}"},
        )
    return org_id


def decision(*, decision_id=None, category="waiting_reply", message_id=MESSAGE, **ev):
    """A brief decision as the reasoning engines emit it."""
    evidence = {
        "person_email": "mulayamabmrc@gmail.com",
        "person_name": "Mulayam",
        "thread_id": THREAD,
        "rfc_message_id": message_id,
        "subject": "Invitation: Meeting",
        "waiting_days": 2.3,
        "commitment": "I'll send the deck",
        "due_at": "2026-08-08T00:00:00+00:00",
        "decision": "ship or hold?",
    }
    evidence.update(ev)
    return {
        "decision_id": str(decision_id or uuid.uuid4()),
        "category": category,
        "title": "Mulayam is waiting for your reply",
        "why": "They emailed and there is no reply from you.",
        "priority_rank": 1,
        "evidence": evidence,
        "reasoning_signal_ids": [],
        "reasoning_version": 2,
        "state_watermark": "wm-1",
    }


class _Brief:
    """Stands in for a DailyBrief row — the enrichment only reads these three."""

    def __init__(self, org_id, decisions):
        self.id = uuid.uuid4()
        self.organization_id = org_id
        self.decisions = decisions
        self.as_of = None
        self.reasoning_version = 2
        self.state_watermark = "wm-1"


async def enrich(engine, org_id, decisions):
    from sqlalchemy.ext.asyncio import async_sessionmaker

    from navis.api.brief_router import _with_executions

    Session = async_sessionmaker(engine, expire_on_commit=False)
    async with Session() as s:
        return await _with_executions(s, _Brief(org_id, decisions))


async def seed_execution(engine, org_id, *, status, d, message_id=MESSAGE,
                         provider_reference=None, attempts=0, last_error=None,
                         action_type="reply_to_email", body="Hi — following up."):
    """Create the execution for a decision's obligation, at a chosen status.

    Goes through the REAL ExecutionService so the obligation key is computed by
    production code, then sets the terminal state directly. No provider is ever
    invoked — nothing is sent.
    """
    from sqlalchemy.ext.asyncio import async_sessionmaker

    from navis.execution.models import ExecutionOrigin
    from navis.execution.proposer import _map
    from navis.execution.service import ExecutionService

    _at, payload = _map(d)
    payload = dict(payload, body=body, requires_draft=False)

    Session = async_sessionmaker(engine, expire_on_commit=False)
    async with Session() as s:
        req, _ = await ExecutionService(s).request(
            org_id, uuid.UUID(d["decision_id"]), action_type, payload,
            reasoning_signal_ids=[], reasoning_version=2, state_watermark="wm-1",
            origin=ExecutionOrigin.PROACTIVE,
        )
        req.status = status
        if provider_reference is not None:
            req.provider_reference = provider_reference
        req.attempts = attempts
        req.last_error = last_error
        await s.commit()
        return req.id


# ── The contract ───────────────────────────────────────────────────────


async def test_no_execution_yields_null(engine):
    org_id = await _org(engine)
    out = await enrich(engine, org_id, [decision()])
    assert out["decisions"][0]["execution"] is None


async def test_findings_never_get_an_execution(engine):
    """risk / priority / opportunity are things to read, not replies to send."""
    org_id = await _org(engine)
    decisions = [decision(category=c) for c in ("risk", "priority", "opportunity")]
    out = await enrich(engine, org_id, decisions)
    assert [d["execution"] for d in out["decisions"]] == [None, None, None]


@pytest.mark.parametrize(
    "status, ref, attempts, err",
    [
        ("requested", None, 0, None),
        ("approved", None, 0, None),
        ("executing", None, 1, None),
        ("executed", {"provider": "gmail", "external_id": GMAIL_ID}, 1, None),
        ("executed", {"provider": "gmail", "external_id": "dryrun-abc", "dry_run": True}, 1, None),
        ("failed", None, 1, "gmail rejected the recipient"),
        ("failed", None, 5, "exhausted"),
        ("rejected", None, 0, None),
        ("cancelled", None, 0, None),
    ],
)
async def test_every_status_reaches_the_decision(engine, status, ref, attempts, err):
    org_id = await _org(engine)
    d = decision()
    await seed_execution(engine, org_id, status=status, d=d,
                         provider_reference=ref, attempts=attempts, last_error=err)

    out = await enrich(engine, org_id, [d])
    ex = out["decisions"][0]["execution"]

    assert ex is not None, f"{status} execution was not attached"
    assert ex["status"] == status
    assert ex["attempts"] == attempts
    assert ex["provider_reference"] == (ref or {})
    if err:
        assert ex["last_error"] == err
    else:
        assert ex["last_error"] is None
    # The wire must stay minimal — no frozen body, no lineage, no watermark.
    assert set(ex) == {
        "execution_id", "status", "provider_reference",
        "attempts", "last_error", "requires_draft",
    }


async def test_a_real_send_is_distinguishable_from_a_dry_run(engine):
    """The frontend decides "sent" — but it can only do so if BOTH reach it."""
    org_id = await _org(engine)
    real, dry = decision(), decision(message_id=OTHER_MESSAGE)
    await seed_execution(engine, org_id, status="executed", d=real,
                         provider_reference={"provider": "gmail", "external_id": GMAIL_ID})
    await seed_execution(engine, org_id, status="executed", d=dry, message_id=OTHER_MESSAGE,
                         provider_reference={"provider": "gmail", "external_id": "dryrun-x",
                                             "dry_run": True})

    out = await enrich(engine, org_id, [real, dry])
    assert out["decisions"][0]["execution"]["provider_reference"]["external_id"] == GMAIL_ID
    assert "dry_run" not in out["decisions"][0]["execution"]["provider_reference"]
    assert out["decisions"][1]["execution"]["provider_reference"]["dry_run"] is True


# ── Resolution is by obligation, not lineage ───────────────────────────


async def test_three_briefs_one_execution(engine):
    """THE regression. Same email, three decision_ids, one sent reply.

    Every one of them must report the SAME execution — otherwise the dashboard
    forgets the reply was sent as soon as a brief is rebuilt.
    """
    org_id = await _org(engine)
    first = decision()
    execution_id = await seed_execution(
        engine, org_id, status="executed", d=first,
        provider_reference={"provider": "gmail", "external_id": GMAIL_ID},
        attempts=1,
    )

    # Later briefs: same inbound email, brand-new decision_ids.
    later = [decision(), decision()]
    out = await enrich(engine, org_id, [first, *later])

    ids = {d["execution"]["execution_id"] for d in out["decisions"]}
    assert ids == {str(execution_id)}
    assert all(d["execution"]["status"] == "executed" for d in out["decisions"])
    # And the decision_ids really did differ — otherwise this proves nothing.
    assert len({d["decision_id"] for d in out["decisions"]}) == 3


async def test_a_new_inbound_message_does_not_inherit_the_old_status(engine):
    """Same thread, new message => new obligation => actionable again."""
    org_id = await _org(engine)
    answered = decision()
    await seed_execution(engine, org_id, status="executed", d=answered,
                         provider_reference={"provider": "gmail", "external_id": GMAIL_ID})

    fresh = decision(message_id=OTHER_MESSAGE)
    out = await enrich(engine, org_id, [answered, fresh])

    assert out["decisions"][0]["execution"]["status"] == "executed"
    assert out["decisions"][1]["execution"] is None, (
        "a genuinely new inbound message inherited the previous reply's status"
    )


async def test_decision_owed_resolves_by_message_id(engine):
    org_id = await _org(engine)
    d = decision(category="decision_owed")
    await seed_execution(engine, org_id, status="requested", d=d)
    assert (await enrich(engine, org_id, [d]))["decisions"][0]["execution"]["status"] == "requested"


async def test_commitment_follow_up_resolves_by_the_promise(engine):
    org_id = await _org(engine)
    d = decision(category="follow_up")
    await seed_execution(engine, org_id, status="executed", d=d,
                         provider_reference={"provider": "gmail", "external_id": GMAIL_ID})

    # A later brief carrying the SAME promise must find the same execution.
    same_promise = decision(category="follow_up")
    out = await enrich(engine, org_id, [same_promise])
    assert out["decisions"][0]["execution"]["status"] == "executed"

    # A DIFFERENT promise on the same thread is different work.
    other = decision(category="follow_up", commitment="I'll send the contract")
    assert (await enrich(engine, org_id, [other]))["decisions"][0]["execution"] is None


async def test_an_unrelated_meeting_execution_is_not_attached(engine):
    """schedule_meeting has no obligation key, so it can never bind to a reply."""
    org_id = await _org(engine)
    d = decision()
    await seed_execution(engine, org_id, status="executed", d=d,
                         action_type="schedule_meeting",
                         provider_reference={"provider": "google", "external_id": "evt-1"})

    out = await enrich(engine, org_id, [d])
    assert out["decisions"][0]["execution"] is None


# ── Shape and efficiency ───────────────────────────────────────────────


async def test_existing_fields_and_order_are_preserved(engine):
    org_id = await _org(engine)
    decisions = [decision(), decision(category="risk"), decision(message_id=OTHER_MESSAGE)]
    out = await enrich(engine, org_id, decisions)

    assert [d["decision_id"] for d in out["decisions"]] == [
        d["decision_id"] for d in decisions
    ]
    for before, after in zip(decisions, out["decisions"]):
        for key, value in before.items():
            assert after[key] == value, f"{key} was altered by enrichment"
    assert {"id", "org_id", "as_of", "reasoning_version", "state_watermark", "decisions"} <= set(out)


async def test_no_n_plus_one(engine):
    """One SELECT for the whole brief, however many decisions it carries."""
    from sqlalchemy import event
    from sqlalchemy.ext.asyncio import async_sessionmaker

    from navis.api.brief_router import _with_executions

    org_id = await _org(engine)
    decisions = [
        decision(message_id=f"<msg-{i}@mail.example>") for i in range(25)
    ]
    for d in decisions[:10]:
        await seed_execution(engine, org_id, status="executed", d=d,
                             message_id=d["evidence"]["rfc_message_id"],
                             provider_reference={"provider": "gmail", "external_id": GMAIL_ID})

    selects: list[str] = []

    @event.listens_for(engine.sync_engine, "before_cursor_execute")
    def _count(conn, cursor, statement, params, context, executemany):
        if statement.lstrip().upper().startswith("SELECT"):
            selects.append(statement)

    Session = async_sessionmaker(engine, expire_on_commit=False)
    async with Session() as s:
        out = await _with_executions(s, _Brief(org_id, decisions))
    event.remove(engine.sync_engine, "before_cursor_execute", _count)

    assert sum(1 for d in out["decisions"] if d["execution"]) == 10
    assert len(selects) == 1, (
        f"expected ONE batched lookup for 25 decisions, issued {len(selects)}"
    )
