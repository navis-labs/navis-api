"""P0-2: real-PostgreSQL concurrency — N workers cannot execute one request twice.

Runs only with NAVIS_TEST_DATABASE_URL (real Postgres; FOR UPDATE SKIP LOCKED and
ON CONFLICT cannot be exercised against fakes). Proves: (a) concurrent idempotent
requests collapse to one row; (b) concurrent execute passes execute an APPROVED
request exactly once.
"""
from __future__ import annotations

import asyncio
import os
import uuid

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("NAVIS_TEST_DATABASE_URL"),
    reason="set NAVIS_TEST_DATABASE_URL to run execution concurrency tests",
)


@pytest.fixture()
async def engine():
    from sqlalchemy.ext.asyncio import create_async_engine

    import navis.registry  # noqa: F401
    from navis.platform.database import Base

    eng = create_async_engine(os.environ["NAVIS_TEST_DATABASE_URL"])
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield eng
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await eng.dispose()


async def _seed_org_user(session, *, plan=None):
    """An org as production actually has it: created AND provisioned.

    `bootstrap` is what the ORG_CREATED handler runs at signup — it writes the
    lifecycle row and seeds the plan's entitlements. Without it the org exists
    but is entitled to nothing, which is not a state any real approved execution
    can be in: the approve endpoint requires EXECUTION plus the action's
    connector capability before a row can reach APPROVED at all.

    This matters now that the worker RE-CHECKS authorization immediately before
    the provider call. An unprovisioned org is correctly refused there, so
    seeding only the Organization row would test the denial path while claiming
    to test concurrency. Provisioning here means these tests exercise the real
    authorization path against Postgres on their way to the thing they assert.
    """
    from navis.identity.models import Organization, User
    from navis.organization.contract import Plan
    from navis.organization.service import OrganizationService

    org = Organization(name="Acme", slug=f"acme-{uuid.uuid4().hex[:6]}")
    user = User(email=f"u-{uuid.uuid4().hex[:6]}@acme.com")
    session.add_all([org, user])
    await session.flush()
    # TEAM is the lowest plan carrying EXECUTION + GMAIL + CALENDAR.
    await OrganizationService(session).bootstrap(org.id, plan=plan or Plan.TEAM)
    await session.flush()
    return org.id, user.id


async def test_concurrent_requests_collapse_to_one_row(engine):
    from sqlalchemy.ext.asyncio import async_sessionmaker
    from navis.execution.service import ExecutionService

    Session = async_sessionmaker(engine, expire_on_commit=False)
    async with Session() as s:
        org_id, _ = await _seed_org_user(s)
        await s.commit()

    decision = uuid.uuid4()
    payload = {"to": "a@x.com", "subject": "Hi"}

    async def one():
        async with Session() as s:
            req, created = await ExecutionService(s).request(
                org_id, decision, "send_email", payload,
                reasoning_signal_ids=[], reasoning_version=1, state_watermark="w",
            )
            await s.commit()
            return req.id, created

    results = await asyncio.gather(*[one() for _ in range(25)], return_exceptions=True)
    ids = {r[0] for r in results if isinstance(r, tuple)}
    created_count = sum(1 for r in results if isinstance(r, tuple) and r[1])
    assert len(ids) == 1            # one deterministic execution_id
    assert created_count == 1       # exactly one creator; rest idempotent no-ops


class SyntheticProvider:
    """A stand-in for a real external system, without leaving the process.

    NOT a dry run. It behaves the way a real connector behaves in every respect
    that concurrency depends on:

      * it TAKES TIME, so the window between claim and result is real and two
        workers genuinely overlap (a synchronous fake would serialise itself and
        prove nothing);
      * it COUNTS its invocations, which is the assertion that matters — one
        database row is not the same claim as one external side effect;
      * it returns a STABLE external reference derived from execution_id, the
        same shape `_google_event_id` / `navis_message_id` use, so a replay is
        recognisable;
      * it is ORG SCOPED: it reads `__ctx__` and refuses without it, exactly as
        the real senders do.

    `_sender` is set so `is_live` reports True — a provider that would dry-run
    is withheld from the production registry and could not model a real send.
    """

    name = "synthetic"
    action_type = "send_email"  # a REAL supported action; only transport is fake

    def __init__(self, delay: float = 0.05) -> None:
        self._sender = self._send
        self._delay = delay
        self.calls: list[tuple[str, str]] = []

    async def _send(self, payload):  # pragma: no cover - shape parity only
        return "unused"

    @property
    def is_live(self) -> bool:
        return True

    unsupported = False

    async def validate(self, payload) -> None:
        if not payload.get("to"):
            raise ValueError("send_email requires 'to'")

    async def execute(self, payload):
        from navis.execution.contract import ProviderReference, ProviderResult

        ctx = payload.get("__ctx__") or {}
        org_id, execution_id = ctx.get("org_id"), ctx.get("execution_id")
        assert org_id and execution_id, "worker must thread __ctx__ to the provider"
        await asyncio.sleep(self._delay)  # a real external call takes time
        self.calls.append((org_id, execution_id))
        return ProviderResult(
            ok=True,
            reference=ProviderReference("synthetic", f"ext-{execution_id}"),
        )

    async def status(self, reference) -> str:  # pragma: no cover
        return "sent"


def _unit_factory(Session):
    """Mirrors production's `session_scope` unit: ONE transaction per phase."""
    from contextlib import asynccontextmanager

    from navis.execution.repository import ExecutionRepository

    @asynccontextmanager
    async def unit():
        async with Session() as s:
            try:
                yield ExecutionRepository(s)
                await s.commit()
            except Exception:
                await s.rollback()
                raise

    return unit


async def test_concurrent_workers_execute_once(engine):
    """N workers race ONE approved request. Exactly one external call happens.

    Drives the worker through `unit=` — the production shape, where claim,
    provider call and result each get their own transaction. The guarantee under
    test is therefore the real one: the committed EXECUTING transition is the
    lease, so a second worker cannot claim the row even though NO database lock
    is held while the provider runs.

    Previously this test used `create_task`, which now refuses at validate(), so
    it asserted `executed == 1` against a path that always failed — a green
    invariant that had silently stopped being tested.
    """
    from sqlalchemy import func, select
    from sqlalchemy.ext.asyncio import async_sessionmaker

    from navis.execution.models import ExecutionOutbox
    from navis.execution.repository import ExecutionRepository
    from navis.execution.service import ExecutionService
    from navis.execution.status import ExecutionStatus
    from navis.execution.worker import execute_pass
    from navis.platform.events import EventType

    Session = async_sessionmaker(engine, expire_on_commit=False)
    async with Session() as s:
        org_id, user_id = await _seed_org_user(s)
        svc = ExecutionService(s)
        req, _ = await svc.request(
            org_id, uuid.uuid4(), "send_email",
            {"to": "a@x.com", "subject": "Hi", "body": "a real body"},
            reasoning_signal_ids=[], reasoning_version=1, state_watermark="w",
        )
        await svc.approve(req.id, user_id)
        await s.commit()

    provider = SyntheticProvider()  # ONE shared instance: the call count is global
    unit = _unit_factory(Session)

    async def worker():
        out = await execute_pass(unit=unit, providers={"send_email": provider})
        return out["executed"]

    totals = await asyncio.gather(*[worker() for _ in range(10)], return_exceptions=True)
    for t in totals:
        assert not isinstance(t, BaseException), f"worker raised: {t!r}"

    # THE assertion: one execution AND one external side effect.
    assert sum(totals) == 1, "more than one worker executed the request"
    assert len(provider.calls) == 1, (
        f"provider was invoked {len(provider.calls)} times — a duplicate external "
        "side effect, which one database row would have hidden"
    )
    assert provider.calls[0][0] == str(org_id)  # org scoping survived the split

    async with Session() as s:
        row = await ExecutionRepository(s).get(req.id)
        assert row.status == ExecutionStatus.EXECUTED.value
        assert row.attempts == 1, "exactly one attempt was consumed"
        assert row.provider_reference["external_id"] == f"ext-{req.id}"
        assert "dry_run" not in row.provider_reference

        executed_events = await s.scalar(
            select(func.count(ExecutionOutbox.id)).where(
                ExecutionOutbox.execution_id == req.id,
                ExecutionOutbox.event_type == EventType.ACTION_EXECUTED.value,
            )
        )
        assert executed_events == 1, "exactly one ActionExecuted event"
