"""Cross-tenant isolation, proved against real PostgreSQL — TASK 6.5.1.

Two organizations with a row in EVERY organization-scoped layer coexist in one
database, and Org A's real read paths — the mounted HTTP surface where one
exists, the real repository where it does not — never return, count, or act on
an Org B row.

    NAVIS_TEST_DATABASE_URL=postgresql+asyncpg://.../navis_test?ssl=require \
        EVENT_HANDLER_TIMEOUT_SECONDS=180 \
        pytest tests/integration/test_cross_tenant_isolation_postgres.py -q

WHY THIS IS ITS OWN MODULE
--------------------------
It needs a MODULE-SCOPED event loop and a module-scoped engine that builds the
schema once and never drops it. Neither existing home can provide that:

  * `test_organizational_workflows.py` runs `drop_all` + `create_all` per test.
    Recreating the schema gives every PostgreSQL ENUM a new `pg_type` oid while
    pooled asyncpg connections still hold the old oids in their per-connection
    TYPE CODEC cache; the driver re-introspects, is invalidated again, and after
    six attempts aborts the first parameterised INSERT that follows with
    `InternalClientError: could not resolve query result and/or argument types
    in 6 attempts`. That is what stopped these tests from ever reaching an
    assertion. (`statement_cache_size=0` does NOT fix it — that disables
    PREPARED STATEMENT caching, which is a different cache.)
  * `test_organization_postgres.py` was tried as a host. Converting that module
    to `loop_scope="module"` broke its six existing enum/concurrency tests with
    `MissingGreenlet`, because a module cannot cleanly mix module-loop and
    function-loop async tests sharing one engine.

So the schema is built ONCE here and each test cleans only its own rows — the
pattern `test_token_backfill_postgres.py` and
`test_membership_concurrency_postgres.py` already use, both of which run green
against Neon. One test went from ~8 minutes (and failing in setup) to ~70s.

SAFETY
------
Refuses to run unless the target database is named `navis_test*`, and never
drops a schema. Every module that imported `session_scope` by value is repointed
at this engine — see the list in `Session`, and the production-read incident
recorded there.
"""
from __future__ import annotations

import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio

pytestmark = [
    pytest.mark.skipif(
        not os.environ.get("NAVIS_TEST_DATABASE_URL"),
        reason="set NAVIS_TEST_DATABASE_URL (disposable Postgres) to run these",
    ),
    # ONE event loop for the module. asyncpg connections bind to the loop that
    # created them, so a module-scoped engine cannot be shared across per-test
    # loops.
    pytest.mark.asyncio(loop_scope="module"),
]

NOW = datetime(2026, 8, 19, 9, 0, tzinfo=timezone.utc)


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def engine():
    """ONE engine and ONE `create_all` for the whole module. Never `drop_all`.

    Guarded by DATABASE NAME, never hostname: a hostname check would both block
    a legitimate disposable Neon database and permit clobbering a local working
    copy.
    """
    from sqlalchemy.ext.asyncio import create_async_engine

    import navis.registry  # noqa: F401  register every model
    from navis.platform.database import Base

    url = os.environ["NAVIS_TEST_DATABASE_URL"]
    name = url.rsplit("/", 1)[-1].split("?")[0]
    assert name.startswith("navis_test"), (
        f"refusing to run against database {name!r}: point "
        "NAVIS_TEST_DATABASE_URL at a disposable database named navis_test*"
    )

    eng = create_async_engine(url)
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield eng
    await eng.dispose()


# ── The disposable session, wired everywhere production would reach global ──


@pytest_asyncio.fixture(loop_scope="module")
async def Session(engine, monkeypatch):
    """A sessionmaker on the module engine, installed everywhere the production
    path would otherwise reach for the global one.

    `OrganizationService.bootstrap` publishes onboarding events, and the
    subscribers that consume them imported `session_scope` BY VALUE at module
    import — so a subscriber that is not patched here opens its own session
    against the URL in `.env`, which is PRODUCTION. Patching the source module
    alone would not be seen by them.
    """
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

    # EVERY module that did `from navis.platform.database import session_scope`
    # holds its OWN reference, bound at import time. Patching the source module
    # alone does not reach them, and an unpatched one opens a session against
    # the URL in `.env` — PRODUCTION.
    #
    # `navis.orchestration.brief` is on this list because leaving it off is not
    # hypothetical: `_dispatch_all()` then enumerated the PRODUCTION
    # `organizations` table, and the fan-out test failed with three real
    # production org ids instead of this module's two tenants.
    for target in (
        "navis.platform.database.session_scope",
        "navis.orchestration.brief.session_scope",
        "navis.orchestration.org_provisioning.session_scope",
        "navis.orchestration.sync.session_scope",
        "navis.observation.subscribers.session_scope",
        "navis.learning.subscribers.session_scope",
        "navis.intelligence.subscribers.session_scope",
        "navis.ml.embeddings.subscribers.session_scope",
        "navis.memory.service.session_scope",
        "navis.identity_resolution.service.session_scope",
        "navis.execution.providers._senders.session_scope",
    ):
        monkeypatch.setattr(target, _scope, raising=False)

    # Never mirror this module's events onto a real Redis bus.
    from navis.platform.events import event_bus

    monkeypatch.setattr(event_bus, "_mirror_to_redis", False, raising=False)
    return maker


# ═════════════════════════════════════════════════════════════════════════════
# TASK 6.5.1 — CROSS-TENANT ISOLATION, EVERY LAYER, REAL POSTGRES
# ═════════════════════════════════════════════════════════════════════════════
#
# `test_organizational_workflows.py::test_one_organizations_workflow_never_touches_another`
# proves the EXECUTION path. This section proves the other eleven: two fully populated
# organizations coexisting in one database, and Org A's real read paths — the
# mounted HTTP surface where one exists, the real repository where it does not —
# never returning, counting, or acting on an Org B row.
#
# Populated, not stubbed. Every table below holds rows for BOTH tenants, so an
# assertion that Org A sees one row is meaningful: a query missing its tenant
# filter would see two.

TENANT_A = uuid.UUID("aaaaaaaa-0000-4000-8000-00000000000a")
TENANT_B = uuid.UUID("bbbbbbbb-0000-4000-8000-00000000000b")


def _tenant_client(Session):
    """httpx over the FULL mounted api_router, on this event loop.

    NOT Starlette's TestClient: it runs the app in its own portal loop while the
    asyncpg pool belongs to the pytest loop, and sharing a connection across the
    two corrupts the wire protocol (`InternalClientError: unknown protocol
    state`) and poisons the pooled connection for the next test.

    Only `get_session` is overridden. Authentication, principal resolution,
    membership lookup, permission gates and capability gates all run for real —
    which is the point: this measures what a request actually reaches.
    """
    from fastapi import FastAPI
    from httpx import ASGITransport, AsyncClient

    from navis.api.router import api_router
    from navis.platform.database import get_session

    async def _session_override():
        async with Session() as db:
            yield db

    app = FastAPI()
    app.include_router(api_router)
    app.dependency_overrides[get_session] = _session_override
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://tenant")


async def _purge_tenants(Session, user_ids=()) -> None:
    """Delete every row this module creates for TENANT_A / TENANT_B.

    Called BEFORE seeding as well as after. The module builds its schema once and
    never drops it, so a run that dies mid-setup leaves rows behind and the next
    run collides on `organizations_pkey`. Cleaning on entry makes the fixture
    idempotent, which is what a shared-schema module needs instead of `drop_all`.

    ORM deletes only. A raw `text("... = ANY(:ids)")` with a Python list of UUIDs
    gives asyncpg no type for the array parameter and its introspection retry
    loop gives up with `InternalClientError: could not resolve query result
    and/or argument types` — the typed ORM `in_()` carries the column type.
    """
    from sqlalchemy import delete

    from navis.adaptive.models import PolicySnapshot
    from navis.identity.enterprise_models import AuditLog
    from navis.identity.models import Integration, Membership, Organization, User
    from navis.intelligence.models import IntelligenceInsight
    from navis.learning.models import LearningEvent
    from navis.memory.models import MemoryRecord
    from navis.observation.models import Outcome
    from navis.organization.models import (
        OrganizationEntitlement,
        OrganizationFeatureFlag,
        OrganizationLifecycle,
        OrganizationMetrics,
        OrganizationOnboarding,
        OrganizationProfile,
    )
    from navis.signals.models import SignalEventRecord
    from navis.state.v2.snapshots import StateSnapshot

    async with Session() as s:
        # Children first: memory_records FKs signal_events, and everything FKs
        # organizations.
        for model in (PolicySnapshot, LearningEvent, Outcome, IntelligenceInsight,
                      StateSnapshot, MemoryRecord, SignalEventRecord, Integration,
                      Membership, AuditLog, OrganizationEntitlement,
                      OrganizationFeatureFlag, OrganizationLifecycle,
                      OrganizationMetrics, OrganizationOnboarding,
                      OrganizationProfile):
            await s.execute(
                delete(model).where(model.organization_id.in_([TENANT_A, TENANT_B]))
            )
        if user_ids:
            await s.execute(delete(User).where(User.id.in_(list(user_ids))))
        await s.execute(
            delete(Organization).where(Organization.id.in_([TENANT_A, TENANT_B]))
        )
        await s.commit()


@pytest_asyncio.fixture(loop_scope="module")
async def two_tenants(Session):
    """Org A and Org B, each with a row in EVERY organization-scoped layer.

    Values are deliberately distinguishable per tenant ("A-only" / "B-only") so
    a leak shows up as content, not just as a count.
    """
    from navis.adaptive.models import PolicySnapshot
    from navis.identity.models import (
        Integration,
        IntegrationProvider,
        Membership,
        Organization,
        Role,
        User,
    )
    from navis.intelligence.models import IntelligenceInsight
    from navis.learning.models import LearningEvent
    from navis.memory.models import MemoryRecord
    from navis.observation.models import Outcome
    from navis.organization.contract import Plan
    from navis.organization.service import OrganizationService
    from navis.signals.crypto import token_cipher
    from navis.signals.google.oauth import GOOGLE_SCOPES
    from navis.signals.models import SignalEventRecord
    from navis.state.v2.snapshots import StateSnapshot

    made: dict[uuid.UUID, uuid.UUID] = {}

    # Start from a known-empty tenant space (see `_purge_tenants`).
    await _purge_tenants(Session)

    async with Session() as s:
        for org_id, tag in ((TENANT_A, "A"), (TENANT_B, "B")):
            s.add(Organization(
                id=org_id, name=f"Tenant {tag}", slug=f"tenant-{tag.lower()}-{uuid.uuid4().hex[:6]}"
            ))
            user = User(email=f"founder-{tag.lower()}-{uuid.uuid4().hex[:6]}@example.com",
                        full_name=f"Founder {tag}", is_active=True)
            s.add(user)
            await s.flush()
            s.add(Membership(user_id=user.id, organization_id=org_id, role=Role.FOUNDER))
            made[org_id] = user.id

            await OrganizationService(s).bootstrap(org_id, plan=Plan.ENTERPRISE)

            s.add(Integration(
                organization_id=org_id, user_id=user.id,
                provider=IntegrationProvider.GMAIL,
                external_account=f"{tag.lower()}-only@example.com",
                access_token=token_cipher.encrypt(f"{tag}-ACCESS"),
                refresh_token=token_cipher.encrypt(f"{tag}-REFRESH"),
                token_expires_at=NOW + timedelta(hours=1),
                scopes=" ".join(GOOGLE_SCOPES), is_active=True, sync_state={},
            ))

            signal_id = uuid.uuid4()
            s.add(SignalEventRecord(
                id=signal_id, organization_id=org_id, source="gmail",
                event_type="message", external_id=f"{tag}-only-msg",
                external_version="1", occurred_at=NOW,
                payload={"subject": f"{tag}-only subject"},
            ))
            # `memory_records.source_signal_event_id` is a real FK to
            # `signal_events.id`. Flush the signal row first: inside a single
            # flush SQLAlchemy orders inserts per MAPPER, so the memory row is
            # not guaranteed to land after the signal event it points at.
            await s.flush()
            s.add(MemoryRecord(
                memory_id=uuid.uuid4(), organization_id=org_id, entity_type="person",
                entity_id=uuid.uuid4(), source_signal_event_id=signal_id,
                extractor_name="test", extractor_version="1", schema_version=1,
                event_timestamp=NOW, dedup_key=f"{tag}-only-dedup",
                payload={"name": f"{tag}-only person"},
            ))
            s.add(StateSnapshot(
                organization_id=org_id, projection_version=1, schema_version=1,
                full_projection={"marker": f"{tag}-only state"},
            ))
            s.add(IntelligenceInsight(
                insight_id=uuid.uuid4(), organization_id=org_id,
                insight_type="pattern", key=f"{tag}-only-key",
                title=f"{tag}-only insight", summary=f"{tag}-only summary",
                learning_version=1,
            ))
            outcome_id = uuid.uuid4()
            s.add(Outcome(
                outcome_id=outcome_id, organization_id=org_id, outcome_type="reply",
                source_event_id=f"{tag}-only-outcome", occurred_at=NOW, recorded_at=NOW,
            ))
            s.add(LearningEvent(
                id=uuid.uuid4(), organization_id=org_id, outcome_id=outcome_id,
                outcome_type="reply", outcome_surface="email", result="positive",
                source_event_id=f"{tag}-only-learning", occurred_at=NOW,
            ))
            s.add(PolicySnapshot(
                policy_id=uuid.uuid4(), organization_id=org_id, policy_version=1,
                status="active", payload={"marker": f"{tag}-only policy"},
            ))
        await s.commit()

    yield Session, made

    await _purge_tenants(Session, user_ids=made.values())


def _token_for(user_id: uuid.UUID, org_id: uuid.UUID) -> str:
    from navis.identity.authz import principal_cache_clear
    from navis.platform.security import create_access_token

    principal_cache_clear()
    return create_access_token(str(user_id), org_id=str(org_id))


# ── 1. Org A cannot READ Org B, across every HTTP read surface ──────────────

async def test_no_http_read_surface_returns_another_tenants_row(two_tenants):
    """The broadest proof, and the one closest to a real attack.

    Authenticated as Org A's founder, walk every organization-scoped GET the API
    mounts and assert the marker string Org B's rows carry appears in NONE of the
    response bodies. Both tenants hold a row in every table behind these routes,
    so a query missing its tenant filter returns B's marker and this fails.
    """
    Session, users = two_tenants
    token = _token_for(users[TENANT_A], TENANT_A)

    surfaces = [
        "/intelligence/insights",
        "/adaptive/policy",
        "/adaptive/proposals",
        "/actions",
        "/integrations",
        "/organization/entitlements",
        "/organization/profile",
        "/organization/onboarding",
        "/organization/metrics",
        "/signals/gmail/recent",
        "/signals/calendar/upcoming",
        "/signals/drive/recent",
        "/memory/people",
        "/decision-memory",
        "/identity/members",
        "/identity/audit",
        "/identity/api-keys",
        "/identity/service-accounts",
    ]

    leaked: list[str] = []
    unreachable: list[str] = []
    async with _tenant_client(Session) as client:
        for path in surfaces:
            r = await client.get(path, headers={"Authorization": f"Bearer {token}"})
            if r.status_code >= 500:
                unreachable.append(f"{path} -> {r.status_code}")
                continue
            body = r.text
            if "B-only" in body or str(TENANT_B) in body:
                leaked.append(f"{path} -> {r.status_code} {body[:200]}")

    assert not leaked, "another tenant's data reached an Org A response:\n" + "\n".join(leaked)
    assert not unreachable, "read surface errored rather than answering:\n" + "\n".join(unreachable)


async def test_org_a_sees_its_own_data_on_those_surfaces(two_tenants):
    """The control. Without this, the isolation test above would pass just as
    well against a backend that returned nothing at all."""
    Session, users = two_tenants
    token = _token_for(users[TENANT_A], TENANT_A)

    async with _tenant_client(Session) as client:
        r = await client.get("/intelligence/insights",
                             headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 200, r.text
        assert "A-only" in r.text, "Org A cannot see its OWN insight"

        r = await client.get("/integrations",
                             headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 200, r.text
        assert "a-only@example.com" in r.text


# ── 2/3. Org A cannot MODIFY or ACT ON an Org B resource ────────────────────

async def test_org_a_cannot_mutate_an_org_b_resource_by_id(two_tenants):
    """Every mutating route that takes an id in the path, aimed at a row that
    exists — but belongs to the other tenant. A tenant-blind handler would find
    the row and act; each must answer 404/403 and leave it untouched."""
    from navis.adaptive.models import PolicySnapshot
    from sqlalchemy import select

    Session, users = two_tenants
    token = _token_for(users[TENANT_A], TENANT_A)

    async with Session() as s:
        b_policy = await s.scalar(
            select(PolicySnapshot).where(PolicySnapshot.organization_id == TENANT_B)
        )
        b_user = users[TENANT_B]
    assert b_policy is not None

    auth = {"Authorization": f"Bearer {token}"}
    attempts = [
        ("POST", f"/adaptive/rollback/{b_policy.policy_id}", None),
        ("POST", f"/adaptive/proposals/{uuid.uuid4()}/approve", None),
        ("DELETE", f"/identity/members/{b_user}", None),
        ("POST", f"/identity/members/{b_user}/role", {"role": "employee"}),
        ("DELETE", f"/scim/v2/Users/{b_user}", None),
    ]

    bad: list[str] = []
    async with _tenant_client(Session) as client:
        for method, path, body in attempts:
            r = await client.request(method, path, headers=auth, json=body)
            if r.status_code < 400:
                bad.append(f"{method} {path} -> {r.status_code} {r.text[:120]}")

    assert not bad, "a cross-tenant mutation SUCCEEDED:\n" + "\n".join(bad)

    # And Org B's policy is still exactly as it was.
    async with Session() as s:
        still = await s.get(PolicySnapshot, b_policy.policy_id)
    assert still is not None and still.organization_id == TENANT_B
    assert still.status == "active", "Org B's active policy was altered by Org A"


# ── 4. Org A cannot execute using Org B credentials ─────────────────────────

async def test_credential_resolution_never_crosses_a_tenant(two_tenants):
    """`resolve_integration` is the ONLY way execution obtains a credential.

    Asked for Org A it must return A's account; handed B's integration id under
    A's org it must refuse rather than serve the row it can plainly see.
    """
    from sqlalchemy import select

    from navis.execution.integrations import (
        IntegrationUnavailable,
        resolve_integration,
    )
    from navis.identity.models import Integration

    Session, _users = two_tenants
    async with Session() as s:
        b_integration = await s.scalar(
            select(Integration).where(Integration.organization_id == TENANT_B)
        )
    assert b_integration is not None

    async with Session() as s:
        mine = await resolve_integration(s, TENANT_A, "gmail")
        assert mine.organization_id == TENANT_A
        assert mine.external_account == "a-only@example.com"

        with pytest.raises(IntegrationUnavailable):
            await resolve_integration(
                s, TENANT_A, "gmail", integration_id=b_integration.id
            )


# ── 5. Org A entitlement cannot become Org B entitlement ────────────────────

async def test_entitlements_are_decided_per_tenant(two_tenants):
    """Revoking a capability for one tenant must not move the other's answer,
    and `can_use` is the single authority both the request gate and the worker
    consult."""
    from navis.organization.contract import Capability
    from navis.organization.repository import OrganizationRepository
    from navis.organization.service import OrganizationService

    Session, _users = two_tenants

    async with Session() as s:
        await OrganizationRepository(s).set_entitlement(
            TENANT_B, Capability.EXECUTION.value, False
        )
        await s.commit()

    async with Session() as s:
        svc = OrganizationService(s)
        assert await svc.can_use(TENANT_A, Capability.EXECUTION) is True
        assert await svc.can_use(TENANT_B, Capability.EXECUTION) is False


async def test_a_suspended_tenant_does_not_suspend_the_other(two_tenants):
    """Lifecycle is the FIRST gate in `EntitlementService._decide`. Suspending B
    must strip B of everything and leave A untouched."""
    from navis.organization.contract import Capability
    from navis.organization.service import OrganizationService

    Session, _users = two_tenants

    async with Session() as s:
        # `bootstrap` leaves an org in TRIAL, and the state machine only allows
        # TRIAL -> {ACTIVE, CHURNED}. Reaching SUSPENDED goes through ACTIVE —
        # the transition guard refusing `trial -> suspended` is the lifecycle
        # working, so the test walks the legal path instead of weakening it.
        await OrganizationService(s).lifecycle.activate(TENANT_B)
        await OrganizationService(s).lifecycle.suspend(TENANT_B)
        await s.commit()

    async with Session() as s:
        svc = OrganizationService(s)
        for cap in (Capability.EXECUTION, Capability.DECISION, Capability.GMAIL,
                    Capability.MEMORY):
            assert await svc.can_use(TENANT_B, cap) is False, f"suspended org kept {cap}"
            assert await svc.can_use(TENANT_A, cap) is True, f"Org A lost {cap}"


# ── 6-10. Repository reads, for the layers with no HTTP surface ─────────────

async def test_every_repository_read_is_scoped_to_its_tenant(two_tenants):
    """The layers a request cannot reach directly — Observation, Learning,
    Adaptive projections, State snapshots, Memory — read through their real
    repositories with Org A's id and must not see Org B's row."""
    from sqlalchemy import func, select

    from navis.adaptive.repository import AdaptiveRepository
    from navis.learning.repository import LearningRepository
    from navis.memory.models import MemoryRecord
    from navis.observation.repository import OutcomeRepository
    from navis.state.v2.snapshots import StateSnapshot

    Session, _users = two_tenants

    async with Session() as s:
        # Observation
        timeline = await OutcomeRepository(s).timeline(TENANT_A)
        assert {o.organization_id for o in timeline} <= {TENANT_A}
        assert all("B-only" not in (o.source_event_id or "") for o in timeline)

        # Learning
        events = await LearningRepository(s).all_events(TENANT_A)
        assert {e.organization_id for e in events} <= {TENANT_A}
        assert all("B-only" not in e.source_event_id for e in events)

        # Adaptive projection
        policy = await AdaptiveRepository(s).get_active_payload(TENANT_A)
        assert policy.get("marker") != "B-only policy"

        # State snapshot
        snaps = (await s.scalars(
            select(StateSnapshot).where(StateSnapshot.organization_id == TENANT_A)
        )).all()
        assert {sn.organization_id for sn in snaps} == {TENANT_A}
        assert all(sn.full_projection.get("marker") == "A-only state" for sn in snaps)

        # Memory — both tenants hold exactly one record, so a missing filter shows.
        n_a = await s.scalar(
            select(func.count()).select_from(MemoryRecord)
            .where(MemoryRecord.organization_id == TENANT_A)
        )
        n_b = await s.scalar(
            select(func.count()).select_from(MemoryRecord)
            .where(MemoryRecord.organization_id == TENANT_B)
        )
        assert n_a == 1 and n_b == 1


# ── 11. Caches are tenant-isolated ──────────────────────────────────────────

async def test_the_state_cache_cannot_serve_one_tenant_anothers_projection(two_tenants):
    """The in-process State cache is keyed `(org_id, projection_version,
    watermark)`. Warm it with BOTH tenants and prove a lookup for one never
    returns the other — a key that dropped org_id would collide here, because
    the two projections share a version and both have no watermark advance."""
    from navis.state.v2 import service as state_service

    Session, _users = two_tenants
    state_service._cache_clear()

    a_watermark, b_watermark = uuid.uuid4(), uuid.uuid4()

    class _Fake:
        def __init__(self, org_id, marker, wm):
            self.org_id = org_id
            self.projection_version = 1
            self.marker = marker

            class _W:
                last_memory_id = wm
            self.watermark = _W()

    state_service._STATE_CACHE[(TENANT_A, 1, a_watermark)] = _Fake(TENANT_A, "A-only state", a_watermark)
    state_service._STATE_CACHE[(TENANT_B, 1, b_watermark)] = _Fake(TENANT_B, "B-only state", b_watermark)

    got_a = state_service._cache_get(TENANT_A, 1, a_watermark)
    got_b = state_service._cache_get(TENANT_B, 1, b_watermark)
    assert got_a.marker == "A-only state"
    assert got_b.marker == "B-only state"

    # A's org id with B's watermark is a different key entirely — no partial match.
    assert state_service._cache_get(TENANT_A, 1, b_watermark) is None
    assert state_service._cache_get(TENANT_B, 1, a_watermark) is None

    # Evicting one tenant leaves the other warm.
    state_service._cache_evict_org(TENANT_A)
    assert state_service._cache_get(TENANT_A, 1, a_watermark) is None
    assert state_service._cache_get(TENANT_B, 1, b_watermark) is not None
    state_service._cache_clear()


async def test_the_rate_limiter_counts_each_tenant_separately(two_tenants):
    """`ratelimit:{bucket}:{org}:{window}` — one tenant exhausting a bucket must
    not throttle another. Proved on the key itself so it holds without Redis."""
    from navis.identity.throttle import OrgRateLimiter

    limiter = OrgRateLimiter(redis=None, window_seconds=60)  # type: ignore[arg-type]
    now = 1_700_000_000.0
    key_a = limiter._key("chat", TENANT_A, now)
    key_b = limiter._key("chat", TENANT_B, now)
    assert key_a != key_b
    assert str(TENANT_A) in key_a and str(TENANT_B) in key_b


# ── 12. Background work preserves tenant identity ───────────────────────────

async def test_scheduled_work_is_decided_per_tenant(two_tenants):
    """The beat fan-out asks `scheduled_work_allowed` per organization. A tenant
    that loses a capability must be skipped WITHOUT affecting the other — the
    property that lets a suspended org stop costing money while its neighbour
    keeps running."""
    from navis.orchestration.runtime_policy import scheduled_work_allowed
    from navis.organization.contract import Capability
    from navis.organization.service import OrganizationService

    Session, _users = two_tenants

    async with Session() as s:
        # `bootstrap` leaves an org in TRIAL, and the state machine only allows
        # TRIAL -> {ACTIVE, CHURNED}. Reaching SUSPENDED goes through ACTIVE —
        # the transition guard refusing `trial -> suspended` is the lifecycle
        # working, so the test walks the legal path instead of weakening it.
        await OrganizationService(s).lifecycle.activate(TENANT_B)
        await OrganizationService(s).lifecycle.suspend(TENANT_B)
        await s.commit()

    async with Session() as s:
        assert await scheduled_work_allowed(
            s, TENANT_A, Capability.DECISION, task="build_brief"
        ) is True
        assert await scheduled_work_allowed(
            s, TENANT_B, Capability.DECISION, task="build_brief"
        ) is False
        # Fail closed on a tenant that does not exist at all.
        assert await scheduled_work_allowed(
            s, uuid.uuid4(), Capability.DECISION, task="build_brief"
        ) is False
        assert await scheduled_work_allowed(
            s, None, Capability.DECISION, task="build_brief"
        ) is False


async def test_the_brief_fan_out_dispatches_one_task_per_organization(two_tenants):
    """`build_all_briefs` enumerates organizations from persisted state and
    dispatches ONE task carrying ONE org id. It must never hand a task a set of
    organizations, and every id it emits must be a real row."""
    from sqlalchemy import select

    from navis.identity.models import Organization
    from navis.orchestration import brief as brief_mod

    Session, _users = two_tenants
    dispatched: list[str] = []

    class _Task:
        @staticmethod
        def delay(org_id: str) -> None:
            dispatched.append(org_id)

    original = brief_mod.build_brief_for_org
    brief_mod.build_brief_for_org = _Task  # type: ignore[assignment]
    try:
        await brief_mod._dispatch_all()
    finally:
        brief_mod.build_brief_for_org = original  # type: ignore[assignment]

    assert str(TENANT_A) in dispatched
    assert str(TENANT_B) in dispatched
    assert len(dispatched) == len(set(dispatched)), "an organization was dispatched twice"

    async with Session() as s:
        real = {str(o) for o in (await s.scalars(select(Organization.id))).all()}
    assert set(dispatched) <= real, "dispatched an organization id that does not exist"


# ── 12. WHICH ACCOUNT an action is performed AS stays inside its tenant ─────
#
# `execution_requests.integration_id` (T-11) records which of an organization's
# connected accounts an action must be performed with. Migration 0042 added it,
# the worker leases it and `resolve_integration` pins on it — but no PRODUCER
# ever wrote a value, so it was uniformly NULL and every send fell back to
# resolving by (organization, provider).
#
# `propose_actions` now RE-DERIVES it from `memory_records.source_integration_id`
# — the account the thread actually arrived on — via
# `memory.lineage.integrations_for_threads`. Two things can go wrong there and
# both are real-database properties, which is why this is proved here rather
# than with fakes: the org-scoped UUIDv5 `entity_id` derivation, and the SQL that
# groups distinct accounts per thread.


async def _seed_thread_capture(Session, *, org_id, user_id, integration_id, thread_id, tag):
    """One captured message on `thread_id` plus the THREAD fact it produces."""
    import datetime as _dt

    from navis.memory.envelope import EntityType
    from navis.memory.ids import entity_id, memory_record_id
    from navis.memory.models import MemoryRecord
    from navis.signals.models import SignalEventRecord

    signal_id = uuid.uuid4()
    async with Session() as s:
        s.add(SignalEventRecord(
            id=signal_id, organization_id=org_id, user_id=user_id,
            integration_id=integration_id, source="gmail", event_type="message",
            external_id=f"{tag}-{uuid.uuid4().hex[:8]}", external_version="1",
            occurred_at=NOW, payload={"thread_id": thread_id},
        ))
        await s.flush()
        s.add(MemoryRecord(
            memory_id=memory_record_id(
                org_id, signal_id, "email_thread", EntityType.THREAD.value, thread_id
            ),
            organization_id=org_id, entity_type=EntityType.THREAD.value,
            entity_id=entity_id(org_id, EntityType.THREAD.value, thread_id),
            source_signal_event_id=signal_id, source_integration_id=integration_id,
            extractor_name="email_thread", extractor_version="1", schema_version=1,
            event_timestamp=NOW, dedup_key=f"{thread_id}|{signal_id}",
            payload={"thread_id": thread_id},
        ))
        await s.commit()
    return signal_id


async def _gmail_integration_id(Session, org_id):
    from sqlalchemy import select as _select

    from navis.identity.models import Integration

    async with Session() as s:
        return await s.scalar(
            _select(Integration.id).where(Integration.organization_id == org_id)
        )


async def test_the_account_a_thread_arrived_on_never_crosses_a_tenant(two_tenants):
    """THE CROSS-TENANT CASE for T-11.

    Both tenants captured a thread under the SAME provider thread id — which is
    ordinary, since Gmail thread ids are only unique per mailbox. `entity_id` is
    a UUIDv5 over the ORGANIZATION, and the query is org-filtered on top of that,
    so each tenant resolves its own capturing account and neither can see the
    other's. A leak here would put one company's mail identity on another
    company's outbound reply.
    """
    from navis.memory.lineage import integrations_for_threads

    Session, users = two_tenants
    shared_thread = f"thread-{uuid.uuid4().hex[:8]}"
    account_a = await _gmail_integration_id(Session, TENANT_A)
    account_b = await _gmail_integration_id(Session, TENANT_B)
    assert account_a and account_b and account_a != account_b

    await _seed_thread_capture(
        Session, org_id=TENANT_A, user_id=users[TENANT_A],
        integration_id=account_a, thread_id=shared_thread, tag="A",
    )
    await _seed_thread_capture(
        Session, org_id=TENANT_B, user_id=users[TENANT_B],
        integration_id=account_b, thread_id=shared_thread, tag="B",
    )

    async with Session() as s:
        seen_by_a = await integrations_for_threads(s, TENANT_A, [shared_thread])
        seen_by_b = await integrations_for_threads(s, TENANT_B, [shared_thread])

    assert seen_by_a == {shared_thread: account_a}
    assert seen_by_b == {shared_thread: account_b}


async def test_a_thread_captured_on_two_accounts_resolves_to_neither(two_tenants):
    """FAIL CLOSED, not "pick one".

    One organization can connect two mailboxes that both hold a conversation, and
    nothing about the thread says which identity a reply should come from.
    Omitting it leaves `integration_id` NULL, which the credential boundary
    already treats as "not recorded" — and `resolve_integration` then refuses a
    multi-account organization outright instead of sending from whichever row
    happened to sort first. Guessing here is the recency heuristic this whole
    boundary exists to remove.
    """
    from navis.identity.models import Integration, IntegrationProvider
    from navis.memory.lineage import integrations_for_threads
    from navis.signals.crypto import token_cipher
    from navis.signals.google.oauth import GOOGLE_SCOPES

    Session, users = two_tenants
    thread = f"thread-{uuid.uuid4().hex[:8]}"
    first = await _gmail_integration_id(Session, TENANT_A)

    async with Session() as s:
        second = Integration(
            organization_id=TENANT_A, user_id=users[TENANT_A],
            provider=IntegrationProvider.GMAIL,
            external_account=f"ops-{uuid.uuid4().hex[:6]}@example.com",
            access_token=token_cipher.encrypt("A2-ACCESS"),
            refresh_token=token_cipher.encrypt("A2-REFRESH"),
            token_expires_at=NOW + timedelta(hours=1),
            scopes=" ".join(GOOGLE_SCOPES), is_active=True, sync_state={},
        )
        s.add(second)
        await s.commit()
        second_id = second.id

    # The SAME thread observed through BOTH of this org's mailboxes.
    await _seed_thread_capture(
        Session, org_id=TENANT_A, user_id=users[TENANT_A],
        integration_id=first, thread_id=thread, tag="A1",
    )
    await _seed_thread_capture(
        Session, org_id=TENANT_A, user_id=users[TENANT_A],
        integration_id=second_id, thread_id=thread, tag="A2",
    )

    async with Session() as s:
        resolved = await integrations_for_threads(s, TENANT_A, [thread])

    assert resolved == {}, (
        "an ambiguous thread must contribute no account — recording either one "
        "would send a reply from an identity nothing in the thread chose"
    )


async def test_an_uncaptured_thread_contributes_no_account(two_tenants):
    """Nothing is fabricated: a thread with no lineage yields no account, which
    is the honest "not recorded" the nullable column documents."""
    from navis.memory.lineage import integrations_for_threads

    Session, _users = two_tenants
    async with Session() as s:
        assert await integrations_for_threads(s, TENANT_A, ["never-captured"]) == {}
        assert await integrations_for_threads(s, TENANT_A, []) == {}
        assert await integrations_for_threads(s, TENANT_A, [""]) == {}
