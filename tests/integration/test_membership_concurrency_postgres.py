"""TASK 6.4.3 Phase 10 — the last-founder race, against real PostgreSQL.

The fake session cannot model row locking, so the one invariant that depends on it
is proved here: two concurrent privileged operations must not each pass their own
last-founder check and together leave the organization with zero founders.

    NAVIS_TEST_DATABASE_URL=postgresql+asyncpg://.../navis_test?ssl=require \
        pytest tests/integration/test_membership_concurrency_postgres.py -q
"""
from __future__ import annotations

import asyncio
import os
import uuid

import pytest
import pytest_asyncio

pytestmark = [
    pytest.mark.skipif(
        not os.environ.get("NAVIS_TEST_DATABASE_URL"),
        reason="set NAVIS_TEST_DATABASE_URL to run Postgres integration tests",
    ),
    pytest.mark.asyncio(loop_scope="module"),
]


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def engine():
    from sqlalchemy.ext.asyncio import create_async_engine

    import navis.registry  # noqa: F401  register all models
    from navis.platform.database import Base

    url = os.environ["NAVIS_TEST_DATABASE_URL"]
    name = url.rsplit("/", 1)[-1].split("?")[0]
    assert name.startswith("navis_test"), f"refusing to run against database {name!r}"

    eng = create_async_engine(url)
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield eng
    await eng.dispose()


@pytest_asyncio.fixture(loop_scope="module")
async def org(engine):
    """An org with TWO founders, plus a cleanup that removes only what we made."""
    from sqlalchemy import delete
    from sqlalchemy.ext.asyncio import async_sessionmaker

    from navis.identity.models import Membership, Organization, Role, User

    Session = async_sessionmaker(engine, expire_on_commit=False)
    async with Session() as db:
        o = Organization(name="Race", slug=f"race-{uuid.uuid4().hex[:8]}")
        db.add(o)
        await db.flush()
        founders = []
        for _ in range(2):
            u = User(email=f"f-{uuid.uuid4().hex[:8]}@race.test")
            db.add(u)
            await db.flush()
            db.add(Membership(user_id=u.id, organization_id=o.id, role=Role.FOUNDER))
            founders.append(u.id)
        await db.commit()
        org_id = o.id

    yield Session, org_id, founders

    async with Session() as db:
        await db.execute(delete(Membership).where(Membership.organization_id == org_id))
        await db.execute(delete(Organization).where(Organization.id == org_id))
        await db.commit()


async def _founder_count(Session, org_id) -> int:
    from sqlalchemy import func, select

    from navis.identity.models import Membership, Role

    async with Session() as db:
        return int(
            await db.scalar(
                select(func.count()).select_from(Membership).where(
                    Membership.organization_id == org_id, Membership.role == Role.FOUNDER
                )
            )
            or 0
        )


async def test_concurrent_remove_and_demote_cannot_zero_the_founders(org):
    """Request A removes founder #1 while Request B demotes founder #2.

    Both read "2 founders" if they are allowed to run unserialized, and both then
    believe they are safe. `_count_founders_locked` takes `SELECT ... FOR UPDATE`
    over the org's founder rows, so the second transaction blocks and re-evaluates
    against the first one's committed result.
    """
    from navis.identity.members import MemberConflict, MemberService
    from navis.identity.models import Role
    from navis.identity.permissions import default_permissions_for_role

    Session, org_id, (f1, f2) = org
    FOUNDER = default_permissions_for_role("founder")
    assert await _founder_count(Session, org_id) == 2

    started = asyncio.Event()

    async def remove_one():
        async with Session() as db:
            await MemberService(db).remove(
                org_id=org_id, user_id=f1, actor_id=f1, actor_permissions=FOUNDER
            )
            started.set()
            # Hold the transaction open so the demotion below genuinely overlaps.
            await asyncio.sleep(0.4)
            await db.commit()

    async def demote_other():
        await started.wait()
        async with Session() as db:
            await MemberService(db).assign_role(
                org_id=org_id, user_id=f2, role=Role.EMPLOYEE,
                actor_id=f2, actor_permissions=FOUNDER,
            )
            await db.commit()

    results = await asyncio.gather(remove_one(), demote_other(), return_exceptions=True)

    # Exactly one must succeed; the loser must be refused by the governance guard.
    failures = [r for r in results if isinstance(r, Exception)]
    assert len(failures) == 1, f"expected exactly one refusal, got {results!r}"
    assert isinstance(failures[0], MemberConflict), failures[0]

    remaining = await _founder_count(Session, org_id)
    assert remaining == 1, "the organization must never be left without a founder"


async def test_concurrent_double_demotion_leaves_one_founder(org):
    from navis.identity.members import MemberConflict, MemberService
    from navis.identity.models import Role
    from navis.identity.permissions import default_permissions_for_role

    Session, org_id, (f1, f2) = org
    FOUNDER = default_permissions_for_role("founder")

    async def demote(uid, delay):
        await asyncio.sleep(delay)
        async with Session() as db:
            await MemberService(db).assign_role(
                org_id=org_id, user_id=uid, role=Role.EMPLOYEE,
                actor_id=uid, actor_permissions=FOUNDER,
            )
            await db.commit()

    results = await asyncio.gather(
        demote(f1, 0.0), demote(f2, 0.0), return_exceptions=True
    )
    failures = [r for r in results if isinstance(r, Exception)]
    assert len(failures) == 1, f"expected exactly one refusal, got {results!r}"
    assert isinstance(failures[0], MemberConflict), failures[0]
    assert await _founder_count(Session, org_id) == 1


# ═════════════════════════════════════════════════════════════════════════════
# TASK 6.4.4 — tenant isolation of the authorization READ path, on real Postgres
# ═════════════════════════════════════════════════════════════════════════════
#
# `/auth/me` now publishes the caller's effective permissions, which means a
# tenancy bug in permission RESOLUTION is no longer only an enforcement bug —
# it is a disclosure. The unit tests prove the shape against a fake session;
# these prove it against the real query planner, the real `IN (org_id, NULL)`
# predicate, and the real enum column, with two organizations that both have an
# `employee` (the ordinary case, not an exotic one).


def _asgi_client(Session):
    """An HTTP client that drives the identity router ON THIS EVENT LOOP.

    NOT `TestClient`. Starlette's TestClient runs the ASGI app in a portal with
    its OWN event loop, while the engine and its connection pool here were
    created on the pytest-asyncio loop. Handing asyncpg connections between two
    loops corrupts the protocol — it surfaces as
    `InternalClientError: got result for unknown protocol state 3` and then as
    cascading failures in whatever test runs next, because the poisoned
    connection goes back into the pool. `httpx.ASGITransport` awaits the app
    directly, so the request, the session and the pool all share one loop.
    """
    from fastapi import FastAPI
    from httpx import ASGITransport, AsyncClient

    from navis.identity.router import router as identity_router
    from navis.platform.database import get_session

    async def _session_override():
        async with Session() as db:
            yield db

    app = FastAPI()
    app.include_router(identity_router)
    app.dependency_overrides[get_session] = _session_override
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


@pytest_asyncio.fixture(loop_scope="module")
async def two_orgs(engine):
    """Two organizations, one user holding a seat in BOTH.

    Org A grants its employees nothing beyond the code defaults. Org B grants
    its employees EXECUTE_ACTIONS. Same role string, same user, two tenants —
    the exact configuration in which a missing tenant filter is invisible in
    single-org testing and catastrophic in production.
    """
    from sqlalchemy import delete
    from sqlalchemy.ext.asyncio import async_sessionmaker

    from navis.identity.enterprise_models import RolePermission
    from navis.identity.models import Membership, Organization, Role, User

    Session = async_sessionmaker(engine, expire_on_commit=False)
    async with Session() as db:
        a = Organization(name="Tenant A", slug=f"tenant-a-{uuid.uuid4().hex[:8]}")
        b = Organization(name="Tenant B", slug=f"tenant-b-{uuid.uuid4().hex[:8]}")
        db.add_all([a, b])
        await db.flush()
        user = User(email=f"dual-{uuid.uuid4().hex[:8]}@example.com", is_active=True)
        db.add(user)
        await db.flush()
        db.add(Membership(user_id=user.id, organization_id=a.id, role=Role.EMPLOYEE))
        db.add(Membership(user_id=user.id, organization_id=b.id, role=Role.EMPLOYEE))
        # Only Org B grants the extra permission.
        db.add(RolePermission(organization_id=b.id, role="employee",
                              permission="execute_actions"))
        await db.commit()
        ids = (a.id, b.id, user.id)

    yield Session, ids

    a_id, b_id, user_id = ids
    async with Session() as db:
        await db.execute(
            delete(RolePermission).where(RolePermission.organization_id.in_([a_id, b_id]))
        )
        await db.execute(delete(Membership).where(Membership.user_id == user_id))
        await db.execute(delete(User).where(User.id == user_id))
        await db.execute(delete(Organization).where(Organization.id.in_([a_id, b_id])))
        await db.commit()


async def test_org_a_principal_never_carries_an_org_b_grant(two_orgs):
    """The isolation itself, through the real resolution path.

    Not an injected Principal: this mints an access token, runs
    `_principal_from_jwt` against Postgres, and reads what came back.
    """
    from navis.identity.authz import _principal_from_jwt, principal_cache_clear
    from navis.identity.permissions import default_permissions_for_role
    from navis.platform.security import create_access_token

    Session, (a_id, b_id, user_id) = two_orgs
    principal_cache_clear()

    async with Session() as db:
        in_a = await _principal_from_jwt(db, create_access_token(str(user_id), org_id=str(a_id)))
    assert in_a.org_id == a_id
    assert "execute_actions" not in in_a.permissions, "Org B's grant reached Org A"
    assert in_a.permissions == default_permissions_for_role("employee")

    # The same user, same role string, other tenant — the grant applies there.
    principal_cache_clear()
    async with Session() as db:
        in_b = await _principal_from_jwt(db, create_access_token(str(user_id), org_id=str(b_id)))
    assert in_b.org_id == b_id
    assert "execute_actions" in in_b.permissions, (
        "the grant did not apply in its OWN org — the previous assertion would then "
        "pass for the wrong reason (a query that returns nothing anywhere)"
    )


async def test_auth_me_reports_only_the_authenticated_organizations_permissions(two_orgs):
    """End to end over HTTP, no dependency overridden except the session.

    A real bearer token goes in; the response's `permissions` must describe the
    org that token authenticated into and no other, while `organizations` lists
    both seats WITHOUT any authorization state attached to the one that is not
    current.
    """
    from navis.identity.authz import principal_cache_clear
    from navis.platform.security import create_access_token

    Session, (a_id, b_id, user_id) = two_orgs
    principal_cache_clear()

    async with _asgi_client(Session) as client:
        token_a = create_access_token(str(user_id), org_id=str(a_id))
        r = await client.get("/auth/me", headers={"Authorization": f"Bearer {token_a}"})
        assert r.status_code == 200, r.text
        body = r.json()

        assert body["organization"]["id"] == str(a_id)
        assert body["role"] == "employee"
        assert "execute_actions" not in body["permissions"], "Org B's grant leaked over HTTP"
        # `view_signals` ALONE, and the missing one is P1-6 rather than a bug.
        # `employee`'s literal in `DEFAULT_ROLE_PERMISSIONS` still names
        # VIEW_DECISIONS, but `_application_only` strips it: VIEW_DECISIONS is
        # in `DEVELOPER_PERMISSIONS`, so no Application role carries it and the
        # Developer surface is reached through `memberships.developer_role`.
        #
        # Kept as an EXACT set rather than relaxed to a membership check. The
        # assertion above proves the cross-tenant grant did not leak; this one
        # proves nothing ELSE arrived either, which is the property that would
        # decay first if a future role change quietly widened the default.
        assert sorted(body["permissions"]) == ["view_signals"]

        # Multi-org is REPORTED (F-2) — both seats visible, one flagged current,
        # neither non-current entry carrying a role or permission.
        listed = {o["id"]: o for o in body["organizations"]}
        assert set(listed) == {str(a_id), str(b_id)}
        assert listed[str(a_id)]["is_current"] is True
        assert listed[str(b_id)]["is_current"] is False
        for entry in body["organizations"]:
            assert set(entry) == {"id", "name", "slug", "is_current"}

        # Authenticating into the other tenant changes the answer, and only the answer.
        principal_cache_clear()
        token_b = create_access_token(str(user_id), org_id=str(b_id))
        r_b = await client.get("/auth/me", headers={"Authorization": f"Bearer {token_b}"})
        body_b = r_b.json()
    assert body_b["organization"]["id"] == str(b_id)
    assert "execute_actions" in body_b["permissions"]


async def test_auth_me_is_401_without_credentials(two_orgs):
    """The contract a compat stub used to break: no credential, no data. Not a
    200 with an empty body that a client renders as 'nothing to see today'."""
    Session, _ids = two_orgs

    async with _asgi_client(Session) as client:
        assert (await client.get("/auth/me")).status_code == 401
        bad = await client.get("/auth/me", headers={"Authorization": "Bearer not-a-jwt"})
    assert bad.status_code == 401


async def test_a_token_naming_an_org_the_user_left_is_refused(two_orgs):
    """The JWT `org` claim is a lookup key, never tenant authority.

    Proved by REMOVING the Org B membership and reusing a token that still
    names Org B — the claim is unchanged and still validly signed; only the
    database changed. Resolution must follow the database.
    """
    from sqlalchemy import delete, select

    from fastapi import HTTPException

    from navis.identity.authz import _principal_from_jwt, principal_cache_clear
    from navis.identity.models import Membership, Role
    from navis.platform.security import create_access_token

    Session, (a_id, b_id, user_id) = two_orgs
    principal_cache_clear()
    token_b = create_access_token(str(user_id), org_id=str(b_id))

    async with Session() as db:
        await db.execute(
            delete(Membership).where(
                Membership.user_id == user_id, Membership.organization_id == b_id
            )
        )
        await db.commit()
    try:
        principal_cache_clear()
        async with Session() as db:
            with pytest.raises(HTTPException) as exc:
                await _principal_from_jwt(db, token_b)
        assert exc.value.status_code == 403, "a token for a revoked seat still resolved"

        # And it did NOT silently fall back to the org the user IS in.
        principal_cache_clear()
        async with Session() as db:
            still_a = await _principal_from_jwt(
                db, create_access_token(str(user_id), org_id=str(a_id))
            )
        assert still_a.org_id == a_id
    finally:
        async with Session() as db:
            exists = await db.scalar(
                select(Membership).where(
                    Membership.user_id == user_id, Membership.organization_id == b_id
                )
            )
            if exists is None:
                db.add(Membership(user_id=user_id, organization_id=b_id, role=Role.EMPLOYEE))
                await db.commit()


async def test_an_entitlement_in_one_org_does_not_reach_the_other(two_orgs):
    """Org A entitlement cannot become Org B entitlement.

    `can_use` is the single capability authority every business domain calls.
    Provisioning one tenant on a plan that carries a capability must not make
    that capability answerable for a tenant provisioned without it.
    """
    from navis.organization.contract import Capability, Plan
    from navis.organization.service import OrganizationService

    Session, (a_id, b_id, _user_id) = two_orgs

    async with Session() as db:
        await OrganizationService(db).bootstrap(a_id, plan=Plan.ENTERPRISE)
        await OrganizationService(db).bootstrap(b_id, plan=Plan.FREE)
        await db.commit()

    async with Session() as db:
        svc = OrganizationService(db)
        a_can = await svc.can_use(a_id, Capability.EXECUTION)
        b_can = await svc.can_use(b_id, Capability.EXECUTION)

    assert a_can is True, "the enterprise tenant lost its own capability"
    assert b_can is False, "the free tenant inherited another organization's entitlement"


async def test_membership_role_is_not_null_in_the_live_schema(engine):
    """Migration 0043's invariant, asserted against the actual column.

    `Base.metadata.create_all` builds this database from the ORM, so this pins
    the model; the migration pins the deployed schema and refuses to apply while
    any NULL role exists.
    """
    from sqlalchemy import text

    async with engine.connect() as conn:
        is_nullable = (
            await conn.execute(
                text(
                    "SELECT is_nullable FROM information_schema.columns "
                    "WHERE table_name = 'memberships' AND column_name = 'role'"
                )
            )
        ).scalar_one()
    assert is_nullable == "NO", "memberships.role is nullable in the live schema"
