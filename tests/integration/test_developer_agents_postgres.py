"""The Agent lifecycle, proved at the HTTP wire against real PostgreSQL.

    NAVIS_TEST_DATABASE_URL=postgresql+asyncpg://.../navis_test?ssl=require \
        EVENT_HANDLER_TIMEOUT_SECONDS=180 \
        pytest tests/integration/test_developer_agents_postgres.py -q

WHAT THIS PROVES, AND WHY AT THE WIRE. `/api/v1/developer/agents` is the first
MUTABLE surface on the Developer boundary, and a write is where tenant isolation
is easiest to lose: a read that forgets its tenant filter returns another
organization's row, but a WRITE that forgets one CHANGES it. So the whole chain
runs for real — credential resolution, principal resolution, membership lookup,
permission gate, capability gate, human-operator gate, service layer, repository
— and only `get_session` is overridden, to point at a disposable database.

The properties, in the order the module asserts them:

  1. an operator creates an agent in their OWN organization and reads it back;
  2. the plaintext secret is returned EXACTLY once and is unrecoverable after;
  3. neither the agent's organization nor its owner can be named by the client —
     `organization_id` in the body changes nothing;
  4. Org A cannot read, edit, revoke or credential Org B's agent by id, on any
     route, and every attempt is a 404 rather than a 403;
  5. an agent's own credential authenticates as that agent and reaches exactly
     the permissions it was granted — no more;
  6. revoking an agent kills every credential it holds, immediately;
  7. revoking ONE credential leaves the agent and its other credentials working;
  8. a machine principal cannot create, edit or revoke an agent;
  9. an operator cannot grant an agent a permission they do not hold themselves;
 10. permission and capability stay separate gates with separate denials;
 11. editing an agent's permissions changes what its LIVE credential can reach;
 12. environment and runtime are closed vocabularies, refused at the edge;
 13. the per-organization write limit and the active-agent ceiling both fire, and
     one tenant's burst never spends another's budget;
 14. 24 interleaved concurrent requests from both tenants never cross;
 15. a REMOVED MEMBER's agent credential keeps working — characterised, not
     asserted as desirable. See the test's own docstring.

MODULE STRUCTURE mirrors `test_developer_boundary_postgres.py` and for the same
reasons documented there: a MODULE-scoped event loop and engine, schema built
once and never dropped, per-module row cleanup instead of `drop_all`, and `httpx`
over `ASGITransport` rather than Starlette's `TestClient`.

SAFETY. Refuses to run unless the target database is named `navis_test*`, and
never drops a schema.
"""
from __future__ import annotations

import asyncio
import os
import uuid
from contextlib import asynccontextmanager

import pytest
import pytest_asyncio

pytestmark = [
    pytest.mark.skipif(
        not os.environ.get("NAVIS_TEST_DATABASE_URL"),
        reason="set NAVIS_TEST_DATABASE_URL (disposable Postgres) to run these",
    ),
    pytest.mark.asyncio(loop_scope="module"),
]

TENANT_A = uuid.UUID("aaaa0000-0000-4000-8000-00000000000a")
TENANT_B = uuid.UUID("aaaa0000-0000-4000-8000-00000000000b")

# The scopes an agent is created with here. Spelled out rather than "everything":
# a credential holding exactly these must reach exactly these.
AGENT_SCOPES = ["view_state", "view_memory"]


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def engine():
    """ONE engine and ONE `create_all` for the whole module. Never `drop_all`."""
    from sqlalchemy.ext.asyncio import create_async_engine

    import navis.registry  # noqa: F401  register every model
    from navis.platform.database import Base

    url = os.environ["NAVIS_TEST_DATABASE_URL"]
    name = url.rsplit("/", 1)[-1].split("?")[0]
    assert name.startswith("navis_test"), (
        f"refusing to run against database {name!r}: point "
        "NAVIS_TEST_DATABASE_URL at a disposable database named navis_test*"
    )

    # Sized for the concurrency test, which issues 24 simultaneous requests and
    # takes one session per request under the `get_session` override.
    eng = create_async_engine(url, pool_size=30, max_overflow=10)
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield eng
    await eng.dispose()


@pytest_asyncio.fixture(loop_scope="module")
async def Session(engine, monkeypatch):
    """A sessionmaker on the module engine, installed everywhere production would
    otherwise reach for the global one — which points at the URL in `.env`."""
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

    from navis.platform.events import event_bus

    monkeypatch.setattr(event_bus, "_mirror_to_redis", False, raising=False)
    return maker


def _client(Session):
    """httpx over the FULL mounted api_router, on this event loop.

    Only `get_session` is overridden. Authentication, principal resolution,
    permission gates, capability gates, the human-operator gate and rate limits
    all run for real.
    """
    from fastapi import FastAPI
    from httpx import ASGITransport, AsyncClient

    from navis.api.router import api_router
    from navis.platform.database import get_session

    async def _session_override():
        """Mirrors production `get_session` EXACTLY, including the commit.

        The read-side boundary module can omit the commit because it only reads.
        A lifecycle module cannot: without it every write in this file would be
        rolled back at session close, each request would still answer 201 from
        the flushed row, and the very next read would find nothing — which is a
        harness bug that looks exactly like a tenant-isolation bug. Committing
        here also means the transaction boundary under test is the one
        production uses: one request, one transaction, committed at the edge.
        """
        async with Session() as db:
            try:
                yield db
                await db.commit()
            except Exception:
                await db.rollback()
                raise

    app = FastAPI()
    app.include_router(api_router)
    app.dependency_overrides[get_session] = _session_override
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://agents")


async def _purge(Session, user_ids=()) -> None:
    """Delete every row this module creates, before seeding as well as after."""
    from sqlalchemy import delete

    from navis.identity.enterprise_models import ApiKey, AuditLog, ServiceAccount
    from navis.identity.models import Membership, Organization, User
    from navis.organization.models import (
        OrganizationEntitlement,
        OrganizationFeatureFlag,
        OrganizationLifecycle,
        OrganizationMetrics,
        OrganizationOnboarding,
        OrganizationProfile,
    )

    async with Session() as s:
        # Children first: api_keys FKs service_accounts, everything FKs organizations.
        for model in (ApiKey, ServiceAccount, Membership, AuditLog,
                      OrganizationEntitlement, OrganizationFeatureFlag,
                      OrganizationLifecycle, OrganizationMetrics,
                      OrganizationOnboarding, OrganizationProfile):
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
async def tenants(Session):
    """Two organizations, each with a FOUNDER (the operator) and an ADMIN.

    The admin exists for the removed-member characterisation: an organization
    cannot lose its last founder, so the member who is removed has to be someone
    else — and `admin` is the role that genuinely holds
    `manage_service_accounts`, which is what makes them able to create an agent
    in the first place.
    """
    from navis.identity.models import Membership, Organization, Role, User
    from navis.organization.contract import Plan
    from navis.organization.service import OrganizationService
    from navis.platform.security import create_access_token

    await _purge(Session)
    made: dict[uuid.UUID, dict] = {}

    async with Session() as s:
        for org_id, tag in ((TENANT_A, "A"), (TENANT_B, "B")):
            s.add(Organization(
                id=org_id, name=f"Tenant {tag}",
                slug=f"agents-{tag.lower()}-{uuid.uuid4().hex[:6]}",
            ))
            founder = User(
                email=f"agent-founder-{tag.lower()}-{uuid.uuid4().hex[:6]}@example.com",
                full_name=f"Founder {tag}", is_active=True,
            )
            admin = User(
                email=f"agent-admin-{tag.lower()}-{uuid.uuid4().hex[:6]}@example.com",
                full_name=f"Admin {tag}", is_active=True,
            )
            s.add_all([founder, admin])
            await s.flush()
            s.add(Membership(user_id=founder.id, organization_id=org_id, role=Role.FOUNDER,
                             developer_role='owner'))
            s.add(Membership(user_id=admin.id, organization_id=org_id, role=Role.ADMIN,
                             developer_role='admin'))
            await OrganizationService(s).bootstrap(org_id, plan=Plan.ENTERPRISE)
            await s.flush()

            made[org_id] = {
                "founder_id": founder.id,
                "admin_id": admin.id,
                "founder_token": create_access_token(str(founder.id), org_id=str(org_id)),
                "admin_token": create_access_token(str(admin.id), org_id=str(org_id)),
            }
        await s.commit()

    yield Session, made

    await _purge(
        Session,
        user_ids=[v["founder_id"] for v in made.values()]
        + [v["admin_id"] for v in made.values()],
    )


@pytest.fixture(autouse=True)
def _generous_write_budget(monkeypatch):
    """Lift the agent-write rate limit for the module, and say why out loud.

    The limit is REAL and it fires: writing this module without this fixture
    produced `429 rate limit exceeded for this organization` at the 21st write,
    against a live Redis, which is the limiter working exactly as intended. What
    a lifecycle test measures is authorization and isolation, not the capacity
    guard, so the budget is raised here and the guard itself is proved by
    `test_the_agent_write_rate_limit_is_enforced_per_organization` below, which
    lowers it deliberately.

    `rate_limit` reads the setting at REQUEST time rather than capturing it at
    import, so this takes effect without re-importing the router.
    """
    from navis.platform.config import settings

    monkeypatch.setattr(settings, "rate_limit_agent_write_per_window", 500, raising=False)


def _human(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _machine(secret: str) -> dict[str, str]:
    return {"X-API-Key": secret}


async def _create_agent(c, token: str, **overrides) -> dict:
    """Create one agent and return the whole `CreatedAgentOut` body."""
    body = {
        "name": "Fixture agent",
        "description": "created by the test fixture",
        "environment": "development",
        "runtime": "claude-code",
        "permissions": AGENT_SCOPES,
        **overrides,
    }
    r = await c.post("/developer/agents", headers=_human(token), json=body)
    assert r.status_code == 201, r.text
    return r.json()


# ═══════════════════════════════════════════════════════════════════════════
# 1 + 2. THE CONTROL — an operator registers an agent and gets one secret
# ═══════════════════════════════════════════════════════════════════════════

async def test_an_operator_registers_an_agent_in_their_own_organization(tenants):
    """The control. Without it every isolation assertion below would pass just as
    well against a surface that created nothing for anybody.

    Runs for BOTH tenants so the fixture is proved symmetric.
    """
    Session, t = tenants

    async with _client(Session) as c:
        for org_id, tag in ((TENANT_A, "A"), (TENANT_B, "B")):
            token = t[org_id]["founder_token"]
            created = await _create_agent(c, token, name=f"{tag}-only agent")

            agent = created["agent"]
            # The organization is the CALLER's, resolved from the membership row.
            assert agent["organization_id"] == str(org_id)
            assert agent["name"] == f"{tag}-only agent"
            assert agent["environment"] == "development"
            assert agent["runtime"] == "claude-code"
            assert agent["status"] == "active"
            assert agent["permissions"] == sorted(AGENT_SCOPES)
            assert agent["credential_count"] == 1
            assert agent["last_used_at"] is None

            # The secret exists here and only here.
            credential = created["credential"]
            assert credential["secret"].startswith("navis_")
            assert credential["prefix"] in credential["secret"]
            assert credential["agent_id"] == agent["id"]
            assert credential["revoked"] is False

            listed = await c.get("/developer/agents", headers=_human(token))
            assert listed.status_code == 200, listed.text
            page = listed.json()
            assert page["total"] >= 1
            assert agent["id"] in {a["id"] for a in page["agents"]}
            # Paged from the first response, not once it hurts.
            assert page["limit"] == 50 and page["offset"] == 0


async def test_the_secret_is_returned_once_and_can_never_be_read_back(tenants):
    """`api_keys` holds a SHA-256, so there is nothing left to leak — proved by
    walking every route that could carry one and asserting it does not.

    The check is on the whole response TEXT rather than a field, because the
    failure being guarded against is a secret appearing somewhere nobody thought
    to look: an error detail, an echoed request, a nested serializer.
    """
    Session, t = tenants
    token = t[TENANT_A]["founder_token"]

    async with _client(Session) as c:
        created = await _create_agent(c, token, name="Once-only agent")
        agent_id = created["agent"]["id"]
        secret = created["credential"]["secret"]

        for path in (
            "/developer/agents",
            f"/developer/agents/{agent_id}",
            f"/developer/agents/{agent_id}/credentials",
            "/developer/me",
        ):
            r = await c.get(path, headers=_human(token))
            assert r.status_code == 200, f"{path} -> {r.status_code} {r.text[:200]}"
            assert secret not in r.text, f"{path} returned the plaintext secret"

        # The credential IS listed — its metadata, and no secret field at all.
        creds = await c.get(f"/developer/agents/{agent_id}/credentials", headers=_human(token))
        rows = creds.json()["credentials"]
        assert len(rows) == 1
        assert "secret" not in rows[0]
        assert rows[0]["prefix"] and rows[0]["prefix"] in secret


# ═══════════════════════════════════════════════════════════════════════════
# 3 + 4. THE CLIENT CANNOT NAME A TENANT, AND CANNOT REACH ANOTHER'S AGENT
# ═══════════════════════════════════════════════════════════════════════════

async def test_no_secret_reaches_a_log_record(tenants, caplog):
    """The credential is minted, returned and never written to a log line.

    Checked over EVERY captured record's rendered message and its `args`,
    because a `%s` interpolation puts the value in `args` rather than in
    `getMessage()` until something formats it — and a log shipper formats it.

    The routes here deliberately log the credential's PUBLIC PREFIX, which is
    what lets an operator correlate "agent created" with the row in the console
    without the secret ever leaving the response. The last assertion is that the
    prefix really did make it, so this test cannot pass by logging nothing.
    """
    import logging

    Session, t = tenants
    token = t[TENANT_A]["founder_token"]

    async with _client(Session) as c:
        with caplog.at_level(logging.DEBUG):
            created = await _create_agent(c, token, name="Logged agent")
            agent_id = created["agent"]["id"]
            extra = await c.post(
                f"/developer/agents/{agent_id}/credentials",
                headers=_human(token), json={"name": "CI"},
            )
            assert extra.status_code == 201, extra.text
            await c.delete(f"/developer/agents/{agent_id}", headers=_human(token))

    secrets = [created["credential"]["secret"], extra.json()["secret"]]
    prefix = created["credential"]["prefix"]

    leaked: list[str] = []
    saw_prefix = False
    for record in caplog.records:
        rendered = f"{record.getMessage()} {record.args!r}"
        for secret in secrets:
            if secret in rendered:
                leaked.append(f"{record.name}: {rendered[:200]}")
            # The random half alone is still the whole secret's entropy.
            if secret.split("_", 2)[2] in rendered:
                leaked.append(f"{record.name} (secret segment): {rendered[:200]}")
        if prefix in rendered:
            saw_prefix = True

    assert not leaked, "a credential secret reached a log record:\n" + "\n".join(leaked)
    assert saw_prefix, "no agent log line was captured — this test proved nothing"


async def test_a_client_supplied_organization_never_moves_the_write(tenants):
    """`organization_id` in the body is not honoured, ignored-with-a-warning, or
    validated — the model has no such field, so it is dropped and the agent is
    created in the caller's own tenant.

    Asserted at the wire rather than by reading the model, because the question
    is what the SERVER does with the request an attacker actually sends.
    """
    Session, t = tenants

    async with _client(Session) as c:
        r = await c.post(
            "/developer/agents",
            headers=_human(t[TENANT_A]["founder_token"]),
            json={
                "name": "Tenant-hopping attempt",
                "permissions": AGENT_SCOPES,
                # Every shape a client might try.
                "organization_id": str(TENANT_B),
                "org_id": str(TENANT_B),
                "owner_user_id": str(t[TENANT_B]["founder_id"]),
                "created_by": str(t[TENANT_B]["founder_id"]),
            },
        )
        assert r.status_code == 201, r.text
        assert r.json()["agent"]["organization_id"] == str(TENANT_A)

        # And it is not visible from B.
        listed_b = await c.get("/developer/agents", headers=_human(t[TENANT_B]["founder_token"]))
        assert "Tenant-hopping attempt" not in listed_b.text


async def test_another_tenants_agent_id_is_a_404_on_every_route(tenants):
    """The broadest proof and the one closest to a real attack.

    B's agent id is a REAL id — it resolves for B on the line above — so a 404
    here is the tenant predicate doing its job rather than a malformed request
    being rejected. 404 and not 403 on purpose: a 403 would confirm the id names
    something real somewhere, which is a cross-tenant existence oracle.
    """
    Session, t = tenants
    a_token = t[TENANT_A]["founder_token"]
    b_token = t[TENANT_B]["founder_token"]

    async with _client(Session) as c:
        b_created = await _create_agent(c, b_token, name="B-only private agent")
        b_agent = b_created["agent"]["id"]
        b_credential = b_created["credential"]["id"]

        # The control: B can reach its own agent.
        mine = await c.get(f"/developer/agents/{b_agent}", headers=_human(b_token))
        assert mine.status_code == 200, mine.text

        attempts = [
            ("GET", f"/developer/agents/{b_agent}", None),
            ("PATCH", f"/developer/agents/{b_agent}", {"name": "seized"}),
            ("DELETE", f"/developer/agents/{b_agent}", None),
            ("GET", f"/developer/agents/{b_agent}/credentials", None),
            ("POST", f"/developer/agents/{b_agent}/credentials", {"name": "stolen key"}),
            ("DELETE", f"/developer/agents/{b_agent}/credentials/{b_credential}", None),
        ]
        wrong: list[str] = []
        for method, path, body in attempts:
            r = await c.request(method, path, headers=_human(a_token), json=body)
            if r.status_code != 404:
                wrong.append(f"{method} {path} -> {r.status_code} {r.text[:160]}")
        assert not wrong, "A reached B's agent:\n" + "\n".join(wrong)

        # B's agent is intact: not renamed, not revoked, credential not minted.
        after = await c.get(f"/developer/agents/{b_agent}", headers=_human(b_token))
        assert after.status_code == 200
        assert after.json()["name"] == "B-only private agent"
        assert after.json()["status"] == "active"
        assert after.json()["credential_count"] == 1


async def test_a_credential_id_from_another_agent_cannot_be_revoked(tenants):
    """Both ids are verified, and the second one under the first.

    A's own agent id plus B's credential id is the interesting case: the agent
    lookup succeeds, so the only thing standing between the caller and another
    agent's credential is the second predicate.
    """
    Session, t = tenants
    a_token = t[TENANT_A]["founder_token"]
    b_token = t[TENANT_B]["founder_token"]

    async with _client(Session) as c:
        a_agent = (await _create_agent(c, a_token, name="A pairing agent"))["agent"]["id"]
        b_created = await _create_agent(c, b_token, name="B pairing agent")
        b_credential = b_created["credential"]["id"]
        b_secret = b_created["credential"]["secret"]

        r = await c.delete(
            f"/developer/agents/{a_agent}/credentials/{b_credential}", headers=_human(a_token)
        )
        assert r.status_code == 404, r.text

        # B's credential still authenticates.
        me = await c.get("/developer/me", headers=_machine(b_secret))
        assert me.status_code == 200, me.text
        assert me.json()["organization"]["id"] == str(TENANT_B)


# ═══════════════════════════════════════════════════════════════════════════
# 5. THE AGENT'S OWN CREDENTIAL — who it is, and exactly what it reaches
# ═══════════════════════════════════════════════════════════════════════════

async def test_an_agent_credential_authenticates_as_that_agent_and_no_further(tenants):
    """The credential -> agent -> organization chain, end to end at the wire.

    `/developer/me` reports the tenant from the `api_keys` row and the agent from
    `api_keys.service_account_id`, and the permission list is exactly what the
    agent was granted. The 403 below is the other half: a route naming a
    permission this agent does not hold is refused even though the ORGANIZATION
    is entitled to the capability behind it.
    """
    Session, t = tenants
    token = t[TENANT_A]["founder_token"]

    async with _client(Session) as c:
        created = await _create_agent(c, token, name="Scoped agent")
        agent_id, secret = created["agent"]["id"], created["credential"]["secret"]
        credential_id = created["credential"]["id"]

        me = await c.get("/developer/me", headers=_machine(secret))
        assert me.status_code == 200, me.text
        body = me.json()
        assert body["organization"]["id"] == str(TENANT_A)
        assert body["principal"]["kind"] == "service_account"
        assert body["principal"]["service_account_id"] == agent_id
        assert body["principal"]["credential_id"] == credential_id
        assert body["principal"]["permissions"] == sorted(AGENT_SCOPES)

        # Granted: view_state and view_memory.
        for path in ("/developer/state", "/developer/memory"):
            r = await c.get(path, headers=_machine(secret))
            assert r.status_code == 200, f"{path} -> {r.status_code} {r.text[:160]}"

        # Not granted: view_learning. The org is ENTERPRISE, so this is a
        # CREDENTIAL refusal (403), never an entitlement one (402).
        denied = await c.get("/developer/learning", headers=_machine(secret))
        assert denied.status_code == 403, denied.text
        assert "view_learning" in denied.text


async def test_an_agent_credential_cannot_change_the_agent_fleet(tenants):
    """Writes are human-only, and this is why.

    An agent that can register agents can mint identities its OWN revocation does
    not reach — so containing a leaked agent credential would stop being a single
    DELETE. The agent below holds `manage_service_accounts`, which is exactly the
    permission the routes gate on, and is still refused: the refusal is about
    what KIND of principal it is, not what it may do.
    """
    Session, t = tenants
    token = t[TENANT_A]["founder_token"]

    async with _client(Session) as c:
        privileged = await _create_agent(
            c, token, name="Would-be recruiter",
            permissions=["view_state", "manage_service_accounts"],
        )
        secret = privileged["credential"]["secret"]
        agent_id = privileged["agent"]["id"]

        # It authenticates and it may READ the fleet — nothing there is a secret.
        listed = await c.get("/developer/agents", headers=_machine(secret))
        assert listed.status_code == 200, listed.text

        refusals = [
            ("POST", "/developer/agents", {"name": "spawned", "permissions": []}),
            ("PATCH", f"/developer/agents/{agent_id}", {"name": "self-promoted"}),
            ("DELETE", f"/developer/agents/{agent_id}", None),
            ("POST", f"/developer/agents/{agent_id}/credentials", {"name": "extra"}),
        ]
        wrong: list[str] = []
        for method, path, body in refusals:
            r = await c.request(method, path, headers=_machine(secret), json=body)
            if r.status_code != 403:
                wrong.append(f"{method} {path} -> {r.status_code} {r.text[:160]}")
        assert not wrong, "a machine principal changed the fleet:\n" + "\n".join(wrong)


# ═══════════════════════════════════════════════════════════════════════════
# 6 + 7. REVOCATION
# ═══════════════════════════════════════════════════════════════════════════

async def test_revoking_an_agent_kills_every_credential_it_holds(tenants):
    """Revocation is one call and it is immediate — no TTL, no cache window.

    Both credentials die, and they die twice over: each `api_keys` row is
    stamped, AND `_principal_from_api_key` re-reads `service_accounts.revoked_at`
    and answers 401 even if a key were somehow missed. The second GET below
    proves the row survives revocation, which is what keeps the audit trail's
    target from pointing at nothing.
    """
    Session, t = tenants
    token = t[TENANT_A]["founder_token"]

    async with _client(Session) as c:
        created = await _create_agent(c, token, name="Doomed agent")
        agent_id = created["agent"]["id"]
        first = created["credential"]["secret"]

        second_response = await c.post(
            f"/developer/agents/{agent_id}/credentials",
            headers=_human(token), json={"name": "CI"},
        )
        assert second_response.status_code == 201, second_response.text
        second = second_response.json()["secret"]

        for secret in (first, second):
            assert (await c.get("/developer/me", headers=_machine(secret))).status_code == 200

        revoked = await c.delete(f"/developer/agents/{agent_id}", headers=_human(token))
        assert revoked.status_code == 200, revoked.text
        assert revoked.json() == {
            "id": agent_id, "status": "revoked", "credentials_revoked": 2
        }

        for secret in (first, second):
            r = await c.get("/developer/me", headers=_machine(secret))
            assert r.status_code == 401, f"a revoked agent's credential still works: {r.text[:160]}"

        # The record remains, and reports the terminal state.
        after = await c.get(f"/developer/agents/{agent_id}", headers=_human(token))
        assert after.status_code == 200
        assert after.json()["status"] == "revoked"
        assert after.json()["credential_count"] == 0

        # Idempotent: a retry after a dropped response is the same outcome.
        again = await c.delete(f"/developer/agents/{agent_id}", headers=_human(token))
        assert again.status_code == 200
        assert again.json()["credentials_revoked"] == 0

        # And a revoked agent is not editable — revocation is terminal.
        edit = await c.patch(
            f"/developer/agents/{agent_id}", headers=_human(token), json={"name": "resurrected"}
        )
        assert edit.status_code == 404, edit.text


async def test_revoking_one_credential_leaves_the_agent_and_its_others_alive(tenants):
    """The narrow operation: one leaked key, retired, without an outage.

    The revoked row stays in the listing marked `revoked` — a credential list
    that hid it could not answer "was the leaked key the one we retired?", which
    is the question the list is opened for.
    """
    Session, t = tenants
    token = t[TENANT_A]["founder_token"]

    async with _client(Session) as c:
        created = await _create_agent(c, token, name="Two-key agent")
        agent_id = created["agent"]["id"]
        keep = created["credential"]["secret"]

        extra = await c.post(
            f"/developer/agents/{agent_id}/credentials",
            headers=_human(token), json={"name": "laptop"},
        )
        assert extra.status_code == 201, extra.text
        drop_id, drop_secret = extra.json()["id"], extra.json()["secret"]

        # An additional credential inherits the AGENT's scopes, never the request's.
        assert extra.json()["scopes"] == sorted(AGENT_SCOPES)

        gone = await c.delete(
            f"/developer/agents/{agent_id}/credentials/{drop_id}", headers=_human(token)
        )
        assert gone.status_code == 200, gone.text
        assert gone.json() == {"id": drop_id, "agent_id": agent_id, "revoked": True}

        assert (await c.get("/developer/me", headers=_machine(drop_secret))).status_code == 401
        assert (await c.get("/developer/me", headers=_machine(keep))).status_code == 200

        listing = await c.get(f"/developer/agents/{agent_id}/credentials", headers=_human(token))
        rows = {row["id"]: row for row in listing.json()["credentials"]}
        assert rows[drop_id]["revoked"] is True
        assert len(rows) == 2, "a revoked credential vanished from the listing"


# ═══════════════════════════════════════════════════════════════════════════
# 9 + 10 + 11. AUTHORIZATION
# ═══════════════════════════════════════════════════════════════════════════

async def test_an_operator_cannot_grant_an_agent_more_than_they_hold(tenants):
    """Agent creation is not a privilege-escalation primitive.

    `admin` deliberately does NOT hold `execute_actions` (see
    `DEFAULT_ROLE_PERMISSIONS`), so an admin asking for it is refused 403 by the
    same `ensure_grantable_scopes` that guards `/identity/api-keys`. An unknown
    permission is a 400 rather than a 403 — a typo is a bad request, not an
    authorization decision, and reporting it as one sends someone to ask for a
    grant that does not exist.
    """
    Session, t = tenants
    admin = t[TENANT_A]["admin_token"]

    async with _client(Session) as c:
        escalation = await c.post(
            "/developer/agents", headers=_human(admin),
            json={"name": "Overreaching agent", "permissions": ["execute_actions"]},
        )
        assert escalation.status_code == 403, escalation.text
        assert "execute_actions" in escalation.text

        typo = await c.post(
            "/developer/agents", headers=_human(admin),
            json={"name": "Typo agent", "permissions": ["view_stat"]},
        )
        assert typo.status_code == 400, typo.text

        # The control: a permission the admin DOES hold is accepted.
        ok = await c.post(
            "/developer/agents", headers=_human(admin),
            json={"name": "Within-bounds agent", "permissions": ["view_state"]},
        )
        assert ok.status_code == 201, ok.text


async def test_permission_and_capability_stay_separate_gates(tenants):
    """403 is about the CREDENTIAL; 402 is about the TENANT. They must not merge.

    The organization is switched off `Capability.STATE` for the duration, which
    is what `LifecycleService` does to a suspended tenant and what a kill switch
    does to everyone. Agent management then answers 402 — a suspended
    organization cannot mint new machine credentials — while a caller lacking the
    permission still gets 403, because authentication and authorization are
    settled before entitlement is consulted.
    """
    from navis.organization.contract import Capability
    from navis.organization.service import OrganizationService

    Session, t = tenants
    founder = t[TENANT_A]["founder_token"]

    async with _client(Session) as c:
        async with Session() as s:
            await OrganizationService(s).entitlements.set(TENANT_A, Capability.STATE, False)
            await s.commit()
        try:
            denied = await c.get("/developer/agents", headers=_human(founder))
            assert denied.status_code == 402, denied.text
            assert "state" in denied.text

            created = await c.post(
                "/developer/agents", headers=_human(founder),
                json={"name": "During suspension", "permissions": []},
            )
            assert created.status_code == 402, created.text
        finally:
            async with Session() as s:
                await OrganizationService(s).entitlements.set(TENANT_A, Capability.STATE, True)
                await s.commit()

        restored = await c.get("/developer/agents", headers=_human(founder))
        assert restored.status_code == 200, restored.text
        assert "During suspension" not in restored.text, "a 402 still created the row"


async def test_editing_permissions_changes_what_the_live_credential_reaches(tenants):
    """The gap this closes, stated as a failure: authorization reads
    `api_keys.scopes`, so editing only the account would change what the console
    displays and NOTHING about what the credential in the agent's environment can
    do. `ServiceAccountService.update` cascades, in the same transaction.

    Proved from the credential's own side — the agent asks `/developer/me` and
    the permission list has actually changed — rather than by re-reading the
    agent record, which is the half that was never in doubt.
    """
    Session, t = tenants
    token = t[TENANT_A]["founder_token"]

    async with _client(Session) as c:
        created = await _create_agent(c, token, name="Re-scoped agent")
        agent_id, secret = created["agent"]["id"], created["credential"]["secret"]

        assert (await c.get("/developer/memory", headers=_machine(secret))).status_code == 200

        patched = await c.patch(
            f"/developer/agents/{agent_id}", headers=_human(token),
            json={"permissions": ["view_state"], "description": "narrowed"},
        )
        assert patched.status_code == 200, patched.text
        assert patched.json()["permissions"] == ["view_state"]
        assert patched.json()["description"] == "narrowed"

        me = await c.get("/developer/me", headers=_machine(secret))
        assert me.json()["principal"]["permissions"] == ["view_state"]

        narrowed = await c.get("/developer/memory", headers=_machine(secret))
        assert narrowed.status_code == 403, (
            "the live credential kept a permission the agent no longer holds: "
            f"{narrowed.status_code}"
        )
        assert (await c.get("/developer/state", headers=_machine(secret))).status_code == 200


async def test_an_unknown_environment_or_runtime_is_refused(tenants):
    """Closed vocabularies, enforced at the edge as well as in the database.

    A 422 rather than a stored free-text value: an unbounded tier or runtime
    cannot be grouped, filtered or counted, which is the only reason to record
    either of them.
    """
    Session, t = tenants
    token = t[TENANT_A]["founder_token"]

    async with _client(Session) as c:
        for field, value in (("environment", "prod"), ("runtime", "some-new-tool")):
            r = await c.post(
                "/developer/agents", headers=_human(token),
                json={"name": "Bad vocabulary", "permissions": [], field: value},
            )
            assert r.status_code == 422, f"{field}={value} -> {r.status_code} {r.text[:160]}"

        blank = await c.post(
            "/developer/agents", headers=_human(token),
            json={"name": "   ", "permissions": []},
        )
        assert blank.status_code == 422, blank.text


async def test_the_agent_write_rate_limit_is_enforced_per_organization(tenants, monkeypatch):
    """The capacity guard, proved rather than assumed — and proved to be TENANT-keyed.

    The bucket key is `Principal.org_id`, which `get_principal` resolved from a
    membership row, so there is no request field that could move a caller into
    another tenant's bucket to spend it or out of their own to escape it. Org B
    keeps its full budget while Org A is throttled, which is the isolation
    property the limiter exists to defend.

    `Retry-After` is asserted because a 429 without one is a client that spins.
    """
    from navis.platform.cache import redis_client
    from navis.platform.config import settings

    Session, t = tenants
    monkeypatch.setattr(settings, "rate_limit_agent_write_per_window", 2, raising=False)

    # Start from an empty bucket for BOTH tenants. The window is shared with the
    # tests above, which have already spent it; without this, Org B would be
    # throttled for a reason that has nothing to do with Org A's burst and the
    # isolation assertion would pass for the wrong reason.
    for org_id in (TENANT_A, TENANT_B):
        keys = [k async for k in redis_client.scan_iter(f"ratelimit:agent_write:{org_id}:*")]
        if keys:
            await redis_client.delete(*keys)

    async with _client(Session) as c:
        statuses = []
        for n in range(5):
            r = await c.post(
                "/developer/agents",
                headers=_human(t[TENANT_A]["founder_token"]),
                json={"name": f"burst-{uuid.uuid4().hex[:6]}-{n}", "permissions": []},
            )
            statuses.append(r.status_code)
            if r.status_code == 429:
                assert r.headers.get("Retry-After"), "429 without Retry-After"

        assert 429 in statuses, f"the write limiter never fired: {statuses}"

        # Org B is untouched — a separate bucket, keyed on ITS organization.
        other = await c.post(
            "/developer/agents",
            headers=_human(t[TENANT_B]["founder_token"]),
            json={"name": f"unthrottled-{uuid.uuid4().hex[:6]}", "permissions": []},
        )
        assert other.status_code == 201, (
            f"one tenant's burst spent another tenant's budget: {other.status_code}"
        )


async def test_the_active_agent_ceiling_is_enforced_and_freed_by_revocation(tenants, monkeypatch):
    """A capacity guard on rows, not an entitlement.

    409 rather than 402: the organization's PLAN is not the reason it was
    refused, and reporting a capacity limit as a billing decision would send an
    operator to upgrade something that would not help. Revoking an agent frees
    its slot — a ceiling that counted tombstones would leave an organization that
    rotates its fleet permanently unable to create another.
    """
    from navis.platform.config import settings

    Session, t = tenants
    token = t[TENANT_B]["founder_token"]

    async with _client(Session) as c:
        existing = (await c.get("/developer/agents", headers=_human(token))).json()
        active = [a for a in existing["agents"] if a["status"] == "active"]
        monkeypatch.setattr(
            settings, "max_agents_per_organization", len(active) + 1, raising=False
        )

        first = await c.post(
            "/developer/agents", headers=_human(token),
            json={"name": f"ceiling-{uuid.uuid4().hex[:6]}", "permissions": []},
        )
        assert first.status_code == 201, first.text

        refused = await c.post(
            "/developer/agents", headers=_human(token),
            json={"name": f"over-{uuid.uuid4().hex[:6]}", "permissions": []},
        )
        assert refused.status_code == 409, refused.text
        assert "active agents" in refused.text

        # Revoke one, and the slot comes back.
        assert (
            await c.delete(
                f"/developer/agents/{first.json()['agent']['id']}", headers=_human(token)
            )
        ).status_code == 200
        again = await c.post(
            "/developer/agents", headers=_human(token),
            json={"name": f"reclaimed-{uuid.uuid4().hex[:6]}", "permissions": []},
        )
        assert again.status_code == 201, again.text


# ═══════════════════════════════════════════════════════════════════════════
# 12. CONCURRENCY
# ═══════════════════════════════════════════════════════════════════════════

async def test_interleaved_agent_traffic_from_two_tenants_never_crosses(tenants):
    """Both tenants writing AND reading at the same time, in one process.

    THE PROPERTY. The tenant must be derived per request and held nowhere else.
    A write makes this sharper than the read-side proof: a create that picked up
    a process-local organization would put one tenant's agent in another's fleet
    permanently, and a sequential test would never surface it.

    24 requests are issued with `asyncio.gather` so they interleave inside one
    event loop, alternating tenant AND operation. Every response must describe
    its own caller's tenant, and afterwards NEITHER fleet may contain a row
    created under the other's credential.
    """
    Session, t = tenants
    marker = uuid.uuid4().hex[:8]

    async with _client(Session) as c:
        async def create(org_id, n):
            r = await c.post(
                "/developer/agents",
                headers=_human(t[org_id]["founder_token"]),
                json={
                    "name": f"conc-{marker}-{'A' if org_id == TENANT_A else 'B'}-{n}",
                    "permissions": AGENT_SCOPES,
                },
            )
            return org_id, "create", r

        async def read(org_id, _n):
            r = await c.get("/developer/agents", headers=_human(t[org_id]["founder_token"]))
            return org_id, "list", r

        jobs = [
            op(org_id, n)
            for n in range(6)
            for org_id in (TENANT_A, TENANT_B)   # interleaved, not batched per tenant
            for op in (create, read)
        ]
        results = await asyncio.gather(*jobs)
        assert len(results) == 24

        crossed: list[str] = []
        for org_id, kind, r in results:
            other = TENANT_B if org_id == TENANT_A else TENANT_A
            if r.status_code not in (200, 201):
                crossed.append(f"{kind} as {org_id} -> {r.status_code} {r.text[:160]}")
                continue
            if str(other) in r.text:
                crossed.append(f"{kind} as {org_id} named {other}: {r.text[:200]}")
            if kind == "create" and r.json()["agent"]["organization_id"] != str(org_id):
                crossed.append(f"create as {org_id} landed in another tenant")

        # And the durable state agrees with every individual response.
        for org_id, tag, other_tag in (
            (TENANT_A, "A", "B"), (TENANT_B, "B", "A")
        ):
            page = await c.get(
                "/developer/agents?limit=200", headers=_human(t[org_id]["founder_token"])
            )
            names = {a["name"] for a in page.json()["agents"]}
            mine = {n for n in names if n.startswith(f"conc-{marker}-{tag}-")}
            theirs = {n for n in names if n.startswith(f"conc-{marker}-{other_tag}-")}
            if len(mine) != 6:
                crossed.append(f"{org_id} kept {len(mine)} of its 6 concurrent agents")
            if theirs:
                crossed.append(f"{org_id}'s fleet contains {sorted(theirs)}")

        assert not crossed, (
            "tenant state crossed a request boundary under concurrency:\n" + "\n".join(crossed)
        )


# ═══════════════════════════════════════════════════════════════════════════
# 13. THE OPEN FINDING — a removed member's agent credential
# ═══════════════════════════════════════════════════════════════════════════

async def test_a_removed_members_agent_credential_survives_their_removal(tenants):
    """CHARACTERISATION, NOT ENDORSEMENT. This records what the system does today
    so that a change to it is a deliberate decision with a failing test attached.

    WHAT HAPPENS. Removing a member bulk-revokes their SESSIONS
    (`MemberService.remove` -> `TokenRevocationService.revoke_all_for_user`, and
    the JWT below is dead immediately). It does NOT revoke `api_keys` rows whose
    `created_by` is that user, and `_principal_from_api_key` never consults
    membership — an API key's tenant is `api_keys.organization_id`, full stop.

    WHY THAT IS DEFENSIBLE FOR AN AGENT. The agent is ORGANIZATION-owned. It has
    its own row, its own grants and its own revocation clock, and `created_by` is
    attribution rather than authority — which is precisely what makes an agent
    something other than "a user plus an API key". A CI agent must not stop
    working because the engineer who registered it changed teams.

    WHY IT IS STILL A REAL RISK. The person who created the credential SAW the
    plaintext once, and after their removal they still hold a working credential
    for an organization they are no longer a member of. Nothing in the product
    surfaces that, and offboarding has no step that would.

    The residual risk is bounded by what the agent holds — `AGENT_SCOPES` here,
    never the departing member's own permissions — and by the operator's ability
    to revoke the agent, which the last assertion exercises. The recommended fix
    is in the task report; it is deliberately not applied here.
    """
    from navis.identity.authz import principal_cache_clear
    from navis.identity.members import MemberService
    from navis.identity.permissions import FOUNDER_PERMISSIONS

    Session, t = tenants
    admin_token = t[TENANT_A]["admin_token"]
    admin_id = t[TENANT_A]["admin_id"]
    founder_id = t[TENANT_A]["founder_id"]

    async with _client(Session) as c:
        created = await _create_agent(c, admin_token, name="Departing member's agent")
        agent_id, secret = created["agent"]["id"], created["credential"]["secret"]

        assert (await c.get("/developer/me", headers=_machine(secret))).status_code == 200

        async with Session() as s:
            await MemberService(s).remove(
                org_id=TENANT_A, user_id=admin_id, actor_id=founder_id,
                actor_permissions=FOUNDER_PERMISSIONS,
            )
            await s.commit()
        # The in-process principal cache is evicted by `revoke_all_for_user`;
        # cleared here as well so this asserts the DATABASE state rather than a
        # cache that happened to be cold.
        principal_cache_clear()

        # The SESSION is dead — this half already works.
        session_after = await c.get("/developer/agents", headers=_human(admin_token))
        assert session_after.status_code == 401, (
            f"a removed member's JWT still authenticates: {session_after.status_code}"
        )

        # The CREDENTIAL is not. This is the finding.
        credential_after = await c.get("/developer/me", headers=_machine(secret))
        assert credential_after.status_code == 200, (
            "the removed-member credential behaviour has CHANGED — if that was "
            "deliberate, update this test and the audit report together"
        )
        assert credential_after.json()["organization"]["id"] == str(TENANT_A)

        # It is reachable and revocable by an operator who remains, which is the
        # mitigation the product has today.
        founder = t[TENANT_A]["founder_token"]
        revoked = await c.delete(f"/developer/agents/{agent_id}", headers=_human(founder))
        assert revoked.status_code == 200, revoked.text
        assert (await c.get("/developer/me", headers=_machine(secret))).status_code == 401
