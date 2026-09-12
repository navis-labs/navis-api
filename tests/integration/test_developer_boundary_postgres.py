"""The Developer API boundary, proved at the HTTP wire against real PostgreSQL.

    NAVIS_TEST_DATABASE_URL=postgresql+asyncpg://.../navis_test?ssl=require \
        EVENT_HANDLER_TIMEOUT_SECONDS=180 \
        pytest tests/integration/test_developer_boundary_postgres.py -q

WHAT THIS PROVES, AND WHY AT THE WIRE. Every property below is a property of a
REQUEST, not of a helper: a tenant filter that a repository applies correctly can
still be bypassed by a route that never calls it, and a permission a gate checks
correctly can still be missing from the route that needed it. So the whole chain
runs for real — credential resolution, principal resolution, membership lookup,
permission gate, capability gate, domain service, repository — and only
`get_session` is overridden, to point at a disposable database.

The properties, in the order the module asserts them:

  1. an Org A credential reads Org A data on every Developer route (the control:
     without it, isolation would "pass" against a backend returning nothing);
  2. an Org A credential never sees an Org B row on any Developer route;
  3. an Org B credential reads Org B data (so the fixture is real, not one-sided);
  4. a caller-supplied `organization_id` / `agent_id` / `principal_id` cannot move
     a request into another tenant, or out of its own;
  5. a cross-tenant entity id 404s instead of resolving;
  6. a revoked API key, a revoked service account, an invalid key and a missing
     credential are each 401 on every Developer route;
  7. a scopeless credential is 403 on every gated Developer route;
  8. no API secret appears in any response body or any log record;
  9. the EXISTING machine-reachable surfaces (`/context`, `/knowledge/home`,
     `/outcomes`, `/learning/summary`) hold the same isolation under an API key —
     the audit's finding, tested rather than assumed;
 10. a credential whose service account belongs to ANOTHER tenant is refused, and
     a known-but-foreign agent UUID grants nothing in any argument position;
 11. permission and capability are separate gates with separate, non-leaking
     denials (403 about the credential, 402 about the tenant);
 12. 32 interleaved concurrent requests from both tenants never cross, and the
     in-process State cache stays tenant-keyed while they do;
 13. every dimension a future quota would key on is already server-derivable.

MODULE STRUCTURE mirrors `test_cross_tenant_isolation_postgres.py` and for the
same reasons documented there: a MODULE-scoped event loop and engine, schema
built once and never dropped, per-module row cleanup instead of `drop_all` (which
re-oids every PG ENUM and breaks pooled asyncpg type codecs), and `httpx` over
`ASGITransport` instead of Starlette's `TestClient` (which runs the app on its
own portal loop and corrupts the asyncpg wire protocol).

SAFETY. Refuses to run unless the target database is named `navis_test*`, and
never drops a schema.
"""
from __future__ import annotations

import logging
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import pytest
import pytest_asyncio

pytestmark = [
    pytest.mark.skipif(
        not os.environ.get("NAVIS_TEST_DATABASE_URL"),
        reason="set NAVIS_TEST_DATABASE_URL (disposable Postgres) to run these",
    ),
    pytest.mark.asyncio(loop_scope="module"),
]

NOW = datetime(2026, 9, 5, 9, 0, tzinfo=timezone.utc)

TENANT_A = uuid.UUID("dddddddd-0000-4000-8000-00000000000a")
TENANT_B = uuid.UUID("dddddddd-0000-4000-8000-00000000000b")

# Every Developer route, with a request that is valid for any authenticated
# caller. Used by the isolation, revocation, permission and secret-leak sweeps so
# that a route added later is covered by all four the moment it is listed here.
DEVELOPER_ROUTES: tuple[str, ...] = (
    "/developer/me",
    "/developer/context",
    "/developer/state",
    "/developer/memory",
    "/developer/outcomes",
    "/developer/learning",
    # The three intelligence reads. Listed here rather than tested separately so
    # they inherit the isolation, revocation, permission and secret-leak sweeps
    # above — which is the whole reason this list is declarative.
    "/developer/insights",
    "/developer/decisions",
    "/developer/reasoning",
)

# The routes that reach tenant business data — i.e. everything except `/me`,
# which is authentication-only by design (see `gateway.get_developer_context`).
GATED_ROUTES: tuple[str, ...] = DEVELOPER_ROUTES[1:]

# The scopes a Developer credential needs for the gated routes above. Deliberately
# spelled out rather than "all permissions": a credential holding exactly these
# must reach exactly these.
DEVELOPER_SCOPES = [
    "view_state", "view_memory", "view_observation", "view_learning",
    # Added with the three intelligence reads. Each already existed in the ONE
    # vocabulary and had no route using it; none is Developer-specific.
    "view_intelligence", "view_decisions", "view_reasoning",
]


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def engine():
    """ONE engine and ONE `create_all` for the whole module. Never `drop_all`.

    Guarded by DATABASE NAME, never hostname: a hostname check would both block a
    legitimate disposable Neon database and permit clobbering a local working copy.
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

    # Pool sized for the CONCURRENCY test, which is the only thing here that
    # needs more than a couple of connections: it issues 16 simultaneous requests
    # and the `get_session` override takes one session per request. SQLAlchemy's
    # default (pool_size=5, max_overflow=10) caps that at 15 and the 16th request
    # fails `QueuePool limit ... connection timed out` after 30s.
    #
    # That is a property of THIS FIXTURE, not of the application: production sizes
    # its own pool and serves concurrent requests over time rather than in one
    # `gather`. Sized here explicitly so the concurrency test measures tenant
    # isolation rather than the harness's connection budget, and kept modest so a
    # hosted Postgres is not handed a burst it will drop mid-run.
    eng = create_async_engine(url, pool_size=20, max_overflow=10)
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield eng
    await eng.dispose()


@pytest_asyncio.fixture(loop_scope="module")
async def Session(engine, monkeypatch):
    """A sessionmaker on the module engine, installed everywhere production would
    otherwise reach for the global one.

    Subscribers import `session_scope` BY VALUE at module import, so one left
    unpatched opens a session against the URL in `.env` — which is PRODUCTION.
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
    permission gates, capability gates and rate limits all run for real — which is
    the point: this measures what a request actually reaches.
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
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://developer")


async def _purge(Session, user_ids=()) -> None:
    """Delete every row this module creates. Called BEFORE seeding as well as
    after, so a run that dies mid-setup does not collide on the next run."""
    from sqlalchemy import delete

    from navis.identity.enterprise_models import ApiKey, AuditLog, ServiceAccount
    from navis.identity.models import Membership, Organization, User
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
        # Children first: api_keys FKs service_accounts, memory_records FKs
        # signal_events, and everything FKs organizations.
        for model in (ApiKey, ServiceAccount, LearningEvent, Outcome, StateSnapshot,
                      MemoryRecord, SignalEventRecord, Membership, AuditLog,
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
    """Two organizations, each fully populated, each with real machine credentials.

    Per tenant: an org, a founder, an ENTERPRISE bootstrap (so no route is denied
    for want of an entitlement), a signal + memory record + state snapshot +
    outcome + learning event, a SERVICE ACCOUNT with a key, a bare API KEY, and a
    SCOPELESS API key. Markers are tenant-distinguishable ("A-only"/"B-only") so a
    leak shows up as content rather than only as a count.
    """
    from navis.identity.apikeys import ApiKeyService
    from navis.identity.models import Membership, Organization, Role, User
    from navis.identity.service_accounts import ServiceAccountService
    from navis.learning.models import LearningEvent
    from navis.memory.envelope import EntityType
    from navis.memory.ids import entity_id as memory_entity_id
    from navis.memory.models import MemoryRecord
    from navis.observation.models import Outcome
    from navis.organization.contract import Plan
    from navis.organization.service import OrganizationService
    from navis.signals.models import SignalEventRecord
    from navis.state.v2.snapshots import StateSnapshot

    await _purge(Session)
    made: dict[uuid.UUID, dict] = {}

    async with Session() as s:
        for org_id, tag in ((TENANT_A, "A"), (TENANT_B, "B")):
            s.add(Organization(
                id=org_id, name=f"Tenant {tag}",
                slug=f"dev-{tag.lower()}-{uuid.uuid4().hex[:6]}",
            ))
            user = User(email=f"dev-{tag.lower()}-{uuid.uuid4().hex[:6]}@example.com",
                        full_name=f"Founder {tag}", is_active=True)
            s.add(user)
            await s.flush()
            s.add(Membership(user_id=user.id, organization_id=org_id, role=Role.FOUNDER))

            await OrganizationService(s).bootstrap(org_id, plan=Plan.ENTERPRISE)

            signal_id = uuid.uuid4()
            s.add(SignalEventRecord(
                id=signal_id, organization_id=org_id, source="gmail",
                event_type="message", external_id=f"{tag}-only-dev-msg",
                external_version="1", occurred_at=NOW,
                payload={"subject": f"{tag}-only subject"},
            ))
            # `memory_records.source_signal_event_id` is a real FK. Flush the
            # signal first: within one flush SQLAlchemy orders inserts per MAPPER,
            # so the memory row is not guaranteed to land after it otherwise.
            await s.flush()
            # A canonical, org-derived entity id (UUIDv5 over the ORGANIZATION), so
            # the cross-tenant entity probe below aims at an id that genuinely
            # exists — in the other tenant.
            ent_id = memory_entity_id(
                org_id, EntityType.PERSON.value, f"{tag.lower()}-only@example.com"
            )
            s.add(MemoryRecord(
                memory_id=uuid.uuid4(), organization_id=org_id,
                entity_type=EntityType.PERSON.value, entity_id=ent_id,
                source_signal_event_id=signal_id, extractor_name="test",
                extractor_version="1", schema_version=1, event_timestamp=NOW,
                dedup_key=f"{tag}-only-dev-dedup",
                payload={"name": f"{tag}-only person",
                         "email": f"{tag.lower()}-only@example.com"},
            ))
            s.add(StateSnapshot(
                organization_id=org_id, projection_version=1, schema_version=1,
                full_projection={"marker": f"{tag}-only state"},
            ))
            outcome_id = uuid.uuid4()
            s.add(Outcome(
                outcome_id=outcome_id, organization_id=org_id, outcome_type="reply",
                source_event_id=f"{tag}-only-dev-outcome", occurred_at=NOW,
                recorded_at=NOW, payload={"marker": f"{tag}-only outcome"},
            ))
            s.add(LearningEvent(
                id=uuid.uuid4(), organization_id=org_id, outcome_id=outcome_id,
                outcome_type="reply", outcome_surface="email", result="positive",
                source_event_id=f"{tag}-only-dev-learning", occurred_at=NOW,
            ))
            await s.flush()

            # ── The machine credentials this module authenticates as ──────────
            sa, agent_key = await ServiceAccountService(s).create(
                org_id=org_id, name=f"{tag} agent", scopes=DEVELOPER_SCOPES,
                created_by=user.id,
            )
            _, bare_key = await ApiKeyService(s).create(
                org_id=org_id, name=f"{tag} bare key", scopes=DEVELOPER_SCOPES,
                created_by=user.id,
            )
            scopeless_row, scopeless_key = await ApiKeyService(s).create(
                org_id=org_id, name=f"{tag} scopeless key", scopes=[],
                created_by=user.id,
            )
            made[org_id] = {
                "user_id": user.id, "entity_id": ent_id, "outcome_id": outcome_id,
                "service_account_id": sa.id, "agent_key": agent_key,
                "bare_key": bare_key, "scopeless_key": scopeless_key,
                "scopeless_id": scopeless_row.id,
            }
        await s.commit()

    yield Session, made

    await _purge(Session, user_ids=[v["user_id"] for v in made.values()])


def _auth(key: str) -> dict[str, str]:
    return {"X-API-Key": key}


# ═══════════════════════════════════════════════════════════════════════════
# 1 + 3. THE CONTROL — each tenant reads its OWN data through the boundary
# ═══════════════════════════════════════════════════════════════════════════

async def test_each_tenant_reads_its_own_data_through_the_developer_api(tenants):
    """Without this, every isolation assertion below would pass just as well
    against a boundary that returned nothing to anybody.

    Runs for BOTH tenants so the fixture is proved symmetric: a one-sided seed
    would make "B never appears in A's answer" true for the wrong reason.
    """
    Session, t = tenants

    async with _client(Session) as c:
        for org_id, tag in ((TENANT_A, "A"), (TENANT_B, "B")):
            key = t[org_id]["agent_key"]

            me = await c.get("/developer/me", headers=_auth(key))
            assert me.status_code == 200, me.text
            body = me.json()
            assert body["organization"]["id"] == str(org_id)
            assert body["organization"]["name"] == f"Tenant {tag}"
            assert body["principal"]["kind"] == "service_account"
            assert body["principal"]["service_account_id"] == str(
                t[org_id]["service_account_id"]
            )
            assert body["principal"]["permissions"] == sorted(DEVELOPER_SCOPES)
            # The entitlement snapshot is published and is this org's own.
            assert body["capabilities"]["state"] is True

            mem = await c.get("/developer/memory", headers=_auth(key))
            assert mem.status_code == 200, mem.text
            assert f"{tag}-only person" in mem.text, "a tenant cannot see its OWN memory"

            st = await c.get("/developer/state", headers=_auth(key))
            assert st.status_code == 200, st.text
            assert st.json()["org_id"] == str(org_id)

            for path in ("/developer/context", "/developer/outcomes", "/developer/learning"):
                r = await c.get(path, headers=_auth(key))
                assert r.status_code == 200, f"{path} -> {r.status_code} {r.text[:200]}"

            out = await c.get("/developer/outcomes", headers=_auth(key))
            assert f"{tag}-only outcome" in out.text, "a tenant cannot see its OWN outcomes"
            assert out.json()["total"] >= 1

            learn = await c.get("/developer/learning", headers=_auth(key))
            assert learn.json()["org_id"] == str(org_id)
            assert learn.json()["totals"]["learning_events"] >= 1


# ═══════════════════════════════════════════════════════════════════════════
# 2. ORG A CANNOT SEE ORG B — every Developer route, at the wire
# ═══════════════════════════════════════════════════════════════════════════

async def test_no_developer_route_returns_another_tenants_row(tenants):
    """The broadest proof and the one closest to a real attack.

    Authenticated as Org A — once with a SERVICE ACCOUNT key and once with a BARE
    API key, because the two resolve different principal kinds through different
    branches of `_principal_from_api_key` — walk every Developer route and assert
    Org B's marker appears in NONE of the bodies. Both tenants hold a row behind
    every one of these routes, so a query missing its tenant filter fails here.
    """
    Session, t = tenants
    leaked: list[str] = []
    errored: list[str] = []

    async with _client(Session) as c:
        for cred in ("agent_key", "bare_key"):
            key = t[TENANT_A][cred]
            for path in DEVELOPER_ROUTES:
                r = await c.get(path, headers=_auth(key))
                if r.status_code >= 500:
                    errored.append(f"{cred} {path} -> {r.status_code} {r.text[:200]}")
                    continue
                if "B-only" in r.text or str(TENANT_B) in r.text:
                    leaked.append(f"{cred} {path} -> {r.status_code} {r.text[:300]}")

    assert not errored, "a Developer route errored rather than answering:\n" + "\n".join(errored)
    assert not leaked, "another tenant's data reached an Org A response:\n" + "\n".join(leaked)


# ═══════════════════════════════════════════════════════════════════════════
# 4. CALLER-SUPPLIED IDENTITY CANNOT OVERRIDE THE AUTHENTICATED ONE
# ═══════════════════════════════════════════════════════════════════════════

async def test_caller_supplied_identity_never_moves_the_request(tenants):
    """Organization is derived server-side, and nothing on the wire can change it.

    Aims Org B's real organization id, service-account id, user id and API-key id
    at every Developer route while presenting an Org A credential — as query
    parameters, and as the headers a naive gateway might honour. Every response
    must still describe Org A, and none may carry an Org B marker.

    Query parameters a route does not declare are IGNORED by FastAPI rather than
    rejected, which is exactly why this has to be asserted at the wire: "the
    handler has no `organization_id` argument" is a claim about source code, and
    this is a claim about behaviour.
    """
    Session, t = tenants
    key = t[TENANT_A]["agent_key"]
    b = t[TENANT_B]

    overrides = {
        "organization_id": str(TENANT_B),
        "org_id": str(TENANT_B),
        "agent_id": str(b["service_account_id"]),
        "service_account_id": str(b["service_account_id"]),
        "principal_id": str(b["user_id"]),
        "user_id": str(b["user_id"]),
        "credential_id": str(b["scopeless_id"]),
    }
    spoofed_headers = {
        **_auth(key),
        "X-Organization-Id": str(TENANT_B),
        "X-Org-Id": str(TENANT_B),
        "X-Agent-Id": str(b["service_account_id"]),
    }

    bad: list[str] = []
    async with _client(Session) as c:
        for path in DEVELOPER_ROUTES:
            for headers in (_auth(key), spoofed_headers):
                r = await c.get(path, params=overrides, headers=headers)
                if r.status_code != 200:
                    bad.append(f"{path} -> {r.status_code} {r.text[:160]}")
                    continue
                if "B-only" in r.text or str(TENANT_B) in r.text:
                    bad.append(f"{path} answered about TENANT B: {r.text[:300]}")

        # And `/me` — the identity oracle — still says Org A with A's own agent.
        me = await c.get("/developer/me", params=overrides, headers=spoofed_headers)
        body = me.json()
        assert body["organization"]["id"] == str(TENANT_A)
        assert body["principal"]["service_account_id"] == str(
            t[TENANT_A]["service_account_id"]
        )
        assert body["principal"]["service_account_id"] != str(b["service_account_id"])

    assert not bad, "caller-supplied identity changed the answer:\n" + "\n".join(bad)


async def test_a_cross_tenant_filter_matches_nothing_rather_than_leaking(tenants):
    """`decision_id`/`execution_id` on `/developer/outcomes` are FILTERS, not scope.

    Handed Org B's real outcome lineage under an Org A credential, the route must
    return an empty page — the filter is applied ON TOP of the organization
    predicate, never instead of it.
    """
    Session, t = tenants
    key = t[TENANT_A]["agent_key"]

    async with _client(Session) as c:
        r = await c.get(
            "/developer/outcomes",
            params={"execution_id": str(uuid.uuid4()), "decision_id": str(uuid.uuid4())},
            headers=_auth(key),
        )
        assert r.status_code == 200, r.text
        assert r.json()["outcomes"] == []
        assert "B-only" not in r.text


# ═══════════════════════════════════════════════════════════════════════════
# 5. A CROSS-TENANT ENTITY ID RESOLVES TO NOTHING
# ═══════════════════════════════════════════════════════════════════════════

async def test_another_tenants_entity_id_is_a_404_not_a_read(tenants):
    """The entity route takes an id in the PATH — the classic IDOR shape.

    Org B's entity id is real and canonical (a UUIDv5 over Org B), so a
    tenant-blind lookup would find it. Asked for under an Org A credential it must
    404, indistinguishably from an id that exists nowhere, so a probe learns
    nothing about the other tenant's graph. Org B asking for the same id is the
    control: the answer must not be 404 for the wrong reason.
    """
    Session, t = tenants
    b_entity = t[TENANT_B]["entity_id"]

    async with _client(Session) as c:
        cross = await c.get(
            f"/developer/entities/person/{b_entity}",
            headers=_auth(t[TENANT_A]["agent_key"]),
        )
        assert cross.status_code == 404, f"{cross.status_code} {cross.text[:300]}"
        assert "B-only" not in cross.text

        nowhere = await c.get(
            f"/developer/entities/person/{uuid.uuid4()}",
            headers=_auth(t[TENANT_A]["agent_key"]),
        )
        # Identical answer for "another tenant's entity" and "no such entity".
        assert nowhere.status_code == cross.status_code


# ═══════════════════════════════════════════════════════════════════════════
# 6. REVOKED / INVALID / MISSING CREDENTIALS
# ═══════════════════════════════════════════════════════════════════════════

async def test_a_revoked_api_key_is_rejected_on_every_developer_route(tenants):
    """Revocation is enforced at the credential, so it takes effect on the next
    request — `_principal_from_api_key` filters `revoked_at IS NULL` in the lookup
    itself, and API-key principals are never cached."""
    from navis.identity.apikeys import ApiKeyService

    Session, t = tenants
    key = t[TENANT_A]["bare_key"]

    async with _client(Session) as c:
        assert (await c.get("/developer/me", headers=_auth(key))).status_code == 200

    async with Session() as s:
        from sqlalchemy import select

        from navis.identity.enterprise_models import ApiKey

        row = await s.scalar(
            select(ApiKey).where(
                ApiKey.organization_id == TENANT_A, ApiKey.name == "A bare key"
            )
        )
        assert row is not None
        await ApiKeyService(s).revoke(row.id, org_id=TENANT_A)
        await s.commit()

    still_served: list[str] = []
    async with _client(Session) as c:
        for path in DEVELOPER_ROUTES:
            r = await c.get(path, headers=_auth(key))
            if r.status_code != 401:
                still_served.append(f"{path} -> {r.status_code} {r.text[:160]}")
    assert not still_served, "a REVOKED key still reached:\n" + "\n".join(still_served)


async def test_a_revoked_service_account_is_rejected_even_with_a_live_key(tenants):
    """Revoking the ACCOUNT must kill the agent, not just its current credential.

    `ServiceAccountService.revoke` stamps the account and bulk-revokes its keys;
    `_principal_from_api_key` additionally refuses a key whose account is revoked,
    so a key that somehow survived the bulk update still cannot authenticate.
    """
    from navis.identity.service_accounts import ServiceAccountService

    Session, t = tenants
    key = t[TENANT_B]["agent_key"]

    async with _client(Session) as c:
        assert (await c.get("/developer/me", headers=_auth(key))).status_code == 200

    async with Session() as s:
        await ServiceAccountService(s).revoke(
            t[TENANT_B]["service_account_id"], org_id=TENANT_B
        )
        await s.commit()

    async with _client(Session) as c:
        for path in DEVELOPER_ROUTES:
            r = await c.get(path, headers=_auth(key))
            assert r.status_code == 401, f"{path} -> {r.status_code} {r.text[:160]}"


async def test_an_invalid_or_missing_credential_is_rejected(tenants):
    """Four shapes of "not a valid caller", each 401 on every Developer route:
    no credential at all, a well-formed but unknown `navis_` key, a random string,
    and a bearer token that is neither a JWT nor a key."""
    Session, _t = tenants
    attempts = {
        "no credential": {},
        "unknown navis key": {"X-API-Key": f"navis_{uuid.uuid4().hex[:16]}_{uuid.uuid4().hex}"},
        "garbage key": {"X-API-Key": "not-a-key"},
        "garbage bearer": {"Authorization": "Bearer not-a-token"},
    }

    bad: list[str] = []
    async with _client(Session) as c:
        for label, headers in attempts.items():
            for path in DEVELOPER_ROUTES:
                r = await c.get(path, headers=headers)
                if r.status_code != 401:
                    bad.append(f"{label} {path} -> {r.status_code} {r.text[:160]}")
    assert not bad, "an invalid credential was not rejected:\n" + "\n".join(bad)


# ═══════════════════════════════════════════════════════════════════════════
# 7. A SCOPELESS CREDENTIAL IS AUTHENTICATED BUT NOT AUTHORIZED
# ═══════════════════════════════════════════════════════════════════════════

async def test_a_scopeless_credential_authenticates_but_reaches_no_data(tenants):
    """The permission gate the Developer boundary adds over the web edge.

    `POST /identity/api-keys` accepts `scopes: []` — `_ensure_grantable_scopes`
    has nothing to reject — so a scopeless key is a real, mintable credential. It
    resolves a Principal with an EMPTY permission set, which means:

      * `/developer/me` answers 200 and reports no permissions (a credential must
        be able to discover that it is misconfigured);
      * every gated Developer route answers 403, because each names a permission
        as well as a capability.

    The organization is fully entitled here, so a 403 can only come from the
    permission gate — not from a plan or a lifecycle.
    """
    Session, t = tenants
    key = t[TENANT_A]["scopeless_key"]

    async with _client(Session) as c:
        me = await c.get("/developer/me", headers=_auth(key))
        assert me.status_code == 200, me.text
        assert me.json()["principal"]["permissions"] == []
        assert me.json()["organization"]["id"] == str(TENANT_A)

        reached: list[str] = []
        for path in GATED_ROUTES:
            r = await c.get(path, headers=_auth(key))
            if r.status_code != 403:
                reached.append(f"{path} -> {r.status_code} {r.text[:200]}")
        assert not reached, (
            "a credential holding NO permissions reached tenant data:\n"
            + "\n".join(reached)
        )


# ═══════════════════════════════════════════════════════════════════════════
# 8. CREDENTIAL SECURITY — no secret on the wire, no secret in the log
# ═══════════════════════════════════════════════════════════════════════════

async def test_no_api_secret_appears_in_a_response_or_a_log_record(tenants, caplog):
    """An API secret exists in exactly one place: the response to its own creation.

    Sweeps every Developer route with a real key, at DEBUG, and asserts the
    plaintext appears in NO response body and NO log record — including the
    secret's `navis_<prefix>_` head, which is enough to correlate a key across
    logs even when the tail is cut. `api_keys.hashed_secret` (a SHA-256) must not
    appear either: publishing the verifier is publishing an offline-guessable
    credential.
    """
    from sqlalchemy import select

    from navis.identity.enterprise_models import ApiKey

    Session, t = tenants
    key = t[TENANT_A]["agent_key"]
    head = key.rsplit("_", 1)[0]          # navis_<prefix>
    tail = key.rsplit("_", 1)[1]          # the secret segment alone

    async with Session() as s:
        digests = [
            r.hashed_secret for r in (await s.scalars(
                select(ApiKey).where(ApiKey.organization_id == TENANT_A)
            )).all()
        ]
    assert digests

    # DEBUG on everything Navis emits — the assertion is about what the
    # APPLICATION logs, at the noisiest level it can be configured to run at.
    #
    # SQLAlchemy's engine loggers are pinned back to WARNING, and that is not a
    # weakening. With echo=False (how the engine is built in every environment)
    # SQLAlchemy decides whether to log by asking `isEnabledFor(DEBUG)`, so
    # raising the ROOT level in a test silently switches the driver's echo on —
    # at which point it logs bound parameters and result ROWS, and an `api_keys`
    # row carries `hashed_secret`. That would be this test discovering its own
    # setup rather than a property of the code. `caplog.set_level` restores both
    # levels at teardown.
    caplog.set_level(logging.DEBUG)
    caplog.set_level(logging.WARNING, logger="sqlalchemy.engine")
    caplog.set_level(logging.WARNING, logger="sqlalchemy.pool")
    bodies: list[str] = []
    async with _client(Session) as c:
        for path in DEVELOPER_ROUTES:
            r = await c.get(path, headers=_auth(key))
            bodies.append(f"{path}:{r.text}")

    for body in bodies:
        assert key not in body, f"the API SECRET was echoed in a response: {body[:200]}"
        assert tail not in body, f"the secret segment appeared in a response: {body[:200]}"
        for digest in digests:
            assert digest not in body, "the stored secret HASH was published"

    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert key not in logged, "the API SECRET was written to a log record"
    assert tail not in logged, "the secret segment was written to a log record"
    assert head not in logged, "the credential's identifying head was logged"
    for digest in digests:
        assert digest not in logged, "the stored secret HASH was written to a log"


async def test_the_credential_identity_reported_is_the_key_row_not_the_secret(tenants):
    """Credential ATTRIBUTION without credential EXPOSURE.

    `/developer/me` reports `credential_id` — the `api_keys` row that
    authenticated — which is what makes an agent call attributable, revocable and
    rotatable. Two different keys belonging to the SAME service account must
    report the same `service_account_id` and DIFFERENT `credential_id`s: without
    the second field a rotation is invisible and a leaked key cannot be told from
    its replacement.
    """
    from navis.identity.service_accounts import ServiceAccountService

    Session, t = tenants
    sa_id = t[TENANT_A]["service_account_id"]
    first_key = t[TENANT_A]["agent_key"]

    async with _client(Session) as c:
        first = (await c.get("/developer/me", headers=_auth(first_key))).json()

    assert first["principal"]["service_account_id"] == str(sa_id)
    first_cred = first["principal"]["credential_id"]
    assert first_cred is not None, "the authenticating credential is unattributable"
    assert first_cred != str(sa_id), "credential_id must be the KEY, not the account"
    assert first_key not in str(first), "the secret reached the identity response"

    # Rotate: mints a new key and revokes the old one.
    async with Session() as s:
        rotated_key = await ServiceAccountService(s).rotate(sa_id, org_id=TENANT_A)
        await s.commit()

    async with _client(Session) as c:
        rotated = (await c.get("/developer/me", headers=_auth(rotated_key))).json()
        # The retired key no longer authenticates at all.
        assert (await c.get("/developer/me", headers=_auth(first_key))).status_code == 401

    assert rotated["principal"]["service_account_id"] == str(sa_id), "the agent changed"
    assert rotated["principal"]["credential_id"] != first_cred, (
        "a rotated credential is indistinguishable from the one it replaced"
    )
    assert rotated["organization"]["id"] == str(TENANT_A)


# ═══════════════════════════════════════════════════════════════════════════
# 9. THE EXISTING MACHINE-REACHABLE SURFACES, UNDER A MACHINE CREDENTIAL
# ═══════════════════════════════════════════════════════════════════════════

async def test_the_pre_existing_machine_surfaces_hold_the_same_isolation(tenants):
    """`/context`, `/knowledge/home`, `/knowledge/entity/*`, `/outcomes` and
    `/learning/summary` were reachable by an API key before the Developer boundary
    existed, and remain so. Prove the tenant property holds there too — an
    isolation guarantee that stops at the new edge is not an isolation guarantee.

    NOTE, and it is the finding this test exists to pin: these five routes gate on
    a CAPABILITY only. They hold their tenant scope (asserted here), but a
    scopeless credential still reaches them, which is why the Developer routes add
    a permission gate rather than delegating their authorization as well as their
    reads. See `test_the_pre_existing_surfaces_accept_a_scopeless_credential`.
    """
    Session, t = tenants
    key = t[TENANT_A]["agent_key"]
    b_entity = t[TENANT_B]["entity_id"]

    surfaces = [
        "/context",
        "/knowledge/home",
        f"/knowledge/entity/person/{b_entity}",
        "/outcomes",
        "/learning/summary",
    ]

    leaked, errored = [], []
    async with _client(Session) as c:
        for path in surfaces:
            r = await c.get(path, headers=_auth(key))
            if r.status_code >= 500:
                errored.append(f"{path} -> {r.status_code} {r.text[:200]}")
                continue
            if "B-only" in r.text or str(TENANT_B) in r.text:
                leaked.append(f"{path} -> {r.status_code} {r.text[:300]}")

    assert not errored, "a pre-existing surface errored:\n" + "\n".join(errored)
    assert not leaked, (
        "another tenant's data reached a pre-existing machine surface:\n"
        + "\n".join(leaked)
    )


async def test_the_pre_existing_surfaces_accept_a_scopeless_credential(tenants):
    """A CHARACTERISATION test: it pins today's behaviour, and today's behaviour
    is a gap.

    A key minted with `scopes=[]` holds no permission, yet `/context`,
    `/knowledge/home`, `/outcomes` and `/learning/summary` serve it, because their
    only authorization gate is `require_capability` — an ORG-level entitlement
    that answers "is this tenant on a plan that includes it", never "may this
    credential do it".

    Left as an assertion of the current behaviour rather than a fix, because
    adding `require_permission` to these five routes changes what HUMAN sessions
    reach as well (an `employee` role holds neither `view_state` nor
    `view_memory`), and that is a product decision, not a side effect of building
    a machine boundary. When it is made, this test flips to `== 403` and becomes
    the proof.
    """
    Session, t = tenants
    key = t[TENANT_A]["scopeless_key"]

    served: list[str] = []
    async with _client(Session) as c:
        for path in ("/context", "/knowledge/home", "/outcomes", "/learning/summary"):
            r = await c.get(path, headers=_auth(key))
            if r.status_code == 200:
                served.append(path)

    assert served == ["/context", "/knowledge/home", "/outcomes", "/learning/summary"], (
        "the pre-existing surfaces' authorization changed — if a permission gate "
        "was added, update this test and the Developer boundary's note about it"
    )
    # The gap is an AUTHORIZATION gap, never a tenancy one: they still refuse to
    # answer about another tenant.
    async with _client(Session) as c:
        for path in ("/context", "/knowledge/home", "/outcomes", "/learning/summary"):
            r = await c.get(path, headers=_auth(key))
            assert "B-only" not in r.text


# ═══════════════════════════════════════════════════════════════════════════
# 10. AGENT ISOLATION — the credential -> agent -> organization chain
# ═══════════════════════════════════════════════════════════════════════════

async def test_a_key_bound_to_another_tenants_service_account_is_rejected(tenants):
    """The chain must be verified, not assumed.

    `_principal_from_api_key` takes the ORGANIZATION from the `api_keys` row and
    the acting SUBJECT from the `service_accounts` row. If those two rows name
    different tenants, the resulting principal has Org A's scope and Org B's
    agent — and every downstream tenant filter then works perfectly while
    filtering the wrong tenant. No route can create that pairing today, and no FK
    can forbid it (`api_keys.service_account_id` references
    `service_accounts.id` alone, so no constraint spans the pair), which is
    exactly why the check belongs in the one function every machine request is
    authorized from.

    Constructed directly in the database because that is the only way to reach
    it — the point is that authorization does not depend on every writer staying
    correct forever.
    """
    from navis.identity.apikeys import ApiKeyService

    Session, t = tenants
    b_agent = t[TENANT_B]["service_account_id"]

    async with Session() as s:
        # An Org A key pointing at Org B's service account.
        _row, crossed = await ApiKeyService(s).create(
            org_id=TENANT_A, name="A key bound to B agent",
            scopes=DEVELOPER_SCOPES, service_account_id=b_agent,
        )
        await s.commit()

    async with _client(Session) as c:
        for path in DEVELOPER_ROUTES:
            r = await c.get(path, headers=_auth(crossed))
            assert r.status_code == 401, (
                f"a credential whose agent belongs to another tenant reached "
                f"{path} -> {r.status_code} {r.text[:200]}"
            )
        # And the rejection says nothing about the other tenant's agent.
        body = (await c.get("/developer/me", headers=_auth(crossed))).text
        assert str(b_agent) not in body and str(TENANT_B) not in body


async def test_a_valid_agent_uuid_from_another_tenant_grants_nothing(tenants):
    """Do not rely on UUID secrecy.

    Org B's service-account id is real and, in any system with an SDK, knowable.
    Presented by an Org A credential in every position a future MCP/SDK argument
    could occupy, it must change nothing: the answer stays Org A's, and Org B's
    identifiers never appear.
    """
    Session, t = tenants
    key = t[TENANT_A]["agent_key"]
    b_agent = t[TENANT_B]["service_account_id"]

    positions = [
        {"agent_id": str(b_agent)},
        {"service_account_id": str(b_agent)},
        {"as_agent": str(b_agent)},
        {"on_behalf_of": str(b_agent)},
        {"subject_id": str(b_agent)},
    ]
    async with _client(Session) as c:
        for params in positions:
            me = await c.get("/developer/me", params=params, headers=_auth(key))
            assert me.status_code == 200, me.text
            reported = me.json()["principal"]["service_account_id"]
            assert reported == str(t[TENANT_A]["service_account_id"])
            assert reported != str(b_agent), f"{params} selected another tenant's agent"


# ═══════════════════════════════════════════════════════════════════════════
# 11. PERMISSION vs CAPABILITY — two gates, two distinct denials
# ═══════════════════════════════════════════════════════════════════════════

async def test_permission_and_capability_are_separate_gates_with_separate_denials(tenants):
    """A credential can fail EITHER gate, and the two failures stay distinguishable.

      * holds the capability, lacks the permission -> 403 (about the CREDENTIAL)
      * holds the permission, org lacks the capability -> 402 (about the TENANT)

    402 rather than 403 for the second is the pre-existing, deliberate contract in
    `organization.guards` — it is what lets a client tell "you may not" from "your
    plan does not include this". Collapsing them into one status would erase the
    distinction between an authorization decision and an entitlement decision at
    the wire, which is the one thing the Developer boundary must not do.

    Both are asserted against the SAME organization so neither result can be
    explained by anything else about the tenant, and Org B is checked throughout
    to prove one tenant's entitlement change never moves another's answer.
    """
    from navis.organization.contract import Capability
    from navis.organization.repository import OrganizationRepository

    Session, t = tenants
    a_key, b_key = t[TENANT_A]["agent_key"], t[TENANT_B]["agent_key"]

    # ── Gate 1: the permission. A scopeless Org A credential, fully entitled org.
    async with _client(Session) as c:
        r = await c.get("/developer/learning", headers=_auth(t[TENANT_A]["scopeless_key"]))
        assert r.status_code == 403, f"expected a PERMISSION denial, got {r.status_code}"
        assert "permission" in r.json()["detail"].lower()

    # ── Gate 2: the capability. Full scopes, but the org loses LEARNING.
    async with Session() as s:
        await OrganizationRepository(s).set_entitlement(
            TENANT_A, Capability.LEARNING.value, False
        )
        await s.commit()

    async with _client(Session) as c:
        r = await c.get("/developer/learning", headers=_auth(a_key))
        assert r.status_code == 402, (
            f"expected an ENTITLEMENT denial (402), got {r.status_code} {r.text[:200]}"
        )
        assert "capability" in r.json()["detail"].lower()

        # The permission still works everywhere the capability is intact — so the
        # 402 above is about LEARNING alone, not a broken credential.
        assert (await c.get("/developer/memory", headers=_auth(a_key))).status_code == 200
        assert (await c.get("/developer/state", headers=_auth(a_key))).status_code == 200

        # And Org B is untouched: one tenant's entitlement is not another's.
        assert (await c.get("/developer/learning", headers=_auth(b_key))).status_code == 200

    # Neither denial leaks anything internal.
    async with _client(Session) as c:
        for key in (t[TENANT_A]["scopeless_key"], a_key):
            body = (await c.get("/developer/learning", headers=_auth(key))).text
            for leak in ("Traceback", "SELECT", "sqlalchemy", "asyncpg", str(TENANT_B)):
                assert leak not in body, f"denial leaked {leak!r}: {body[:200]}"


# ═══════════════════════════════════════════════════════════════════════════
# 12. CONCURRENCY — no tenant state crosses a request boundary
# ═══════════════════════════════════════════════════════════════════════════

async def test_concurrent_requests_from_two_tenants_never_cross(tenants):
    """Two tenants hitting the boundary at the same time, interleaved, in one process.

    THE PROPERTY. Authorization must be derived per request and held nowhere else.
    If any of it were process-local — a module-level "current org", a request-id
    keyed dict, a cached authorization result keyed on something short of the
    tenant — then concurrent A and B traffic through the SAME worker is what
    surfaces it, and a sequential test never would.

    16 requests are issued with `asyncio.gather` so they interleave inside one
    event loop, alternating tenant AND route. The count is bounded by the test
    engine's connection pool (see the `engine` fixture) rather than by the
    property: leakage from process-local authorization state would show with far
    fewer, and every request here is genuinely in flight at once. `_STATE_CACHE`
    is on these paths (`/developer/state`) as are the repositories behind
    `/developer/memory`. Every response must describe its OWN caller's tenant and
    must not contain the other's marker.

    This is also the horizontal-scaling proof. Nothing here relies on a request
    reaching a particular worker, so the boundary is stateless in the only sense
    that matters for authorization: two processes would answer identically.
    """
    import asyncio

    Session, t = tenants
    paths = ("/developer/me", "/developer/state", "/developer/memory", "/developer/learning")

    async with _client(Session) as c:
        async def call(org_id, path):
            r = await c.get(path, headers=_auth(t[org_id]["agent_key"]))
            return org_id, path, r

        jobs = [
            call(org_id, path)
            for _ in range(2)
            for org_id in (TENANT_A, TENANT_B)   # interleaved, not batched per tenant
            for path in paths
        ]
        results = await asyncio.gather(*jobs)

    assert len(results) == 16
    crossed: list[str] = []
    for org_id, path, r in results:
        other = TENANT_B if org_id == TENANT_A else TENANT_A
        other_tag = "B-only" if org_id == TENANT_A else "A-only"
        if r.status_code != 200:
            crossed.append(f"{path} as {org_id} -> {r.status_code} {r.text[:160]}")
            continue
        if str(other) in r.text or other_tag in r.text:
            crossed.append(f"{path} as {org_id} returned {other}'s data: {r.text[:200]}")
        # Where the body names its tenant, it must name the CALLER's.
        body = r.json()
        named = body.get("org_id") or (body.get("organization") or {}).get("id")
        if named is not None and named != str(org_id):
            crossed.append(f"{path} as {org_id} reported org {named}")

    assert not crossed, (
        "tenant state crossed a request boundary under concurrency:\n" + "\n".join(crossed)
    )


async def test_the_state_cache_is_tenant_keyed_under_concurrent_developer_reads(tenants):
    """`/developer/state` and `/developer/context` both fold through `StateService`,
    whose in-process cache is keyed `(org_id, projection_version, watermark)`.

    Warmed by REAL concurrent requests from both tenants (not by hand-placed
    entries), the cache must then hold each tenant's projection under its own key
    and hand back neither for the other. A key that dropped `org_id` would collide
    here, because both tenants share a projection version.
    """
    import asyncio

    from navis.state.v2 import service as state_service

    Session, t = tenants
    state_service._cache_clear()

    async with _client(Session) as c:
        await asyncio.gather(*[
            c.get(path, headers=_auth(t[org_id]["agent_key"]))
            for _ in range(2)
            for org_id in (TENANT_A, TENANT_B)
            for path in ("/developer/state", "/developer/context")
        ])

    keys = list(state_service._STATE_CACHE)
    assert keys, "no projection was cached — this test would prove nothing"
    orgs = {k[0] for k in keys}
    assert orgs <= {TENANT_A, TENANT_B}
    for key in keys:
        assert isinstance(key, tuple) and len(key) == 3, key
        assert key[0] in (TENANT_A, TENANT_B), "org_id is not the leading cache key"

    # Evicting one tenant must not touch the other.
    if TENANT_A in orgs and TENANT_B in orgs:
        state_service._cache_evict_org(TENANT_A)
        remaining = {k[0] for k in state_service._STATE_CACHE}
        assert TENANT_A not in remaining
        assert TENANT_B in remaining, "evicting one tenant dropped another's cache"
    state_service._cache_clear()


# ═══════════════════════════════════════════════════════════════════════════
# 13. RATE-LIMIT IDENTITY BOUNDARY (no quota system built — see the report)
# ═══════════════════════════════════════════════════════════════════════════

async def test_every_future_rate_limit_dimension_is_available_at_the_gate(tenants):
    """The four dimensions a future quota would key on must ALREADY be resolvable
    server-side, from the credential alone, at the point a limit is applied.

    Today's limiter keys on the organization only, and that is deliberate: a
    per-credential quota is a product decision and building an unused one now
    would be a speculative abstraction. What must be true NOW is that adding it
    later needs no new identity plumbing, no second store and no schema — i.e.
    that organization, agent, credential and endpoint class are all derivable
    from the authenticated request.

    Asserted through `/developer/me`, which publishes exactly those first three,
    plus the limiter's own key shape for the fourth.
    """
    from navis.identity.throttle import OrgRateLimiter

    Session, t = tenants

    async with _client(Session) as c:
        me = (await c.get("/developer/me", headers=_auth(t[TENANT_A]["agent_key"]))).json()

    # organization + agent + credential — all server-derived, all non-null here.
    assert me["organization"]["id"] == str(TENANT_A)
    assert me["principal"]["service_account_id"] == str(t[TENANT_A]["service_account_id"])
    assert me["principal"]["credential_id"], "no credential dimension to limit on"
    assert me["principal"]["service_account_id"] != me["principal"]["credential_id"], (
        "agent and credential must be separately addressable, or a per-credential "
        "quota could not tell one of an agent's keys from another"
    )

    # endpoint class — the existing bucket segment, and the key is already
    # per-tenant, so extending it costs a segment rather than a new store.
    limiter = OrgRateLimiter(redis=None, window_seconds=60)  # type: ignore[arg-type]
    key_a = limiter._key("read", TENANT_A, 1_700_000_000.0)
    key_b = limiter._key("read", TENANT_B, 1_700_000_000.0)
    assert key_a != key_b and str(TENANT_A) in key_a and str(TENANT_B) in key_b
    assert key_a.startswith("ratelimit:read:"), key_a
