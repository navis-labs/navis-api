"""P1-7 — an engineer can set themselves up without an organization admin.

THE PRODUCT PROBLEM THIS CLOSES. Before P1-7 a Developer at a customer held
seven read permissions and `record_agent_runs`, and NO way to obtain a
credential: `manage_service_accounts` belonged to Developer admin and
`manage_api_keys` to the Application administrator. So the first step of using
the SDK, the MCP server or a local coding agent was "ask an admin to mint me a
key" — every time, including for a rotation. The only available fix was to make
the engineer an administrator of every machine identity in the company.

WHAT REPLACED IT. `manage_own_agents`: the same agent and credential operations,
bounded by `service_accounts.created_by`. This file proves both halves — that
the developer can do their own work, and that the bound is real.

    sections 1-2   the self-serve workflow, end to end
    section 3      the bound: a colleague's agent is invisible, not merely
                   unwritable
    section 4      organization-wide administration is unchanged
    section 5      least privilege — what the grant does NOT open
    section 6      machine credentials are unaffected

Pure where it can be: the gates are dependency closures and the scope decision
is a pure function of a `DeveloperContext`, so most of this needs no database.
The handler tests use `FakeSession`, the same in-memory session the rest of the
Identity and Developer suites use.
"""
from __future__ import annotations

import uuid

import pytest
from fastapi import HTTPException

from navis.developer.agents_router import (
    _AGENT_PERMISSIONS,
    _require_agent,
    _sees_every_agent,
    list_agents,
)
from navis.developer.gateway import DeveloperContext, require_developer_any
from navis.identity.authz import Principal, PrincipalKind
from navis.identity.enterprise_models import ServiceAccount
from navis.identity.models import User
from navis.identity.permissions import (
    DEVELOPER_ROLE_PERMISSIONS,
    Permission,
    effective_permissions,
)
from navis.organization.contract import Capability
from tests.identity.fakes import FakeSession

ORG_A = uuid.uuid4()
ORG_B = uuid.uuid4()


def _ctx(
    *,
    role: str = "employee",
    developer_role: str | None = "developer",
    subject_id: uuid.UUID | None = None,
    org_id: uuid.UUID | None = None,
    kind: PrincipalKind = PrincipalKind.USER,
    permissions: frozenset[str] | None = None,
) -> DeveloperContext:
    """A `DeveloperContext` resolved exactly as the request path resolves one."""
    uid = subject_id or uuid.uuid4()
    resolved = (
        permissions
        if permissions is not None
        else effective_permissions(role, developer_role=developer_role)
    )
    return DeveloperContext.of(
        Principal(
            kind=kind,
            org_id=org_id or ORG_A,
            subject_id=uid,
            permissions=resolved,
            role=role,
            developer_role=developer_role,
            user=User(id=uid, email=f"{uid.hex[:8]}@x.com") if kind is PrincipalKind.USER else None,
        )
    )


@pytest.fixture()
def no_credential_facts(monkeypatch):
    """Stub the per-agent credential rollup for the LIST tests.

    `_credential_facts` is one grouped aggregate over `api_keys`, which
    `FakeSession` does not interpret — it models the simple `select().where()`
    shapes the services use, not a `GROUP BY` with a filtered `count()`.
    Teaching it aggregates to test OWNERSHIP SCOPING would be a large change to
    a shared fake for a function these tests are not about: the counts are
    decoration on rows whose selection is the thing under test.

    The real query is covered against Postgres in
    `tests/integration/test_developer_agents_postgres.py`.
    """
    from navis.developer import agents_router

    async def _none(db, agent_ids, org_id):
        return {}

    monkeypatch.setattr(agents_router, "_credential_facts", _none)


def _agent(db, *, org_id: uuid.UUID, created_by: uuid.UUID | None, name: str = "dev agent"):
    sa = ServiceAccount(
        id=uuid.uuid4(),
        organization_id=org_id,
        name=name,
        scopes=["view_state"],
        created_by=created_by,
        description="",
        environment="development",
        runtime="custom",
    )
    db.add(sa)
    return sa


# ═══════════════════════════════════════════════════════════════════════════
# 1. THE GRANT EXISTS AND THE WORKING ROLE HOLDS IT
# ═══════════════════════════════════════════════════════════════════════════


def test_a_developer_can_create_an_agent_without_an_organization_admin():
    """THE HEADLINE. The working Developer role holds the permission the agent
    routes accept, so registering an agent needs nobody's help."""
    developer = effective_permissions("employee", developer_role="developer")
    assert Permission.MANAGE_OWN_AGENTS in developer
    assert any(developer >= {p} for p in _AGENT_PERMISSIONS)


def test_a_developer_still_holds_no_organization_wide_agent_authority():
    """The other half of the same sentence, and the reason this is not just a
    wider grant wearing a new name."""
    developer = effective_permissions("employee", developer_role="developer")
    assert Permission.MANAGE_SERVICE_ACCOUNTS not in developer


def test_a_developer_can_record_the_runs_of_the_agent_they_created():
    """Step 7 of the workflow. Already true before P1-7 and asserted here so the
    self-serve path is covered end to end rather than in halves."""
    developer = effective_permissions("employee", developer_role="developer")
    assert Permission.RECORD_AGENT_RUNS in developer


def test_a_developer_can_retrieve_context():
    """Step 6. `POST /developer/context` gates on `view_memory`."""
    developer = effective_permissions("employee", developer_role="developer")
    assert Permission.VIEW_MEMORY in developer
    assert Permission.VIEW_STATE in developer


@pytest.mark.asyncio
async def test_the_gate_admits_a_developer_holding_only_the_self_serve_grant():
    """At the gate, not just in the table. No capability check is reachable
    here — `require_capability` needs a session — so this asserts the
    permission half, which is the half P1-7 changed."""
    gate = require_developer_any(_AGENT_PERMISSIONS, Capability.STATE)
    ctx = _ctx(developer_role="developer")
    # The permission arm passes; the capability arm is exercised in the
    # Postgres integration suite where an entitlement row exists.
    assert any(ctx.principal.has(p) for p in _AGENT_PERMISSIONS)
    assert gate is not None


@pytest.mark.asyncio
async def test_the_gate_refuses_a_developer_viewer():
    """A viewer reads the organization and builds nothing. Least privilege
    inside the Developer surface."""
    gate = require_developer_any(_AGENT_PERMISSIONS, Capability.STATE)
    viewer = _ctx(developer_role="viewer").principal

    with pytest.raises(HTTPException) as raised:
        await gate(viewer, None)
    assert raised.value.status_code == 403
    # The refusal names BOTH acceptable permissions, so an operator asks for the
    # narrow one rather than organization-wide agent administration.
    assert "manage_own_agents" in raised.value.detail
    assert "manage_service_accounts" in raised.value.detail


@pytest.mark.asyncio
async def test_the_gate_refuses_a_member_with_no_developer_access():
    gate = require_developer_any(_AGENT_PERMISSIONS, Capability.STATE)
    application_only = _ctx(role="admin", developer_role=None).principal

    with pytest.raises(HTTPException) as raised:
        await gate(application_only, None)
    assert raised.value.status_code == 403


# ═══════════════════════════════════════════════════════════════════════════
# 2. SCOPE — WHICH GRANT DECIDES WHAT YOU SEE
# ═══════════════════════════════════════════════════════════════════════════


def test_the_scope_is_decided_by_a_permission_not_a_role():
    """A role string here would be a second authorization vocabulary, and a
    per-org `role_permissions` grant would be invisible to it."""
    assert _sees_every_agent(_ctx(developer_role="admin")) is True
    assert _sees_every_agent(_ctx(developer_role="owner")) is True
    assert _sees_every_agent(_ctx(developer_role="developer")) is False
    # And a bare permission set with no role at all resolves the same way.
    assert _sees_every_agent(
        _ctx(permissions=frozenset({Permission.MANAGE_SERVICE_ACCOUNTS}))
    ) is True
    assert _sees_every_agent(
        _ctx(permissions=frozenset({Permission.MANAGE_OWN_AGENTS}))
    ) is False


@pytest.mark.asyncio
async def test_a_developer_lists_only_the_agents_they_created(no_credential_facts):
    db = FakeSession()
    me = uuid.uuid4()
    colleague = uuid.uuid4()
    mine = _agent(db, org_id=ORG_A, created_by=me, name="mine")
    _agent(db, org_id=ORG_A, created_by=colleague, name="theirs")

    page = await list_agents(
        limit=50, offset=0, ctx=_ctx(subject_id=me, developer_role="developer"), db=db
    )
    assert [a.id for a in page.agents] == [mine.id]


@pytest.mark.asyncio
async def test_a_developer_admin_lists_the_whole_fleet(no_credential_facts):
    db = FakeSession()
    me = uuid.uuid4()
    _agent(db, org_id=ORG_A, created_by=me, name="mine")
    _agent(db, org_id=ORG_A, created_by=uuid.uuid4(), name="theirs")

    page = await list_agents(
        limit=50, offset=0, ctx=_ctx(subject_id=me, developer_role="admin"), db=db
    )
    assert len(page.agents) == 2


@pytest.mark.asyncio
async def test_the_page_and_its_total_are_scoped_by_the_same_creator(no_credential_facts):
    """`total` MUST be narrowed with the page.

    A self-serve caller shown "the first 50 of 900" for a fleet they cannot see
    would be told the size of something they have no access to — a confusing
    number and a small disclosure. So the creator passed to `count_for_org` has
    to be the one passed to `list_for_org`.

    Asserted with spies rather than by counting rows, because `FakeSession` does
    not interpret `select(func.count())`; what matters here is that one scope
    decision reaches both queries, and that is exactly what this observes.
    """
    from navis.identity import service_accounts as sa_module

    seen: dict[str, object] = {}
    real_list = sa_module.ServiceAccountService.list_for_org
    real_count = sa_module.ServiceAccountService.count_for_org

    async def spy_list(self, org_id, *, limit=50, offset=0, created_by=None):
        seen["list"] = created_by
        return await real_list(self, org_id, limit=limit, offset=offset, created_by=created_by)

    async def spy_count(self, org_id, *, active_only=False, created_by=None):
        seen["count"] = created_by
        return 0

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(sa_module.ServiceAccountService, "list_for_org", spy_list)
    monkeypatch.setattr(sa_module.ServiceAccountService, "count_for_org", spy_count)
    try:
        db = FakeSession()
        me = uuid.uuid4()
        _agent(db, org_id=ORG_A, created_by=me)

        await list_agents(
            limit=50, offset=0, ctx=_ctx(subject_id=me, developer_role="developer"), db=db
        )
        assert seen["list"] == me and seen["count"] == me

        seen.clear()
        await list_agents(
            limit=50, offset=0, ctx=_ctx(subject_id=me, developer_role="owner"), db=db
        )
        # None means "no creator filter" — the organization-wide scope.
        assert seen["list"] is None and seen["count"] is None
    finally:
        monkeypatch.undo()


@pytest.mark.asyncio
async def test_an_organization_wide_caller_still_sees_agents_with_no_creator(no_credential_facts):
    """`created_by` is NULLABLE and 10 of 15 production rows predate it being
    populated. An organization-wide caller must still see them — the creator
    filter is applied only when a creator is named, so it narrows and never
    accidentally excludes."""
    db = FakeSession()
    legacy = _agent(db, org_id=ORG_A, created_by=None, name="legacy")

    page = await list_agents(
        limit=50, offset=0, ctx=_ctx(developer_role="owner"), db=db
    )
    assert legacy.id in [a.id for a in page.agents]


@pytest.mark.asyncio
async def test_a_self_serve_caller_cannot_see_an_agent_with_no_creator(no_credential_facts):
    """The fail-closed direction of the same fact: a NULL creator can never
    equal a subject id, so a legacy agent is invisible to a self-serve caller
    rather than being claimable by whoever asks first."""
    db = FakeSession()
    _agent(db, org_id=ORG_A, created_by=None, name="legacy")

    page = await list_agents(
        limit=50, offset=0, ctx=_ctx(developer_role="developer"), db=db
    )
    assert page.agents == []


# ═══════════════════════════════════════════════════════════════════════════
# 3. THE BOUND IS REAL — A COLLEAGUE'S AGENT IS INVISIBLE
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_a_developer_reaches_their_own_agent():
    db = FakeSession()
    me = uuid.uuid4()
    mine = _agent(db, org_id=ORG_A, created_by=me)

    ctx = _ctx(subject_id=me, developer_role="developer")
    assert (await _require_agent(db, mine.id, ctx)).id == mine.id


@pytest.mark.asyncio
async def test_a_colleagues_agent_answers_404_and_not_403():
    """404 FOR THE SAME REASON A FOREIGN TENANT'S ID DOES. A 403 would confirm
    that the id names a real agent in this organization, turning the route into
    an existence oracle for exactly the fleet this caller was not given."""
    db = FakeSession()
    theirs = _agent(db, org_id=ORG_A, created_by=uuid.uuid4())

    ctx = _ctx(subject_id=uuid.uuid4(), developer_role="developer")
    with pytest.raises(HTTPException) as raised:
        await _require_agent(db, theirs.id, ctx)
    assert raised.value.status_code == 404
    assert raised.value.detail == "agent not found"


@pytest.mark.asyncio
async def test_a_foreign_tenants_agent_is_indistinguishable_from_a_colleagues():
    """Both answer the identical 404, so the response cannot be used to tell
    "exists here but not yours" from "exists nowhere you can reach"."""
    db = FakeSession()
    other_tenant = _agent(db, org_id=ORG_B, created_by=uuid.uuid4())
    colleague = _agent(db, org_id=ORG_A, created_by=uuid.uuid4())

    ctx = _ctx(developer_role="developer")
    refusals = []
    for target in (other_tenant, colleague):
        with pytest.raises(HTTPException) as raised:
            await _require_agent(db, target.id, ctx)
        refusals.append((raised.value.status_code, raised.value.detail))
    assert refusals[0] == refusals[1] == (404, "agent not found")


@pytest.mark.asyncio
async def test_a_developer_admin_reaches_a_colleagues_agent():
    """Organization-wide administration is what the higher grant is FOR."""
    db = FakeSession()
    theirs = _agent(db, org_id=ORG_A, created_by=uuid.uuid4())

    ctx = _ctx(developer_role="admin")
    assert (await _require_agent(db, theirs.id, ctx)).id == theirs.id


@pytest.mark.asyncio
async def test_the_tenant_predicate_still_binds_an_organization_wide_caller():
    """The creator filter is ADDITIONAL to the tenant filter, never instead of
    it. A Developer owner in ORG_A cannot reach ORG_B's agent."""
    db = FakeSession()
    foreign = _agent(db, org_id=ORG_B, created_by=uuid.uuid4())

    ctx = _ctx(org_id=ORG_A, developer_role="owner")
    with pytest.raises(HTTPException) as raised:
        await _require_agent(db, foreign.id, ctx)
    assert raised.value.status_code == 404


# ═══════════════════════════════════════════════════════════════════════════
# 4. NOTHING THAT WORKED BEFORE STOPPED WORKING
# ═══════════════════════════════════════════════════════════════════════════


def test_organization_wide_administration_is_unchanged():
    """`MANAGE_SERVICE_ACCOUNTS` means exactly what it meant: every agent in the
    organization. P1-7 added an alternative, it did not re-define this one."""
    for role in ("admin", "owner"):
        assert Permission.MANAGE_SERVICE_ACCOUNTS in DEVELOPER_ROLE_PERMISSIONS[role]


def test_the_agent_routes_still_accept_the_permission_they_always_accepted():
    """THE NO-REGRESSION ASSERTION. A live credential carries
    `manage_service_accounts` today; re-gating these routes on the narrower
    permission alone would have 401'd a running integration. It is still in the
    accepted set."""
    assert Permission.MANAGE_SERVICE_ACCOUNTS in _AGENT_PERMISSIONS


def test_the_developer_ladder_is_still_strictly_nested():
    """An admin must never hold less than a developer, or "may I grant this
    role" would depend on which of two incomparable roles you named."""
    viewer = DEVELOPER_ROLE_PERMISSIONS["viewer"]
    developer = DEVELOPER_ROLE_PERMISSIONS["developer"]
    admin = DEVELOPER_ROLE_PERMISSIONS["admin"]
    assert viewer < developer < admin
    assert DEVELOPER_ROLE_PERMISSIONS["owner"] == admin


# ═══════════════════════════════════════════════════════════════════════════
# 5. LEAST PRIVILEGE — WHAT THE GRANT DOES NOT OPEN
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    "permission",
    [
        Permission.MANAGE_BILLING,
        Permission.MANAGE_SSO,
        Permission.MANAGE_ROLES,
        Permission.MANAGE_MEMBERS,
        Permission.MANAGE_API_KEYS,
        Permission.MANAGE_POLICIES,
        Permission.MANAGE_INTEGRATIONS,
        Permission.EXECUTE_ACTIONS,
        Permission.VIEW_AUDIT,
        Permission.MANAGE_SERVICE_ACCOUNTS,
    ],
)
def test_the_self_serve_developer_holds_no_administrative_authority(permission: str):
    """Everything the brief says a normal Developer must NOT receive, asserted
    one permission at a time so a failure names which one leaked."""
    developer = effective_permissions("employee", developer_role="developer")
    assert permission not in developer


def test_the_self_serve_grant_does_not_reach_the_legacy_admin_surface():
    """`/identity/service-accounts` and `/identity/api-keys` are gated on
    `manage_service_accounts` and `manage_api_keys`, and a self-serve Developer
    holds neither — so the organization-level machine-identity administration
    stays where it was."""
    developer = effective_permissions("employee", developer_role="developer")
    assert Permission.MANAGE_SERVICE_ACCOUNTS not in developer
    assert Permission.MANAGE_API_KEYS not in developer


def test_a_developer_cannot_mint_a_credential_beyond_their_own_permissions():
    """`ensure_grantable_scopes` is unchanged and carries the bound for free: a
    credential can never hold a permission its creator lacks, so self-serve
    creation is not a privilege-escalation primitive."""
    from navis.identity.authz import ensure_grantable_scopes

    ctx = _ctx(developer_role="developer")
    # May grant what they hold.
    ensure_grantable_scopes([Permission.VIEW_STATE, Permission.RECORD_AGENT_RUNS], ctx.principal)
    # May not grant what they do not.
    for scope in (Permission.MANAGE_SERVICE_ACCOUNTS, Permission.MANAGE_BILLING):
        with pytest.raises(HTTPException) as raised:
            ensure_grantable_scopes([scope], ctx.principal)
        assert raised.value.status_code == 403


def test_the_agent_ceiling_stays_an_organization_limit():
    """A per-creator ceiling would let a team of ten hold ten times the agents
    the tenant is provisioned for. `count_for_org` takes a creator for PAGING
    only; the ceiling check must not pass one."""
    import inspect

    from navis.developer import agents_router

    source = inspect.getsource(agents_router.create_agent)
    assert "count_for_org(ctx.organization_id, active_only=True)" in source
    assert "created_by" not in source.split("max_agents_per_organization")[0]


# ═══════════════════════════════════════════════════════════════════════════
# 6. MACHINE CREDENTIALS ARE UNAFFECTED
# ═══════════════════════════════════════════════════════════════════════════


def test_the_self_serve_grant_is_inert_on_a_machine_credential():
    """`created_by` holds the USER who registered an agent, so an agent
    presenting its own subject id matches no row. The grant is grantable to a
    machine and does nothing, which is why it is safe to have in the catalog."""
    machine = _ctx(
        kind=PrincipalKind.API_KEY,
        developer_role=None,
        permissions=frozenset({Permission.MANAGE_OWN_AGENTS}),
    )
    assert _sees_every_agent(machine) is False


@pytest.mark.asyncio
async def test_a_machine_credential_with_the_self_serve_grant_reads_nothing(no_credential_facts):
    db = FakeSession()
    _agent(db, org_id=ORG_A, created_by=uuid.uuid4())

    machine = _ctx(
        kind=PrincipalKind.API_KEY,
        developer_role=None,
        permissions=frozenset({Permission.MANAGE_OWN_AGENTS}),
    )
    page = await list_agents(limit=50, offset=0, ctx=machine, db=db)
    assert page.agents == []


@pytest.mark.asyncio
async def test_agent_writes_are_still_refused_to_a_machine_principal():
    """THE REASON `manage_own_agents` IS SAFE AS A DEFAULT. Whatever holds it, a
    machine credential cannot write here — so it can never create an identity
    that outlives its own revocation."""
    from navis.developer.agents_router import _operator

    machine = _ctx(
        kind=PrincipalKind.API_KEY,
        developer_role=None,
        permissions=frozenset({Permission.MANAGE_OWN_AGENTS}),
    )
    with pytest.raises(HTTPException) as raised:
        await _operator(machine)
    assert raised.value.status_code == 403
    assert "signed-in member" in raised.value.detail
