"""TASK 6.4.3 — membership administration over HTTP.

Service-level tests prove the rule; these prove the ROUTES apply it, return safe
status codes, and cannot be steered by request-supplied identity. All four
membership-mutating endpoints are covered:

    POST   /identity/members/invite
    DELETE /identity/members/{user_id}
    POST   /identity/members/{user_id}/role
    DELETE /scim/v2/Users/{user_id}      (the second removal path)

TASK 6.4.4 adds the READ side of the same contract at the bottom of this file:
`GET /auth/me`, which publishes what a membership entitles its holder to. The
write path and the read path have to agree about what a role means, and they
now agree by construction — both resolve through `Principal.permissions`.
"""
from __future__ import annotations

import uuid

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from navis.identity.admin_router import router as admin_router
from navis.identity.authz import (
    Principal,
    PrincipalKind,
    get_principal,
    require_permission,
)
from navis.identity.models import Membership, Organization, Role, User
from navis.identity.permissions import default_permissions_for_role
from navis.identity.router import router as identity_router
from navis.identity.scim_router import router as scim_router
from navis.platform.database import get_session
from tests.identity.fakes import FakeSession

ORG_A = uuid.uuid4()
ORG_B = uuid.uuid4()


def _principal(org_id, role: str, subject_id=None) -> Principal:
    """A principal whose permissions are derived from the role, exactly as
    `_principal_from_jwt` builds them."""
    return Principal(
        kind=PrincipalKind.USER,
        org_id=org_id,
        subject_id=subject_id or uuid.uuid4(),
        permissions=default_permissions_for_role(role),
        role=role,
        user=User(id=subject_id or uuid.uuid4(), email="actor@x.com"),
    )


@pytest.fixture()
def env():
    """An app wired to a fake session, with a swappable authenticated principal."""
    db = FakeSession()
    state: dict = {"principal": None}

    app = FastAPI()
    app.include_router(admin_router)
    app.include_router(scim_router)
    app.dependency_overrides[get_session] = lambda: db
    app.dependency_overrides[get_principal] = lambda: state["principal"]

    def seed(org_id, role: Role) -> uuid.UUID:
        uid = uuid.uuid4()
        db.add(User(id=uid, email=f"{uid.hex[:8]}@x.com"))
        db.add(Membership(user_id=uid, organization_id=org_id, role=role))
        return uid

    def as_(role: str, org_id=ORG_A):
        state["principal"] = _principal(org_id, role)
        return state["principal"]

    return TestClient(app, raise_server_exceptions=False), db, seed, as_


def _roles(db, org_id):
    """Roles currently persisted for an org — the ground truth a denial must leave
    untouched. Read straight from the fake's store so it cannot be confused by the
    same query layer under test."""
    return sorted(
        m.role.value if isinstance(m.role, Role) else str(m.role)
        for m in db.store.get(Membership, [])
        if m.organization_id == org_id
    )


# ── Route-level permission gate ──────────────────────────────────────────────

def test_employee_cannot_reach_membership_administration(env):
    client, db, seed, as_ = env
    target = seed(ORG_A, Role.EMPLOYEE)
    as_("employee")
    assert client.delete(f"/identity/members/{target}").status_code == 403
    assert client.post(f"/identity/members/{target}/role", json={"role": "admin"}).status_code == 403
    assert client.post("/identity/members/invite", json={"email": "x@y.com", "role": "employee"}).status_code == 403


def test_admin_lacks_manage_roles_is_403_not_500(env):
    """`admin` DOES hold MANAGE_ROLES; `coo` does not. The route gate must be the
    thing that answers, with a clean 403."""
    client, db, seed, as_ = env
    target = seed(ORG_A, Role.EMPLOYEE)
    as_("coo")
    assert client.post(f"/identity/members/{target}/role", json={"role": "employee"}).status_code == 403


# ── D. Higher-privileged target, over HTTP ───────────────────────────────────

def test_admin_removing_a_ceo_is_403(env):
    client, db, seed, as_ = env
    victim = seed(ORG_A, Role.CEO)
    as_("admin")
    r = client.delete(f"/identity/members/{victim}")
    assert r.status_code == 403
    assert _roles(db, ORG_A) == ["ceo"], "target survives"


def test_admin_demoting_a_ceo_is_403(env):
    client, db, seed, as_ = env
    victim = seed(ORG_A, Role.CEO)
    as_("admin")
    r = client.post(f"/identity/members/{victim}/role", json={"role": "employee"})
    assert r.status_code == 403
    assert _roles(db, ORG_A) == ["ceo"]


def test_admin_promoting_to_founder_is_403(env):
    client, db, seed, as_ = env
    target = seed(ORG_A, Role.EMPLOYEE)
    as_("admin")
    r = client.post(f"/identity/members/{target}/role", json={"role": "founder"})
    assert r.status_code == 403
    assert _roles(db, ORG_A) == ["employee"]


def test_scim_deprovision_of_a_higher_target_is_403_not_404(env):
    """The second removal path. It previously mapped every MemberError subclass to
    404, so a privilege denial read as 'already gone' and an IdP would stop
    retrying while believing the user was deprovisioned."""
    client, db, seed, as_ = env
    victim = seed(ORG_A, Role.CEO)
    as_("admin")
    r = client.delete(f"/scim/v2/Users/{victim}")
    assert r.status_code == 403
    assert _roles(db, ORG_A) == ["ceo"]


# ── B. Allowed cases still work ──────────────────────────────────────────────

def test_founder_may_remove_a_ceo(env):
    client, db, seed, as_ = env
    victim = seed(ORG_A, Role.CEO)
    as_("founder")
    assert client.delete(f"/identity/members/{victim}").status_code == 200
    assert _roles(db, ORG_A) == []


def test_admin_may_remove_an_employee(env):
    client, db, seed, as_ = env
    victim = seed(ORG_A, Role.EMPLOYEE)
    as_("admin")
    assert client.delete(f"/identity/members/{victim}").status_code == 200
    assert _roles(db, ORG_A) == []


# ── A. Cross-org, over HTTP ──────────────────────────────────────────────────

def test_founder_of_a_cannot_touch_a_member_of_b(env):
    client, db, seed, as_ = env
    victim_b = seed(ORG_B, Role.EMPLOYEE)
    as_("founder", org_id=ORG_A)
    assert client.delete(f"/identity/members/{victim_b}").status_code == 404
    assert client.post(f"/identity/members/{victim_b}/role", json={"role": "admin"}).status_code == 404
    assert client.delete(f"/scim/v2/Users/{victim_b}").status_code == 404
    assert _roles(db, ORG_B) == ["employee"], "org B untouched"


# ── G. Request manipulation ──────────────────────────────────────────────────

def test_organization_id_in_the_body_is_ignored(env):
    """Org scope comes from Principal.org_id. A body field naming another tenant
    must not redirect the operation."""
    client, db, seed, as_ = env
    victim_b = seed(ORG_B, Role.EMPLOYEE)
    as_("founder", org_id=ORG_A)
    r = client.post(
        f"/identity/members/{victim_b}/role",
        json={"role": "admin", "organization_id": str(ORG_B), "org_id": str(ORG_B)},
    )
    assert r.status_code == 404
    assert _roles(db, ORG_B) == ["employee"]


def test_injected_permissions_in_the_body_are_ignored(env):
    """Actor permissions come from the Principal, never the request."""
    client, db, seed, as_ = env
    victim = seed(ORG_A, Role.CEO)
    as_("admin")
    r = client.delete(f"/identity/members/{victim}")
    assert r.status_code == 403
    r = client.post(
        f"/identity/members/{victim}/role",
        json={"role": "employee", "permissions": ["execute_actions", "manage_policies"],
              "actor_permissions": ["execute_actions"]},
    )
    assert r.status_code in (403, 422)
    assert _roles(db, ORG_A) == ["ceo"]


def test_unknown_role_value_is_rejected_by_validation(env):
    client, db, seed, as_ = env
    target = seed(ORG_A, Role.EMPLOYEE)
    as_("founder")
    r = client.post(f"/identity/members/{target}/role", json={"role": "sovereign"})
    assert r.status_code == 422
    assert _roles(db, ORG_A) == ["employee"]


# ── E. Last founder, over HTTP ───────────────────────────────────────────────

def test_sole_founder_removal_is_409(env):
    client, db, seed, as_ = env
    sole = seed(ORG_A, Role.FOUNDER)
    as_("founder")
    assert client.delete(f"/identity/members/{sole}").status_code == 409
    assert _roles(db, ORG_A) == ["founder"]


def test_sole_founder_demotion_is_409(env):
    client, db, seed, as_ = env
    sole = seed(ORG_A, Role.FOUNDER)
    as_("founder")
    assert client.post(f"/identity/members/{sole}/role", json={"role": "employee"}).status_code == 409
    assert _roles(db, ORG_A) == ["founder"]


# ── H. Machine principal ─────────────────────────────────────────────────────

def test_machine_principal_obeys_the_same_scope_rule(env):
    """An API key carries ONLY its own scopes as permissions (no role), so it is
    bound by the same containment rule. A key scoped to manage_members may reach
    the route, but may still not act on a target holding more than it does."""
    client, db, seed, as_ = env
    victim = seed(ORG_A, Role.CEO)
    machine = Principal(
        kind=PrincipalKind.API_KEY,
        org_id=ORG_A,
        subject_id=uuid.uuid4(),
        permissions=frozenset({"manage_members", "manage_roles"}),
        scopes=["manage_members", "manage_roles"],
    )
    client.app.dependency_overrides[get_principal] = lambda: machine  # type: ignore[attr-defined]
    assert client.delete(f"/identity/members/{victim}").status_code == 403
    assert _roles(db, ORG_A) == ["ceo"]

    # It is ALSO denied an `employee`, and that is correct rather than surprising:
    # `employee` holds view_decisions/view_signals, which this key does not. The
    # containment rule makes no exception for machines — a key must hold a SUPERSET
    # of what its target holds, so a narrowly-scoped key cannot administer members
    # by virtue of being a machine.
    weak = seed(ORG_A, Role.EMPLOYEE)
    assert client.delete(f"/identity/members/{weak}").status_code == 403

    # A key provisioned with the target's permissions as well may proceed. This is
    # the honest cost of the rule: administering a member requires holding what
    # they hold.
    capable = Principal(
        kind=PrincipalKind.API_KEY,
        org_id=ORG_A,
        subject_id=uuid.uuid4(),
        permissions=frozenset({"manage_members", "view_decisions", "view_signals"}),
        scopes=["manage_members", "view_decisions", "view_signals"],
    )
    client.app.dependency_overrides[get_principal] = lambda: capable  # type: ignore[attr-defined]
    assert client.delete(f"/identity/members/{weak}").status_code == 200


# ── Response hygiene ─────────────────────────────────────────────────────────

def test_denials_do_not_leak_internal_authorization_detail(env):
    client, db, seed, as_ = env
    victim = seed(ORG_A, Role.CEO)
    as_("admin")
    body = client.delete(f"/identity/members/{victim}").text.lower()
    for leak in ("manage_policies", "execute_actions", "frozenset", "traceback", "sql"):
        assert leak not in body, f"response leaked {leak!r}"


# ═════════════════════════════════════════════════════════════════════════════
# TASK 6.4.4 — GET /auth/me: the frontend authorization READ contract
# ═════════════════════════════════════════════════════════════════════════════
#
# Three findings converge on this one response:
#
#   F-1  a membership with no usable role crashed the endpoint (500) instead of
#        reporting "no role, no permissions";
#   F-4  the caller's effective permissions were not published at all, forcing
#        every client to keep a second copy of the backend's permission table;
#   F-2  a single organization was reported with no way to tell whether the
#        account had others.
#
# The tests below pin all three, plus the invariants that must survive them:
# permissions are the ACTIVE org's only, machine principals stay rejected, and
# nothing secret rides along.


@pytest.fixture()
def me_env():
    """An app serving `/auth/me` over the fake session.

    Mirrors the `env` fixture above but mounts the identity router and seeds the
    Organization rows the response reads. `as_me` installs the principal AND the
    matching membership row, because /auth/me now reads both: the principal for
    authorization state, the memberships for the organization list.
    """
    db = FakeSession()
    state: dict = {"principal": None}

    app = FastAPI()
    app.include_router(identity_router)
    app.dependency_overrides[get_session] = lambda: db
    app.dependency_overrides[get_principal] = lambda: state["principal"]

    db.add(Organization(id=ORG_A, name="Acme", slug="acme"))
    db.add(Organization(id=ORG_B, name="Other Co", slug="other-co"))

    def as_me(role, *, org_id=ORG_A, also_member_of=(), permissions=None):
        uid = uuid.uuid4()
        user = User(
            id=uid, email="founder@acme.example.com", full_name="A Founder",
            is_active=True, is_verified=True,
            hashed_password="$2b$12$notarealhashbutlongenough",
            google_sub="google-subject-1234567890",
        )
        db.add(user)
        role_value = role.value if isinstance(role, Role) else role
        db.add(Membership(user_id=uid, organization_id=org_id, role=role))
        for other in also_member_of:
            db.add(Membership(user_id=uid, organization_id=other, role=Role.EMPLOYEE))
        state["principal"] = Principal(
            kind=PrincipalKind.USER, org_id=org_id, subject_id=uid,
            permissions=(
                frozenset(permissions) if permissions is not None
                else default_permissions_for_role(role_value)
            ),
            role=role_value, user=user,
        )
        return state["principal"]

    return TestClient(app, raise_server_exceptions=False), db, as_me


# ── F-1: a membership can never crash /auth/me ───────────────────────────────

def test_me_returns_the_role_for_a_valid_founder(me_env):
    client, _db, as_me = me_env
    as_me(Role.FOUNDER)
    r = client.get("/auth/me")
    assert r.status_code == 200
    body = r.json()
    assert body["role"] == "founder"
    assert body["organization"]["id"] == str(ORG_A)


def test_me_returns_the_role_for_a_valid_member(me_env):
    client, _db, as_me = me_env
    as_me(Role.EMPLOYEE)
    r = client.get("/auth/me")
    assert r.status_code == 200
    assert r.json()["role"] == "employee"


def test_me_reports_a_null_role_instead_of_500(me_env):
    """F-1, the reported defect. `memberships.role` was nullable at the database
    level, and the route built `Role("")` from it — ValueError, 500, and a user
    with a perfectly valid seat locked out of establishing a session."""
    client, _db, as_me = me_env
    as_me(None)
    r = client.get("/auth/me")
    assert r.status_code == 200, r.text
    assert r.json()["role"] is None


def test_me_reports_an_unknown_role_as_null_instead_of_500(me_env):
    """The same crash from the other direction: a role string this build does not
    know (enum extended by migration ahead of a deploy, or hand-edited data).
    Reported as null — never guessed, never raised."""
    client, _db, as_me = me_env
    as_me("regional_manager", permissions=frozenset())
    r = client.get("/auth/me")
    assert r.status_code == 200, r.text
    assert r.json()["role"] is None


def test_a_null_role_grants_no_permissions(me_env):
    """Fail-closed is the point. Tolerating the row must not soften it: no role
    means no permissions, not 'default' permissions."""
    client, _db, as_me = me_env
    as_me(None)
    assert client.get("/auth/me").json()["permissions"] == []


async def test_authorization_stays_fail_closed_for_a_null_role():
    """The gate, not just the report. A principal built from a NULL-role
    membership must be refused by `require_permission` on every protected
    route — /auth/me answering 200 changes nothing about that."""
    from fastapi import HTTPException

    from navis.identity.permissions import Permission

    nulled = Principal(
        kind=PrincipalKind.USER, org_id=ORG_A, subject_id=uuid.uuid4(),
        permissions=default_permissions_for_role(None), role=None,
    )
    assert nulled.permissions == frozenset()
    for perm in (Permission.VIEW_DECISIONS, Permission.EXECUTE_ACTIONS,
                 Permission.MANAGE_MEMBERS):
        with pytest.raises(HTTPException) as exc:
            await require_permission(perm)(principal=nulled)
        assert exc.value.status_code == 403


def test_membership_role_is_not_null_in_the_orm():
    """The invariant made explicit (migration 0043). The annotation said
    `Mapped[Role]` while the column allowed NULL for the whole life of the
    table; the write side can no longer produce the row the readers above
    defend against."""
    assert Membership.__table__.c.role.nullable is False
    # No server default, deliberately: an INSERT that forgets the role must fail
    # loudly rather than silently mint an org owner.
    assert Membership.__table__.c.role.server_default is None


# ── F-4: effective permissions, published once ───────────────────────────────

def test_me_publishes_the_principals_effective_permissions(me_env):
    """The frontend must be able to read its authorization state rather than
    recompute it from a duplicated role table (or discover it by provoking
    403s). What it reads is exactly what the gate checks."""
    client, _db, as_me = me_env
    principal = as_me(Role.CEO)
    perms = client.get("/auth/me").json()["permissions"]
    assert perms == sorted(principal.permissions)
    assert "execute_actions" in perms          # ceo holds it
    assert "manage_api_keys" not in perms      # ceo does not


def test_me_permissions_include_per_org_grants_not_just_role_defaults(me_env):
    """The reason a client-side copy of the table cannot be correct: an org's
    `role_permissions` rows exist only in the database. An employee granted
    EXECUTE_ACTIONS by their org reads that permission here; a client deriving
    from the role alone would render the button disabled for a user the backend
    would happily let through."""
    client, _db, as_me = me_env
    as_me(
        Role.EMPLOYEE,
        permissions=default_permissions_for_role("employee") | {"execute_actions"},
    )
    perms = client.get("/auth/me").json()["permissions"]
    assert "execute_actions" in perms
    # The role's own remaining default, so this asserts defaults UNION grants
    # rather than just the grant. `view_decisions` used to stand here; it is a
    # Developer-surface permission and is no longer an `employee` default.
    assert "view_signals" in perms


def test_me_permissions_are_sorted_and_are_plain_strings(me_env):
    """Stable wire shape — clients diff and cache this."""
    client, _db, as_me = me_env
    as_me(Role.ADMIN)
    perms = client.get("/auth/me").json()["permissions"]
    assert perms == sorted(perms)
    assert all(isinstance(p, str) for p in perms)


def test_me_rejects_machine_principals_unchanged(me_env):
    """API keys and service accounts are not human sessions. The existing
    contract (401 from `get_current_context`) is preserved — publishing
    permissions did not open /auth/me to machines."""
    client, _db, as_me = me_env
    as_me(Role.FOUNDER)
    client.app.dependency_overrides[get_principal] = lambda: Principal(  # type: ignore[attr-defined]
        kind=PrincipalKind.API_KEY, org_id=ORG_A, subject_id=uuid.uuid4(),
        permissions=frozenset({"manage_members"}), scopes=["manage_members"],
    )
    assert client.get("/auth/me").status_code == 401


def test_me_leaks_no_credential_material(me_env):
    """An authorization contract must not become a credential contract."""
    client, _db, as_me = me_env
    as_me(Role.FOUNDER)
    body = client.get("/auth/me").text
    for secret in ("hashed_password", "$2b$", "google_sub", "google-subject",
                   "tokens_valid_from", "refresh_token", "access_token"):
        assert secret not in body, f"/auth/me leaked {secret!r}"


# ── F-2: multi-org is REPORTED, never switchable ─────────────────────────────

def test_me_lists_every_organization_the_user_belongs_to(me_env):
    client, _db, as_me = me_env
    as_me(Role.FOUNDER, also_member_of=(ORG_B,))
    orgs = client.get("/auth/me").json()["organizations"]
    assert {o["id"] for o in orgs} == {str(ORG_A), str(ORG_B)}


def test_me_flags_exactly_one_current_organization(me_env):
    """Criterion 4 of the handoff gate: the active tenant is deterministic. It is
    `Principal.org_id`, which comes from the membership row — never from the
    JWT's org claim, and never from anything in the request."""
    client, _db, as_me = me_env
    as_me(Role.FOUNDER, also_member_of=(ORG_B,))
    body = client.get("/auth/me").json()
    current = [o for o in body["organizations"] if o["is_current"]]
    assert len(current) == 1
    assert current[0]["id"] == str(ORG_A) == body["organization"]["id"]


def test_single_org_user_still_gets_a_populated_list(me_env):
    """The distinction the old shape erased: "one organization" now reads
    differently from "we only told you about one"."""
    client, _db, as_me = me_env
    as_me(Role.FOUNDER)
    orgs = client.get("/auth/me").json()["organizations"]
    assert len(orgs) == 1 and orgs[0]["is_current"] is True


def test_other_organizations_carry_no_authorization_state(me_env):
    """Org identity only. Publishing a role for another tenant would hand a
    client everything it needs to derive that tenant's permissions — exactly the
    cross-org inference this contract exists to prevent."""
    client, _db, as_me = me_env
    as_me(Role.FOUNDER, also_member_of=(ORG_B,))
    for org in client.get("/auth/me").json()["organizations"]:
        assert set(org) == {"id", "name", "slug", "is_current"}


def test_no_organization_switch_endpoint_exists():
    """Multi-org switching is Phase 7. This response is a READ; there is no
    mechanism to re-scope a token, and nothing here may grow into one by
    accident."""
    from fastapi.routing import APIRoute

    from navis.api.router import api_router

    paths = {r.path for r in api_router.routes if isinstance(r, APIRoute)}
    for banned in ("/auth/switch-org", "/auth/organizations/{org_id}/switch",
                   "/organization/switch", "/switch-org"):
        assert banned not in paths


# ── Tenant isolation: the resolved principal, end to end ─────────────────────

async def test_org_a_permissions_never_include_an_org_b_grant():
    """The real resolution path, not an overridden principal.

    `grants_for` filters `role_permissions` on `(organization_id IN (this_org,
    NULL))`. A grant Org B made to the same ROLE STRING must not reach a
    principal authenticated into Org A — two orgs both having an `employee` is
    the normal case, not an exotic one.
    """
    from navis.identity.authz import _principal_from_jwt, principal_cache_clear
    from navis.identity.enterprise_models import RolePermission
    from navis.platform.security import create_access_token

    principal_cache_clear()
    db = FakeSession()
    uid = uuid.uuid4()
    db.add(User(id=uid, email="shared@x.com", is_active=True))
    db.add(Membership(user_id=uid, organization_id=ORG_A, role=Role.EMPLOYEE))
    # Org B grants its employees EXECUTE_ACTIONS. Org A grants nothing.
    db.add(RolePermission(organization_id=ORG_B, role="employee", permission="execute_actions"))

    principal = await _principal_from_jwt(  # type: ignore[arg-type]
        db, create_access_token(str(uid), org_id=str(ORG_A))
    )

    assert principal.org_id == ORG_A
    assert "execute_actions" not in principal.permissions, "Org B's grant leaked into Org A"
    assert principal.permissions == default_permissions_for_role("employee")


async def test_a_grant_in_the_callers_own_org_does_reach_the_principal():
    """The other half of the previous test — proving the isolation above is a
    tenant filter and not simply a query that never returns anything."""
    from navis.identity.authz import _principal_from_jwt, principal_cache_clear
    from navis.identity.enterprise_models import RolePermission
    from navis.platform.security import create_access_token

    principal_cache_clear()
    db = FakeSession()
    uid = uuid.uuid4()
    db.add(User(id=uid, email="local@x.com", is_active=True))
    db.add(Membership(user_id=uid, organization_id=ORG_A, role=Role.EMPLOYEE))
    db.add(RolePermission(organization_id=ORG_A, role="employee", permission="execute_actions"))

    principal = await _principal_from_jwt(  # type: ignore[arg-type]
        db, create_access_token(str(uid), org_id=str(ORG_A))
    )
    assert "execute_actions" in principal.permissions


async def test_a_jwt_org_claim_without_a_membership_is_refused():
    """The security invariant in one test: the JWT's `org` claim is a LOOKUP KEY,
    never tenant authority. A token minted naming Org B for a user who is only a
    member of Org A resolves no membership and is refused — it does not fall
    back to Org A, and it does not grant Org B."""
    from fastapi import HTTPException

    from navis.identity.authz import _principal_from_jwt, principal_cache_clear
    from navis.platform.security import create_access_token

    principal_cache_clear()
    db = FakeSession()
    uid = uuid.uuid4()
    db.add(User(id=uid, email="a-only@x.com", is_active=True))
    db.add(Membership(user_id=uid, organization_id=ORG_A, role=Role.FOUNDER))

    with pytest.raises(HTTPException) as exc:
        await _principal_from_jwt(  # type: ignore[arg-type]
            db, create_access_token(str(uid), org_id=str(ORG_B))
        )
    assert exc.value.status_code == 403


async def test_principal_org_id_comes_from_the_membership_row():
    """`Principal.org_id` is `membership.organization_id`, read from the
    database — the claim is only ever used to find that row."""
    from navis.identity.authz import _principal_from_jwt, principal_cache_clear
    from navis.platform.security import create_access_token

    principal_cache_clear()
    db = FakeSession()
    uid = uuid.uuid4()
    db.add(User(id=uid, email="m@x.com", is_active=True))
    membership = Membership(user_id=uid, organization_id=ORG_A, role=Role.FOUNDER)
    db.add(membership)

    principal = await _principal_from_jwt(  # type: ignore[arg-type]
        db, create_access_token(str(uid), org_id=str(ORG_A))
    )
    assert principal.org_id is membership.organization_id


def test_me_resolves_the_principal_exactly_once(me_env):
    """/auth/me asks for both `CurrentContext` and `Principal`. FastAPI caches a
    dependency per request, so that is ONE resolution — not a second pass
    through JWT decode, revocation lookup, membership query and grant query on
    an endpoint every page load hits."""
    client, _db, as_me = me_env
    principal = as_me(Role.FOUNDER)
    calls = {"n": 0}

    def counting_principal():
        calls["n"] += 1
        return principal

    client.app.dependency_overrides[get_principal] = counting_principal  # type: ignore[attr-defined]
    assert client.get("/auth/me").status_code == 200
    assert calls["n"] == 1, f"principal resolved {calls['n']}x — dependency cache lost"


# ── P0-8: switching to another seat the caller already holds ─────────────────
#
# `/auth/me` has always LISTED every organization the caller belongs to, while
# both login paths resolved `primary_membership` — the oldest. Every other seat
# was therefore visible and unreachable, and the only action a person had left
# was to sign up again, which is how a duplicate organization gets created.
# `POST /auth/switch` closes that, and these pin that it closes it without
# becoming a way to name a tenant you do not belong to.


def test_switch_issues_a_session_for_another_seat_the_user_holds(me_env):
    client, _db, as_me = me_env
    as_me(Role.FOUNDER, org_id=ORG_A, also_member_of=(ORG_B,))

    r = client.post("/auth/switch", json={"organization_id": str(ORG_B)})

    assert r.status_code == 200
    assert r.json()["access_token"]


def test_switch_refuses_an_organization_the_user_is_not_a_member_of(me_env):
    """The request names an organization; the MEMBERSHIP ROW decides.

    Refused with the same wording and status `_principal_from_jwt` uses for a
    forged `org` claim, so a non-member cannot distinguish "not yours" from
    "does not exist" and the route is not an organization oracle.
    """
    client, _db, as_me = me_env
    as_me(Role.FOUNDER, org_id=ORG_A)  # ORG_B exists, but this user has no seat

    r = client.post("/auth/switch", json={"organization_id": str(ORG_B)})

    assert r.status_code == 403
    assert client.post(
        "/auth/switch", json={"organization_id": str(uuid.uuid4())}
    ).status_code == 403


def test_switch_creates_no_organization_and_no_membership(me_env):
    """It is a re-scope, not a provisioning path.

    The failure this guards against is the tempting "switch to an org you are
    not in by creating it", which would turn the fix for duplicate
    organizations into a new source of them.
    """
    client, db, as_me = me_env
    as_me(Role.FOUNDER, org_id=ORG_A, also_member_of=(ORG_B,))
    orgs_before = len(db.store.get(Organization, []))
    memberships_before = len(db.store.get(Membership, []))

    client.post("/auth/switch", json={"organization_id": str(ORG_B)})
    client.post("/auth/switch", json={"organization_id": str(uuid.uuid4())})

    assert len(db.store.get(Organization, [])) == orgs_before
    assert len(db.store.get(Membership, [])) == memberships_before


# ── P1-8: the Developer team routes admit a Developer administrator ─────────
#
# Three routes administer Developer team membership and all three were gated
# on an Application permission, so a Developer owner or admin who is an
# Application `employee` was refused by the ROUTE before any service-layer
# rule could apply. These assert the gate, not the grant: whether the
# assignment itself is allowed is `_ensure_grantable`'s answer and is covered
# in `test_developer_access_model.py`.


def _developer_admin(org_id=ORG_A, subject_id=None):
    """An Application `employee` whose Developer role is `admin` — the tech
    lead at a Developer-first company, and the exact principal that could not
    reach these routes before P1-8."""
    from navis.identity.permissions import effective_permissions

    uid = subject_id or uuid.uuid4()
    return Principal(
        kind=PrincipalKind.USER,
        org_id=org_id,
        subject_id=uid,
        permissions=effective_permissions("employee", developer_role="admin"),
        role="employee",
        user=User(id=uid, email="lead@x.com"),
    )


def test_a_developer_admin_may_list_the_team(env):
    client, _db, seed, _as = env
    seed(ORG_A, Role.EMPLOYEE)
    client.app.dependency_overrides[get_principal] = _developer_admin  # type: ignore[attr-defined]

    assert client.get("/identity/members").status_code == 200


def test_a_developer_admin_may_invite(env):
    client, _db, _seed, _as = env
    client.app.dependency_overrides[get_principal] = _developer_admin  # type: ignore[attr-defined]

    response = client.post(
        "/identity/members/invite",
        json={"email": "new@x.com", "role": "employee", "developer_role": "developer"},
    )
    assert response.status_code == 201, response.text


def test_a_developer_admin_may_reach_the_access_route(env):
    client, _db, seed, _as = env
    target = seed(ORG_A, Role.EMPLOYEE)
    client.app.dependency_overrides[get_principal] = _developer_admin  # type: ignore[attr-defined]

    response = client.post(
        f"/identity/members/{target}/access", json={"developer_role": "developer"}
    )
    assert response.status_code == 200, response.text


def test_a_developer_admin_may_not_remove_a_member(env):
    """DELIBERATELY NOT EXTENDED. Removing a member evicts them from the
    ORGANIZATION — dropping their Application access and bulk-revoking their
    tokens — which is an organizational act, not a Developer one. The
    Developer-scoped equivalent is revoking Developer access, which is the
    route above with `developer_role: null`."""
    client, _db, seed, _as = env
    target = seed(ORG_A, Role.EMPLOYEE)
    client.app.dependency_overrides[get_principal] = _developer_admin  # type: ignore[attr-defined]

    assert client.delete(f"/identity/members/{target}").status_code == 403


def test_an_ordinary_developer_may_not_administer_the_team(env):
    """`developer` is the self-serve building role, not an administrator."""
    from navis.identity.permissions import effective_permissions

    client, _db, seed, _as = env
    target = seed(ORG_A, Role.EMPLOYEE)
    uid = uuid.uuid4()
    client.app.dependency_overrides[get_principal] = lambda: Principal(  # type: ignore[attr-defined]
        kind=PrincipalKind.USER, org_id=ORG_A, subject_id=uid,
        permissions=effective_permissions("employee", developer_role="developer"),
        role="employee", user=User(id=uid, email="dev@x.com"),
    )

    assert client.get("/identity/members").status_code == 403
    assert client.post(
        "/identity/members/invite", json={"email": "x@y.com", "role": "employee"}
    ).status_code == 403
    assert client.post(
        f"/identity/members/{target}/access", json={"developer_role": "viewer"}
    ).status_code == 403


def test_an_application_admin_still_reaches_these_routes(env):
    """The disjunction is ADDITIVE — every caller that passed before still
    passes on the Application permission alone."""
    client, _db, seed, as_ = env
    seed(ORG_A, Role.EMPLOYEE)
    as_("admin")

    assert client.get("/identity/members").status_code == 200
    assert client.post(
        "/identity/members/invite", json={"email": "app@x.com", "role": "employee"}
    ).status_code == 201
