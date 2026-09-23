"""Identity V2: RBAC enforcement, token revocation, API keys, service accounts,
audit, member management, multi-tenancy, migration bootstrap, and the import guard.
"""
from __future__ import annotations

import ast
import pathlib
import uuid

import pytest

from navis.identity import permissions as P
from navis.identity.apikeys import ApiKeyService, _hash
from navis.identity.audit import AuditAction, AuditService
from navis.identity.authz import Principal, require_permission
from navis.identity.members import MemberError, MemberService
from navis.identity.models import Membership, OrganizationKind, Role, User
from navis.identity.permissions import Permission
from navis.identity.service_accounts import ServiceAccountService
from navis.identity.tokens import TokenRevocationService, token_predates_bulk_revoke
from tests.identity.fakes import ACTOR, ORG_A, ORG_B, FakeSession

IDENTITY = pathlib.Path(__file__).resolve().parent.parent.parent / "navis" / "identity"
MIGRATIONS = pathlib.Path(__file__).resolve().parent.parent.parent / "migrations" / "versions"
FORBIDDEN = {"signals", "memory", "state", "reasoning", "decisions",
             "execution", "observation", "learning", "intelligence", "adaptive"}


# ---- RBAC: permission resolution + enforcement ----

def test_default_role_permissions_are_data():
    assert P.Permission.EXECUTE_ACTIONS in P.default_permissions_for_role("founder")
    # `employee` is the minimal APPLICATION role. Since P1-6 it no longer carries
    # `view_decisions` — a Developer-surface permission that gated
    # `/developer/decisions`, so holding it gave every employee a Developer read
    # nobody had granted them.
    assert P.default_permissions_for_role("employee") == frozenset(
        {Permission.VIEW_SIGNALS}
    )
    assert Permission.EXECUTE_ACTIONS not in P.default_permissions_for_role("employee")


def test_effective_permissions_merges_grants():
    eff = P.effective_permissions("employee", [Permission.EXECUTE_ACTIONS])
    assert Permission.EXECUTE_ACTIONS in eff  # per-org grant augments defaults


async def test_require_permission_allows_and_denies():
    from fastapi import HTTPException

    founder = Principal("user", ORG_A, ACTOR, P.default_permissions_for_role("founder"), role="founder")
    employee = Principal("user", ORG_A, ACTOR, P.default_permissions_for_role("employee"), role="employee")

    dep = require_permission(Permission.EXECUTE_ACTIONS)
    assert await dep(principal=founder) is founder
    with pytest.raises(HTTPException) as exc:
        await dep(principal=employee)
    assert exc.value.status_code == 403


# ---- token revocation ----

async def test_jti_denylist_blocks_token():
    db = FakeSession()
    svc = TokenRevocationService(db)
    assert await svc.is_jti_revoked("j1") is False
    await svc.revoke_jti("j1", user_id=ACTOR, expires_at=None)
    assert await svc.is_jti_revoked("j1") is True


async def test_bulk_revoke_watermark_rejects_old_tokens():
    from datetime import datetime, timezone

    db = FakeSession()
    user = User(email="f@x.com", is_active=True)
    db.add(user)
    await TokenRevocationService(db).revoke_all_for_user(user.id)
    # A token issued before the watermark is rejected; one issued after is fine.
    old = {"iat": 1}
    future = {"iat": int(datetime.now(timezone.utc).timestamp()) + 10_000}
    assert token_predates_bulk_revoke(old, user) is True
    assert token_predates_bulk_revoke(future, user) is False


# ---- principal cache (hot-path DB skip on the JWT auth pipeline) ----

async def test_principal_cache_hit_skips_downstream_db_queries():
    """A pre-populated cache MUST make _principal_from_jwt return without touching
    the DB. Regression guard for the /context auth-hot-path optimization."""
    from navis.identity.authz import (
        Principal,
        PrincipalKind,
        _principal_cache_put,
        _principal_from_jwt,
        principal_cache_clear,
    )
    from navis.platform.security import create_access_token

    principal_cache_clear()
    user_id = uuid.uuid4()
    org_id = ORG_A
    cached = Principal(
        kind=PrincipalKind.USER, org_id=org_id, subject_id=user_id,
        permissions=frozenset(), role="founder", user=None,
    )
    token = create_access_token(str(user_id), org_id=str(org_id))
    # Extract the jti the exact same way _principal_from_jwt does.
    from navis.platform.security import decode_token

    payload = decode_token(token, expected_type="access")
    _principal_cache_put(payload["jti"], cached)

    class _RaisingSession:
        async def get(self, *a, **kw):     raise AssertionError("db.get called on cache hit")
        async def scalar(self, *a, **kw):  raise AssertionError("db.scalar called on cache hit")
        async def scalars(self, *a, **kw): raise AssertionError("db.scalars called on cache hit")

    got = await _principal_from_jwt(_RaisingSession(), token)  # type: ignore[arg-type]
    assert got is cached


async def test_principal_cache_miss_populates_then_hits():
    """First call runs the DB pipeline and caches; second call is a hit."""
    from navis.identity.authz import (
        _PRINCIPAL_CACHE,
        _principal_from_jwt,
        principal_cache_clear,
    )
    from navis.platform.security import create_access_token, decode_token

    principal_cache_clear()
    db = FakeSession()
    user = User(email="cache@x.com", is_active=True)
    db.add(user)
    db.add(Membership(user_id=user.id, organization_id=ORG_A, role=Role.FOUNDER))

    token = create_access_token(str(user.id), org_id=str(ORG_A))
    p1 = await _principal_from_jwt(db, token)
    assert p1.subject_id == user.id
    payload = decode_token(token, expected_type="access")
    assert payload["jti"] in _PRINCIPAL_CACHE

    # Second call with a session that raises on any DB access — must serve the cache.
    class _RaisingSession:
        async def get(self, *a, **kw):     raise AssertionError("cached path touched DB")
        async def scalar(self, *a, **kw):  raise AssertionError("cached path touched DB")
        async def scalars(self, *a, **kw): raise AssertionError("cached path touched DB")

    p2 = await _principal_from_jwt(_RaisingSession(), token)  # type: ignore[arg-type]
    assert p2.subject_id == user.id


async def test_revoke_jti_evicts_the_principal_cache_synchronously():
    """Explicit logout MUST propagate immediately — the cache must not serve
    a revoked jti even within the TTL."""
    from navis.identity.authz import (
        _PRINCIPAL_CACHE,
        Principal,
        PrincipalKind,
        _principal_cache_put,
        principal_cache_clear,
    )

    principal_cache_clear()
    jti = "jti-to-revoke"
    _principal_cache_put(jti, Principal(
        PrincipalKind.USER, ORG_A, ACTOR, frozenset(), role="founder", user=None,
    ))
    assert jti in _PRINCIPAL_CACHE

    db = FakeSession()
    await TokenRevocationService(db).revoke_jti(jti, user_id=ACTOR, expires_at=None)
    assert jti not in _PRINCIPAL_CACHE   # evicted synchronously by revoke_jti


async def test_revoke_all_for_user_evicts_every_cached_session_for_that_user():
    """Bulk revoke (deactivation / 'log out everywhere') MUST invalidate every
    outstanding session, not only the token that happened to present a jti."""
    from navis.identity.authz import (
        _PRINCIPAL_CACHE,
        Principal,
        PrincipalKind,
        _principal_cache_put,
        principal_cache_clear,
    )

    principal_cache_clear()
    victim_user = uuid.uuid4()
    other_user = uuid.uuid4()
    # Three sessions for the victim, one for someone else.
    for jti in ("v1", "v2", "v3"):
        _principal_cache_put(jti, Principal(
            PrincipalKind.USER, ORG_A, victim_user, frozenset(), role="founder", user=None,
        ))
    _principal_cache_put("other", Principal(
        PrincipalKind.USER, ORG_A, other_user, frozenset(), role="founder", user=None,
    ))
    assert set(_PRINCIPAL_CACHE) == {"v1", "v2", "v3", "other"}

    db = FakeSession()
    db.add(User(id=victim_user, email="v@x.com", is_active=True))
    await TokenRevocationService(db).revoke_all_for_user(victim_user)
    assert set(_PRINCIPAL_CACHE) == {"other"}    # victim's sessions gone; unrelated user survives


async def test_principal_cache_respects_ttl():
    """A stale entry (past TTL) must not be served — the request pays the DB cost
    and re-populates."""
    import time as _time

    from navis.identity.authz import (
        _PRINCIPAL_CACHE,
        Principal,
        PrincipalKind,
        _principal_cache_get,
        principal_cache_clear,
    )

    principal_cache_clear()
    jti = "ttl-jti"
    # Manually inject an already-expired entry (expires_at in the past).
    _PRINCIPAL_CACHE[jti] = (
        _time.monotonic() - 1.0,
        ACTOR,
        Principal(PrincipalKind.USER, ORG_A, ACTOR, frozenset(), role="founder", user=None),
    )
    assert _principal_cache_get(jti) is None       # expired -> miss
    assert jti not in _PRINCIPAL_CACHE             # lazily evicted on get


async def test_principal_cache_is_bounded_lru():
    """Cache size never exceeds _PRINCIPAL_CACHE_MAX — the oldest entries evict."""
    from navis.identity.authz import (
        _PRINCIPAL_CACHE,
        _PRINCIPAL_CACHE_MAX,
        Principal,
        PrincipalKind,
        _principal_cache_put,
        principal_cache_clear,
    )

    principal_cache_clear()
    for i in range(_PRINCIPAL_CACHE_MAX + 8):
        _principal_cache_put(f"jti-{i}", Principal(
            PrincipalKind.USER, ORG_A, ACTOR, frozenset(), role="founder", user=None,
        ))
    assert len(_PRINCIPAL_CACHE) == _PRINCIPAL_CACHE_MAX
    # The oldest 8 evicted; the newest _MAX survive.
    assert "jti-0" not in _PRINCIPAL_CACHE
    assert f"jti-{_PRINCIPAL_CACHE_MAX + 7}" in _PRINCIPAL_CACHE


# ---- API key lifecycle ----

async def test_api_key_create_verify_revoke():
    db = FakeSession()
    svc = ApiKeyService(db)
    key, plaintext = await svc.create(org_id=ORG_A, name="ci", scopes=[Permission.VIEW_STATE])
    assert plaintext.startswith("navis_") and key.hashed_secret == _hash(plaintext)
    assert key.hashed_secret != plaintext  # never stored in clear

    verified = await svc.verify(plaintext)
    assert verified is not None and verified.id == key.id
    assert await svc.verify("navis_deadbeef_nope") is None

    await svc.revoke(key.id, org_id=ORG_A)
    assert await svc.verify(plaintext) is None  # revoked keys do not authenticate


# ---- service account lifecycle (rotation revokes prior credential) ----

async def test_service_account_rotation_revokes_old_key():
    db = FakeSession()
    sa_svc = ServiceAccountService(db)
    sa, secret1 = await sa_svc.create(org_id=ORG_A, name="worker", scopes=[Permission.EXECUTE_ACTIONS])
    keys = ApiKeyService(db)
    assert await keys.verify(secret1) is not None

    secret2 = await sa_svc.rotate(sa.id, org_id=ORG_A)
    assert await keys.verify(secret1) is None      # old credential dead
    assert await keys.verify(secret2) is not None  # new credential lives

    await sa_svc.revoke(sa.id, org_id=ORG_A)
    assert await keys.verify(secret2) is None       # revoke kills all credentials


# ---- tenant isolation: cross-org mutation must be impossible (P0 regression guards) ----

async def test_cross_org_api_key_revoke_is_blocked():
    db = FakeSession()
    svc = ApiKeyService(db)
    key, plaintext = await svc.create(org_id=ORG_A, name="a", scopes=[])
    # ORG_B tries to revoke ORG_A's key -> treated as not-found, key stays valid.
    assert await svc.revoke(key.id, org_id=ORG_B) is None
    assert await svc.verify(plaintext) is not None  # untouched
    # Owning org can revoke.
    assert await svc.revoke(key.id, org_id=ORG_A) is not None
    assert await svc.verify(plaintext) is None


async def test_cross_org_service_account_rotate_revoke_is_blocked():
    import pytest

    db = FakeSession()
    sa_svc = ServiceAccountService(db)
    sa, secret = await sa_svc.create(org_id=ORG_A, name="w", scopes=[])
    with pytest.raises(LookupError):
        await sa_svc.rotate(sa.id, org_id=ORG_B)   # cannot mint a cross-tenant credential
    with pytest.raises(LookupError):
        await sa_svc.revoke(sa.id, org_id=ORG_B)   # cannot revoke another org's SA
    assert await ApiKeyService(db).verify(secret) is not None  # ORG_A's creds intact


# ---- audit logging ----

async def test_audit_records_and_lists():
    db = FakeSession()
    audit = AuditService(db)
    await audit.record(AuditAction.LOGIN, org_id=ORG_A, actor_id=ACTOR)
    await audit.record(AuditAction.API_KEY_CREATE, org_id=ORG_A, actor_id=ACTOR, target_id="k1")
    rows = await audit.list_for_org(ORG_A)
    assert {r.action for r in rows} == {AuditAction.LOGIN, AuditAction.API_KEY_CREATE}


# ---- member management + token revoke on removal ----

async def test_member_invite_assign_remove():
    from navis.identity.permissions import ALL_PERMISSIONS

    db = FakeSession()
    svc = MemberService(db)
    founder = ALL_PERMISSIONS  # the actor is a founder, so may grant any role
    m = await svc.invite(org_id=ORG_A, email="new@x.com", role=Role.EMPLOYEE, actor_id=ACTOR, actor_permissions=founder)
    assert m.role == Role.EMPLOYEE
    with pytest.raises(MemberError):
        await svc.invite(org_id=ORG_A, email="new@x.com", role=Role.EMPLOYEE, actor_id=ACTOR, actor_permissions=founder)

    await svc.assign_role(org_id=ORG_A, user_id=m.user_id, role=Role.ADMIN, actor_id=ACTOR, actor_permissions=founder)
    member = await db.get(Membership, m.id)
    assert member.role == Role.ADMIN

    await svc.remove(org_id=ORG_A, user_id=m.user_id, actor_id=ACTOR, actor_permissions=founder)
    assert await svc.list_members(ORG_A) == []
    # Removal bulk-revokes the user's tokens.
    user = await db.get(User, m.user_id)
    assert user.tokens_valid_from is not None
    # And every mutation was audited.
    actions = {a.action for a in await AuditService(db).list_for_org(ORG_A)}
    assert {AuditAction.MEMBER_INVITE, AuditAction.ROLE_CHANGE, AuditAction.MEMBER_REMOVE} <= actions


# ---- P0-4: the personal workspace -------------------------------------------
#
# A solo developer had to invent a company to reach the Developer surface, and
# the tenant that came out was indistinguishable from a real one. These pin the
# smallest model that fixes it: `organizations.kind` carries PRODUCT meaning,
# and nothing else changes. Authorization still reads `memberships.role` alone.


def test_signup_can_omit_an_organization_name():
    """Omitting it is the explicit signal for "I have no company"."""
    from navis.identity.schemas import SignupRequest

    named = SignupRequest(
        email="dev@acme.com", password="Str0ngPass!x", organization_name="Acme"
    )
    assert named.organization_name == "Acme"

    solo = SignupRequest(email="dev@acme.com", password="Str0ngPass!x")
    assert solo.organization_name is None, (
        "a solo developer must be able to sign up without naming a company"
    )


def test_a_personal_workspace_is_named_after_the_person_not_their_address():
    """The label is rendered in the console and in `/developer/me`; an email
    address is more than a display name needs to carry."""
    from navis.identity.service import _personal_workspace_name

    assert _personal_workspace_name("Dana Reed", "dana@acme.com") == "Dana Reed"
    assert _personal_workspace_name(None, "dana@acme.com") == "dana"
    assert _personal_workspace_name("   ", "dana@acme.com") == "dana"
    # Total, never empty: `OrgLifecycleSeam.create` rejects a blank name and a
    # signup must not fail for want of a display string.
    assert _personal_workspace_name(None, "@acme.com") == "Personal workspace"


def test_org_creation_still_defaults_to_a_company():
    """Every existing call site produces exactly the row it produced before."""
    import inspect

    from navis.identity.models import OrganizationKind
    from navis.identity.org_service import OrgLifecycleSeam

    default = inspect.signature(OrgLifecycleSeam.create).parameters["kind"].default
    assert default is OrganizationKind.COMPANY


async def test_a_personal_workspace_refuses_a_second_member():
    """"One person's own space" with three members is a false statement about
    the tenant, and the classification is read by the console and the
    Developer surface."""
    from navis.identity.models import Organization, OrganizationKind
    from navis.identity.permissions import ALL_PERMISSIONS

    db = FakeSession()
    workspace = Organization(name="Dana Reed", slug="dana-reed", kind=OrganizationKind.PERSONAL.value)
    workspace.id = ORG_A
    db.add(workspace)

    with pytest.raises(MemberError, match="personal workspace"):
        await MemberService(db).invite(
            org_id=ORG_A, email="colleague@acme.com", role=Role.EMPLOYEE,
            actor_id=ACTOR, actor_permissions=ALL_PERMISSIONS,
        )


async def test_a_company_still_accepts_members():
    """The guard is about the workspace KIND, not a new restriction on invites."""
    from navis.identity.models import Organization, OrganizationKind
    from navis.identity.permissions import ALL_PERMISSIONS

    db = FakeSession()
    company = Organization(name="Acme", slug="acme", kind=OrganizationKind.COMPANY.value)
    company.id = ORG_A
    db.add(company)

    membership = await MemberService(db).invite(
        org_id=ORG_A, email="colleague@acme.com", role=Role.EMPLOYEE,
        actor_id=ACTOR, actor_permissions=ALL_PERMISSIONS,
    )
    assert membership.role == Role.EMPLOYEE


def test_a_workspace_owner_can_administer_their_own_workspace():
    """WHY THE CREATOR'S APPLICATION ROLE IS NOT COSMETIC (P0-9B).

    P0-9B asked whether a workspace creator should really be an Application
    `founder`, since the console was rendering "Application role: Founder" on
    a one-person workspace and making it look like a company. The console was
    fixed to say "Workspace ownership: Owner" there — a LABEL — and this test
    is the reason the stored role was left alone.

    These three permissions are the ones a workspace owner needs to administer
    their OWN workspace, and all three live on the Application role:

        MANAGE_BILLING       PUT /organization/plan — without it a solo
                             developer can never upgrade their own plan.
        MANAGE_MEMBERS       PUT /organization/profile.
        MANAGE_API_KEYS      the /identity/api-keys lifecycle.

    None of them is reachable from any Developer role, so "give the creator
    Developer owner and no Application role" would produce a tenant whose
    owner cannot pay for it. Worse, `MemberService._ensure_grantable` requires
    a granter to already hold what they grant, so nobody could ever restore
    them — the same unrecoverable state the last-founder and last-Developer-
    owner guards exist to prevent.

    Kept as a property of the ROLE, not of the workspace kind: authorization
    must not read `kind` at all, which the test below pins.
    """
    from navis.identity.permissions import (
        DEVELOPER_ROLE_PERMISSIONS,
        Permission,
        default_permissions_for_role,
    )

    founder = default_permissions_for_role(Role.FOUNDER.value)
    self_administration = {
        Permission.MANAGE_BILLING,
        Permission.MANAGE_MEMBERS,
        Permission.MANAGE_API_KEYS,
    }
    assert self_administration <= founder

    # And no Developer role supplies any of them, which is what makes the
    # Application role load-bearing rather than redundant.
    for developer_role, granted in DEVELOPER_ROLE_PERMISSIONS.items():
        assert not (self_administration & granted), (
            f"Developer role '{developer_role}' now grants workspace "
            "self-administration — the Application role may no longer be the "
            "only thing standing between an owner and their own billing"
        )


def test_workspace_kind_is_not_an_authorization_input():
    """THE INVARIANT THIS WHOLE CHANGE RESTS ON.

    `kind` classifies what a tenant IS. If it ever reaches
    `effective_permissions` or a gate, Navis has a second authorization input
    and the two are free to disagree — which is the failure the surface-RBAC
    migration deliberately avoided by shipping storage only.
    """
    import inspect

    from navis.identity import authz, permissions

    for module in (permissions, authz):
        source = inspect.getsource(module)
        assert "OrganizationKind" not in source, (
            f"{module.__name__} reads the workspace kind — authorization must "
            "depend on memberships.role alone"
        )
        # A TENANT's kind specifically. `Permission.kind` is an unrelated field
        # on the permission vocabulary itself, so a bare `.kind` match would
        # read that as a violation and this test would be noise.
        for attribute in ("organization.kind", "org.kind", "Organization.kind"):
            assert attribute not in source, (
                f"{module.__name__} branches on {attribute} — a tenant's "
                "classification must not reach an authorization decision"
            )


# ---- P0-5: workspace ownership at the Google edge ---------------------------
#
# P0-4 made the PASSWORD path able to create a personal workspace. Google could
# not: it created a company named after the person for everybody, so a solo
# developer arriving through the only button the console shows got a tenant
# that claimed to be a company. These pin the fix and, more importantly, its
# blast radius — intent is honoured when STATED and inferred never.


def test_google_signup_can_state_a_workspace_kind():
    from navis.identity.schemas import GoogleAuthRequest

    personal = GoogleAuthRequest(code="x", workspace_kind=OrganizationKind.PERSONAL)
    assert personal.workspace_kind is OrganizationKind.PERSONAL

    company = GoogleAuthRequest(
        code="x", workspace_kind=OrganizationKind.COMPANY, organization_name="Dignatara Space"
    )
    assert company.organization_name == "Dignatara Space"


def test_google_signup_without_a_stated_kind_is_unchanged():
    """THE COMPATIBILITY PROPERTY. Every caller that existed before this field
    sends neither, and must keep creating exactly the company it created."""
    from navis.identity.schemas import GoogleAuthRequest

    request = GoogleAuthRequest(code="x")
    assert request.workspace_kind is None
    assert request.organization_name is None


def test_a_personal_google_workspace_refuses_a_company_name():
    """A caller asking for a personal workspace AND naming a company believes
    it is creating something other than what it asked for. Refused rather than
    silently dropped — dropping it is how somebody discovers a week later that
    their workspace is called their own name."""
    import pydantic

    from navis.identity.schemas import GoogleAuthRequest

    with pytest.raises(pydantic.ValidationError, match="named after its owner"):
        GoogleAuthRequest(
            code="x",
            workspace_kind=OrganizationKind.PERSONAL,
            organization_name="Dignatara Space",
        )


def test_the_session_contract_publishes_the_workspace_kind():
    """Published so a surface can stop asking for what cannot apply.

    `OnboardingStage` is an Application ladder — profile, Gmail, calendar,
    first brief. A personal workspace never climbs it, so a console that
    cannot tell the kinds apart nags a developer forever about onboarding they
    have nothing to finish. That is the fake Application activation step this
    field removes.
    """
    from navis.identity.schemas import OrganizationOut

    assert "kind" in OrganizationOut.model_fields
    rendered = OrganizationOut(
        id=ORG_A, name="Dana Reed", slug="dana-reed", kind=OrganizationKind.PERSONAL
    )
    assert rendered.kind is OrganizationKind.PERSONAL


async def test_refresh_is_bound_to_its_user():
    """Service-layer defense-in-depth: a refresh token can only be rotated by the
    user it was issued to, even if the router passed the wrong user."""
    from navis.identity.models import User
    from navis.identity.service import AuthError, IdentityService

    db = FakeSession()
    svc = IdentityService(db)
    alice = User(email="alice@x.com", is_active=True)
    bob = User(email="bob@x.com", is_active=True)
    db.add(alice)
    db.add(bob)
    _, refresh = await svc.issue_tokens(alice, ORG_A)
    with pytest.raises(AuthError):
        await svc.rotate_refresh(refresh, bob, ORG_A)   # Bob can't use Alice's token
    a2, r2 = await svc.rotate_refresh(refresh, alice, ORG_A)  # Alice can
    assert a2 and r2


async def test_refresh_rejects_expired_session():
    from datetime import datetime, timedelta, timezone

    from navis.identity.models import Session, User
    from navis.identity.service import AuthError, IdentityService
    from navis.platform.security import decode_token

    db = FakeSession()
    svc = IdentityService(db)
    user = User(email="u@x.com", is_active=True)
    db.add(user)
    _, refresh = await svc.issue_tokens(user, ORG_A)
    # Force the session past expiry (service must enforce, not just trust JWT exp).
    sid = decode_token(refresh, expected_type="refresh")["sid"]
    import uuid as _uuid
    (await db.get(Session, _uuid.UUID(sid))).expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
    with pytest.raises(AuthError):
        await svc.rotate_refresh(refresh, user, ORG_A)


async def test_refresh_reuse_detection_kills_family():
    from navis.identity.audit import AuditAction
    from navis.identity.enterprise_models import AuditLog
    from navis.identity.models import User
    from navis.identity.service import AuthError, IdentityService

    db = FakeSession()
    svc = IdentityService(db)
    user = User(email="u@x.com", is_active=True)
    db.add(user)
    _, refresh1 = await svc.issue_tokens(user, ORG_A)
    await svc.rotate_refresh(refresh1, user, ORG_A)  # refresh1 now spent
    # Replay the already-used refresh1 -> reuse detected.
    with pytest.raises(AuthError):
        await svc.rotate_refresh(refresh1, user, ORG_A)
    # The whole family is killed (bulk-revoke watermark) + a security event is logged.
    assert user.tokens_valid_from is not None
    reuse_events = [
        a for a in db.store.get(AuditLog, []) if a.action == AuditAction.REFRESH_REUSE_DETECTED
    ]
    assert len(reuse_events) == 1 and reuse_events[0].actor_id == user.id


async def test_logout_revokes_session_so_refresh_is_blocked():
    """The exploitable P1: logout must end the session, not just the access token —
    otherwise a logged-out client refreshes into a fresh valid token."""
    import uuid as _uuid

    from navis.identity.models import Session
    from navis.identity.service import AuthError, IdentityService
    from navis.platform.security import decode_token

    db = FakeSession()
    svc = IdentityService(db)
    user = User(email="u@x.com", is_active=True)
    db.add(user)

    access, refresh = await svc.issue_tokens(user, ORG_A)
    payload = decode_token(access, expected_type="access")
    sid = payload["sid"]
    assert sid  # token is bound to its session

    # Refresh works BEFORE logout.
    a2, r2 = await svc.rotate_refresh(refresh, user, ORG_A)
    assert a2 and r2

    # Log out the (new) session, then its refresh token must be dead.
    p2 = decode_token(a2, expected_type="access")
    await svc.logout(jti=p2.get("jti"), user_id=user.id, expires_at=None, sid=p2["sid"])
    revoked = await db.get(Session, _uuid.UUID(p2["sid"]))
    assert revoked.revoked is True
    with pytest.raises(AuthError):
        await svc.rotate_refresh(r2, user, ORG_A)  # refresh-after-logout blocked


async def test_invite_normalizes_email_to_prevent_duplicate_identities():
    from navis.identity.permissions import ALL_PERMISSIONS

    db = FakeSession()
    svc = MemberService(db)
    await svc.invite(org_id=ORG_A, email="  John@Company.COM ", role=Role.EMPLOYEE,
                     actor_id=ACTOR, actor_permissions=ALL_PERMISSIONS)
    # Same identity, different casing/whitespace -> resolves to the SAME user -> already a member.
    with pytest.raises(MemberError):
        await svc.invite(org_id=ORG_A, email="john@company.com", role=Role.EMPLOYEE,
                         actor_id=ACTOR, actor_permissions=ALL_PERMISSIONS)
    members = await svc.list_members(ORG_A)
    assert len(members) == 1  # exactly one identity, not two


# ---- authorization escalation guards (V4) ----

async def test_admin_cannot_escalate_to_founder():
    from navis.identity.members import MemberForbidden
    from navis.identity.permissions import default_permissions_for_role

    db = FakeSession()
    svc = MemberService(db)
    admin_perms = default_permissions_for_role("admin")  # has MANAGE_ROLES, not founder powers
    # admin invites an employee (allowed — subset of admin perms)
    m = await svc.invite(org_id=ORG_A, email="e@x.com", role=Role.EMPLOYEE, actor_id=ACTOR, actor_permissions=admin_perms)
    # admin tries to promote them to FOUNDER (superuser) -> blocked
    with pytest.raises(MemberForbidden):
        await svc.assign_role(org_id=ORG_A, user_id=m.user_id, role=Role.FOUNDER, actor_id=ACTOR, actor_permissions=admin_perms)


async def test_cannot_remove_or_demote_last_founder():
    from navis.identity.members import MemberConflict
    from navis.identity.permissions import ALL_PERMISSIONS
    from tests.identity.fakes import ACTOR
    import uuid as _uuid

    db = FakeSession()
    svc = MemberService(db)
    founder_uid = _uuid.uuid4()
    db.add(Membership(user_id=founder_uid, organization_id=ORG_A, role=Role.FOUNDER))
    with pytest.raises(MemberConflict):
        await svc.remove(org_id=ORG_A, user_id=founder_uid, actor_id=ACTOR, actor_permissions=ALL_PERMISSIONS)
    with pytest.raises(MemberConflict):
        await svc.assign_role(org_id=ORG_A, user_id=founder_uid, role=Role.EMPLOYEE, actor_id=ACTOR, actor_permissions=ALL_PERMISSIONS)


# ---- multi-tenant isolation ----

async def test_api_keys_isolated_per_org():
    db = FakeSession()
    svc = ApiKeyService(db)
    await svc.create(org_id=ORG_A, name="a", scopes=[])
    await svc.create(org_id=ORG_B, name="b", scopes=[])
    assert {k.organization_id for k in await svc.list_for_org(ORG_A)} == {ORG_A}
    assert {k.organization_id for k in await svc.list_for_org(ORG_B)} == {ORG_B}


# ---- migration bootstrap: base creates the foundational tables ----

def test_base_migration_creates_identity_tables():
    src = (MIGRATIONS / "0000_identity_base.py").read_text(encoding="utf-8")
    created = {
        n.args[0].value
        for n in ast.walk(ast.parse(src))
        if isinstance(n, ast.Call) and getattr(n.func, "attr", None) == "create_table"
    }
    assert {"organizations", "users", "memberships", "sessions", "teams", "integrations"} <= created


def test_chain_is_anchored_to_base():
    base = (MIGRATIONS / "0000_identity_base.py").read_text(encoding="utf-8")
    first = (MIGRATIONS / "0001_signal_events_contract.py").read_text(encoding="utf-8")
    assert "down_revision: Union[str, None] = None" in base
    assert '"0000_identity_base"' in first  # 0001 now depends on the base


# ---- architecture guard: Identity imports no business domain ----

def test_identity_imports_no_business_domain():
    violations = []
    for py in IDENTITY.rglob("*.py"):
        for node in ast.walk(ast.parse(py.read_text(encoding="utf-8"), filename=str(py))):
            mods = []
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                mods = [node.module]
            for mod in mods:
                parts = mod.split(".")
                if len(parts) >= 2 and parts[0] == "navis" and parts[1] in FORBIDDEN:
                    violations.append(f"{py.name} imports {mod}")
    assert not violations, "Identity must depend on no business domain:\n" + "\n".join(violations)
