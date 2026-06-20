"""Identity V2: RBAC enforcement, token revocation, API keys, service accounts,
audit, member management, multi-tenancy, migration bootstrap, and the import guard.
"""
from __future__ import annotations

import ast
import pathlib

import pytest

from navis.identity import permissions as P
from navis.identity.apikeys import ApiKeyService, _hash
from navis.identity.audit import AuditAction, AuditService
from navis.identity.authz import Principal, require_permission
from navis.identity.members import MemberError, MemberService
from navis.identity.models import Membership, Role, User
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
    assert P.default_permissions_for_role("employee") == frozenset(
        {Permission.VIEW_DECISIONS, Permission.VIEW_SIGNALS}
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

    await svc.remove(org_id=ORG_A, user_id=m.user_id, actor_id=ACTOR)
    assert await svc.list_members(ORG_A) == []
    # Removal bulk-revokes the user's tokens.
    user = await db.get(User, m.user_id)
    assert user.tokens_valid_from is not None
    # And every mutation was audited.
    actions = {a.action for a in await AuditService(db).list_for_org(ORG_A)}
    assert {AuditAction.MEMBER_INVITE, AuditAction.ROLE_CHANGE, AuditAction.MEMBER_REMOVE} <= actions


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
        await svc.remove(org_id=ORG_A, user_id=founder_uid, actor_id=ACTOR)
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
    src = (MIGRATIONS / "0000_identity_base.py").read_text()
    created = {
        n.args[0].value
        for n in ast.walk(ast.parse(src))
        if isinstance(n, ast.Call) and getattr(n.func, "attr", None) == "create_table"
    }
    assert {"organizations", "users", "memberships", "sessions", "teams", "integrations"} <= created


def test_chain_is_anchored_to_base():
    base = (MIGRATIONS / "0000_identity_base.py").read_text()
    first = (MIGRATIONS / "0001_signal_events_contract.py").read_text()
    assert "down_revision: Union[str, None] = None" in base
    assert '"0000_identity_base"' in first  # 0001 now depends on the base


# ---- architecture guard: Identity imports no business domain ----

def test_identity_imports_no_business_domain():
    violations = []
    for py in IDENTITY.rglob("*.py"):
        for node in ast.walk(ast.parse(py.read_text(), filename=str(py))):
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
