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

    await svc.revoke(key.id)
    assert await svc.verify(plaintext) is None  # revoked keys do not authenticate


# ---- service account lifecycle (rotation revokes prior credential) ----

async def test_service_account_rotation_revokes_old_key():
    db = FakeSession()
    sa_svc = ServiceAccountService(db)
    sa, secret1 = await sa_svc.create(org_id=ORG_A, name="worker", scopes=[Permission.EXECUTE_ACTIONS])
    keys = ApiKeyService(db)
    assert await keys.verify(secret1) is not None

    secret2 = await sa_svc.rotate(sa.id)
    assert await keys.verify(secret1) is None      # old credential dead
    assert await keys.verify(secret2) is not None  # new credential lives

    await sa_svc.revoke(sa.id)
    assert await keys.verify(secret2) is None       # revoke kills all credentials


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
    db = FakeSession()
    svc = MemberService(db)
    m = await svc.invite(org_id=ORG_A, email="new@x.com", role=Role.EMPLOYEE, actor_id=ACTOR)
    assert m.role == Role.EMPLOYEE
    with pytest.raises(MemberError):
        await svc.invite(org_id=ORG_A, email="new@x.com", role=Role.EMPLOYEE, actor_id=ACTOR)

    await svc.assign_role(org_id=ORG_A, user_id=m.user_id, role=Role.ADMIN, actor_id=ACTOR)
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
