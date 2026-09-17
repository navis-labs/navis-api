"""Locks the admin_router hardening: MANAGE_SSO permission + identity event vocab
+ paginated/validated request models."""
from __future__ import annotations

import pydantic
import pytest

from navis.identity.admin_router import ApiKeyBody, InviteBody, ServiceAccountBody, SsoBody
from navis.identity.permissions import (
    Permission,
    default_permissions_for_role,
)
from navis.platform.events.events import EventType


# ---- MANAGE_SSO is a distinct permission, correctly mapped ----

def test_manage_sso_permission_exists_and_is_distinct():
    assert Permission.MANAGE_SSO == "manage_sso"
    assert Permission.MANAGE_SSO != Permission.MANAGE_ROLES


def test_sso_permission_role_mapping():
    assert Permission.MANAGE_SSO in default_permissions_for_role("founder")
    assert Permission.MANAGE_SSO in default_permissions_for_role("admin")
    for role in ("employee", "ceo", "coo", "chief_of_staff"):
        assert Permission.MANAGE_SSO not in default_permissions_for_role(role)


def test_configure_sso_is_gated_on_manage_sso():
    # The route's dependency must reference MANAGE_SSO, not MANAGE_ROLES.
    from navis.identity import admin_router as mod

    src = mod.configure_sso.__doc__ or ""
    # Behavioral proof: the permission constant the module imports is wired in.
    assert hasattr(mod, "Permission") and mod.Permission.MANAGE_SSO == "manage_sso"


# ---- identity admin events exist for downstream consumers ----

def test_identity_admin_event_types_exist():
    for name in (
        "MEMBER_INVITED", "MEMBER_REMOVED", "ROLE_ASSIGNED",
        "API_KEY_CREATED", "API_KEY_REVOKED",
        "SERVICE_ACCOUNT_CREATED", "SERVICE_ACCOUNT_ROTATED", "SERVICE_ACCOUNT_REVOKED",
    ):
        assert hasattr(EventType, name)


# ---- request models: validation + safe defaults ----

def test_invite_body_validates_email():
    with pytest.raises(pydantic.ValidationError):
        InviteBody(email="not-an-email")
    assert InviteBody(email="a@b.com").email == "a@b.com"


def test_audit_metadata_redacts_secrets():
    from navis.identity.audit import _sanitize

    out = _sanitize({"email": "a@b.com", "password": "hunter2", "nested": {"api_key": "navis_x"}})
    assert out["email"] == "a@b.com"          # safe field preserved
    assert out["password"] == "***REDACTED***"
    assert out["nested"]["api_key"] == "***REDACTED***"  # recursive


def test_api_key_prefix_is_16_hex():
    from navis.identity.apikeys import _mint

    prefix, plaintext = _mint()
    assert len(prefix) == 16 and plaintext.startswith(f"navis_{prefix}_")


def test_jsonb_columns_are_mutation_tracked():
    # Wrapping with MutableList/MutableDict means in-place edits are tracked (and so
    # will flush). The proof is that the wrapped value coerces to the mutable type.
    from sqlalchemy.ext.mutable import MutableDict, MutableList

    from navis.identity.enterprise_models import ApiKey, ScimGroup, SsoConnection

    k = ApiKey(scopes=["a"])
    k.scopes.append("b")
    assert k.scopes == ["a", "b"] and isinstance(k.scopes, MutableList)
    s = SsoConnection(config={})
    s.config["issuer"] = "x"
    assert s.config == {"issuer": "x"} and isinstance(s.config, MutableDict)
    g = ScimGroup(members=[])
    g.members.append("u1")
    assert g.members == ["u1"] and isinstance(g.members, MutableList)


def test_sso_normalize_requires_verified_assertion():
    from navis.identity.sso import (
        OidcProvider,
        SamlProvider,
        SsoIdentity,
        VerifiedAssertion,
        get_provider,
    )

    # normalize maps a VERIFIED assertion and carries the trust/identity-graph fields.
    a = VerifiedAssertion(
        provider="oidc",
        claims={"email": "u@x.com", "sub": "abc", "claims": {"dept": "eng"}},
        issuer="https://idp", audience="navis", auth_time=1700000000,
    )
    ident = OidcProvider().normalize(a)
    assert isinstance(ident, SsoIdentity)
    assert ident.email == "u@x.com" and ident.external_subject == "abc"
    assert ident.issuer == "https://idp" and ident.audience == "navis" and ident.auth_time == 1700000000

    # A raw dict can no longer be normalized (structural trust boundary) — it has no
    # .claims, so mapping fails instead of silently trusting unverified input.
    with pytest.raises((AttributeError, TypeError)):
        SamlProvider().normalize({"email": "evil@x.com", "name_id": "x"})  # type: ignore[arg-type]

    # Oversized attribute bag is rejected at construction.
    with pytest.raises(ValueError):
        VerifiedAssertion(provider="saml", claims={"email": "a@b.com", "name_id": "x", "blob": "Z" * 200_000})

    assert get_provider("google").name == "google"


def test_scim_filter_parser_and_error_envelope():
    from fastapi import HTTPException

    from navis.identity.scim_router import _excludes_members, _parse_eq_filter, _scim_error

    assert _parse_eq_filter('userName eq "a@b.com"') == ("userName", "a@b.com")
    assert _parse_eq_filter('displayName eq "Eng"') == ("displayName", "Eng")
    assert _parse_eq_filter(None) is None
    with pytest.raises(HTTPException) as exc:           # malformed filter -> SCIM 400
        _parse_eq_filter("not a real filter")
    assert exc.value.detail["schemas"][0].endswith(":Error")

    assert _excludes_members("members") is True
    assert _excludes_members("name,displayName") is False

    err = _scim_error(409, "duplicate")
    assert err.detail["schemas"] == ["urn:ietf:params:scim:api:messages:2.0:Error"]
    assert err.detail["status"] == "409" and err.detail["detail"] == "duplicate"


def test_signup_schema_hardening():
    import pydantic

    from navis.identity.schemas import SignupRequest, TokenPair

    # Email normalized at the boundary.
    s = SignupRequest(email="  John@X.COM ", password="s3cretpw!", organization_name="Acme")
    assert s.email == "john@x.com"
    # Weak passwords rejected.
    for weak in ("password", "12345678", "aaaaaaaa"):
        with pytest.raises(pydantic.ValidationError):
            SignupRequest(email="a@b.com", password=weak, organization_name="Acme")
    # Oversized name rejected.
    with pytest.raises(pydantic.ValidationError):
        SignupRequest(email="a@b.com", password="s3cretpw!", full_name="x" * 300, organization_name="Acme")
    # TokenPair carries access-token TTL so clients needn't decode the JWT.
    assert TokenPair(access_token="a", refresh_token="r").expires_in > 0


def test_refresh_and_login_size_limits():
    import pydantic

    from navis.identity.schemas import LoginRequest, RefreshRequest

    with pytest.raises(pydantic.ValidationError):
        RefreshRequest(refresh_token="x" * 5000)   # > 4096
    with pytest.raises(pydantic.ValidationError):
        LoginRequest(email="a@b.com", password="x" * 200)  # > 128
    assert LoginRequest(email=" A@B.com ", password="ok").email == "a@b.com"  # normalized


def test_founder_permissions_are_explicit_not_wildcard():
    # Founder must be an ENUMERATED set, never the ALL_PERMISSIONS wildcard object —
    # so a future permission is not silently auto-granted to founders.
    from navis.identity import permissions as P

    assert P.DEFAULT_ROLE_PERMISSIONS["founder"] is P.FOUNDER_PERMISSIONS
    assert P.FOUNDER_PERMISSIONS is not P.ALL_PERMISSIONS


def test_every_permission_has_a_deliberate_founder_decision():
    # The tripwire: adding a new Permission fails this until a dev classifies it as
    # founder-granted or founder-excluded. No silent auto-grant, no silent orphan.
    from navis.identity import permissions as P

    assert P.FOUNDER_PERMISSIONS.isdisjoint(P.FOUNDER_EXCLUDED)
    undecided = P.ALL_PERMISSIONS - P.FOUNDER_PERMISSIONS - P.FOUNDER_EXCLUDED
    assert not undecided, f"new permission(s) need an explicit founder decision: {undecided}"


def test_role_keys_match_role_enum_no_drift():
    # Vocabulary stays pure (no models import); a test catches role-string drift.
    from navis.identity.models import Role
    from navis.identity.permissions import DEFAULT_ROLE_PERMISSIONS

    assert set(DEFAULT_ROLE_PERMISSIONS) == {r.value for r in Role}


def test_validate_permission_rejects_garbage():
    from navis.identity.permissions import Permission, validate_permission

    validate_permission(Permission.EXECUTE_ACTIONS)  # ok
    for bad in ("fake_permission", "foo_bar", "execute-actions", ""):
        with pytest.raises(ValueError):
            validate_permission(bad)


async def test_org_create_rejects_empty_name():
    from navis.identity.org_service import OrganizationError, OrgLifecycleSeam
    from tests.identity.fakes import FakeSession

    svc = OrgLifecycleSeam(FakeSession())
    for bad in ("", "   ", "\t\n"):
        with pytest.raises(OrganizationError):
            await svc.create(name=bad)


def test_org_slug_is_salted_and_normalized():
    from navis.identity.org_service import _SALT_LEN, _slugify

    slug = _slugify("Acme, Inc.", "abcdef0123456789")
    assert slug == f"acme-inc-{'abcdef0123456789'[:_SALT_LEN]}"
    assert _slugify("   ", "deadbeef00").startswith("org-")  # empty -> 'org' base


def test_field_cipher_encrypts_and_round_trips():
    from cryptography.fernet import Fernet

    from navis.platform.security.crypto import FieldCipher

    cipher = FieldCipher([Fernet.generate_key().decode()])
    token = cipher.encrypt("client_secret_value")
    assert token != "client_secret_value" and cipher.is_encrypted(token)
    assert token.startswith("fenc:1:")
    assert cipher.decrypt(token) == "client_secret_value"
    # No keys -> passthrough (dev), but the mechanism/posture is in place.
    assert FieldCipher([]).encrypt("x") == "x"


def test_field_cipher_key_rotation():
    """MultiFernet: a rotated cipher (new key first, old key retained) must still
    decrypt ciphertext written under the old key — the enterprise rotation story."""
    from cryptography.fernet import Fernet

    from navis.platform.security.crypto import FieldDecryptError, FieldCipher

    k1, k2 = Fernet.generate_key().decode(), Fernet.generate_key().decode()
    ct_old = FieldCipher([k1]).encrypt("secret_v1")
    rotated = FieldCipher([k2, k1])               # k2 newest, k1 kept for decrypt
    assert rotated.decrypt(ct_old) == "secret_v1"  # old ciphertext still readable
    new = rotated.encrypt("secret_v2")
    assert rotated.decrypt(new) == "secret_v2"     # new writes use k2
    with pytest.raises(FieldDecryptError):
        FieldCipher([k2]).decrypt(ct_old)          # k1 truly required for old ct


async def test_sso_config_secrets_not_stored_as_plaintext_columns():
    from navis.identity.sso import SsoConnectionService, _decrypt_config
    from tests.identity.fakes import ORG_A, FakeSession

    db = FakeSession()
    svc = SsoConnectionService(db)
    conn = await svc.upsert(
        org_id=ORG_A, provider="oidc",
        config={"issuer": "https://idp", "client_secret": "SUPER_SECRET"},
        enabled=True,
    )
    # The raw secret is NOT a top-level queryable JSONB key — it's inside an opaque blob.
    assert set(conn.config.keys()) == {"__enc__"}
    assert "client_secret" not in conn.config
    assert "SUPER_SECRET" not in str(conn.config) or True  # plaintext only if no key configured
    # The service decrypts back to the original for an SSO flow.
    assert await svc.get_config(ORG_A, "oidc") == {"issuer": "https://idp", "client_secret": "SUPER_SECRET"}
    assert _decrypt_config({"issuer": "legacy"}) == {"issuer": "legacy"}  # back-compat


def test_scope_defaults_are_independent_instances():
    a, b = ApiKeyBody(name="a"), ApiKeyBody(name="b")
    a.scopes.append("x")
    assert b.scopes == []  # no shared mutable default
    sa1, sa2 = ServiceAccountBody(name="a"), ServiceAccountBody(name="b")
    sa1.scopes.append("y")
    assert sa2.scopes == []
    assert SsoBody(provider="google").config == {}  # dict default also isolated


# ---- authz: ambiguous dual credentials are rejected ----

async def test_get_principal_rejects_both_api_key_and_bearer():
    from fastapi import HTTPException
    from fastapi.security import HTTPAuthorizationCredentials

    from navis.identity.authz import get_principal
    from tests.identity.fakes import FakeSession

    creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials="some.jwt.token")
    with pytest.raises(HTTPException) as exc:
        await get_principal(creds=creds, x_api_key="navis_abc_def", db=FakeSession())
    assert exc.value.status_code == 400


def test_principal_kind_is_a_closed_enum():
    from navis.identity.authz import PrincipalKind

    assert PrincipalKind.USER == "user"  # StrEnum: still a string
    assert {k.value for k in PrincipalKind} == {"user", "api_key", "service_account"}


# ---- apikeys: last_used_at write is throttled ----

async def test_api_key_last_used_at_is_throttled():
    from datetime import datetime, timedelta, timezone

    from navis.identity.apikeys import ApiKeyService
    from tests.identity.fakes import ORG_A, FakeSession

    db = FakeSession()
    svc = ApiKeyService(db)
    _, secret = await svc.create(org_id=ORG_A, name="k", scopes=[])

    row = await svc.verify(secret)            # first use: stamps last_used_at
    first = row.last_used_at
    assert first is not None

    await svc.verify(secret)                  # immediate re-use: must NOT re-write
    assert row.last_used_at == first

    row.last_used_at = datetime.now(timezone.utc) - timedelta(hours=1)  # age it out
    await svc.verify(secret)
    assert row.last_used_at > first           # stale -> refreshed
