"""Expiration PARITY — every credential minting path can bound a lifetime.

WHY THIS FILE EXISTS SEPARATELY FROM `test_credential_expiration.py`. That file
pins the DECISION: `ApiKeyService.check` refuses an expired credential, NULL
still means forever, expired and revoked stay distinguishable. It proves the
enforcement is correct and central.

This one pins REACH. `expires_at` existed on `ApiKeyService.create` and on
`ServiceAccountService.issue_credential`, and on nothing else — so four of the
five ways to mint a credential could only produce one that never expires. A
bounded lifetime nobody can select from the path they actually use is a feature
that exists in the schema and not in the product.

THE FIVE PATHS, AND THE ONE FUNCTION UNDER ALL OF THEM:

    POST /developer/agents                       -> create_agent
    POST /developer/agents/{id}/credentials      -> issue_credential
    POST /identity/api-keys                      -> ApiKeyService.create
    POST /identity/service-accounts              -> create -> create_agent
    POST /identity/service-accounts/{id}/rotate  -> rotate
                                                        |
                                                        v
                                             ApiKeyService.create

`ApiKey(...)` is constructed in exactly one place, so that arrow is the whole
inventory rather than a list someone has to keep current. The tests below assert
the pass-through at the service layer and — critically — that the FLOOR under it
is not bypassed by any of them: a past expiry is refused identically whichever
door it arrives through, because only one function decides.

WHAT IS DELIBERATELY NOT HERE. No test asserts a default expiry, because there
is none: every path still defaults to NULL. Forcing a lifetime on credentials
that did not have one would be an outage scheduled for whenever the date
arrived, and `0052` says so in prose. The tests that a default did NOT appear
are as load-bearing as the ones that the option did.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from navis.identity.apikeys import ApiKeyService
from navis.identity.authz import Principal, PrincipalKind, get_principal
from navis.identity.models import User
from navis.identity.permissions import effective_permissions
from navis.identity.service_accounts import ServiceAccountService
from navis.platform.database import get_session
from tests.identity.fakes import ACTOR, ORG_A, FakeSession

FUTURE = datetime.now(timezone.utc) + timedelta(days=30)
PAST = datetime.now(timezone.utc) - timedelta(minutes=1)
SCOPES = ["view_memory"]


def _iso(moment: datetime) -> str:
    return moment.isoformat()


def _rows(db, model):
    """The rows a `FakeSession` holds for one model, in insertion order.

    `FakeSession` keys its store by model class and does not track a
    transaction — no `commit`, no `rollback` — so these tests read the store
    directly and never assert on rollback. The one place that matters is called
    out where it applies.
    """
    return list(db.store.get(model, []))


# ── 1. Agent registration — the path a developer actually uses ──────────


async def test_agent_registration_can_bound_its_first_credential():
    """`POST /developer/agents` mints the identity AND the first credential.

    Before parity this was the ONE path most developers ever touch and the one
    that could not express a lifetime — so the only way to get a bounded
    credential was to register an agent, then call a second endpoint the console
    did not surface.
    """
    db = FakeSession()
    _sa, key, plaintext = await ServiceAccountService(db).create_agent(
        org_id=ORG_A, name="Navis Engineer", scopes=SCOPES, created_by=ACTOR,
        expires_at=FUTURE,
    )
    assert key.expires_at == FUTURE
    # Bounded, and still usable right now — an expiry in the future is not a
    # refusal. If this ever fails, registration mints dead credentials.
    assert await ApiKeyService(db).verify(plaintext) is key


async def test_agent_registration_still_defaults_to_no_expiry():
    """THE COMPATIBILITY PROPERTY, at the edge rather than in the service.

    Every caller that omits the field — which is every caller that existed
    before it — must get exactly the credential it got before.
    """
    db = FakeSession()
    _sa, key, _plaintext = await ServiceAccountService(db).create_agent(
        org_id=ORG_A, name="unbounded", scopes=SCOPES, created_by=ACTOR,
    )
    assert key.expires_at is None


async def test_the_agent_row_itself_carries_no_expiry():
    """The lifetime belongs to the CREDENTIAL, never to the identity.

    An agent that expired would take its historical attribution with it: runs,
    outcomes and audit rows point at `service_accounts.id`, and the whole reason
    a credential can be replaced without re-registering is that the identity
    outlives the secret. A `service_accounts.expires_at` would undo that, so
    this asserts the column does not exist rather than trusting nobody adds it.
    """
    db = FakeSession()
    sa, _key, _plaintext = await ServiceAccountService(db).create_agent(
        org_id=ORG_A, name="durable", scopes=SCOPES, created_by=ACTOR, expires_at=FUTURE,
    )
    assert not hasattr(sa, "expires_at")
    assert sa.revoked_at is None, "revocation is the only way an agent ends"


# ── 2. The legacy service-account paths ─────────────────────────────────


async def test_service_account_creation_can_bound_its_first_credential():
    """`POST /identity/service-accounts` delegates to `create_agent`, so parity
    there is parity here — asserted rather than assumed, because the two-value
    return is a separate signature that could have dropped the argument."""
    db = FakeSession()
    _sa, plaintext = await ServiceAccountService(db).create(
        org_id=ORG_A, name="worker", scopes=SCOPES, created_by=ACTOR, expires_at=FUTURE,
    )
    row = await ApiKeyService(db).verify(plaintext)
    assert row is not None and row.expires_at == FUTURE


async def test_rotation_can_bound_the_replacement_credential():
    db = FakeSession()
    svc = ServiceAccountService(db)
    sa, _key, _first = await svc.create_agent(
        org_id=ORG_A, name="rotating", scopes=SCOPES, created_by=ACTOR,
    )
    plaintext = await svc.rotate(sa.id, org_id=ORG_A, expires_at=FUTURE)
    row = await ApiKeyService(db).verify(plaintext)
    assert row is not None and row.expires_at == FUTURE


async def test_rotation_revokes_the_old_credential_and_the_new_one_works():
    """Parity must not have changed what rotation IS. The old secret dies, the
    new one lives, and the agent binding survives both."""
    db = FakeSession()
    svc = ServiceAccountService(db)
    sa, _key, first = await svc.create_agent(
        org_id=ORG_A, name="rotating", scopes=SCOPES, created_by=ACTOR,
    )
    second = await svc.rotate(sa.id, org_id=ORG_A, expires_at=FUTURE)

    assert await ApiKeyService(db).verify(first) is None, "rotation revokes the old key"
    new_row = await ApiKeyService(db).verify(second)
    assert new_row is not None
    assert new_row.service_account_id == sa.id, "still the same agent identity"
    assert new_row.scopes == SCOPES, "scopes come from the account, not the request"


async def test_rotation_does_not_inherit_the_retired_expiry():
    """DOCUMENTED, NOT ACCIDENTAL — and the reason it is tested.

    An account can hold credentials with different lifetimes, so there is no
    single expiry to carry over: the longest would extend a bound somebody
    chose, the shortest would shorten one. Rotation therefore takes the caller's
    instruction or none.

    The consequence is a real downgrade and this test is where it is written
    down: rotating a dated credential WITHOUT passing `expires_at` yields one
    that never expires. `ServiceAccountService.rotate` logs when that happens so
    an audit can find it.
    """
    db = FakeSession()
    svc = ServiceAccountService(db)
    sa, _key, _first = await svc.create_agent(
        org_id=ORG_A, name="dated", scopes=SCOPES, created_by=ACTOR, expires_at=FUTURE,
    )
    replacement = await svc.rotate(sa.id, org_id=ORG_A)

    row = await ApiKeyService(db).verify(replacement)
    assert row is not None
    assert row.expires_at is None, "rotation without an expiry produces an unbounded key"


# ── 3. The floor is not bypassable from any door ────────────────────────


@pytest.mark.parametrize("path", ["create_agent", "create", "rotate", "issue_credential"])
async def test_a_past_expiry_is_refused_on_every_minting_path(path):
    """ONE decision, four doors.

    `ApiKeyService.create` refuses an expiry already in the past. Every path
    above reaches it, so none of them needs — or is allowed — its own copy of
    that rule. A path that accepted a dead credential would be minting a support
    ticket, and a path with its OWN validation would be the second implementation
    that eventually disagrees with this one.
    """
    db = FakeSession()
    svc = ServiceAccountService(db)

    if path == "create_agent":
        call = svc.create_agent(
            org_id=ORG_A, name="x", scopes=SCOPES, created_by=ACTOR, expires_at=PAST
        )
    elif path == "create":
        call = svc.create(
            org_id=ORG_A, name="x", scopes=SCOPES, created_by=ACTOR, expires_at=PAST
        )
    else:
        sa, _key, _pt = await svc.create_agent(
            org_id=ORG_A, name="host", scopes=SCOPES, created_by=ACTOR,
        )
        call = (
            svc.rotate(sa.id, org_id=ORG_A, expires_at=PAST)
            if path == "rotate"
            else svc.issue_credential(
                sa.id, org_id=ORG_A, name="extra", created_by=ACTOR, expires_at=PAST
            )
        )

    with pytest.raises(ValueError, match="must be in the future"):
        await call


async def test_no_minting_path_reimplements_the_expiry_rule():
    """Structural, and the property the whole task depends on.

    If `expires_at` is ever compared against `now` outside `apikeys.py`, there
    are two answers to "is this credential still alive" and the wrong one will
    win some day. The routers and the service-account layer may PASS the value;
    they may not judge it.
    """
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[2] / "navis"
    for module in [
        "identity/service_accounts.py",
        "identity/admin_router.py",
        "developer/agents_router.py",
    ]:
        source = (root / module).read_text()
        # `credential_status_of` is the shared DISPLAY derivation and is allowed;
        # an inline comparison of an expiry against the clock is not.
        assert "expires_at <=" not in source, module
        assert "expires_at <" not in source, module
        assert "expires_at >" not in source, module


# ── 4. At the wire ──────────────────────────────────────────────────────
#
# The service tests above prove the value reaches `ApiKeyService.create`. These
# prove an HTTP CLIENT can put it there — which is the half that was missing,
# since `expires_at` was always settable in Python and settable through no route
# but one.


def _operator_principal(
    role: str = "founder", developer_role: str | None = "owner"
) -> Principal:
    """A founder session, resolved from BOTH surfaces as production resolves it.

    The Developer role is not decoration here: `SCOPES` below contains
    `view_memory`, a Developer-surface permission, and `ensure_grantable_scopes`
    refuses to mint a credential carrying a permission the operator does not
    hold. A founder holds it through `developer_role='owner'` — which is what
    signup writes and what `0053_developer_access` backfilled — so omitting it
    would model a founder who cannot exist.
    """
    uid = uuid.uuid4()
    return Principal(
        kind=PrincipalKind.USER,
        org_id=ORG_A,
        subject_id=uid,
        permissions=effective_permissions(role, developer_role=developer_role),
        role=role,
        developer_role=developer_role,
        user=User(id=uid, email="operator@x.com"),
    )


@pytest.fixture()
def admin_client():
    """`/identity` wired to a fake session and a founder principal."""
    from navis.identity.admin_router import router as admin_router

    db = FakeSession()
    app = FastAPI()
    app.include_router(admin_router)
    app.dependency_overrides[get_session] = lambda: db
    app.dependency_overrides[get_principal] = _operator_principal
    return TestClient(app, raise_server_exceptions=False), db


def test_api_key_creation_accepts_an_expiry_over_http(admin_client):
    client, db = admin_client
    response = client.post(
        "/identity/api-keys",
        json={"name": "CI", "scopes": SCOPES, "expires_at": _iso(FUTURE)},
    )
    assert response.status_code == 201, response.text
    body = response.json()
    # Echoed, so a caller knows what it got without a second read.
    assert body["expires_at"] is not None
    # And it is on the ROW, not merely in the response.
    from navis.identity.enterprise_models import ApiKey

    assert _rows(db, ApiKey)[-1].expires_at is not None


def test_api_key_creation_without_an_expiry_is_unchanged(admin_client):
    """The contract every existing caller depends on."""
    from navis.identity.enterprise_models import ApiKey

    client, db = admin_client
    response = client.post("/identity/api-keys", json={"name": "forever", "scopes": SCOPES})
    assert response.status_code == 201, response.text
    assert response.json()["expires_at"] is None
    assert _rows(db, ApiKey)[-1].expires_at is None


def test_api_key_creation_refuses_a_past_expiry_with_400(admin_client):
    """400, not 422: the field parsed, the instant it names is the problem."""
    client, _db = admin_client
    response = client.post(
        "/identity/api-keys",
        json={"name": "born dead", "scopes": SCOPES, "expires_at": _iso(PAST)},
    )
    assert response.status_code == 400, response.text
    assert "future" in response.json()["detail"]


def test_the_api_key_list_reports_expiry_and_the_four_state_status(admin_client):
    """THE CONSISTENCY FIX.

    This list reported `revoked` and nothing else, so a key that aged out
    yesterday appeared with `revoked: false` — and the console rendered it under
    a heading that promised it would authenticate on the next request. `status`
    is the same derivation the agent credential list uses, so a row that appears
    in both (every agent credential does) cannot be described two ways.
    """
    from navis.identity.enterprise_models import ApiKey

    client, db = admin_client
    client.post("/identity/api-keys", json={"name": "forever", "scopes": SCOPES})
    client.post(
        "/identity/api-keys",
        json={"name": "dated", "scopes": SCOPES, "expires_at": _iso(FUTURE)},
    )
    # Age one row past its expiry. Minting one already dead is refused, and
    # ageing a row is what time does.
    _rows(db, ApiKey)[-1].expires_at = PAST

    rows = client.get("/identity/api-keys").json()["api_keys"]
    by_name = {row["name"]: row for row in rows}

    assert by_name["forever"]["expires_at"] is None
    assert by_name["forever"]["status"] == "active"

    assert by_name["dated"]["expires_at"] is not None
    assert by_name["dated"]["status"] == "expired"
    # `revoked` is KEPT alongside `status`, and the two answer different
    # questions: nobody cut this key off, it simply reached the end of its life.
    assert by_name["dated"]["revoked"] is False


def test_rotation_accepts_an_expiry_over_http(admin_client):
    """The rotate route took no body at all before. An absent body must still
    work, because that is what every existing caller sends."""
    from navis.identity.enterprise_models import ApiKey

    client, db = admin_client
    created = client.post("/identity/service-accounts", json={"name": "w", "scopes": SCOPES})
    assert created.status_code == 201, created.text
    sa_id = created.json()["id"]

    assert client.post(f"/identity/service-accounts/{sa_id}/rotate").status_code == 200
    assert _rows(db, ApiKey)[-1].expires_at is None, "no body means no expiry, as before"

    dated = client.post(
        f"/identity/service-accounts/{sa_id}/rotate", json={"expires_at": _iso(FUTURE)}
    )
    assert dated.status_code == 200, dated.text
    assert _rows(db, ApiKey)[-1].expires_at is not None


@pytest.fixture()
def agents_client():
    """`/developer/agents` with the operator gate satisfied.

    `_operator` is overridden rather than reimplemented: it is the dependency
    that requires a signed-in member, and the point of these tests is the
    expiry, not the gate. `tests/integration/test_developer_agents_postgres.py`
    exercises the real gates against real SQL.
    """
    from navis.developer import agents_router as mod

    db = FakeSession()
    principal = _operator_principal()
    ctx = mod.DeveloperContext(
        principal=principal,
        organization_id=ORG_A,
        kind="user",
        subject_id=principal.subject_id,
        service_account_id=None,
        credential_id=None,
        permissions=principal.permissions,
    )

    app = FastAPI()
    app.include_router(mod.router)
    app.dependency_overrides[get_session] = lambda: db
    app.dependency_overrides[mod._operator] = lambda: ctx
    # THE THROTTLE TOO, and it is not optional. `_WRITE_LIMIT` is a separate
    # route dependency that resolves `get_principal` itself and then calls
    # Redis — so with only `_operator` overridden these routes answer 401 (no
    # credential to authenticate) or 500 (no Redis), and neither is the thing
    # under test here. `.dependency` is the closure `Depends` wraps, which is
    # the key `dependency_overrides` matches on.
    app.dependency_overrides[mod._WRITE_LIMIT.dependency] = lambda: principal
    return TestClient(app, raise_server_exceptions=False), db


def test_agent_registration_accepts_a_credential_expiry_over_http(agents_client):
    """The path the console drives, and the one the whole task is about."""
    client, _db = agents_client
    response = client.post(
        "/developer/agents",
        json={
            "name": "Navis Engineer",
            "runtime": "claude-code",
            "permissions": SCOPES,
            "credential_expires_at": _iso(FUTURE),
        },
    )
    assert response.status_code == 201, response.text
    credential = response.json()["credential"]
    assert credential["expires_at"] is not None
    assert credential["status"] == "active", "a future expiry is not a refusal"
    # The secret is returned here and only here.
    assert credential["secret"].startswith("navis_")


def test_agent_registration_without_an_expiry_is_unchanged(agents_client):
    client, _db = agents_client
    response = client.post(
        "/developer/agents", json={"name": "unbounded", "permissions": SCOPES}
    )
    assert response.status_code == 201, response.text
    assert response.json()["credential"]["expires_at"] is None
    assert response.json()["credential"]["status"] == "active"


def test_agent_registration_refuses_a_past_expiry_with_400(agents_client):
    client, db = agents_client
    response = client.post(
        "/developer/agents",
        json={"name": "born dead", "permissions": SCOPES, "credential_expires_at": _iso(PAST)},
    )
    assert response.status_code == 400, response.text
    assert "future" in response.json()["detail"]


def test_a_rejected_expiry_mints_no_credential(agents_client):
    """`create_agent` FLUSHES the agent row before minting the credential, so a
    refusal at the credential has to undo it.

    WHAT THIS CAN AND CANNOT PROVE HERE. `FakeSession` models no transaction —
    it has no `commit` and no `rollback`, and the override installs it as a
    plain value rather than through `get_session`'s try/except — so the rollback
    that protects this in production is NOT observable in this test and is not
    asserted. What is asserted is the part this repository controls: the route
    answers 400 rather than 201, and no credential row exists for the agent, so
    nothing was handed out. The rollback itself is a property of `get_session`
    (which rolls back on any exception, and `HTTPException` is one) and is
    exercised against real SQL in
    `tests/integration/test_developer_agents_postgres.py`.
    """
    from navis.identity.enterprise_models import ApiKey

    client, db = agents_client
    response = client.post(
        "/developer/agents",
        json={"name": "born dead", "permissions": SCOPES, "credential_expires_at": _iso(PAST)},
    )
    assert response.status_code == 400
    # The decisive fact: no credential row was written, so no secret was
    # returned and nothing can authenticate as this agent.
    assert _rows(db, ApiKey) == []
