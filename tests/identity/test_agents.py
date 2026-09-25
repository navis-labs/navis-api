"""Agents: the vocabulary, the wire contract, and the service rules.

WHAT THIS FILE IS FOR, AND WHAT IT IS NOT. These are the checks that do not need
a database: a closed vocabulary, a derived status, a request model that has no
identity field to send, and the service rules that are pure enough for the
in-memory `FakeSession` to model honestly. Everything that depends on real SQL —
tenant predicates at the wire, the permission cascade reaching a live credential,
revocation, concurrency — is proved against real PostgreSQL in
`tests/integration/test_developer_agents_postgres.py`, because a fake session
that "passes" a tenant filter it does not implement is worse than no test.
"""
from __future__ import annotations

import pytest

from navis.identity import agents as A
from navis.identity.apikeys import ApiKeyService
from navis.identity.permissions import Permission, VALID_PERMISSIONS
from navis.identity.service_accounts import ServiceAccountService
from tests.identity.fakes import ACTOR, ORG_A, ORG_B, FakeSession


# ── The vocabulary ──────────────────────────────────────────────────────────


def test_the_agent_vocabularies_are_closed_and_match_the_published_contract():
    """These exact strings are `AGENT_RUNTIMES` in `@navis/types` and the values a
    database CHECK constraint admits. A change here that is not made in all three
    places is a console offering a runtime the API refuses."""
    assert A.AGENT_RUNTIMES == frozenset(
        {"claude-code", "codex", "cursor", "gemini", "openai-agents", "custom"}
    )
    assert A.AGENT_ENVIRONMENTS == frozenset({"development", "staging", "production"})


def test_status_is_derived_from_the_revocation_clock_and_nothing_else():
    """`revoked_at` is what authentication reads to refuse a revoked agent, so it
    is what the wire `status` must be computed from. A stored status column would
    be a second authority the auth path never consults."""
    from datetime import datetime, timezone

    assert A.status_of(None) == "active"
    assert A.status_of(datetime.now(timezone.utc)) == "revoked"
    # There is deliberately no third state: revocation is terminal.
    assert {s.value for s in A.AgentStatus} == {"active", "revoked"}


def test_environment_and_runtime_are_not_authorization_inputs():
    """Structural: no gate may branch on either.

    An agent tagged `production` holds exactly the permissions its scopes name.
    If either word ever appears in the authorization module, someone has added a
    second, undocumented authorization axis next to `Permission`.
    """
    import pathlib

    authz = (
        pathlib.Path(__file__).resolve().parents[2] / "navis" / "identity" / "authz.py"
    ).read_text()
    assert "environment" not in authz
    assert "runtime" not in authz


# ── The wire contract ───────────────────────────────────────────────────────


def test_no_agent_request_model_carries_an_identity_field():
    """The organization, the owner and the agent are resolved server-side.

    Checked on the MODEL rather than only at the wire so the guarantee survives a
    route being rewritten: a field that does not exist cannot be honoured, cannot
    be ignored-with-a-warning, and cannot be quietly trusted later.
    """
    from navis.developer.agent_schemas import (
        CreateAgentCredentialRequest,
        CreateAgentRequest,
        UpdateAgentRequest,
    )

    banned = {
        "organization_id", "org_id", "tenant_id", "user_id", "owner_user_id",
        "principal_id", "subject_id", "service_account_id", "created_by",
        "agent_id", "scopes",
    }
    for model in (CreateAgentRequest, UpdateAgentRequest, CreateAgentCredentialRequest):
        assert not (set(model.model_fields) & banned), model.__name__


def test_a_create_request_rejects_a_blank_name_and_an_unknown_vocabulary():
    from pydantic import ValidationError

    from navis.developer.agent_schemas import CreateAgentRequest

    # Whitespace passes `min_length` and renders as an unidentifiable row.
    with pytest.raises(ValidationError):
        CreateAgentRequest(name="   ")
    with pytest.raises(ValidationError):
        CreateAgentRequest(name="ok", environment="prod")     # not the enum value
    with pytest.raises(ValidationError):
        CreateAgentRequest(name="ok", runtime="some-new-tool")

    ok = CreateAgentRequest(name="  Padded  ")
    assert ok.name == "Padded"          # stripped, not merely accepted
    assert ok.environment.value == "development"
    assert ok.runtime.value == "custom"
    assert ok.permissions == []          # nothing is granted by omission


def test_only_the_creation_responses_can_carry_a_secret():
    """A route cannot leak a secret by returning the wrong model, because only two
    models have anywhere to put one."""
    from navis.developer import agent_schemas as S

    with_secret = {
        name for name, model in vars(S).items()
        if isinstance(model, type) and hasattr(model, "model_fields")
        and "secret" in model.model_fields
    }
    # Exactly one model, and it is the one returned only by the two POST handlers.
    # `CreatedAgentOut` reaches the secret through this model rather than
    # declaring a field of its own, which is why it is not in this set.
    assert with_secret == {"CreatedAgentCredentialOut"}
    assert "secret" not in S.AgentCredentialOut.model_fields
    assert "secret" not in S.AgentOut.model_fields


def test_agent_permissions_are_the_one_permission_vocabulary():
    """There is no `agent_permissions` table and no second scope language.

    The console offers, the API validates and the credential carries the SAME
    strings a human role is granted from. An agent and a person are authorized by
    the same words, so a grant means one thing in this system.
    """
    from navis.developer.agents_router import _AGENT_CAPABILITY

    assert Permission.MANAGE_SERVICE_ACCOUNTS in VALID_PERMISSIONS
    # The routes gate on an EXISTING permission and an EXISTING capability.
    from navis.organization.contract import Capability

    assert isinstance(_AGENT_CAPABILITY, Capability)


def test_the_escalation_guard_has_exactly_one_implementation():
    """`identity.admin_router` and `developer.agents_router` must not each hold a
    copy of "may this principal grant this scope" — one copy gets a fix and the
    other does not, and the divergence is invisible until a credential is minted
    through the wrong door."""
    from navis.developer.agents_router import ensure_grantable_scopes as agents_guard
    from navis.identity import admin_router
    from navis.identity.authz import ensure_grantable_scopes

    assert admin_router._ensure_grantable_scopes is ensure_grantable_scopes
    assert agents_guard is ensure_grantable_scopes


# ── The service rules ───────────────────────────────────────────────────────


async def test_creating_an_agent_stores_its_descriptive_fields_and_one_credential():
    db = FakeSession()
    sa, key, plaintext = await ServiceAccountService(db).create_agent(
        org_id=ORG_A, name="deploy bot", scopes=[Permission.VIEW_STATE],
        created_by=ACTOR, description="ships things", environment="production",
        runtime="claude-code",
    )
    assert sa.organization_id == ORG_A
    assert (sa.description, sa.environment, sa.runtime) == (
        "ships things", "production", "claude-code"
    )
    assert sa.revoked_at is None
    # The credential is bound to BOTH the organization and the agent.
    assert key.organization_id == ORG_A and key.service_account_id == sa.id
    assert key.scopes == [Permission.VIEW_STATE]
    assert plaintext.startswith("navis_") and key.prefix in plaintext
    # Only a digest is stored.
    assert plaintext not in key.hashed_secret


async def test_the_original_two_value_create_still_works_and_shares_one_path():
    """`/identity/service-accounts` predates the Agents API and keeps its contract;
    both write the same row through the same function."""
    db = FakeSession()
    sa, plaintext = await ServiceAccountService(db).create(
        org_id=ORG_A, name="worker", scopes=[Permission.EXECUTE_ACTIONS]
    )
    assert await ApiKeyService(db).verify(plaintext) is not None
    # Defaults, so a service account made the old way is a valid agent record.
    assert (sa.description, sa.environment, sa.runtime) == ("", "development", "custom")


async def test_changing_an_agents_permissions_reaches_its_live_credentials():
    """Authorization reads `api_keys.scopes`, so the account's list is a template.

    Editing only the account would change what a console displays and nothing
    about what the credential in the agent's environment can reach.
    """
    db = FakeSession()
    svc = ServiceAccountService(db)
    sa, key, _ = await svc.create_agent(
        org_id=ORG_A, name="a", scopes=[Permission.VIEW_STATE, Permission.VIEW_MEMORY]
    )
    await svc.update(sa.id, org_id=ORG_A, scopes=[Permission.VIEW_STATE])

    assert sa.scopes == [Permission.VIEW_STATE]
    assert key.scopes == [Permission.VIEW_STATE], "the live credential kept a dropped scope"


async def test_a_revoked_credential_is_not_re_granted_by_a_permission_edit():
    """The cascade targets ACTIVE keys only. Rewriting a revoked row's scopes
    would edit a tombstone and make the audit trail describe a key that had
    already been retired."""
    db = FakeSession()
    svc = ServiceAccountService(db)
    sa, key, _ = await svc.create_agent(
        org_id=ORG_A, name="a", scopes=[Permission.VIEW_STATE]
    )
    await svc.keys.revoke(key.id, org_id=ORG_A)
    await svc.update(sa.id, org_id=ORG_A, scopes=[Permission.VIEW_MEMORY])

    assert sa.scopes == [Permission.VIEW_MEMORY]
    assert key.scopes == [Permission.VIEW_STATE], "a revoked credential was rewritten"


async def test_an_additional_credential_inherits_the_agents_scopes():
    """A credential is exactly as capable as the agent it belongs to — the mint
    call takes no scopes, so this cannot be used to widen anything."""
    db = FakeSession()
    svc = ServiceAccountService(db)
    sa, _, _ = await svc.create_agent(
        org_id=ORG_A, name="a", scopes=[Permission.VIEW_STATE]
    )
    extra, plaintext = await svc.issue_credential(sa.id, org_id=ORG_A, name="CI")

    assert extra.scopes == [Permission.VIEW_STATE]
    assert extra.service_account_id == sa.id
    assert await ApiKeyService(db).verify(plaintext) is not None


async def test_a_revoked_agent_cannot_be_given_a_new_credential():
    """A key minted for a revoked agent would be valid at the `api_keys` row and
    dead at the gate — a credential that authenticates nowhere."""
    db = FakeSession()
    svc = ServiceAccountService(db)
    sa, _, _ = await svc.create_agent(org_id=ORG_A, name="a", scopes=[])
    await svc.revoke(sa.id, org_id=ORG_A)

    with pytest.raises(LookupError):
        await svc.issue_credential(sa.id, org_id=ORG_A, name="too late")


async def test_every_agent_operation_is_scoped_to_the_owning_organization():
    """A row in another tenant is indistinguishable from one that does not exist —
    which is what lets the HTTP edge answer 404 without leaking existence."""
    db = FakeSession()
    svc = ServiceAccountService(db)
    sa, _, _ = await svc.create_agent(org_id=ORG_A, name="a", scopes=[])

    assert await svc.get(sa.id, org_id=ORG_B) is None
    assert await svc.get(sa.id, org_id=ORG_A) is not None
    for call in (
        lambda: svc.update(sa.id, org_id=ORG_B, name="seized"),
        lambda: svc.revoke(sa.id, org_id=ORG_B),
        lambda: svc.rotate(sa.id, org_id=ORG_B),
        lambda: svc.issue_credential(sa.id, org_id=ORG_B, name="stolen"),
    ):
        with pytest.raises(LookupError):
            await call()
    assert (await svc.get(sa.id, org_id=ORG_A)).name == "a"
