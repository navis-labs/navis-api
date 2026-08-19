# pyright: reportArgumentType=false, reportOptionalMemberAccess=false
"""The generic integration / credential boundary.

Two things are being protected here:

  1. CREDENTIAL ISOLATION — a credential is only ever resolved under exactly one
     organization, and nothing a Decision or a payload says can change which one.
  2. PROVIDER NEUTRALITY — the generic Execution layer (worker, router, service,
     action registry) contains no connector authentication, so a new connector is
     a resolver registration and nothing else.
"""
from __future__ import annotations

import ast
import pathlib
import uuid

import pytest

from navis.execution.actions import (
    ActionRisk,
    ActionSpec,
    ApprovalPolicy,
    register_action,
    unregister_action,
)
from navis.execution.contract import ProviderReference, ProviderResult
from navis.execution.integrations import (
    IntegrationUnavailable,
    has_integration_resolver,
    register_integration_resolver,
    resolve_integration,
)
from navis.execution.providers.base import Provider
from navis.execution.retry import ExecutionRetryPolicy
from navis.execution.service import ExecutionService
from navis.execution.status import ExecutionStatus
from navis.execution.worker import execute_pass
from navis.organization.billing import BillingService
from navis.organization.contract import Capability, Plan
from navis.organization.entitlements import EntitlementService
from navis.platform.events import EventType
from tests.execution.fakes import ORG_A, ORG_B, USER, FakeExecutionRepo
from tests.organization.fakes import FakeOrgRepo

EXECUTION_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "navis" / "execution"
FAST = ExecutionRetryPolicy(max_attempts=1, initial_backoff=0.0, max_backoff=0.0, jitter=0.0)


# ── PART 12: dependency direction ──────────────────────────────────────

# Modules that make up the GENERIC execution engine. Connector authentication
# living in any of them is the defect this task removes: each new connector
# would have to edit every one.
_GENERIC = ("worker.py", "router.py", "service.py", "actions.py", "summary.py",
            "repository.py", "integrations.py", "retry.py", "recovery.py",
            "dead_letter.py", "status.py", "contract.py")

# Signals modules that implement provider AUTHENTICATION. Read-model helpers
# (calendar reconcile, draft thread context) are a different dependency and are
# deliberately not covered by this rule.
_CREDENTIAL_MODULES = ("navis.signals.google.resolve", "navis.signals.google.oauth",
                       "navis.signals.crypto", "navis.signals.google.transport")


def _imports(path: pathlib.Path) -> set[str]:
    tree = ast.parse(path.read_text())
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            out.add(node.module)
        elif isinstance(node, ast.Import):
            out.update(a.name for a in node.names)
    return out


def test_generic_execution_contains_no_connector_authentication():
    violations = []
    for name in _GENERIC:
        for mod in _imports(EXECUTION_DIR / name):
            if mod in _CREDENTIAL_MODULES:
                violations.append(f"{name} imports {mod}")
    assert not violations, (
        "generic Execution must reach credentials only through "
        "navis.execution.integrations:\n" + "\n".join(violations)
    )


def test_the_google_adapter_is_the_only_credential_implementation():
    """Exactly one module in Execution may know Google authentication exists."""
    owners = sorted(
        p.relative_to(EXECUTION_DIR).as_posix()
        for p in EXECUTION_DIR.rglob("*.py")
        if _imports(p) & set(_CREDENTIAL_MODULES)
    )
    assert owners == ["providers/_senders.py"], owners


# ── PART 3: organization isolation ─────────────────────────────────────


async def test_credential_resolution_without_an_organization_fails_closed():
    with pytest.raises(IntegrationUnavailable):
        await resolve_integration(object(), None, "gmail")


async def test_an_unregistered_provider_cannot_resolve_a_credential():
    """A provider with no registered credential authority cannot act.

    Slack used to be the example here; it now has a resolver (5.4.1), which is
    itself the proof that the extension point works. The invariant is unchanged
    and is stated against a provider that genuinely has no authority — every
    future connector is in exactly this state until its resolver ships.
    """
    assert has_integration_resolver("slack"), "slack shipped its resolver in 5.4.1"
    assert not has_integration_resolver("notion")
    with pytest.raises(IntegrationUnavailable, match="no credential resolver"):
        await resolve_integration(object(), ORG_A, "notion")


@pytest.fixture()
def synthetic_connector():
    """Register a connector the way a future one will: one ActionSpec, one
    credential resolver. No generic Execution code is touched."""
    seen: list[uuid.UUID] = []
    vault = {ORG_A: "token-for-org-A", ORG_B: "token-for-org-B"}

    async def _resolver(db, org_id):
        seen.append(org_id)
        if org_id not in vault:
            raise IntegrationUnavailable("not connected", provider="synthetic")
        return vault[org_id]

    register_integration_resolver("synthetic", _resolver)
    register_action(ActionSpec(
        action_type="synthetic_write", provider="synthetic",
        capability=Capability.EXECUTION, risk=ActionRisk.EXTERNAL_WRITE,
        approval=ApprovalPolicy.REQUIRED,
    ))
    try:
        yield seen
    finally:
        unregister_action("synthetic_write")
        from navis.execution.integrations import _RESOLVERS

        _RESOLVERS.pop("synthetic", None)


class _CredentialProvider(Provider):
    """A sender that resolves its credential through the boundary and records
    which one it got — the shape every real sender uses."""

    name = "synthetic"
    action_type = "synthetic_write"

    def __init__(self) -> None:
        self._sender = self
        self.used: list[tuple[uuid.UUID, str]] = []

    async def validate(self, payload):
        pass

    async def execute(self, payload):
        org_id = uuid.UUID(payload["__ctx__"]["org_id"])
        credential = await resolve_integration(object(), org_id, "synthetic")
        self.used.append((org_id, credential))
        return ProviderResult(ok=True, reference=ProviderReference("synthetic", "s-1"))

    async def status(self, reference):
        return "ok"


async def _entitled(*orgs):
    org_repo = FakeOrgRepo()
    ent = EntitlementService(org_repo)
    for org in orgs:
        await BillingService(org_repo, ent).set_plan(org, Plan.ENTERPRISE)
    return ent


async def _approved(repo, org_id, payload=None):
    svc = ExecutionService(repo=repo)
    req, _ = await svc.request(
        org_id, uuid.uuid4(), "synthetic_write", payload or {"record": "r1"},
        reasoning_signal_ids=[], reasoning_version=1, state_watermark="w",
    )
    await svc.approve(req.id, USER)
    return req


async def test_each_organization_resolves_only_its_own_credential(synthetic_connector):
    repo = FakeExecutionRepo()
    repo.entitlements = await _entitled(ORG_A, ORG_B)
    await _approved(repo, ORG_A)
    await _approved(repo, ORG_B)

    provider = _CredentialProvider()
    await execute_pass(repo, providers={"synthetic_write": provider}, policy=FAST)

    assert sorted(provider.used) == sorted(
        [(ORG_A, "token-for-org-A"), (ORG_B, "token-for-org-B")]
    )
    assert sorted(synthetic_connector) == sorted([ORG_A, ORG_B])


async def test_a_payload_cannot_choose_which_organizations_credential_is_used(
    synthetic_connector,
):
    """THE cross-tenant attack. A Decision payload — which originates from
    reasoning over signals, or from a chat turn a user typed — carries a forged
    `__ctx__` naming another organization. The worker stamps the execution
    context from the CLAIMED ROW, after the payload, so the row's org always
    wins. Natural language can never select credentials.
    """
    repo = FakeExecutionRepo()
    repo.entitlements = await _entitled(ORG_A, ORG_B)
    forged = {
        "record": "r1",
        "__ctx__": {
            "org_id": str(ORG_B),
            "execution_id": str(uuid.uuid4()),
            "decision_id": str(uuid.uuid4()),
        },
    }
    req = await _approved(repo, ORG_A, forged)

    provider = _CredentialProvider()
    await execute_pass(repo, providers={"synthetic_write": provider}, policy=FAST)

    assert provider.used == [(ORG_A, "token-for-org-A")]
    assert synthetic_connector == [ORG_A]
    assert req.organization_id == ORG_A


# ── PART 13: authorization BEFORE credential resolution ────────────────


async def test_an_unauthorized_org_never_reaches_credential_resolution(
    synthetic_connector,
):
    """Authentication is not authorization, and the order proves it: a revoked
    org is refused by the entitlement re-check, so no credential lookup happens
    at all. A credential's existence can never imply permission.
    """
    repo = FakeExecutionRepo()
    org_repo = FakeOrgRepo()
    ent = EntitlementService(org_repo)
    await BillingService(org_repo, ent).set_plan(ORG_A, Plan.ENTERPRISE)
    repo.entitlements = ent
    req = await _approved(repo, ORG_A)

    await ent.set(ORG_A, Capability.EXECUTION, False)

    provider = _CredentialProvider()
    await execute_pass(repo, providers={"synthetic_write": provider}, policy=FAST)

    assert provider.used == []
    assert synthetic_connector == [], "no credential may be touched before authorization"
    assert (await repo.get(req.id)).status == ExecutionStatus.FAILED.value


# ── PART 6: credential lifecycle ───────────────────────────────────────


async def test_a_missing_credential_fails_permanently_and_never_succeeds(
    synthetic_connector,
):
    """Authorized, but not connected. Must fail BEFORE any side effect, must not
    be reported as success, and must not burn the retry budget rediscovering that
    the account is still not connected."""
    # Fully ENTITLED, but absent from the connector's credential vault — the
    # org bought the capability and never connected the account.
    unconnected = uuid.UUID("99999999-9999-9999-9999-999999999999")
    repo = FakeExecutionRepo()
    repo.entitlements = await _entitled(unconnected)
    req = await _approved(repo, unconnected)

    provider = _CredentialProvider()
    result = await execute_pass(
        repo, providers={"synthetic_write": provider}, policy=FAST, max_attempts=3
    )

    assert result["executed"] == 0
    row = await repo.get(req.id)
    assert row.status == ExecutionStatus.FAILED.value
    assert not row.provider_reference, "no receipt for something that never happened"
    # Permanent: attempts exhausted in one pass, dead-lettered immediately.
    assert row.attempts >= 3
    assert len(repo.dead_letters) == 1
    assert "integration" in repo.dead_letters[0].error
    # Never a success event.
    assert EventType.ACTION_EXECUTED.value not in [e.event_type for e in repo.events]


# ── PART 7: no credential leakage ──────────────────────────────────────


async def test_no_credential_material_reaches_the_row_outbox_or_dead_letter(
    synthetic_connector,
):
    """Everything written here is read by Observation, the dashboard and the DLQ
    console. A token in any of them is a breach."""
    repo = FakeExecutionRepo()
    repo.entitlements = await _entitled(ORG_A)
    req = await _approved(repo, ORG_A)

    provider = _CredentialProvider()
    await execute_pass(repo, providers={"synthetic_write": provider}, policy=FAST)

    secret = "token-for-org-A"
    row = await repo.get(req.id)
    surfaces = [
        str(row.payload), str(row.provider_reference), str(row.last_error),
        str([e.payload for e in repo.events]),
        str([d.payload for d in repo.dead_letters]),
        str([d.error for d in repo.dead_letters]),
    ]
    for surface in surfaces:
        assert secret not in surface
        assert "access_token" not in surface
        assert "refresh_token" not in surface
    # The credential WAS used — this is not passing by never resolving one.
    assert provider.used == [(ORG_A, secret)]


def test_the_execution_context_carries_no_credential_fields():
    """`__ctx__` is what the worker threads into every payload. It must carry
    identifiers only — adding a token here would put one in the row, the outbox
    and every Observation event at once."""
    source = (EXECUTION_DIR / "worker.py").read_text()
    ctx = source[source.index('"__ctx__": {'):source.index('"__ctx__": {') + 260]
    for forbidden in ("token", "secret", "credential", "authorization", "Bearer"):
        assert forbidden not in ctx.lower()
