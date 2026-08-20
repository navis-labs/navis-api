# pyright: reportArgumentType=false, reportOptionalMemberAccess=false, reportIncompatibleMethodOverride=false
# ^ Test-double typing, matching tests/organization/test_organization_v1.py: the
#   org services are annotated for the real OrganizationRepository but are given
#   the structural in-memory FakeOrgRepo, and the tripwire provider narrows its
#   own return type to NoReturn. Both are inherent to the fake-based unit style.
"""Post-approval authorization: approval is intent, not a standing permission.

Every test here drives the REAL `EntitlementService` (over the in-memory org
repository) through the REAL worker path. Nothing about the decision is stubbed —
`_decide` makes every call, exactly as it does in production. The provider is a
tripwire that fails the test if it is ever reached.

The scenario under test:

    T0 org authorized -> T1 request -> T2 human approves
    -> T3 authorization is revoked
    -> T4 worker claims the APPROVED row

If the side effect still happens at T4, the kill switch does not work.
"""
from __future__ import annotations

import uuid

from navis.execution.contract import ProviderReference, ProviderResult
from navis.execution.providers.base import Provider
from navis.execution.retry import ExecutionRetryPolicy
from navis.execution.service import ExecutionService
from navis.execution.status import ExecutionStatus
from navis.execution.worker import execute_pass
from navis.organization.billing import BillingService
from navis.organization.contract import Capability, Plan
from navis.organization.entitlements import EntitlementService
from navis.organization.feature_flags import FeatureFlagService, kill_switch_flag
from navis.organization.lifecycle import LifecycleService
from navis.platform.events import EventType
from tests.execution.fakes import ORG_A, ORG_B, USER, FakeExecutionRepo
from tests.organization.fakes import FakeOrgRepo

FAST = ExecutionRetryPolicy(max_attempts=2, initial_backoff=0.0, max_backoff=0.0, jitter=0.0)

PAYLOAD = {"to": "someone@example.com", "subject": "s", "body": "a real body"}


class _Tripwire(Provider):
    """Reaching `execute()` IS the failure. A dry run would hide the bug, so this
    provider is `is_live` and raises instead."""

    name = "gmail"
    action_type = "send_email"

    def __init__(self) -> None:
        self._sender = self  # is_live -> True
        self.validated = 0
        self.executed = 0

    async def validate(self, payload):
        self.validated += 1

    async def execute(self, payload):
        self.executed += 1
        raise AssertionError(
            "PROVIDER REACHED: an unauthorized organization performed a real "
            "external side effect"
        )

    async def status(self, reference):
        return "sent"


class _Working(_Tripwire):
    """Same shape, but actually succeeds — for the 'authorization restored' leg."""

    async def execute(self, payload):
        self.executed += 1
        return ProviderResult(
            ok=True, reference=ProviderReference("gmail", "real-msg-1")
        )


async def _entitled_org(org_id=ORG_A, plan=Plan.ENTERPRISE):
    """A fully provisioned organization, through the real billing path."""
    org_repo = FakeOrgRepo()
    ent = EntitlementService(org_repo)
    await BillingService(org_repo, ent).set_plan(org_id, plan, billing_authorized=True)
    return org_repo, ent


async def _approved_send(repo, org_id=ORG_A):
    svc = ExecutionService(repo=repo)
    req, _ = await svc.request(
        org_id, uuid.uuid4(), "send_email", PAYLOAD,
        reasoning_signal_ids=["s1"], reasoning_version=1, state_watermark="w",
    )
    await svc.approve(req.id, USER)
    assert req.status == ExecutionStatus.APPROVED.value
    return req


def _event_types(repo):
    return [e.event_type for e in repo.events]


# ── PART 2: post-approval revocation ───────────────────────────────────


async def test_suspended_organization_cannot_execute_an_approved_action():
    repo = FakeExecutionRepo()
    org_repo, ent = await _entitled_org()
    repo.entitlements = ent
    req = await _approved_send(repo)

    # T3: the org is suspended AFTER the human approved.
    await LifecycleService(org_repo).activate(ORG_A)
    await LifecycleService(org_repo).suspend(ORG_A)

    provider = _Tripwire()
    result = await execute_pass(
        repo, providers={"send_email": provider}, policy=FAST, max_attempts=3
    )

    assert provider.executed == 0
    assert provider.validated == 0, "validation must not run either — no provider contact"
    assert result["failed"] == 1
    assert (await repo.get(req.id)).status == ExecutionStatus.FAILED.value


async def test_revoking_the_blanket_execution_capability_stops_queued_work():
    repo = FakeExecutionRepo()
    _org_repo, ent = await _entitled_org()
    repo.entitlements = ent
    await _approved_send(repo)

    await ent.set(ORG_A, Capability.EXECUTION, False)

    provider = _Tripwire()
    await execute_pass(repo, providers={"send_email": provider}, policy=FAST, max_attempts=3)
    assert provider.executed == 0


async def test_revoking_only_the_connector_capability_stops_queued_work():
    """EXECUTION is still held. GMAIL is not. The blanket gate alone would have
    let this through — that is the gap this check closes."""
    repo = FakeExecutionRepo()
    _org_repo, ent = await _entitled_org()
    repo.entitlements = ent
    req = await _approved_send(repo)

    await ent.set(ORG_A, Capability.GMAIL, False)
    assert await ent.can_use(ORG_A, Capability.EXECUTION) is True

    provider = _Tripwire()
    await execute_pass(repo, providers={"send_email": provider}, policy=FAST, max_attempts=3)

    assert provider.executed == 0
    row = await repo.get(req.id)
    assert row.status == ExecutionStatus.FAILED.value
    assert "gmail" in (row.last_error or ""), "the audit trail must name the capability"


async def test_the_approval_record_survives_an_authorization_denial():
    """Denial is not disapproval. The human's decision stays on the record —
    what changed is the organization's current authorization, not their intent."""
    repo = FakeExecutionRepo()
    _org_repo, ent = await _entitled_org()
    repo.entitlements = ent
    req = await _approved_send(repo)
    approved_by, approved_at = req.approved_by, req.approved_at

    await ent.set(ORG_A, Capability.GMAIL, False)
    await execute_pass(repo, providers={"send_email": _Tripwire()}, policy=FAST, max_attempts=3)

    row = await repo.get(req.id)
    assert row.approved_by == approved_by == USER
    assert row.approved_at == approved_at


# ── PART 3: the kill switch actually stops queued side effects ─────────


async def test_kill_switch_stops_an_already_approved_action_then_releases_it():
    """The whole point of a kill switch: thrown during an incident, it must stop
    work that is ALREADY approved and sitting in the queue — and lifting it must
    let that work resume without an operator replay."""
    repo = FakeExecutionRepo()
    org_repo, ent = await _entitled_org()
    repo.entitlements = ent
    req = await _approved_send(repo)
    flags = FeatureFlagService(org_repo)

    # Incident: kill the connector globally (org_id=None is the global switch).
    await flags.set(None, kill_switch_flag(Capability.GMAIL.value), kind="kill_switch", enabled=True)

    blocked = _Tripwire()
    await execute_pass(repo, providers={"send_email": blocked}, policy=FAST, max_attempts=3)
    assert blocked.executed == 0
    row = await repo.get(req.id)
    assert row.status == ExecutionStatus.FAILED.value
    assert row.attempts == 1, "one attempt spent, retry budget intact"

    # Incident over. No re-approval, no replay — the row is still claimable.
    await flags.set(None, kill_switch_flag(Capability.GMAIL.value), kind="kill_switch", enabled=False)

    working = _Working()
    result = await execute_pass(repo, providers={"send_email": working}, policy=FAST, max_attempts=3)

    assert working.executed == 1
    assert result["executed"] == 1
    row = await repo.get(req.id)
    assert row.status == ExecutionStatus.EXECUTED.value
    assert row.provider_reference == {"provider": "gmail", "external_id": "real-msg-1"}


async def test_a_persistently_unauthorized_action_dead_letters_instead_of_spinning():
    """Bounded. A revocation that is never lifted must terminate at the DLQ with
    a stated reason, not re-check forever."""
    repo = FakeExecutionRepo()
    _org_repo, ent = await _entitled_org()
    repo.entitlements = ent
    req = await _approved_send(repo)
    await ent.set(ORG_A, Capability.GMAIL, False)

    provider = _Tripwire()
    for _ in range(4):
        await execute_pass(repo, providers={"send_email": provider}, policy=FAST, max_attempts=2)

    assert provider.executed == 0
    assert (await repo.get(req.id)).attempts >= 2
    letters = [d for d in repo.dead_letters if d.execution_id == req.id]
    assert len(letters) == 1, "exactly one dead letter, not one per pass"
    assert "no longer authorized" in letters[0].error


# ── CASE D: the GLOBAL execution kill switch ───────────────────────────


async def test_kill_execution_stops_every_queued_action_regardless_of_connector():
    """`kill:execution` is the blast-radius lever: one flag, all side effects,
    every connector. Distinct from `kill:gmail`, which stops one connector."""
    repo = FakeExecutionRepo()
    org_repo, ent = await _entitled_org()
    repo.entitlements = ent
    req = await _approved_send(repo)

    await FeatureFlagService(org_repo).set(
        None, kill_switch_flag(Capability.EXECUTION.value), kind="kill_switch", enabled=True
    )

    provider = _Tripwire()
    await execute_pass(repo, providers={"send_email": provider}, policy=FAST, max_attempts=3)

    assert provider.executed == 0
    row = await repo.get(req.id)
    assert row.status == ExecutionStatus.FAILED.value
    assert "execution" in (row.last_error or "")


# ── Enterprise scenario 3: plan downgrade after approval ───────────────


async def test_a_plan_downgrade_after_approval_blocks_the_queued_action():
    """TEAM includes EXECUTION; FREE does not. A downgrade between approval and
    the worker pass must stop the side effect — billing state IS authorization."""
    repo = FakeExecutionRepo()
    org_repo = FakeOrgRepo()
    ent = EntitlementService(org_repo)
    billing = BillingService(org_repo, ent)
    await billing.set_plan(ORG_A, Plan.TEAM, billing_authorized=True)
    repo.entitlements = ent
    req = await _approved_send(repo)
    assert await ent.can_use(ORG_A, Capability.EXECUTION) is True

    await billing.set_plan(ORG_A, Plan.FREE, billing_authorized=True)

    provider = _Tripwire()
    await execute_pass(repo, providers={"send_email": provider}, policy=FAST, max_attempts=3)

    assert provider.executed == 0
    assert (await repo.get(req.id)).status == ExecutionStatus.FAILED.value


# ── Enterprise scenario 4: one connector revoked, another untouched ────


async def test_revoking_slack_blocks_slack_and_leaves_gmail_alone():
    """The capability comes from the ActionSpec, never from a mapping inside the
    worker: revoking SLACK stops `send_slack` and leaves `send_email` (whose spec
    names GMAIL) running."""
    repo = FakeExecutionRepo()
    _org_repo, ent = await _entitled_org()
    repo.entitlements = ent

    svc = ExecutionService(repo=repo)
    slack_req, _ = await svc.request(
        ORG_A, uuid.uuid4(), "send_slack", {"channel": "#general", "text": "hi"},
        reasoning_signal_ids=[], reasoning_version=1, state_watermark="w",
    )
    await svc.approve(slack_req.id, USER)
    email_req = await _approved_send(repo)

    await ent.set(ORG_A, Capability.SLACK, False)

    class _SlackTripwire(_Tripwire):
        name = "slack"
        action_type = "send_slack"

    slack, email = _SlackTripwire(), _Working()
    await execute_pass(
        repo,
        providers={"send_slack": slack, "send_email": email},
        policy=FAST, max_attempts=3,
    )

    assert slack.executed == 0
    assert (await repo.get(slack_req.id)).status == ExecutionStatus.FAILED.value
    assert email.executed == 1, "revoking SLACK must not affect a GMAIL action"
    assert (await repo.get(email_req.id)).status == ExecutionStatus.EXECUTED.value


# ── Enterprise scenario 8: authorization wins over credential ──────────


async def test_a_denied_execution_performs_zero_credential_lookups():
    """THE hard invariant, measured on the REAL gmail credential resolver slot.

    A perfectly valid Google credential exists. Authorization has been revoked.
    The credential must never be consulted: a token can never imply permission,
    and a denied execution must not even touch the vault.
    """
    from navis.execution import integrations

    lookups: list[uuid.UUID] = []

    async def _counting_resolver(db, org_id):
        lookups.append(org_id)
        return "a-perfectly-valid-credential"

    # Swap the REAL gmail resolver so the count is taken at the actual production
    # boundary rather than at a parallel one built for the test.
    original = integrations._RESOLVERS["gmail"]
    integrations._RESOLVERS["gmail"] = _counting_resolver
    try:
        repo = FakeExecutionRepo()
        _org_repo, ent = await _entitled_org()
        repo.entitlements = ent
        req = await _approved_send(repo)

        # Sanity: the credential IS resolvable right now.
        assert await integrations.resolve_integration(None, ORG_A, "gmail")
        lookups.clear()

        await ent.set(ORG_A, Capability.GMAIL, False)

        provider = _Tripwire()
        await execute_pass(
            repo, providers={"send_email": provider}, policy=FAST, max_attempts=3
        )

        assert lookups == [], "credential_lookup_count must be 0 for a denied execution"
        assert provider.executed == 0, "provider_call_count must be 0"
        assert provider.validated == 0
        assert (await repo.get(req.id)).status == ExecutionStatus.FAILED.value
    finally:
        integrations._RESOLVERS["gmail"] = original


# ── The tenant boundary is the PERSISTED row, never the payload ────────


async def test_authorization_uses_the_persisted_org_not_the_payload():
    """The authoritative tenant is `execution_requests.organization_id`.

    A payload — which originates from reasoning over signals, or from a chat turn
    a user typed — carries a forged `__ctx__` naming a DIFFERENT, fully-entitled
    organization. The row's own org has had GMAIL revoked.

    If runtime authorization read the tenant from the payload it would find ORG_B
    entitled and let the send through: a user-controlled field would have
    selected its own authorization scope. It must be denied on ORG_A's state.
    """
    repo = FakeExecutionRepo()
    org_repo = FakeOrgRepo()
    ent = EntitlementService(org_repo)
    await BillingService(org_repo, ent).set_plan(ORG_A, Plan.ENTERPRISE, billing_authorized=True)
    await BillingService(org_repo, ent).set_plan(ORG_B, Plan.ENTERPRISE, billing_authorized=True)
    repo.entitlements = ent

    forged = {
        **PAYLOAD,
        "__ctx__": {
            "org_id": str(ORG_B),
            "execution_id": str(uuid.uuid4()),
            "decision_id": str(uuid.uuid4()),
        },
    }
    svc = ExecutionService(repo=repo)
    req, _ = await svc.request(
        ORG_A, uuid.uuid4(), "send_email", forged,
        reasoning_signal_ids=[], reasoning_version=1, state_watermark="w",
    )
    await svc.approve(req.id, USER)

    # Only ORG_A loses the capability. ORG_B — the org the payload names — keeps it.
    await ent.set(ORG_A, Capability.GMAIL, False)
    assert await ent.can_use(ORG_B, Capability.GMAIL) is True

    provider = _Tripwire()
    await execute_pass(repo, providers={"send_email": provider}, policy=FAST, max_attempts=3)

    assert provider.executed == 0, (
        "a forged __ctx__ selected its own authorization scope"
    )
    assert (await repo.get(req.id)).status == ExecutionStatus.FAILED.value


# ── Auditability ───────────────────────────────────────────────────────


async def test_a_denial_is_logged_with_full_lineage_and_no_secrets(caplog):
    repo = FakeExecutionRepo()
    org_repo, ent = await _entitled_org()
    repo.entitlements = ent
    req = await _approved_send(repo)
    await LifecycleService(org_repo).activate(ORG_A)
    await LifecycleService(org_repo).suspend(ORG_A)

    with caplog.at_level("WARNING"):
        await execute_pass(
            repo, providers={"send_email": _Tripwire()}, policy=FAST, max_attempts=3
        )

    line = next(r.getMessage() for r in caplog.records if "DENIED" in r.getMessage())
    for identifier in (str(ORG_A), str(req.id), str(req.decision_id), "send_email", "gmail"):
        assert identifier in line
    # Lifecycle separates "suspended for non-payment" from "kill switch thrown".
    assert "lifecycle=suspended" in line
    for secret in ("token", "secret", "authorization:", "bearer"):
        assert secret not in line.lower()


# ── PART 8: an authorization failure is never a success ────────────────


async def test_an_authorization_denial_never_emits_action_executed():
    repo = FakeExecutionRepo()
    _org_repo, ent = await _entitled_org()
    repo.entitlements = ent
    await _approved_send(repo)
    await ent.set(ORG_A, Capability.GMAIL, False)

    await execute_pass(repo, providers={"send_email": _Tripwire()}, policy=FAST, max_attempts=3)

    types = _event_types(repo)
    assert EventType.ACTION_EXECUTED.value not in types
    assert EventType.ACTION_FAILED.value in types
    # Observation keys outcomes off these events, so a denial lands as a real
    # failure outcome with full lineage — never as a silent drop.
    failed = [e for e in repo.events if e.event_type == EventType.ACTION_FAILED.value][0]
    assert failed.organization_id == ORG_A
    assert "no longer authorized" in failed.payload["error"]


# ── PART 10: organization isolation ────────────────────────────────────


async def test_revoking_one_org_does_not_affect_another():
    repo = FakeExecutionRepo()
    org_repo = FakeOrgRepo()
    ent = EntitlementService(org_repo)
    await BillingService(org_repo, ent).set_plan(ORG_A, Plan.ENTERPRISE, billing_authorized=True)
    await BillingService(org_repo, ent).set_plan(ORG_B, Plan.ENTERPRISE, billing_authorized=True)
    repo.entitlements = ent

    req_a = await _approved_send(repo, ORG_A)
    req_b = await _approved_send(repo, ORG_B)

    # Only Org A loses the capability.
    await ent.set(ORG_A, Capability.GMAIL, False)

    provider = _Working()
    await execute_pass(repo, providers={"send_email": provider}, policy=FAST, max_attempts=3)

    assert (await repo.get(req_a.id)).status == ExecutionStatus.FAILED.value
    assert (await repo.get(req_b.id)).status == ExecutionStatus.EXECUTED.value
    assert provider.executed == 1, "exactly one org reached the provider"


async def test_an_orgs_kill_switch_does_not_kill_another_orgs_work():
    repo = FakeExecutionRepo()
    org_repo = FakeOrgRepo()
    ent = EntitlementService(org_repo)
    await BillingService(org_repo, ent).set_plan(ORG_A, Plan.ENTERPRISE, billing_authorized=True)
    await BillingService(org_repo, ent).set_plan(ORG_B, Plan.ENTERPRISE, billing_authorized=True)
    repo.entitlements = ent
    req_a = await _approved_send(repo, ORG_A)
    req_b = await _approved_send(repo, ORG_B)

    # Org-scoped kill switch — must not leak to the other tenant.
    await FeatureFlagService(org_repo).set(
        ORG_A, kill_switch_flag(Capability.GMAIL.value), kind="kill_switch", enabled=True
    )

    await execute_pass(repo, providers={"send_email": _Working()}, policy=FAST, max_attempts=3)

    assert (await repo.get(req_a.id)).status == ExecutionStatus.FAILED.value
    assert (await repo.get(req_b.id)).status == ExecutionStatus.EXECUTED.value


# ── PART 7: the gate protects the REAL production provider ─────────────


def test_the_production_gmail_provider_is_live_and_wired_to_the_real_sender():
    """Not a test double. `production_providers()` is what the Celery worker
    passes to `execute_pass`, and its send_email provider must hold the actual
    `gmail_sender` — the function that resolves the org's integration, encodes
    the RFC 2822 message and POSTs it through GoogleTransport."""
    from navis.execution.providers import production_providers
    from navis.execution.providers._senders import gmail_sender

    provider = production_providers()["send_email"]
    assert provider.is_live
    assert provider._sender is gmail_sender
    assert not provider.unsupported


async def test_an_unauthorized_org_cannot_reach_the_real_production_sender():
    """The strongest form of the guarantee: the registry here is the REAL one,
    with the REAL Google senders wired in. If the gate leaked, `gmail_sender`
    would run — resolve an integration, decrypt tokens, and POST to Gmail.

    It never gets that far, and the proof is that no database session is ever
    opened: `gmail_sender`'s first act is `session_scope()`, and there is no
    database configured in this process for it to reach.
    """
    from navis.execution.providers import production_providers

    repo = FakeExecutionRepo()
    _org_repo, ent = await _entitled_org()
    repo.entitlements = ent
    req = await _approved_send(repo)

    await ent.set(ORG_A, Capability.GMAIL, False)

    result = await execute_pass(
        repo, providers=production_providers(), policy=FAST, max_attempts=3
    )

    assert result["executed"] == 0
    row = await repo.get(req.id)
    assert row.status == ExecutionStatus.FAILED.value
    assert "no longer authorized" in row.last_error
    # No provider_reference at all — nothing was attempted, let alone simulated.
    assert not row.provider_reference


def test_no_production_provider_can_report_a_dry_run():
    """A dry-run reference from the production registry would surface to
    Observation as a real success. `_production_safe` withholds any provider that
    could produce one; this asserts the property that guard exists for."""
    from navis.execution.providers import production_providers

    for action_type, provider in production_providers().items():
        assert provider.is_live or provider.unsupported, action_type


# ── PART 9: the re-check does not weaken idempotency ───────────────────


async def test_a_receipt_from_a_previous_attempt_still_closes_out_after_revocation():
    """A row carrying a provider reference has ALREADY had its side effect. That
    is bookkeeping, not a new action, so it must still be recorded — refusing it
    would lose the receipt and let a later pass re-run the provider, which is the
    duplicate-send the reference exists to prevent."""
    repo = FakeExecutionRepo()
    _org_repo, ent = await _entitled_org()
    repo.entitlements = ent
    req = await _approved_send(repo)
    req.provider_reference = {"provider": "gmail", "external_id": "already-sent-1"}

    await ent.set(ORG_A, Capability.GMAIL, False)

    provider = _Tripwire()
    result = await execute_pass(
        repo, providers={"send_email": provider}, policy=FAST, max_attempts=3
    )

    assert provider.executed == 0, "the provider is NOT re-called"
    assert result["executed"] == 1
    row = await repo.get(req.id)
    assert row.status == ExecutionStatus.EXECUTED.value
    assert row.provider_reference["external_id"] == "already-sent-1"


async def test_the_recheck_costs_one_query_per_execution():
    """Both capabilities resolve in a SINGLE round trip. Asking per-capability
    would put an extra Neon RTT in front of every side effect."""
    repo = FakeExecutionRepo()
    org_repo, ent = await _entitled_org()
    repo.entitlements = ent
    await _approved_send(repo)

    calls: list[list[str]] = []
    original = org_repo.snapshot_capabilities

    async def _counting(org_id, capabilities):
        calls.append(list(capabilities))
        return await original(org_id, capabilities)

    org_repo.snapshot_capabilities = _counting
    await execute_pass(repo, providers={"send_email": _Working()}, policy=FAST, max_attempts=3)

    assert len(calls) == 1, f"expected one batched read, got {len(calls)}"
    assert set(calls[0]) == {"execution", "gmail"}
