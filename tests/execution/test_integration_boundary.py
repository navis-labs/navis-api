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
    tree = ast.parse(path.read_text(encoding="utf-8"))
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
        await BillingService(org_repo, ent).set_plan(org, Plan.ENTERPRISE, billing_authorized=True)
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
    await BillingService(org_repo, ent).set_plan(ORG_A, Plan.ENTERPRISE, billing_authorized=True)
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
    source = (EXECUTION_DIR / "worker.py").read_text(encoding="utf-8")
    ctx = source[source.index('"__ctx__": {'):source.index('"__ctx__": {') + 260]
    for forbidden in ("token", "secret", "credential", "authorization", "Bearer"):
        assert forbidden not in ctx.lower()


# ── T-11: an execution binds to ONE connected account ───────────────────────
#
# `organization_id` says WHOSE action this is; it never said which of that
# organization's accounts to act AS. With Gmail A and Gmail B both connected, the
# credential came from (org, provider) and the tie was broken by RECENCY — so a
# reply to a message that arrived at B could be sent from A. Right tenant, wrong
# identity, wrong thread, and no error anywhere.


class _PinnedIntegration:
    def __init__(
        self,
        *,
        org_id,
        provider="gmail",
        active=True,
        scopes="https://www.googleapis.com/auth/gmail.send",
        access_token="enc:1:at",
        refresh_token="enc:1:rt",
    ):
        self.id = uuid.uuid4()
        self.organization_id = org_id
        self.provider = provider
        self.is_active = active
        self.external_account = "founder@navislabs.in"
        self.access_token = access_token
        self.refresh_token = refresh_token
        self.token_expires_at = None
        self.scopes = scopes


class _PinDB:
    """`db.get(Integration, id)` over a fixed set of rows."""

    def __init__(self, rows):
        self._rows = {r.id: r for r in rows}
        self.gets = 0

    async def get(self, _model, row_id):
        self.gets += 1
        return self._rows.get(row_id)


ORG_X = uuid.UUID("aaaaaaaa-0000-0000-0000-000000000001")
ORG_Y = uuid.UUID("bbbbbbbb-0000-0000-0000-000000000002")


async def test_a_pinned_account_is_returned_exactly():
    row = _PinnedIntegration(org_id=ORG_X)
    got = await resolve_integration(_PinDB([row]), ORG_X, "gmail", integration_id=row.id)
    assert got is row


async def test_a_pinned_account_from_another_org_is_refused():
    """THE CROSS-TENANT CASE. Ownership is re-checked against the org the WORKER
    read off the persisted execution row, so a pinned id belonging to another
    tenant resolves to nothing — and says the same thing a non-existent id says,
    so an execution cannot probe another organization's integration ids."""
    theirs = _PinnedIntegration(org_id=ORG_Y)
    db = _PinDB([theirs])

    with pytest.raises(IntegrationUnavailable) as exc:
        await resolve_integration(db, ORG_X, "gmail", integration_id=theirs.id)
    owned_msg = str(exc.value)

    with pytest.raises(IntegrationUnavailable) as exc2:
        await resolve_integration(db, ORG_X, "gmail", integration_id=uuid.uuid4())
    assert str(exc2.value) == owned_msg, "the error distinguishes 'yours' from 'exists'"


async def test_a_pinned_account_never_falls_back_to_another_account():
    """The whole point: an unavailable PIN must refuse, not silently resolve the
    org's other account. A wrong-account send is not fixed by a retry."""
    gone = uuid.uuid4()
    other = _PinnedIntegration(org_id=ORG_X)  # same org, would satisfy the unpinned path

    with pytest.raises(IntegrationUnavailable):
        await resolve_integration(_PinDB([other]), ORG_X, "gmail", integration_id=gone)


async def test_a_pinned_account_of_the_wrong_family_is_refused():
    slack_row = _PinnedIntegration(org_id=ORG_X, provider="slack")
    with pytest.raises(IntegrationUnavailable, match="cannot perform a gmail action"):
        await resolve_integration(
            _PinDB([slack_row]), ORG_X, "gmail", integration_id=slack_row.id
        )


async def test_a_sibling_google_row_is_accepted_for_a_gmail_action():
    """One Google consent writes gmail/google_calendar/drive rows that share a
    token and an address, so pinning any of them pins the same real account."""
    for family in ("gmail", "google_calendar", "drive"):
        row = _PinnedIntegration(org_id=ORG_X, provider=family)
        got = await resolve_integration(
            _PinDB([row]), ORG_X, "gmail", integration_id=row.id
        )
        assert got is row, f"{family} row rejected for a gmail action"


async def test_a_disconnected_pinned_account_is_refused_with_reconsent():
    row = _PinnedIntegration(org_id=ORG_X, active=False)
    with pytest.raises(IntegrationUnavailable) as exc:
        await resolve_integration(_PinDB([row]), ORG_X, "gmail", integration_id=row.id)
    assert exc.value.reconsent_required is True


async def test_a_pin_still_requires_an_organization():
    """The pin does not bypass the org requirement — both are checked."""
    row = _PinnedIntegration(org_id=ORG_X)
    with pytest.raises(IntegrationUnavailable, match="requires an organization"):
        await resolve_integration(_PinDB([row]), None, "gmail", integration_id=row.id)


def test_the_worker_takes_the_pin_from_the_persisted_row_not_the_payload():
    """`__ctx__` is written AFTER the payload spread, so a payload key cannot
    supply its own account any more than it can supply its own organization."""
    import inspect

    from navis.execution import worker as W

    src = inspect.getsource(W._execute_one)
    assert "integration_id=getattr(req, \"integration_id\", None)" in src

    run_src = inspect.getsource(W._run_leased)
    assert "**lease.payload" in run_src
    assert run_src.index("**lease.payload") < run_src.index('"__ctx__"')


def test_the_execution_row_records_the_account():
    """The column exists, is nullable (historical rows recorded no account and
    inventing one would fabricate lineage), and is indexed for the ownership
    re-check."""
    from navis.execution.models import ExecutionRequest

    col = ExecutionRequest.__table__.columns["integration_id"]
    assert col.nullable is True
    assert col.index is True
    assert [fk.target_fullname for fk in col.foreign_keys] == ["integrations.id"]
    assert next(iter(col.foreign_keys)).ondelete == "SET NULL"


async def test_intake_persists_the_pin():
    import inspect

    from navis.execution.service import ExecutionService as ES

    assert "integration_id" in inspect.signature(ES.request).parameters
    src = inspect.getsource(ES.request)
    assert "integration_id=integration_id" in src, "accepted but never persisted"


# ── PINNING MUST NEVER BE THE WEAKER PATH (T-11) ───────────────────────
#
# `_ACCEPTED` answers a STRUCTURAL question — may a Google-family row serve a
# gmail action? — because one consent writes gmail/google_calendar/drive rows
# that share a token. It says nothing about whether the grant carried the scope
# the action needs, or whether the row still holds tokens.
#
# The org-scoped resolvers have always asked both: `resolve_gmail_send_integration`
# filters on `gmail_send_granted` AND on stored tokens before returning anything.
# `_resolve_pinned` asked neither, so recording an account SKIPPED the gate that
# not recording one enforced. Reproduced before the fix: a `gmail.readonly` row
# and a token-less row BOTH resolved when pinned while the unpinned resolver
# refused each with `reconsent_required=True`.
#
# The damage is not a wrong account — the pin prevents that — it is losing the
# permanent, actionable refusal ("reconnect Google to grant send access") and
# getting an opaque provider 403 at send time instead, which the worker
# classifies TRANSIENT and retries the full budget.


async def test_a_pinned_gmail_account_without_send_scope_is_refused():
    read_only = _PinnedIntegration(
        org_id=ORG_X, scopes="https://www.googleapis.com/auth/gmail.readonly"
    )
    with pytest.raises(IntegrationUnavailable) as exc:
        await resolve_integration(
            _PinDB([read_only]), ORG_X, "gmail", integration_id=read_only.id
        )
    # PERMANENT + actionable, exactly as the unpinned resolver reports it.
    assert exc.value.reconsent_required is True
    assert "send" in str(exc.value).lower()


async def test_a_pinned_gmail_account_without_tokens_is_refused():
    tokenless = _PinnedIntegration(org_id=ORG_X, access_token=None, refresh_token=None)
    with pytest.raises(IntegrationUnavailable) as exc:
        await resolve_integration(
            _PinDB([tokenless]), ORG_X, "gmail", integration_id=tokenless.id
        )
    assert exc.value.reconsent_required is True


async def test_a_pinned_google_account_without_a_calendar_scope_is_refused():
    """A gmail row is structurally acceptable for a calendar action (same
    consent), so `_ACCEPTED` lets it through — the SCOPE is what decides."""
    gmail_only = _PinnedIntegration(
        org_id=ORG_X, scopes="https://www.googleapis.com/auth/gmail.send"
    )
    with pytest.raises(IntegrationUnavailable) as exc:
        await resolve_integration(
            _PinDB([gmail_only]), ORG_X, "calendar", integration_id=gmail_only.id
        )
    assert exc.value.reconsent_required is True

    capable = _PinnedIntegration(
        org_id=ORG_X, scopes="https://www.googleapis.com/auth/calendar.events"
    )
    got = await resolve_integration(
        _PinDB([capable]), ORG_X, "calendar", integration_id=capable.id
    )
    assert got is capable


async def test_a_pinned_slack_install_without_write_scope_is_refused():
    read_only = _PinnedIntegration(
        org_id=ORG_X, provider="slack", scopes="channels:history"
    )
    with pytest.raises(IntegrationUnavailable) as exc:
        await resolve_integration(
            _PinDB([read_only]), ORG_X, "slack", integration_id=read_only.id
        )
    assert exc.value.reconsent_required is True


async def test_pinning_is_never_more_permissive_than_not_pinning():
    """The invariant behind the three tests above, stated once.

    For every row the org-scoped resolver REFUSES, pinning that same row must
    also refuse. Anything else means recording the account bought a weaker
    credential check than leaving it NULL — which would make the T-11 producer a
    regression rather than a fix.
    """
    from navis.signals.google import resolve as gresolve

    class _Scalars:
        def __init__(self, rows):
            self._rows = rows

        async def scalars(self, _stmt):
            class _R:
                def __init__(s, rows):
                    s._r = rows

                def all(s):
                    return s._r

            return _R(self._rows)

        async def get(self, _model, row_id):
            return {r.id: r for r in self._rows}.get(row_id)

    refused_by_unpinned = [
        _PinnedIntegration(
            org_id=ORG_X, scopes="https://www.googleapis.com/auth/gmail.readonly"
        ),
        _PinnedIntegration(org_id=ORG_X, access_token=None, refresh_token=None),
    ]
    for row in refused_by_unpinned:
        db = _Scalars([row])
        with pytest.raises(RuntimeError):  # GmailSendScopeInsufficient / NotConnected
            await gresolve.resolve_gmail_send_integration(db, ORG_X)
        with pytest.raises(IntegrationUnavailable):
            await resolve_integration(db, ORG_X, "gmail", integration_id=row.id)


# ── THE PRODUCER SIDE OF T-11 ──────────────────────────────────────────
#
# The column, the lease, the `__ctx__` field and the pinning resolver all shipped
# together; no PRODUCER ever wrote a value, so `integration_id` was uniformly
# NULL and every send fell back to resolving by (organization, provider).
#
# That fallback is fail-closed, which is precisely why the gap mattered: the
# resolvers refuse a multi-account organization with "recreate the action from
# the message it answers" — and with no producer recording an account, recreating
# it produced another NULL. The refusal had no remedy, so every proactive reply
# for such an organization was permanently unsendable.


async def test_the_proactive_producer_records_the_account_it_derived():
    """`propose_actions` threads the thread's own capturing account into intake.

    The id is RE-DERIVED from persisted memory lineage (proved against real
    Postgres in `tests/integration/test_memory_postgres.py`) — never read from a
    payload, which is the same rule the worker applies to the tenant.
    """
    from navis.orchestration import brief as brief_module

    org = uuid.uuid4()
    account = uuid.uuid4()
    captured: dict = {}

    class _Brief:
        decisions = [
            {
                "decision_id": str(uuid.uuid4()),
                "category": "waiting_reply",
                "title": "They are waiting on a reply",
                "reasoning_signal_ids": [],
                "reasoning_version": 1,
                "state_watermark": "w-1",
                "evidence": {
                    "person_email": "john@partner.com",
                    "thread_id": "thread-release",
                    "rfc_message_id": "<m1@partner.com>",
                    "subject": "Product Release",
                },
            }
        ]

    class _Decisions:
        def __init__(self, _db):
            pass

        async def latest_brief(self, _org):
            return _Brief()

    class _Svc:
        def __init__(self, _db):
            pass

        async def request(self, *args, **kwargs):
            captured.update(kwargs)
            return object(), True

    async def _lineage(_db, _org, thread_ids):
        assert list(thread_ids) == ["thread-release"], "thread not read from the payload"
        return {"thread-release": account}

    import navis.decisions.service as decisions_service
    import navis.execution.service as execution_service
    import navis.memory.lineage as lineage_module

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(decisions_service, "DecisionService", _Decisions)
        mp.setattr(execution_service, "ExecutionService", _Svc)
        mp.setattr(lineage_module, "integrations_for_threads", _lineage)
        created = await brief_module.propose_actions(None, org)

    assert created == 1
    assert captured["integration_id"] == account, (
        "the proactive producer dropped the account it had already derived"
    )


async def test_the_producer_records_nothing_when_the_thread_is_ambiguous():
    """FAIL CLOSED, not "pick one". A thread captured on two accounts leaves
    `integration_id` NULL, and the credential boundary then refuses a
    multi-account organization outright rather than guessing an identity."""
    from navis.orchestration import brief as brief_module

    captured: dict = {}

    class _Brief:
        decisions = [
            {
                "decision_id": str(uuid.uuid4()),
                "category": "waiting_reply",
                "title": "t",
                "evidence": {
                    "person_email": "john@partner.com",
                    "thread_id": "thread-shared",
                    "rfc_message_id": "<m1@partner.com>",
                },
            }
        ]

    class _Decisions:
        def __init__(self, _db):
            pass

        async def latest_brief(self, _org):
            return _Brief()

    class _Svc:
        def __init__(self, _db):
            pass

        async def request(self, *args, **kwargs):
            captured.update(kwargs)
            return object(), True

    async def _ambiguous(_db, _org, _threads):
        return {}  # two accounts hold this thread -> the helper omits it

    import navis.decisions.service as decisions_service
    import navis.execution.service as execution_service
    import navis.memory.lineage as lineage_module

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(decisions_service, "DecisionService", _Decisions)
        mp.setattr(execution_service, "ExecutionService", _Svc)
        mp.setattr(lineage_module, "integrations_for_threads", _ambiguous)
        await brief_module.propose_actions(None, uuid.uuid4())

    assert captured["integration_id"] is None


# ── The API read surfaces resolve credentials through the SAME authority ──


async def test_the_calendar_availability_surface_refuses_an_ambiguous_org():
    """`/workspace/calendar/available-slots` must not rank connected accounts.

    It used to run its own query — provider=GOOGLE_CALENDAR, is_active, ORDER BY
    created_at DESC, `scalar()` — so an org holding founder@ and ops@ (a shape
    `uq_org_provider_account` permits) silently got whichever was connected last.
    That is the recency tie-break T-11 removed from the execution path, and it
    decided WHOSE free/busy the founder picks a slot from.

    The canonical resolver refuses the ambiguity instead. Asserting on the 409
    rather than on the query shape means a future rewrite of the helper cannot
    quietly reintroduce ranking and still pass.
    """
    from fastapi import HTTPException

    from navis.api import workspace_router

    async def _ambiguous(_db, _org, _provider, **_kw):
        raise IntegrationUnavailable(
            "this organization has more than one connected Google account",
            provider="calendar",
        )

    # The helper imports `resolve_integration` INSIDE the function, so the name
    # resolves against this module at call time — patching it here is what the
    # helper will actually see.
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("navis.execution.integrations.resolve_integration", _ambiguous)
        with pytest.raises(HTTPException) as exc:
            await workspace_router._resolve_google_calendar(None, ORG_A)

    assert exc.value.status_code == 409
    assert "more than one" in str(exc.value.detail)


async def test_the_calendar_availability_surface_returns_the_resolved_account():
    """Delegation, not a second lookup: whatever the authority returns is used."""
    from navis.api import workspace_router

    sentinel = object()

    async def _resolved(_db, org_id, provider, **_kw):
        assert org_id == ORG_A and provider == "calendar"
        return sentinel

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("navis.execution.integrations.resolve_integration", _resolved)
        got = await workspace_router._resolve_google_calendar(None, ORG_A)

    assert got is sentinel
