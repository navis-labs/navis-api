"""The universal Action Contract: one action, one spec, one provider.

These tests protect ARCHITECTURAL invariants, not behaviour of a particular
connector. The load-bearing one is `test_a_non_google_action_executes_through_the
_same_universal_path`: a synthetic provider that shares no code with Gmail or
Calendar goes intent -> spec -> request -> approval -> worker -> provider ->
normalized result -> Observation envelope. If that passes, the path is genuinely
universal rather than merely compatible with Google.
"""
from __future__ import annotations

import ast
import pathlib
import uuid
from types import SimpleNamespace
from typing import Any

import pytest

from navis.execution.actions import (
    ActionRisk,
    ActionSpec,
    ApprovalPolicy,
    UnknownAction,
    all_specs,
    credential_preflight,
    get_spec,
    register_action,
    require_spec,
    unregister_action,
)
from navis.execution.contract import ProviderReference, ProviderResult
from navis.execution.models import ActionType
from navis.execution.providers import (
    _DEFAULT_PROVIDERS,
    get_provider,
    production_providers,
)
from navis.execution.providers.base import Provider
from navis.execution.service import ExecutionService
from navis.execution.status import ExecutionStatus
from navis.execution.worker import execute_pass
from navis.organization.contract import Capability
from navis.platform.events import EventType
from tests.execution.fakes import ORG_A, ORG_B, USER, FakeExecutionRepo

EXECUTION_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "navis" / "execution"


# ── 1-4: the registry is complete and authoritative ────────────────────


def test_every_action_has_exactly_one_provider_and_every_provider_has_a_spec():
    specs = all_specs()
    assert set(specs) == set(_DEFAULT_PROVIDERS), (
        "the spec registry and the provider wiring must be the same set of "
        "actions; a spec with no provider dead-letters after approval, a "
        "provider with no spec is unreachable"
    )
    for action_type, spec in specs.items():
        assert _DEFAULT_PROVIDERS[action_type].name == spec.provider


def test_action_type_enum_mirrors_the_registry():
    """`ActionType` is readable constants for callers, NOT a second registry.
    actions.py asserts this at import; asserting it here names the invariant."""
    assert {a.value for a in ActionType} == set(all_specs())


def test_every_action_declares_capability_risk_and_approval():
    for action_type, spec in all_specs().items():
        assert isinstance(spec.capability, Capability), action_type
        assert isinstance(spec.risk, ActionRisk), action_type
        # Everything Navis can do touches an external system, so everything
        # needs a human. This is the metadata form of that claim.
        assert spec.approval is ApprovalPolicy.REQUIRED, action_type


def test_action_capabilities_use_the_existing_connector_vocabulary():
    """No new capability names are invented by the action contract — each action
    consumes a capability the plan model already sells."""
    assert require_spec("send_email").capability is Capability.GMAIL
    assert require_spec("reply_to_email").capability is Capability.GMAIL
    assert require_spec("schedule_meeting").capability is Capability.CALENDAR
    assert require_spec("cancel_meeting").capability is Capability.CALENDAR
    assert require_spec("send_slack").capability is Capability.SLACK


# ── 5-6: unknown action / unknown provider fail closed ─────────────────


async def test_unknown_action_is_refused_at_intake():
    repo = FakeExecutionRepo()
    with pytest.raises(UnknownAction):
        await ExecutionService(repo=repo).request(
            ORG_A, uuid.uuid4(), "notion.create_page", {"title": "x"},
            reasoning_signal_ids=[], reasoning_version=1, state_watermark="w",
        )
    assert repo.rows == {}, "no row may exist for an action Navis cannot perform"
    assert repo.events == [], "and nothing may be published about it"


def test_unknown_action_resolves_to_no_provider():
    assert get_provider("notion.create_page") is None


def test_a_provider_without_a_spec_is_unreachable():
    """Wiring a provider without registering its action must NOT make it
    dispatchable — the spec carries the capability, risk and approval metadata,
    so dispatching without one runs a side effect no gate ever evaluated."""

    class _Rogue(Provider):
        name = "rogue"
        action_type = "rogue_write"

        async def validate(self, payload):
            pass

        async def execute(self, payload):
            raise AssertionError("must never be reached")

        async def status(self, reference):
            return "x"

    assert get_provider("rogue_write", {"rogue_write": _Rogue()}) is None


# ── 9, 12: dry-run safety ──────────────────────────────────────────────


def test_production_registry_exposes_no_simulating_provider():
    for action_type, provider in production_providers().items():
        assert provider.is_live or provider.unsupported, (
            f"{action_type} would simulate a side effect in production"
        )


def test_slack_went_live_through_the_seam_not_around_it():
    """Slack was withheld from production until it had a REAL sender, and now
    ships live. What must remain true is that it got there the same way every
    connector will: a spec, a capability, and a sender that actually posts —
    never by relaxing the dry-run guard."""
    spec = get_spec("send_slack")
    assert spec is not None and spec.capability is Capability.SLACK
    provider = production_providers().get("send_slack")
    assert provider is not None, "slack has a real sender and must be exposed"
    assert provider.is_live and not provider.unsupported


# ── 7, 15: the per-action capability gate ──────────────────────────────


class _EntitlementDB:
    """The one query `OrganizationRepository.resolve_capability` makes.

    Stands in for Postgres so the REAL authorization path — spec.capability ->
    OrganizationService.can_use -> EntitlementService._decide — is exercised in
    process. Nothing about the decision is stubbed; only the row fetch is.

    `row` is the ExecutionRequest `_require_org` will find, for the tests that
    drive the whole approve handler rather than the capability gate alone.
    """

    def __init__(
        self, granted: dict[uuid.UUID, set[str]], row: Any | None = None
    ) -> None:
        self.granted = granted
        self.asked: list[tuple[str, str]] = []
        self.row = row

    async def get(self, _model, _pk):
        return self.row

    async def execute(self, _stmt, params):
        org, cap = params["org"], params["cap"]
        self.asked.append((org, cap))
        allowed = cap in self.granted.get(uuid.UUID(org), set())
        row = {"status": "active", "plan": "team", "explicit": allowed, "killed": False}
        return type(
            "_R", (), {"mappings": lambda self: type("_M", (), {"one": lambda self: row})()}
        )()


async def test_an_action_is_refused_when_its_connector_capability_is_not_held():
    """EXECUTION is a blanket gate — it does not answer "may this org send
    Slack?". An org holding EXECUTION but with SLACK revoked (or `kill:gmail`
    enabled) could previously still run those actions, because the connector
    capability was never consulted.
    """
    from fastapi import HTTPException

    from navis.execution.router import _require_action_capability

    db = _EntitlementDB({ORG_A: {"gmail"}})

    spec = await _require_action_capability(db, ORG_A, "send_email")
    assert spec.capability is Capability.GMAIL

    with pytest.raises(HTTPException) as exc:
        await _require_action_capability(db, ORG_A, "send_slack")
    assert exc.value.status_code == 402
    assert "slack" in str(exc.value.detail)


async def test_one_orgs_capability_never_authorizes_another():
    from fastapi import HTTPException

    from navis.execution.router import _require_action_capability

    db = _EntitlementDB({ORG_A: {"gmail"}, ORG_B: set()})

    await _require_action_capability(db, ORG_A, "send_email")
    with pytest.raises(HTTPException) as exc:
        await _require_action_capability(db, ORG_B, "send_email")
    assert exc.value.status_code == 402
    # The gate asked about the ORG it was given, both times.
    assert db.asked == [(str(ORG_A), "gmail"), (str(ORG_B), "gmail")]


async def test_an_unregistered_action_cannot_be_approved():
    """A persisted row whose action type is no longer registered must not accept
    an approval it can never act on."""
    from fastapi import HTTPException

    from navis.execution.router import _require_action_capability

    with pytest.raises(HTTPException) as exc:
        await _require_action_capability(
            _EntitlementDB({ORG_A: {"gmail"}}), ORG_A, "legacy_action"
        )
    assert exc.value.status_code == 409


# ── the connector pre-flight: declared once, honoured for every action ──


def test_every_action_with_a_credential_authority_declares_a_preflight():
    """An action that will resolve a credential must be able to say it CANNOT be
    performed before a human approves it.

    Only `reply_to_email` used to carry a pre-flight. So approving a
    chat-composed email, a meeting, or a Slack post for an org whose grant could
    not perform it succeeded — and the founder learned about it from a dead
    letter minutes later instead of from the button they had just pressed.

    Scoped to providers that HAVE a credential authority, so a deliberately
    unsupported action is not required to invent one. `actions.py` asserts this
    at import; naming it here says why it exists.
    """
    from navis.execution.integrations import has_integration_resolver

    missing = sorted(
        action_type
        for action_type, spec in all_specs().items()
        if has_integration_resolver(spec.provider) and spec.preflight is None
    )
    assert not missing, f"actions that resolve credentials with no preflight: {missing}"


@pytest.fixture()
def preflighted_action():
    """A non-draftable action with its own credential authority.

    Deliberately NOT Gmail or Calendar: the point is that the generic approve
    handler asks whatever the spec supplies, for any connector, without knowing
    what the connector is.
    """
    from navis.execution import integrations

    asked: list[uuid.UUID] = []
    refuse = {"value": False}

    async def _resolver(db: Any, org_id: uuid.UUID) -> Any:
        asked.append(org_id)
        if refuse["value"]:
            raise integrations.IntegrationUnavailable(
                "Synthetic is connected read-only. Reconnect it to grant write "
                "access.",
                provider="synthetic",
                reconsent_required=True,
            )
        return object()

    integrations.register_integration_resolver("synthetic", _resolver, replace=True)
    spec = register_action(ActionSpec(
        action_type="synthetic_write",
        provider="synthetic",
        capability=Capability.EXECUTION,
        risk=ActionRisk.EXTERNAL_WRITE,
        approval=ApprovalPolicy.REQUIRED,
        preflight=credential_preflight("synthetic"),
    ), replace=True)
    try:
        yield SimpleNamespace(spec=spec, asked=asked, refuse=refuse)
    finally:
        unregister_action("synthetic_write")
        # No public de-registration exists (a connector's resolver is meant to
        # live for the process), so the fixture reaches in to undo its own
        # registration rather than leaking it into the next test.
        integrations._RESOLVERS.pop("synthetic", None)


def _pending_row(action_type: str, org_id: uuid.UUID = ORG_A) -> SimpleNamespace:
    return SimpleNamespace(
        id=uuid.uuid4(), organization_id=org_id, decision_id=uuid.uuid4(),
        action_type=action_type, status=ExecutionStatus.REQUESTED.value,
        payload={"record_id": "rec-1"}, reasoning_signal_ids=[],
        source_signal_event_ids=[], reasoning_version=1, state_watermark="w",
        provider_reference={}, attempts=0, last_error=None, origin="chat",
    )


async def test_approve_runs_the_preflight_for_an_action_that_needs_no_draft(
    preflighted_action,
):
    """The pre-flight used to be nested INSIDE the draft branch of `approve`.

    `reply_to_email` was the only spec declaring one, so the nesting was
    invisible — and it silently disqualified every other action from ever having
    one: a pre-flight registered on a non-draftable action would simply never
    run, with nothing in the spec to say so.

    A refusal must produce a 409 carrying the CONNECTOR's own message, and must
    leave the row REQUESTED — the point of checking here is that no state moves.
    """
    from fastapi import HTTPException

    from navis.execution.router import approve

    preflighted_action.refuse["value"] = True
    row = _pending_row("synthetic_write")
    db = _EntitlementDB({ORG_A: {"execution"}}, row=row)
    principal = SimpleNamespace(org_id=ORG_A, subject_id=USER)

    with pytest.raises(HTTPException) as exc:
        await approve(row.id, principal, db)

    assert exc.value.status_code == 409
    assert "Reconnect" in str(exc.value.detail)
    assert preflighted_action.asked == [ORG_A], "the preflight was never asked"
    assert row.status == ExecutionStatus.REQUESTED.value, (
        "a refused approval must not move the row"
    )


async def test_authorization_is_decided_before_any_credential_is_resolved(
    preflighted_action,
):
    """Possession of a credential must never imply permission to use it.

    So the entitlement gate is asked FIRST and independently: an org that may not
    run this action is refused without anything looking up a token for it. Order,
    not just outcome — a 402 that resolved a credential on the way would mean the
    two questions had been collapsed into one.
    """
    from fastapi import HTTPException

    from navis.execution.router import approve

    row = _pending_row("synthetic_write")
    db = _EntitlementDB({ORG_A: set()}, row=row)  # EXECUTION not granted

    with pytest.raises(HTTPException) as exc:
        await approve(row.id, SimpleNamespace(org_id=ORG_A, subject_id=USER), db)

    assert exc.value.status_code == 402
    assert preflighted_action.asked == [], (
        "a credential was resolved for an org that is not authorized to act"
    )


# ── 16: approval cannot be bypassed by the spec ────────────────────────


async def test_a_spec_cannot_authorize_execution_without_approval():
    """The hard boundary is the status machine, not the metadata. Even a spec
    that declares approval NOT_REQUIRED must not let a REQUESTED row execute:
    `claim_executable` selects APPROVED (or retryable FAILED) rows and nothing
    reads `spec.approval` to skip the transition.
    """
    real = require_spec("send_email")
    register_action(
        ActionSpec(
            action_type="send_email", provider="gmail", capability=Capability.GMAIL,
            risk=ActionRisk.IRREVERSIBLE_COMMUNICATION,
            approval=ApprovalPolicy.NOT_REQUIRED,
        ),
        replace=True,
    )
    try:
        repo = FakeExecutionRepo()
        req, _ = await ExecutionService(repo=repo).request(
            ORG_A, uuid.uuid4(), "send_email",
            {"to": "a@b.com", "subject": "s", "body": "b"},
            reasoning_signal_ids=[], reasoning_version=1, state_watermark="w",
        )
        result = await execute_pass(repo)
        assert result == {"executed": 0, "failed": 0, "skipped": 0}
        assert (await repo.get(req.id)).status == ExecutionStatus.REQUESTED.value
    finally:
        # Put the real spec back. NOT via importlib.reload: reloading rebinds
        # ActionSpec/ActionRisk to new class objects while every other module
        # still holds the old ones, so `isinstance` checks elsewhere would start
        # failing depending on test order.
        register_action(real, replace=True)


# ── 20: the generic worker carries no connector knowledge ──────────────


def test_the_generic_worker_never_imports_a_connector():
    """`worker.py` used to import CALENDAR_MUTATION_ACTION_TYPES from Signals and
    branch on it. Every connector needing a write-back would have added another
    branch. Connector knowledge now reaches the worker only through ActionSpec.
    """
    tree = ast.parse((EXECUTION_DIR / "worker.py").read_text())
    imported = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    leaks = sorted(m for m in imported if m.startswith("navis.signals"))
    assert not leaks, f"worker.py imports connector modules: {leaks}"


def test_the_generic_router_and_worker_do_not_branch_on_action_type():
    """A layer that tests `action_type == "..."` is a layer that needs editing
    for every new connector. The spec is how those questions get answered now."""
    offenders: list[str] = []
    for name in ("worker.py", "router.py", "summary.py", "service.py"):
        source = (EXECUTION_DIR / name).read_text()
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.Compare):
                continue
            src = ast.get_source_segment(source, node) or ""
            if "action_type" in src and ("==" in src or " in " in src):
                offenders.append(f"{name}: {src.strip()[:90]}")
    assert not offenders, "generic layers branch on action_type:\n" + "\n".join(offenders)


# ── 13-15, 17-19: THE PROOF — a non-Google action, same universal path ──


class _SyntheticProvider(Provider):
    """A connector that shares nothing with Google: its own name, its own
    payload shape, its own external id space."""

    name = "synthetic"
    action_type = "synthetic_write"

    def __init__(self) -> None:
        self._sender = self._send  # is_live -> True, so production-registry safe
        self.calls: list[dict[str, Any]] = []

    async def _send(self, payload):  # pragma: no cover - shape only
        return "x"

    async def validate(self, payload: dict[str, Any]) -> None:
        from navis.execution.providers.base import ProviderValidationError

        if not payload.get("record_id"):
            raise ProviderValidationError("synthetic_write requires 'record_id'")

    async def execute(self, payload: dict[str, Any]) -> ProviderResult:
        self.calls.append(payload)
        return ProviderResult(
            ok=True,
            reference=ProviderReference("synthetic", f"syn-{payload['record_id']}"),
        )

    async def status(self, reference: dict[str, Any]) -> str:
        return "written"


@pytest.fixture()
def synthetic_action():
    """Register a non-Google action the way a future connector will.

    `register_action` is the production extension point, not a test hook — this
    is exactly the call a Notion or Jira module makes at import.
    """
    spec = register_action(ActionSpec(
        action_type="synthetic_write",
        provider="synthetic",
        capability=Capability.EXECUTION,
        risk=ActionRisk.EXTERNAL_WRITE,
        approval=ApprovalPolicy.REQUIRED,
    ))
    try:
        yield spec
    finally:
        unregister_action("synthetic_write")


async def test_a_non_google_action_executes_through_the_same_universal_path(
    synthetic_action,
):
    """intent -> ActionSpec -> request -> approval -> worker -> provider ->
    normalized ProviderResult -> Observation envelope.

    Nothing in this test touches Gmail, Calendar, Google or Signals.
    """
    repo = FakeExecutionRepo()
    provider = _SyntheticProvider()
    svc = ExecutionService(repo=repo)

    # Intake accepts it purely because a spec exists.
    req, created = await svc.request(
        ORG_A, uuid.uuid4(), "synthetic_write", {"record_id": "rec-1"},
        reasoning_signal_ids=["s1"], reasoning_version=3, state_watermark="w-9",
    )
    assert created and req.status == ExecutionStatus.REQUESTED.value

    # Approval is the same hard boundary.
    result = await execute_pass(repo, providers={"synthetic_write": provider})
    assert result["executed"] == 0, "unapproved work is not claimable"

    await svc.approve(req.id, USER)
    result = await execute_pass(repo, providers={"synthetic_write": provider})

    assert result["executed"] == 1
    assert provider.calls and provider.calls[0]["record_id"] == "rec-1"
    row = await repo.get(req.id)
    assert row.status == ExecutionStatus.EXECUTED.value
    assert row.provider_reference == {
        "provider": "synthetic", "external_id": "syn-rec-1",
    }


async def test_observation_receives_the_same_envelope_whatever_the_provider(
    synthetic_action,
):
    """Observation subscribes to ACTION_EXECUTED and reads `outcome`. The
    envelope must not vary by connector, or Learning would need per-provider
    handling for what is the same fact: an action happened."""

    async def _executed_outcome(action_type, payload, provider):
        repo = FakeExecutionRepo()
        svc = ExecutionService(repo=repo)
        req, _ = await svc.request(
            ORG_A, uuid.uuid4(), action_type, payload,
            reasoning_signal_ids=["s1"], reasoning_version=3, state_watermark="w-9",
        )
        await svc.approve(req.id, USER)
        await execute_pass(repo, providers={action_type: provider})
        events = [e for e in repo.events if e.event_type == EventType.ACTION_EXECUTED.value]
        assert len(events) == 1
        return events[0].payload["outcome"]

    from navis.execution.providers.gmail import GmailProvider

    synthetic = await _executed_outcome(
        "synthetic_write", {"record_id": "rec-1"}, _SyntheticProvider()
    )
    gmail = await _executed_outcome(
        "send_email",
        {"to": "a@b.com", "subject": "s", "body": "a real body"},
        GmailProvider(sender=lambda payload: _resolved("gmail-1")),
    )

    assert set(synthetic) == set(gmail), "outcome envelope differs by provider"
    for key in ("org_id", "reasoning_signal_ids", "reasoning_version", "state_watermark"):
        assert synthetic[key] == gmail[key]
    # The only thing that legitimately varies is which system was touched.
    assert synthetic["provider_reference"]["provider"] == "synthetic"
    assert gmail["provider_reference"]["provider"] == "gmail"


async def _resolved(value):
    return value


async def test_execution_ids_stay_deterministic_and_org_isolated(synthetic_action):
    """Same org + same decision + same payload -> the same execution id (replay
    converges). A DIFFERENT org -> a different id, always: the org is part of the
    idempotency key, so two tenants proposing identical work can never collide on
    one row."""
    repo = FakeExecutionRepo()
    svc = ExecutionService(repo=repo)
    decision = uuid.uuid4()
    payload = {"record_id": "rec-1"}

    first, created_a = await svc.request(
        ORG_A, decision, "synthetic_write", payload,
        reasoning_signal_ids=[], reasoning_version=1, state_watermark="w",
    )
    again, created_b = await svc.request(
        ORG_A, decision, "synthetic_write", payload,
        reasoning_signal_ids=[], reasoning_version=1, state_watermark="w",
    )
    other_org, _ = await svc.request(
        ORG_B, decision, "synthetic_write", payload,
        reasoning_signal_ids=[], reasoning_version=1, state_watermark="w",
    )

    assert created_a and not created_b
    assert first.id == again.id
    assert other_org.id != first.id
    assert other_org.organization_id == ORG_B
