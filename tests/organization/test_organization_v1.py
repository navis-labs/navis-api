# pyright: reportArgumentType=false, reportOptionalMemberAccess=false
# ^ Test-double typing: services are annotated for the real OrganizationRepository, but
#   these tests inject the in-memory FakeOrgRepo (structural, not a nominal subclass), and
#   intentionally access getter results they know are present. Both are inherent to the
#   fake-based unit style; suppressed file-wide rather than threading asserts/Protocols.
"""Organization V1: onboarding funnel, entitlements, feature flags, lifecycle,
activation/TTFI, plan gating, multi-tenancy, and the import guard.
"""
from __future__ import annotations

import ast
import pathlib
from types import SimpleNamespace

import pytest

from navis.organization.activation import ActivationService, compute_ttfi_seconds
from navis.organization.billing import BillingService
from navis.organization.contract import (
    Capability,
    LifecycleStatus,
    OnboardingEvent,
    OnboardingStage,
    Plan,
    capabilities_for_plan,
)
from navis.organization.entitlements import EntitlementService
from navis.organization.feature_flags import FeatureFlagService, kill_switch_flag
from navis.organization.lifecycle import LifecycleError, LifecycleService
from navis.organization.metrics import MetricsService
from navis.organization.onboarding import OnboardingService
from tests.organization.fakes import ACTOR, ORG_A, ORG_B, FakeOrgRepo, FakePublisher

ORG_PKG = pathlib.Path(__file__).resolve().parent.parent.parent / "navis" / "organization"
FORBIDDEN = {"signals", "memory", "state", "reasoning", "decisions",
             "execution", "observation", "learning", "intelligence", "adaptive"}


# ---- onboarding funnel ----

async def test_onboarding_advances_forward_and_emits_events():
    repo, pub = FakeOrgRepo(), FakePublisher()
    svc = OnboardingService(repo, publisher=pub)
    for stage in (OnboardingStage.CREATED, OnboardingStage.FIRST_SIGNAL, OnboardingStage.FIRST_BRIEF):
        await svc.advance(ORG_A, stage)
    row = await repo.get_onboarding(ORG_A)
    assert row.stage == OnboardingStage.FIRST_BRIEF.value
    assert OnboardingEvent.FIRST_BRIEF in [e for e, _ in pub.events]
    # reaching first brief activates the org
    assert (await repo.get_lifecycle(ORG_A)).status == LifecycleStatus.ACTIVE.value


async def test_onboarding_is_monotonic_no_backwards():
    repo = FakeOrgRepo()
    svc = OnboardingService(repo)
    await svc.advance(ORG_A, OnboardingStage.STATE_READY)
    await svc.advance(ORG_A, OnboardingStage.FIRST_SIGNAL)  # earlier -> ignored
    assert (await repo.get_onboarding(ORG_A)).stage == OnboardingStage.STATE_READY.value


# ---- TTFI + activation ----

async def test_ttfi_and_activation_score():
    repo = FakeOrgRepo()
    svc = OnboardingService(repo)
    # CREATED is stamped by bootstrap in production (advance() is a no-op when
    # re-asserting the current stage), so set the CREATED clock the same way here.
    row = await repo.ensure_onboarding(ORG_A)
    row.stage_timestamps = {OnboardingStage.CREATED.value: "2026-06-01T00:00:00+00:00"}
    await svc.advance(ORG_A, OnboardingStage.FIRST_BRIEF)
    metrics = await repo.get_metrics(ORG_A)
    assert metrics.ttfi_seconds is not None and metrics.ttfi_seconds >= 0
    assert 0 < metrics.activation_score <= 100


def test_ttfi_none_without_first_brief():
    from types import SimpleNamespace

    onboarding = SimpleNamespace(stage_timestamps={OnboardingStage.CREATED.value: "2026-06-01T00:00:00+00:00"})
    assert compute_ttfi_seconds(onboarding) is None


async def test_activation_recompute_idempotent():
    repo = FakeOrgRepo()
    await OnboardingService(repo).advance(ORG_A, OnboardingStage.CREATED)
    m1 = await ActivationService(repo).recompute(ORG_A)
    m2 = await ActivationService(repo).recompute(ORG_A)
    assert m1.activation_score == m2.activation_score


# ---- entitlements + plan gating ----

async def test_plan_defaults_gate_capabilities():
    repo = FakeOrgRepo()
    ent = EntitlementService(repo)
    await BillingService(repo, ent).set_plan(ORG_A, Plan.FREE, billing_authorized=True)
    assert await ent.can_use(ORG_A, Capability.MEMORY) is True        # in FREE
    assert await ent.can_use(ORG_A, Capability.EXECUTION) is False     # not in FREE
    assert await ent.can_use(ORG_A, Capability.INTELLIGENCE) is False

    await BillingService(repo, ent).set_plan(ORG_A, Plan.ENTERPRISE, billing_authorized=True)
    assert await ent.can_use(ORG_A, Capability.ADAPTIVE) is True       # ENTERPRISE has all


async def test_can_use_unknown_capability_denies_not_raises():
    # can_use resolves by value, so a capability outside the enum must deny (False),
    # never raise ValueError -> 500. Guards the single-query resolution path.
    repo = FakeOrgRepo()
    ent = EntitlementService(repo)
    await BillingService(repo, ent).set_plan(ORG_A, Plan.ENTERPRISE, billing_authorized=True)
    assert await ent.can_use(ORG_A, "not_a_real_capability") is False


async def test_entitlement_decision_does_not_depend_on_environment():
    """The capability gate must answer the same in dev, staging and production.

    `_decide` used to return True for everything whenever `settings.is_production`
    was False — and ENVIRONMENT defaults to development when unset, so one missing
    env var on a production box granted every capability to every org. This is the
    regression test for that: the decision reads org state only.
    """
    from navis.platform.config import settings

    repo = FakeOrgRepo()
    ent = EntitlementService(repo)
    await BillingService(repo, ent).set_plan(ORG_A, Plan.FREE, billing_authorized=True)

    original = settings.environment
    try:
        for env in ("development", "staging", "production"):
            settings.environment = env
            assert await ent.can_use(ORG_A, Capability.EXECUTION) is False
            assert await ent.can_use(ORG_A, Capability.MEMORY) is True
    finally:
        settings.environment = original


async def test_unknown_plan_and_status_deny_rather_than_raise():
    """`plan` and `status` are free text in the database, so a legacy, renamed or
    hand-edited value reaches the decision function. It must deny — raising here
    would turn every gated request for that org into a 500, and falling through
    would authorize an org whose tier nobody can name."""
    repo = FakeOrgRepo()
    ent = EntitlementService(repo)
    await BillingService(repo, ent).set_plan(ORG_A, Plan.ENTERPRISE, billing_authorized=True)

    lifecycle = await repo.ensure_lifecycle(ORG_A)
    # Drop the seeded explicit grant so resolution actually reaches the plan branch.
    del repo.entitlements[(ORG_A, Capability.EXECUTION.value)]

    lifecycle.plan = "legacy_gold"
    assert await ent.can_use(ORG_A, Capability.EXECUTION) is False

    lifecycle.plan = Plan.ENTERPRISE.value
    lifecycle.status = "deleted"
    assert await ent.can_use(ORG_A, Capability.EXECUTION) is False


async def test_unprovisioned_org_gets_free_tier_only():
    """No lifecycle row = the window between org creation and async provisioning.
    It resolves to the minimum tier, never to a capability that can mutate the
    outside world."""
    repo = FakeOrgRepo()
    ent = EntitlementService(repo)
    assert await ent.can_use(ORG_A, Capability.MEMORY) is True        # FREE
    assert await ent.can_use(ORG_A, Capability.EXECUTION) is False    # not FREE
    assert await ent.can_use(ORG_A, Capability.DECISION) is False


async def test_explicit_entitlement_overrides_plan():
    repo = FakeOrgRepo()
    ent = EntitlementService(repo)
    await BillingService(repo, ent).set_plan(ORG_A, Plan.FREE, billing_authorized=True)
    await ent.set(ORG_A, Capability.EXECUTION, True)   # explicit grant beyond plan
    assert await ent.can_use(ORG_A, Capability.EXECUTION) is True


async def test_suspended_org_can_use_nothing():
    repo = FakeOrgRepo()
    ent = EntitlementService(repo)
    await BillingService(repo, ent).set_plan(ORG_A, Plan.ENTERPRISE, billing_authorized=True)
    assert await ent.can_use(ORG_A, Capability.MEMORY) is True
    # suspend requires ACTIVE first (TRIAL->SUSPENDED is illegal by design)
    await LifecycleService(repo).activate(ORG_A)
    await LifecycleService(repo).suspend(ORG_A)
    assert await ent.can_use(ORG_A, Capability.MEMORY) is False


# ---- feature flags / kill switch ----

async def test_kill_switch_forces_capability_off():
    repo = FakeOrgRepo()
    ent = EntitlementService(repo)
    await BillingService(repo, ent).set_plan(ORG_A, Plan.ENTERPRISE, billing_authorized=True)
    assert await ent.can_use(ORG_A, Capability.INTELLIGENCE) is True
    await FeatureFlagService(repo).set(ORG_A, kill_switch_flag(Capability.INTELLIGENCE.value), kind="kill_switch", enabled=True)
    assert await ent.can_use(ORG_A, Capability.INTELLIGENCE) is False


async def test_flag_org_override_beats_global():
    repo = FakeOrgRepo()
    flags = FeatureFlagService(repo)
    await flags.set(None, "beta_x", enabled=False)     # global default off
    await flags.set(ORG_A, "beta_x", enabled=True)     # org override on
    assert await flags.evaluate(ORG_A, "beta_x") is True
    assert await flags.evaluate(ORG_B, "beta_x") is False  # other org sees global


# ---- lifecycle transitions ----

async def test_lifecycle_legal_and_illegal_transitions():
    repo = FakeOrgRepo()
    lc = LifecycleService(repo)
    activated = await lc.activate(ORG_A)
    assert activated.status == LifecycleStatus.ACTIVE.value and activated.activated_at is not None
    await lc.churn(ORG_A)
    with pytest.raises(LifecycleError):
        await lc.activate(ORG_A)  # CHURNED is terminal


# ---- multi-tenant isolation ----

async def test_entitlements_isolated_per_org():
    repo = FakeOrgRepo()
    ent = EntitlementService(repo)
    await BillingService(repo, ent).set_plan(ORG_A, Plan.ENTERPRISE, billing_authorized=True)
    await BillingService(repo, ent).set_plan(ORG_B, Plan.FREE, billing_authorized=True)
    assert await ent.can_use(ORG_A, Capability.EXECUTION) is True
    assert await ent.can_use(ORG_B, Capability.EXECUTION) is False


async def test_metrics_rollup_shape():
    repo = FakeOrgRepo()
    await OnboardingService(repo).advance(ORG_A, OnboardingStage.CREATED)
    roll = await MetricsService(repo).rollup(ORG_A)
    assert set(roll) == {"status", "plan", "stage", "signals_count", "memory_count", "ttfi_seconds", "activation_score"}


# ---- snapshot N+1 fix: bulk capability resolution ----

async def test_snapshot_agrees_with_can_use_across_all_capabilities():
    """snapshot() (1 SQL roundtrip) must return identical decisions to can_use()
    (1 SQL roundtrip per cap) for every Capability. Guards against drift between the
    single-row and multi-row resolution paths — both feed through _decide."""
    repo = FakeOrgRepo()
    ent = EntitlementService(repo)
    await BillingService(repo, ent).set_plan(ORG_A, Plan.PRO, billing_authorized=True)
    # Mix in: an explicit grant beyond plan + an org-specific kill switch.
    await ent.set(ORG_A, Capability.EXECUTION, True)
    await FeatureFlagService(repo).set(
        ORG_A, kill_switch_flag(Capability.MEMORY.value), kind="kill_switch", enabled=True
    )

    snap = await ent.snapshot(ORG_A)
    # Shape: one entry per Capability value, nothing more, nothing less.
    assert set(snap.keys()) == {c.value for c in Capability}
    # Every value matches can_use() for that capability — no drift.
    for cap in Capability:
        assert snap[cap.value] == await ent.can_use(ORG_A, cap), f"drift on {cap.value}"


async def test_snapshot_respects_suspended_lifecycle():
    """SUSPENDED orgs deny EVERYTHING through both can_use and snapshot."""
    repo = FakeOrgRepo()
    ent = EntitlementService(repo)
    await BillingService(repo, ent).set_plan(ORG_A, Plan.ENTERPRISE, billing_authorized=True)
    await LifecycleService(repo).activate(ORG_A)
    await LifecycleService(repo).suspend(ORG_A)
    snap = await ent.snapshot(ORG_A)
    assert all(v is False for v in snap.values())


# ---- capability contract sanity ----

def test_plan_capability_monotonicity():
    free = capabilities_for_plan(Plan.FREE)
    pro = capabilities_for_plan(Plan.PRO)
    team = capabilities_for_plan(Plan.TEAM)
    ent = capabilities_for_plan(Plan.ENTERPRISE)
    assert free < pro < team < ent  # each tier strictly supersets the prior
    assert ent == frozenset(Capability)


#: What the DEVELOPER surface cannot function without — one capability per stage
#: of the loop it actually serves.
#:
#:   memory       the retrieval floor `POST /developer/context` gates on
#:   state        what is currently true, plus the agent registry's own gate
#:   reasoning    the organization's conclusions; without it the context
#:                bundle withholds the reasoning layer
#:   observation  the agent execution ledger — tasks, runs, steps, outcomes
#:
#: Not the whole Developer surface: `decision`, `intelligence` and `learning`
#: stay paid on purpose, and a FREE org is correctly told so through
#: `layers.withheld`. These four are the ones whose absence leaves nothing for
#: a developer to evaluate.
DEVELOPER_FLOOR = frozenset({
    Capability.MEMORY, Capability.STATE, Capability.REASONING, Capability.OBSERVATION,
})


def test_every_plan_grants_the_developer_floor():
    """A self-serve developer must be able to read context and record runs.

    THE REGRESSION THIS PINS (P0-4). `observation` was ENTERPRISE-only, so on
    the FREE tier every tenant gets by signing up, the entire agent execution
    ledger answered 402 — an agent platform whose free tier cannot record what
    an agent did. Seven of the Developer surface's fourteen route groups were
    unreachable; with this floor it is four, and each of those four is a
    deliberate paid layer that reports itself as withheld rather than empty.

    Dropping any of these from any plan takes the Developer product below
    usable for that tier, which is a decision to make deliberately and not by
    editing a set literal.
    """
    for plan in Plan:
        missing = sorted(c.value for c in DEVELOPER_FLOOR - capabilities_for_plan(plan))
        assert missing == [], f"{plan.value} cannot serve the Developer surface: missing {missing}"


def test_the_developer_floor_does_not_quietly_become_the_whole_matrix():
    """The floor is a floor, not a way to make everything free.

    If `decision`, `intelligence`, `learning` or `execution` ever land in FREE,
    that is a pricing change and it should be argued for, not arrive as a side
    effect of widening the floor.
    """
    free = capabilities_for_plan(Plan.FREE)
    for paid in (
        Capability.DECISION, Capability.INTELLIGENCE,
        Capability.LEARNING, Capability.EXECUTION,
    ):
        assert paid not in free, f"{paid.value} became free — was that deliberate?"


# ---- architecture guard ----

def test_organization_imports_no_business_domain():
    violations = []
    for py in ORG_PKG.rglob("*.py"):
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
    assert not violations, "Organization must not import a business domain:\n" + "\n".join(violations)


# ---- T-4: plan escalation authority ----------------------------------------
#
# `PUT /organization/plan` gates on Permission.MANAGE_BILLING, which every FOUNDER
# holds, and the first user of every org IS a founder. `set_plan` then wrote
# EXPLICIT entitlement rows via `seed_for_plan`, and explicit rows outrank plan
# defaults in `_decide` — so `{"plan": "enterprise"}` permanently granted the
# caller's own org all fourteen capabilities, for free, indistinguishably from a
# real upgrade.


async def test_free_cannot_self_upgrade_to_enterprise():
    """THE T-4 REGRESSION. Unauthorized escalation raises and changes NOTHING."""
    from navis.organization.billing import PlanEscalationDenied

    repo = FakeOrgRepo()
    ent = EntitlementService(repo)
    billing = BillingService(repo, ent)
    await billing.set_plan(ORG_A, Plan.FREE, billing_authorized=True)

    with pytest.raises(PlanEscalationDenied) as exc:
        await billing.set_plan(ORG_A, Plan.ENTERPRISE, actor_id=ACTOR)

    # Names what would have been granted — an operator reading the log/402 needs it.
    assert "execution" in str(exc.value)
    # The plan did not move and no entitlement row was rewritten.
    assert (await billing.get_plan(ORG_A)) is Plan.FREE
    assert await ent.can_use(ORG_A, Capability.EXECUTION) is False
    assert await ent.can_use(ORG_A, Capability.ADAPTIVE) is False
    # ...and what FREE legitimately grants is untouched.
    assert await ent.can_use(ORG_A, Capability.MEMORY) is True


async def test_every_intermediate_upgrade_is_refused_too():
    """Not just the jump to ENTERPRISE — any capability-GAINING move is refused."""
    from navis.organization.billing import PlanEscalationDenied

    repo = FakeOrgRepo()
    ent = EntitlementService(repo)
    billing = BillingService(repo, ent)
    await billing.set_plan(ORG_A, Plan.FREE, billing_authorized=True)

    for target in (Plan.PRO, Plan.TEAM, Plan.ENTERPRISE):
        with pytest.raises(PlanEscalationDenied):
            await billing.set_plan(ORG_A, target)
    assert (await billing.get_plan(ORG_A)) is Plan.FREE


async def test_downgrade_stays_self_service():
    """Giving up capabilities harms nobody and needs no billing authority."""
    repo = FakeOrgRepo()
    ent = EntitlementService(repo)
    billing = BillingService(repo, ent)
    await billing.set_plan(ORG_A, Plan.ENTERPRISE, billing_authorized=True)

    await billing.set_plan(ORG_A, Plan.FREE)  # unauthorized, but a reduction

    assert (await billing.get_plan(ORG_A)) is Plan.FREE
    assert await ent.can_use(ORG_A, Capability.EXECUTION) is False
    assert await ent.can_use(ORG_A, Capability.MEMORY) is True


async def test_an_authorized_plan_change_still_works_and_reseeds():
    """The billing seam is unchanged for a legitimate authority."""
    repo = FakeOrgRepo()
    ent = EntitlementService(repo)
    billing = BillingService(repo, ent)
    await billing.set_plan(ORG_A, Plan.FREE, billing_authorized=True)

    await billing.set_plan(ORG_A, Plan.ENTERPRISE, billing_authorized=True)

    assert (await billing.get_plan(ORG_A)) is Plan.ENTERPRISE
    for cap in Capability:
        assert await ent.can_use(ORG_A, cap) is True, f"{cap.value} not reseeded"


async def test_a_lateral_plan_change_is_not_an_escalation():
    """Refusal is on the CAPABILITY SET, not on a ranking of plan names: a move
    that grants nothing new is allowed even without authority."""
    repo = FakeOrgRepo()
    ent = EntitlementService(repo)
    billing = BillingService(repo, ent)
    await billing.set_plan(ORG_A, Plan.ENTERPRISE, billing_authorized=True)

    await billing.set_plan(ORG_A, Plan.ENTERPRISE)  # no-op re-assert, no authority
    assert (await billing.get_plan(ORG_A)) is Plan.ENTERPRISE


async def test_a_corrupt_stored_plan_denies_rather_than_grants():
    """An unparseable plan column resolves to FREE — the same fallback `_decide`
    uses — so a hand-edited row cannot become a free upgrade."""
    from navis.organization.billing import PlanEscalationDenied

    repo = FakeOrgRepo()
    ent = EntitlementService(repo)
    billing = BillingService(repo, ent)
    await billing.set_plan(ORG_A, Plan.FREE, billing_authorized=True)
    (await repo.ensure_lifecycle(ORG_A)).plan = "platinum-unlimited"

    with pytest.raises(PlanEscalationDenied):
        await billing.set_plan(ORG_A, Plan.ENTERPRISE)


async def test_unknown_plan_is_rejected_before_any_write():
    """`PlanBody.plan` is typed to the closed Plan enum, so an unknown plan is a
    422 at the edge and never reaches BillingService."""
    with pytest.raises(ValueError):
        Plan("platinum-unlimited")

    from navis.organization.router import PlanBody

    with pytest.raises(Exception):
        PlanBody(plan="platinum-unlimited")


async def test_escalation_denial_survives_every_environment():
    """Environment-blind by construction. `require_capability` once returned
    unchecked whenever is_production was false — and ENVIRONMENT defaults to
    development — so one missing variable removed the boundary. Authorization must
    never learn about deployments; dev breadth comes from what bootstrap
    PROVISIONS, not from what the gate allows."""
    from navis.organization.billing import PlanEscalationDenied
    from navis.platform.config import settings

    original = settings.environment
    try:
        for env in ("development", "staging", "production"):
            settings.environment = env
            repo = FakeOrgRepo()
            billing = BillingService(repo, EntitlementService(repo))
            await billing.set_plan(ORG_A, Plan.FREE, billing_authorized=True)
            with pytest.raises(PlanEscalationDenied):
                await billing.set_plan(ORG_A, Plan.ENTERPRISE)
    finally:
        settings.environment = original


async def test_the_http_route_never_passes_billing_authority():
    """The flag is keyword-only and server-side: pin that the route cannot set it,
    and that bootstrap (the legitimate provisioning authority) does.

    Asserted on the PARSED call, not the source text — the route's own docstring
    explains the flag, so a substring check matches the explanation and passes for
    the wrong reason (it did, on first run).
    """
    import ast
    import inspect
    import textwrap

    from navis.organization import router as R
    from navis.organization.service import OrganizationService

    def _set_plan_kwargs(fn) -> set[str]:
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
        return {
            kw.arg
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "set_plan"
            for kw in node.keywords
            if kw.arg
        }

    assert "billing_authorized" not in _set_plan_kwargs(R.set_plan)
    assert "billing_authorized" in _set_plan_kwargs(OrganizationService.bootstrap)

    # The route must translate the refusal rather than 500 on it.
    assert "PlanEscalationDenied" in {
        n.id
        for n in ast.walk(ast.parse(textwrap.dedent(inspect.getsource(R.set_plan))))
        if isinstance(n, ast.Name)
    }

    # No request model may carry it either — that would make it caller-supplied.
    assert "billing_authorized" not in R.PlanBody.model_fields


# ---- T-3: lifecycle has a production entry point ---------------------------
#
# `suspend()` and `churn()` were written, tested, and called by NOTHING: the only
# production caller of LifecycleService was onboarding -> activate. So no org could
# leave TRIAL/ACTIVE, which made the first and most authoritative gate in
# `_decide` — the lifecycle check — dead code, and with it the execution worker's
# "the org is suspended" refusal.


def test_lifecycle_task_rejects_an_unknown_status_without_touching_the_db(monkeypatch):
    from navis.orchestration import org_provisioning as P

    def _boom():
        raise AssertionError("must validate before opening a session")

    monkeypatch.setattr(P, "session_scope", _boom)

    out = P.set_organization_lifecycle(str(ORG_A), "platinum")
    assert "unknown lifecycle status" in out["error"]
    assert set(out["allowed"]) == {"active", "suspended", "churned"}


def test_lifecycle_task_refuses_to_reset_an_org_to_trial(monkeypatch):
    """TRIAL is the PROVISIONING state. Moving an org back to it would hand a
    suspended tenant a fresh trial, so it is not operator-settable."""
    from navis.orchestration import org_provisioning as P

    def _boom():
        raise AssertionError("must validate before opening a session")

    monkeypatch.setattr(P, "session_scope", _boom)

    out = P.set_organization_lifecycle(str(ORG_A), "trial")
    assert "not an operator-settable status" in out["error"]


def test_lifecycle_task_drives_the_existing_service(monkeypatch):
    """The task adds no rules of its own — it calls LifecycleService, which owns
    the transition table."""
    import contextlib

    from navis.orchestration import org_provisioning as P

    calls: list[str] = []

    class _Lifecycle:
        async def activate(self, org_id):
            calls.append("activate")
            return SimpleNamespace(status="active")

        async def suspend(self, org_id):
            calls.append("suspend")
            return SimpleNamespace(status="suspended")

        async def churn(self, org_id):
            calls.append("churn")
            return SimpleNamespace(status="churned")

    class _Svc:
        def __init__(self, _db):
            self.lifecycle = _Lifecycle()

    @contextlib.asynccontextmanager
    async def _scope():
        yield object()

    monkeypatch.setattr(P, "session_scope", _scope)
    monkeypatch.setattr(P, "OrganizationService", _Svc)

    for status in ("suspended", "active", "churned"):
        out = P.set_organization_lifecycle(str(ORG_A), status)
        assert out == {"org_id": str(ORG_A), "status": status}
    assert calls == ["suspend", "activate", "churn"]


def test_lifecycle_task_reports_an_illegal_transition_instead_of_raising(monkeypatch):
    """CHURNED is terminal, so this is also what stops a churned tenant being
    revived. An operator gets a stated reason, not a traceback."""
    import contextlib

    from navis.orchestration import org_provisioning as P

    class _Lifecycle:
        async def activate(self, org_id):
            raise LifecycleError("illegal transition churned -> active")

    class _Svc:
        def __init__(self, _db):
            self.lifecycle = _Lifecycle()

    @contextlib.asynccontextmanager
    async def _scope():
        yield object()

    monkeypatch.setattr(P, "session_scope", _scope)
    monkeypatch.setattr(P, "OrganizationService", _Svc)

    out = P.set_organization_lifecycle(str(ORG_A), "active")
    assert "illegal transition" in out["error"]


async def test_a_lifecycle_transition_evicts_that_orgs_cached_principals():
    """A suspension must bite now, not on the 30s principal-cache TTL — and it
    must not evict any OTHER tenant's sessions."""
    import uuid as _uuid

    from navis.identity.authz import (
        Principal,
        PrincipalKind,
        _PRINCIPAL_CACHE,
        _principal_cache_put,
        principal_cache_clear,
    )

    principal_cache_clear()
    try:
        for org in (ORG_A, ORG_A, ORG_B):
            _principal_cache_put(
                _uuid.uuid4().hex,
                Principal(
                    kind=PrincipalKind.USER, org_id=org,
                    subject_id=_uuid.uuid4(), permissions=frozenset(),
                ),
            )
        assert len(_PRINCIPAL_CACHE) == 3

        repo = FakeOrgRepo()
        lc = LifecycleService(repo)
        await lc.activate(ORG_A)
        await lc.suspend(ORG_A)

        remaining = [p.org_id for _e, _u, p in _PRINCIPAL_CACHE.values()]
        assert remaining == [ORG_B], "evicted the wrong tenant's sessions"
    finally:
        principal_cache_clear()


async def test_a_lifecycle_transition_is_audited_and_metered():
    from navis.platform.monitoring.metrics import metrics

    repo = FakeOrgRepo()
    audit = _RecordingAudit()
    lc = LifecycleService(repo, audit)

    before = metrics.snapshot()["counters"].get(
        "organization.lifecycle.transition{from=active,to=suspended}", 0
    )
    await lc.activate(ORG_A, actor_id=ACTOR)
    await lc.suspend(ORG_A, actor_id=ACTOR)
    after = metrics.snapshot()["counters"].get(
        "organization.lifecycle.transition{from=active,to=suspended}", 0
    )

    assert after == before + 1
    assert [a for a, _ in audit.records] == [
        "org.lifecycle.active", "org.lifecycle.suspended"
    ]
    # Idempotent re-assert writes nothing and audits nothing.
    await lc.suspend(ORG_A, actor_id=ACTOR)
    assert len(audit.records) == 2


class _RecordingAudit:
    def __init__(self) -> None:
        self.records: list[tuple] = []

    async def record(self, action, *, org_id=None, actor_id=None, target_id=None, metadata=None):
        self.records.append((action, org_id))


def test_no_http_route_can_change_lifecycle():
    """Suspension is an OPERATOR action. Every HTTP principal is scoped to one
    organization, so a route here would let a founder reactivate their own
    suspended org and make suspension advisory."""
    import inspect

    from navis.organization import router as R

    src = inspect.getsource(R)
    for lever in ("lifecycle.suspend", "lifecycle.churn", "lifecycle.activate"):
        assert lever not in src, f"{lever} became reachable over HTTP"
