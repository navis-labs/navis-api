"""Organization V1: onboarding funnel, entitlements, feature flags, lifecycle,
activation/TTFI, plan gating, multi-tenancy, and the import guard.
"""
from __future__ import annotations

import ast
import pathlib

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
    await svc.advance(ORG_A, OnboardingStage.CREATED)
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
    await BillingService(repo, ent).set_plan(ORG_A, Plan.FREE)
    assert await ent.can_use(ORG_A, Capability.MEMORY) is True        # in FREE
    assert await ent.can_use(ORG_A, Capability.EXECUTION) is False     # not in FREE
    assert await ent.can_use(ORG_A, Capability.INTELLIGENCE) is False

    await BillingService(repo, ent).set_plan(ORG_A, Plan.ENTERPRISE)
    assert await ent.can_use(ORG_A, Capability.ADAPTIVE) is True       # ENTERPRISE has all


async def test_explicit_entitlement_overrides_plan():
    repo = FakeOrgRepo()
    ent = EntitlementService(repo)
    await BillingService(repo, ent).set_plan(ORG_A, Plan.FREE)
    await ent.set(ORG_A, Capability.EXECUTION, True)   # explicit grant beyond plan
    assert await ent.can_use(ORG_A, Capability.EXECUTION) is True


async def test_suspended_org_can_use_nothing():
    repo = FakeOrgRepo()
    ent = EntitlementService(repo)
    await BillingService(repo, ent).set_plan(ORG_A, Plan.ENTERPRISE)
    assert await ent.can_use(ORG_A, Capability.MEMORY) is True
    # suspend requires ACTIVE first (TRIAL->SUSPENDED is illegal by design)
    await LifecycleService(repo).activate(ORG_A)
    await LifecycleService(repo).suspend(ORG_A)
    assert await ent.can_use(ORG_A, Capability.MEMORY) is False


# ---- feature flags / kill switch ----

async def test_kill_switch_forces_capability_off():
    repo = FakeOrgRepo()
    ent = EntitlementService(repo)
    await BillingService(repo, ent).set_plan(ORG_A, Plan.ENTERPRISE)
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
    await BillingService(repo, ent).set_plan(ORG_A, Plan.ENTERPRISE)
    await BillingService(repo, ent).set_plan(ORG_B, Plan.FREE)
    assert await ent.can_use(ORG_A, Capability.EXECUTION) is True
    assert await ent.can_use(ORG_B, Capability.EXECUTION) is False


async def test_metrics_rollup_shape():
    repo = FakeOrgRepo()
    await OnboardingService(repo).advance(ORG_A, OnboardingStage.CREATED)
    roll = await MetricsService(repo).rollup(ORG_A)
    assert set(roll) == {"status", "plan", "stage", "signals_count", "memory_count", "ttfi_seconds", "activation_score"}


# ---- capability contract sanity ----

def test_plan_capability_monotonicity():
    free = capabilities_for_plan(Plan.FREE)
    pro = capabilities_for_plan(Plan.PRO)
    team = capabilities_for_plan(Plan.TEAM)
    ent = capabilities_for_plan(Plan.ENTERPRISE)
    assert free < pro < team < ent  # each tier strictly supersets the prior
    assert ent == frozenset(Capability)


# ---- architecture guard ----

def test_organization_imports_no_business_domain():
    violations = []
    for py in ORG_PKG.rglob("*.py"):
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
    assert not violations, "Organization must not import a business domain:\n" + "\n".join(violations)
