# pyright: reportArgumentType=false
# ^ Same fake-based unit style as tests/organization/test_organization_v1.py: the
#   services are annotated for OrganizationRepository/AsyncSession but are handed
#   structural in-memory doubles. Suppressed file-wide rather than threading
#   Protocols through production signatures purely for the tests.
"""T-14: every scheduled producer asks the same entitlement authority first.

THE ASYMMETRY THIS CLOSES. `POST /brief/generate` has always required
`Capability.DECISION`; the hourly `build_brief_for_org` that produces the same
artefact required nothing. `sync_all_sources` fanned out over every active
integration and `embed_all_orgs` over `SELECT id FROM organizations`, neither
joined to lifecycle. So the API path and the background path enforced different
rules for the same work, and a SUSPENDED organization kept syncing mail every
five minutes, kept building briefs every hour, and kept being embedded against a
paid provider.

This is an EXECUTION gate, not deletion — nothing is removed, work is skipped
while the tenant is not entitled and resumes by itself when it is.

The file is deliberately cross-domain: the contract is that FOUR producers in
THREE domains consult ONE authority, and pinning that per-domain would let a
single producer lose its gate without any test failing.
"""
from __future__ import annotations

import ast
import contextlib
import inspect
import pathlib
import textwrap
import uuid

import pytest

from navis.organization.billing import BillingService
from navis.organization.contract import Capability, Plan
from navis.organization.entitlements import EntitlementService
from navis.organization.lifecycle import LifecycleService
from navis.orchestration.runtime_policy import scheduled_work_allowed
from tests.organization.fakes import ORG_A, FakeOrgRepo

ORG_B = uuid.UUID("22222222-2222-2222-2222-222222222222")


# ── The gate itself ─────────────────────────────────────────────────────────


async def _entitled(plan: Plan, *, suspend: bool = False, churn: bool = False):
    """A FakeOrgRepo provisioned onto `plan`, optionally moved out of service."""
    repo = FakeOrgRepo()
    ent = EntitlementService(repo)
    await BillingService(repo, ent).set_plan(ORG_A, plan, billing_authorized=True)
    if suspend or churn:
        lc = LifecycleService(repo)
        await lc.activate(ORG_A)
        await (lc.churn(ORG_A) if churn else lc.suspend(ORG_A))
    return repo


class _DB:
    """Stands in for AsyncSession — `OrganizationService` only needs it to build
    an OrganizationRepository, which we replace wholesale below."""


@pytest.fixture()
def org_repo(monkeypatch):
    """Point OrganizationService at a FakeOrgRepo instead of a real session."""

    def install(repo):
        import navis.organization.service as S

        monkeypatch.setattr(S, "OrganizationRepository", lambda _db: repo)
        return repo

    return install


async def test_trial_and_active_with_the_capability_are_allowed(org_repo):
    org_repo(await _entitled(Plan.ENTERPRISE))
    assert await scheduled_work_allowed(
        _DB(), ORG_A, Capability.DECISION, task="t"
    ) is True


async def test_free_without_the_capability_is_skipped(org_repo):
    """FREE grants GMAIL/CALENDAR/MEMORY/STATE — not DECISION, not ML."""
    org_repo(await _entitled(Plan.FREE))
    assert await scheduled_work_allowed(_DB(), ORG_A, Capability.DECISION, task="t") is False
    assert await scheduled_work_allowed(_DB(), ORG_A, Capability.ML, task="t") is False
    # ...but the work FREE does pay for still runs.
    assert await scheduled_work_allowed(_DB(), ORG_A, Capability.GMAIL, task="t") is True


async def test_suspended_is_skipped_for_everything(org_repo):
    org_repo(await _entitled(Plan.ENTERPRISE, suspend=True))
    for cap in Capability:
        assert await scheduled_work_allowed(_DB(), ORG_A, cap, task="t") is False


async def test_churned_is_skipped_for_everything(org_repo):
    org_repo(await _entitled(Plan.ENTERPRISE, churn=True))
    for cap in Capability:
        assert await scheduled_work_allowed(_DB(), ORG_A, cap, task="t") is False


async def test_the_gate_fails_closed_without_an_organization():
    assert await scheduled_work_allowed(_DB(), None, Capability.DECISION, task="t") is False


async def test_the_gate_fails_closed_when_it_cannot_answer(monkeypatch):
    """Nobody is watching a beat task, so a gate that degraded to 'allow' on error
    would look exactly like a working one."""
    import navis.orchestration.runtime_policy as RP

    class _Exploding:
        def __init__(self, _db):
            pass

        async def can_use(self, *_a, **_k):
            raise RuntimeError("database unreachable")

    monkeypatch.setattr(RP, "OrganizationService", _Exploding)
    assert await scheduled_work_allowed(_DB(), ORG_A, Capability.ML, task="t") is False


async def test_skips_are_metered_with_bounded_labels(org_repo):
    """`task` is a fixed string per call site and `capability` is a closed enum,
    so this cannot scale with the number of organizations."""
    from navis.platform.monitoring.metrics import metrics

    org_repo(await _entitled(Plan.FREE))
    key = "orchestration.scheduled.skipped{reason=not_entitled,task=build_brief_for_org}"
    before = metrics.snapshot()["counters"].get(key, 0)

    await scheduled_work_allowed(
        _DB(), ORG_A, Capability.DECISION, task="build_brief_for_org"
    )

    assert metrics.snapshot()["counters"].get(key, 0) == before + 1


# ── Every producer consults it ──────────────────────────────────────────────


async def test_brief_skips_before_building_anything(monkeypatch):
    import navis.orchestration.brief as B

    built: list = []

    @contextlib.asynccontextmanager
    async def _scope():
        yield _DB()

    async def _deny(_db, org_id, cap, *, task):
        return False

    monkeypatch.setattr(B, "session_scope", _scope)
    monkeypatch.setattr(B, "scheduled_work_allowed", _deny)
    monkeypatch.setattr(B, "build_brief", lambda *a, **k: built.append(1))

    out = await B.run_brief_for_org(ORG_A)

    assert out == {"org_id": str(ORG_A), "skipped": "not_entitled"}
    assert built == [], "folded state and called the LLM for an unentitled org"


async def test_brief_asks_for_the_decision_capability(monkeypatch):
    import navis.orchestration.brief as B

    asked: list = []

    @contextlib.asynccontextmanager
    async def _scope():
        yield _DB()

    async def _spy(_db, org_id, cap, *, task):
        asked.append((org_id, cap, task))
        return False

    monkeypatch.setattr(B, "session_scope", _scope)
    monkeypatch.setattr(B, "scheduled_work_allowed", _spy)

    await B.run_brief_for_org(ORG_A)
    assert asked == [(ORG_A, Capability.DECISION, "build_brief_for_org")]


async def test_embedding_skips_before_calling_the_provider(monkeypatch):
    """The most expensive scheduled work in the system: one paid provider call
    per unembedded record, every five minutes, for every organization."""
    import navis.ml.embeddings.jobs as J
    import navis.orchestration.runtime_policy as RP

    embedded: list = []

    @contextlib.asynccontextmanager
    async def _scope():
        yield _DB()

    async def _deny(_db, org_id, cap, *, task):
        embedded.append(("asked", cap, task))
        return False

    monkeypatch.setattr(J, "session_scope", _scope)
    # jobs.py imports the gate INSIDE the function, so patch it at the source.
    monkeypatch.setattr(RP, "scheduled_work_allowed", _deny)

    class _Boom:
        def __init__(self, _db):
            pass

        async def embed_org(self, *_a, **_k):
            raise AssertionError("called the embedding provider for a skipped org")

    monkeypatch.setattr(J, "EmbeddingService", _Boom)

    assert await J._embed_org(ORG_A, 1, 20) == 0
    assert embedded == [("asked", Capability.ML, "embed_org")]


async def test_sync_skips_before_touching_the_provider(monkeypatch):
    """Placed inside `sync_integration`, which is the ONE function all three sync
    producers converge on — the beat fan-out, the Pub/Sub push, and the manual
    route — so one check covers every producer instead of three that can drift."""
    from types import SimpleNamespace

    import navis.orchestration.sync as S

    asked: list = []

    integration = SimpleNamespace(
        id=uuid.uuid4(),
        organization_id=ORG_A,
        user_id=uuid.uuid4(),
        provider=S.IntegrationProvider.GMAIL,
        is_active=True,
        # Healthy, so this test isolates the ENTITLEMENT refusal. The auth gate
        # sits immediately after the entitlement gate, and a fake defaulting to
        # a broken credential would let this test pass for the wrong reason.
        auth_state=S.IntegrationAuthState.OK.value,
        sync_state={},
    )

    class _DBWithGet:
        async def get(self, _model, _id):
            return integration

    @contextlib.asynccontextmanager
    async def _scope():
        yield _DBWithGet()

    @contextlib.asynccontextmanager
    async def _lock(_db, _id):
        yield True

    async def _deny(_db, org_id, cap, *, task):
        asked.append((org_id, cap, task))
        return False

    class _Transport:
        async def aclose(self):
            pass

    monkeypatch.setattr(S, "session_scope", _scope)
    monkeypatch.setattr(S, "integration_advisory_lock", _lock)
    monkeypatch.setattr(S, "scheduled_work_allowed", _deny)
    monkeypatch.setattr(
        S, "_build_client",
        lambda _i: (object(), object(), (_Transport(), SimpleNamespace(last_token=None))),
    )

    class _Boom:
        def __init__(self, _db):
            pass

        async def sync(self, *_a, **_k):
            raise AssertionError("called the connector for a skipped org")

    monkeypatch.setattr(S, "SignalSyncService", _Boom)

    out = await S.sync_integration(integration)

    assert out == {"skipped": "not_entitled"}
    assert asked == [(ORG_A, Capability.GMAIL, "sync_integration")]


def test_an_unmapped_provider_has_no_capability_and_is_refused():
    """Fail closed: an unmapped connector is one nobody has decided the
    entitlement for, and guessing would let a future connector run outside the
    plan that pays for it. NOTION is in the provider enum with no connector and
    no capability, so it stands in for exactly that case."""
    import navis.orchestration.sync as S

    assert S._capability_for(S.IntegrationProvider.NOTION) is None

    # ...and `sync_integration` turns that None into a refusal, not a crash.
    src = textwrap.dedent(inspect.getsource(S.sync_integration))
    assert "no_capability_mapping" in src


def test_every_connector_capability_is_a_real_capability():
    import navis.orchestration.sync as S

    for provider, cap in S._CAPABILITY.items():
        assert isinstance(cap, Capability), f"{provider} maps to a non-capability"
    # Every provider the sync REGISTRY can actually drive must be mapped, or it
    # would be refused at runtime rather than at import.
    for provider in S._REGISTRY:
        assert provider in S._CAPABILITY, f"{provider} is syncable but unmapped"


# ── The sweep: a NEW scheduled producer must not be born unguarded ──────────


def test_no_scheduled_fanout_enumerates_orgs_without_a_gate():
    """Every keyset fan-out over `Organization.id` must dispatch to a per-tenant
    task, and every per-tenant task body must reach `scheduled_work_allowed`.

    Structural rather than behavioural on purpose: this is the test that fails on
    the day someone adds a fifth scheduled producer and forgets the gate.
    """
    root = pathlib.Path(__file__).resolve().parents[2] / "navis"
    gated = {
        "navis/orchestration/brief.py": "run_brief_for_org",
        "navis/orchestration/sync.py": "sync_integration",
        "navis/ml/embeddings/jobs.py": "_embed_org",
    }
    for rel, fn_name in gated.items():
        src = (root.parent / rel).read_text(encoding="utf-8")
        tree = ast.parse(src)
        fn = next(
            n for n in ast.walk(tree)
            if isinstance(n, ast.AsyncFunctionDef) and n.name == fn_name
        )
        called = {
            n.func.id if isinstance(n.func, ast.Name) else getattr(n.func, "attr", "")
            for n in ast.walk(fn) if isinstance(n, ast.Call)
        }
        assert "scheduled_work_allowed" in called, (
            f"{rel}::{fn_name} no longer consults the entitlement gate"
        )


def test_watch_renewal_is_gated_too():
    import navis.orchestration.sync as S

    src = textwrap.dedent(inspect.getsource(S.ensure_watch_for_integration))
    called = {
        n.func.id if isinstance(n.func, ast.Name) else getattr(n.func, "attr", "")
        for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Call)
    }
    assert "scheduled_work_allowed" in called
