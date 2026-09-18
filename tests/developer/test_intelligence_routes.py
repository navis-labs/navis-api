"""The three intelligence reads on the Developer boundary.

WHAT THESE PIN. Not "does the route return 200" — the Postgres boundary module
sweeps that, along with isolation, revocation and permission denial, for every
path listed in its `DEVELOPER_ROUTES`. These pin the four properties that are
decided by how the handlers are WIRED, that a happy-path run by a founder
holding every permission would not notice, and that are each one line away from
being wrong:

  1. `/decisions` is a READ. It must never reach `build_brief`, which folds
     State and writes `reasoning_signals`, `intelligence_insights` and
     `daily_briefs` — the most expensive way a read could mutate the pipeline.
  2. `/reasoning` FAILS CLOSED on its join. A conclusion with no observation at
     the current watermark reports `current: null`, never a substituted
     neighbouring observation — the defect the observation table was split out
     to remove.
  3. Freshness is TRI-STATE. "State has no snapshot" is not "fresh".
  4. Each route gates on the permission and capability that already existed in
     the ONE vocabulary, and invents no word of its own.
"""
from __future__ import annotations

import importlib
import inspect
import uuid
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from navis.developer.gateway import DeveloperContext

# `importlib`, not `from navis.developer import router`: that package's
# `__init__` binds the name `router` to the APIRouter OBJECT (see its own note on
# why `agents_router` is re-exported under a different name), so the plain import
# would hand these tests a router instead of the module whose handlers they call.
module = importlib.import_module("navis.developer.router")
from navis.identity.authz import Principal, PrincipalKind
from navis.identity.permissions import VALID_PERMISSIONS, Permission
from navis.organization.contract import Capability

ORG = uuid.UUID("eeee0000-0000-4000-8000-00000000000a")
OTHER_ORG = uuid.UUID("eeee0000-0000-4000-8000-00000000000b")
WATERMARK = "1a605bc5-7c54-51e2-828b-48ddad44dbde"
OLDER = "f8675a7f-77cc-538d-9bd3-f198344f4b53"
AS_OF = datetime(2026, 8, 24, 18, 46, 59, tzinfo=timezone.utc)


def _ctx(*permissions: str) -> DeveloperContext:
    """A context shaped exactly as `DeveloperContext.of` builds one."""
    subject = uuid.uuid4()
    perms = frozenset(permissions)
    return DeveloperContext(
        principal=Principal(
            kind=PrincipalKind.SERVICE_ACCOUNT,
            org_id=ORG,
            subject_id=subject,
            permissions=perms,
        ),
        organization_id=ORG,
        kind=PrincipalKind.SERVICE_ACCOUNT.value,
        subject_id=subject,
        service_account_id=subject,
        credential_id=None,
        permissions=perms,
    )


class _DB:
    """A session that fails loudly if a handler tries to use it directly.

    Every read on these routes goes through a domain service or the reasoning
    store, all of which are stubbed per test. A handler reaching past them with
    a query of its own is a second read semantics, and it shows up here.
    """

    async def execute(self, *a, **k):  # pragma: no cover - must not be reached
        raise AssertionError("handler issued its own query instead of using a service")

    scalar = scalars = get = execute


def _signal(
    signal_id: str,
    *,
    signal_type: str = "commitment_slipping",
    severity: float = 0.5,
    confidence: float = 0.5,
    why: str = "first observation",
) -> SimpleNamespace:
    return SimpleNamespace(
        signal_id=uuid.UUID(signal_id),
        signal_type=signal_type,
        entity_type="commitment",
        entity_id="f33778d8-a54e-50bb-87b9-2d74d83584e2",
        title="Commitment slipping",
        why=why,
        evidence={"age_days": 7.16},
        confidence=confidence,
        severity=severity,
        engine_name="commitment_reasoning",
        engine_version="2.0.0",
        state_as_of=AS_OF,
        state_watermark=OLDER,
        reasoning_version=4,
    )


def _observation(signal_id: str, *, severity: float, confidence: float) -> SimpleNamespace:
    return SimpleNamespace(
        signal_id=uuid.UUID(signal_id),
        why="current observation",
        confidence=confidence,
        severity=severity,
        evidence={"age_days": 14.27},
        state_as_of=AS_OF,
        state_watermark=WATERMARK,
    )


# ── 1. `/decisions` is a read ───────────────────────────────────────────────


def _imported_and_called(handler) -> set[str]:
    """Every name the handler IMPORTS or CALLS, read from its AST.

    Deliberately not a substring search over `inspect.getsource`: this handler's
    docstring names `build_brief` in order to explain why it does not call it,
    and a text match cannot tell the prose from the code. The AST can.
    """
    import ast
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(handler)))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Name):
                names.add(fn.id)
            elif isinstance(fn, ast.Attribute):
                names.add(fn.attr)
    return names


def test_decisions_never_reaches_the_brief_builder():
    """`/brief/today` rebuilds a brief whose watermark has moved. This route
    must compose the two READ halves instead, or a Developer read writes four
    tables and spends the whole pipeline."""
    names = _imported_and_called(module.developer_decisions)

    assert "build_brief" not in names, "a Developer read must not build a brief"
    assert "brief_for_today" not in names, "that helper rebuilds on a moved watermark"
    assert {"latest_brief", "_with_executions", "current_state_watermark"} <= names


async def test_decisions_returns_null_without_building_one(monkeypatch):
    """An org that has never had a brief reports `brief: null` — a real state,
    and one a read must not paper over by manufacturing a brief."""
    import navis.decisions.service as decisions_service
    import navis.orchestration.brief as orchestration

    class _Svc:
        def __init__(self, db):
            pass

        async def latest_brief(self, org_id):
            assert org_id == ORG
            return None

    monkeypatch.setattr(decisions_service, "DecisionService", _Svc)
    monkeypatch.setattr(
        orchestration, "current_state_watermark",
        lambda db, org_id: _async(WATERMARK),
    )

    body = await module.developer_decisions(ctx=_ctx(), db=_DB())

    assert body["brief"] is None
    assert body["decisions"] == [] and body["count"] == 0
    assert body["current_state_watermark"] == WATERMARK
    # No brief means no comparison, so staleness is unknown rather than False.
    assert body["stale"] is None


@pytest.mark.parametrize(
    "brief_watermark, current, expected",
    [
        (WATERMARK, WATERMARK, False),   # describes the current snapshot
        (OLDER, WATERMARK, True),        # State has moved past it
        (OLDER, None, None),             # no snapshot: NOT "fresh"
    ],
)
async def test_staleness_is_tri_state(monkeypatch, brief_watermark, current, expected):
    """Same rule as `brief_router._freshness.current`: null means the comparison
    could not be made. A client rendering null as up-to-date is making a claim
    this route deliberately refuses to make."""
    _stub_brief(monkeypatch, state_watermark=brief_watermark, current=current)

    body = await module.developer_decisions(ctx=_ctx(), db=_DB())

    assert body["stale"] is expected
    assert body["state_watermark"] == brief_watermark
    assert body["current_state_watermark"] == current


async def test_decisions_publishes_the_web_edges_own_enrichment(monkeypatch):
    """The decision list is `_with_executions`' output verbatim — execution,
    action, provenance and explanation included. A Developer-specific reshaping
    would be a second wire vocabulary for one set of facts."""
    _stub_brief(monkeypatch, state_watermark=WATERMARK, current=WATERMARK)

    body = await module.developer_decisions(ctx=_ctx(), db=_DB())

    assert body["count"] == 1
    assert body["decisions"][0]["explanation"] == "because"
    assert body["decisions"][0]["provenance"] == {"verified": True}
    assert body["as_of"] == AS_OF.isoformat()
    assert body["reasoning_version"] == 4


# ── 2. `/reasoning` fails closed on the join ────────────────────────────────


async def test_unobserved_conclusion_reports_current_null(monkeypatch):
    """A conclusion with no observation at the current watermark is reported as
    unobserved. It is never filled from the identity row, whose values are the
    FIRST observation — that substitution is how a live decision came to report
    11.85 days against a cited signal recording 7.16."""
    a = "d9170d07-3936-5a2e-abae-b0282d91fb18"
    b = "2af920ca-5df8-5164-9a5c-1dde1bb29798"
    _stub_reasoning(
        monkeypatch,
        signals=[_signal(a, severity=0.63, confidence=0.73), _signal(b, severity=0.60)],
        observations={a: _observation(a, severity=0.8756, confidence=0.9)},
    )

    body = await module.developer_reasoning(
        signal_type=None, limit=100, ctx=_ctx(), db=_DB()
    )

    observed, unobserved = body["signals"][0], body["signals"][1]
    assert observed["signal_id"] == a
    assert observed["current"]["severity"] == 0.8756
    assert observed["current"]["why"] == "current observation"
    # The identity row is still published, and still says what it says.
    assert observed["first_observed"]["severity"] == 0.63
    assert observed["first_observed"]["why"] == "first observation"

    assert unobserved["signal_id"] == b
    assert unobserved["current"] is None
    assert unobserved["first_observed"]["severity"] == 0.60


async def test_reasoning_ranks_on_current_severity_not_the_stored_one(monkeypatch):
    """The stored severity is the first observation's. Ranking a list on it
    would be the same stale-value mistake one layer up — so an unobserved
    conclusion sorts last rather than competing on a number that no longer
    describes anything."""
    high_now = "aaaaaaaa-0000-4000-8000-00000000000a"
    high_then = "bbbbbbbb-0000-4000-8000-00000000000b"
    _stub_reasoning(
        monkeypatch,
        # `high_then` looks the most severe on its identity row and has no
        # current observation; `high_now` looks mild and is the live problem.
        signals=[_signal(high_then, severity=0.99), _signal(high_now, severity=0.10)],
        observations={high_now: _observation(high_now, severity=0.87, confidence=0.9)},
    )

    body = await module.developer_reasoning(
        signal_type=None, limit=100, ctx=_ctx(), db=_DB()
    )

    assert [s["signal_id"] for s in body["signals"]] == [high_now, high_then]
    assert body["order"] == "current.severity.desc"


async def test_reasoning_reports_the_watermark_it_resolved_current_against(monkeypatch):
    """Without it a caller cannot tell WHICH snapshot `current` describes, and a
    null `current` is indistinguishable from a build that does not report one."""
    a = "d9170d07-3936-5a2e-abae-b0282d91fb18"
    _stub_reasoning(monkeypatch, signals=[_signal(a)], observations={})

    body = await module.developer_reasoning(
        signal_type=None, limit=100, ctx=_ctx(), db=_DB()
    )

    assert body["state_watermark"] == WATERMARK
    assert body["signals"][0]["current"] is None


@pytest.mark.parametrize("asked, expected", [(0, 1), (1, 1), (10_000, 200), (-5, 1)])
async def test_reasoning_limit_is_clamped(monkeypatch, asked, expected):
    """The log grows with every brief, so an unbounded read is a read that gets
    slower forever. Clamped rather than 422'd, matching every other read handler
    in this codebase — those are called directly in tests, where a `Query`
    default arrives as a `Query` object instead of an int."""
    seen: dict = {}
    _stub_reasoning(monkeypatch, signals=[], observations={}, capture=seen)

    body = await module.developer_reasoning(
        signal_type=None, limit=asked, ctx=_ctx(), db=_DB()
    )

    assert seen["limit"] == expected
    assert body["limit"] == expected


async def test_reasoning_passes_the_type_filter_through_as_a_filter(monkeypatch):
    """`signal_type` narrows a query that is ALREADY org-scoped. It is applied
    on top of the tenant predicate inside the store, so an unrecognised value
    matches nothing rather than widening the scope."""
    seen: dict = {}
    _stub_reasoning(monkeypatch, signals=[], observations={}, capture=seen)

    body = await module.developer_reasoning(
        signal_type="commitment_slipping", limit=50, ctx=_ctx(), db=_DB()
    )

    assert seen["org_id"] == ORG
    assert seen["signal_type"] == "commitment_slipping"
    assert body["signal_type"] == "commitment_slipping"


# ── 3. Insights are Intelligence's own projection ───────────────────────────


async def test_insights_publish_intelligences_own_serialization(monkeypatch):
    """Read through `IntelligenceService.get_insights` and serialized with
    Intelligence's own `_serialize`, so the Developer API and the web app cannot
    disagree about what an insight is or which generation of it is current."""
    import navis.intelligence.repository as repo_module
    import navis.intelligence.service as service_module

    row = SimpleNamespace(
        insight_id=uuid.UUID("b123db6f-3ac5-59e0-b72b-3cc38c37dab6"),
        insight_type="situation_intelligence",
        key="relationship:e57ffa81",
        title="cooling relationship and an unresolved commitment",
        summary="Reasoning reached 2 separate conclusions.",
        evidence={"signals": []},
        sample_size=2,
        confidence=0.8858,
        supporting_learning_event_ids=[],
        learning_version=1,
        created_at=AS_OF,
    )

    class _Repo:
        def __init__(self, db):
            pass

    class _Svc:
        def __init__(self, repo):
            pass

        async def get_insights(self, org_id):
            assert org_id == ORG
            return [row]

    monkeypatch.setattr(repo_module, "IntelligenceRepository", _Repo)
    monkeypatch.setattr(service_module, "IntelligenceService", _Svc)

    body = await module.developer_insights(ctx=_ctx(), db=_DB())

    assert body["count"] == 1
    got = body["insights"][0]
    assert got["insight_id"] == str(row.insight_id)
    assert got["insight_type"] == "situation_intelligence"
    assert got["confidence"] == 0.8858
    assert got["created_at"] == AS_OF.isoformat()


# ── 4. The gates are the existing vocabulary ────────────────────────────────


@pytest.mark.parametrize(
    "handler, permission, capability",
    [
        ("developer_insights", "VIEW_INTELLIGENCE", "INTELLIGENCE"),
        ("developer_decisions", "VIEW_DECISIONS", "DECISION"),
        ("developer_reasoning", "VIEW_REASONING", "REASONING"),
    ],
)
def test_each_route_gates_on_the_layer_it_reads(handler, permission, capability):
    """The permission names the LAYER, not the boundary. Borrowing a
    neighbouring word (as `/outcomes` briefly borrowed `VIEW_DECISIONS`) widens
    the least-privileged role's meaning by way of a machine edge.

    Both gates, always: a permission answers "may this CREDENTIAL", a capability
    answers "is this TENANT entitled", and the web-edge routes these sit beside
    check only the second. Pinned structurally for every Developer route by
    `test_every_developer_route_gates_on_a_permission_and_a_capability`; named
    per route here so the mapping to a layer is explicit.
    """
    source = inspect.getsource(getattr(module, handler))

    assert f"Permission.{permission}" in source
    assert f"Capability.{capability}" in source
    assert getattr(Permission, permission) in VALID_PERMISSIONS
    assert getattr(Capability, capability) in set(Capability)


def test_the_three_permissions_are_not_developer_specific():
    """No `developer_admin`, no second RBAC, no scope vocabulary of its own: an
    agent and a human are authorized by the same words.

    RESOLVED THROUGH THE FULL EVALUATOR SINCE P1-6. These three are Developer-
    surface permissions, so a founder now reaches them through
    `memberships.developer_role` rather than through their Application role —
    but through the SAME `effective_permissions` and the same vocabulary, which
    is what this test is actually about. Asserting against
    `default_permissions_for_role` alone would now be asserting that one half of
    a two-half resolution contains everything, which was never the claim.
    """
    from navis.identity.permissions import effective_permissions

    founder = effective_permissions("founder", developer_role="owner")
    for permission in (
        Permission.VIEW_INTELLIGENCE, Permission.VIEW_DECISIONS, Permission.VIEW_REASONING
    ):
        assert permission in VALID_PERMISSIONS
        assert permission in founder, f"{permission} unreachable by a real role"


def test_no_intelligence_route_accepts_an_organization():
    """The tenant comes from the credential. These handlers take no identity
    parameter in any position — pinned globally by
    `test_no_developer_route_accepts_an_identity_parameter`, and named here so
    the reason is attached to the routes themselves."""
    for handler in ("developer_insights", "developer_decisions", "developer_reasoning"):
        params = set(inspect.signature(getattr(module, handler)).parameters)
        assert not params & {"organization_id", "org_id", "tenant_id", "user_id"}
        assert "ctx" in params and "db" in params


# ── helpers ─────────────────────────────────────────────────────────────────


def _async(value):
    """A coroutine returning `value` — for stubbing an awaited helper."""
    async def _run():
        return value

    return _run()


def _stub_brief(monkeypatch, *, state_watermark, current):
    """One brief, enriched as `_with_executions` enriches it."""
    import navis.api.brief_router as brief_router
    import navis.decisions.service as decisions_service
    import navis.orchestration.brief as orchestration

    brief = SimpleNamespace(
        id=uuid.uuid4(), organization_id=ORG, as_of=AS_OF,
        reasoning_version=4, state_watermark=state_watermark, decisions=[{}],
    )

    class _Svc:
        def __init__(self, db):
            pass

        async def latest_brief(self, org_id):
            return brief

    async def _enriched(db, b, freshness=None):
        assert b is brief
        return {
            "id": str(b.id),
            "as_of": b.as_of.isoformat(),
            "reasoning_version": b.reasoning_version,
            "state_watermark": b.state_watermark,
            "decisions": [{
                "decision_id": "595a212e", "title": "Commitment slipping",
                "execution": None, "action": None,
                "provenance": {"verified": True}, "explanation": "because",
            }],
        }

    monkeypatch.setattr(decisions_service, "DecisionService", _Svc)
    monkeypatch.setattr(brief_router, "_with_executions", _enriched)
    monkeypatch.setattr(
        orchestration, "current_state_watermark",
        lambda db, org_id: _async(current),
    )


def _stub_reasoning(monkeypatch, *, signals, observations, capture=None):
    """The reasoning store's two reads, with the arguments they were given."""
    import navis.orchestration.brief as orchestration
    import navis.reasoning.store as store

    async def _list(db, org_id, *, signal_type=None, entity_id=None,
                    reasoning_version=None, limit=200):
        if capture is not None:
            capture.update(
                org_id=org_id, signal_type=signal_type, limit=limit,
                entity_id=entity_id,
            )
        return list(signals)

    async def _observed(db, org_id, signal_ids, *, state_watermark):
        assert org_id == ORG, "the observation join must carry the tenant predicate"
        assert state_watermark == WATERMARK
        ids = {str(s) for s in signal_ids}
        return {k: v for k, v in observations.items() if k in ids}

    monkeypatch.setattr(store, "list_signals", _list)
    monkeypatch.setattr(store, "observations_at", _observed)
    monkeypatch.setattr(
        orchestration, "current_state_watermark",
        lambda db, org_id: _async(WATERMARK),
    )
