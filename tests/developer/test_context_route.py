"""`POST /developer/context` — the task-aware bundle.

WHAT THESE PIN. Four properties that are decided by wiring, are invisible in a
happy-path call by a founder holding every permission, and are each one line
away from being wrong:

  1. IT DOES NOT WRITE. The State read must go through `read_only_state`, which
     never folds, rebuilds or disposes. The alternatives both mutate:
     `incremental_update` saves a snapshot every call, `get_state` deletes and
     rebuilds a corrupt one. A read that folds an organization's memory as a
     side effect of being asked a question is not a read.
  2. IT IS THE SAME COMPOSER `/chat` USES. Not a copy, not a Developer variant.
     If these two ever diverge, an agent and a founder are looking at different
     organizations.
  3. IT DEGRADES BY PERMISSION, AND SAYS SO. A credential that cannot read
     Intelligence gets a bundle without it AND an explicit note naming what was
     withheld — because a thin bundle and a quiet organization are
     indistinguishable otherwise.
  4. NO MODEL IS CALLED. The bundle is the product; the caller does the
     reasoning.
"""
from __future__ import annotations

import ast
import importlib
import inspect
import textwrap
import uuid
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from navis.developer.gateway import DeveloperContext
from navis.identity.authz import Principal, PrincipalKind
from navis.identity.permissions import Permission
from navis.ml.rag.grounding import GroundedContext, Grounding
from navis.orchestration.context import ComposedContext, ContextLayers
from navis.organization.contract import Capability

# `importlib`, not `from navis.developer import ...`: that package's `__init__`
# binds `router` to the APIRouter object, so a plain import can hand back a
# router where a module was meant.
mod = importlib.import_module("navis.developer.context_router")

ORG = uuid.UUID("dddd0000-0000-4000-8000-00000000000c")
AS_OF = datetime(2026, 8, 24, 18, 46, 59, tzinfo=timezone.utc)
#: DELIBERATELY EARLIER THAN `AS_OF`, by eighteen days — the exact shape the
#: live organization exhibits. `as_of` is when Navis last folded; this is when
#: the newest fact actually happened. Choosing two values that differ is the
#: whole point of the fixture: equal ones would let a handler that returned
#: `as_of` twice pass every assertion below.
NEWEST_FACT_AT = datetime(2026, 8, 6, 18, 46, 59, tzinfo=timezone.utc)
WATERMARK = "1a605bc5-7c54-51e2-828b-48ddad44dbde"

ALL_PERMISSIONS = frozenset({
    Permission.VIEW_MEMORY, Permission.VIEW_STATE, Permission.VIEW_REASONING,
    Permission.VIEW_INTELLIGENCE, Permission.VIEW_LEARNING,
})
ALL_CAPABILITIES = {
    Capability.MEMORY.value: True, Capability.STATE.value: True,
    Capability.REASONING.value: True, Capability.INTELLIGENCE.value: True,
    Capability.LEARNING.value: True,
}


def _ctx(permissions: frozenset[str]) -> DeveloperContext:
    subject = uuid.uuid4()
    return DeveloperContext(
        principal=Principal(
            kind=PrincipalKind.SERVICE_ACCOUNT, org_id=ORG,
            subject_id=subject, permissions=permissions,
        ),
        organization_id=ORG,
        kind=PrincipalKind.SERVICE_ACCOUNT.value,
        subject_id=subject,
        service_account_id=subject,
        credential_id=None,
        permissions=permissions,
    )


class _DB:
    """A session no handler on this path may query directly."""

    async def execute(self, *a, **k):  # pragma: no cover
        raise AssertionError("the context route issued its own query")


def _stub(monkeypatch, *, capabilities=None, facts=None, state_available=True):
    """Capture what the composer was asked for, and answer with a fixed bundle."""
    import navis.developer.context_router as router_module
    import navis.organization.service as org_service

    seen: dict = {}

    class _Entitlements:
        async def snapshot(self, org_id):
            seen["entitlements_org"] = org_id
            return ALL_CAPABILITIES if capabilities is None else capabilities

    class _OrgService:
        def __init__(self, db):
            pass

        entitlements = _Entitlements()

    async def _build(db, org_id, task, **kw):
        seen["org_id"] = org_id
        seen["task"] = task
        seen.update(kw)
        bundle = GroundedContext(
            org_id=str(org_id), question=task,
            facts=list(facts if facts is not None else [
                Grounding("memory", "m1", "an email about billing"),
                Grounding("state", "e1", "commitment: ship the invoice fix"),
            ]),
            token_estimate=42, as_of=AS_OF,
        )
        return ComposedContext(
            context=bundle,
            derived=[f for f in bundle.facts if f.kind in {"state", "reasoning"}],
            state_watermark=WATERMARK if state_available else None,
            as_of=AS_OF if state_available else None,
            newest_fact_at=NEWEST_FACT_AT if state_available else None,
            state_available=state_available,
            subject=None,
            expanded=False,
            layers=kw.get("layers") or ContextLayers(),
        )

    monkeypatch.setattr(org_service, "OrganizationService", _OrgService)
    monkeypatch.setattr(router_module, "build_grounded_context", _build)
    return seen


# ── 1. It does not write ────────────────────────────────────────────────────


async def test_the_state_read_is_read_only(monkeypatch):
    """`read_only=True`, always and unconditionally. This is the whole
    difference between this route and `GET /developer/context`, which folds and
    saves a snapshot on every call."""
    seen = _stub(monkeypatch)

    await mod.developer_context_for_task(
        body=mod.DeveloperContextIn(task="change billing"),
        ctx=_ctx(ALL_PERMISSIONS), db=_DB(),
    )

    assert seen["read_only"] is True


def _identifiers(module_or_fn) -> set[str]:
    """Every name this CODE references. Docstrings and comments excluded.

    Deliberately not a substring search: this module's own prose names
    `incremental_update` and `rebuild_org` in order to explain why it does not
    call them, and a text match cannot tell the explanation from the call.
    """
    tree = ast.parse(textwrap.dedent(inspect.getsource(module_or_fn)))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.ImportFrom):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
    return names


def test_the_handler_never_asks_for_a_mutating_state_read():
    """Structural backstop for the test above: the three State entry points that
    write must not be reachable from this module by name."""
    referenced = _identifiers(mod)
    for mutating in ("incremental_update", "rebuild_org", "_dispose"):
        assert mutating not in referenced, (
            f"{mutating} writes; this route must not call it"
        )


def test_no_model_is_called():
    """The bundle IS the answer. A generator here would put a Navis model between
    an agent and its own organization's facts — and the agent is a better one."""
    names = set()
    tree = ast.parse(textwrap.dedent(inspect.getsource(mod)))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    assert not any(n.startswith("navis.ml.generation") for n in names)
    source = inspect.getsource(mod)
    for generator in ("explanation(", "recommendation(", "summary(", "MLProvider"):
        assert generator not in source, f"{generator} would generate an answer"


# ── 2. It is the same composer /chat uses ───────────────────────────────────


def test_it_calls_the_shared_composer_not_a_local_one():
    """One implementation, two callers. A Developer-specific composition is the
    failure the whole extraction exists to prevent."""
    source = inspect.getsource(mod)
    assert "from navis.orchestration.context import" in source
    assert "build_grounded_context" in source

    import navis.api.chat_router as chat

    assert "build_grounded_context" in inspect.getsource(chat), (
        "/chat must call the same composer, or the two surfaces have diverged"
    )


def test_chat_keeps_the_mutating_state_read():
    """The Application path is NOT changed to suit the Developer one.

    `/chat` composes WITHOUT `read_only`, so State still repairs an absent or
    corrupt snapshot exactly as it did before the extraction. Read off the call
    itself rather than the source text — the docstring beside it explains this
    choice in words, and a substring match would read the explanation as the
    thing it describes.
    """
    import navis.api.chat_router as chat

    tree = ast.parse(textwrap.dedent(inspect.getsource(chat._grounded_answer)))
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "build_grounded_context"
    ]
    assert len(calls) == 1, "the chat path must compose exactly once"
    passed = {kw.arg for kw in calls[0].keywords}
    assert "read_only" not in passed, (
        "the chat path must not have acquired the Developer read-only posture"
    )


# ── 3. It degrades by permission, and says so ───────────────────────────────


async def test_a_fully_scoped_credential_gets_every_layer(monkeypatch):
    seen = _stub(monkeypatch)

    out = await mod.developer_context_for_task(
        body=mod.DeveloperContextIn(task="change billing"),
        ctx=_ctx(ALL_PERMISSIONS), db=_DB(),
    )

    layers: ContextLayers = seen["layers"]
    assert (layers.state, layers.reasoning, layers.intelligence, layers.learning) == (
        True, True, True, True
    )
    assert out["layers"]["withheld"] == []
    assert set(out["layers"]["included"]) == {
        "memory", "state", "reasoning", "intelligence", "learning"
    }


@pytest.mark.parametrize(
    "missing, layer, requires",
    [
        (Permission.VIEW_STATE, "state", "view_state"),
        (Permission.VIEW_REASONING, "reasoning", "view_reasoning"),
        (Permission.VIEW_INTELLIGENCE, "intelligence", "view_intelligence"),
        (Permission.VIEW_LEARNING, "learning", "view_learning"),
    ],
)
async def test_a_missing_permission_withholds_its_layer_and_names_it(
    monkeypatch, missing, layer, requires
):
    """Withheld, never silently absent. An agent that cannot tell "you may not
    read this" from "there is nothing here" will report the second."""
    seen = _stub(monkeypatch)

    out = await mod.developer_context_for_task(
        body=mod.DeveloperContextIn(task="change billing"),
        ctx=_ctx(ALL_PERMISSIONS - {missing}), db=_DB(),
    )

    assert getattr(seen["layers"], layer) is False
    assert layer not in out["layers"]["included"]
    withheld = {w["layer"]: w for w in out["layers"]["withheld"]}
    assert withheld[layer]["reason"] == "permission"
    assert withheld[layer]["requires"] == requires


async def test_a_missing_capability_is_reported_as_a_plan_problem(monkeypatch):
    """403 and 402 mean different things and the fix is different — one is the
    credential's, the other the organization's plan. A partial answer must keep
    them apart for the same reason the gate chain does."""
    seen = _stub(
        monkeypatch,
        capabilities={**ALL_CAPABILITIES, Capability.INTELLIGENCE.value: False},
    )

    out = await mod.developer_context_for_task(
        body=mod.DeveloperContextIn(task="change billing"),
        ctx=_ctx(ALL_PERMISSIONS), db=_DB(),
    )

    assert seen["layers"].intelligence is False
    withheld = {w["layer"]: w for w in out["layers"]["withheld"]}
    assert withheld["intelligence"]["reason"] == "capability"
    assert withheld["intelligence"]["requires"] == "intelligence"


async def test_unreadable_entitlements_fail_closed(monkeypatch):
    """A layer withheld costs the bundle depth. A layer wrongly included is a
    permission that did not hold."""
    import navis.organization.service as org_service

    class _Broken:
        def __init__(self, db):
            pass

        class entitlements:  # noqa: N801
            @staticmethod
            async def snapshot(org_id):
                raise RuntimeError("entitlements unavailable")

    _stub(monkeypatch)
    monkeypatch.setattr(org_service, "OrganizationService", _Broken)

    out = await mod.developer_context_for_task(
        body=mod.DeveloperContextIn(task="change billing"),
        ctx=_ctx(ALL_PERMISSIONS), db=_DB(),
    )

    assert out["layers"]["included"] == ["memory"]
    assert {w["layer"] for w in out["layers"]["withheld"]} == {
        "state", "reasoning", "intelligence", "learning"
    }


def test_memory_is_the_floor_not_a_filtered_layer():
    """A bundle with no retrieved memory is not a degraded context — it is the
    projection `GET /developer/context` already serves. So the gate requires it
    rather than the filter withholding it."""
    source = inspect.getsource(mod.developer_context_for_task)
    assert "Permission.VIEW_MEMORY" in source
    assert "Capability.MEMORY" in source
    assert "memory" not in {name for name, _p, _c in mod._OPTIONAL_LAYERS}


# ── 4. Freshness, identity, and the wire contract ───────────────────────────


async def test_freshness_travels_with_the_bundle(monkeypatch):
    _stub(monkeypatch)

    out = await mod.developer_context_for_task(
        body=mod.DeveloperContextIn(task="change billing"),
        ctx=_ctx(ALL_PERMISSIONS), db=_DB(),
    )

    assert out["freshness"] == {
        "as_of": AS_OF.isoformat(),
        "newest_fact_at": NEWEST_FACT_AT.isoformat(),
        "state_watermark": WATERMARK,
        "state_available": True,
    }


async def test_observation_time_and_event_time_are_reported_separately(monkeypatch):
    """The two clocks must not be collapsed, and the gap must survive the wire.

    THIS IS THE BUG THIS FIELD EXISTS FOR. On the live organization `as_of` reads
    as today while the newest thing that actually happened is eighteen days old —
    because re-extraction writes memory rows now that carry old event
    timestamps. A handler that returned `as_of` for both, or dropped
    `newest_fact_at` when it was older, would tell an agent a stale picture is
    current, and every assertion in the test above would still pass.
    """
    _stub(monkeypatch)

    out = await mod.developer_context_for_task(
        body=mod.DeveloperContextIn(task="change billing"),
        ctx=_ctx(ALL_PERMISSIONS), db=_DB(),
    )

    freshness = out["freshness"]
    assert freshness["as_of"] != freshness["newest_fact_at"], (
        "observation time and event time were collapsed into one value"
    )
    assert freshness["newest_fact_at"] < freshness["as_of"], (
        "the newest fact must be able to predate the last fold — that gap is "
        "the entire signal that ingestion has stopped"
    )


async def test_an_organization_with_no_snapshot_says_so(monkeypatch):
    """`state_available: false` means "we declined to build one", NOT "this
    organization is empty". A read must not fold an org's memory to answer."""
    _stub(monkeypatch, state_available=False,
          facts=[Grounding("memory", "m1", "an email about billing")])

    out = await mod.developer_context_for_task(
        body=mod.DeveloperContextIn(task="change billing"),
        ctx=_ctx(ALL_PERMISSIONS), db=_DB(),
    )

    assert out["freshness"]["state_available"] is False
    assert out["freshness"]["as_of"] is None
    assert out["freshness"]["newest_fact_at"] is None, (
        "no snapshot means no event-time frontier either — a date here would be "
        "invented"
    )
    assert out["freshness"]["state_watermark"] is None
    assert out["count"] == 1


async def test_facts_carry_their_layer_and_lineage(monkeypatch):
    """A retrieved email, a State entity and a Reasoning conclusion are three
    different kinds of claim. Flattening them would make the bundle unjudgeable."""
    _stub(monkeypatch, facts=[
        Grounding("memory", "m1", "an email", occurred_at=AS_OF),
        Grounding("state", "e1", "commitment: ship it",
                  lineage=("memory:m9", "signal:s9")),
    ])

    out = await mod.developer_context_for_task(
        body=mod.DeveloperContextIn(task="change billing"),
        ctx=_ctx(ALL_PERMISSIONS), db=_DB(),
    )

    memory, state = out["facts"]
    assert memory["kind"] == "memory" and memory["occurred_at"] == AS_OF.isoformat()
    assert memory["lineage"] == []
    assert state["kind"] == "state" and state["lineage"] == ["memory:m9", "signal:s9"]


def test_the_request_body_names_no_identity():
    """The tenant is the credential's. Pinned globally by
    `test_no_developer_route_accepts_an_identity_parameter`, which follows the
    body `$ref` into components; named here so the reason sits with the route."""
    fields = set(mod.DeveloperContextIn.model_fields)
    assert fields == {"task", "since", "until", "token_budget"}
    for banned in ("organization_id", "org_id", "tenant_id", "user_id", "agent_id"):
        assert banned not in fields


async def test_the_task_is_echoed_and_the_org_comes_from_the_credential(monkeypatch):
    seen = _stub(monkeypatch)
    task = "I am about to modify the billing integration."

    out = await mod.developer_context_for_task(
        body=mod.DeveloperContextIn(task=task), ctx=_ctx(ALL_PERMISSIONS), db=_DB()
    )

    assert seen["org_id"] == ORG
    assert seen["entitlements_org"] == ORG
    assert seen["task"] == task
    assert out["task"] == task


@pytest.mark.parametrize("bad", ["", "x" * 2001])
def test_the_task_is_bounded(bad):
    """Empty is not a task. A pasted document is not a task either — it is what
    the retriever would embed verbatim."""
    with pytest.raises(Exception):
        mod.DeveloperContextIn(task=bad)


async def test_optional_window_and_budget_reach_the_composer(monkeypatch):
    """Every optional field exists because the Core already accepts it. None was
    invented to make the API look richer."""
    seen = _stub(monkeypatch)
    since = datetime(2026, 8, 1, tzinfo=timezone.utc)
    until = datetime(2026, 8, 25, tzinfo=timezone.utc)

    await mod.developer_context_for_task(
        body=mod.DeveloperContextIn(
            task="change billing", since=since, until=until, token_budget=4000
        ),
        ctx=_ctx(ALL_PERMISSIONS), db=_DB(),
    )

    assert seen["since"] == since
    assert seen["until"] == until
    assert seen["token_budget"] == 4000


async def test_an_omitted_budget_leaves_the_core_default(monkeypatch):
    """Not forwarded as None — the composer's default IS the application's, and
    passing None would override it with whatever this module thought."""
    seen = _stub(monkeypatch)

    await mod.developer_context_for_task(
        body=mod.DeveloperContextIn(task="change billing"),
        ctx=_ctx(ALL_PERMISSIONS), db=_DB(),
    )

    assert "token_budget" not in seen


# ── 5. Truthful source state (what is actually watching) ────────────────────


def _source(
    source: str,
    label: str,
    *,
    connected: bool = True,
    auth_state: str = "ok",
    last_attempt_at=None,
    last_success_at=None,
    last_failure_reason: str | None = None,
):
    """A `SourceHealth` with healthy defaults.

    THE REAL TYPE, not a `SimpleNamespace`. A hand-rolled stand-in would keep
    passing after a field is added to the production dataclass, which is exactly
    how the fakes elsewhere in this suite went stale — and a source-health fake
    that silently lacks `auth_state` is the one fake whose drift would hide the
    bug this whole phase is about.
    """
    from navis.api.workspace_fallback import SourceHealth

    return SourceHealth(
        source=source,
        label=label,
        connected=connected,
        auth_state=auth_state,
        last_attempt_at=last_attempt_at,
        last_success_at=last_success_at,
        last_failure_reason=last_failure_reason,
    )


async def test_the_bundle_names_the_systems_that_actually_feed_it(monkeypatch):
    """An agent reasoning about an ABSENCE needs to know what was looking.

    Navis ingests Gmail, Calendar, Drive, Slack and Notion. No code host, no
    issue tracker, no observability system. So silence about a repository is
    not evidence that nothing was said about it — it is evidence that nothing
    was ever watching one, and the bundle has to say which.
    """
    import navis.api.workspace_fallback as fallback

    _stub(monkeypatch)

    async def _health(db, org_id):
        assert org_id == ORG
        return [
            _source("gmail", "Gmail"),
            _source("google_calendar", "Google Calendar"),
            _source("drive", "Google Drive"),
        ]

    monkeypatch.setattr(fallback, "source_health", _health)

    out = await mod.developer_context_for_task(
        body=mod.DeveloperContextIn(task="modify the payments service"),
        ctx=_ctx(ALL_PERMISSIONS), db=_DB(),
    )

    assert out["sources"]["connected"] == ["Gmail", "Google Calendar", "Google Drive"]


async def test_a_connected_source_is_not_claimed_to_be_healthy(monkeypatch):
    """`connected` is INTENT. `detail` is health. Conflating them is the bug.

    A revoked Google grant stays `is_active = true` forever — the user still
    wants Gmail connected, Google just stopped honouring the token. Before
    `detail` existed, that rendered identically to a working connection, so an
    agent reading `sources.connected` treated a blind pipeline as a quiet
    company and reported "nothing to flag" with total confidence.
    """
    import navis.api.workspace_fallback as fallback

    _stub(monkeypatch)

    async def _health(db, org_id):
        return [
            _source("gmail", "Gmail", auth_state="reconsent_required",
                    last_success_at=NEWEST_FACT_AT,
                    last_failure_reason="google refresh rejected (invalid_grant)"),
            _source("slack", "Slack", connected=False),
        ]

    monkeypatch.setattr(fallback, "source_health", _health)

    out = await mod.developer_context_for_task(
        body=mod.DeveloperContextIn(task="modify the payments service"),
        ctx=_ctx(ALL_PERMISSIONS), db=_DB(),
    )

    # Backwards compatible: Gmail is still reported as connected, because the
    # user has in fact connected it. That field's meaning does not change.
    assert out["sources"]["connected"] == ["Gmail"]

    detail = {d["source"]: d for d in out["sources"]["detail"]}
    # BROKEN — connected, but nothing is arriving and nothing will until a
    # person re-authorises.
    assert detail["gmail"]["connected"] is True
    assert detail["gmail"]["auth_state"] == "reconsent_required"
    assert detail["gmail"]["last_success_at"] == NEWEST_FACT_AT.isoformat()
    assert "invalid_grant" in detail["gmail"]["last_failure_reason"]
    # DISCONNECTED — reported too, because "you turned Slack off" and "you never
    # connected Slack" are different explanations for the same silence.
    assert detail["slack"]["connected"] is False
    assert "Slack" not in out["sources"]["connected"]


async def test_unreadable_sources_degrade_to_empty_not_to_a_failed_bundle(monkeypatch):
    """An empty list is the safe direction: an agent that under-trusts an
    absence is right more often than one that assumes Navis was watching a
    system it cannot see."""
    import navis.api.workspace_fallback as fallback

    _stub(monkeypatch)

    async def _broken(db, org_id):
        raise RuntimeError("integrations unavailable")

    monkeypatch.setattr(fallback, "source_health", _broken)

    out = await mod.developer_context_for_task(
        body=mod.DeveloperContextIn(task="modify the payments service"),
        ctx=_ctx(ALL_PERMISSIONS), db=_DB(),
    )

    assert out["sources"]["connected"] == []
    assert out["sources"]["detail"] == [], (
        "an unreadable health check must not fabricate a healthy source"
    )
    assert out["count"] == 2, "the bundle itself must survive"
