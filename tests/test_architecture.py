"""Architecture enforcement: repo-wide structural invariants.

Rule 2/9 of the constitution: platform is shared infrastructure only. It must
never import a business domain — otherwise infrastructure becomes a hidden
dependency hub and the dependency graph inverts. This test fails the build if
any `navis/platform/**` module imports a business domain.

The second guard here (TASK 6.4.4 / F-5) is the route table: every mounted route
must resolve a Principal unless it is on a short, named list of endpoints that
authenticate by another mechanism.
"""
from __future__ import annotations

import ast
import pathlib

PLATFORM_DIR = pathlib.Path(__file__).resolve().parent.parent / "navis" / "platform"

BUSINESS_DOMAINS = {
    "identity",
    "identity_resolution",
    "signals",
    "memory",
    "state",
    "reasoning",
    "decisions",
    "execution",
    "observation",
    "learning",
    "intelligence",
    "ml",
    "orchestration",
    "api",
}


def _imported_modules(path: pathlib.Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    mods: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                mods.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods.add(node.module)
    return mods


def test_platform_does_not_import_business_domains():
    violations: list[str] = []
    for py in PLATFORM_DIR.rglob("*.py"):
        for mod in _imported_modules(py):
            parts = mod.split(".")
            if len(parts) >= 2 and parts[0] == "navis" and parts[1] in BUSINESS_DOMAINS:
                rel = py.relative_to(PLATFORM_DIR.parent.parent)
                violations.append(f"{rel} imports {mod}")
    assert not violations, "platform leaks into business domains:\n" + "\n".join(violations)


def test_platform_modules_all_import_cleanly():
    """Importing every platform module must not raise (no broken wiring)."""
    import importlib

    modules = [
        "navis.platform.config",
        "navis.platform.database",
        "navis.platform.cache",
        "navis.platform.queue",
        "navis.platform.events",
        "navis.platform.security",
        "navis.platform.monitoring",
        "navis.platform.monitoring.health",
        "navis.platform.monitoring.middleware",
        "navis.platform.monitoring.sentry",
        "navis.platform.storage",
    ]
    for m in modules:
        importlib.import_module(m)


# ---- TASK 6.4.4 / F-5: the route table is authenticated by default ----
#
# The endpoints that legitimately serve an anonymous caller. Each one
# authenticates by a mechanism that is not a Principal, and each is named here
# so ADDING to this set is a deliberate, reviewable act rather than a side
# effect of forgetting a dependency.
#
#   /auth/*        mint or exchange credentials. `/auth/logout` and
#                  `/auth/refresh` validate the token they are handed.
#   OAuth callback verifies a signed `state` parameter.
#   /webhooks/*    verify a provider signature (Google Pub/Sub JWT, Slack
#                  signing secret) before trusting the body.
#
# NOTHING THAT SERVES TENANT BUSINESS DATA MAY BE ADDED. That is the failure
# this guard exists to prevent: `compat_router` served `/decide/stack`,
# `/skills`, `/signals`, `/execution/slack/blockers` and
# `/workspace/meetings/{id}/brief` as anonymous 200s with empty payloads, so a
# dead session and a quiet Tuesday looked identical to every client.
_UNAUTHENTICATED_BY_DESIGN = {
    ("POST", "/auth/signup"),
    ("POST", "/auth/login"),
    ("POST", "/auth/google"),
    ("POST", "/auth/refresh"),
    ("POST", "/auth/logout"),
    ("GET", "/integrations/google/callback"),
    ("POST", "/webhooks/google/pubsub"),
    ("POST", "/webhooks/slack/events"),
    ("POST", "/webhooks/observation/{provider}"),
}


def _routes_resolving_no_principal() -> set[tuple[str, str]]:
    from fastapi.routing import APIRoute

    from navis.api.router import api_router
    from navis.identity.authz import get_principal
    from navis.identity.dependencies import get_current_context

    auth_deps = {get_principal, get_current_context}

    def resolves(dep, seen: set[int] | None = None) -> bool:
        seen = seen if seen is not None else set()
        if id(dep) in seen:
            return False
        seen.add(id(dep))
        if dep.call in auth_deps:
            return True
        return any(resolves(sub, seen) for sub in dep.dependencies)

    return {
        (method, r.path)
        for r in api_router.routes
        if isinstance(r, APIRoute)
        for method in r.methods
        if not resolves(r.dependant)
    }


def test_every_mounted_route_resolves_a_principal():
    """No route may serve a request without establishing WHO is asking.

    Checked structurally (walk each route's dependency tree for
    `get_principal` / `get_current_context`) rather than by calling endpoints,
    so a route added tomorrow is covered without anyone writing a test for it.
    """
    unauthenticated = _routes_resolving_no_principal()
    unexpected = unauthenticated - _UNAUTHENTICATED_BY_DESIGN
    assert not unexpected, (
        "these routes serve anonymous callers and are not on the allowlist:\n"
        + "\n".join(f"  {m} {p}" for m, p in sorted(unexpected))
        + "\n\nA protected resource must answer 401 when unauthenticated. If the "
        "route genuinely authenticates by another mechanism (signature, signed "
        "state), add it to _UNAUTHENTICATED_BY_DESIGN with the reason."
    )


def test_unauthenticated_allowlist_has_no_dead_entries():
    """The allowlist shrinks when a route is removed or gains a principal.

    Without this, a deleted exception stays on the list forever and quietly
    pre-approves the next route that happens to reuse the path.
    """
    stale = _UNAUTHENTICATED_BY_DESIGN - _routes_resolving_no_principal()
    assert not stale, f"allowlist entries no longer apply: {sorted(stale)}"


# ---- The Developer boundary: structural guarantees, checked on every run ----
#
# `tests/integration/test_developer_boundary_postgres.py` proves the BEHAVIOUR —
# tenant isolation, revocation, secret handling — but it skips unless
# NAVIS_TEST_DATABASE_URL is set, so it cannot be the only thing standing between
# a new Developer route and a missing gate. These four guards need no database
# and run in the default suite, and like the route-table guard above they are
# STRUCTURAL: a route added tomorrow is covered without anyone writing a test.

def _developer_routes():
    from fastapi.routing import APIRoute

    from navis.api.router import api_router

    return [
        r for r in api_router.routes
        if isinstance(r, APIRoute) and r.path.startswith("/developer")
    ]


def _closure_values(dep) -> set:
    """Every value captured by the dependency closures under `dep`.

    `require_permission(perm)` and `require_capability(cap)` both return a closure
    over their argument, so walking the dependency tree and reading the cells is
    how a test discovers WHICH permission and WHICH capability a route enforces —
    without calling anything or needing a request.

    Recurses through captured CALLABLES as well as sub-dependencies, because
    `require_developer` composes the two gates by calling them rather than by
    declaring them: its closure holds the `require_permission` and
    `require_capability` closures, and the strings live one level further in. A
    walker that read only the outermost cells would find nothing and report every
    route as ungated — which is a test that fails safe, but never passes.

    ALSO FLATTENS CAPTURED TUPLES AND LISTS. `require_developer_any` takes the
    several permissions that may reach a route (P1-7: organization-wide agent
    administration OR the self-serve own-agents grant) and closes over the
    SEQUENCE, so the permission strings sit one container deep. Without this the
    agent routes read as `permissions=[]` and the gate test below reported them
    ungated, which was a false alarm about real gates rather than a real finding.
    The invariant it protects is unchanged: at least one genuine `Permission`
    must still be discoverable per route.
    """
    found: set = set()
    seen_fns: set[int] = set()

    def read_fn(fn) -> None:
        if fn is None or id(fn) in seen_fns:
            return
        seen_fns.add(id(fn))
        for cell in getattr(fn, "__closure__", None) or ():
            try:
                value = cell.cell_contents
            except ValueError:  # an empty cell, mid-definition
                continue
            if callable(value):
                read_fn(value)
                continue
            # A sequence of gate arguments — record the MEMBERS, not the
            # container, so a permission inside one is as visible as a
            # permission captured directly. Deliberately not recursive beyond
            # one level and deliberately not applied to str/bytes (which are
            # themselves iterable and would explode into characters).
            if isinstance(value, (tuple, list, frozenset, set)):
                for member in value:
                    try:
                        found.add(member)
                    except TypeError:
                        continue
                continue
            try:
                found.add(value)
            except TypeError:  # unhashable capture — not a permission or capability
                continue

    def walk(d, seen: set[int]) -> None:
        if id(d) in seen:
            return
        seen.add(id(d))
        read_fn(getattr(d, "call", None))
        for sub in d.dependencies:
            walk(sub, seen)

    walk(dep, set())
    return found


def test_the_developer_api_is_mounted_on_the_one_application():
    """One app, one schema. The Developer routes must be reachable under the
    existing `/api/v1` prefix and appear in the existing OpenAPI document — a
    second FastAPI application is the failure this asserts against."""
    from navis.main import create_app

    app = create_app()
    schema = app.openapi()
    paths = {p for p in schema["paths"] if "/developer" in p}
    assert paths, "the Developer API is absent from the generated OpenAPI schema"
    assert all(p.startswith("/api/v1/developer/") for p in paths), sorted(paths)
    # Every route is tagged, and every response body has a NAMED component — a
    # generated MCP/SDK/CLI client emits `object` for an untyped handler.
    #
    # The SUCCESS response, whatever its status. Pinning "200" was correct while
    # the boundary was read-only; `POST /developer/agents` answers 201, and a
    # check that skipped it would have left the one route that returns a secret
    # unverified. Any 2xx counts, and exactly one is expected per operation.
    for path in paths:
        for method, op in schema["paths"][path].items():
            assert op.get("tags") == ["developer"], f"{method} {path} is untagged"
            success = [code for code in op["responses"] if code.startswith("2")]
            assert len(success) == 1, f"{method} {path} declares {success} success codes"
            ok = op["responses"][success[0]]["content"]["application/json"]["schema"]
            assert "$ref" in ok, f"{method} {path} has no named response schema"


def test_every_developer_route_gates_on_a_permission_and_a_capability():
    """The Developer boundary's whole authorization contract, structurally.

    Every data route must enforce BOTH an existing `Permission` (may this
    CREDENTIAL?) and an existing `Capability` (is this TENANT entitled?). The
    web-edge routes these delegate to gate on the capability alone, so a route
    that inherits its gates by accident instead of declaring them would be
    reachable by a credential holding no permissions at all.

    `/developer/me` is the one exception and is named here: it reports what a
    credential IS, so gating it on a permission would stop a misconfigured
    credential from discovering that it is misconfigured.
    """
    from navis.identity.permissions import VALID_PERMISSIONS
    from navis.organization.contract import Capability

    unauthorized: list[str] = []
    for route in _developer_routes():
        if route.path == "/developer/me":
            continue
        captured = _closure_values(route.dependant)
        perms = {v for v in captured if isinstance(v, str) and v in VALID_PERMISSIONS}
        caps = {v for v in captured if isinstance(v, Capability)}
        if not perms or not caps:
            unauthorized.append(
                f"{route.path}: permissions={sorted(perms)} capabilities={sorted(c.value for c in caps)}"
            )
    assert not unauthorized, (
        "a Developer route does not gate on BOTH a permission and a capability:\n"
        + "\n".join(unauthorized)
    )


#: A name a caller must NEVER be able to send to the Developer boundary, in any
#: position. Every one of these selects a TENANT or a SUBJECT, and all of them are
#: resolved from the credential — so accepting one, even to ignore it as
#: `/context` does for frontend compatibility, is a parameter a client will
#: eventually believe in.
_BANNED_IDENTITY_NAMES = frozenset({
    "organization_id", "org_id", "tenant_id",
    "principal_id", "user_id", "subject_id", "owner_user_id",
    "service_account_id", "created_by", "actor_id",
})

#: Names that address a RESOURCE rather than select an identity, and are therefore
#: allowed — but ONLY as path parameters. `/developer/agents/{agent_id}` names one
#: of the caller's own agents and is looked up under `Principal.org_id`, so a
#: foreign id answers 404 rather than reaching a row. The same word as a QUERY
#: parameter or a BODY field would be a different thing entirely: something the
#: caller supplies alongside a resolved resource, which is how "filter" quietly
#: becomes "scope".
_RESOURCE_PATH_NAMES = frozenset({"agent_id", "credential_id"})


def test_no_developer_route_accepts_an_identity_parameter():
    """Organization, agent and principal are derived server-side, always.

    THE BOUNDARY IS NO LONGER READ-ONLY, and this test is what keeps that from
    quietly becoming a hole. `/developer/agents` accepts request bodies, so
    "declares no request body" is no longer the invariant. The invariant is
    narrower and stronger:

      * no identity name (above) anywhere — path, query, header or body;
      * `agent_id`/`credential_id` only in the PATH, never as a query parameter
        and never in a body;
      * every request body schema is checked field by field, following its
        `$ref` into `components`, because a banned name hidden one level down is
        exactly the thing a parameter-only check would miss.
    """
    from navis.main import create_app

    schema = create_app().openapi()
    components = schema.get("components", {}).get("schemas", {})

    def body_fields(operation: dict) -> set[str]:
        """Top-level property names of an operation's JSON request body."""
        content = (operation.get("requestBody") or {}).get("content", {})
        media = content.get("application/json", {})
        ref = media.get("schema", {}).get("$ref")
        if not ref:
            return set()
        model = components.get(ref.rsplit("/", 1)[-1], {})
        return set(model.get("properties", {}))

    offenders: list[str] = []
    for path, ops in schema["paths"].items():
        if "/developer" not in path:
            continue
        for method, op in ops.items():
            where = f"{method.upper()} {path}"
            for param in op.get("parameters", []):
                name, location = param["name"], param.get("in")
                if name in _BANNED_IDENTITY_NAMES:
                    offenders.append(f"{where} accepts {name} as a {location} parameter")
                elif name in _RESOURCE_PATH_NAMES and location != "path":
                    offenders.append(f"{where} accepts {name} as a {location} parameter")
            for field in sorted(body_fields(op) & (_BANNED_IDENTITY_NAMES | _RESOURCE_PATH_NAMES)):
                offenders.append(f"{where} accepts {field} in its request body")

    assert not offenders, (
        "a Developer route lets the caller name an identity:\n" + "\n".join(offenders)
    )


def test_the_developer_agent_lifecycle_never_writes_from_a_machine_principal():
    """Registering machines is a HUMAN operation on this boundary.

    An agent that can create agents can mint identities its OWN revocation does
    not reach, so containing a leaked agent credential would stop being a single
    `DELETE`. Every mutating route on the AGENT LIFECYCLE therefore depends on
    `_operator`, which asserts `PrincipalKind.USER` on top of the permission and
    capability gates.

    SCOPED TO THE LIFECYCLE ROUTER, deliberately. This test used to sweep every
    mutating `/developer/*` route, which was the same set at the time — the
    agents router was the boundary's only writer. It is not any more: the agent
    EXECUTION LEDGER (`/developer/tasks`, `/developer/runs`) is written BY agents,
    and must be.

    The distinction is not a relaxation, it is the actual invariant. What makes a
    machine write dangerous here is MINTING AN IDENTITY, not writing a row. A run
    record creates no principal, carries no scopes, and grants nothing; refusing
    it to machines would mean the only party that knows what an agent did is the
    only party forbidden from saying so, and the ledger would simply be empty.
    `test_the_execution_ledger_cannot_mint_an_identity` below is the other half
    of this pair, and it is what keeps the narrowing honest.
    """
    from fastapi.routing import APIRoute

    from navis.developer.agents_router import _operator
    from navis.developer.agents_router import router as agents_router

    lifecycle_paths = {r.path for r in agents_router.routes if isinstance(r, APIRoute)}
    mutating = [
        r for r in _developer_routes()
        if isinstance(r, APIRoute)
        and r.path in lifecycle_paths
        and r.methods & {"POST", "PATCH", "PUT", "DELETE"}
    ]
    assert mutating, "no mutating agent-lifecycle route found — has the agents router moved?"

    ungated = [
        f"{sorted(r.methods)} {r.path}"
        for r in mutating
        if not any(d.call is _operator for d in r.dependant.dependencies)
    ]
    assert not ungated, (
        "a mutating agent-lifecycle route does not require a human operator:\n"
        + "\n".join(ungated)
    )


def test_the_execution_ledger_cannot_mint_an_identity():
    """The other half of the pair above.

    The ledger is writable by machines, so the thing that must be impossible is
    that a ledger write ever creates a principal or widens one. Two properties
    hold that line, and both are asserted here rather than trusted:

      * the ledger's writer gate is `record_agent_runs` — NOT
        `manage_service_accounts`, which is the permission that mints agents and
        credentials. An agent holding the ledger permission therefore cannot
        reach a single identity route.
      * no ledger route touches the identity tables. The service imports
        `ServiceAccount` to VALIDATE that an agent belongs to the tenant, and
        writes only its own four tables.
    """
    import inspect

    from navis.developer import run_service, runs_router
    from navis.identity.permissions import Permission

    source = inspect.getsource(runs_router)
    assert f'_writer = require_developer(Permission.RECORD_AGENT_RUNS' in source
    assert "MANAGE_SERVICE_ACCOUNTS" not in source, (
        "the execution ledger must not name the identity-minting permission"
    )

    # The service may READ ServiceAccount to check tenancy; it must never add or
    # delete one. `session.add(` of an identity model would be minting.
    service_source = inspect.getsource(run_service)
    for minting in ("ServiceAccount(", "ApiKey(", "self._db.delete("):
        assert minting not in service_source, (
            f"the execution ledger service must not perform {minting!r}"
        )


def test_the_developer_context_derives_identity_only_from_the_principal():
    """`DeveloperContext` is a projection of `Principal`, and is immutable.

    Frozen so a handler cannot re-point a request at another tenant halfway
    through, and `slots=True` so an attribute the class does not declare cannot be
    attached at runtime. The agent is reported only for a SERVICE_ACCOUNT
    principal: a bare API key has no machine identity behind it, and reporting its
    key id as an "agent" would invent one the database never created.
    """
    import dataclasses
    import uuid as _uuid

    import pytest

    from navis.developer.gateway import DeveloperContext
    from navis.identity.authz import Principal, PrincipalKind

    org, sa, key = _uuid.uuid4(), _uuid.uuid4(), _uuid.uuid4()

    agent = DeveloperContext.of(Principal(
        kind=PrincipalKind.SERVICE_ACCOUNT, org_id=org, subject_id=sa,
        permissions=frozenset({"view_state"}), scopes=["view_state"], credential_id=key,
    ))
    assert agent.organization_id == org
    assert agent.service_account_id == sa
    assert agent.credential_id == key

    bare = DeveloperContext.of(Principal(
        kind=PrincipalKind.API_KEY, org_id=org, subject_id=key,
        permissions=frozenset(), credential_id=key,
    ))
    assert bare.service_account_id is None, "a bare API key is not an agent"
    assert bare.credential_id == key

    human = DeveloperContext.of(Principal(
        kind=PrincipalKind.USER, org_id=org, subject_id=_uuid.uuid4(),
        permissions=frozenset(), role="founder",
    ))
    assert human.service_account_id is None and human.credential_id is None

    assert dataclasses.is_dataclass(DeveloperContext)
    with pytest.raises(dataclasses.FrozenInstanceError):
        agent.organization_id = _uuid.uuid4()  # type: ignore[misc]
    # An UNDECLARED attribute is refused too. `slots=True` gives no `__dict__` to
    # put it in and the frozen `__setattr__` runs first, so the rejection surfaces
    # as TypeError on CPython rather than the AttributeError bare slots produce.
    # Which exception it is does not matter; that it cannot be set does.
    with pytest.raises((AttributeError, TypeError)):
        agent.impersonated_org = _uuid.uuid4()  # type: ignore[attr-defined]


def test_the_permission_vocabulary_covers_every_readable_pipeline_stage():
    """The VIEW_* vocabulary is pipeline-shaped, and the shape must stay complete.

    Observation and Learning each have a `Capability` and an HTTP reader, and for
    a while had no `Permission` — so the Developer boundary had to borrow
    `VIEW_DECISIONS` for both. That borrowing is what this guard prevents from
    recurring: a stage that gains a read surface needs its own word, or an edge
    will quietly widen someone else's.

    Execution is the known, deliberate remaining hole: it has `EXECUTE_ACTIONS`
    (a write) and no read permission, because nothing reads it over HTTP yet. It
    is named here so adding an execution reader trips this test.
    """
    from navis.identity.permissions import ALL_PERMISSIONS, Permission
    from navis.organization.contract import Capability

    readable_stages = {
        "memory": Permission.VIEW_MEMORY,
        "state": Permission.VIEW_STATE,
        "reasoning": Permission.VIEW_REASONING,
        "decision": Permission.VIEW_DECISIONS,
        "observation": Permission.VIEW_OBSERVATION,
        "learning": Permission.VIEW_LEARNING,
        "intelligence": Permission.VIEW_INTELLIGENCE,
    }
    for capability, permission in readable_stages.items():
        assert capability in {c.value for c in Capability}, capability
        assert permission in ALL_PERMISSIONS, (
            f"pipeline stage {capability!r} has a Capability but no Permission — "
            "a route for it would have to borrow another stage's word"
        )

    # The hole, stated so it cannot be forgotten rather than left implicit.
    assert "view_execution" not in ALL_PERMISSIONS, (
        "an execution read permission was added — give it a route and add it to "
        "`readable_stages` above, or remove it"
    )


def test_the_application_and_developer_authorization_asymmetry_is_deliberate():
    """Application API authorization != Developer API authorization.

    The four web-edge read routes gate on a CAPABILITY only; every Developer data
    route gates on a permission AND a capability. That asymmetry is a decision
    (a human session has a UI and a member behind it; a machine credential has an
    explicit scope list and neither), and this pins BOTH halves.

    It fails in either direction — if a Developer route drops to capability-only,
    or if an application route silently gains a permission gate. The second is
    not a bug, but it IS a role migration that revokes access from real
    `employee` users, so it must be a reviewed edit here rather than a side
    effect somewhere else.
    """
    from fastapi.routing import APIRoute

    from navis.api.router import api_router
    from navis.identity.permissions import VALID_PERMISSIONS

    capability_only = {
        "/context", "/knowledge/home", "/outcomes", "/learning/summary",
    }
    for route in api_router.routes:
        if not isinstance(route, APIRoute) or route.path not in capability_only:
            continue
        perms = {
            v for v in _closure_values(route.dependant)
            if isinstance(v, str) and v in VALID_PERMISSIONS
        }
        assert not perms, (
            f"{route.path} gained a permission gate ({sorted(perms)}). That is a "
            "role migration — `employee` holds neither view_state nor view_memory "
            "— so update this guard, the note in `navis/developer/router.py`, and "
            "the characterisation test in the Developer integration module."
        )


def test_removed_compat_stubs_are_not_mounted():
    """The five deleted stubs must not come back under any router.

    Each returned an unauthenticated 200 with a plausible empty business
    payload — `{"decisions": []}`, `{"blockers": []}`, a meeting brief for an
    event nobody checked the caller could see. The frontend's `fetchJsonOr`
    already treats 404 as "not shipped" for four of them, and the meeting-brief
    drawer renders an explicit error state for the fifth, so absence is the
    honest answer for all five.
    """
    from fastapi.routing import APIRoute

    from navis.api.router import api_router

    gone = {
        "/decide/stack",
        "/skills",
        "/signals",
        "/execution/slack/blockers",
        "/workspace/meetings/{event_id}/brief",
    }
    mounted = {r.path for r in api_router.routes if isinstance(r, APIRoute)}
    assert not (gone & mounted), f"deleted compat stubs are mounted again: {sorted(gone & mounted)}"
    # `/signals` was also a sibling of the REAL signals prefix; the domain routes
    # under it must be untouched by the removal.
    assert "/signals/gmail/recent" in mounted


# ── ONE INTELLIGENCE PIPELINE, TWO CONSUMERS ────────────────────────────────
#
# The invariant this whole platform rests on: the Navis application and the
# Developer boundary must reach organizational truth through the SAME code. A
# second composer would not announce itself — both surfaces would keep working,
# both suites would keep passing, and the two would drift a field at a time
# until a founder and their agent disagreed about their own company.
#
# Nothing but reading the call graph catches that, so these read it.


def _calls_in(fn) -> set[str]:
    """Every function this CODE calls. Docstrings and comments excluded.

    Deliberately not a substring search over the source: these modules explain
    in prose which alternatives they do NOT use, and a text match cannot tell
    an explanation from a call.
    """
    import ast
    import inspect
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                names.add(func.id)
            elif isinstance(func, ast.Attribute):
                names.add(func.attr)
    return names


def test_the_application_and_the_developer_api_share_one_context_composer():
    """`/chat` and `POST /developer/context` must call the same function.

    Not "an equivalent one". The same one, by name, from the same module — so
    retrieval, ranking, the State fold, the reasoning pass, the intelligence
    read and the token budget are one implementation with two callers.
    """
    import importlib

    from navis.api.chat_router import _grounded_answer

    developer = importlib.import_module("navis.developer.context_router")

    assert "build_grounded_context" in _calls_in(_grounded_answer), (
        "the chat path no longer calls the shared composer"
    )
    assert "build_grounded_context" in _calls_in(
        developer.developer_context_for_task
    ), "the Developer context route no longer calls the shared composer"


def test_neither_surface_reimplements_retrieval_ranking_or_scoring():
    """The composer owns them, and nothing above it may acquire its own.

    A ranking or retrieval call appearing in a ROUTER is the first move of a
    second intelligence engine: it works, it is small, and it is a different
    answer to "what matters to this organization".
    """
    import importlib
    import inspect

    from navis.api import chat_router

    developer = importlib.import_module("navis.developer.context_router")

    # The names that mean "this module decided relevance for itself".
    #
    # Each is the ENTRY POINT to one of the composer's jobs, so naming any of
    # them here is the whole bypass rather than a step toward one. Bare
    # `search` and `embed` are deliberately NOT in this set: `re.search` and a
    # compiled pattern's `.search()` are how `chat_router` classifies intent,
    # which is its own job and not retrieval. A vector search on this path can
    # only be reached through `MemoryRetriever`, which is listed.
    forbidden = {
        "MemoryRetriever",   # retrieval
        "rerank",            # ranking
        "ContextBuilder",    # assembly + the token budget
        "run_reasoning_v2",  # the reasoning pass
        "scope_for",         # the State projection
    }
    for module in (chat_router, developer):
        found = set()
        for _name, obj in vars(module).items():
            if not (inspect.isfunction(obj) or inspect.iscoroutinefunction(obj)):
                continue
            if inspect.getmodule(obj) is not module:
                continue  # imported, not defined here
            found |= _calls_in(obj) & forbidden
        assert not found, (
            f"{module.__name__} performs its own {sorted(found)} — the composer owns these"
        )


def test_only_the_developer_boundary_asks_for_the_read_only_state_read():
    """The one deliberate asymmetry, asserted so it stays deliberate.

    `/chat` keeps `get_state`, which repairs an absent or corrupt snapshot by
    rebuilding it — a human session is entitled to spend that. The Developer
    boundary passes `read_only=True`, so a machine asking what an organization
    knows cannot trigger a full fold of its memory by asking.

    If this ever inverts, either the Application got slower or the Developer
    boundary started writing.
    """
    import ast
    import importlib
    import inspect
    import textwrap

    from navis.api.chat_router import _grounded_answer

    developer = importlib.import_module("navis.developer.context_router")

    def read_only_kwarg(fn) -> object:
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "build_grounded_context"
            ):
                for kw in node.keywords:
                    if kw.arg == "read_only":
                        return ast.literal_eval(kw.value)
                return None
        raise AssertionError("no composer call found")

    assert read_only_kwarg(_grounded_answer) is None, (
        "the chat path must keep the repairing State read it has always used"
    )
    assert read_only_kwarg(developer.developer_context_for_task) is True, (
        "the Developer context route must never fold, rebuild or dispose State"
    )
