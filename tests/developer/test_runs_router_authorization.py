"""The execution ledger's authorization wiring, asserted declaratively.

WHAT THESE PIN. Not "does a request succeed" — that is the integration suite's
job against a real database — but the four properties decided by how the router
is WIRED, and which a passing end-to-end test would not notice if they silently
changed:

  1. reads and writes are gated on DIFFERENT permissions;
  2. no route on this surface accepts an organization;
  3. a machine caller cannot choose which agent it writes for;
  4. every list route is bounded.

Each is a one-line change away from being wrong, and each would be invisible in a
happy-path run by a founder who holds every permission.
"""
from __future__ import annotations

import inspect
import uuid

import pytest
from fastapi import HTTPException

from navis.developer import runs_router as module
from navis.developer.gateway import DeveloperContext
from navis.developer.lifecycle import DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE
from navis.identity.authz import Principal, PrincipalKind
from navis.identity.permissions import Permission, default_permissions_for_role

ORG = uuid.UUID("cccc0000-0000-4000-8000-00000000000a")


def _ctx(kind: PrincipalKind, service_account_id: uuid.UUID | None) -> DeveloperContext:
    """A context shaped exactly as `DeveloperContext.of` builds one."""
    subject = service_account_id or uuid.uuid4()
    return DeveloperContext(
        principal=Principal(
            kind=kind,
            org_id=ORG,
            subject_id=subject,
            permissions=frozenset({Permission.RECORD_AGENT_RUNS}),
        ),
        organization_id=ORG,
        kind=kind.value,
        subject_id=subject,
        service_account_id=service_account_id,
        credential_id=None,
        permissions=frozenset({Permission.RECORD_AGENT_RUNS}),
    )


# ── 1. Read and write are different permissions ─────────────────────────────


def test_reads_and_writes_are_gated_on_different_permissions():
    """A read permission must never authorize a write. If these ever collapse
    into one, anyone who can READ the ledger can FORGE execution history."""
    source = inspect.getsource(module)

    assert "_reader = require_developer(Permission.VIEW_OBSERVATION" in source
    assert "_writer = require_developer(Permission.RECORD_AGENT_RUNS" in source


def test_the_write_permission_is_not_the_agent_administration_one():
    """`manage_service_accounts` would mean only an agent that can MINT other
    agents may report what it did."""
    source = inspect.getsource(module)
    assert "_writer = require_developer(Permission.MANAGE_SERVICE_ACCOUNTS" not in source


def test_the_write_permission_exists_in_the_shared_vocabulary():
    """A permission, not a Developer-specific role. It resolves through the same
    table every other gate reads."""
    assert Permission.RECORD_AGENT_RUNS == "record_agent_runs"
    assert Permission.RECORD_AGENT_RUNS in default_permissions_for_role("founder")
    assert Permission.RECORD_AGENT_RUNS in default_permissions_for_role("admin")
    assert Permission.RECORD_AGENT_RUNS not in default_permissions_for_role("employee")


def test_every_route_declares_one_of_the_two_gates():
    """A route that forgot its dependency would be authenticated-only."""
    for route in module.router.routes:
        source = inspect.getsource(route.endpoint)
        assert "_reader" in source or "_writer" in source, f"{route.path} declares neither gate"


# ── 2. No route accepts an organization ─────────────────────────────────────


def test_no_request_model_carries_an_organization():
    """A field like this is one somebody eventually trusts."""
    from navis.developer import run_schemas

    forbidden = {"organization_id", "org_id", "tenant_id", "organization", "tenant"}
    for name, obj in vars(run_schemas).items():
        fields = getattr(obj, "model_fields", None)
        if fields is None or not name.endswith("Request"):
            continue
        leaked = forbidden & set(fields)
        assert not leaked, f"{name} accepts {leaked}"


def test_no_route_takes_an_organization_path_or_query_parameter():
    for route in module.router.routes:
        assert "organization" not in route.path
        assert "tenant" not in route.path
        params = inspect.signature(route.endpoint).parameters
        for forbidden in ("organization_id", "org_id", "tenant_id"):
            assert forbidden not in params, f"{route.path} takes {forbidden}"


# ── 3. A machine caller cannot choose its agent ─────────────────────────────


def test_an_agent_credential_always_records_against_itself():
    """THE ISOLATION PROPERTY. `service_account_id` comes from the `api_keys` row
    that authenticated the request, so there is no request an agent can
    construct that writes to another agent's ledger."""
    own = uuid.uuid4()
    someone_else = uuid.uuid4()
    ctx = _ctx(PrincipalKind.SERVICE_ACCOUNT, own)

    # It asks to write for another agent, and is given itself.
    assert module._acting_agent(ctx, someone_else) == own
    assert module._acting_agent(ctx, None) == own


def test_a_human_must_name_the_agent_explicitly():
    """A member has no service account of their own, so there is nothing to
    default to — and guessing would attribute work to an arbitrary agent."""
    ctx = _ctx(PrincipalKind.USER, None)
    named = uuid.uuid4()

    assert module._acting_agent(ctx, named) == named

    with pytest.raises(HTTPException) as raised:
        module._acting_agent(ctx, None)
    assert raised.value.status_code == 400


def test_a_bare_api_key_with_no_agent_behind_it_must_name_one():
    """`service_account_id` is None for a bare key. Falling back to the key's own
    id would invent an agent that does not exist."""
    ctx = _ctx(PrincipalKind.API_KEY, None)

    with pytest.raises(HTTPException):
        module._acting_agent(ctx, None)


# ── 4. Lists are bounded ────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "path", ["/developer/tasks", "/developer/runs", "/developer/runs/{run_id}/steps"]
)
def test_every_list_route_bounds_its_page(path: str):
    """An unbounded list endpoint is how a shared table becomes an outage."""
    route = next(r for r in module.router.routes if r.path == path and "GET" in r.methods)
    query = inspect.signature(route.endpoint).parameters["limit"].default

    assert query.default == DEFAULT_PAGE_SIZE
    # FastAPI keeps the constraints as annotated-types metadata, not as
    # attributes on the Query object.
    ceilings = [getattr(m, "le", None) for m in query.metadata]
    assert MAX_PAGE_SIZE in ceilings, f"{path} does not cap its page size"


# ── Refusal mapping ─────────────────────────────────────────────────────────


def test_an_unaddressable_resource_is_404_not_403():
    """403 would confirm the id names something real in another tenant."""
    from navis.developer.run_service import InvalidTransition, UnknownAgent

    assert module._refuse(UnknownAgent("x")).status_code == 404
    assert module._refuse(InvalidTransition("x")).status_code == 409


def test_an_oversized_payload_is_refused_with_400():
    from navis.developer.sanitize import PayloadTooLarge

    assert module._refuse(PayloadTooLarge("too big")).status_code == 400


def test_no_response_model_can_carry_a_secret():
    """A route cannot leak a credential by returning the wrong model, because no
    model on this surface has a field that could hold one."""
    from navis.developer import run_schemas

    forbidden = {"secret", "token", "password", "api_key", "credential", "hashed_secret"}
    for name, obj in vars(run_schemas).items():
        fields = getattr(obj, "model_fields", None)
        if fields is None:
            continue
        leaked = forbidden & set(fields)
        assert not leaked, f"{name} exposes {leaked}"


# ── 5. Every handler's names actually resolve ───────────────────────────────
#
# This section exists because they did not. `start_run` shipped calling
# `_acting_agent(ctx, agent_id)` with no `agent_id` in its signature and read
# `body.task_id` from a model that has no such field, so the ONE route that
# creates a Run answered 500 to every request. Nothing caught it: the suite
# exercises `AgentLifecycleService.start_run` directly, and a route nobody calls
# through the app is a route whose body is never executed.
#
# A test for that one line would have been a test for that one line. The guard
# below is for the CLASS — a handler referencing a name that exists nowhere is a
# guaranteed 500, and it is invisible until someone makes the request.


def _phantom_globals(function) -> set[str]:
    """Names a function loads as globals that resolve nowhere.

    LOAD_GLOBAL only — attribute access (`ctx.organization_id`) also appears in
    `co_names` and is not a name this can judge.
    """
    import builtins
    import dis

    resolvable = set(function.__globals__) | set(dir(builtins))
    return {
        instruction.argval
        for instruction in dis.get_instructions(function)
        if instruction.opname == "LOAD_GLOBAL" and instruction.argval not in resolvable
    }


@pytest.mark.parametrize(
    "route", [r for r in module.router.routes], ids=lambda r: f"{sorted(r.methods)} {r.path}"
)
def test_no_handler_references_a_name_that_does_not_exist(route):
    """A phantom global is a NameError on the first real request."""
    phantom = _phantom_globals(route.endpoint)
    assert not phantom, f"{route.path} references undefined {phantom}"


@pytest.mark.parametrize("route", [r for r in module.router.routes], ids=lambda r: r.path)
def test_no_handler_reads_a_field_its_request_model_lacks(route):
    """`body.<field>` must name a real field on the declared request model."""
    import dis

    body = inspect.signature(route.endpoint).parameters.get("body")
    if body is None:
        return

    # RESOLVE THE ANNOTATION FIRST. The module is `from __future__ import
    # annotations`, so `body.annotation` is the STRING "StartRunRequest" and
    # `hasattr(..., "model_fields")` is False for every route — which is how the
    # first version of this guard passed against the very bug it was written to
    # catch. A test that cannot fail is not a test.
    model = getattr(module, body.annotation, None) if isinstance(body.annotation, str) else body.annotation
    assert model is not None, f"{route.path} declares a body this test cannot resolve"
    assert hasattr(model, "model_fields"), f"{route.path} body is not a pydantic model"

    declared = set(model.model_fields)
    read = {
        instruction.argval
        for instruction in dis.get_instructions(route.endpoint)
        if instruction.opname == "LOAD_ATTR"
        and instruction.argval not in {"value"}  # `.value` unwraps an enum
    }
    # Only names read off `body` matter; intersecting with the model's own
    # vocabulary is not enough, so compare against what a body access could be.
    source = inspect.getsource(route.endpoint)
    from_body = {name for name in read if f"body.{name}" in source}

    assert from_body <= declared, (
        f"{route.path} reads {from_body - declared} off {model.__name__}, "
        f"which declares {sorted(declared)}"
    )


def test_a_run_takes_its_task_from_the_path_and_its_agent_from_the_task():
    """THE Agent -> Task -> Run INVARIANT, at the routing layer.

    A task id in the BODY would let a caller open a run against one task while
    addressing another, which is exactly the arbitrary agent/task/run
    combination the path shape was chosen to make unrepresentable. And the
    handler must not pass an agent at all — the service reads it from the task.
    """
    from navis.developer.run_schemas import StartRunRequest

    assert "task_id" not in StartRunRequest.model_fields

    source = inspect.getsource(module.start_run)
    assert "task_id=task_id" in source, "the run must take the PATH task id"
    assert "body.task_id" not in source, "a body task id would break the invariant"
    assert "agent_id" not in source, "the agent comes from the task, never the caller"


# ── 6. Every gate names something that exists ───────────────────────────────
#
# Also written after the fact. `_WRITE_LIMIT` referenced
# `rate_limit_write_per_window`, a Settings field that does not exist, so every
# ledger WRITE raised AttributeError inside the dependency and answered 500
# before the handler ran. Invisible for the same reason as the `start_run` bug:
# `rate_limit` reads the attribute at REQUEST time, so nothing fails at import,
# and no test made the request.
#
# Asserted across the WHOLE app rather than this router, because the next
# occurrence will be in whichever router is written next.


def test_every_rate_limit_names_a_setting_that_exists():
    """A missing Settings field is a 500 on first use, not an import error."""
    import re
    from pathlib import Path

    from navis.platform.config.settings import settings

    root = Path(__file__).resolve().parents[2] / "navis"
    missing: list[str] = []
    for path in root.rglob("*.py"):
        for bucket, field in re.findall(
            r'rate_limit\(\s*"([^"]+)"\s*,\s*"([^"]+)"\s*\)', path.read_text()
        ):
            if not hasattr(settings, field):
                missing.append(f"{path.relative_to(root)}: {bucket} -> {field}")

    assert not missing, "rate_limit references a Settings field that does not exist: " + "; ".join(
        missing
    )


def test_the_ledger_write_budget_is_not_the_agent_registration_budget():
    """Different access patterns, so different ceilings.

    An organization registers a handful of agents and then reads with them for
    months, which is why `agent_write` is 20. A single run legitimately appends
    many steps in quick succession, so reusing that ceiling would refuse
    ordinary recording and make the ledger lossy exactly when an agent is
    busiest.
    """
    from navis.platform.config.settings import settings

    source = inspect.getsource(module)
    assert "rate_limit_agent_write_per_window" not in source
    assert settings.rate_limit_run_write_per_window > settings.rate_limit_agent_write_per_window
