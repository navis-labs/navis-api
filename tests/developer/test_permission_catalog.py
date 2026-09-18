"""The permission catalog — the thing that stops a management surface drifting.

WHAT THESE PIN, AND WHY EACH ONE EARNS ITS PLACE.

The Developer Console shipped a hand-written list of four agent permissions in
TypeScript while the Developer boundary had grown to gate on nine. Nothing
failed. No test broke. `record_agent_runs` simply could not be selected in the
screen whose entire purpose is to grant it, and the only way to find out was to
try to use a credential and read the 403.

That failure had one cause: the list and the routes lived in different
languages in different repositories, with nothing asserting they agreed. So the
tests below are mostly about AGREEMENT rather than behaviour —

  1. the catalog describes the whole vocabulary, so a new permission cannot
     appear in the console as a bare string with no meaning;
  2. the agent-assignable set matches what Developer routes actually gate on,
     which is the specific drift that produced this file;
  3. the catalog is inert — it grants nothing, gates nothing, and removing an
     entry cannot widen or narrow what anybody may do.

(3) matters most. A catalog that started deciding things would be a second
authorization axis next to `Permission`, which is the failure the Developer
boundary exists to prevent.
"""
from __future__ import annotations

import inspect
import re

# NOT `from navis.developer import router` — the package re-exports the
# `APIRouter` instance under that name, which shadows the submodule.
from navis.developer.router import developer_permissions
from navis.identity.permissions import (
    AGENT_ASSIGNABLE_PERMISSIONS,
    PERMISSION_CATALOG,
    VALID_PERMISSIONS,
    Permission,
    PermissionKind,
    permission_info,
)

# ── 1. The catalog covers the vocabulary ────────────────────────────────────


def test_every_permission_is_described():
    """A permission with no catalog entry renders in the console as a bare
    string an operator cannot act on. Adding one to `Permission` must mean
    deciding what it means."""
    undescribed = VALID_PERMISSIONS - set(PERMISSION_CATALOG)
    assert undescribed == set(), f"permissions with no catalog entry: {sorted(undescribed)}"


def test_the_catalog_invents_nothing():
    """The reverse direction. A catalog entry for a permission the vocabulary
    does not define would be offered by the console and then rejected by
    `validate_permission` at 400 — a grant that cannot succeed."""
    invented = set(PERMISSION_CATALOG) - VALID_PERMISSIONS
    assert invented == set(), f"catalog describes unknown permissions: {sorted(invented)}"


def test_every_entry_states_a_kind_the_console_can_render():
    kinds = {PermissionKind.READ, PermissionKind.WRITE, PermissionKind.ADMIN}
    for name, info in PERMISSION_CATALOG.items():
        assert info.kind in kinds, f"{name} has unrenderable kind {info.kind!r}"
        assert info.summary.strip() != "", f"{name} has no summary"


def test_record_agent_runs_is_described_as_a_write():
    """The permission this whole exercise existed to make grantable. It opens
    the only write a machine principal may perform, so a console that rendered
    it beside the read permissions would be describing it wrongly."""
    info = PERMISSION_CATALOG[Permission.RECORD_AGENT_RUNS]
    assert info.kind == PermissionKind.WRITE
    assert info.sensitive is True


def test_permission_info_never_raises_for_an_undescribed_permission():
    """A management screen that 500s because somebody added a permission
    without a summary is worse than one that renders the bare string."""
    info = permission_info("a_permission_that_does_not_exist")
    assert info.permission == "a_permission_that_does_not_exist"
    assert info.summary == ""


# ── 2. The anti-drift guard ─────────────────────────────────────────────────


def _permissions_gated_on_by_developer_routes() -> set[str]:
    """Every `Permission.X` named anywhere under `navis/developer/`.

    Read from the SOURCE rather than from a list someone maintains, because a
    maintained list is precisely what drifted.
    """
    import pathlib

    import navis.developer as developer_pkg

    root = pathlib.Path(developer_pkg.__file__).parent
    found: set[str] = set()
    for path in root.glob("*.py"):
        for attr in re.findall(r"Permission\.([A-Z_]+)", path.read_text()):
            value = getattr(Permission, attr, None)
            if isinstance(value, str):
                found.add(value)
    return found


def test_agent_assignable_matches_developer_routes():
    """THE TEST THIS FILE EXISTS FOR.

    If a Developer route starts gating on a permission that is not
    agent-assignable, the console cannot grant it and the route is unreachable
    for every agent — silently, until somebody debugs a 403. That is exactly
    how `record_agent_runs` became ungrantable.
    """
    gated_on = _permissions_gated_on_by_developer_routes()
    unreachable = gated_on - AGENT_ASSIGNABLE_PERMISSIONS
    assert unreachable == set(), (
        "Developer routes gate on permissions the console cannot grant to an "
        f"agent: {sorted(unreachable)}. Add them to AGENT_ASSIGNABLE_PERMISSIONS "
        "or stop gating on them."
    )


def test_agent_assignable_permissions_are_all_real():
    assert AGENT_ASSIGNABLE_PERMISSIONS <= VALID_PERMISSIONS


def test_agent_assignable_includes_record_agent_runs():
    """Regression pin for the original bug, stated in its own terms."""
    assert Permission.RECORD_AGENT_RUNS in AGENT_ASSIGNABLE_PERMISSIONS


# ── 3. The catalog decides nothing ──────────────────────────────────────────


def test_the_catalog_is_not_an_authorization_input():
    """`ensure_grantable_scopes` is the only rule that decides what may be
    granted. If it ever consulted the catalog, a documentation edit would
    become a privilege change."""
    from navis.identity import authz

    source = inspect.getsource(authz.ensure_grantable_scopes)
    assert "PERMISSION_CATALOG" not in source
    assert "AGENT_ASSIGNABLE" not in source


def test_the_catalog_route_reports_grantability_from_the_caller_not_the_catalog():
    """`grantable` must mirror the caller's own permissions — the same rule
    the write path enforces. Deriving it from the catalog instead would let
    the console offer a grant the Core then refuses."""
    source = inspect.getsource(developer_permissions)
    assert "held = set(ctx.permissions)" in source
    assert "grantable=name in held" in source


def test_the_catalog_route_is_gated_and_names_no_organization():
    """Consistent with the agent surface it serves, and — like every Developer
    route — it takes no tenant parameter.

    SINCE P1-7 "consistent with the agent surface" IS THE ASSERTION, rather than
    a particular permission name. The agent routes accept either
    `manage_service_accounts` or `manage_own_agents`, and this route has to
    accept exactly the same callers: a self-serve developer allowed to register
    an agent and handed an empty catalog would be given a form with nothing to
    tick. Pinning the SHARED constant is what makes the two unable to drift,
    which naming one permission here could not do.
    """
    from navis.developer.gateway import AGENT_MANAGEMENT_PERMISSIONS

    source = inspect.getsource(developer_permissions)
    assert "require_developer_any(AGENT_MANAGEMENT_PERMISSIONS" in source
    # And the constant is the pair the agent routes gate on.
    assert set(AGENT_MANAGEMENT_PERMISSIONS) == {
        Permission.MANAGE_SERVICE_ACCOUNTS,
        Permission.MANAGE_OWN_AGENTS,
    }

    signature = inspect.signature(developer_permissions)
    assert "organization_id" not in signature.parameters
    assert "organization" not in signature.parameters


def test_the_catalog_and_the_agent_routes_admit_the_same_callers():
    """The invariant the previous test's constant exists to protect, stated
    directly: both modules must gate on the SAME object, not on equal copies."""
    from navis.developer.agents_router import _AGENT_PERMISSIONS
    from navis.developer.gateway import AGENT_MANAGEMENT_PERMISSIONS

    assert _AGENT_PERMISSIONS is AGENT_MANAGEMENT_PERMISSIONS
