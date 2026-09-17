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
