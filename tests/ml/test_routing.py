"""Phase 4 regression tests — capability router.

Asserts the router is capability-based (never a model name in the call), DETERMINISTIC (pure
function of capability + configured set), backward-compatible (empty preference → the
deployment's configured-default order), and separate from construction (delegates to the
registry's get_provider / get_embedding_provider and reads no secrets). The preference
*mechanism* is exercised via a monkeypatched map — production ships with no vendor opinions.
"""
from __future__ import annotations

from navis.ml import routing
from navis.ml.routing import Capability


def _stub(monkeypatch, configured):
    """Route through a recorder: capture the provider NAME the router asks the registry for."""
    seen: list[str | None] = []
    monkeypatch.setattr(routing, "configured_providers", lambda: configured)
    monkeypatch.setattr(routing, "get_provider", lambda name=None: (seen.append(name), name)[1])
    return seen


# --- default policy (empty preference → configured-default order; backward-compatible) ------

def test_route_defaults_to_first_configured(monkeypatch):
    seen = _stub(monkeypatch, ("azure", "deterministic"))
    assert routing.route(Capability.REASONING) == "azure"  # no vendor opinion → configured default
    assert seen == ["azure"]


def test_route_default_is_deterministic(monkeypatch):
    _stub(monkeypatch, ("gemini", "azure", "deterministic"))
    a = routing.route(Capability.ACCURATE)
    b = routing.route(Capability.ACCURATE)
    assert a == b == "gemini"  # first configured, same every time


# --- preference MECHANISM (populated only from evidence; exercised here via monkeypatch) -----

def test_route_honors_explicit_preference(monkeypatch):
    seen = _stub(monkeypatch, ("azure", "claude", "deterministic"))
    monkeypatch.setattr(routing, "_CAPABILITY_PREFERENCE", {Capability.REASONING: ("claude", "azure")})
    assert routing.route(Capability.REASONING) == "claude"  # preferred + configured
    assert seen == ["claude"]


def test_route_skips_unconfigured_preference(monkeypatch):
    seen = _stub(monkeypatch, ("azure", "deterministic"))
    monkeypatch.setattr(routing, "_CAPABILITY_PREFERENCE", {Capability.REASONING: ("claude", "azure")})
    assert routing.route(Capability.REASONING) == "azure"  # claude not configured → next preferred
    assert seen == ["azure"]


def test_route_falls_back_when_no_preferred_configured(monkeypatch):
    seen = _stub(monkeypatch, ("deterministic",))
    monkeypatch.setattr(routing, "_CAPABILITY_PREFERENCE", {Capability.TOOL_CALLING: ("claude", "azure")})
    assert routing.route(Capability.TOOL_CALLING) == "deterministic"  # terminal fallback
    assert seen == ["deterministic"]


# --- embedding is pinned in the registry, never routed via generation preference ------------

def test_embedding_routes_to_pinned_home(monkeypatch):
    monkeypatch.setattr(routing, "get_embedding_provider", lambda: "EMB")
    assert routing.route(Capability.EMBEDDING) == "EMB"


# --- routing owns capability resolution ONLY — no business/product vocabulary ---------------

def test_routing_exposes_no_business_vocabulary():
    for removed in ("Task", "Tier", "provider_for", "tier_for", "TASK_TIER"):
        assert not hasattr(routing, removed), f"routing must not know a business concept: {removed}"


def test_callers_use_capability_not_vendor_strings():
    assert Capability.REASONING.value == "reasoning"
    assert {c.value for c in Capability} >= {"reasoning", "embedding", "tool_calling", "structured_output"}
