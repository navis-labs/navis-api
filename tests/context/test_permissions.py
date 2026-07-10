"""Security regression tests for navis/context/permissions.py: ABAC fail-closed, RBAC
union, monotonic scope narrowing, default-deny, policy validation + fingerprint.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest

from navis.context.contracts import ActorRef, PermissionScope
from navis.context.exceptions import PermissionConfigurationError
from navis.context.models import ContextScope, ScopeEntry, SelectionReason
from navis.context.permissions import (
    DefaultDenyPermissionFilter,
    RolePermissionResolver,
    intersect_scopes,
)
from navis.memory.envelope import EntityType
from navis.state.v2.models import OrganizationalState, Watermark
from tests.state.fakes import ORG_A

_PERSON = EntityType.PERSON.value
_TR = SelectionReason.TASK_RELEVANT.value


def _raw(eid, fields):
    return {"entity_id": eid, "first_seen": None, "last_seen": None,
            "fields": fields, "lists": {}, "links": {}, "provenance": {}}


def _state(persons):
    return OrganizationalState(org_id=ORG_A, as_of=datetime(2026, 6, 1, tzinfo=timezone.utc),
                              watermark=Watermark(), entities={_PERSON: persons})


def _scope(*ids):
    return ContextScope((_PERSON,), tuple(ScopeEntry(i, _PERSON, _TR) for i in ids))


def _filter(state, scope, *, allowed=(_PERSON,), predicates=()):
    perms = PermissionScope(allow_all=False, allowed_entity_types=tuple(allowed),
                            attribute_predicates=tuple(predicates))
    out = DefaultDenyPermissionFilter().resolve(ActorRef.system(), perms, scope, state)
    return {e.entity_id for e in out.entries}


# --- ABAC fail-closed (the headline security fix) ---------------------------

def test_abac_missing_field_is_denied_not_matched_as_none():
    # P1 lacks the "clearance" field; a predicate must NOT match it (old code let an absent
    # field stringify to "None" and match the literal "None").
    state = _state({"P1": _raw("P1", {}), "P2": _raw("P2", {"clearance": "None"})})
    assert _filter(state, _scope("P1", "P2"), predicates=(("clearance", "None"),)) == {"P2"}


def test_abac_present_field_match_and_mismatch():
    state = _state({"P1": _raw("P1", {"role": "from"}), "P2": _raw("P2", {"role": "to"})})
    assert _filter(state, _scope("P1", "P2"), predicates=(("role", "from"),)) == {"P1"}


def test_abac_entity_absent_from_state_is_denied():
    state = _state({"P1": _raw("P1", {"role": "from"})})  # P2 not in state
    assert _filter(state, _scope("P1", "P2"), predicates=(("role", "from"),)) == {"P1"}


def test_abac_all_predicates_conjunctive():
    state = _state({
        "P1": _raw("P1", {"role": "from", "region": "us"}),
        "P2": _raw("P2", {"role": "from", "region": "eu"}),
    })
    kept = _filter(state, _scope("P1", "P2"), predicates=(("role", "from"), ("region", "us")))
    assert kept == {"P1"}  # P2 fails the region predicate


def test_rbac_type_not_allowed_is_denied():
    state = _state({"P1": _raw("P1", {})})
    assert _filter(state, _scope("P1"), allowed=()) == set()  # default-deny: no types allowed


# --- RBAC resolution: additive union, default-deny --------------------------

def test_roles_are_additive_union():
    policy = {"a": frozenset({_PERSON}), "b": frozenset({EntityType.COMPANY.value})}
    actor = ActorRef(actor_id=uuid.uuid4(), kind="user", roles=("a", "b"))
    scope = RolePermissionResolver(policy).resolve(actor)
    assert set(scope.allowed_entity_types) == {_PERSON, EntityType.COMPANY.value}


def test_unknown_role_grants_nothing():
    actor = ActorRef(actor_id=uuid.uuid4(), kind="user", roles=("ghost-role",))
    assert RolePermissionResolver({"a": frozenset({_PERSON})}).resolve(actor).allowed_entity_types == ()


def test_system_actor_is_unrestricted():
    assert RolePermissionResolver().resolve(ActorRef.system()).allow_all is True


# --- intersect_scopes: monotonic narrowing, no escalation -------------------

def test_intersect_narrows_types_never_widens():
    granted = PermissionScope(allowed_entity_types=(_PERSON, EntityType.COMPANY.value))
    requested = PermissionScope(allowed_entity_types=(_PERSON, EntityType.COMMITMENT.value))
    eff = intersect_scopes(requested, granted)
    assert set(eff.allowed_entity_types) == {_PERSON}  # ∩ — narrower than either side
    assert set(eff.allowed_entity_types) <= set(granted.allowed_entity_types)
    assert set(eff.allowed_entity_types) <= set(requested.allowed_entity_types)


def test_requested_allow_all_is_bounded_by_grant():
    granted = PermissionScope(allowed_entity_types=(_PERSON,))
    eff = intersect_scopes(PermissionScope.system(), granted)  # caller asks for everything
    assert eff.allow_all is False and set(eff.allowed_entity_types) == {_PERSON}  # bounded


def test_predicates_union_is_more_restrictive():
    eff = intersect_scopes(
        PermissionScope(allowed_entity_types=(_PERSON,), attribute_predicates=(("a", "1"),)),
        PermissionScope(allowed_entity_types=(_PERSON,), attribute_predicates=(("b", "2"),)),
    )
    assert set(eff.attribute_predicates) == {("a", "1"), ("b", "2")}  # both apply (conjunctive)


# --- Policy validation + fingerprint (fail-fast + replay) -------------------

def test_policy_with_unknown_type_is_rejected():
    with pytest.raises(PermissionConfigurationError):
        RolePermissionResolver({"a": frozenset({"persons"})})  # typo of "person"


def test_resolver_fingerprint_is_deterministic_and_policy_sensitive():
    a = RolePermissionResolver({"r": frozenset({_PERSON})})
    b = RolePermissionResolver({"r": frozenset({_PERSON})})
    c = RolePermissionResolver({"r": frozenset({EntityType.COMPANY.value})})
    assert a.fingerprint() == b.fingerprint()
    assert a.fingerprint() != c.fingerprint()  # policy change → new fingerprint


# --- Tenant/integrity guard: never authorize an entity not in this org's State ----

def test_unbacked_entity_is_denied():
    state = _state({"P1": _raw("P1", {})})           # GHOST is NOT in State
    assert _filter(state, _scope("P1", "GHOST")) == {"P1"}  # foreign id denied (no predicates)


# --- Bootstrap policy misuse prevention -------------------------------------

def test_bootstrap_flag_is_set_correctly():
    assert RolePermissionResolver().is_bootstrap is True
    assert RolePermissionResolver({"viewer": frozenset({_PERSON})}).is_bootstrap is False


def test_bootstrap_policy_warns_only_for_non_system_actor():
    import logging
    from navis.context.permissions import logger as perm_logger
    records: list[logging.LogRecord] = []
    handler = logging.Handler()
    handler.emit = records.append  # type: ignore[method-assign]
    perm_logger.addHandler(handler)
    perm_logger.setLevel(logging.WARNING)
    try:
        RolePermissionResolver().resolve(ActorRef(actor_id=uuid.uuid4(), kind="user", roles=("viewer",)))
        RolePermissionResolver().resolve(ActorRef.system())  # system must NOT warn
        RolePermissionResolver({"viewer": frozenset({_PERSON})}).resolve(
            ActorRef(actor_id=uuid.uuid4(), kind="user", roles=("viewer",))
        )  # injected policy must NOT warn
    finally:
        perm_logger.removeHandler(handler)
    warnings = [r.getMessage() for r in records if "BOOTSTRAP" in r.getMessage()]
    assert len(warnings) == 1  # exactly the non-system + bootstrap case


# --- Replay: RBAC policy is part of the cache identity ----------------------

def test_rbac_policy_fingerprint_changes_context_id():
    from navis.context.contracts import ContextRequest, compute_context_id
    from navis.state.v2.builders import build_projection
    from tests.state.fakes import corpus
    state = build_projection(ORG_A, corpus())
    req = ContextRequest.full_org(ORG_A)
    a = compute_context_id(req, state, rbac_policy_fingerprint="policyA")
    b = compute_context_id(req, state, rbac_policy_fingerprint="policyB")
    assert a != b  # a policy change re-keys the cache (no stale-permission serve)


async def test_policy_change_invalidates_cache_end_to_end():
    from navis.context import ContextService, ContextRequest, ContextTask
    from navis.state.v2.builders import build_projection
    from tests.context.fakes import FakeStateReader
    from tests.state.fakes import corpus
    state = build_projection(ORG_A, corpus())
    actor = ActorRef(actor_id=uuid.uuid4(), kind="user", roles=("viewer",))
    req = ContextRequest(
        org_id=ORG_A, task=ContextTask.RELATIONSHIPS, actor=actor,
        permissions=PermissionScope(allowed_entity_types=(_PERSON, EntityType.COMPANY.value)),
    )
    svc_a = ContextService(state_service=FakeStateReader(state),
                           permission_resolver=RolePermissionResolver({"viewer": frozenset({_PERSON})}))
    svc_b = ContextService(state_service=FakeStateReader(state),
                           permission_resolver=RolePermissionResolver({"viewer": frozenset({EntityType.COMPANY.value})}))
    ba = await svc_a.build(req)
    bb = await svc_b.build(req)
    assert ba.context_id != bb.context_id  # different policy → different cache key
