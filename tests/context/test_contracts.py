"""Unit tests for navis/context/contracts.py hardening: value-object validation,
unambiguous fingerprinting, deterministic serialization, and bundle invariants.
"""
from __future__ import annotations

import dataclasses
import uuid
from datetime import datetime, timezone

import pytest

from navis.context import (
    ActorRef,
    ContextRequest,
    ContextService,
    ContextTask,
    PermissionScope,
    TokenBudget,
)
from navis.context.contracts import (
    CONTEXT_VERSION,
    ContextBundle,
    _digest,
    compute_context_id,
)
from navis.context.models import BudgetReport
from navis.state.v2.builders import build_projection
from tests.context.fakes import FakeStateReader
from tests.state.fakes import ORG_A, corpus


def _svc():
    return ContextService(state_service=FakeStateReader(build_projection(ORG_A, corpus())))


# --- TokenBudget validation -------------------------------------------------

@pytest.mark.parametrize("bad", [0, -1, -1000, True])
def test_invalid_token_budget_rejected(bad):
    with pytest.raises(ValueError):
        TokenBudget(max_tokens=bad)


def test_valid_token_budget_accepted():
    assert TokenBudget(1).max_tokens == 1
    assert TokenBudget.unbounded().max_tokens is None


# --- policy_fingerprint / token validation ----------------------------------

@pytest.mark.parametrize("bad", ["", "x" * 257, "has\x1fcontrol", "tab\tchar"])
def test_invalid_policy_fingerprint_rejected(bad):
    with pytest.raises(ValueError):
        ContextRequest(ORG_A, ContextTask.FULL_ORG, ActorRef.system(),
                       PermissionScope.system(), policy_fingerprint=bad)


# --- UUID / type validation -------------------------------------------------

def test_invalid_org_id_rejected():
    with pytest.raises(ValueError):
        ContextRequest("not-a-uuid", ContextTask.FULL_ORG, ActorRef.system(), PermissionScope.system())  # type: ignore[arg-type]


def test_invalid_actor_kind_rejected():
    with pytest.raises(ValueError):
        ActorRef(actor_id=uuid.uuid4(), kind="root")  # type: ignore[arg-type]


def test_allow_all_with_constraints_rejected():
    with pytest.raises(ValueError):
        PermissionScope(allow_all=True, allowed_entity_types=("person",))


# --- Fingerprint: deterministic + unambiguous -------------------------------

def test_digest_is_deterministic():
    assert _digest("a", "b", "c") == _digest("a", "b", "c")


def test_digest_has_no_delimiter_ambiguity():
    # Different field groupings must NEVER collide (the old \x1f-join could).
    assert _digest("a", "bc") != _digest("ab", "c")
    assert _digest("a\x1fb", "c") != _digest("a", "b\x1fc")
    assert _digest("", "ab") != _digest("a", "b")


def test_fingerprints_are_stable_across_instances():
    a1 = ActorRef(actor_id=uuid.UUID(int=7), kind="user", roles=("b", "a"))
    a2 = ActorRef(actor_id=uuid.UUID(int=7), kind="user", roles=("a", "b"))  # order-insensitive
    assert a1.fingerprint() == a2.fingerprint()
    p1 = PermissionScope(allowed_entity_types=("b", "a"))
    p2 = PermissionScope(allowed_entity_types=("a", "b"))
    assert p1.fingerprint() == p2.fingerprint()


# --- Deterministic context_id -----------------------------------------------

async def test_context_id_deterministic_and_version_folded():
    state = build_projection(ORG_A, corpus())
    req = ContextRequest.full_org(ORG_A)
    assert compute_context_id(req, state) == compute_context_id(req, state)
    # Folding a different version yields a different id (migration safety).
    assert compute_context_id(req, state, context_version=CONTEXT_VERSION) != \
        compute_context_id(req, state, context_version=CONTEXT_VERSION + 1)


# --- Serialization round-trip ----------------------------------------------

async def test_bundle_round_trip_is_lossless_and_deterministic():
    bundle = await _svc().build(ContextRequest.full_org(ORG_A))
    d1 = bundle.to_dict()
    again = ContextBundle.from_dict(d1).to_dict()
    assert again == d1
    import json
    assert json.dumps(d1, sort_keys=True) == json.dumps(again, sort_keys=True)


# --- ContextBundle.validate() invariants ------------------------------------

async def test_validate_rejects_admitted_mismatch():
    good = await _svc().build(ContextRequest.full_org(ORG_A))
    with pytest.raises(ValueError):
        dataclasses.replace(good, budget_report=BudgetReport(0, 999, 0, None))


async def test_validate_rejects_logical_clock_drift():
    good = await _svc().build(ContextRequest.full_org(ORG_A))
    with pytest.raises(ValueError):
        dataclasses.replace(good, created_at=datetime(2000, 1, 1, tzinfo=timezone.utc))


async def test_validate_rejects_empty_context_id():
    good = await _svc().build(ContextRequest.full_org(ORG_A))
    with pytest.raises(ValueError):
        dataclasses.replace(good, context_id="")
