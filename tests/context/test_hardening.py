"""Regression tests for the production-hardening blockers (B1-B8).

Each test pins a previously-broken behaviour so it can never regress.
"""
from __future__ import annotations

import dataclasses
import logging
import uuid

from navis.context import (
    ActorRef,
    ContextRequest,
    ContextService,
    ContextTask,
    PermissionScope,
    TokenBudget,
)
from navis.context.contracts import (
    CONTEXT_SCHEMA_VERSION,
    CONTEXT_VERSION,
    ContextBundle,
    compute_context_id,
)
from navis.context.repository import ContextCacheRepository
from navis.memory.envelope import EntityType
from navis.state.v2.builders import build_projection
from tests.context.fakes import FakeStateReader, MemoryContextCache
from tests.state.fakes import ORG_A, corpus


def _state():
    return build_projection(ORG_A, corpus())


def _svc(state=None, cache=None):
    return ContextService(state_service=FakeStateReader(state or _state()), cache=cache)


def _req(task=ContextTask.FULL_ORG, *, actor=None, permissions=None, budget=None):
    return ContextRequest(
        org_id=ORG_A, task=task,
        actor=actor or ActorRef.system(),
        permissions=permissions or PermissionScope.system(),
        budget=budget or TokenBudget(),
    )


# --- B1: budget is part of cache identity -----------------------------------

def test_budget_changes_context_id():
    state = _state()
    a = compute_context_id(_req(budget=TokenBudget(10)), state)
    b = compute_context_id(_req(budget=TokenBudget.unbounded()), state)
    assert a != b


async def test_cache_never_serves_wrong_budget():
    cache = MemoryContextCache()
    svc = _svc(cache=cache)
    small = await svc.build(_req(ContextTask.COMMITMENTS, budget=TokenBudget(5)))
    big = await svc.build(_req(ContextTask.COMMITMENTS, budget=TokenBudget.unbounded()))
    assert small.context_id != big.context_id            # no collision
    assert small.budget_report.admitted <= big.budget_report.admitted


# --- B2: no dead fields -----------------------------------------------------

def test_request_has_no_dead_fields():
    names = {f.name for f in dataclasses.fields(ContextRequest)}
    assert "pinned_watermark" not in names and "goal" not in names


# --- B3: FULL_ORG embeds an INDEPENDENT state -------------------------------

async def test_full_org_state_is_independent_copy():
    state = _state()
    before = state.to_dict()
    bundle = await _svc(state).build(_req())
    assert bundle.state is not state                      # not aliased
    assert bundle.state.to_dict() == before               # equal content (parity)
    for bucket in bundle.state.entities.values():          # mutate the bundle
        for raw in bucket.values():
            raw["fields"]["__mutated__"] = True
    assert state.to_dict() == before                      # source untouched


# --- B4: RBAC resolver bounds the caller (no escalation) ---------------------

async def test_resolver_bounds_caller_no_escalation():
    # guest grants only person+company; the caller asks for allow_all → bounded.
    actor = ActorRef(actor_id=uuid.uuid4(), kind="user", roles=("guest",))
    bundle = await _svc().build(_req(actor=actor, permissions=PermissionScope.system()))
    types = {e.entity_type for e in bundle.scope.entries}
    assert types <= {EntityType.PERSON.value, EntityType.COMPANY.value}
    assert EntityType.COMMITMENT.value not in types


async def test_unknown_role_is_default_deny():
    actor = ActorRef(actor_id=uuid.uuid4(), kind="user", roles=("nobody",))
    bundle = await _svc().build(_req(actor=actor, permissions=PermissionScope.system()))
    assert bundle.scope.entries == ()


async def test_system_actor_unrestricted():
    bundle = await _svc().build(_req())
    assert bundle.budget_report.admitted == 8


# --- B5: audit trail (no PII) ------------------------------------------------

async def test_audit_record_is_emitted():
    from navis.context.telemetry import audit_logger
    records: list[logging.LogRecord] = []
    handler = logging.Handler()
    handler.emit = records.append  # type: ignore[method-assign]
    audit_logger.addHandler(handler)
    audit_logger.setLevel(logging.INFO)
    try:
        await _svc().build(_req())
    finally:
        audit_logger.removeHandler(handler)
    assert records, "expected an audit record per build"
    msg = records[0].getMessage()
    for field in ("context_id=", "perm_fp=", "policy_fp=", "entities=", "tokens=",
                  "latency_ms=", "actor_kind=", "actor_fp="):
        assert field in msg
    assert "email" not in msg and "@" not in msg  # no PII / entity content


# --- B6: cache robustness ----------------------------------------------------

class _Row:
    def __init__(self, cv, sv, bundle):
        self.context_version, self.schema_version, self.bundle = cv, sv, bundle


class _FakeDB:
    def __init__(self, row):
        self._row = row

    async def scalar(self, _stmt):
        return self._row


async def test_cache_get_rejects_version_mismatch():
    repo = ContextCacheRepository(_FakeDB(_Row(CONTEXT_VERSION + 1, CONTEXT_SCHEMA_VERSION, {})))  # type: ignore[arg-type]
    assert await repo.get("x") is None


async def test_cache_get_recovers_from_corruption():
    repo = ContextCacheRepository(_FakeDB(_Row(CONTEXT_VERSION, CONTEXT_SCHEMA_VERSION, {"garbage": True})))  # type: ignore[arg-type]
    assert await repo.get("x") is None  # from_dict would raise → graceful miss


async def test_cache_get_miss_returns_none():
    repo = ContextCacheRepository(_FakeDB(None))  # type: ignore[arg-type]
    assert await repo.get("x") is None


# --- B7: unbounded budget fast path -----------------------------------------

async def test_unbounded_budget_skips_token_serialization():
    bundle = await _svc().build(_req())
    assert bundle.budget_report.token_estimate == 0    # fast path: no per-entity serialization
    assert bundle.budget_report.admitted == 8


async def test_bounded_budget_accounts_tokens():
    bundle = await _svc().build(_req(budget=TokenBudget(50)))
    assert bundle.budget_report.token_estimate > 0
    assert bundle.budget_report.token_estimate <= 50
