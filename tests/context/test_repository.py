"""In-process tests for navis/context/repository.py (no DB): SQL construction for the
content-addressed upsert / org-scoped delete / created_at prune, and get()'s version-mismatch
+ corruption recovery. Real Postgres semantics live in tests/integration/test_context_postgres.py.
"""
from __future__ import annotations

import types
from datetime import datetime, timezone

from sqlalchemy.dialects import postgresql

from navis.context import ContextRequest, ContextService, ContextTask, ActorRef, PermissionScope
from navis.context.contracts import CONTEXT_SCHEMA_VERSION, CONTEXT_VERSION
from navis.context.repository import ContextCacheRepository, ContextSnapshot
from navis.state.v2.builders import build_projection
from tests.context.fakes import FakeStateReader
from tests.state.fakes import ORG_A, corpus


class _RecSession:
    def __init__(self, scalar_row=None):
        self.stmts = []
        self._scalar_row = scalar_row

    async def execute(self, stmt):
        self.stmts.append(stmt)
        return types.SimpleNamespace(rowcount=0)

    async def scalar(self, _stmt):
        return self._scalar_row


def _sql(stmt):
    return str(stmt.compile(dialect=postgresql.dialect())).upper()


async def _bundle():
    state = build_projection(ORG_A, corpus())
    return await ContextService(state_service=FakeStateReader(state)).build(
        ContextRequest(org_id=ORG_A, task=ContextTask.COMMITMENTS,
                       actor=ActorRef.system(), permissions=PermissionScope.system())
    )


# --- SQL construction -------------------------------------------------------

async def test_put_is_self_healing_upsert():
    sess = _RecSession()
    await ContextCacheRepository(sess).put(await _bundle())  # type: ignore[arg-type]
    sql = _sql(sess.stmts[0])
    assert "INSERT INTO CONTEXT_BUNDLES" in sql
    assert "ON CONFLICT" in sql and "DO UPDATE" in sql  # self-heals on a same-id shape change


async def test_delete_is_org_scoped():
    sess = _RecSession()
    await ContextCacheRepository(sess).delete(ORG_A)  # type: ignore[arg-type]
    sql = _sql(sess.stmts[0])
    assert sql.startswith("DELETE FROM CONTEXT_BUNDLES") and "ORGANIZATION_ID" in sql


async def test_prune_targets_created_at():
    sess = _RecSession()
    n = await ContextCacheRepository(sess).prune(datetime(2020, 1, 1, tzinfo=timezone.utc))  # type: ignore[arg-type]
    sql = _sql(sess.stmts[0])
    assert sql.startswith("DELETE FROM CONTEXT_BUNDLES") and "CREATED_AT" in sql
    assert n == 0  # rowcount from the fake


# --- get(): graceful invalidation + corruption recovery (no DB) -------------

def _row(*, cv, sv, bundle):
    return ContextSnapshot(context_id="x", organization_id=ORG_A, context_version=cv,
                           schema_version=sv, bundle=bundle)


async def test_get_miss_returns_none():
    assert await ContextCacheRepository(_RecSession(scalar_row=None)).get("x") is None  # type: ignore[arg-type]


async def test_get_rejects_version_mismatch():
    row = _row(cv=CONTEXT_VERSION + 1, sv=CONTEXT_SCHEMA_VERSION, bundle={})
    assert await ContextCacheRepository(_RecSession(scalar_row=row)).get("x") is None  # type: ignore[arg-type]
    row2 = _row(cv=CONTEXT_VERSION, sv=CONTEXT_SCHEMA_VERSION + 1, bundle={})
    assert await ContextCacheRepository(_RecSession(scalar_row=row2)).get("x") is None  # type: ignore[arg-type]


async def test_get_recovers_from_corrupt_bundle():
    row = _row(cv=CONTEXT_VERSION, sv=CONTEXT_SCHEMA_VERSION, bundle={"garbage": True})
    assert await ContextCacheRepository(_RecSession(scalar_row=row)).get("x") is None  # type: ignore[arg-type]


async def test_get_round_trips_a_valid_bundle():
    bundle = await _bundle()
    row = _row(cv=CONTEXT_VERSION, sv=CONTEXT_SCHEMA_VERSION, bundle=bundle.to_dict())
    got = await ContextCacheRepository(_RecSession(scalar_row=row)).get(bundle.context_id)  # type: ignore[arg-type]
    assert got is not None and got.context_id == bundle.context_id
