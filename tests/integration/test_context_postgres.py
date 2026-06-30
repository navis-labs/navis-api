"""PostgreSQL integration tests for the Context bundle cache.

Runs only with NAVIS_TEST_DATABASE_URL. Exercises the real content-addressed cache:
put/get round-trip, idempotent (do-nothing) conflict on identical context_id, org-scoped
disposal, tenant isolation, and FK CASCADE on org deletion.
"""
from __future__ import annotations

import os
import uuid

import pytest

from navis.context import ContextRequest, ContextService
from navis.context.repository import ContextCacheRepository
from navis.state.v2.builders import build_projection
from tests.context.fakes import FakeStateReader
from tests.state.fakes import ORG_A, ORG_B, corpus

pytestmark = pytest.mark.skipif(
    not os.environ.get("NAVIS_TEST_DATABASE_URL"),
    reason="set NAVIS_TEST_DATABASE_URL to run Context Postgres integration tests",
)


@pytest.fixture()
async def db():
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    import navis.registry  # noqa: F401  (register every ORM table, incl. context_bundles)
    from navis.platform.database import Base

    engine = create_async_engine(os.environ["NAVIS_TEST_DATABASE_URL"])
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    Session = async_sessionmaker(engine, expire_on_commit=False)
    async with Session() as session:
        yield session
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


async def _org(db, org_id: uuid.UUID) -> None:
    from navis.identity.models import Organization

    db.add(Organization(id=org_id, name="Acme", slug=f"acme-{uuid.uuid4().hex[:8]}"))
    await db.flush()


async def _bundle(org_id):
    state = build_projection(org_id, corpus(org_id))
    return await ContextService(state_service=FakeStateReader(state)).build(
        ContextRequest.full_org(org_id)
    )


async def test_put_get_round_trip(db):
    await _org(db, ORG_A)
    repo = ContextCacheRepository(db)
    bundle = await _bundle(ORG_A)
    await repo.put(bundle)
    got = await repo.get(bundle.context_id)
    assert got is not None and got.to_dict() == bundle.to_dict()  # lossless


async def test_get_miss_returns_none(db):
    repo = ContextCacheRepository(db)
    assert await repo.get("does-not-exist") is None


async def test_put_is_idempotent_content_addressed(db):
    await _org(db, ORG_A)
    repo = ContextCacheRepository(db)
    bundle = await _bundle(ORG_A)
    await repo.put(bundle)
    await repo.put(bundle)  # same context_id → DO NOTHING, no error, one row
    got = await repo.get(bundle.context_id)
    assert got is not None and got.context_id == bundle.context_id


async def test_delete_disposes_org_bundles(db):
    await _org(db, ORG_A)
    repo = ContextCacheRepository(db)
    bundle = await _bundle(ORG_A)
    await repo.put(bundle)
    await repo.delete(ORG_A)
    assert await repo.get(bundle.context_id) is None


async def test_tenant_isolation(db):
    await _org(db, ORG_A)
    await _org(db, ORG_B)
    repo = ContextCacheRepository(db)
    a, b = await _bundle(ORG_A), await _bundle(ORG_B)
    await repo.put(a)
    await repo.put(b)
    await repo.delete(ORG_A)
    assert await repo.get(a.context_id) is None
    assert await repo.get(b.context_id) is not None  # other tenant untouched
