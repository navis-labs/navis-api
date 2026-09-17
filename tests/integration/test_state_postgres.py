"""PostgreSQL integration tests for State V2 SnapshotRepository.

Runs only with NAVIS_TEST_DATABASE_URL. Exercises the real SQL that the in-memory
fakes cannot: the monotonic ON CONFLICT upsert (stale-snapshot protection), the
GDPR delete primitive, projection-version isolation, and tenant isolation — all
against a live database. Snapshot content is a pure projection of in-memory
MemoryRecordViews, so no signal/memory rows need seeding (only an Organization for
the FK).
"""
from __future__ import annotations

import os
import uuid

import pytest

from navis.state.v2.builders import build_projection
from navis.state.v2.models import PROJECTION_VERSION
from navis.state.v2.repository import SnapshotRepository
from tests.state.fakes import ORG_A, ORG_B, corpus

pytestmark = pytest.mark.skipif(
    not os.environ.get("NAVIS_TEST_DATABASE_URL"),
    reason="set NAVIS_TEST_DATABASE_URL to run State Postgres integration tests",
)


@pytest.fixture()
async def db():
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    import navis.registry  # noqa: F401  (register every ORM table)
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


async def test_save_and_latest_round_trip(db):
    await _org(db, ORG_A)
    repo = SnapshotRepository(db)
    state = build_projection(ORG_A, corpus())
    await repo.save(state)
    snap = await repo.latest(ORG_A, projection_version=state.projection_version)
    assert snap is not None
    assert snap.full_projection == state.to_dict()  # lossless persistence


async def test_upsert_is_monotonic_stale_write_is_ignored(db):
    await _org(db, ORG_A)
    repo = SnapshotRepository(db)
    full = build_projection(ORG_A, corpus())          # newest watermark
    older = build_projection(ORG_A, corpus()[:5])     # older watermark (subset)
    assert older.watermark.to_dict() != full.watermark.to_dict()

    await repo.save(full)
    await repo.save(older)  # stale: must NOT regress the cache
    snap = await repo.latest(ORG_A, projection_version=full.projection_version)
    assert snap is not None and snap.full_projection == full.to_dict()

    # A genuinely newer write still wins.
    await repo.delete(ORG_A)
    await repo.save(older)
    await repo.save(full)
    snap2 = await repo.latest(ORG_A, projection_version=full.projection_version)
    assert snap2 is not None and snap2.full_projection == full.to_dict()


async def test_duplicate_save_is_idempotent(db):
    await _org(db, ORG_A)
    repo = SnapshotRepository(db)
    state = build_projection(ORG_A, corpus())
    await repo.save(state)
    await repo.save(state)  # same watermark; <= guard allows, content identical
    snap = await repo.latest(ORG_A, projection_version=state.projection_version)
    assert snap is not None and snap.full_projection == state.to_dict()


async def test_delete_disposes_cache(db):
    await _org(db, ORG_A)
    repo = SnapshotRepository(db)
    await repo.save(build_projection(ORG_A, corpus()))
    await repo.delete(ORG_A)
    assert await repo.latest(ORG_A, projection_version=1) is None


async def test_version_isolation(db):
    await _org(db, ORG_A)
    repo = SnapshotRepository(db)
    state = build_projection(ORG_A, corpus())
    await repo.save(state)
    assert await repo.latest(ORG_A, projection_version=state.projection_version) is not None
    assert await repo.latest(ORG_A, projection_version=999) is None  # other version absent


async def test_tenant_isolation(db):
    # Reads the CURRENT projection version rather than a literal, exactly like
    # `test_version_isolation` above. This test is about one tenant's snapshot
    # never being visible to another; pinning the literal `1` made it fail for
    # the unrelated reason that the projection version moved on.
    await _org(db, ORG_A)
    await _org(db, ORG_B)
    repo = SnapshotRepository(db)
    await repo.save(build_projection(ORG_A, corpus(ORG_A)))
    await repo.save(build_projection(ORG_B, corpus(ORG_B)))
    a = await repo.latest(ORG_A, projection_version=PROJECTION_VERSION)
    b = await repo.latest(ORG_B, projection_version=PROJECTION_VERSION)
    assert a is not None and b is not None
    assert a.organization_id == ORG_A and b.organization_id == ORG_B
    # Disposing one tenant never touches the other.
    await repo.delete(ORG_A)
    assert await repo.latest(ORG_A, projection_version=PROJECTION_VERSION) is None
    assert await repo.latest(ORG_B, projection_version=PROJECTION_VERSION) is not None


async def test_half_set_watermark_is_rejected(db):
    # A partial-write watermark (one half NULL) must be impossible to persist.
    import datetime as _dt

    from sqlalchemy.exc import IntegrityError

    from navis.state.v2.snapshots import StateSnapshot

    await _org(db, ORG_A)
    db.add(StateSnapshot(
        organization_id=ORG_A, projection_version=1, schema_version=1,
        watermark_timestamp=_dt.datetime.now(_dt.timezone.utc),
        watermark_memory_id=None,  # half-set → violates ck_state_snapshots_watermark_paired
        as_of=None, full_projection={},
    ))
    with pytest.raises(IntegrityError):
        await db.flush()
