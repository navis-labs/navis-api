"""StateService: rebuild, get, incremental + watermark recovery (with fakes)."""
from __future__ import annotations

from navis.state.v2.builders import build_projection
from navis.state.v2.models import order_key
from navis.state.v2.service import StateService
from tests.state.fakes import ORG_A, FakeReader, FakeSnapshots, corpus


def _svc(records):
    reader = FakeReader(records)
    snaps = FakeSnapshots()
    return StateService(reader=reader, snapshots=snaps), reader, snaps


async def test_rebuild_then_get_round_trips():
    svc, _, _ = _svc(corpus())
    rebuilt = await svc.rebuild_org(ORG_A)
    fetched = await svc.get_state(ORG_A)
    assert fetched.to_dict() == rebuilt.to_dict()  # snapshot serialization is lossless


async def test_get_state_rebuilds_when_no_snapshot():
    svc, _, _ = _svc(corpus())
    state = await svc.get_state(ORG_A)
    assert state.counts()["person"] >= 1


async def test_incremental_matches_full_via_service():
    records = sorted(corpus(), key=lambda r: order_key(r.event_timestamp, r.memory_id))
    half = len(records) // 2

    # Reader starts with the first half; build a snapshot.
    svc, reader, _ = _svc(records[:half])
    await svc.rebuild_org(ORG_A)

    # New memory arrives; incremental folds only the records after the watermark.
    reader.add(records[half:])
    incr = await svc.incremental_update(ORG_A)

    full = build_projection(ORG_A, records)
    assert incr.to_dict() == full.to_dict()


async def test_incremental_noop_when_no_new_records():
    svc, _, _ = _svc(corpus())
    first = await svc.rebuild_org(ORG_A)
    again = await svc.incremental_update(ORG_A)
    assert again.to_dict() == first.to_dict()  # nothing past the watermark → unchanged
