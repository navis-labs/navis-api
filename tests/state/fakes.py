"""Controlled memory-record corpus + fake reader/snapshots for State V2 tests."""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

from navis.memory.envelope import EntityType
from navis.state.v2.builders import MemoryRecordView
from navis.state.v2.models import OrganizationalState, order_key

NS = uuid.UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
ORG_A = uuid.UUID("11111111-1111-1111-1111-111111111111")
ORG_B = uuid.UUID("22222222-2222-2222-2222-222222222222")
BASE = datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)


def rec(entity_type: EntityType, natural_key: str, payload: dict, *, offset: int, tag: str,
        org_id: uuid.UUID = ORG_A) -> MemoryRecordView:
    return MemoryRecordView(
        memory_id=uuid.uuid5(NS, f"mem:{org_id}:{tag}"),
        org_id=org_id,
        entity_type=entity_type.value,
        entity_id=uuid.uuid5(NS, f"ent:{org_id}:{entity_type.value}:{natural_key}"),
        event_timestamp=BASE + timedelta(seconds=offset),
        payload=payload,
        source_signal_event_id=uuid.uuid5(NS, f"sig:{org_id}:{tag}"),
        schema_version=1,
        confidence=0.9,
    )


def corpus(org_id: uuid.UUID = ORG_A) -> list[MemoryRecordView]:
    """A realistic multi-entity, multi-version corpus (same ts for the first two
    to exercise the memory_id tiebreak)."""
    return [
        rec(EntityType.PERSON, "alice@acme.com", {"email": "alice@acme.com", "name": "Alice", "role": "from"},
            offset=0, tag="p-alice", org_id=org_id),
        rec(EntityType.PERSON, "bob@acme.com", {"email": "bob@acme.com", "name": "Bob", "role": "to"},
            offset=0, tag="p-bob", org_id=org_id),  # same ts → tiebreak by memory_id
        rec(EntityType.COMPANY, "acme.com", {"domain": "acme.com"},
            offset=1, tag="c-acme", org_id=org_id),
        rec(EntityType.RELATIONSHIP, "alice@acme.com", {"person_email": "alice@acme.com", "direction": "inbound"},
            offset=2, tag="r-alice-in", org_id=org_id),
        rec(EntityType.RELATIONSHIP, "alice@acme.com", {"person_email": "alice@acme.com", "direction": "outbound"},
            offset=3, tag="r-alice-out", org_id=org_id),  # accumulate directions, advance last_seen
        rec(EntityType.COMMITMENT, "msg-1:0", {"text": "I'll send the deck", "speaker_email": "alice@acme.com"},
            offset=4, tag="cm-1", org_id=org_id),
        rec(EntityType.THREAD, "thread-9", {"thread_id": "thread-9", "subject": "terms", "message_id": "m1"},
            offset=5, tag="t-m1", org_id=org_id),
        rec(EntityType.THREAD, "thread-9", {"thread_id": "thread-9", "subject": "terms", "message_id": "m2"},
            offset=6, tag="t-m2", org_id=org_id),  # accumulate message_ids
        rec(EntityType.MEETING, "evt-1", {"title": "Board", "status": "confirmed",
                                          "attendees": ["alice@acme.com", "bob@acme.com"]},
            offset=7, tag="mt-1", org_id=org_id),
        rec(EntityType.MEETING, "evt-1", {"title": "Board", "status": "cancelled",
                                          "attendees": ["alice@acme.com"]},
            offset=8, tag="mt-2", org_id=org_id),  # latest-wins: status + attendees
        rec(EntityType.DECISION, "msg-1:3", {"text": "We decided to raise"},
            offset=9, tag="d-1", org_id=org_id),
    ]


class FakeReader:
    def __init__(self, records=None) -> None:
        self._records = list(records or [])

    def add(self, records) -> None:
        self._records.extend(records)

    async def stream(self, org_id, *, after=None, batch=5000):
        rows = sorted(
            (r for r in self._records if r.org_id == org_id),
            key=lambda r: order_key(r.event_timestamp, r.memory_id),
        )
        for r in rows:
            if after is None or after.passed(r.event_timestamp, r.memory_id):
                yield r

    async def has_records_after(self, org_id, watermark) -> bool:
        """Mirror MemoryReader.has_records_after: True iff any record past `watermark`."""
        return any(
            r.org_id == org_id and watermark.passed(r.event_timestamp, r.memory_id)
            for r in self._records
        )


class _Snap:
    def __init__(self, full_projection: dict) -> None:
        self.full_projection = full_projection


class FakeSnapshots:
    def __init__(self) -> None:
        self.store: dict = {}

    async def latest(self, org_id, *, projection_version):
        return self.store.get((org_id, projection_version))

    async def watermark_only(self, org_id, *, projection_version):
        """Mirror SnapshotRepository.watermark_only — return (ts, mid) or None."""
        import uuid as _uuid
        from datetime import datetime as _dt

        snap = self.store.get((org_id, projection_version))
        if snap is None:
            return None
        wm = (snap.full_projection or {}).get("watermark") or {}
        ts_str = wm.get("last_timestamp")
        mid_str = wm.get("last_memory_id")
        return (
            _dt.fromisoformat(ts_str) if ts_str else None,
            _uuid.UUID(mid_str) if mid_str else None,
        )

    async def save(self, state: OrganizationalState) -> None:
        self.store[(state.org_id, state.projection_version)] = _Snap(state.to_dict())

    async def delete(self, org_id, *, projection_version=None) -> None:
        # Mirrors SnapshotRepository.delete: drop one version, or all versions for the org.
        if projection_version is None:
            for k in [k for k in self.store if k[0] == org_id]:
                del self.store[k]
        else:
            self.store.pop((org_id, projection_version), None)
