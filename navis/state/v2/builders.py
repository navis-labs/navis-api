"""Projection builders — full rebuild and incremental, sharing one reducer.

Determinism by construction:
  * records are sorted by (event_timestamp, memory_id) before folding;
  * the fold is a left-fold of a per-record reducer (`StateBuilder.apply`);
  * full build = fold ALL records from empty; incremental = continue the SAME
    fold from a prior projection with records strictly after its watermark.
Because the reducer is monotone in fold order (latest-non-null-wins scalars, union
lists, fixed first_seen), incremental output == full rebuild output, always.

Cross-links are recomputed from scratch in `finalize` over the (identical) entity
sets, so they match for full and incremental too.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable

from navis.memory.envelope import EntityType
from navis.state.v2.folds import compute_links
from navis.state.v2.models import OrganizationalState, Watermark, order_key


@dataclass(frozen=True, slots=True)
class MemoryRecordView:
    """Read-only view of one memory_records row handed to the projection."""

    memory_id: uuid.UUID
    org_id: uuid.UUID
    entity_type: str
    entity_id: uuid.UUID
    event_timestamp: datetime
    payload: dict[str, Any]
    source_signal_event_id: uuid.UUID
    schema_version: int
    confidence: float


def _add_unique(lst: list, value) -> None:
    if value is not None and value not in lst:
        lst.append(value)
        lst.sort(key=lambda x: (str(type(x)), str(x)))  # deterministic order


class StateBuilder:
    """Mutable working accumulator. Deterministic given the input fold order."""

    def __init__(self, org_id: uuid.UUID) -> None:
        self.org_id = org_id
        self.entities: dict[str, dict[str, dict]] = {}
        self.last_ts: datetime | None = None
        self.last_mid: uuid.UUID | None = None

    @classmethod
    def from_state(cls, state: OrganizationalState) -> "StateBuilder":
        b = cls(state.org_id)
        # Copy entities so we never mutate the caller's projection.
        b.entities = {
            et: {eid: _deep_copy_raw(raw) for eid, raw in bucket.items()}
            for et, bucket in state.entities.items()
        }
        b.last_ts = state.watermark.last_timestamp
        b.last_mid = state.watermark.last_memory_id
        return b

    def apply(self, rec: MemoryRecordView) -> None:
        bucket = self.entities.setdefault(rec.entity_type, {})
        eid = str(rec.entity_id)
        ts_iso = rec.event_timestamp.isoformat()
        raw = bucket.get(eid)
        if raw is None:
            raw = {
                "entity_id": eid,
                "first_seen": ts_iso,  # creating record is earliest (sorted order)
                "last_seen": ts_iso,
                "fields": {},
                "lists": {},
                "links": {},
                "provenance": {"memory_ids": [], "signal_ids": []},
            }
            bucket[eid] = raw
        else:
            raw["last_seen"] = ts_iso  # monotonic: applied in ascending order

        # Latest NON-NULL wins. A sparser newer record (e.g. a Slack authorship event
        # carrying email=None) must never null out a field a richer earlier record
        # established. None is "unknown here", not "cleared" — Memory is additive, so a
        # null carries no information and is skipped. Order-independent and replay-safe:
        # the surviving value is always the latest non-null by (event_timestamp, memory_id),
        # so incremental == full still holds (prev never stores a null-clobbered value).
        for k, v in rec.payload.items():
            if v is not None:
                raw["fields"][k] = v

        # Provenance (sorted-unique).
        _add_unique(raw["provenance"]["memory_ids"], str(rec.memory_id))
        _add_unique(raw["provenance"]["signal_ids"], str(rec.source_signal_event_id))

        self._accumulate_lists(rec.entity_type, raw, rec.payload)

        # Watermark advances to the newest folded record.
        self.last_ts = rec.event_timestamp
        self.last_mid = rec.memory_id

    @staticmethod
    def _accumulate_lists(entity_type: str, raw: dict, payload: dict) -> None:
        lists = raw["lists"]
        if entity_type == EntityType.THREAD.value:
            _add_unique(lists.setdefault("message_ids", []), payload.get("message_id"))
        elif entity_type == EntityType.RELATIONSHIP.value:
            _add_unique(lists.setdefault("directions", []), payload.get("direction"))
        elif entity_type == EntityType.PROJECT.value:
            _add_unique(lists.setdefault("document_names", []), payload.get("name"))

    def finalize(self, *, as_of: datetime | None) -> OrganizationalState:
        compute_links(self.entities)  # recompute cross-links from scratch
        return OrganizationalState(
            org_id=self.org_id,
            as_of=as_of if as_of is not None else self.last_ts,
            watermark=Watermark(self.last_ts, self.last_mid),
            entities=self.entities,
        )


def _deep_copy_raw(raw: dict) -> dict:
    import copy

    return copy.deepcopy(raw)


def _sorted(records: Iterable[MemoryRecordView]) -> list[MemoryRecordView]:
    return sorted(records, key=lambda r: order_key(r.event_timestamp, r.memory_id))


def build_projection(
    org_id: uuid.UUID,
    records: Iterable[MemoryRecordView],
    *,
    as_of: datetime | None = None,
) -> OrganizationalState:
    """Full rebuild: fold ALL records from empty."""
    builder = StateBuilder(org_id)
    for rec in _sorted(records):
        builder.apply(rec)
    return builder.finalize(as_of=as_of)


def continue_projection(
    prev: OrganizationalState,
    new_records: Iterable[MemoryRecordView],
    *,
    as_of: datetime | None = None,
) -> OrganizationalState:
    """Incremental: continue the fold from a prior projection with newer records."""
    builder = StateBuilder.from_state(prev)
    for rec in _sorted(new_records):
        builder.apply(rec)
    return builder.finalize(as_of=as_of)
