"""Projection builders — full rebuild and incremental, sharing one reducer.

Determinism by construction:
  * records are folded in (event_timestamp, memory_id) order — see `order_key`;
  * the fold is a left-fold of a per-record reducer (`StateBuilder.apply`);
  * full build = fold ALL records from empty; incremental = continue the SAME
    fold from a prior projection with records the cursor has not yet seen.
Because the reducer is monotone in fold order (latest-non-null-wins scalars, union
lists, fixed first_seen), incremental output == full rebuild output — PROVIDED the
incremental delta folds strictly after the prior `fold_frontier`. That precondition
is checked by `StateService.incremental_update`, which falls back to a full rebuild
rather than folding out of order. Nothing here assumes it.

TWO CLOCKS, NEVER CONFLATED (see `models.py`):
  * `event_timestamp` drives the FOLD order and every stored `first_seen`/`last_seen`.
    It may be in the future — a calendar meeting's `event_timestamp` is its START.
  * `observed_at` drives the RESUME cursor and `as_of`. It is `memory_records.created_at`,
    assigned by Postgres at insert, so it can never exceed wall clock.

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
from navis.state.v2.models import (
    FoldFrontier,
    OrganizationalState,
    Watermark,
    observation_key,
    order_key,
)


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
    # WHEN NAVIS LEARNED THIS FACT — `memory_records.created_at`, verbatim.
    #
    # Distinct from `event_timestamp` (when the fact HAPPENED) and load-bearing:
    # it is the only key the resume cursor may use, because it is the only one
    # that cannot run ahead of wall clock. `MemoryReader._view` always populates
    # it, so it is never None on any production path.
    #
    # Optional ONLY so that hand-built projections (pure `build_projection`
    # corpora in tests) stay constructible. A record without an observation time
    # does not advance the cursor at all — it is folded and counted, but the
    # projection reports no observation horizon, which fails CLOSED: a snapshot
    # with an unset cursor cannot be resumed and forces a rebuild.
    observed_at: datetime | None = None


def _add_unique(lst: list, value) -> None:
    if value is not None and value not in lst:
        lst.append(value)
        lst.sort(key=lambda x: (str(type(x)), str(x)))  # deterministic order


class StateBuilder:
    """Mutable working accumulator. Deterministic given the input fold order."""

    def __init__(self, org_id: uuid.UUID) -> None:
        self.org_id = org_id
        self.entities: dict[str, dict[str, dict]] = {}
        # Observation cursor (resume position) — max over `observed_at`.
        self.obs_ts: datetime | None = None
        self.obs_mid: uuid.UUID | None = None
        # Event frontier (ordering proof) — max over `event_timestamp`.
        self.evt_ts: datetime | None = None
        self.evt_mid: uuid.UUID | None = None
        # Completeness proof.
        self.record_count: int = 0

    @classmethod
    def from_state(cls, state: OrganizationalState) -> "StateBuilder":
        b = cls(state.org_id)
        # Copy entities so we never mutate the caller's projection.
        b.entities = {
            et: {eid: _deep_copy_raw(raw) for eid, raw in bucket.items()}
            for et, bucket in state.entities.items()
        }
        b.obs_ts = state.watermark.last_timestamp
        b.obs_mid = state.watermark.last_memory_id
        b.evt_ts = state.fold_frontier.last_event_timestamp
        b.evt_mid = state.fold_frontier.last_memory_id
        b.record_count = state.record_count
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

        # Both cursors advance by MAX, never by "last one applied".
        #
        # That distinction is the whole fix. Taking the last-applied record made
        # the cursor a function of fold order, so a meeting whose event_timestamp
        # is tomorrow — folded last because event order put it last — became the
        # organization's clock. A max over `observed_at` cannot do that: the
        # newest thing Navis has LEARNED is bounded by when it looked.
        #
        # Max is also order-independent, so a full rebuild and an incremental
        # fold over the same records produce the identical cursor whatever
        # sequence they arrive in.
        if rec.observed_at is not None and (
            self.obs_ts is None
            or self.obs_mid is None
            or observation_key(rec.observed_at, rec.memory_id)
            > observation_key(self.obs_ts, self.obs_mid)
        ):
            self.obs_ts = rec.observed_at
            self.obs_mid = rec.memory_id

        if (
            self.evt_ts is None
            or self.evt_mid is None
            or order_key(rec.event_timestamp, rec.memory_id)
            > order_key(self.evt_ts, self.evt_mid)
        ):
            self.evt_ts = rec.event_timestamp
            self.evt_mid = rec.memory_id

        self.record_count += 1

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
            # THE OBSERVATION HORIZON, not the newest event. An explicit `as_of`
            # still wins — that is the point-in-time query path — but the derived
            # default is now the moment of the newest fact Navis had LEARNED,
            # which is bounded by wall clock and can never be a future meeting.
            as_of=as_of if as_of is not None else self.obs_ts,
            watermark=Watermark(self.obs_ts, self.obs_mid),
            entities=self.entities,
            record_count=self.record_count,
            fold_frontier=FoldFrontier(self.evt_ts, self.evt_mid),
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
