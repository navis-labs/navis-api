"""State V2 — a deterministic projection over Memory V2 (`memory_records`).

State answers one question: "What is true right now?" It is a PROJECTION, never a
source of truth; it is disposable and fully rebuildable from Memory. It computes
ZERO intelligence (no risk/health/priority/score/overdue/...). Those belong to
Reasoning, which reads OrganizationalState.

Determinism law: the projection is a pure function of (memory_records, as_of).
No datetime.now(), no uuid4, no randomness, no mutable counters. Records are folded
in (event_timestamp, memory_id) order — when the fact HAPPENED.

Resume is a different question and uses a different clock. The cursor advances in
(created_at, memory_id) order — when Navis LEARNED the fact — because event time is
not observation-monotone: a calendar meeting's `event_timestamp` is its START, so an
event-time cursor can point past wall clock and strand everything observed after it.
Incremental == full rebuild still holds, but now it is CHECKED rather than assumed:
State proves completeness against a record count and ordering against a fold frontier,
and rebuilds whenever either proof fails. See `models.py` and `service.py`.

Module map (to the requested layout):
    models.py     -> models/ + organizational_state/ + watermarks/
    folds.py      -> folds/
    builders.py   -> builders/ + rebuild/ + incremental/
    repository.py -> repository/
    snapshots.py  -> snapshots/
    service.py    -> service/   (StateService is the sole interface)
"""
from navis.state.v2.models import OrganizationalState, Watermark
from navis.state.v2.scope import (
    Citation,
    EntityLink,
    EntityRef,
    Intent,
    MorningBrief,
    ScopedView,
    scope_for,
)
from navis.state.v2.service import StateService

__all__ = [
    "OrganizationalState",
    "Watermark",
    "StateService",
    # Scoping utility — read-only over State (see scope.py header).
    "Intent",
    "MorningBrief",
    "ScopedView",
    "EntityRef",
    "EntityLink",
    "Citation",
    "scope_for",
]
