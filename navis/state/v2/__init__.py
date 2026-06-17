"""State V2 — a deterministic projection over Memory V2 (`memory_records`).

State answers one question: "What is true right now?" It is a PROJECTION, never a
source of truth; it is disposable and fully rebuildable from Memory. It computes
ZERO intelligence (no risk/health/priority/score/overdue/...). Those belong to
Reasoning, which reads OrganizationalState.

Determinism law: the projection is a pure function of (memory_records, as_of).
No datetime.now(), no uuid4, no randomness, no mutable counters. Records are folded
in (event_timestamp, memory_id) order. Incremental == full rebuild, by construction.

Module map (to the requested layout):
    models.py     -> models/ + organizational_state/ + watermarks/
    folds.py      -> folds/
    builders.py   -> builders/ + rebuild/ + incremental/
    repository.py -> repository/
    snapshots.py  -> snapshots/
    service.py    -> service/   (StateService is the sole interface)
"""
from navis.state.v2.models import OrganizationalState, Watermark
from navis.state.v2.service import StateService

__all__ = ["OrganizationalState", "Watermark", "StateService"]
