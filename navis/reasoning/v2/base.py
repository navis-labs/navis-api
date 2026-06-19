"""Engine base + context. The state's `as_of` is the only clock.

`ReasoningContext` carries the projection's logical clock and provenance so engines
can compute time-deltas deterministically (as_of - last_seen) without ever touching
datetime.now(). A `prior` OrganizationalState (also State V2) may be supplied for
trend reasoning — still State-only, never Memory/Signals.
"""
from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime

from navis.reasoning.v2.contract import SignalDraft
from navis.state.v2.models import OrganizationalState


@dataclass(frozen=True, slots=True)
class ReasoningContext:
    org_id: uuid.UUID
    as_of: datetime | None
    watermark: str | None
    reasoning_version: int
    prior: OrganizationalState | None = None


def days_between(as_of: datetime | None, last_seen_iso: str | None) -> float | None:
    """Deterministic age in days using the projection clock (NOT now())."""
    if as_of is None or not last_seen_iso:
        return None
    try:
        last = datetime.fromisoformat(last_seen_iso)
    except (TypeError, ValueError):
        return None
    return (as_of - last).total_seconds() / 86400.0


class ReasoningEngine(ABC):
    name: str
    version: str

    @abstractmethod
    def analyze(self, state: OrganizationalState, ctx: ReasoningContext) -> list[SignalDraft]:
        """Pure: OrganizationalState -> SignalDraft[]. No I/O, no clock, no LLM."""
        raise NotImplementedError
