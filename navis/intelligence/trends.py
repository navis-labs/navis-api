"""Trend Engine — the temporal dimension Learning's snapshot aggregates lack.

Deterministic per-subject (signal / decision) outcome trend: compare the success rate
of the RECENT window of learning_events vs the PRIOR window, to surface what is
improving or degrading. This is genuinely NEW information — learning_aggregates are
CUMULATIVE counts with no time dimension, so a snapshot projection cannot see change.

Pure + deterministic: logical time only (the caller supplies events ordered by
occurred_at; this module never reads a clock), a fixed even split, decisive-only rates
(neutrals excluded, exactly like Learning's success_rate), and a total sorted order.

Scope (deliberate): subjects are the reasoning signals and decisions Learning encodes.
Trends in people / relationship / customer / project HEALTH are NOT built here — that
data lives in State/Memory, which Intelligence must never read (the Learning-only
boundary). A degrading-relationship trend belongs to a State-reading layer.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

MIN_SAMPLES = 5          # decisive samples required in BOTH windows to state a trend
_THRESHOLD = 0.1         # min success-rate delta to call a direction (else "stable")


@dataclass(frozen=True, slots=True)
class Trend:
    subject_kind: str    # "signal" | "decision"
    subject: str
    recent_rate: float
    prior_rate: float
    direction: str       # "improving" | "degrading" | "stable"
    recent_samples: int
    prior_samples: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "subject_kind": self.subject_kind,
            "subject": self.subject,
            "recent_rate": self.recent_rate,
            "prior_rate": self.prior_rate,
            "direction": self.direction,
            "recent_samples": self.recent_samples,
            "prior_samples": self.prior_samples,
        }


def _bump(acc: dict, key: tuple[str, str], window: str, positive: int) -> None:
    slot = acc.setdefault(key, {"recent": [0, 0], "prior": [0, 0]})
    slot[window][0] += positive
    slot[window][1] += 1


def build_trends(recent_first_events: Iterable[Any], *, min_samples: int = MIN_SAMPLES) -> list[Trend]:
    """Recent-first learning_events -> per-subject success-rate trends.

    Events are ordered newest-first (occurred_at DESC) by the caller; the newer half is
    the RECENT window, the older half the PRIOR window. Each event exposes
    `.reasoning_signal_ids`, `.decision_id`, `.result`. Only decisive (success/failure)
    events count toward a rate; neutrals are ignored. Deterministic + replay-safe."""
    events = list(recent_first_events)
    mid = len(events) // 2
    acc: dict[tuple[str, str], dict[str, list[int]]] = {}
    for window, chunk in (("recent", events[:mid]), ("prior", events[mid:])):
        for e in chunk:
            if e.result not in ("success", "failure"):
                continue  # neutral (e.g. action_executed) never counts toward a success rate
            positive = 1 if e.result == "success" else 0
            for sid in (e.reasoning_signal_ids or []):
                _bump(acc, ("signal", str(sid)), window, positive)
            if e.decision_id is not None:
                _bump(acc, ("decision", str(e.decision_id)), window, positive)

    trends: list[Trend] = []
    for (kind, subject), w in sorted(acc.items()):
        rt, rn = w["recent"]
        pt, pn = w["prior"]
        if rn < min_samples or pn < min_samples:
            continue  # need enough decisive evidence in BOTH windows to state a trend
        recent_rate = round(rt / rn, 4)
        prior_rate = round(pt / pn, 4)
        delta = recent_rate - prior_rate
        direction = (
            "improving" if delta > _THRESHOLD
            else "degrading" if delta < -_THRESHOLD
            else "stable"
        )
        trends.append(Trend(kind, subject, recent_rate, prior_rate, direction, rn, pn))
    return trends
