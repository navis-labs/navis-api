"""Regression: the slipping/cooling/stalled predicate has ONE home (risk.py) reused by both the
entity engines and the PriorityEngine meta-counts, so a count can never drift from the signals
the entity engines emit. Deterministic (projection clock; not-open / missing timestamp = no hit).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from navis.reasoning.v2.engines import meta, risk
from navis.reasoning.v2.engines.risk import (
    COMMITMENT_SLIPPING_DAYS,
    PROJECT_STALLED_DAYS,
    RELATIONSHIP_COOLING_DAYS,
    cooling_age,
    slipping_age,
    stalled_age,
)

_AS_OF = datetime(2026, 1, 20, tzinfo=timezone.utc)


def _iso(days_ago: int) -> str:
    return (_AS_OF - timedelta(days=days_ago)).isoformat()


def test_slipping_predicate():
    assert slipping_age(SimpleNamespace(status="open", last_seen=_iso(19)), _AS_OF) >= COMMITMENT_SLIPPING_DAYS
    assert slipping_age(SimpleNamespace(status="closed", last_seen=_iso(19)), _AS_OF) is None  # not open
    assert slipping_age(SimpleNamespace(status="open", last_seen=_iso(1)), _AS_OF) is None      # under threshold
    assert slipping_age(SimpleNamespace(status="open", last_seen=None), _AS_OF) is None          # no timestamp


def test_cooling_predicate():
    assert cooling_age(SimpleNamespace(last_interaction_at=_iso(19)), _AS_OF) >= RELATIONSHIP_COOLING_DAYS
    assert cooling_age(SimpleNamespace(last_interaction_at=_iso(1)), _AS_OF) is None
    assert cooling_age(SimpleNamespace(last_interaction_at=None), _AS_OF) is None


def test_stalled_predicate():
    assert stalled_age(SimpleNamespace(last_seen=_iso(19)), _AS_OF) >= PROJECT_STALLED_DAYS
    assert stalled_age(SimpleNamespace(last_seen=_iso(1)), _AS_OF) is None
    assert stalled_age(SimpleNamespace(last_seen=None), _AS_OF) is None


def test_predicate_is_single_home():
    # meta reuses risk's predicate — the SAME function object, so PriorityEngine counts can never
    # disagree with the signals the entity engines emit.
    assert meta.slipping_age is risk.slipping_age
    assert meta.cooling_age is risk.cooling_age
    assert meta.stalled_age is risk.stalled_age
