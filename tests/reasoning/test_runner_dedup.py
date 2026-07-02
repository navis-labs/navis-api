"""Regression: the runner's output is de-duplicated by the content-addressed signal_id (unique
ids), keep-first, deterministically ordered. This is the invariant every future signal source
(an additive AI runner) must meet. Today's engines emit unique ids, so real output is unchanged;
a duplicate is injected via a fake engine to prove the collapse.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from types import SimpleNamespace

from navis.reasoning.v2 import service
from navis.reasoning.v2.base import ReasoningContext, ReasoningEngine
from navis.reasoning.v2.contract import SignalDraft, SignalType


def _fake_state():
    return SimpleNamespace(
        org_id=uuid.UUID("11111111-1111-1111-1111-111111111111"),
        as_of=datetime(2026, 1, 1, tzinfo=timezone.utc),
        watermark=SimpleNamespace(last_memory_id=uuid.uuid4()),
    )


class _DupEngine(ReasoningEngine):
    name = "dup_test"
    version = "1.0.0"

    def analyze(self, state, ctx: ReasoningContext) -> list[SignalDraft]:
        draft = SignalDraft(
            signal_type=SignalType.PRIORITY_LOAD, entity_type=None, entity_id=None,
            title="t", why="w", confidence=0.5, severity=0.5, dedup_key="k",
        )
        return [draft, draft]  # same identity twice → same signal_id


def test_runner_collapses_duplicate_signal_ids(monkeypatch):
    monkeypatch.setattr(service, "ENGINES", [_DupEngine()])
    signals = service.run_reasoning_v2(_fake_state())  # type: ignore[arg-type]
    ids = [s.signal_id for s in signals]
    assert len(ids) == 1 and len(set(ids)) == 1  # two identical drafts collapsed to one signal


def test_runner_output_has_unique_ids_and_is_ordered(monkeypatch):
    # Two distinct drafts (different severity) → both kept, ordered by severity desc.
    class _TwoEngine(ReasoningEngine):
        name = "two_test"
        version = "1.0.0"

        def analyze(self, state, ctx):
            return [
                SignalDraft(SignalType.PRIORITY_LOAD, None, None, "low", "w", 0.5, 0.3, "a"),
                SignalDraft(SignalType.OPPORTUNITY, None, None, "high", "w", 0.5, 0.9, "b"),
            ]

    monkeypatch.setattr(service, "ENGINES", [_TwoEngine()])
    signals = service.run_reasoning_v2(_fake_state())  # type: ignore[arg-type]
    ids = [s.signal_id for s in signals]
    assert len(ids) == len(set(ids)) == 2                 # unique ids
    assert [s.severity for s in signals] == [0.9, 0.3]    # severity desc ordering
