"""Regression: SignalDraft structurally validates the fields that cross into Decision — confidence
and severity in [0,1] (they feed Decision's weight = confidence*severity) and a non-empty dedup_key
(anchors the signal_id). A future/buggy engine emitting an out-of-range value is rejected at draft
construction and ISOLATED by the runner (the pass continues) — never reaching Decision. Today's
engines all emit valid values, so real output is unchanged.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from navis.reasoning.v2 import service
from navis.reasoning.v2.base import ReasoningEngine
from navis.reasoning.v2.contract import SignalDraft, SignalType

_T = SignalType.PRIORITY_LOAD


def _fake_state():
    return SimpleNamespace(
        org_id=uuid.UUID("11111111-1111-1111-1111-111111111111"),
        as_of=datetime(2026, 1, 1, tzinfo=timezone.utc),
        watermark=SimpleNamespace(last_memory_id=uuid.uuid4()),
    )


def test_signal_draft_accepts_valid_ranges():
    SignalDraft(_T, None, None, "t", "w", 0.0, 0.0, "k")  # boundaries valid
    SignalDraft(_T, None, None, "t", "w", 1.0, 1.0, "k")


@pytest.mark.parametrize("bad", [-0.01, 1.01, 5.0, -1.0])
def test_signal_draft_rejects_out_of_range_confidence_and_severity(bad):
    with pytest.raises(ValueError, match="confidence"):
        SignalDraft(_T, None, None, "t", "w", bad, 0.5, "k")
    with pytest.raises(ValueError, match="severity"):
        SignalDraft(_T, None, None, "t", "w", 0.5, bad, "k")


def test_signal_draft_rejects_empty_dedup_key():
    with pytest.raises(ValueError, match="dedup_key"):
        SignalDraft(_T, None, None, "t", "w", 0.5, 0.5, "")


def test_runner_isolates_engine_that_emits_invalid_signal(monkeypatch):
    class _BadEngine(ReasoningEngine):
        name = "bad"
        version = "1.0.0"

        def analyze(self, state, ctx):
            return [SignalDraft(_T, None, None, "t", "w", 0.5, 5.0, "k")]  # severity=5 → raises

    monkeypatch.setattr(service, "ENGINES", [_BadEngine()])
    assert service.run_reasoning_v2(_fake_state()) == []  # bad engine isolated; pass continues
