"""Regression tests for navis/context/telemetry.py — the Context layer's only logging
boundary. Proves it is best-effort (no logger/formatter/sink/serialization failure can fail
an assembly), that latency is always recorded, that the log schema is stable, that formatting
is deterministic, and that audit records carry no PII.
"""
from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone

import pytest

from navis.context import metrics as ctx_metrics
from navis.context import telemetry


# --- recorders --------------------------------------------------------------

class _RecLogger:
    """Captures the fully-rendered message (msg % args) — also exercises real formatting."""

    def __init__(self, *, boom: bool = False):
        self.lines: list[str] = []
        self._boom = boom

    def info(self, msg, *args):
        if self._boom:
            raise RuntimeError("logger sink down")
        self.lines.append(msg % args if args else msg)


class _RecSink:
    """Stand-in for the platform metrics SINK (the thing the façade calls)."""

    def __init__(self, *, boom: bool = False):
        self.incrs: list = []
        self.obs: list = []
        self._boom = boom

    def incr(self, name, value=1.0, labels=None):
        if self._boom:
            raise RuntimeError("metrics sink down")
        self.incrs.append((name, value, labels))

    def observe_ms(self, name, ms, labels=None):
        if self._boom:
            raise RuntimeError("metrics sink down")
        self.obs.append((name, ms, labels))


ORG = uuid.UUID("11111111-1111-1111-1111-111111111111")


def _audit(logger_rec, **over):
    kw = dict(
        org_id=ORG, actor_fingerprint="ab12cd", actor_kind="user", task="commitments",
        context_id="deadbeef", permission_fingerprint="perm99", policy_fingerprint="pol77",
        entity_count=5, bundle_tokens=128,
        state_as_of=datetime(2026, 6, 29, tzinfo=timezone.utc), latency_ms=1.5, cache_hit=False,
    )
    kw.update(over)
    telemetry.log_audit(**kw)


# --- Best-effort: a logger failure can never fail an assembly ----------------

def test_log_built_swallows_logger_failure(monkeypatch):
    monkeypatch.setattr(telemetry, "logger", _RecLogger(boom=True))
    # Must not raise — log_built runs on the non-isolated build path.
    telemetry.log_built(org_id=ORG, task="commitments", context_id="x",
                        admitted=1, dropped=0, cache_hit=True)


def test_log_audit_swallows_logger_failure(monkeypatch):
    sink = _RecSink()
    monkeypatch.setattr(ctx_metrics, "metrics", sink)
    monkeypatch.setattr(telemetry, "audit_logger", _RecLogger(boom=True))
    _audit(None)  # must not raise...
    assert sink.incrs == [("context.audit", 1.0, {"task": "commitments", "cache_hit": "false"})]
    # ...and the audit metric still fires even though the log line blew up.


def test_log_audit_swallows_isoformat_failure(monkeypatch):
    rec = _RecLogger()
    monkeypatch.setattr(telemetry, "audit_logger", rec)
    monkeypatch.setattr(ctx_metrics, "metrics", _RecSink())

    class _BadTs:
        def isoformat(self):
            raise ValueError("clock exploded")

    _audit(None, state_as_of=_BadTs())  # arg-eval path must not propagate
    assert "state_as_of= latency_ms=" in rec.lines[0]  # rendered as empty, did not crash


def test_log_audit_handles_none_state_as_of(monkeypatch):
    rec = _RecLogger()
    monkeypatch.setattr(telemetry, "audit_logger", rec)
    monkeypatch.setattr(ctx_metrics, "metrics", _RecSink())
    _audit(None, state_as_of=None)
    assert "state_as_of=" in rec.lines[0]  # renders "" rather than crashing


# --- Metrics sink failure can never fail an assembly ------------------------

def test_log_audit_swallows_metrics_sink_failure(monkeypatch):
    monkeypatch.setattr(telemetry, "audit_logger", _RecLogger())
    monkeypatch.setattr(ctx_metrics, "metrics", _RecSink(boom=True))
    _audit(None)  # façade swallows the sink error → no propagation


# --- timed(): always records, never masks, nesting-safe ---------------------

def test_timed_always_records_latency(monkeypatch):
    sink = _RecSink()
    monkeypatch.setattr(ctx_metrics, "metrics", sink)
    with telemetry.timed("assembly_ms", labels={"task": "t"}):
        pass
    assert len(sink.obs) == 1
    name, ms, labels = sink.obs[0]
    assert name == "context.assembly_ms" and ms >= 0.0 and labels == {"task": "t"}


def test_timed_records_latency_even_when_body_raises(monkeypatch):
    sink = _RecSink()
    monkeypatch.setattr(ctx_metrics, "metrics", sink)
    with pytest.raises(ValueError):  # the block's own error is NOT swallowed
        with telemetry.timed("assembly_ms"):
            raise ValueError("boom")
    assert len(sink.obs) == 1  # ...but latency was still recorded in `finally`


def test_timed_metrics_sink_failure_never_masks_body(monkeypatch):
    monkeypatch.setattr(ctx_metrics, "metrics", _RecSink(boom=True))
    # A failing metrics sink in `finally` must not raise / mask anything.
    with telemetry.timed("assembly_ms"):
        pass


def test_timed_nesting_is_independent(monkeypatch):
    sink = _RecSink()
    monkeypatch.setattr(ctx_metrics, "metrics", sink)
    with telemetry.timed("outer"):
        with telemetry.timed("inner"):
            pass
    assert {n for n, _, _ in sink.obs} == {"context.outer", "context.inner"}


# --- Schema stability + deterministic formatting ----------------------------

def _keys(line: str) -> list[str]:
    return re.findall(r"(\w+)=", line)


def test_built_log_schema_is_stable(monkeypatch):
    rec = _RecLogger()
    monkeypatch.setattr(telemetry, "logger", rec)
    telemetry.log_built(org_id=ORG, task="commitments", context_id="x",
                        admitted=2, dropped=1, cache_hit=False)
    assert _keys(rec.lines[0]) == ["org", "task", "context_id", "admitted", "dropped", "cache_hit"]


def test_audit_log_schema_is_stable(monkeypatch):
    rec = _RecLogger()
    monkeypatch.setattr(telemetry, "audit_logger", rec)
    monkeypatch.setattr(ctx_metrics, "metrics", _RecSink())
    _audit(rec)
    assert _keys(rec.lines[0]) == [
        "org", "actor_fp", "actor_kind", "task", "context_id", "perm_fp", "policy_fp",
        "entities", "tokens", "state_as_of", "latency_ms", "cache_hit",
    ]


def test_audit_formatting_is_deterministic(monkeypatch):
    rec = _RecLogger()
    monkeypatch.setattr(telemetry, "audit_logger", rec)
    monkeypatch.setattr(ctx_metrics, "metrics", _RecSink())
    _audit(rec)
    _audit(rec)
    assert rec.lines[0] == rec.lines[1]  # same inputs -> byte-identical record


# --- PII review -------------------------------------------------------------

def test_audit_record_carries_no_pii(monkeypatch):
    rec = _RecLogger()
    monkeypatch.setattr(telemetry, "audit_logger", rec)
    monkeypatch.setattr(ctx_metrics, "metrics", _RecSink())
    # Even if an email/name somehow reached a fingerprint slot, the schema only renders the
    # fields we pass — assert no PII-shaped value and no PII-named key leaks.
    _audit(rec)
    line = rec.lines[0]
    assert "@" not in line  # no email
    forbidden_keys = {"email", "name", "content", "payload", "prompt", "document", "body"}
    assert not (forbidden_keys & set(_keys(line)))
