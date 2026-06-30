"""Regression tests for navis/context/metrics.py: namespacing, name/value validation,
high-cardinality label rejection, caller-mapping immutability, and best-effort emission.
"""
from __future__ import annotations

import pytest

from navis.context import metrics as ctx_metrics


class _Recorder:
    def __init__(self, *, boom: bool = False):
        self.incrs: list = []
        self.obs: list = []
        self._boom = boom

    def incr(self, name, value=1.0, labels=None):
        if self._boom:
            raise RuntimeError("sink down")
        self.incrs.append((name, value, labels))

    def observe_ms(self, name, ms, labels=None):
        if self._boom:
            raise RuntimeError("sink down")
        self.obs.append((name, ms, labels))


@pytest.fixture()
def rec(monkeypatch):
    r = _Recorder()
    monkeypatch.setattr(ctx_metrics, "metrics", r)
    return r


# --- Namespacing + happy path -----------------------------------------------

def test_incr_emits_prefixed_value_and_labels(rec):
    ctx_metrics.incr("cache_hit", labels={"task": "full_org"}, value=2.0)
    assert rec.incrs == [("context.cache_hit", 2.0, {"task": "full_org"})]


def test_observe_ms_emits_prefixed(rec):
    ctx_metrics.observe_ms("assembly_ms", 12.5, labels={"task": "x"})
    assert rec.obs == [("context.assembly_ms", 12.5, {"task": "x"})]


def test_empty_labels_become_none(rec):
    ctx_metrics.incr("assembled", labels={})
    assert rec.incrs == [("context.assembled", 1.0, None)]


# --- Name validation (Rule 5) -----------------------------------------------

@pytest.mark.parametrize("bad", ["", "   ", "Bad", "a-b", "a\nb", "a b", "1abc", "x" * 200, "café"])
def test_invalid_names_are_skipped(rec, bad):
    ctx_metrics.incr(bad)
    ctx_metrics.observe_ms(bad, 1.0)
    assert rec.incrs == [] and rec.obs == []


# --- Value validation (Rule 7) ----------------------------------------------

@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), True])
def test_non_finite_or_bool_value_is_skipped(rec, bad):
    ctx_metrics.incr("assembled", value=bad)
    assert rec.incrs == []


@pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf")])
def test_observe_ms_rejects_negative_and_nonfinite(rec, bad):
    ctx_metrics.observe_ms("assembly_ms", bad)
    assert rec.obs == []


def test_observe_ms_zero_is_valid(rec):
    ctx_metrics.observe_ms("assembly_ms", 0.0)
    assert rec.obs == [("context.assembly_ms", 0.0, None)]


# --- High-cardinality label protection (Rule 6) + immutability (Rule 4) -----

def test_forbidden_label_keys_are_dropped(rec):
    ctx_metrics.incr("assembled", labels={"user_id": "u1", "context_id": "c", "task": "t"})
    assert rec.incrs == [("context.assembled", 1.0, {"task": "t"})]


def test_all_labels_forbidden_becomes_none(rec):
    ctx_metrics.incr("assembled", labels={"email": "a@b.com"})
    assert rec.incrs == [("context.assembled", 1.0, None)]


def test_caller_label_mapping_is_not_mutated(rec):
    original = {"user_id": "u1", "task": "t"}
    ctx_metrics.incr("assembled", labels=original)
    assert original == {"user_id": "u1", "task": "t"}  # untouched


# --- Best-effort emission (Rule 10) -----------------------------------------

def test_sink_failure_is_swallowed(monkeypatch):
    monkeypatch.setattr(ctx_metrics, "metrics", _Recorder(boom=True))
    # Must not raise — a metrics outage can never break product correctness.
    ctx_metrics.incr("assembled")
    ctx_metrics.observe_ms("assembly_ms", 1.0)


# --- Stable namespace constant (Rule 9) -------------------------------------

def test_namespace_prefix_is_stable():
    assert ctx_metrics._PREFIX == "context."
