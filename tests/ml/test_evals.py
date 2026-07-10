"""Regression tests for navis/ml/evals — the production bugs fixed in the evals audit:

  * prompt_fingerprint is INJECTIVE (a `|` in a part can no longer alias two revisions to the
    same fingerprint → drift can't slip past regression detection).
  * recall_at_k guards non-positive k (a negative k no longer becomes a negative slice that
    silently inflates recall).
  * mean_recall_at_k is exported (one home for recall aggregation).
  * evals stay OFFLINE and provider-independent: the modules import only stdlib.
"""
from __future__ import annotations

import ast
import os

import navis.ml.evals as evals
from navis.ml.evals import mean_recall_at_k, prompt_fingerprint, recall_at_k


# --- prompt_fingerprint: injective (the real bug) ----------------------------

def test_prompt_fingerprint_is_injective_across_delimiters():
    # Under the old `|`-join these two DIFFERENT revisions collided ("a|1|b|c|d" both times),
    # so a real prompt edit went undetected. They must now differ.
    a = prompt_fingerprint("a", 1, "b|c", "d")
    b = prompt_fingerprint("a", 1, "b", "c|d")
    assert a != b


def test_prompt_fingerprint_deterministic_and_field_sensitive():
    base = prompt_fingerprint("brief", 1, "sys", "user")
    assert base == prompt_fingerprint("brief", 1, "sys", "user")   # deterministic
    assert base != prompt_fingerprint("brief", 2, "sys", "user")   # version
    assert base != prompt_fingerprint("brief", 1, "SYS", "user")   # system text
    assert base != prompt_fingerprint("brief", 1, "sys", "user2")  # rendered output
    assert base != prompt_fingerprint("draft", 1, "sys", "user")   # id
    assert len(base) == 16


# --- recall_at_k: correct statistic ------------------------------------------

def test_recall_at_k_valid_is_unchanged():
    assert recall_at_k(["a", "b", "c"], frozenset({"b", "z"}), k=3) == 0.5


def test_recall_at_k_guards_nonpositive_k():
    # Old behaviour: k=-1 -> retrieved[:-1] = ["a","b"] -> recall 1.0 (WRONG). Now 0.0.
    assert recall_at_k(["a", "b", "c"], frozenset({"a"}), k=-1) == 0.0
    assert recall_at_k(["a", "b", "c"], frozenset({"a"}), k=0) == 0.0
    assert recall_at_k([], frozenset(), k=5) == 1.0  # empty relevant set convention


def test_mean_recall_at_k_aggregates():
    cases = [(["a", "b"], frozenset({"a"})), (["x"], frozenset({"y"}))]
    assert mean_recall_at_k(cases, k=2) == 0.5   # (1.0 + 0.0) / 2
    assert mean_recall_at_k([], k=2) == 0.0
    assert "mean_recall_at_k" in evals.__all__  # exported: one home for aggregation


# --- invariant: evals are offline + provider-independent ---------------------

def _import_roots(path: str) -> set[str]:
    tree = ast.parse(open(path).read())
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def test_evals_import_only_stdlib_never_navis():
    # evals must never depend on providers/gateway/routing/inference — it operates on plain
    # data (strings, ids), so it can never influence production inference (Q7/Q8).
    d = os.path.dirname(evals.__file__)
    for fname in ("regression_eval.py", "generation_eval.py", "retrieval_eval.py"):
        roots = _import_roots(os.path.join(d, fname))
        assert roots <= {"__future__", "hashlib", "re", "dataclasses"}, (fname, roots)
