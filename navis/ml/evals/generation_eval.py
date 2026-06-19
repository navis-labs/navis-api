"""Generation grounding fidelity: does the output rely only on provided facts?

Deterministic heuristic — measures how much of the output's vocabulary is supported
by the grounded context (plus stopwords). A high score means low fabrication risk.
This is a guardrail metric, not a human-quality score.
"""
from __future__ import annotations

import re

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    "the a an and or but of to in on for with is are was were be this that it as at by from".split()
)


def grounding_score(output: str, context_text: str) -> float:
    out = [t for t in _TOKEN.findall(output.lower()) if t not in _STOP]
    if not out:
        return 1.0
    ctx = set(_TOKEN.findall(context_text.lower()))
    supported = sum(1 for t in out if t in ctx)
    return round(supported / len(out), 4)


def is_grounded(output: str, context_text: str, *, threshold: float = 0.6) -> bool:
    return grounding_score(output, context_text) >= threshold
