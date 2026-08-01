"""Narrative generation (Decision -> narrative; never narrative -> Decision).

`recommendation` is the one advisory generator: it may rank facts by urgency/impact
so the founder can decide what to do first, but it still never invents facts and
never fires actions.
"""
from navis.ml.generation.narratives import (
    Narrative,
    brief,
    draft,
    explanation,
    generate,
    recommendation,
    summary,
)
from navis.ml.generation.prompts import REGISTRY, Prompt, get_prompt

__all__ = [
    "Narrative",
    "generate",
    "brief",
    "summary",
    "draft",
    "explanation",
    "recommendation",
    "Prompt",
    "get_prompt",
    "REGISTRY",
]
