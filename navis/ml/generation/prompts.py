"""Centralized, versioned prompts. The ONLY place prompt text lives.

Each prompt has a stable id + integer version. Changing wording bumps the version,
so telemetry and regression evals can attribute quality to an exact prompt revision.
Business logic NEVER lives in a prompt — prompts only instruct narration over the
already-decided, grounded facts.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Prompt:
    id: str
    version: int
    system: str


_GROUND_RULE = (
    " You may only use the grounded facts provided. Cite facts as [n]. If the facts "
    "do not answer the question, say so. Never invent facts. Never make decisions, "
    "recommendations, priorities, or take actions — narrate only."
)

REGISTRY: dict[str, Prompt] = {
    "brief_narrative": Prompt(
        "brief_narrative", 1,
        "You are a concise executive writer. Summarize what matters from the grounded "
        "facts into a short morning narrative for a founder." + _GROUND_RULE,
    ),
    "summary": Prompt(
        "summary", 1,
        "You are a precise summarizer. Produce a faithful summary of the grounded facts."
        + _GROUND_RULE,
    ),
    "draft": Prompt(
        "draft", 1,
        "You are a writing assistant drafting a message grounded ONLY in the facts."
        + _GROUND_RULE,
    ),
    "explanation": Prompt(
        "explanation", 1,
        "You explain WHY something is true using only the grounded facts and their "
        "citations." + _GROUND_RULE,
    ),
}


def get_prompt(prompt_id: str) -> Prompt:
    if prompt_id not in REGISTRY:
        raise KeyError(f"unknown prompt: {prompt_id}")
    return REGISTRY[prompt_id]
