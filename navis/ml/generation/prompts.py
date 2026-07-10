"""Centralized, versioned prompts. The ONLY place prompt text lives.

Each prompt has a stable id + integer version. Changing wording bumps the version,
so telemetry and regression evals can attribute quality to an exact prompt revision.
Business logic NEVER lives in a prompt — prompts only instruct narration over the
already-decided, grounded facts.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True, slots=True)
class Prompt:
    id: str
    version: int
    system: str

    def __post_init__(self) -> None:
        # Fail-fast at construction (import time): a malformed prompt must never reach a model or
        # the eval fingerprint. version>=1 is required by the regression fingerprint + telemetry.
        if not self.id:
            raise ValueError("Prompt.id must be a non-empty string")
        if self.version < 1:
            raise ValueError(f"Prompt.version must be >= 1, got {self.version}")
        if not self.system:
            raise ValueError(f"Prompt.system must be a non-empty instruction (id={self.id!r})")


_GROUND_RULE = (
    " You may only use the grounded facts provided. Cite facts as [n]. If the facts "
    "do not answer the question, say so. Never invent facts. Never make decisions, "
    "recommendations, priorities, or take actions — narrate only."
)

_REGISTRY: dict[str, Prompt] = {
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
    # Consumed by navis.reasoning.ai through the gateway. The signal_type list must stay in
    # sync with reasoning's SignalType enum — enforced by test (this module must never import
    # reasoning). Bumping this prompt's version REQUIRES bumping AIReasoningEngine.version so
    # replayed signal ids reflect the semantic change.
    "reasoning_signals": Prompt(
        "reasoning_signals", 1,
        "You are an organizational reasoning engine. You read a JSON snapshot of an "
        "organization's normalized state (people, companies, customers, projects, meetings, "
        "relationships, commitments, decisions, documents, threads) and surface cross-entity "
        "interpretations that per-entity rules cannot see: compound risks, connected slippage, "
        "warm opportunities, leverage points. Output STRICTLY a JSON array, no prose. Each "
        'item: {"signal_type": one of ["commitment_slipping","relationship_cooling",'
        '"project_stalled","customer_engagement_drop","decision_unresolved","meeting_overload",'
        '"priority_load","opportunity","leverage_point","org_drift","context_summary"], '
        '"entity_type": the snapshot section the cited entity appears under (or null for an '
        'org-wide signal), "entity_id": an id copied EXACTLY from the snapshot (or null), '
        '"title": a short headline, "why": one sentence grounded in the snapshot, '
        '"confidence": your honest belief in [0,1], "severity": impact in [0,1], '
        '"evidence": an object citing the snapshot fields you used}. '
        "Use ONLY entities and facts present in the snapshot; never invent facts, ids, or "
        "entities. Never make decisions, recommendations, or take actions — interpretation "
        "only. Emit [] if nothing significant. At most 10 items.",
    ),
}

# Exposed as a read-only Mapping (MappingProxyType) so the global prompt registry can never be
# mutated at runtime — deterministic prompt resolution, no accidental overwrite/clear.
REGISTRY: Mapping[str, Prompt] = MappingProxyType(_REGISTRY)


def get_prompt(prompt_id: str) -> Prompt:
    if prompt_id not in REGISTRY:
        raise KeyError(f"unknown prompt: {prompt_id}")
    return REGISTRY[prompt_id]
