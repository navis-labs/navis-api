"""Provider-independent model routing.

Tasks declare a TIER (fast vs deep); the router maps tier → a concrete provider,
choosing the cheapest model that fits. Because everything goes through MLProvider,
swapping the underlying vendor changes nothing for callers.

  fast  → GPT-4o-mini (Azure)  /  Claude Haiku   — summaries, drafts
  deep  → Claude Sonnet/Opus   /  GPT-4o          — briefs, explanations
"""
from __future__ import annotations

import enum

from navis.ml.providers.base import MLProvider
from navis.ml.providers.registry import get_provider


class Tier(str, enum.Enum):
    FAST = "fast"
    DEEP = "deep"


class Task(str, enum.Enum):
    SUMMARY = "summary"
    DRAFT = "draft"
    BRIEF_NARRATIVE = "brief_narrative"
    EXPLANATION = "explanation"
    RECOMMENDATION = "recommendation"


TASK_TIER: dict[Task, Tier] = {
    Task.SUMMARY: Tier.FAST,
    Task.DRAFT: Tier.FAST,
    Task.BRIEF_NARRATIVE: Tier.DEEP,
    Task.EXPLANATION: Tier.DEEP,
    # Prioritization weighs urgency vs impact across the whole context — deep tier.
    Task.RECOMMENDATION: Tier.DEEP,
}


def provider_for(task: Task) -> MLProvider:
    """Resolve the provider for a task's tier (deterministic default when offline)."""
    # The configured provider already encodes the vendor; tier influences the model
    # the generation layer requests. We return the active provider; deep/fast model
    # choice is expressed via the provider's own fast/deep models where available.
    return get_provider()


def tier_for(task: Task) -> Tier:
    return TASK_TIER[task]
