"""Feature Engine — reusable organizational features derived from Learning.

Converts a LearningView (the once-computed Learning read model) into named,
deterministic org-level features, so the insight generators — and the future
deterministic engines (Phases 3-4) — READ a feature instead of re-deriving it from
raw aggregates. Pure + deterministic: same view => same features. No clock, no uuid,
no ML, and NO Learning re-scan (the view already holds the counts).

Scope (deliberate): features here are ONLY those derivable from Learning
(outcome/decision/signal aggregates + patterns). Features that need meeting /
relationship / customer / project / team data — meeting_frequency, response_latency,
relationship_strength, customer_health, team_velocity, project_health,
communication_health — are NOT built here: that data lives in State/Memory, which
Intelligence must never read (the Learning-only boundary, enforced by
test_no_forbidden_pipeline_imports). Emitting empty slots for them would be
speculative and dishonest; they belong to a layer that can see that data.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # annotation only — avoids a runtime import cycle with insights.py
    from navis.intelligence.insights import LearningView

# Outcome-type vocabulary — mirrors Learning's attribution result mapping.
_POSITIVE = frozenset({"reply_received", "meeting_accepted", "meeting_completed"})
_NEGATIVE = frozenset({"action_failed", "no_response"})
_ATTEMPT = "action_executed"


@dataclass(frozen=True, slots=True)
class OrgFeatures:
    """Named, reusable org-level features — the output of the Feature Engine.

    Raw counts are the primitives; the rates are derived properties (None until there
    is decisive evidence, so a consumer can distinguish "0%" from "not measured yet")."""

    positive_outcomes: int      # replies + meetings accepted/completed
    negative_outcomes: int      # action_failed + no_response
    executed_attempts: int      # action_executed
    reliable_patterns: int      # learning_patterns with kind == "reliable"
    total_patterns: int

    @property
    def decisive_outcomes(self) -> int:
        return self.positive_outcomes + self.negative_outcomes

    @property
    def engagement_rate(self) -> float | None:
        """Positive outcomes per executed action (capped at 1.0). None if no attempts."""
        if not self.executed_attempts:
            return None
        return round(min(1.0, self.positive_outcomes / self.executed_attempts), 4)

    @property
    def outcome_positive_rate(self) -> float | None:
        """Positive / decisive outcomes. None until there are decisive outcomes."""
        d = self.decisive_outcomes
        return round(self.positive_outcomes / d, 4) if d else None

    @property
    def signal_reliability_rate(self) -> float | None:
        """Reliable / total recurring signal patterns. None if no patterns."""
        return round(self.reliable_patterns / self.total_patterns, 4) if self.total_patterns else None


def compute_features(view: "LearningView") -> OrgFeatures:
    """LearningView -> named org features. Pure + deterministic; in-memory over the
    view's already-loaded aggregates/patterns (never re-scans Learning)."""
    positive = sum(a.total for a in view.outcome_aggs if a.key in _POSITIVE)
    negative = sum(a.total for a in view.outcome_aggs if a.key in _NEGATIVE)
    attempts = sum(a.total for a in view.outcome_aggs if a.key == _ATTEMPT)
    reliable = sum(1 for p in view.patterns if p.kind == "reliable")
    return OrgFeatures(
        positive_outcomes=positive,
        negative_outcomes=negative,
        executed_attempts=attempts,
        reliable_patterns=reliable,
        total_patterns=len(view.patterns),
    )
