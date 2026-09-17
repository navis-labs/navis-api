"""GroundedContext — the ONLY thing an LLM is ever allowed to see.

The LLM never touches the database. It receives a GroundedContext: a bounded, cited
bundle of facts drawn from the truth layers (Memory/State/Learning/Intelligence).
Every fact carries a source so generated narrative is auditable back to a real row.
Truth is never authored by the model — only narrated.

Each fact may carry the `occurred_at` timestamp of the underlying event. Rendering
prefixes the fact with an ISO date so the LLM can answer time-scoped questions
("how many meetings this week") from the facts alone, without a separate tool call.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True, slots=True)
class Grounding:
    """One cited fact. `source` is the lineage pointer (e.g. a memory_id)."""

    kind: str            # memory | state | reasoning | learning | intelligence
    source: str          # id / key the fact came from
    text: str
    occurred_at: datetime | None = None
    # Deeper provenance, when the fact was DERIVED rather than retrieved. A State
    # entity is a fold over many memory records, so its single `source` (the
    # entity_id) cannot express what it was built from — this carries the drill-down
    # without inventing a second provenance system.
    #
    # Prefixed, self-describing ids ("memory:<uuid>", "signal:<uuid>"), matching the
    # convention `api/workspace_fallback.py` already uses for its `source` values.
    # Optional and last, so every existing construction site — including the
    # positional ones in tests/ml — is unaffected. Never rendered into the prompt;
    # `render()` emits `source` only, so this adds no tokens.
    lineage: tuple[str, ...] = ()

    def render_line(self, index: int) -> str:
        """This fact EXACTLY as it appears in the prompt.

        The single definition of a rendered fact. `GroundedContext.render` emits
        it and `ContextBuilder`'s budget loop costs it, so what is counted and
        what is sent are the same string — previously the budget counted `text`
        alone while the prompt also carried the index, date and `(kind:source)`
        citation, undercounting every fact by roughly a third.
        """
        when = f" [{self.occurred_at.date().isoformat()}]" if self.occurred_at else ""
        return f"[{index}]{when} ({self.kind}:{self.source}) {self.text}"


@dataclass(frozen=True, slots=True)
class GroundedContext:
    org_id: str
    question: str
    facts: list[Grounding] = field(default_factory=list)
    token_estimate: int = 0
    as_of: datetime | None = None

    def render(self) -> str:
        """The exact text handed to the model — facts only, each cited.

        When a fact carries `occurred_at`, we emit its UTC ISO date right after
        the citation index so the model can reason about "this week" / "today"
        directly from the fact list.
        """
        lines = [f"Question: {self.question}"]
        # Anchor date: without this, the model has no reference for "today" /
        # "tomorrow" / "this week" and will guess from fact dates or training data.
        if self.as_of is not None:
            lines.append(f"Today's date is: {self.as_of.date().isoformat()} (UTC)")
        lines += ["", "Grounded facts (cite by [n]):"]
        lines += [f.render_line(i) for i, f in enumerate(self.facts, 1)]
        return "\n".join(lines)

    def sources(self) -> list[str]:
        return [f.source for f in self.facts]
