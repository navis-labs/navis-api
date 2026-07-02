"""GroundedContext — the ONLY thing an LLM is ever allowed to see.

The LLM never touches the database. It receives a GroundedContext: a bounded, cited
bundle of facts drawn from the truth layers (Memory/State/Learning/Intelligence).
Every fact carries a source so generated narrative is auditable back to a real row.
Truth is never authored by the model — only narrated.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Grounding:
    """One cited fact. `source` is the lineage pointer (e.g. a memory_id)."""

    kind: str            # memory | state | learning | intelligence
    source: str          # id / key the fact came from
    text: str


@dataclass(frozen=True, slots=True)
class GroundedContext:
    org_id: str
    question: str
    facts: tuple[Grounding, ...] = ()
    token_estimate: int = 0

    def __post_init__(self) -> None:
        # Immutability guarantee: GroundedContext is the ONLY contract this package exports, and
        # the model treats its facts as ground truth. A `list` would let a caller mutate evidence
        # after construction (ctx.facts.append(...)) — injecting an un-cited "fact" the model
        # would then narrate as grounded. Coerce to a tuple so `frozen=True` actually holds for
        # the evidence. Any iterable (incl. existing list callers) is accepted — backward-compatible.
        if not isinstance(self.facts, tuple):
            object.__setattr__(self, "facts", tuple(self.facts))

    def render(self) -> str:
        """The exact text handed to the model — facts only, each cited."""
        lines = [f"Question: {self.question}", "", "Grounded facts (cite by [n]):"]
        for i, f in enumerate(self.facts, 1):
            lines.append(f"[{i}] ({f.kind}:{f.source}) {f.text}")
        return "\n".join(lines)

    def sources(self) -> list[str]:
        return [f.source for f in self.facts]
