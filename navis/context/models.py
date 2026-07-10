"""Internal value objects for the Context layer.

Pure, frozen, serializable — no behaviour beyond deterministic projection of their own
fields. These describe WHAT was assembled (scope, evidence, budget), never WHY in any
interpretive sense (no scores, no risk — that is Reasoning's job).

Every object validates and CANONICALIZES itself at construction, so a ContextScope is
always deterministically ordered and duplicate-free regardless of how it was built, and a
BudgetReport can never hold a nonsensical count. Validation is pure (no clock/IO), so
replay determinism is preserved.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any


class SelectionReason(str, enum.Enum):
    """Deterministic, structural reason an entity entered the scope. Auditable, never
    interpretive (no 'important'/'risky' — those are Reasoning concerns)."""

    TASK_RELEVANT = "task_relevant"      # the task's spec selects this entity type
    LINK_EXPANSION = "link_expansion"    # reached by one-hop traversal of State links


_SELECTION_REASONS = frozenset(r.value for r in SelectionReason)


@dataclass(frozen=True, slots=True)
class ScopeEntry:
    entity_id: str
    entity_type: str
    reason: str  # a SelectionReason VALUE (kept as str to preserve the serialized contract)

    def __post_init__(self) -> None:
        if not isinstance(self.entity_id, str) or not self.entity_id:
            raise ValueError("ScopeEntry.entity_id must be a non-empty string")
        if not isinstance(self.entity_type, str) or not self.entity_type:
            raise ValueError("ScopeEntry.entity_type must be a non-empty string")
        # Strong validation without changing the field's str contract: the value MUST be a
        # known SelectionReason. (Storing the enum directly would force selectors/service —
        # out of scope here — to pass enums; validating the value gives the same safety
        # while keeping the str serialization byte-identical.)
        if self.reason not in _SELECTION_REASONS:
            raise ValueError(f"ScopeEntry.reason must be a SelectionReason value, got {self.reason!r}")

    def to_dict(self) -> dict[str, str]:
        return {"entity_id": self.entity_id, "entity_type": self.entity_type, "reason": self.reason}

    @classmethod
    def from_dict(cls, d: dict[str, str]) -> "ScopeEntry":
        return cls(entity_id=d["entity_id"], entity_type=d["entity_type"], reason=d["reason"])


@dataclass(frozen=True, slots=True)
class ContextScope:
    """The deterministic selection result: which entity types were considered and which
    entities (with their reason) survived selection + permission filtering + budgeting.

    Canonicalized at construction: entity_types are sorted+deduped, entries are sorted by
    entity_id, and duplicate entity_ids are rejected — so the scope (and therefore the
    serialized bundle) is byte-identical across replays no matter the construction order."""

    entity_types: tuple[str, ...]      # canonical: sorted, unique
    entries: tuple[ScopeEntry, ...]    # canonical: sorted by entity_id, unique entity_id

    def __post_init__(self) -> None:
        ids = [e.entity_id for e in self.entries]
        if len(set(ids)) != len(ids):
            raise ValueError("ContextScope contains duplicate entity_ids")
        # Canonicalize for determinism (idempotent for already-sorted input). object.__setattr__
        # is the supported way to normalize fields of a frozen dataclass in __post_init__.
        object.__setattr__(self, "entity_types", tuple(sorted(set(self.entity_types))))
        object.__setattr__(self, "entries", tuple(sorted(self.entries, key=lambda e: e.entity_id)))

    def entity_ids(self) -> tuple[str, ...]:
        return tuple(e.entity_id for e in self.entries)

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_types": list(self.entity_types),
            "entries": [e.to_dict() for e in self.entries],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ContextScope":
        return cls(
            entity_types=tuple(d.get("entity_types", [])),
            entries=tuple(ScopeEntry.from_dict(e) for e in d.get("entries", [])),
        )


@dataclass(frozen=True, slots=True)
class BudgetReport:
    """Pure accounting of the budgeting pass — fully explainable, no judgement."""

    token_estimate: int
    admitted: int
    dropped: int
    max_tokens: int | None

    def __post_init__(self) -> None:
        for name in ("token_estimate", "admitted", "dropped"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"BudgetReport.{name} must be a non-negative int, got {value!r}")
        if self.max_tokens is not None:
            if not isinstance(self.max_tokens, int) or isinstance(self.max_tokens, bool) or self.max_tokens <= 0:
                raise ValueError(f"BudgetReport.max_tokens must be > 0 or None, got {self.max_tokens!r}")
        if self.max_tokens is not None and self.token_estimate > self.max_tokens:
            raise ValueError("BudgetReport.token_estimate must not exceed max_tokens")

    def to_dict(self) -> dict[str, Any]:
        return {
            "token_estimate": self.token_estimate,
            "admitted": self.admitted,
            "dropped": self.dropped,
            "max_tokens": self.max_tokens,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "BudgetReport":
        return cls(
            token_estimate=int(d["token_estimate"]),
            admitted=int(d["admitted"]),
            dropped=int(d["dropped"]),
            max_tokens=d.get("max_tokens"),
        )
