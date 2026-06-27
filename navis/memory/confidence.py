"""Confidence contract for Memory V2.

`MemoryRecord.confidence` = P(the fact is true *as extracted*) — a FORMATION-time property
assigned by the extractor, on a standardized scale (no more magic numbers). It is NOT part of
identity (`memory_id`/`identity_tuple` exclude it), so standardizing values never changes
record identity or breaks replay determinism.

Ownership:
  * Extractors ASSIGN confidence at formation, from the levels below.
  * State/Reasoning may COMPUTE a derived corroboration score (N sources, recency, agreement)
    but NEVER mutate the record — they append a new (derived) belief. Confidence is assigned
    once, refined downstream only by supersession.

Levels (use the named constant, never a literal):
"""
from __future__ import annotations

# Fact copied VERBATIM from a structured field — its existence is certain (a calendar event,
# a channel/document id, an attendee's stated email). No inference.
STRUCTURAL = 1.0

# Reliable parse of a structured source where a small parse/normalization step is involved
# (sender of an email, a user/identity snapshot, an observed thread or relationship).
HIGH = 0.9

# Structured but materially INFERRED (a Drive folder treated as a project; a company inferred
# from an email domain).
MEDIUM = 0.7

# Matched a deterministic text pattern that usually — but not always — indicates the fact
# (a commitment phrase: "I'll send … by Friday").
HEURISTIC = 0.55

# Weaker deterministic pattern with more false positives (a decision phrase).
LOW = 0.5

__all__ = ["STRUCTURAL", "HIGH", "MEDIUM", "HEURISTIC", "LOW"]
