"""Collapse an answer so the founder is told each thing once.

Memory extracts several facts from a single message — a thread, the people on
it, the relationships it evidences — and each is independently retrievable. That
is right for a knowledge store and wrong for an answer: when the message matters,
every fact derived from it scores alike, and the founder gets told the same thing
five times while five other events wait outside the top-k.

    Email thread "Accepted: Meeting with mulayam…"   \\
    Person Mulayam <mulayamabmrc@gmail.com>           |  ONE event,
    Person Mulayam <mulayamabmrc@gmail.com>           |  five answer slots
    Received email from mulayamabmrc@gmail.com        |
    Received email from mulayamabmrc@gmail.com       /

TWO NOTIONS OF "THE SAME THING", and event alone was not enough. Collapsing on
the source event removes the repetition ABOVE, but not the repetition that
arrives across events: an entity is re-observed on every message it touches, so
one person accumulates many near-identical facts with a DIFFERENT event each.
Measured on the live corpus — person 5.7 memory records per entity,
relationship 5.6, company 3.5, against thread 1.1, meeting 1.2, document 1.6.

Event-collapse cannot touch that, and it was observable in production: a
retrieval for an address returned the identical string twice inside one
eight-slot answer. Worse than the repetition, it is a systematic bias — the
high-multiplicity types crowd the low-multiplicity ones out of every answer, so
threads, meetings and documents lose slots to restatements of the same person.
Collapsing on entity as well is what keeps an answer's entity types diverse.

This runs at RETRIEVAL time only. Nothing is deleted, no memory changes, no
embedding changes: the collapsed facts remain in the store and stay retrievable
by any other query. It is purely a decision about what to put in front of the
user right now.

The representative is the highest-RANKED fact of its event — deliberately not a
new judgement about which entity type is most informative. The ranking already
blends relevance and importance and has been validated end to end; layering a
second, unvalidated preference on top of it would be inventing a rule where one
already exists. Facts with unknown lineage are never merged, because "unknown"
is not evidence of sameness.
"""
from __future__ import annotations

import uuid
from collections.abc import Callable, Iterable
from typing import TypeVar

T = TypeVar("T")


def collapse_by_key(
    ranked: Iterable[T],
    key_of: Callable[[T], uuid.UUID | None],
) -> list[T]:
    """Keep the first item of each key, preserving the input order.

    `ranked` must already be in best-first order — this makes no ordering
    judgement of its own, it only removes later repeats of a key that is
    already represented. Items whose key is unknown (`None`) all survive.

    Deliberately generic: the retriever applies it once per ENTITY and once per
    EVENT (see `retriever.search`). Both are the same operation over a different
    notion of "the same thing", so there is one implementation, not two.
    """
    seen: set[uuid.UUID] = set()
    out: list[T] = []
    for item in ranked:
        key = key_of(item)
        if key is None:
            out.append(item)  # unknown lineage is not evidence of duplication
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


# The original name, kept because event-collapse is still one of the two passes
# and existing callers/tests read more clearly with it.
collapse_by_event = collapse_by_key
