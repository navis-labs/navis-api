"""Collapse an answer to one fact per underlying event.

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


def collapse_by_event(
    ranked: Iterable[T],
    event_of: Callable[[T], uuid.UUID | None],
) -> list[T]:
    """Keep the first item of each event, preserving the input order.

    `ranked` must already be in best-first order — this makes no ordering
    judgement of its own, it only removes later repeats of an event that is
    already represented. Items whose event is unknown (`None`) all survive.
    """
    seen: set[uuid.UUID] = set()
    out: list[T] = []
    for item in ranked:
        event = event_of(item)
        if event is None:
            out.append(item)  # unknown lineage is not evidence of duplication
            continue
        if event in seen:
            continue
        seen.add(event)
        out.append(item)
    return out
