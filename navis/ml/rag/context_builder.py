"""ContextBuilder — assemble a GroundedContext from the truth layers.

Pulls retrieved Memory facts (+ optionally State/Learning/Intelligence facts passed
in) into a token-budgeted, cited bundle. Compression strategy: rank by relevance,
then greedily admit facts until the token budget is hit (drop the tail, never
summarize away the citation). The model only ever sees what survives this budget.

When the question names a time range (e.g. "this week"), pass `since` / `until`
to scope the retriever — semantic similarity alone can't distinguish "meetings
this week" from meetings mentioned three months ago.
"""
from __future__ import annotations

import re
import uuid
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession

from navis.ml.rag.grounding import GroundedContext, Grounding
from navis.ml.retrieval.retriever import MemoryRetriever

_WORD = re.compile(r"\S+")


def estimate_tokens(text: str) -> int:
    # ~0.75 words/token; conservative integer estimate without a tokenizer dependency.
    return max(1, int(len(_WORD.findall(text)) / 0.75))


class ContextBuilder:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def build(
        self,
        org_id: uuid.UUID,
        question: str,
        *,
        k: int = 8,
        token_budget: int = 1500,
        extra: list[Grounding] | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        now: datetime | None = None,
    ) -> GroundedContext:
        hits = await MemoryRetriever(self.db).search(
            org_id, question, k=k, since=since, until=until
        )
        candidates: list[Grounding] = [
            Grounding(
                kind="memory",
                source=str(h.memory_id),
                text=h.source_text,
                occurred_at=h.event_timestamp,
            )
            for h in hits
        ]
        # Caller-supplied State/Learning/Intelligence facts rank after memory hits.
        candidates += list(extra or [])

        admitted: list[Grounding] = []
        used = estimate_tokens(question)
        for fact in candidates:
            cost = estimate_tokens(fact.text)
            if used + cost > token_budget:
                continue  # compression: drop the tail, keep citations intact
            admitted.append(fact)
            used += cost

        return GroundedContext(
            org_id=str(org_id),
            question=question,
            facts=admitted,
            token_estimate=used,
            as_of=now,
        )
