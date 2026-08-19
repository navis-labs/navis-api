"""ContextBuilder — assemble a GroundedContext from the truth layers.

Pulls retrieved Memory facts, what the organization has LEARNED from its own
outcomes (+ optionally State/Intelligence facts passed in) into a token-budgeted,
cited bundle. Compression strategy: rank by relevance, then greedily admit facts
until the token budget is hit (drop the tail, never summarize away the citation).
The model only ever sees what survives this budget.

When the question names a time range (e.g. "this week"), pass `since` / `until`
to scope the retriever — semantic similarity alone can't distinguish "meetings
this week" from meetings mentioned three months ago.

WHY LEARNING IS READ HERE AND STATE IS NOT. `navis.ml` may not import
`navis.state` or `navis.reasoning` (AST invariants in tests/ml/test_ml_layer.py),
so those arrive pre-flattened from the composition root via `extra`. Learning is
different and deliberately so: the same invariant states that "ML may READ
memory/state/learning/intelligence" and omits `learning` from ML's forbidden
list. So ML consumes Learning directly — through `learning_snapshot()`, Learning's
own read contract, never its tables.
"""
from __future__ import annotations

import re
import uuid
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession

from navis.learning.metrics import LearningSnapshot, learning_snapshot
from navis.ml.rag.grounding import GroundedContext, Grounding
from navis.ml.retrieval.reranker import _overlap
from navis.ml.retrieval.retriever import MemoryRetriever
from navis.platform.monitoring import get_logger

logger = get_logger("ml.rag.context")

_WORD = re.compile(r"\S+")

LEARNING_KIND = "learning"

# How each outcome surface is NAMED to the model. The three are phrased apart on
# purpose: a delivery failure and a customer's silence are different facts about
# the organization, and a model handed "3 failures" with no surface would fold
# them into one notion of "things went wrong" — reintroducing, in the prompt, the
# exact conflation Learning's two-axis classification exists to prevent.
_SURFACE_PHRASING = {
    "delivery": (
        "Action delivery (whether the send itself worked — a channel/credential "
        "fact, NOT a judgement about whether acting was right)"
    ),
    "judgement": (
        "Human judgement on proposals Navis made (a person's verdict — the most "
        "direct feedback on its reasoning)"
    ),
    "world": (
        "World response after acting (what people actually did — the only "
        "surface that answers whether it worked)"
    ),
}

# Composition bands. Keyed on the fact's own `kind`, so this stays
# connector-agnostic: a Gmail thread, a Jira issue and a Zoom meeting all arrive
# as retrieved evidence and are treated identically. Nothing here knows which
# system produced a fact.
#
# Kinds that are CONCLUSIONS — what the organization worked out. Few, dense, and
# the thing a founder actually asked for, so they always lead.
_CONCLUSION_KINDS = frozenset({"reasoning", "intelligence"})
# Kinds that DESCRIBE the organization's current state. Numerous (a ScopedView
# routinely carries 20+), which is why only a bounded, question-ranked slice may
# precede retrieval — see `_compose`.
_STATE_KINDS = frozenset({"state", "learning"})

# How many State facts may sit between the conclusions and the retrieved
# evidence. Sized from the measured regression: ordering purely by kind put 8
# conclusions + 22 State entities ahead of all 8 retrieved facts, and six
# questions that need retrieval detail ("which documents are being worked on",
# "which companies are we in conversation with") went from answered to
# "cannot be determined" — not because evidence was dropped (8/8 were retained,
# ~900 of 1500 tokens) but because the model anchored on 30 derived facts before
# reaching any. Six is enough for the State that explains a conclusion without
# burying what it was derived from; the rest still appear, after the evidence.
_LEADING_STATE_FACTS = 6


def _compose(candidates: list[Grounding], question: str) -> list[Grounding]:
    """Order the context: conclusions, the State that explains them, evidence.

    Three bands rather than a flat sort by kind:

        conclusions            reasoning / intelligence — always first
        leading State          the `_LEADING_STATE_FACTS` most question-relevant
        retrieved evidence     memory + raw workspace facts, in relevance order
        remaining State        still present, still citable, read later

    The middle band is what a flat kind-sort got wrong. A ScopedView carries 20+
    entities; putting all of them ahead of retrieval buried the evidence under
    facts the question never asked about.

    QUESTION-AWARE VIA EXISTING MACHINERY. The State band is ordered by the same
    lexical overlap the reranker already uses to score retrieval hits — one
    shared definition of "does this text answer this question", not a second
    ranking system. Ties keep the caller's order (`scope_for`'s own ranking) via
    the positional tiebreak, so the result is fully deterministic: same
    (question, candidates) in, same order out, no clock and no model.

    Nothing is dropped and no provenance changes. This decides reading order,
    and — only if the budget binds — who is dropped first.
    """
    conclusions: list[Grounding] = []
    state: list[tuple[float, int, Grounding]] = []
    evidence: list[Grounding] = []
    for i, g in enumerate(candidates):
        if g.kind in _CONCLUSION_KINDS:
            conclusions.append(g)
        elif g.kind in _STATE_KINDS:
            state.append((-_overlap(question, g.text), i, g))
        else:
            evidence.append(g)
    state.sort()  # by -overlap, then original position
    ranked_state = [g for _s, _i, g in state]
    # The State tail goes AFTER the evidence — it is not dropped.
    #
    # What made 30 derived facts harmful was that all of them PRECEDED the 8
    # retrieved ones: the model anchored on them and then declined for lack of
    # specifics. Six questions went from answered to "the grounded facts do not
    # provide specific information", with 8/8 evidence still in the prompt. That
    # is an ordering failure, and bounding the LEADING band is what fixes it.
    #
    # Truncating the list as well was over-correction. It cost coverage the
    # budget was never asking for — measured on the real org, 16 of 22 State
    # entities were discarded on every turn while 520-1000 of the 1500 tokens
    # went unspent — and it left questions about org-wide breadth ("which
    # companies are we in conversation with") unanswerable from a context whose
    # single company had been squeezed out of a six-slot band.
    #
    # Appending costs retrieval nothing: the budget loop below admits in this
    # order, so every retrieved fact is admitted BEFORE the first tail entity is
    # considered. The tail spends only what the evidence left over, and is the
    # first thing dropped when the budget does bind.
    return [
        *conclusions,
        *ranked_state[:_LEADING_STATE_FACTS],
        *evidence,
        *ranked_state[_LEADING_STATE_FACTS:],
    ]


def estimate_tokens(text: str) -> int:
    # ~0.75 words/token; conservative integer estimate without a tokenizer dependency.
    return max(1, int(len(_WORD.findall(text)) / 0.75))


# How many signal-reliability facts may enter one context. The token budget would
# bound this anyway; the cap bounds the WORK, so an org with hundreds of tracked
# signals does not render hundreds of strings to discard most of them.
_MAX_SIGNAL_FACTS = 5


def learning_facts(snapshot: LearningSnapshot) -> list[Grounding]:
    """What the organization has learned from its own outcomes, as cited facts.

    PURE — a function of the snapshot alone. No clock, no model, no database, no
    randomness, so the same (org, learning_version) always renders the same facts
    in the same order. That is what makes a learned fact reproducible rather than
    a thing the model saw once.

    NOTHING IS INFERRED HERE. Two sources, both already decided by Learning:

      * `patterns` — Learning's own derived reliability facts, already gated on
        decisive sample count. This layer renders them; it does not re-threshold
        them, and it does not compute a reliability of its own. A second opinion
        about what "reliable" means is exactly what the contract prevents.
      * `outcomes_by_surface()` — counts, grouped by Learning's own taxonomy.

    An organization with no outcomes yet produces NO facts. Not a zero, not an
    "insufficient data" placeholder — nothing, so a model cannot cite the absence
    of evidence as evidence.
    """
    if snapshot.learning_version == 0:
        return []

    facts: list[Grounding] = []
    version_tag = f"learning_version:{snapshot.learning_version}"

    # 1. Delivery / judgement / world, kept apart and labelled as such.
    by_surface = snapshot.outcomes_by_surface()
    for surface, phrasing in _SURFACE_PHRASING.items():
        aggs = by_surface.get(surface)
        if not aggs:
            continue  # this org has never had an outcome of this kind
        counts = ", ".join(f"{a.total} {a.key}" for a in aggs)
        facts.append(
            Grounding(
                kind=LEARNING_KIND,
                source=f"outcome_surface:{surface}",
                text=f"{phrasing}: {counts}.",
                occurred_at=snapshot.as_of,
                lineage=(version_tag,),
            )
        )

    # 2. Which reasoning signals have actually proved out. Deterministic pick:
    #    most decisive samples first, then signal_id to break ties.
    ranked = sorted(snapshot.patterns, key=lambda p: (-p.samples, p.signal_id))
    for pattern in ranked[:_MAX_SIGNAL_FACTS]:
        supporting = snapshot.lineage.get(pattern.signal_id, [])
        facts.append(
            Grounding(
                kind=LEARNING_KIND,
                source=f"signal_pattern:{pattern.signal_id}",
                text=(
                    f"When reasoning signal {pattern.signal_id} fired, the outcome "
                    f"was positive {pattern.success_rate:.0%} of {pattern.samples} "
                    f"decisive outcomes — this organization has found it "
                    f"{pattern.kind}."
                ),
                occurred_at=snapshot.as_of,
                lineage=(
                    version_tag,
                    f"signal:{pattern.signal_id}",
                    *(f"learning_event:{e}" for e in supporting),
                ),
            )
        )
    return facts


async def learning_groundings(db: AsyncSession, org_id: uuid.UUID) -> list[Grounding]:
    """Read this org's learning through Learning's contract and render it.

    The ONLY entry point from ML into Learning. It calls `learning_snapshot`,
    which is org-scoped in every statement it runs, so isolation is a property of
    the contract rather than of this caller remembering to filter.
    """
    return learning_facts(await learning_snapshot(db, org_id))


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
        include_learning: bool = True,
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
        # What the org has LEARNED from its own outcomes. Read AFTER retrieval and
        # wrapped, deliberately on both counts: retrieval is the answer, learning
        # is enrichment, and enrichment is never worth failing the turn for. A
        # Learning outage degrades the context to exactly what it was before this
        # existed — the same contract `_state_groundings` keeps in the API layer.
        #
        # This is also why ML PULLS at query time instead of subscribing to
        # LEARNING_UPDATED: the Observation -> Learning -> Intelligence chain runs
        # synchronously inside one event handler, and hanging feature computation
        # off it would put ML latency on the path that records outcomes.
        learned: list[Grounding] = []
        if include_learning:
            try:
                learned = await learning_groundings(self.db, org_id)
            except Exception:  # noqa: BLE001 — enrichment never fails the turn
                logger.exception("learning groundings unavailable org=%s", org_id)

        candidates += learned
        candidates += list(extra or [])
        # Order by WHAT THE FACT IS, not by which layer supplied it: conclusions
        # lead, a bounded question-ranked slice of State explains them, then the
        # evidence they were derived from. See `_compose` for why the State slice
        # is bounded. Nothing is dropped and no provenance changes.
        candidates = _compose(candidates, question)

        # Cost what is actually SENT, not just the fact text. The prompt carries
        # a header plus, per fact, an index, a date and a `(kind:source)`
        # citation whose source is a uuid — together roughly a third more tokens
        # than `text` alone, so a context declared at 1500 rendered near 2100.
        # Both sides now go through the same `render`/`render_line` definition,
        # which is what keeps them from drifting apart again.
        admitted: list[Grounding] = []
        header = GroundedContext(
            org_id=str(org_id), question=question, facts=[], as_of=now
        ).render()
        used = estimate_tokens(header)
        for fact in candidates:
            cost = estimate_tokens(fact.render_line(len(admitted) + 1))
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
