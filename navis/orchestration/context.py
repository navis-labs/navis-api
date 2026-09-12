"""build_grounded_context — the ONE task-aware context composer.

WHAT THIS IS. Given an organization and a task in words, assemble the bounded,
cited bundle of organizational facts that answering it requires: retrieved
Memory, the current State projection, the conclusions Reasoning drew from it,
what Intelligence holds as true, and what Learning has measured. The result is a
`GroundedContext` — facts with sources, nothing generated.

WHY IT LIVES HERE. Every line below used to live inside `api/chat_router.py`,
where exactly one caller could reach it. `navis.ml` may not import `navis.state`
or `navis.reasoning` and vice versa (AST-pinned in tests/ml/test_ml_layer.py), so
the join between them has to happen in a composition layer — and `orchestration`
is the one this codebase already names for that: `orchestration/brief.py` joins
State, Reasoning, Intelligence, Decision and Adaptive for the morning brief. This
is the same seam doing the same kind of work for a different question.

    orchestration/brief.py    State + Reasoning + Intelligence -> Decision
    orchestration/context.py  State + Reasoning + Memory + ML  -> GroundedContext
                              + Intelligence + Learning

ONE IMPLEMENTATION, TWO CALLERS, AND THAT IS THE POINT:

    /chat                     -> build_grounded_context(...) -> LLM -> prose
    POST /developer/context   -> build_grounded_context(...) -> the bundle itself

A human gets it narrated; an agent gets it structured. Neither gets a different
organization. If these two ever disagree about what is true of a company, it will
be because of a bug in one function rather than a divergence between two products.

THE TASK SHAPES RETRIEVAL, NOT SELECTION. Worth stating plainly because it is the
most likely thing to be misread: `scope_for(state, MorningBrief(), ...)` returns
the same org-wide slice whatever the task says. Task-awareness enters through
`MemoryRetriever` — the query is embedded, searched against the organization's
vectors and reranked — and through subject resolution and the time window. There
is deliberately NO second ranking system for the State layer: inventing one for
the Developer API would be a second opinion about what matters to an organization,
which is the whole failure this module exists to prevent.

MUTATION IS THE CALLER'S CHOICE, AND IT IS EXPLICIT. `read_only=True` reads State
through `StateService.read_only_state`, which never folds, never rebuilds and
never disposes; `read_only=False` uses `get_state`, which serves cache but repairs
a corrupt or absent snapshot by deleting and rebuilding it. `/chat` keeps the
second — that is what it has always done and a human session is entitled to spend
the org's compute — and the Developer boundary takes the first, because a machine
asking "what does this organization know?" must not be able to trigger a full fold
of its memory by asking. NOTHING ELSE ON THIS PATH WRITES: every other dependency
was scanned and is a read.

BEST-EFFORT BY CONSTRUCTION. State unavailable, identity resolution down, the
expansion hop failing — each costs the bundle some depth and never the answer.
The one thing that is never best-effort is honesty about what happened: a bundle
assembled without State says so, rather than looking like an organization with
nothing in it.
"""
from __future__ import annotations

import asyncio
import re
import uuid
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from navis.identity_resolution.search import search_persons
from navis.memory.envelope import EntityType
from navis.memory.lineage import classifications_by_entity
from navis.ml.rag.context_builder import ContextBuilder, SessionFactory, estimate_tokens
from navis.ml.rag.grounding import GroundedContext, Grounding
from navis.platform.monitoring import get_logger
from navis.platform.monitoring.metrics import metrics, timed
from navis.reasoning.v2 import run_reasoning_v2
from navis.state.v2 import MorningBrief, StateService, scope_for
from navis.state.v2.models import OrganizationalState
from navis.state.v2.scope import EntityRef, ScopedView

logger = get_logger("orchestration.context")


# ── Budgets and vocabulary ──────────────────────────────────────────────────

# Bound the State read. State is cache-first here (never `incremental_update`),
# but a cold cache is a full fold over memory_records and was measured at ~13s
# against a serverless Postgres — unacceptable inside a chat turn. On timeout we
# degrade to the memory-only bundle rather than making the caller wait.
#
# 4.0s is sized to a MEASURED distribution, not chosen on principle:
#
#     warm, steady state      p50 1.4s   p95 1.6s
#     first call in a process     3.9s   (snapshot read + deserialize)
#     genuinely cold fold         5.4s   (still correctly refused)
#
# 2.5s was tried and produced 26/26 timeouts, because the residual is network
# RTT to a remote database and no code change removes it.
REASONING_GROUNDING_TIMEOUT_S = 4.0

# `Grounding.kind` for a fact derived from a ReasoningSignal, and for one scoped
# State entity. The documented vocabulary is memory | state | learning |
# intelligence; `kind` is a free string with no validation layer.
REASONING_KIND = "reasoning"
CONTEXT_KIND = "state"

# Kinds DERIVED from State rather than retrieved for the task. They are org-wide
# and present on nearly every turn, so they must never count as "retrieval found
# something" — doing so permanently suppresses the workspace fallback and
# silently loses raw Calendar/Gmail evidence.
DERIVED_KINDS = frozenset({REASONING_KIND, CONTEXT_KIND})

# Keep one entity's rendered fact short: the bundle is a budgeted list of many
# facts, not a record dump. State's full field dict stays out of the prompt.
CONTEXT_LABEL_MAX = 120
CONTEXT_DETAIL_MAX = 90
CONTEXT_TEXT_MAX = 320
# How many members of a list-valued field (attendees, participants, owners) are
# named before the rest are summarised as "+N more".
CONTEXT_LIST_ITEMS = 4

# The one token budget for this path — the same value `ContextBuilder.build`
# enforces, so both branches bound the bundle identically.
CONTEXT_TOKEN_BUDGET = 1500


def compose_within_budget(base: GroundedContext, extra: list[Grounding]) -> GroundedContext:
    """Append derived facts to a context under the SAME token budget.

    The workspace fallback constructs its GroundedContext directly instead of
    going through ContextBuilder, so the builder's budget never applied to it.
    That was harmless while it carried ~12 short facts; adding State (22) and
    Reasoning (10) pushed the rendered prompt to ~1777 tokens — past the 1500 the
    retrieval branch enforces. Same policy as ContextBuilder: admit greedily in
    order, drop the tail, never summarise away a citation.

    `token_estimate` is recomputed here; `replace` alone would carry the
    pre-append value forward and under-report the real size.

    Costs the RENDERED line per fact, matching ContextBuilder — the citation
    prefix and date are real tokens, and counting `text` alone under-reported
    every fact by roughly a third.
    """
    admitted = list(base.facts)
    used = estimate_tokens(replace(base, facts=[]).render()) + sum(
        estimate_tokens(f.render_line(i)) for i, f in enumerate(admitted, 1)
    )
    for fact in extra:
        cost = estimate_tokens(fact.render_line(len(admitted) + 1))
        if used + cost > CONTEXT_TOKEN_BUDGET:
            continue
        admitted.append(fact)
        used += cost
    return replace(base, facts=admitted, token_estimate=used)


# ── How one State entity is rendered ────────────────────────────────────────
#
# The rules for what may be rendered:
#   * EXISTING fields only, read by name off `EntityRef.fields`. Nothing is
#     computed, inferred, defaulted or joined — an absent field renders nothing
#     rather than a guess, so a thin entity stays honestly thin.
#   * A WHITELIST per entity type, never the dict. `fields` is whatever the
#     connector's payload carried, so dumping it would leak raw connector shape
#     and break the "State carries pointers, not bodies" rule.
#   * BOUNDED and DETERMINISTIC: fixed field order, clipped values, capped list
#     members, capped line. Same entity in, same line out, no clock, no model.
#   * Bodies stay out. A thread's `snippet` and a document's contents are
#     retrieval's job; State says what EXISTS and who it involves.

# Which field names the label is taken from, most identifying first. Covers
# every type State can hold. Without `person_email`, relationships rendered as
# bare UUIDs — useless grounding.
LABEL_FIELDS_DEFAULT = (
    "name", "title", "subject", "person_email", "email", "domain", "text",
)
# A commitment or a decision IS its text. The default order reaches `subject`
# first, which labelled every commitment with the SUBJECT LINE OF THE EMAIL it
# was extracted from. The subject is still rendered — as `re`, where it belongs.
LABEL_FIELDS: dict[str, tuple[str, ...]] = {
    EntityType.COMMITMENT.value: ("text",) + LABEL_FIELDS_DEFAULT,
    EntityType.DECISION.value: ("text",) + LABEL_FIELDS_DEFAULT,
}

# The decision-relevant fields per entity type, in render order: (prefix, keys).
# Two keys means a name/address pair — "Abhi <a@b.com>" when both are present,
# whichever exists when only one is. An empty prefix renders the value alone.
DETAIL_FIELDS: dict[str, tuple[tuple[str, tuple[str, ...]], ...]] = {
    # WHO OWES IT, BY WHEN, ABOUT WHAT. `speaker_*` is the promise's owner in
    # State's own terms — `folds.compute_links` builds the commitment's `owner`
    # link from exactly this address.
    EntityType.COMMITMENT.value: (
        ("owner", ("speaker_name", "speaker_email")),
        ("status", ("status",)),
        # Both due fields render: `due_at` only exists when the promise stated a
        # date, and `due_text` preserves what was actually said ("end of week").
        ("due", ("due_at",)),
        ("due stated as", ("due_text",)),
        ("promised", ("promised_at",)),
        ("re", ("subject",)),
    ),
    EntityType.DECISION.value: (
        ("status", ("status",)),
        # "record" (one the org made) vs "request" (one the org owes) — the
        # difference between an answer and an obligation.
        ("kind", ("kind",)),
        ("asked by", ("asked_by_name", "asked_by")),
        ("asked", ("asked_at",)),
        ("re", ("subject",)),
    ),
    # WHEN and WHO — the two things every meeting question asks for.
    EntityType.MEETING.value: (
        ("starts", ("starts_at",)),
        ("ends", ("ends_at",)),
        ("attendees", ("attendees",)),
        ("status", ("status",)),
    ),
    # WHO is on the other side and WHICH WAY it last moved. `direction ==
    # "inbound"` means they spoke last and nobody has replied — the fact behind
    # every "still open / unanswered" question.
    EntityType.THREAD.value: (
        ("last message from", ("from_name", "from_email")),
        ("direction", ("direction",)),
        ("category", ("category",)),
        ("participants", ("participants",)),
    ),
    # The address is what makes a person joinable to a thread, a meeting and a
    # relationship. `category` says whether this "person" is a newsletter — a
    # founder must not be told to re-engage noreply@.
    EntityType.PERSON.value: (
        ("", ("email",)),
        ("category", ("category",)),
    ),
    EntityType.RELATIONSHIP.value: (
        ("direction", ("direction",)),
        ("via", ("via",)),
        ("category", ("category",)),
    ),
    # A company folds from an email domain and carries only that; the label
    # already shows it, so `render_details` drops it as a duplicate.
    EntityType.COMPANY.value: (
        ("domain", ("domain",)),
    ),
    EntityType.DOCUMENT.value: (
        ("owners", ("owners",)),
        ("status", ("status",)),
    ),
    EntityType.PROJECT.value: (
        ("status", ("status",)),
    ),
    EntityType.CUSTOMER.value: (
        ("ref", ("external_ref",)),
    ),
    EntityType.CHANNEL.value: (
        ("visibility", ("visibility",)),
    ),
}


def field_text(value: Any) -> str | None:
    """One field as text, or None when it carries nothing to say.

    Lists are capped at `CONTEXT_LIST_ITEMS` with the remainder counted, so an
    80-attendee meeting costs the same bounded number of tokens as a 2-attendee
    one. Booleans and numbers are deliberately NOT rendered: no field on the
    whitelist is one, and stringifying `False` would assert a fact the fold
    never established.
    """
    if isinstance(value, str):
        return value.strip()[:CONTEXT_DETAIL_MAX] or None
    if isinstance(value, (list, tuple)):
        members = [v.strip() for v in value if isinstance(v, str) and v.strip()]
        if not members:
            return None
        shown = ", ".join(m[:CONTEXT_DETAIL_MAX] for m in members[:CONTEXT_LIST_ITEMS])
        extra = len(members) - CONTEXT_LIST_ITEMS
        return f"{shown} +{extra} more" if extra > 0 else shown
    return None


def label_from_fields(entity_type: str, fields: dict[str, Any]) -> str | None:
    """The entity's human name, or None when State knows no name for it.

    None rather than the entity_id: a caller rendering a signal's subject wants
    to say nothing rather than quote a uuid.
    """
    for key in LABEL_FIELDS.get(entity_type, LABEL_FIELDS_DEFAULT):
        text = field_text(fields.get(key))
        if text:
            return text[:CONTEXT_LABEL_MAX]
    return None


def entity_label(entity: EntityRef) -> str:
    """A human-readable name for a scoped entity, whatever type it is.

    Deliberately source-agnostic: it reads State's normalised field names, never
    anything connector-shaped. A new source that folds into the same entity types
    renders here with no change.
    """
    return label_from_fields(entity.entity_type, entity.fields) or entity.entity_id


def render_details(entity_type: str, fields: dict[str, Any], label: str) -> str:
    """The whitelisted fields of one entity, as `prefix value; prefix value`.

    A detail equal to the label is dropped rather than repeated — a company whose
    only field is the domain it is already named after adds no information for
    the tokens it costs.
    """
    parts: list[str] = []
    for prefix, keys in DETAIL_FIELDS.get(entity_type, ()):
        values = [t for t in (field_text(fields.get(k)) for k in keys) if t]
        if not values or values[0] == label:
            continue
        rendered = values[0] if len(values) == 1 else f"{values[0]} <{values[1]}>"
        parts.append(f"{prefix} {rendered}" if prefix else rendered)
    return "; ".join(parts)


def scoped_view_to_groundings(view: ScopedView) -> list[Grounding]:
    """ScopedView -> cited facts. Pure; no I/O, no org lookup of its own.

    Takes the already-org-scoped view rather than an org_id precisely so this
    cannot fetch the wrong tenant's state: every entity here came from one
    `OrganizationalState` built for one org.

    Provenance is preserved, not discarded — `source` is the State entity_id (a
    deterministic, org-scoped identity) and `lineage` carries the entity's
    `memory_ids` / `signal_ids` so a citation can still be drilled to the raw
    record. The ScopedView itself never crosses into `navis.ml`.
    """
    groundings: list[Grounding] = []
    for entity in view.entities:
        citation = view.citations.get(entity.entity_id)
        occurred_at: datetime | None = None
        if entity.last_seen:
            try:
                occurred_at = datetime.fromisoformat(entity.last_seen)
            except ValueError:  # non-ISO last_seen — a fact without a date is fine
                occurred_at = None
        label = entity_label(entity)
        details = render_details(entity.entity_type, entity.fields, label)
        groundings.append(
            Grounding(
                kind=CONTEXT_KIND,
                source=entity.entity_id,
                # type: label — the decision-relevant fields State already holds
                # — (reason) — the projection's own explanation of why this
                # entity made the cut ("open, oldest activity").
                text=(
                    f"{entity.entity_type}: {label}"
                    + (f" — {details}" if details else "")
                    + f" ({entity.reason})"
                )[:CONTEXT_TEXT_MAX],
                occurred_at=occurred_at,
                lineage=(
                    tuple(f"memory:{m}" for m in citation.memory_ids)
                    + tuple(f"signal:{s}" for s in citation.signal_ids)
                    if citation
                    else ()
                ),
            )
        )
    return groundings


async def reset_session(db: AsyncSession) -> None:
    """Return a session to a usable state after best-effort work failed on it.

    `asyncio.wait_for` CANCELS the coroutine it is waiting on. When that
    coroutine is a multi-statement DB read, the cancellation lands mid
    transaction and SQLAlchemy marks it invalid — every later statement on that
    session then raises `PendingRollbackError`, whatever it was trying to do.

    That is what turned a soft degradation into a 500: the timeout branch
    correctly returned `[]` and logged a warning, but handed the CALLER a
    poisoned session, and the very next thing the composer does is retrieval.
    The enrichment failing must cost the bundle some context — never the turn.

    Rolling back discards nothing: everything on this path is read-only, and the
    request session commits only at the end of `get_session`.
    """
    try:
        await db.rollback()
    except Exception:  # noqa: BLE001 — a cleanup failure must not mask the original
        logger.exception("context: could not reset session after failed enrichment")


def reasoning_text(state: OrganizationalState, signal: Any) -> str:
    """One ReasoningSignal as a cited conclusion, NAMING whom it is about.

    A signal already carries its subject as `(entity_type, entity_id)`, but the
    id is a uuid, so rendering `title — why` alone dropped the only part a reader
    can act on. Measured on the real org: four RELATIONSHIP_COOLING signals
    reached the model as four identical lines, and every answer about who had
    gone quiet could only say "several relationships".

    The name is resolved through the SAME `label_from_fields` the ScopedView path
    uses, against the state the signal was computed from — one definition of what
    an entity is called, and no second lookup or ranking.
    """
    title, why = signal.title, signal.why
    if not signal.entity_type or not signal.entity_id:
        return f"{title} — {why}"
    raw = state.entities.get(signal.entity_type, {}).get(signal.entity_id)
    subject = label_from_fields(signal.entity_type, (raw or {}).get("fields") or {})
    if not subject or subject in title:
        return f"{title} — {why}"
    return f"{title} ({signal.entity_type}: {subject}) — {why}"


async def relevance_evidence(db: AsyncSession, org_id: uuid.UUID) -> dict[str, frozenset[str]]:
    """Organizational-relevance evidence for Reasoning, read at the composition root.

    Reasoning performs no I/O, so the classification sets it decides relevance
    from are fetched here and handed in (see `reasoning/v2/base.py`).

    Degrades to `{}` on its OWN rather than inside the caller's try-block: an
    empty map means UNKNOWN, and every engine's documented response to UNKNOWN is
    to keep its prior behaviour. Failing this read must cost the bundle some
    precision, never the whole State/Context enrichment.
    """
    try:
        return await classifications_by_entity(
            db, org_id, entity_type=EntityType.RELATIONSHIP.value
        )
    except Exception:  # noqa: BLE001 — precision is never worth failing the turn
        logger.exception("context: relevance evidence unavailable org=%s", org_id)
        metrics.incr("context.relevance_evidence.failed")
        await reset_session(db)
        return {}


@dataclass(frozen=True, slots=True)
class StateGrounding:
    """State and Reasoning as cited facts, with what they describe.

    `available` is the field that keeps this honest. An organization whose State
    could not be read and an organization with nothing in it produce the same
    empty `groundings`, and a consumer must be able to tell them apart —
    especially the Developer boundary, which refuses to rebuild a missing
    snapshot and would otherwise report a cold tenant as an empty one.
    """

    groundings: list[Grounding]
    watermark: str | None
    as_of: datetime | None
    #: THE NEWEST EVENT TIME folded into the projection — when the most recent
    #: thing Navis knows about actually HAPPENED.
    #:
    #: NOT the same as `as_of`, and the gap between them is the point. `as_of` is
    #: the observation horizon: the moment Navis last LEARNED something. A
    #: backfill, a re-extraction or a replay writes memory rows today that carry
    #: last month's event timestamps, which advances `as_of` to now while the
    #: organization's actual knowledge stands still. A reader given only `as_of`
    #: reads that as "current" and is wrong by however long ingestion has been
    #: broken.
    #:
    #: Read from State's existing `fold_frontier`, which the builder has always
    #: computed for its own correctness (it is how an out-of-order record is
    #: detected and forced into a rebuild). Nothing new is derived here — the
    #: value existed and simply had no way out of State.
    newest_fact_at: datetime | None
    available: bool


async def state_groundings(
    db: AsyncSession, org_id: uuid.UUID, *, read_only: bool = False
) -> StateGrounding:
    """Current organizational understanding, as cited facts.

    State is read ONCE and feeds the two parallel deterministic read-side
    computations, exactly as the architecture defines them:

        State ├─ scope_for(...)       -> ScopedView   -> Grounding(kind="state")
              └─ run_reasoning_v2()   -> Signal[]     -> Grounding(kind="reasoning")

    Context does NOT feed Reasoning and Reasoning never sees the model. What
    crosses into `navis.ml` is a flat `list[Grounding]` — never a ScopedView,
    never a ReasoningSignal, never an OrganizationalState.

    `read_only` PICKS THE STATE READ, and it is the only difference between what
    the two callers do here:

        False  `get_state`        — serves cache, but DISPOSES and REBUILDS a
                                    snapshot that is absent or corrupt. What
                                    `/chat` has always done.
        True   `read_only_state`  — returns the stored projection or None, and
                                    writes nothing under any circumstance.

    Policy is intentionally None for `scope_for`: it only tunes `brief_max_items`
    and resolving it costs a second round trip inside a latency-bounded path.
    Both callers take the default per-type cap, so both see the same slice.

    Best-effort. State unavailable, slow, or empty all degrade to no groundings,
    which leaves the memory-only path exactly as it was.
    """
    try:
        with timed("context.state_groundings"):
            service = StateService(db)
            state = await asyncio.wait_for(
                service.read_only_state(org_id) if read_only
                else service.get_state(org_id),
                timeout=REASONING_GROUNDING_TIMEOUT_S,
            )
            if state is None:
                # Only reachable on the read-only path: there is no usable
                # snapshot and we declined to build one. Honest absence.
                metrics.incr("context.state_groundings.absent")
                return StateGrounding([], None, None, None, available=False)
            view = scope_for(state, MorningBrief(), None, as_of=state.as_of)
            signals = run_reasoning_v2(
                state, evidence=await relevance_evidence(db, org_id)
            )
    except asyncio.TimeoutError:
        # Expected on a cold projection — a normal degradation, not an incident.
        # Logged WITHOUT a traceback so a cold start doesn't emit a 100-line
        # stack on every first request.
        logger.warning(
            "context: state read exceeded %.1fs, composing without state/reasoning org=%s",
            REASONING_GROUNDING_TIMEOUT_S, org_id,
        )
        metrics.incr("context.state_groundings.timeout")
        await reset_session(db)
        return StateGrounding([], None, None, None, available=False)
    except Exception:  # noqa: BLE001 — enrichment is never worth failing the turn
        logger.exception("context: state grounding unavailable org=%s", org_id)
        metrics.incr("context.state_groundings.failed")
        await reset_session(db)
        return StateGrounding([], None, None, None, available=False)

    context_groundings = scoped_view_to_groundings(view)
    reasoning_groundings = [
        Grounding(
            kind=REASONING_KIND,
            # The deterministic ReasoningSignal id — a real lineage pointer, so a
            # citation resolves back to the engine + watermark that produced it.
            source=str(s.signal_id),
            text=reasoning_text(state, s),
            occurred_at=s.state_as_of,
        )
        for s in signals
    ]
    metrics.incr("context.state_groundings.context", value=len(context_groundings))
    metrics.incr("context.state_groundings.reasoning", value=len(reasoning_groundings))
    watermark = state.watermark.last_memory_id
    # Context (what IS the case) before Reasoning (what it implies), so a
    # squeezed token budget keeps the factual layer and drops interpretation
    # first. Conclusions before the entities they describe: ContextBuilder
    # re-sorts by kind anyway, but the fallback path composes this list
    # directly, so the order has to be right at the source too.
    return StateGrounding(
        groundings=[*reasoning_groundings, *context_groundings],
        watermark=str(watermark) if watermark else None,
        as_of=state.as_of,
        # State has always carried this; it simply had no way out. `fold_frontier`
        # is maintained by the builder for its own correctness — it is how a
        # record that arrives late but happened early is detected and forced into
        # a rebuild — so it is the frontier of EVENT time, which is exactly the
        # question `as_of` cannot answer. Read, never computed.
        newest_fact_at=state.fold_frontier.last_event_timestamp,
        available=True,
    )


# ── Who the task is about ───────────────────────────────────────────────────

EMAIL_IN_QUESTION = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
# A capitalized run of one or two words — the shape a person's name takes in an
# English question. Sentence-initial words match too; `NOT_A_NAME` filters the
# ones that are grammar rather than identity.
CAPITALIZED_RUN = re.compile(r"\b([A-Z][a-z]{1,})(?:\s+([A-Z][a-z]{1,}))?\b")
# Capitalized tokens that are never a person. Question openers dominate because
# they are sentence-initial and therefore always capitalized. Kept small on
# purpose: the real filter is the identity graph — a word that matches no person
# costs one indexed lookup and resolves to nothing.
NOT_A_NAME = frozenset(
    {
        "who", "what", "when", "where", "why", "how", "which", "whose",
        "is", "are", "was", "were", "do", "does", "did", "can", "could",
        "should", "would", "will", "shall", "the", "a", "an", "my", "our",
        "their", "his", "her", "its", "this", "that", "these", "those",
        "tell", "show", "give", "find", "list", "explain", "summarise",
        "summarize", "recap", "navis", "today", "tomorrow", "yesterday",
        "monday", "tuesday", "wednesday", "thursday", "friday", "saturday",
        "sunday", "january", "february", "march", "april", "may", "june",
        "july", "august", "september", "october", "november", "december",
    }
)
# How many distinct candidate tokens we spend a lookup on. Each is one indexed
# query; a task naming three capitalized things is already unusual.
MAX_SUBJECT_CANDIDATES = 3


@dataclass(frozen=True, slots=True)
class SubjectResolution:
    """What the identity graph said about the person this task names.

    Exactly one of these holds:
      * `match` set        — one person, unambiguously
      * `candidates` set   — several people matched; the AMBIGUITY IS THE ANSWER
                             and is surfaced to the caller, never resolved by
                             picking the first
      * both empty         — the task names nobody we know; compose unscoped
    """

    match: Any = None                      # PersonMatch | None
    candidates: tuple[Any, ...] = ()       # tuple[PersonMatch, ...]
    query: str | None = None

    @property
    def ambiguous(self) -> bool:
        return len(self.candidates) > 1


def subject_candidates(question: str) -> list[str]:
    """Plausible person tokens in a task, most specific first.

    Literal addresses lead: an email is an unambiguous reference to a person, and
    it is the one form that needs no guessing at all.
    """
    out: list[str] = []
    seen: set[str] = set()

    def add(token: str) -> None:
        t = token.strip()
        if len(t) < 2 or t.lower() in seen:
            return
        seen.add(t.lower())
        out.append(t)

    for email in EMAIL_IN_QUESTION.findall(question):
        add(email)
    for first, second in CAPITALIZED_RUN.findall(question):
        if first.lower() in NOT_A_NAME:
            continue
        # Try the full run before the bare first name: "Priya Sharma" is a more
        # specific reference than "Priya", and `search_persons` is a substring
        # match, so the longer query narrows rather than widens.
        if second and second.lower() not in NOT_A_NAME:
            add(f"{first} {second}")
        add(first)
    return out[:MAX_SUBJECT_CANDIDATES]


async def resolve_question_subject(
    db: AsyncSession, org_id: uuid.UUID, question: str
) -> SubjectResolution:
    """Ask the EXISTING identity graph who this task is about.

    Org-scoped by `search_persons`, which filters on `org_id` in every statement
    it runs — the same call the scheduling branch already makes, so a person is
    resolved identically whether the founder asks about them or an agent does. No
    second identity system, no second notion of who someone is.

    Best-effort: identity resolution failing must cost the bundle its entity
    scoping, never the bundle. Returns an empty resolution on any error.
    """
    candidates = subject_candidates(question)
    if not candidates:
        return SubjectResolution()
    try:
        for token in candidates:
            result = await search_persons(db, org_id, token)
            if not result.matches:
                continue
            if result.single is not None:
                return SubjectResolution(match=result.single, query=token)
            # Several people carry this name. Do NOT pick one.
            return SubjectResolution(candidates=result.matches, query=token)
    except Exception:  # noqa: BLE001 — scoping is never worth failing the turn
        logger.exception("context: identity resolution unavailable org=%s", org_id)
        metrics.incr("context.subject_resolution.failed")
        await reset_session(db)
    return SubjectResolution()


def retrieval_query(question: str, subject: SubjectResolution) -> str:
    """The text RETRIEVAL embeds — the task, plus who it is actually about.

    The caller's words are what is shown (`GroundedContext.question` is
    unchanged); this is only what the vector search looks for. The difference
    matters because memory records contain a person's ADDRESS far more often than
    their display name — a thread is `from: abhishek@acme.com`, not "Abhishek" —
    so embedding the bare question searched for a string the corpus barely holds.

    Appends the canonical aliases the identity graph resolved. Nothing is
    invented: every token added is an alias already recorded for that person.
    """
    if subject.match is None:
        return question
    aliases = [a for a in (*subject.match.emails, *subject.match.display_names) if a]
    if not aliases:
        return question
    return f"{question} {' '.join(aliases)}"


# ── Context sufficiency for an entity-centric task ──────────────────────────
#
# MEASURED, NOT ASSUMED. Against the live organization, all eight entity-centric
# questions in the 7.5.1 probe produced a byte-identical 53-fact bundle at
# ~1450/1500 tokens, of which only 9-11 facts mentioned the subject at all — and
# the first seven read were org-wide insights that never named them. The
# question-dependent part was the 8 retrieved memory facts: 15% of the bundle,
# sitting behind 18 derived facts.
#
# So the gate asks the only question that matters for an entity task: did we
# actually admit anything ABOUT this person? Not a score, not a confidence — a
# count of facts whose text names one of their known aliases.
MIN_SUBJECT_FACTS = 4
# Which entity types we fetch by id when it did not. Person is who they are;
# relationship is the interaction history and is the type measurably unreachable
# by ranking; commitment is what is outstanding between us. Meetings and threads
# already retrieve well by similarity (they carry names and subjects).
EXPANSION_ENTITY_TYPES = ("person", "relationship", "commitment")
# Newest N per entity type. Four covers "the recent moments" without letting one
# heavily-observed counterparty flood a 1500-token budget.
EXPANSION_PER_ENTITY = 4


def subject_terms(subject: SubjectResolution | None) -> tuple[str, ...]:
    """Lower-cased alias strings that count as naming this person."""
    if subject is None or subject.match is None:
        return ()
    return tuple(
        a.lower()
        for a in (*subject.match.emails, *subject.match.display_names)
        if a
    )


def mentions_subject(text: str, terms: tuple[str, ...]) -> bool:
    low = (text or "").lower()
    return any(t in low for t in terms)


async def entity_expansion(
    db: AsyncSession, org_id: uuid.UUID, subject: SubjectResolution
) -> list[Grounding]:
    """ONE targeted, deterministic hop: this person's own memory records.

    Runs only when the sufficiency gate says retrieval did not find enough about
    the subject, so a task retrieval already answered costs nothing extra.
    Exactly one additional round trip, never a loop.

    NOT A SECOND RETRIEVAL SYSTEM. There is no embedding, no scoring and no
    ranking here — the entity ids are COMPUTED from the aliases the identity
    graph already resolved (`memory.ids.entity_id` is uuid5 over organization +
    type + natural key) and the rows are read through Memory's own
    `records_for_entities`. Same table, same org filter, same provenance.

    Every fact carries its real `memory_id` as the citation source and its
    signal/integration lineage in `lineage`, so an expanded fact is exactly as
    traceable as a retrieved one — signal_event -> memory_record -> citation.
    """
    from navis.memory.ids import entity_id as compute_entity_id
    from navis.memory.lineage import records_for_entities
    # The pure, model-free renderer that produced every `source_text` in the
    # corpus. Called HERE rather than inside Memory because a pipeline domain may
    # not import `navis.ml`; orchestration is the composition root that may
    # touch both — the same argument `state_groundings` makes.
    from navis.ml.embeddings.service import memory_text

    aliases = [a for a in subject.match.emails if a]
    if not aliases:
        return []
    ids = [
        compute_entity_id(org_id, etype, alias)
        for alias in aliases
        for etype in EXPANSION_ENTITY_TYPES
    ]
    try:
        records = await records_for_entities(
            db, org_id, ids, per_entity_limit=EXPANSION_PER_ENTITY
        )
    except Exception:  # noqa: BLE001 — expansion is never worth failing the turn
        logger.exception("context: entity expansion unavailable org=%s", org_id)
        metrics.incr("context.entity_expansion.failed")
        await reset_session(db)
        return []

    metrics.incr("context.entity_expansion.facts", value=len(records))
    out: list[Grounding] = []
    for r in records:
        text = memory_text(r)
        if not text:
            continue  # nothing semantic to say; never render an empty fact
        out.append(
            Grounding(
                # Retrieved evidence, because that is what it is: a memory
                # record. It reaches the consumer in the evidence band beside the
                # vector hits, not in the conclusion band.
                kind="memory",
                source=str(r.memory_id),
                text=text,
                occurred_at=r.event_timestamp,
                lineage=(
                    *((f"signal:{r.source_signal_event_id}",)
                      if r.source_signal_event_id else ()),
                    # The connected account this fact was captured on — the ONLY
                    # honest source of a provider claim. Absent when the record
                    # did not record one; never inferred from the text.
                    *((f"integration:{r.source_integration_id}",)
                      if r.source_integration_id else ()),
                ),
            )
        )
    return out


# ── The composer ────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class ContextLayers:
    """Which truth layers the CALLER is entitled to receive.

    NOT AN AUTHORIZATION DECISION — the decision has already been made by the
    time one of these arrives. This is the RESULT of it, expressed so the
    composer can honour it without knowing what a permission is. `/chat` builds
    the all-true default because a human session's gate is a capability on the
    whole surface; the Developer boundary builds one per credential.

    Memory is deliberately absent, because it is the FLOOR rather than a layer:
    a bundle with no retrieved memory is not a degraded context, it is the
    morning-brief projection that `GET /developer/context` already serves. A
    caller that may not read memory has no business on this route at all, so
    that check belongs at the gate and not here.
    """

    state: bool = True
    reasoning: bool = True
    intelligence: bool = True
    learning: bool = True


@dataclass(frozen=True, slots=True)
class ComposedContext:
    """A GroundedContext, plus what it describes and how it was assembled.

    The extra fields exist because a bundle alone cannot answer "is this
    current?" or "why is there no state here?", and a consumer that cannot ask
    those will present a cold or stale organization as a complete one.
    """

    context: GroundedContext
    #: The State + Reasoning groundings AS ASSEMBLED, before the token budget
    #: admitted or dropped any of them.
    #:
    #: Not derivable from `context.facts`: the budget may have dropped some, and
    #: a caller composing its own fallback bundle needs what was available
    #: rather than what survived. `/chat` uses exactly this when retrieval found
    #: nothing and it falls back to raw workspace evidence.
    derived: list[Grounding]
    #: The memory record State had folded to. None when State was unavailable.
    state_watermark: str | None
    #: State's logical time — WHEN NAVIS LAST LEARNED SOMETHING.
    as_of: datetime | None
    #: WHEN THE NEWEST THING NAVIS KNOWS ABOUT ACTUALLY HAPPENED.
    #:
    #: Published beside `as_of` rather than instead of it because the two are
    #: both true and answer different questions, and a consumer needs both to
    #: tell a quiet organization from a broken pipeline: `as_of` moving while
    #: this stands still is precisely the signature of ingestion having stopped.
    #: None when State was unavailable, or when nothing has been folded yet.
    newest_fact_at: datetime | None
    #: Whether the State/Reasoning layers made it into the bundle at all.
    state_available: bool
    #: Who the task was resolved to, when the identity graph found exactly one.
    subject: SubjectResolution | None
    #: True when retrieval found too little about the subject and the one
    #: expansion hop ran. Operationally the difference between "similarity
    #: answered this" and "we had to fetch the person by id".
    expanded: bool
    #: The layers this bundle was ALLOWED to draw on. A consumer must be able to
    #: tell "the organization has no insights" from "this caller may not read
    #: them" — otherwise a narrowly-scoped credential reads a thin bundle as a
    #: complete picture of the company.
    layers: "ContextLayers" 


async def build_grounded_context(
    db: AsyncSession,
    org_id: uuid.UUID,
    task: str,
    *,
    subject: SubjectResolution | None = None,
    read_only: bool = False,
    layers: ContextLayers | None = None,
    k: int = 8,
    token_budget: int = CONTEXT_TOKEN_BUDGET,
    since: datetime | None = None,
    until: datetime | None = None,
    now: datetime | None = None,
    sessions: SessionFactory | None = None,
) -> ComposedContext:
    """Assemble the organizational context a task requires. THE one composer.

    Every layer is read through the service that owns it, and nothing here
    re-derives, re-ranks or re-scores what another layer already decided:

        Memory        MemoryRetriever            embed -> pgvector -> rerank
        State         StateService               cached projection
        Context       scope_for(MorningBrief)    org-wide slice, pure
        Reasoning     run_reasoning_v2           pure, no I/O
        Intelligence  IntelligenceRepository     current generation only
        Learning      learning_snapshot          Learning's own contract
        Identity      search_persons             the existing identity graph

    `subject` may be supplied by a caller that has already resolved it (the chat
    path resolves it early to decide whether to answer at all); when omitted it
    is resolved here. Passing it in avoids a second identity lookup, never a
    different answer.

    `read_only` is forwarded to `state_groundings` and is the ONLY behavioural
    difference between the two callers. See that function.

    WHAT THIS DOES NOT DO: call a model. The bundle is the product. `/chat` hands
    the result to a generator afterwards; the Developer boundary returns it. A
    composer that generated prose would make every consumer pay for an opinion
    they may not want, and would put a model between an agent and its own
    organization's facts.
    """
    layers = layers or ContextLayers()
    # CONCURRENCY IS GATED ON `read_only`, NOT OFFERED AS A CHOICE.
    #
    # Separate sessions are separate transactions. On the read-only path that
    # costs nothing — every branch reads and none of them folds, rebuilds or
    # disposes — so three transactions observe the same committed database as
    # one would. `/chat` runs with `read_only=False`, where `state_groundings`
    # may REBUILD a corrupt or absent snapshot; splitting a write across
    # sessions it cannot see would be a different operation, not a faster one.
    # So the Application keeps the sequential path it has always had, and the
    # flag that already means "this call only reads" is the one that decides.
    #
    # AND ONLY WHEN THERE IS SOMETHING TO OVERLAP. Retrieval is the one branch
    # that always runs; Learning and Intelligence are the ones that make the
    # builder's gather a gather rather than a single await. With neither of them
    # enabled — a credential holding `view_memory` and `view_state` and nothing
    # else — concurrency buys no parallelism and still pays for the machinery,
    # measured at ~1s slower over five interleaved runs. Fan out when the work
    # fans out, and not before.
    concurrent = (
        sessions is not None
        and read_only
        and (layers.intelligence or layers.learning)
    )

    # Subject resolution and the State read share no data, so they no longer
    # queue behind one another. Both are needed before the builder runs:
    # `subject` shapes the retrieval query, `grounded` supplies the derived
    # layers it composes against.
    async def _subject() -> SubjectResolution:
        if subject is not None:
            return subject
        if not concurrent:
            return await resolve_question_subject(db, org_id, task)
        async with sessions() as s:  # type: ignore[misc]
            return await resolve_question_subject(s, org_id, task)

    async def _grounded() -> StateGrounding:
        # State is read ONCE and produces both derived layers, so a caller
        # entitled to neither must not pay for the read at all.
        if not (layers.state or layers.reasoning):
            return StateGrounding([], None, None, None, available=False)
        if not concurrent:
            return await state_groundings(db, org_id, read_only=read_only)
        async with sessions() as s:  # type: ignore[misc]
            return await state_groundings(s, org_id, read_only=read_only)

    if concurrent:
        with timed("context.subject_state_concurrent"):
            subject, grounded = await asyncio.gather(_subject(), _grounded())
    else:
        subject = await _subject()
        grounded = await _grounded()
    derived = grounded.groundings
    if not layers.reasoning:
        derived = [g for g in derived if g.kind != REASONING_KIND]
    if not layers.state:
        derived = [g for g in derived if g.kind != CONTEXT_KIND]
    terms = subject_terms(subject)
    query = retrieval_query(task, subject) if subject.match is not None else None

    # ── SUFFICIENCY GATE ─────────────────────────────────────────────────
    # Runs INSIDE the single build, on what retrieval actually returned, so the
    # decision costs no extra embedding and no second vector scan.
    #
    # It judges the RETRIEVAL SLICE only. State, Reasoning, Intelligence and
    # Learning are org-wide by construction and present on every turn, so
    # counting them would declare every context sufficient and the gate would
    # never fire — which is precisely how a 53-fact bundle with 9 facts about the
    # subject looked "full".
    expanded = False

    async def _expand(retrieved: list[Grounding]) -> list[Grounding]:
        nonlocal expanded
        if not terms:
            return []
        found = sum(1 for f in retrieved if mentions_subject(f.text, terms))
        if found >= MIN_SUBJECT_FACTS:
            return []
        metrics.incr("context.insufficient")
        got = await entity_expansion(db, org_id, subject)  # type: ignore[arg-type]
        expanded = len(got) > 0
        logger.info(
            "context expanded org=%s retrieved_about_subject=%d expanded=%d",
            org_id, found, len(got),
        )
        return got

    context = await ContextBuilder(db).build(
        org_id, task, k=k, token_budget=token_budget,
        extra=derived,
        # The builder's own seams, used rather than filtered afterwards: a layer
        # the caller may not read is never FETCHED, so an unauthorized read does
        # not happen and then get discarded.
        include_learning=layers.learning,
        include_intelligence=layers.intelligence,
        since=since, until=until, now=now,
        retrieval_query=query,
        # What the task is about, so composition can put facts that name the
        # subject ahead of org-wide boilerplate.
        subject_terms=terms,
        expand=_expand if terms else None,
        # Retrieval, Learning and Intelligence concurrently, one session each.
        sessions=sessions if concurrent else None,
    )
    return ComposedContext(
        context=context,
        derived=derived,
        state_watermark=grounded.watermark,
        as_of=grounded.as_of,
        newest_fact_at=grounded.newest_fact_at,
        state_available=grounded.available,
        subject=subject,
        expanded=expanded,
        layers=layers,
    )
