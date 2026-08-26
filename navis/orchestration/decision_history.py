"""The decision lifecycle, as a READ MODEL over canonical data.

    Decision -> Human choice -> Execution -> Outcome -> Learning

WHY THIS IS A COMPOSITION ROOT AND NOT A DOMAIN. The five stages live in four
bounded contexts that may not import each other — Decision owns `daily_briefs`
and `decision_dismissals`, Execution owns `execution_requests`, Observation owns
`outcomes`, and Learning's tables are readable only through
`navis.learning.metrics`. Joining them is exactly the job `navis/orchestration`
already does for the write path (`brief.py` reads State for Reasoning and
Adaptive policy for Decision). This is the read-side seam at the same boundary.

NOTHING IS PERSISTED. There is no history table, no new decision store, no
duplicate of `daily_briefs.decisions`. Every value below is read from a
canonical row and folded in memory, so replaying the same data rebuilds the same
history and dropping this module loses nothing.

── THE EPISODE KEY, and why it is not `decision_id` ──────────────────────────

`compute_decision_id` hashes (org, reasoning_version, state_watermark, signal),
so THE SAME UNRESOLVED COMMITMENT GETS A NEW DECISION ID EVERY TIME STATE MOVES.
Live, one commitment-slipping signal has produced 18 distinct decision ids
across 18 briefs. Keying a history on `decision_id` would therefore render 18
rows for one thing that happened once — the "meaningless duplicate history"
failure mode — and would also fail to connect a dismissal recorded at one
watermark to an execution created at another.

`compute_signal_id` hashes (org, reasoning_version, engine, engine_version,
type, entity, dedup_key) and NOT the watermark, so a reasoning conclusion keeps
ONE id for as long as it keeps being true. It is already the identity every
downstream layer records: `execution_requests.reasoning_signal_ids`,
`outcomes.reasoning_signal_ids`, `learning_events.reasoning_signal_ids`, and
`learning_aggregates(dimension='signal')` — Learning has been aggregating on it
all along. So the key is not new; it is the one the data already uses.

FALLBACK TO `decision_id`, for decisions that never came from Reasoning. A chat
proposal mints its own decision id and carries `reasoning_signal_ids: []`; that
id is not watermark-derived, so it is stable on its own and each such proposal
is honestly its own one-shot episode. `episode_key_kind` says which rule applied
rather than leaving a client to guess from the shape of a uuid.

`entity_type` / `entity_id` are PUBLISHED but deliberately not the key: they are
null for org-wide signals (`meeting_overload`), and two different engines firing
on one entity are two different conclusions a founder must be able to tell apart.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from navis.decisions.explanation import explain
from navis.decisions.service import DecisionService
from navis.execution.proposer import decision_action
from navis.execution.repository import ExecutionRepository
from navis.execution.status import ExecutionStatus
from navis.execution.summary import execution_summary
from navis.learning.metrics import learning_events_for
from navis.observation.repository import OutcomeRepository
from navis.observation.router import serialize_outcome

# How many briefs one history call folds. A brief is rebuilt whenever the state
# watermark moves, so this bounds a row count that grows with SYNC FREQUENCY
# rather than with the organization. Reported on the envelope as
# `scanned.briefs`, so a truncated history is visible rather than silent.
MAX_BRIEFS_SCANNED = 200

# Per-stage read bounds. Each is a page of the newest rows in its own log.
MAX_EXECUTIONS_SCANNED = 200
MAX_OUTCOMES_SCANNED = 200
MAX_DISMISSALS_SCANNED = 500

# How many episodes one response carries.
DEFAULT_EPISODE_LIMIT = 50


# ── Stage vocabularies ───────────────────────────────────────────────────────
#
# THREE STATES THAT MUST NEVER COLLAPSE INTO ZERO, at every stage:
#
#   <a real state>  something happened and this is what it was
#   "pending"       this stage is expected and has not happened yet
#   "not_applicable" this stage will never happen for this episode
#
# The third is the one a count destroys. A `risk` decision maps to no action
# type, so it has no execution and never will; reporting that as "0 executions"
# is indistinguishable from a reply that was proposed and never sent. Both
# render as the same number and only one of them is a problem.


class ChoiceStage:
    PENDING = "pending"        # still on the brief, nobody has decided
    DISMISSED = "dismissed"    # the founder said this did not need them
    ACTED = "acted"            # a human approved an execution for it


class ExecutionStage:
    NOT_APPLICABLE = "not_applicable"  # `decision_action` returns None: nothing to run
    PENDING = "pending"                # actionable, no execution row yet
    # Otherwise: the ExecutionRequest's own status, verbatim.


class OutcomeStage:
    OBSERVED = "observed"
    PENDING = "pending"
    NONE = "none"


class LearningStage:
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    PENDING = "pending"
    NONE = "none"


# `learning_events.result` -> the stage word. Learning's vocabulary
# (`LearningResult`), mapped once, here, rather than re-derived per surface.
_RESULT_STAGE = {
    "success": LearningStage.POSITIVE,
    "failure": LearningStage.NEGATIVE,
    "neutral": LearningStage.NEUTRAL,
}


@dataclass
class _Episode:
    """One reasoning conclusion, and everything that has happened about it."""

    key: str
    key_kind: str
    signal_id: str | None = None
    signal_type: str | None = None
    category: str | None = None
    entity_type: str | None = None
    entity_id: str | None = None
    latest_decision: dict[str, Any] | None = None
    latest_decision_id: str | None = None
    decision_ids: set[str] = field(default_factory=set)
    first_seen_at: datetime | None = None
    last_seen_at: datetime | None = None
    is_current: bool = False
    dismissal: dict[str, Any] | None = None
    executions: list[Any] = field(default_factory=list)
    outcomes: list[Any] = field(default_factory=list)
    learning: list[Any] = field(default_factory=list)


def _episode_key(
    decision_id: str | None, reasoning_signal_ids: Any
) -> tuple[str, str] | None:
    """(key, key_kind) for one lineage-bearing row. See the module docstring.

    PURE, and the single place the rule is written. Execution, Observation,
    Learning and Decision rows all reach this function, so a row cannot land in
    one episode on one code path and another on a second.
    """
    for raw in reasoning_signal_ids or ():
        sid = str(raw).strip()
        if sid:
            return sid, "reasoning_signal"
    if decision_id:
        return str(decision_id), "decision"
    return None


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def _sort_time(value: datetime | None) -> datetime:
    """Sort key that tolerates a null and a naive timestamp alike.

    `daily_briefs.as_of` is nullable (a brief built before the org's first sync
    records no moment) and Postgres hands back tz-aware values while a fixture
    may not, so comparing them directly raises. Null sorts oldest.
    """
    if value is None:
        return datetime.min.replace(tzinfo=timezone.utc)
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def _fold_briefs(briefs: list[Any]) -> tuple[dict[str, _Episode], dict[str, str]]:
    """Collapse every brief decision into episodes. Newest brief first.

    Returns the episodes and a `decision_id -> episode key` index — the index is
    what lets a dismissal, whose `decision_ref` is one watermark's decision id,
    reach the episode that decision belongs to.
    """
    episodes: dict[str, _Episode] = {}
    by_decision_id: dict[str, str] = {}

    for position, brief in enumerate(briefs):
        as_of = brief.as_of
        for decision in brief.decisions or []:
            decision_id = decision.get("decision_id")
            keyed = _episode_key(decision_id, decision.get("reasoning_signal_ids"))
            if keyed is None:
                continue
            key, key_kind = keyed
            episode = episodes.get(key)
            if episode is None:
                # FIRST SIGHTING IS THE NEWEST, because briefs arrive newest
                # first. So the decision snapshot an episode publishes is the
                # most recent one — the values a founder would see on Today —
                # and never a stale copy from an older watermark.
                episode = _Episode(
                    key=key,
                    key_kind=key_kind,
                    signal_id=key if key_kind == "reasoning_signal" else None,
                    signal_type=decision.get("signal_type"),
                    category=decision.get("category"),
                    entity_type=decision.get("entity_type"),
                    entity_id=decision.get("entity_id"),
                    latest_decision=decision,
                    latest_decision_id=str(decision_id) if decision_id else None,
                    last_seen_at=as_of,
                    is_current=position == 0,
                )
                episodes[key] = episode
            if decision_id:
                episode.decision_ids.add(str(decision_id))
                by_decision_id[str(decision_id)] = key
            # Briefs descend, so every later visit is older.
            episode.first_seen_at = as_of
    return episodes, by_decision_id


def _attach(
    episodes: dict[str, _Episode],
    by_decision_id: dict[str, str],
    rows: list[Any],
    bucket: str,
    *,
    make_episode: bool,
) -> None:
    """File lineage-bearing rows onto their episodes.

    `make_episode` is True for Execution and Observation because an execution
    can exist for a decision no surviving brief still carries — a chat proposal,
    or a brief older than `MAX_BRIEFS_SCANNED`. Dropping those rows would hide
    real, completed work from the only page that reports it. It is False for
    Learning: a learning event always follows an outcome, so its episode has
    already been created by the outcome pass, and inventing one here would
    produce an episode with a learning result and no fact behind it.
    """
    for row in rows:
        keyed = _episode_key(
            str(row.decision_id) if row.decision_id else None,
            row.reasoning_signal_ids,
        )
        if keyed is None:
            continue
        key, key_kind = keyed
        episode = episodes.get(key)
        if episode is None and key_kind == "decision":
            # A decision id that IS on a brief resolves through the index; the
            # episode is keyed on that brief's signal, not on the decision id.
            mapped = by_decision_id.get(key)
            if mapped is not None:
                episode = episodes.get(mapped)
        if episode is None:
            if not make_episode:
                continue
            episode = _Episode(
                key=key,
                key_kind=key_kind,
                signal_id=key if key_kind == "reasoning_signal" else None,
            )
            episodes[key] = episode
            if key_kind == "decision":
                by_decision_id.setdefault(key, key)
        if row.decision_id:
            episode.decision_ids.add(str(row.decision_id))
        getattr(episode, bucket).append(row)


def _execution_stage(
    episode: _Episode, action: Any | None
) -> tuple[str, dict[str, Any] | None]:
    """Where this episode stands in Execution, and the row that says so.

    THE FURTHEST-ALONG EXECUTION WINS, not the most recently created: a retried
    obligation can hold one EXECUTED row and one later FAILED attempt, and
    reporting the attempt would tell a founder nothing was sent when something
    was.

    `action` is `decision_action`'s answer — the SAME function `/brief/today`
    calls to decide whether to offer an Execute affordance — so a card that
    offers no action and a history that reports `not_applicable` cannot
    disagree.
    """
    if episode.executions:
        terminal = [
            e for e in episode.executions
            if e.status == ExecutionStatus.EXECUTED.value
        ]
        chosen = (terminal or sorted(
            episode.executions, key=lambda e: _sort_time(e.created_at)
        ))[-1]
        return chosen.status, execution_summary(chosen)
    if episode.latest_decision is None:
        # No stored decision to ask about — an execution-only episode whose
        # every row has been read above, so this branch is unreachable in
        # practice. Fail toward "we cannot say" rather than "nothing to do".
        return ExecutionStage.PENDING, None
    if action is None:
        return ExecutionStage.NOT_APPLICABLE, None
    return ExecutionStage.PENDING, None


def _outcome_stage(episode: _Episode, execution_stage: str) -> str:
    """Whether the world has answered yet.

    `observed` — an outcome is recorded.
    `pending`  — something that PRODUCES outcomes has already happened (an
                 execution ran, or a human recorded a verdict) and the outcome
                 has not landed. One is expected.
    `none`     — no outcome exists and none is in flight, either because no
                 action is possible for this decision or because nobody has
                 taken one yet.

    The load-bearing distinction is `pending` vs `none`: "we are waiting on the
    world" versus "nothing is on its way". NEITHER IS ZERO, and a count cannot
    express either — which is how the previous surface rendered a finding with
    nothing to run and a reply that was sent but unanswered identically.
    """
    if episode.outcomes:
        return OutcomeStage.OBSERVED
    if episode.dismissal is not None:
        return OutcomeStage.PENDING
    if execution_stage in (ExecutionStage.NOT_APPLICABLE,):
        return OutcomeStage.NONE
    if execution_stage == ExecutionStage.PENDING:
        return OutcomeStage.NONE
    return OutcomeStage.PENDING


def _learning_stage(episode: _Episode, outcome_stage: str) -> str:
    """What the organization took from this, in Learning's own vocabulary.

    The NEWEST learning event decides, and its `result` is mapped through
    `_RESULT_STAGE` — Learning's classification, never a second judgement made
    here. An episode with several events reports the latest; the counts for all
    of them are on `/learning/summary` under this episode's signal key.
    """
    if episode.learning:
        newest = sorted(episode.learning, key=lambda e: _sort_time(e.occurred_at))[-1]
        return _RESULT_STAGE.get(newest.result, LearningStage.NEUTRAL)
    if outcome_stage == OutcomeStage.OBSERVED:
        return LearningStage.PENDING
    if outcome_stage == OutcomeStage.PENDING:
        return LearningStage.PENDING
    return LearningStage.NONE


def _last_activity(episode: _Episode) -> datetime:
    """The most recent thing that happened anywhere on this episode."""
    times = [_sort_time(episode.last_seen_at)]
    times += [_sort_time(e.created_at) for e in episode.executions]
    times += [_sort_time(o.occurred_at) for o in episode.outcomes]
    times += [_sort_time(le.occurred_at) for le in episode.learning]
    if episode.dismissal is not None:
        times.append(_sort_time(episode.dismissal["_at"]))
    return max(times)


def _serialize(episode: _Episode) -> dict[str, Any]:
    """One episode, as the wire sees it."""
    # ONE derivation, shared by the stage and the explanation — exactly as
    # `brief_router._with_executions` does it. Computing it twice is how the two
    # surfaces come to disagree about whether Navis can act on a decision.
    action = decision_action(episode.latest_decision) if episode.latest_decision else None
    execution_stage, execution = _execution_stage(episode, action)
    outcome_stage = _outcome_stage(episode, execution_stage)
    learning_stage = _learning_stage(episode, outcome_stage)

    # The founder-facing projection, from the SAME pure function `/brief/today`
    # uses, over a decision enriched with `action` the SAME way — so a history
    # card and a current card read identically, down to `can_navis_act`.
    # None for an episode with no stored decision (a chat proposal), which is
    # honest: there is no reasoning conclusion to explain.
    explanation = (
        explain({**episode.latest_decision, "action": action.to_dict() if action else None})
        if episode.latest_decision
        else None
    )

    dismissal = episode.dismissal
    if dismissal is not None:
        choice_stage = ChoiceStage.DISMISSED
    elif any(e.approved_at is not None for e in episode.executions):
        choice_stage = ChoiceStage.ACTED
    else:
        choice_stage = ChoiceStage.PENDING

    return {
        # ── Identity ──────────────────────────────────────────────────────
        "episode_key": episode.key,
        "episode_key_kind": episode.key_kind,
        "signal_id": episode.signal_id,
        "signal_type": episode.signal_type,
        "category": episode.category,
        "entity_type": episode.entity_type,
        "entity_id": episode.entity_id,
        # HOW MANY WATERMARK-SCOPED DECISION IDS THIS ONE CONCLUSION HAS HELD.
        # Published because it is the honest answer to "how long has this been
        # on my plate?" and because it is the number that would have become
        # duplicate history rows under a decision-id key.
        "decision_instances": len(episode.decision_ids),
        "latest_decision_id": episode.latest_decision_id,
        # Whether the newest brief still raises this. False means Reasoning
        # stopped concluding it — resolved, aged out, or superseded.
        "is_current": episode.is_current,
        "first_seen_at": _iso(episode.first_seen_at),
        "last_seen_at": _iso(episode.last_seen_at),
        # ── Presentation, verbatim from `decisions.explanation` ───────────
        "explanation": explanation,
        # ── The five stages ───────────────────────────────────────────────
        "stages": {
            "decision": "open" if episode.is_current else "closed",
            "choice": choice_stage,
            "execution": execution_stage,
            "outcome": outcome_stage,
            "learning": learning_stage,
        },
        # ── The facts behind each stage ───────────────────────────────────
        "choice": (
            None
            if dismissal is None
            else {k: v for k, v in dismissal.items() if not k.startswith("_")}
        ),
        # The newest/furthest-along execution, in the same shape `/brief/today`
        # attaches to a decision (`execution.summary.execution_summary`).
        "execution": execution,
        "outcomes": [serialize_outcome(o) for o in episode.outcomes],
        "learning": [
            {
                "learning_event_id": le.learning_event_id,
                "outcome_id": le.outcome_id,
                "outcome_type": le.outcome_type,
                # Both axes, never one — see `learning.attribution`.
                "outcome_surface": le.outcome_surface,
                "result": le.result,
                "occurred_at": _iso(le.occurred_at),
            }
            for le in sorted(episode.learning, key=lambda e: _sort_time(e.occurred_at))
        ],
    }


async def decision_history(
    db,
    org_id: uuid.UUID,
    *,
    limit: int = DEFAULT_EPISODE_LIMIT,
) -> dict[str, Any]:
    """This organization's decision lifecycle, newest activity first.

    Five bounded reads, each through the boundary that owns its table:
    `DecisionService` for briefs and dismissals, `ExecutionRepository` for
    executions, `OutcomeRepository` for outcomes, and `learning.metrics` for
    learning events. No SELECT in this module touches a table it does not
    reach through one of those.

    ORG-SCOPED AT EVERY READ. `org_id` is passed to all five and none of them
    accepts a wider scope, so an episode cannot be assembled from two tenants'
    rows even if their ids collided.

    READ-ONLY. Nothing here writes, transitions, rebuilds or proposes — in
    particular it never calls `build_brief`, so opening the page cannot
    recompute the pipeline.
    """
    decisions = DecisionService(db)
    briefs = await decisions.recent_briefs(org_id, limit=MAX_BRIEFS_SCANNED)
    episodes, by_decision_id = _fold_briefs(briefs)

    executions = list(
        await ExecutionRepository(db).recent_for_org(org_id, limit=MAX_EXECUTIONS_SCANNED)
    )
    outcomes = list(
        await OutcomeRepository(db).recent(org_id, limit=MAX_OUTCOMES_SCANNED)
    )
    learning = list(await learning_events_for(db, org_id))

    _attach(episodes, by_decision_id, executions, "executions", make_episode=True)
    _attach(episodes, by_decision_id, outcomes, "outcomes", make_episode=True)
    _attach(episodes, by_decision_id, learning, "learning", make_episode=False)

    # Dismissals last: `decision_ref` is opaque, so it is resolved against BOTH
    # indexes — the decision-id index built from the briefs, and the episode
    # keys themselves (a client may legitimately dismiss by signal id, which the
    # dismiss route's contract allows).
    for row in await decisions.dismissals(org_id, limit=MAX_DISMISSALS_SCANNED):
        ref = str(row.decision_ref)
        key = by_decision_id.get(ref) or (ref if ref in episodes else None)
        if key is None:
            # A dismissal of a decision no scanned brief carries. Recorded and
            # kept — it is still real feedback — but it has no episode to join,
            # and attaching it to a neighbouring one would attribute a founder's
            # verdict to a conclusion they never saw.
            continue
        episode = episodes[key]
        # Newest dismissal wins; `dismissals` returns them newest first.
        if episode.dismissal is None:
            episode.dismissal = {
                "kind": "dismissed",
                "reason": row.reason,
                "decision_ref": ref,
                "at": _iso(row.created_at),
                "_at": row.created_at,
            }

    ordered = sorted(episodes.values(), key=_last_activity, reverse=True)
    page = ordered[:limit]

    return {
        "episodes": [_serialize(e) for e in page],
        "total": len(page),
        # HOW MANY EPISODES EXIST versus how many this page carries. The gap is
        # the honest "and N more"; a client that cannot see it reports the page
        # size as a count.
        "available": len(ordered),
        # WHAT THIS ANSWER IS A WINDOW ONTO. Each bound is reported with what it
        # actually read, so a truncated history is visible on the wire rather
        # than looking like a complete one.
        "scanned": {
            "briefs": len(briefs),
            "brief_limit": MAX_BRIEFS_SCANNED,
            "executions": len(executions),
            "outcomes": len(outcomes),
            "learning_events": len(learning),
        },
    }
