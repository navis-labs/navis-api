"""The AGENT EXECUTION vocabulary — Task, Run, Step, Tool, Model, Outcome.

WHAT THIS DOMAIN IS, AND WHAT IT IS NOT. An agent Run records what an EXTERNAL
agent did: what it was asked, which context it was handed, which tools it called,
which model generated its answer, and what came of it. Navis does not execute any
of it. The agent executes; this is the ledger it writes to.

THAT MAKES IT A DIFFERENT DOMAIN FROM `navis/execution`, AND THE TWO MUST NOT
MERGE. `execution_requests` is Navis acting on the organization's behalf —
`send_email`, `schedule_meeting` — through the decision -> execution -> outbox ->
observation pipeline, with providers it owns and an approval model it enforces.
An agent run has no proposer, no obligation, no provider and no approval: it is
an account of work done elsewhere. They share the word "execution" and nothing
else. Collapsing them would put an external agent's self-reported timeline into
the aggregate the application trusts to decide what it has already done.

THE CHAIN, AND WHY EACH LINK IS SEPARATE:

    Agent      service_accounts       WHO acted. Already exists; no new identity.
      |
    Task       agent_tasks            WHAT was asked. Survives its attempts.
      |
    Run        agent_runs             ONE attempt. A task may have many.
      |
    Step       agent_run_steps        The ordered, observable timeline.
      |
    Outcome    agent_run_outcomes     What came of it. A run can succeed and
                                      still produce a bad outcome.

Task is separate from Run so a retry does not lose the logical identity of the
work. Outcome is separate from Run because "the process finished" and "the result
was any good" are different questions, and equating them makes the second one
unaskable — which is precisely the question Learning will need later.

NOTHING HERE IS AN AUTHORIZATION INPUT. Like `agents.py`, every vocabulary below
is descriptive. No gate reads a step type, a tool kind or a run status. The only
authorization facts are the Permission and the Capability the routes name, and
the organization the credential resolves to.

Pure module — no DB, no imports from other domains.
"""
from __future__ import annotations

import enum


class TaskStatus(str, enum.Enum):
    """The logical state of the WORK, not of any one attempt.

    A task is `open` while more attempts may happen, and reaches a terminal state
    when the operator or the agent says the work itself is finished. It is
    deliberately NOT derived from its runs: three failed runs do not make a task
    failed — the fourth may succeed — and a task closed by hand while a run is
    still in flight is a real situation the model has to hold.
    """

    OPEN = "open"
    COMPLETED = "completed"
    #: Abandoned without completing. Distinct from `completed`: it records that
    #: nobody is going to attempt this again, which is a different fact from
    #: "the last attempt failed".
    ABANDONED = "abandoned"


class RunStatus(str, enum.Enum):
    """The lifecycle of ONE attempt.

    `queued` and `running` are live; the other three are terminal. The split
    between `failed` and `cancelled` is worth keeping: a failure is the agent's
    or the environment's, a cancellation is a decision someone made, and a
    reliability figure that counts them together is not a reliability figure.
    """

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


#: Terminal states — a run in one of these may never transition again.
TERMINAL_RUN_STATUSES: frozenset[str] = frozenset(
    {RunStatus.COMPLETED.value, RunStatus.FAILED.value, RunStatus.CANCELLED.value}
)

#: The transitions the service permits. A run moves forward only.
#:
#: Written as DATA rather than as branches in the service so the whole state
#: machine is readable in one place and testable without a database. Anything not
#: listed is refused — including every terminal -> anything edge, which is what
#: stops a completed run being re-opened and its timeline rewritten.
ALLOWED_RUN_TRANSITIONS: dict[str, frozenset[str]] = {
    RunStatus.QUEUED.value: frozenset(
        {RunStatus.RUNNING.value, RunStatus.CANCELLED.value, RunStatus.FAILED.value}
    ),
    RunStatus.RUNNING.value: frozenset(
        {RunStatus.COMPLETED.value, RunStatus.FAILED.value, RunStatus.CANCELLED.value}
    ),
    RunStatus.COMPLETED.value: frozenset(),
    RunStatus.FAILED.value: frozenset(),
    RunStatus.CANCELLED.value: frozenset(),
}


def may_transition(current: str, target: str) -> bool:
    """Whether a run may move from `current` to `target`.

    An unknown `current` yields False rather than raising: the caller is a
    service handling a row that may predate a vocabulary change, and refusing the
    transition is the safe answer.
    """
    return target in ALLOWED_RUN_TRANSITIONS.get(current, frozenset())


class StepType(str, enum.Enum):
    """What KIND of thing happened at this point in the run.

    One closed set over one ordered table, rather than a table per kind. The Run
    Detail surface renders a single timeline — context fetched, tool called,
    model generated, outcome produced — and three tables would have to be merged
    and re-sorted on every read to produce it. The type is what lets a reader,
    and a query, tell them apart.
    """

    #: The agent asked Navis for organizational context. The provenance link.
    CONTEXT_FETCH = "context_fetch"
    #: The agent called something outside itself: MCP, REST, a database, a
    #: Navis capability, or a handoff to another agent.
    TOOL_CALL = "tool_call"
    #: A model generated something. Carries the economics columns.
    MODEL_GENERATION = "model_generation"
    #: The agent produced its result. The step that an outcome attaches to.
    OUTCOME = "outcome"


class StepStatus(str, enum.Enum):
    """A step is in flight, or it finished one of three ways."""

    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    #: Started and abandoned because the run ended around it — distinct from a
    #: failure, which is the step's own.
    SKIPPED = "skipped"


class ToolKind(str, enum.Enum):
    """What sort of tool was invoked. Coarse ON PURPOSE.

    Five buckets an operator can group by, not an open registry of every tool
    anyone will ever call — the tool's own NAME carries that, and a free-text
    kind cannot be counted. `OTHER` is first-class, not a fallback for
    misconfiguration.
    """

    MCP = "mcp"
    REST = "rest"
    DATABASE = "database"
    #: A Navis capability reached through the Developer API.
    NAVIS = "navis"
    #: Work delegated to another agent.
    HANDOFF = "handoff"
    OTHER = "other"


class ErrorCategory(str, enum.Enum):
    """Why a step failed, in terms an operator can act on.

    Closed so failures are groupable. The distinction that matters most is
    `refused` versus `error`: a permission or policy refusal is the platform
    working correctly and the agent asking for something it may not have, while
    an error is something broken. Counting them together hides both.
    """

    TIMEOUT = "timeout"
    #: 4xx-shaped: the agent asked for something it may not have.
    REFUSED = "refused"
    #: The agent sent something malformed.
    INVALID = "invalid"
    RATE_LIMITED = "rate_limited"
    #: 5xx-shaped: the tool or model broke.
    ERROR = "error"
    CANCELLED = "cancelled"


class CostBasis(str, enum.Enum):
    """WHERE a cost figure came from. The distinction is the whole point.

    A provider's API returns TOKEN COUNTS, not money. Every currency figure on a
    step is therefore produced by multiplying those counts against a pricing
    table someone wrote down — which makes it an ESTIMATE, however carefully it
    was computed. Recording it without saying so is how "our platform says this
    cost $4.12" ends up in front of somebody comparing it to an invoice.

    So a cost is stored with the basis it was derived on, and the two are not
    interchangeable:

      `ESTIMATED` — Navis or the agent multiplied usage by a pricing snapshot.
        `pricing_version` says WHICH snapshot, because a figure computed against
        last quarter's prices is not wrong, it is stale, and those are different
        problems with different fixes.

      `PROVIDER_REPORTED` — the provider itself returned a cost for this
        generation. Rare, and still not an invoice, but it is the provider's
        arithmetic rather than ours.

    NULL basis with a NULL cost means nothing was costed, which is honest. A cost
    WITHOUT a basis is refused at the edge: an uncited number is the one thing
    this vocabulary exists to prevent.
    """

    ESTIMATED = "estimated"
    PROVIDER_REPORTED = "provider_reported"


class OutcomeStatus(str, enum.Enum):
    """Whether the run's RESULT was any good.

    Deliberately not the run's status. A run that completes can still produce a
    wrong answer, and a run that fails can still have produced something useful
    before it did. `UNKNOWN` is the honest default: nothing evaluates agent
    outcomes yet, and recording an unjudged result as a success would poison the
    first metric anyone builds on this.
    """

    SUCCESS = "success"
    FAILURE = "failure"
    #: Produced something, but not what was asked for.
    PARTIAL = "partial"
    UNKNOWN = "unknown"


# ── Frozen sets for database CHECK constraints ──────────────────────────────
# Mirrors `agents.py`: the edge validates, and so does the database, because the
# edge is one of several writers over a table's life (a migration, a fixture, an
# operator's psql session) and a free-text status cannot be grouped later.

TASK_STATUSES: frozenset[str] = frozenset(s.value for s in TaskStatus)
RUN_STATUSES: frozenset[str] = frozenset(s.value for s in RunStatus)
STEP_TYPES: frozenset[str] = frozenset(s.value for s in StepType)
STEP_STATUSES: frozenset[str] = frozenset(s.value for s in StepStatus)
TOOL_KINDS: frozenset[str] = frozenset(k.value for k in ToolKind)
ERROR_CATEGORIES: frozenset[str] = frozenset(c.value for c in ErrorCategory)
OUTCOME_STATUSES: frozenset[str] = frozenset(s.value for s in OutcomeStatus)
COST_BASES: frozenset[str] = frozenset(b.value for b in CostBasis)


# ── Payload bounds ──────────────────────────────────────────────────────────
#
# A run ledger written by an external agent is an UNTRUSTED, UNBOUNDED input. If
# the platform stores whatever arrives, one agent looping on a large prompt fills
# a shared table, and the first symptom is somebody else's list endpoint timing
# out. These are the ceilings; `sanitize.py` enforces them.
#
# Deliberately small. This is a ledger of WHAT HAPPENED, not a transcript store:
# raw prompts and completions are the agent's to keep, and holding them here
# would make Navis the largest repository of its customers' model traffic for no
# capability it offers in return.

#: Longest task description accepted. Longer is refused, never truncated —
#: silently storing half a sentence makes the record lie about what was asked.
MAX_TASK_DESCRIPTION = 2000

#: Longest human-readable step label.
MAX_STEP_LABEL = 200

#: Longest tool name recorded.
MAX_TOOL_NAME = 200

#: Serialized bytes allowed in any one bounded metadata object.
MAX_METADATA_BYTES = 4096

#: Keys allowed in a metadata object. Caps breadth as well as size, so a caller
#: cannot send four thousand one-byte keys.
MAX_METADATA_KEYS = 32

#: Steps one run may hold. A run that needs more is a run whose timeline nobody
#: will read; the write is refused rather than the ledger silently truncated.
MAX_STEPS_PER_RUN = 1000

#: Default and maximum page sizes for every list endpoint in this domain.
DEFAULT_PAGE_SIZE = 50
MAX_PAGE_SIZE = 200
