"""ActionSpec — the ONE authoritative description of an executable action.

WHY THIS FILE EXISTS
--------------------
Everything a caller needs to know about an action was previously spread across
six places that nothing kept in agreement:

  * which action types exist      -> `ActionType` enum (models.py)
  * which provider runs one       -> `_DEFAULT_PROVIDERS` / `production_providers()`
  * which capability gates it     -> a single blanket `Capability.EXECUTION` on
                                     every execution route (router.py)
  * whether it needs a draft      -> `action_type == REPLY_TO_EMAIL` branches in
                                     router.py and summary.py
  * whether it has a connector
    precondition                  -> `_gmail_send_ready()`, a Gmail-shaped check
                                     hardcoded in the generic approve handler
  * whether it writes back into
    the read model                -> `CALENDAR_MUTATION_ACTION_TYPES`, imported
                                     into the generic worker from Signals

Adding Notion or Jira meant finding all six. Missing one produced a defect that
only appeared in production: an action type with no provider dead-letters after
approval, a connector with no capability check executes for an org that never
bought it, a new draftable action silently reports `requires_draft: false`.

So: ONE action, ONE spec. Every generic layer (service, router, worker, summary,
provider registry) asks this module instead of testing `action_type == ...`.

WHAT THIS FILE DELIBERATELY DOES NOT OWN
----------------------------------------
*Payload validation.* There is already exactly one validation boundary —
`Provider.validate()`, invoked by `execute_with_retry` before any side effect —
and it is in the right place: the connector owns the shape of its own payload,
and it is the last gate before the external call. Adding a schema here would
create the second source of truth this module exists to remove.

*Credentials.* Execution never resolves a credential. A provider is handed a
`sender` closure that owns its own integration lookup (see providers/_senders.py),
so Execution already depends on "call this sender", never on "this provider is
Google, therefore use the Google resolver". `preflight` below is the one place a
connector may expose a *readiness question* to the generic approve path, and it
is a callable the connector supplies — not a credential type Execution knows.

*Multi-step work.* No plan, no steps, no DAG. A spec describes one action. It is
shaped so a future Step can hold one, not so it can hold a workflow today.
"""
from __future__ import annotations

import enum
import uuid
from dataclasses import dataclass
from typing import Any, Protocol

from navis.execution.contract import ProviderReference
from navis.organization.contract import Capability


class UnknownAction(Exception):
    """No ActionSpec is registered for this action_type.

    The type is deliberately distinct from ProviderValidationError: this is
    "Navis does not know what that is", raised at INTAKE, before a row exists —
    not "the payload for a known action is malformed", raised before execution.
    """


class ActionPreconditionFailed(Exception):
    """A connector-level precondition for approving this action does not hold.

    Raised by `ActionSpec.preflight` (e.g. "Gmail is connected read-only, so this
    reply cannot be sent"). The approve handler turns it into a 409 with this
    message shown verbatim — the user is told what to fix, which is the entire
    point of checking before the draft is frozen rather than dead-lettering
    minutes later.
    """


class ActionRisk(str, enum.Enum):
    """How much damage performing this action can do.

    Only two members, because only two are real today. A `LOW`/read-only tier
    would be speculative: every action Navis can currently perform mutates
    something outside Navis. The distinction that IS real is whether the mutation
    lands in a system (an event moves) or in a human's inbox (a message that
    cannot be unsent).
    """

    # Mutates an object in an external system. Reversible in principle — the
    # event can be moved back, the page can be edited.
    EXTERNAL_WRITE = "external_write"
    # Delivers a message a person will read. There is no undo. Calendar verbs
    # notify attendees as a side channel, but their PRIMARY effect is the object
    # mutation; these are the actions whose primary effect is the message.
    IRREVERSIBLE_COMMUNICATION = "irreversible_communication"


class ApprovalPolicy(str, enum.Enum):
    """Whether a human must approve before the side effect runs.

    DECLARATIVE ONLY, and that is load-bearing. The hard boundary is the status
    machine: `claim_executable` selects APPROVED (or retryable FAILED) rows and
    nothing else, and REQUESTED -> APPROVED is the single edge into APPROVED
    (status.py), reachable only from the approve endpoint. No code path reads
    this field to SKIP approval, so a mis-registered spec — or a model that
    writes one — cannot turn an unapproved action into an executed one.

    `NOT_REQUIRED` exists so the registry invariant below can be stated at all
    ("everything that touches the outside world is REQUIRED"). Nothing is
    registered with it today.
    """

    REQUIRED = "required"
    NOT_REQUIRED = "not_required"


class Preflight(Protocol):
    """A connector's readiness question, asked before a human approves.

    Raises `ActionPreconditionFailed` when the action cannot be performed with
    the org's current grants. Returns None otherwise. Takes the session it should
    use so it participates in the request's transaction rather than opening its
    own.
    """

    async def __call__(self, db: Any, org_id: uuid.UUID) -> None: ...


class Reconciler(Protocol):
    """Write the just-mutated external object back into Navis's read model.

    Runs with NO transaction open (worker Phase B) and must never raise in a way
    that changes the execution outcome — the side effect already happened.

    `override` is the low-level connector reconciler the worker may be handed for
    testing; the adapter uses it in place of its production default, so that
    injection point keeps working without the worker knowing what a calendar is.
    """

    async def __call__(
        self,
        org_id: uuid.UUID,
        action_type: str,
        payload: dict[str, Any],
        reference: ProviderReference | None,
        *,
        override: Any = None,
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class ActionSpec:
    """Everything the generic layers need to know about one action.

    Every field earns its place by removing a hardcoded branch that exists in
    production today; none is speculative.
    """

    # The value stored in `execution_requests.action_type` (String(40)). This is
    # the identity — the DB column is already an open string, so registering a
    # new action needs no migration.
    action_type: str
    # `Provider.name`. Feeds the rate limiter key and the ProviderReference, and
    # lets the registry prove one action resolves to exactly one provider.
    provider: str
    # The CONNECTOR capability, from the existing `Capability` enum — no new
    # capability vocabulary is invented here. This is checked IN ADDITION to the
    # blanket `Capability.EXECUTION` gate on the execution routes, which is what
    # closes the gap where an org with EXECUTION but an explicitly denied (or
    # kill-switched) GMAIL could still send mail.
    capability: Capability
    risk: ActionRisk
    approval: ApprovalPolicy
    # True when this action is proposed before its content exists and must go
    # through a draft before it can be approved. Replaces the
    # `action_type == REPLY_TO_EMAIL` tests in router.py and summary.py.
    supports_draft: bool = False
    # Optional connector readiness check, run before a draft is frozen.
    preflight: Preflight | None = None
    # Optional read-model write-back. `None` for actions that mutate nothing
    # Navis mirrors. Replaces the worker's CALENDAR_MUTATION_ACTION_TYPES branch.
    reconciler: Reconciler | None = None


# ── Connector adapters ─────────────────────────────────────────────────────
#
# These live in Execution rather than in Signals so that Signals stays unchanged
# and unaware of Execution. They are the ONLY connector-shaped code in the
# generic path, and they are reached exclusively through a spec field — never by
# a layer testing `action_type == ...`.


def credential_preflight(provider: str) -> Preflight:
    """A pre-flight asking the integration boundary whether `provider` is usable.

    PROVIDER-NEUTRAL. This module used to import `navis.signals.google.resolve`
    and catch Google's exception types, which put connector authentication inside
    the generic action registry — the one place every future connector would have
    had to edit. It now names a provider and asks the boundary; Notion or Jira get
    the same pre-flight by registering a resolver, with no change here.

    Asks the SAME boundary the sender will use, so "will approve let this
    through?" and "can the worker actually act?" stay the same question by
    construction. Runs before a draft is frozen: discovering a missing grant at
    send time costs a dead letter the founder had no way to anticipate.
    """

    async def _preflight(db: Any, org_id: uuid.UUID) -> None:
        from navis.execution.integrations import (
            IntegrationUnavailable,
            integration_preflight,
        )

        try:
            await integration_preflight(db, org_id, provider)
        except IntegrationUnavailable as exc:
            # The connector wrote this message and it is safe to surface: it names
            # the provider and the action to take, never credential material.
            raise ActionPreconditionFailed(str(exc)) from exc

    return _preflight


async def _calendar_reconcile(
    org_id: uuid.UUID,
    action_type: str,
    payload: dict[str, Any],
    reference: ProviderReference | None,
    *,
    override: Any = None,
) -> None:
    """Pull the just-mutated Google event's CURRENT state into signal_events.

    The Google-event-id extraction lives HERE, behind the spec, which is what
    lets the worker call `spec.reconciler(...)` with no idea that calendars
    exist. The production reconciler is imported lazily so registering this spec
    does not drag the Signals stack into every module that imports the registry.
    """
    from navis.platform.monitoring import get_logger
    from navis.signals.calendar.reconcile import event_id_for

    external_id = event_id_for(
        action_type, payload, reference.external_id if reference else None
    )
    if not external_id:
        get_logger("execution.actions").warning(
            "%s produced no calendar event id; read model not reconciled",
            action_type,
        )
        return
    reconciler = override
    if reconciler is None:
        from navis.signals.calendar.reconcile import reconcile_calendar_event

        reconciler = reconcile_calendar_event
    await reconciler(org_id, external_id)


# ── The registry ───────────────────────────────────────────────────────────

_SPECS: dict[str, ActionSpec] = {}


def register_action(spec: ActionSpec, *, replace: bool = False) -> ActionSpec:
    """Register one action. This is the connector extension point.

    A future Notion/Jira module registers its actions through this call; nothing
    else needs to change for the generic path to route, gate, approve and execute
    them. Duplicate registration raises unless `replace` is explicit, so two
    connectors cannot silently claim the same action_type — the failure mode that
    would let one provider's payload reach another provider.
    """
    if not replace and spec.action_type in _SPECS:
        raise ValueError(
            f"action_type {spec.action_type!r} is already registered to provider "
            f"{_SPECS[spec.action_type].provider!r}; one action has one spec"
        )
    _SPECS[spec.action_type] = spec
    return spec


def unregister_action(action_type: str) -> None:
    """Remove a registration. Exists for connectors that register conditionally
    (and for tests that register a synthetic action and clean up after)."""
    _SPECS.pop(action_type, None)


def get_spec(action_type: str) -> ActionSpec | None:
    """The spec, or None. Callers that must fail closed use `require_spec`."""
    return _SPECS.get(action_type)


def require_spec(action_type: str) -> ActionSpec:
    """The spec, or raise. THE fail-closed entry point.

    Unknown action types raise rather than returning a permissive default, so a
    typo, a stale client, or a model inventing an action name cannot produce a
    row that looks executable.
    """
    spec = _SPECS.get(action_type)
    if spec is None:
        raise UnknownAction(
            f"unknown action_type {action_type!r}; registered: "
            f"{', '.join(sorted(_SPECS)) or '(none)'}"
        )
    return spec


def awaiting_draft(action_type: str, payload: dict[str, Any] | None) -> bool:
    """Is this row a placeholder Navis proposed but has not written yet?

    THE single answer to that question. `router._serialize` and
    `summary.execution_summary` are two surfaces reporting the same fact to the
    same card; when each computed it, a new draftable action could be draftable
    on one screen and not the other.

    An action that does not support drafts can never be a placeholder, so the
    spec is consulted first and the connector's payload check only runs for
    actions that actually have a draft stage.
    """
    spec = _SPECS.get(action_type)
    if spec is None or not spec.supports_draft:
        return False
    # Lazy: providers/__init__ imports this module, so a module-level import
    # would close the cycle.
    from navis.execution.providers.reply import requires_draft

    return requires_draft(payload)


def all_specs() -> dict[str, ActionSpec]:
    """A copy of the registry. Copy, not the live dict, so a caller iterating it
    cannot mutate the source of truth by accident."""
    return dict(_SPECS)


# ── The registered actions ─────────────────────────────────────────────────
#
# Capabilities use EXISTING `Capability` members only. No pricing or capability
# vocabulary is invented here: `PLAN_CAPABILITIES` in organization/contract.py
# stays the authority on what a plan grants, and these specs only name which of
# those existing capabilities each action consumes.

# EVERY LIVE CONNECTOR ANSWERS THE READINESS QUESTION.
#
# `reply_to_email` was the only action carrying a pre-flight, so approving a
# chat-composed email, a meeting, or a Slack post for an org whose grant could
# not perform it succeeded — and the dead letter minutes later was the first
# anyone heard of it. The credential question is identical for all of them
# ("can this org act on this provider right now?"), it is asked of the SAME
# boundary the sender will use, and the connector writes its own message. So it
# belongs on every spec whose provider is live, not on one of them.
#
# `create_task` is deliberately excluded: its provider is `unsupported` and
# refuses at validate(), and there is no credential resolver to ask.

register_action(ActionSpec(
    action_type="send_email",
    provider="gmail",
    capability=Capability.GMAIL,
    risk=ActionRisk.IRREVERSIBLE_COMMUNICATION,
    approval=ApprovalPolicy.REQUIRED,
    preflight=credential_preflight("gmail"),
))

register_action(ActionSpec(
    action_type="reply_to_email",
    provider="gmail",
    capability=Capability.GMAIL,
    risk=ActionRisk.IRREVERSIBLE_COMMUNICATION,
    approval=ApprovalPolicy.REQUIRED,
    # Proposed by Navis before a body exists; the draft is what a human can
    # actually have reviewed. See drafts/contract.py.
    supports_draft=True,
    preflight=credential_preflight("gmail"),
))

register_action(ActionSpec(
    action_type="schedule_meeting",
    provider="calendar",
    capability=Capability.CALENDAR,
    # EXTERNAL_WRITE, not COMMUNICATION: the primary effect is the event. Google
    # notifies attendees as a side channel (`send_google_invite`), which is why
    # this still requires approval.
    risk=ActionRisk.EXTERNAL_WRITE,
    approval=ApprovalPolicy.REQUIRED,
    # Asks for the WRITE grant specifically — an org connected under the
    # pre-v3 read-only calendar scope can still read availability, and would
    # discover it cannot book only when the insert 403s inside a worker retry.
    preflight=credential_preflight("calendar"),
    reconciler=_calendar_reconcile,
))

register_action(ActionSpec(
    action_type="reschedule_meeting",
    provider="calendar",
    capability=Capability.CALENDAR,
    risk=ActionRisk.EXTERNAL_WRITE,
    approval=ApprovalPolicy.REQUIRED,
    preflight=credential_preflight("calendar"),
    reconciler=_calendar_reconcile,
))

register_action(ActionSpec(
    action_type="cancel_meeting",
    provider="calendar",
    capability=Capability.CALENDAR,
    risk=ActionRisk.EXTERNAL_WRITE,
    approval=ApprovalPolicy.REQUIRED,
    preflight=credential_preflight("calendar"),
    reconciler=_calendar_reconcile,
))

register_action(ActionSpec(
    action_type="send_slack",
    provider="slack",
    capability=Capability.SLACK,
    risk=ActionRisk.IRREVERSIBLE_COMMUNICATION,
    approval=ApprovalPolicy.REQUIRED,
    # The Slack analogue of the Gmail send scope: an install without
    # `chat:write` can read a workspace and cannot post to it.
    preflight=credential_preflight("slack"),
))

register_action(ActionSpec(
    action_type="create_task",
    provider="tasks",
    # There is no TASKS capability and inventing one would be a pricing decision
    # this task has no authority to make. EXECUTION — the blanket capability
    # every execution route already requires — is the honest placeholder: the
    # provider is `unsupported` and refuses at validate(), so this action can
    # never execute regardless. Give it a real capability when a real task
    # backend is wired in.
    capability=Capability.EXECUTION,
    risk=ActionRisk.EXTERNAL_WRITE,
    approval=ApprovalPolicy.REQUIRED,
))


# ── Import-time invariants ─────────────────────────────────────────────────
#
# These run at import, so a mismatch fails the process at startup rather than
# producing a dead letter in production weeks later.

def _assert_registry_consistent() -> None:
    from navis.execution.models import ActionType

    enum_types = {a.value for a in ActionType}
    spec_types = set(_SPECS)
    if enum_types != spec_types:
        raise RuntimeError(
            "ActionType and the ActionSpec registry disagree. "
            f"enum-only={sorted(enum_types - spec_types)} "
            f"spec-only={sorted(spec_types - enum_types)}. "
            "The registry is the source of truth; `ActionType` exists only as "
            "readable constants for callers and must mirror it exactly."
        )
    from navis.execution.integrations import has_integration_resolver

    for spec in _SPECS.values():
        if spec.approval is not ApprovalPolicy.REQUIRED:
            raise RuntimeError(
                f"action {spec.action_type!r} touches an external system "
                f"(risk={spec.risk.value}) but does not require approval"
            )
        # An action whose provider HAS a credential authority must ask it before
        # a human approves. The alternative is the failure this pairs with the
        # capability gate to remove: approval succeeds, the worker claims the
        # row, and the org learns its grant cannot perform the action from a
        # dead letter instead of from the button they just pressed.
        #
        # Scoped to providers that actually have a resolver, so a deliberately
        # unsupported action (`create_task`, whose provider refuses at
        # validate()) is not required to invent one.
        if has_integration_resolver(spec.provider) and spec.preflight is None:
            raise RuntimeError(
                f"action {spec.action_type!r} resolves credentials through "
                f"provider {spec.provider!r} but declares no preflight; an "
                "approvable action must be able to say it cannot be performed "
                "BEFORE a human approves it"
            )


_assert_registry_consistent()
