"""The generic integration / credential boundary.

WHAT THE GENERIC EXECUTION LAYER IS ALLOWED TO KNOW
---------------------------------------------------
    organization + provider  ->  "is there a usable credential, or not?"

That is all. It does not know what OAuth is, that Google refreshes tokens, that
Gmail scopes differ from Calendar scopes, or that Slack uses a bot token. Each
provider's own module owns its authentication and registers a resolver here.

THE SHAPE THIS EXISTS TO PREVENT
--------------------------------
    if provider == "gmail":   resolve_google_token()
    elif provider == "slack": resolve_slack_token()
    elif provider == "notion": resolve_notion_token()

...appearing in the worker, the router, or the action registry. Before this
module, two GENERIC files already imported `navis.signals.google.resolve`
directly — `execution/router.py` (the draft sign-off) and `execution/actions.py`
(the approve pre-flight). Each new connector would have added a branch to both.

AUTHENTICATION IS NOT AUTHORIZATION
-----------------------------------
Resolving a credential answers "can Navis physically act as this org on this
system". It says NOTHING about whether the org is permitted to. The entitlement
gate (organization lifecycle -> EXECUTION -> the ActionSpec's connector
capability -> kill switch) is a separate question, asked FIRST and independently,
in `worker._authorization_denial`. Possession of a token must never imply
permission, so nothing in this module consults entitlements and nothing in the
authorization path consults credentials.

ORGANIZATION SCOPE IS MANDATORY
-------------------------------
There is no resolver signature that omits `org_id`, and `resolve_integration`
refuses a falsy one before dispatching. There is no global credential, no
process-level token cache, no developer fallback and no "most recent
integration" path — a credential is always looked up under exactly one
organization or not at all.
"""
from __future__ import annotations

import uuid
from importlib import import_module
from typing import Any, Callable, Protocol

from navis.platform.monitoring import get_logger

logger = get_logger("execution.integrations")


class IntegrationUnavailable(Exception):
    """No usable credential for (organization, provider).

    PERMANENT by contract. A missing, revoked, unscoped or token-less
    integration cannot be fixed by trying again in thirty seconds — it needs a
    human to reconnect the account. The worker therefore treats this like a
    validation error: fail once, dead-letter with this message, do not burn the
    retry budget rediscovering that Google is still not connected.

    Deliberately distinct from a provider BUSINESS failure (Gmail rejected the
    recipient) and from an AUTHORIZATION denial (the org is no longer entitled).
    Three different causes, three different things a human must do about them.

    `reconsent_required` separates "connect this account" from "reconnect it to
    grant more" — the two produce different instructions in the UI, and the
    existing Google resolvers already draw exactly that distinction.

    Carries NO credential material: the message names the provider and what the
    user must do, never a token, header or encrypted blob. It reaches the dead
    letter, the outbox and Observation, so it must stay safe to print.
    """

    def __init__(
        self, message: str, *, provider: str, reconsent_required: bool = False
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.reconsent_required = reconsent_required


class IntegrationResolver(Protocol):
    """A provider's own credential lookup.

    Returns an opaque handle — whatever that provider's sender needs (for Google
    it is the `Integration` row). The generic layer NEVER inspects it; only the
    module that registered the resolver knows its shape. Raises
    `IntegrationUnavailable` when the org has no usable credential.

    Takes the CALLER'S session so resolution joins the caller's transaction
    instead of opening its own.
    """

    async def __call__(self, db: Any, org_id: uuid.UUID) -> Any: ...


# provider name (ActionSpec.provider) -> "module.path:function".
#
# Stored as a string and imported on demand so this module stays free of the
# connector stacks. Importing the Google adapter eagerly would drag the whole
# Signals dependency chain into every process that merely wants to ask whether a
# credential exists.
_RESOLVERS: dict[str, str] = {
    "gmail": "navis.execution.providers._senders:resolve_gmail_credential",
    "calendar": "navis.execution.providers._senders:resolve_calendar_credential",
    "slack": "navis.execution.providers._senders:resolve_slack_credential",
}


def register_integration_resolver(
    provider: str, target: str | IntegrationResolver, *, replace: bool = False
) -> None:
    """Register a provider's credential resolver. The connector extension point.

    `target` is either a callable or a "module.path:function" string (preferred —
    it keeps the import lazy). Duplicate registration raises unless `replace` is
    explicit, so two connectors cannot silently claim one provider and hand the
    wrong org's credential shape to a sender.
    """
    if not replace and provider in _RESOLVERS:
        raise ValueError(
            f"provider {provider!r} already has a credential resolver; "
            "one provider has one credential authority"
        )
    _RESOLVERS[provider] = target  # type: ignore[assignment]


def has_integration_resolver(provider: str) -> bool:
    return provider in _RESOLVERS


def _load(provider: str) -> IntegrationResolver:
    target = _RESOLVERS[provider]
    if callable(target):
        return target
    module_path, _, attr = target.partition(":")
    return getattr(import_module(module_path), attr)


async def resolve_integration(
    db: Any,
    org_id: uuid.UUID,
    provider: str,
    *,
    integration_id: uuid.UUID | None = None,
) -> Any:
    """The org's credential for this provider, or raise.

    THE only way generic Execution obtains a credential. Every caller — the
    approve pre-flight, the draft sign-off, the production senders — goes through
    here, so organization scoping is enforced in one place rather than re-argued
    at each call site.

    `integration_id` PINS THE ACCOUNT (T-11). When an execution recorded which
    connected account it is for, that account is the only acceptable answer: the
    row is loaded directly, re-checked against `org_id` and the provider, and
    returned or refused. There is no path where a pinned account falls back to a
    different one — a wrong-account send is a real failure (wrong identity, wrong
    thread) that no retry fixes, so refusing is the honest outcome.

    When it is None the org-scoped resolver runs as before, but the resolvers
    themselves now refuse an organization holding SEVERAL candidate accounts
    rather than breaking the tie by recency.
    """
    if not org_id:
        # Fail closed. A resolver that ran without an organization would be free
        # to pick "some" integration, which is precisely the cross-tenant leak
        # this boundary exists to make impossible.
        raise IntegrationUnavailable(
            "credential resolution requires an organization", provider=provider
        )
    if provider not in _RESOLVERS:
        # An action whose provider has no registered credential authority cannot
        # act. Silence here would mean a sender improvising its own lookup.
        raise IntegrationUnavailable(
            f"no credential resolver is registered for provider {provider!r}; "
            "this connector cannot perform write actions in this deployment",
            provider=provider,
        )
    if integration_id is not None:
        return await _resolve_pinned(db, org_id, provider, integration_id)
    return await _load(provider)(db, org_id)


# Which Integration.provider values may serve each ActionSpec provider.
#
# DECLARED BY THE CONNECTOR, NOT LISTED HERE. A connector knows its own family —
# the Google grant writes one Integration row per granted service, all sharing a
# token, so a `gmail` action can legitimately be pinned to any of them, while a
# `slack` action can never be served by a Google row. Naming those provider values
# in this module would put connector knowledge back into the generic layer, which
# `tests/execution/test_runtime_hardening.py::
# test_generic_execution_contains_no_drive_or_docs_branch` exists to prevent (and
# caught on the first attempt at this). The strings live in
# `providers/_senders.py`, next to the resolvers that already own this knowledge.
_ACCEPTED: dict[str, frozenset[str]] = {}

# provider name -> the connector's own "can THIS row actually do it?" check.
#
# See `register_accepted_providers` for why this is a separate question from
# `_ACCEPTED`, and `_resolve_pinned` for why omitting it made pinning the weaker
# path.
PinVerifier = Callable[[Any], None]
_PIN_VERIFIERS: dict[str, PinVerifier] = {}


def register_accepted_providers(
    provider: str,
    accepts: frozenset[str] | set[str],
    *,
    verify: PinVerifier | None = None,
) -> None:
    """Declare which `Integration.provider` values may serve this action provider.

    Called by the connector module alongside its credential resolver. A provider
    that never registers cannot be pinned at all (see `_resolve_pinned`) — the
    fail-closed direction, because an unregistered family is one nobody has
    decided the compatibility rules for.

    `verify` IS WHAT KEEPS PINNING FROM BEING THE WEAKER PATH (T-11).

    `accepts` answers a structural question — may a `gmail` row serve a `calendar`
    action? — and one Google consent writes a gmail, a google_calendar and a drive
    row that all share a token, so the answer is yes for the whole family. It says
    nothing about whether THIS row was granted the scope the action needs, or
    whether it still holds tokens. The org-scoped resolvers have always asked
    both: `resolve_gmail_send_integration` filters on `gmail_send_granted` and on
    stored tokens before it returns anything.

    `_resolve_pinned` asked neither. Reproduced before this parameter existed: a
    Gmail row carrying only `gmail.readonly`, and a send-scoped row with no stored
    tokens, BOTH resolved when pinned while the unpinned resolver refused each
    with `reconsent_required`. The consequence is not a wrong account — the pin
    prevents that — it is losing the permanent, actionable refusal ("reconnect
    Google to grant send access") and getting an opaque provider 403 at send time
    instead, which the worker classifies as transient and retries five times.

    The callable is the CONNECTOR's, and raises `IntegrationUnavailable` in its
    own words. Optional so a connector that has no capability distinction
    registers nothing and behaves exactly as before.
    """
    _ACCEPTED[provider] = frozenset(accepts)
    if verify is not None:
        _PIN_VERIFIERS[provider] = verify


async def _resolve_pinned(
    db: Any, org_id: uuid.UUID, provider: str, integration_id: uuid.UUID
) -> Any:
    """Load exactly the pinned integration, or refuse. Never substitutes.

    Five checks, each fail-closed, in the order that leaks least: existence,
    OWNERSHIP, provider compatibility, active status, then the connector's own
    capability check. Ownership is checked against the caller's trusted `org_id` —
    which the worker takes from the persisted execution row — so a pinned id
    belonging to another tenant is indistinguishable from one that does not exist.

    The last check is what makes a pinned resolution AT LEAST as strict as an
    unpinned one; see `register_accepted_providers`.
    """
    from navis.identity.models import Integration

    # Force the connector module to import so its `register_accepted_providers`
    # side effect has run. `_load` is the same lazy import the unpinned path uses,
    # so this costs nothing extra on any request that was going to resolve anyway.
    _load(provider)

    row = await db.get(Integration, integration_id)
    if row is None or row.organization_id != org_id:
        # Deliberately one message for both cases: an execution must not be able
        # to probe whether an integration id exists in some other organization.
        raise IntegrationUnavailable(
            "the connected account this action was created for is no longer "
            "available; reconnect the account or recreate the action",
            provider=provider,
        )
    row_provider = row.provider.value if hasattr(row.provider, "value") else str(row.provider)
    if row_provider not in _ACCEPTED.get(provider, frozenset()):
        raise IntegrationUnavailable(
            f"the connected account this action was created for cannot perform a "
            f"{provider} action",
            provider=provider,
        )
    if not row.is_active:
        raise IntegrationUnavailable(
            f"the {provider} account this action was created for has been "
            "disconnected; reconnect it to run this action",
            provider=provider,
            reconsent_required=True,
        )
    verifier = _PIN_VERIFIERS.get(provider)
    if verifier is not None:
        # Raises IntegrationUnavailable in the connector's own words. Not wrapped:
        # the connector already writes the message a human has to act on, and
        # re-phrasing it here would put connector knowledge back in this module.
        verifier(row)
    return row


async def integration_preflight(db: Any, org_id: uuid.UUID, provider: str) -> None:
    """Raise `IntegrationUnavailable` unless the org can act on this provider.

    The read-only form of `resolve_integration`, used before a human approves so
    they are told to reconnect BEFORE reviewing and freezing a draft, rather than
    discovering it from a dead letter minutes later.
    """
    await resolve_integration(db, org_id, provider)
