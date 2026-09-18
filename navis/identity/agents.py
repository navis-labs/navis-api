"""The AGENT vocabulary — the product's word for a machine identity.

AN AGENT IS A SERVICE ACCOUNT. There is no `agents` table and there must not be
one. `service_accounts` already carries every structural fact an agent needs — an
organization, a name, a scope list, a revocation clock, timestamps — and
`authz._principal_from_api_key` already closes the credential -> agent ->
organization chain against it, including the tenant-mismatch refusal. A second
table would be a second machine identity, a second thing to revoke, and a second
answer to "which tenant is this" the day the two disagree.

What the product needs on top of that row is DESCRIPTIVE, not structural: what
the thing is for, which deployment tier it runs against, and which runtime is
behind it. Those are three columns on the existing table (migration 0046), and
this module is the closed vocabulary they are checked against.

NOTHING HERE IS AN AUTHORIZATION INPUT. Environment and runtime are recorded for
operators and are never consulted by a gate: an agent tagged `production` holds
exactly the permissions its scopes name, and `custom` is a first-class runtime
rather than a fallback. Making either change behaviour would put a second,
undocumented authorization axis next to `Permission` — which is the failure the
Developer boundary exists to prevent.

Pure module — no DB, no imports from other domains, mirroring `permissions.py`.
"""
from __future__ import annotations

import enum
from datetime import datetime, timedelta, timezone


class AgentEnvironment(str, enum.Enum):
    """Which deployment tier this agent runs against.

    A DOMAIN fact owned by the organization, deliberately NOT the process's own
    `settings.environment`. They share three words and mean different things: an
    operator may register a `production` agent from a laptop, and a staging
    deployment may hold `development` agents. Coupling them would let a config
    variable silently re-label somebody's fleet.
    """

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class AgentRuntime(str, enum.Enum):
    """The agent framework behind the credential. Observability only.

    Matches `AGENT_RUNTIMES` in `@navis/types` so the console and the backend
    name the same set. A value outside it is rejected at the edge and by a
    database CHECK rather than stored as free text: an unbounded label cannot be
    grouped, filtered or counted, which is the only reason to record it at all.
    """

    CLAUDE_CODE = "claude-code"
    CODEX = "codex"
    CURSOR = "cursor"
    GEMINI = "gemini"
    OPENAI_AGENTS = "openai-agents"
    CUSTOM = "custom"


AGENT_ENVIRONMENTS: frozenset[str] = frozenset(e.value for e in AgentEnvironment)
AGENT_RUNTIMES: frozenset[str] = frozenset(r.value for r in AgentRuntime)


class AgentStatus(str, enum.Enum):
    """Derived, never stored.

    `service_accounts.revoked_at` is the one authority on whether an agent may
    act — it is what `authz._principal_from_api_key` reads to answer 401, and it
    is set by the same call that revokes the agent's keys. A `status` COLUMN
    would be a second authority that the authentication path does not consult,
    so the two would eventually disagree and the wrong one would win.

    Revocation is terminal on purpose. There is no `disabled` state to come back
    from: re-enabling an identity whose credentials were handed out and then
    revoked is a new agent, not the old one waking up.
    """

    ACTIVE = "active"
    REVOKED = "revoked"


def status_of(revoked_at: object | None) -> str:
    """The wire `status` for a service-account row. One derivation, one answer."""
    return AgentStatus.REVOKED.value if revoked_at is not None else AgentStatus.ACTIVE.value


class CredentialStatus(str, enum.Enum):
    """What state one `api_keys` row is in. Derived, never stored.

    Same reasoning as `AgentStatus`: a `status` column would be a second
    authority the authentication path never consults. This is computed from the
    two timestamps that path DOES read, so the console, the API and
    `verify()` cannot disagree about whether a credential works.

    REVOKED OUTRANKS EXPIRED when a row carries both. Revocation is a decision
    somebody made and is terminal; an expiry that has also passed adds nothing a
    reader can act on, and reporting "expired" for a key an operator
    deliberately cut off would misattribute the reason it stopped.

    EXPIRING IS A WARNING, NOT A STATE THE SERVER ENFORCES. A credential that is
    expiring still authenticates exactly like an active one — the distinction
    exists so an operator can rotate before an outage rather than after one.
    """

    ACTIVE = "active"
    EXPIRING = "expiring"
    EXPIRED = "expired"
    REVOKED = "revoked"


#: How long before `expires_at` a credential starts reporting `expiring`.
#:
#: Seven days is chosen to be longer than a weekend plus a working day: the
#: point of the warning is that somebody has time to mint a replacement, switch
#: over and verify it before the old one stops, which is the rotation sequence
#: OpenAI's guidance describes. A shorter window would fire when it is already
#: an emergency; a much longer one would sit yellow for so long that it stops
#: being read as a warning at all.
CREDENTIAL_EXPIRY_WARNING = timedelta(days=7)


def credential_status_of(
    revoked_at: datetime | None,
    expires_at: datetime | None,
    *,
    now: datetime | None = None,
) -> str:
    """The wire `status` for a credential. One derivation, one answer.

    `now` is injectable so a caller rendering a page of credentials stamps them
    all at the same instant — two rows in one response disagreeing about what
    time it is would be a real, if small, lie.
    """
    if revoked_at is not None:
        return CredentialStatus.REVOKED.value
    if expires_at is None:
        return CredentialStatus.ACTIVE.value
    moment = now or datetime.now(timezone.utc)
    # A naive `expires_at` cannot come from the database (the column is
    # timestamptz) but can come from a test or a fixture, and comparing naive to
    # aware raises. Treat it as UTC rather than failing a status read.
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    if expires_at <= moment:
        return CredentialStatus.EXPIRED.value
    if expires_at - moment <= CREDENTIAL_EXPIRY_WARNING:
        return CredentialStatus.EXPIRING.value
    return CredentialStatus.ACTIVE.value
