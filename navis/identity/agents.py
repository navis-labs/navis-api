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
