"""Wire contracts for `/api/v1/developer/agents`.

WHY THIS FILE IS SEPARATE FROM `schemas.py`. That module's contract is "nothing
here is a request model — the foundation is read-only", and it is worth keeping
true: every model in it describes something the Developer boundary READS, and a
reader can check that claim by opening one file. Agents are the boundary's first
mutable resource, so their request models live next to their response models
here rather than quietly falsifying a docstring next door.

NO MODEL BELOW CARRIES AN IDENTITY INPUT. There is no `organization_id`, no
`owner_user_id`, no `principal_id` and no `created_by` on any request. The tenant
is `Principal.org_id` in every handler, and the agent id is a PATH parameter that
is looked up under that tenant — so naming another organization's agent id
produces a 404, not a cross-tenant read. `AgentOut.organization_id` is a
response-only echo of the tenant the server resolved, present so a client can
assert it is talking to the organization it thinks it is.

NO SECRET IS READABLE. `CreatedAgentOut` and `CreatedAgentCredentialOut` are the
only models with a `secret` field, they are returned only by the two POST
handlers, and the value is never stored — `api_keys.hashed_secret` holds a
SHA-256. Every list and every GET returns `AgentCredentialOut`, which has no
secret field at all, so a route cannot leak one by returning the wrong model.
"""
from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, Field, field_validator

from navis.identity.agents import AgentEnvironment, AgentRuntime

#: Names are shown to operators and stored verbatim. Bounded to the column width
#: so an over-long value is a 422 with a field name rather than a database error.
_NAME_MAX = 255
_DESCRIPTION_MAX = 500


class AgentOut(BaseModel):
    """One agent, as the server resolved it. Never carries a secret."""

    id: uuid.UUID = Field(
        description=(
            "The agent's stable id — the `service_accounts` row, and the same "
            "value `/developer/me` reports as `principal.service_account_id` "
            "when this agent's credential is the caller."
        )
    )
    organization_id: uuid.UUID = Field(
        description="The tenant this agent belongs to, derived from the caller's credential."
    )
    name: str
    description: str
    environment: AgentEnvironment = Field(
        description="Deployment tier this agent runs against. Recorded, never enforced."
    )
    runtime: AgentRuntime = Field(
        description="Agent framework behind the credential. Observability only."
    )
    status: str = Field(
        description=(
            "active | revoked. DERIVED from the revocation clock, not stored: "
            "the same field authentication reads to refuse a revoked agent."
        )
    )
    permissions: list[str] = Field(
        description=(
            "The permissions this agent's credentials carry — the SAME vocabulary "
            "human roles are granted from (`identity.permissions`). Sorted."
        )
    )
    credential_count: int = Field(
        description="Active (unrevoked) credentials bound to this agent."
    )
    last_used_at: datetime | None = Field(
        default=None,
        description=(
            "The most recent authenticated request by any of this agent's "
            "credentials, accurate to ~15 minutes (the write is throttled). Null "
            "until the agent authenticates for the first time."
        ),
    )
    created_at: datetime
    updated_at: datetime


class AgentPageOut(BaseModel):
    """A page of agents. Paged from the first response, not once it hurts."""

    agents: list[AgentOut]
    count: int = Field(description="Agents in THIS page.")
    total: int = Field(description="Agents this organization holds, all pages.")
    limit: int
    offset: int


class AgentCredentialOut(BaseModel):
    """Credential METADATA. There is deliberately no secret field on this model."""

    id: uuid.UUID = Field(
        description=(
            "The `api_keys` row — the handle for revocation and the value "
            "`/developer/me` reports as `principal.credential_id`."
        )
    )
    agent_id: uuid.UUID
    organization_id: uuid.UUID
    name: str
    prefix: str = Field(
        description="Public, non-secret lookup prefix (16 hex chars). Safe to display."
    )
    scopes: list[str] = Field(description="Inherited from the agent. Sorted.")
    revoked: bool
    last_used_at: datetime | None = None
    created_at: datetime


class AgentCredentialPageOut(BaseModel):
    credentials: list[AgentCredentialOut]
    count: int
    limit: int
    offset: int


class CreatedAgentCredentialOut(AgentCredentialOut):
    """The ONE response that carries a plaintext secret.

    The value exists in this response and nowhere else — not in the database, not
    in a log line, not in a URL. It cannot be re-read, and no GET returns this
    model. An interface that appears to show an existing secret is either lying
    or leaking.
    """

    secret: str = Field(
        description="Full `navis_<prefix>_<secret>` plaintext. Shown once. Never stored."
    )


class CreatedAgentOut(BaseModel):
    """A new agent plus its first credential — the only time both exist together."""

    agent: AgentOut
    credential: CreatedAgentCredentialOut


class _AgentFields(BaseModel):
    """The editable surface, shared by create and update so they cannot drift."""

    @field_validator("name", check_fields=False)
    @classmethod
    def _name_not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            # A whitespace name passes `min_length=1` and renders as an empty row
            # in every list that shows it. Reject at the edge rather than storing
            # something no operator can identify later.
            raise ValueError("name must not be blank")
        return stripped


class CreateAgentRequest(_AgentFields):
    """What an operator supplies. Everything else is assigned by the server.

    `permissions` MAY be empty. An agent with no permissions authenticates and
    reads nothing — every gated Developer route answers 403 — which is a
    legitimate "registered, not yet granted" state and is how an operator stages
    a credential before deciding its reach. The console requires at least one
    because silently creating a key that can read nothing is a support ticket a
    week later; the API does not, because refusing it would make the staged state
    unreachable.
    """

    name: str = Field(max_length=_NAME_MAX, description="Shown wherever this agent appears.")
    description: str = Field(default="", max_length=_DESCRIPTION_MAX)
    environment: AgentEnvironment = AgentEnvironment.DEVELOPMENT
    runtime: AgentRuntime = AgentRuntime.CUSTOM
    permissions: list[str] = Field(
        default_factory=list,
        description=(
            "Permissions to grant, from `identity.permissions`. Each is refused "
            "unless the CALLER already holds it, so creation can never escalate."
        ),
    )


class UpdateAgentRequest(_AgentFields):
    """A partial edit. An omitted field is left alone; `null` is not accepted.

    `permissions`, when present, REPLACES the grant list and cascades to every
    live credential of this agent in the same transaction — see
    `ServiceAccountService.update`. A partial-add contract would need a removal
    contract to match it, and the two together are how a grant list drifts from
    what an operator believes they approved.
    """

    name: str | None = Field(default=None, max_length=_NAME_MAX)
    description: str | None = Field(default=None, max_length=_DESCRIPTION_MAX)
    environment: AgentEnvironment | None = None
    runtime: AgentRuntime | None = None
    permissions: list[str] | None = None


class CreateAgentCredentialRequest(BaseModel):
    """Mint an ADDITIONAL credential for an existing agent.

    There is no `scopes` field, and that is the point: a credential's reach is
    the AGENT's reach. Accepting scopes here would make the console's agent
    permission screen a summary of several different answers, and would need a
    rule for what happens when the agent's grants later change.
    """

    name: str = Field(
        max_length=_NAME_MAX,
        description="How this credential is recognised in the list — 'CI', 'laptop'.",
    )

    @field_validator("name")
    @classmethod
    def _name_not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("name must not be blank")
        return stripped


class AgentRevokedOut(BaseModel):
    """The terminal state of an agent, reported so a client need not re-read it."""

    id: uuid.UUID
    status: str = Field(description="Always 'revoked'.")
    credentials_revoked: int = Field(
        description="Credentials that were active and are now not."
    )


class CredentialRevokedOut(BaseModel):
    id: uuid.UUID
    agent_id: uuid.UUID
    revoked: bool = Field(description="Always true.")
