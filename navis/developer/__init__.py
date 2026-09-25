"""Developer Platform — the machine-access boundary over Navis Platform Core.

WHAT THIS IS. One authenticated, permission-gated, capability-gated, tenant-scoped
edge that a machine client consumes. MCP servers, SDKs and CLIs are *clients of
this boundary*; they are not implemented here and must never acquire business
logic of their own. Whatever an agent can do through `/api/v1/developer/*` is
whatever the existing Navis domains already do, reached through the existing
identity, permission and capability systems.

WHAT THIS IS NOT, AND MUST NEVER BECOME:

    * a second FastAPI application — the routers mount on the existing
      `api_router`, so Developer routes are in the existing OpenAPI schema;
    * a second database — nothing here declares an ORM model or a migration.
      `agents_router` is the test of that rule and passes it: an AGENT is a
      `service_accounts` row and an agent CREDENTIAL is an `api_keys` row, both
      written through `identity.service_accounts`, because a second machine
      identity is a second thing to revoke and a second answer to "which tenant";
    * a second identity / organization / permission / capability system —
      `gateway.py` composes `identity.authz.get_principal`,
      `identity.authz.require_permission` and
      `organization.guards.require_capability` and adds no rule of its own;
    * a second copy of any domain read — every data route delegates to the
      handler the web edge already calls, so the two surfaces cannot drift.

THE INVARIANT, restated as code review would apply it: this package may import
Identity, Organization and the domain read surfaces. Nothing may import THIS
package except `navis.api.router`, which mounts it. A dependency arrow pointing
the other way means a domain has started to know about the Developer edge, which
is the moment the boundary stops being a boundary.
"""
# Named `agent_lifecycle_router`, not `agents_router`: the latter is the MODULE's
# own name, and re-binding it here would leave `navis.developer.agents_router`
# meaning the router object in one import order and the module in another.
from navis.developer.agents_router import router as agent_lifecycle_router
from navis.developer.gateway import DeveloperContext, require_developer
from navis.developer.router import router

__all__ = ["DeveloperContext", "agent_lifecycle_router", "require_developer", "router"]
