"""Aggregate router. Each domain owns its routes; the edge just mounts them.

EVERY ROUTE MOUNTED HERE RESOLVES A PRINCIPAL, with four deliberate exceptions
that authenticate by other means: `/auth/*` (mints credentials, and `/auth/logout`
validates its own bearer), `/integrations/google/callback` (signed OAuth state),
and the `/webhooks/*` endpoints (provider signature verification).

`compat_router` used to be a fifth exception and was not one of these. It served
`/decide/stack`, `/skills`, `/signals`, `/execution/slack/blockers` and
`/workspace/meetings/{id}/brief` as unauthenticated 200s with empty payloads —
so an anonymous caller, an expired session and a genuinely empty organization
were INDISTINGUISHABLE on the wire, and a client whose token had died rendered
"nothing to do today" instead of signing the user back in. It was deleted in
TASK 6.4.4 rather than authenticated: none of the five had a backing
implementation, so gating them would have made the fabricated payload an
*authenticated* lie instead of an anonymous one. A 404 is the truth — the
backend does not serve these — and the frontend's `fetchJsonOr` already treats
404 as "not shipped" for four of them, while the fifth (the meeting brief) has
an error state it renders instead.

DO NOT reintroduce a stub router. A route that returns a plausible empty
business payload without checking who is asking is an outage that looks like
a quiet day.
"""
from __future__ import annotations

from fastapi import APIRouter

from navis.adaptive.router import router as adaptive_router
from navis.api.admin_router import router as admin_router
from navis.api.brief_router import router as brief_router
from navis.api.chat_router import router as chat_router
from navis.api.context_router import router as context_router
from navis.api.knowledge_router import router as knowledge_router
from navis.api.memory_router import router as memory_router
from navis.api.workspace_router import router as workspace_router
from navis.developer.agents_router import router as developer_agents_router
from navis.developer.context_router import router as developer_context_router
from navis.developer.router import router as developer_router
from navis.developer.runs_router import router as developer_runs_router
from navis.execution.router import router as execution_router
from navis.identity.admin_router import router as identity_admin_router
from navis.identity.router import router as identity_router
from navis.identity.scim_router import router as scim_router
from navis.identity_resolution.router import router as identity_resolution_router
from navis.intelligence.router import router as intelligence_router
from navis.learning.router import router as learning_router
from navis.observation.router import router as observation_router
from navis.observation.webhooks.router import router as observation_webhooks_router
from navis.organization.router import router as organization_router
from navis.signals.read_router import router as signals_read_router
from navis.signals.router import router as signals_router

api_router = APIRouter()
api_router.include_router(identity_router)
api_router.include_router(identity_admin_router)
api_router.include_router(scim_router)
api_router.include_router(organization_router)
api_router.include_router(signals_router)
api_router.include_router(signals_read_router)
api_router.include_router(identity_resolution_router)
api_router.include_router(brief_router)
api_router.include_router(context_router)
# Canonical organizational entities, projected from State V2. Registered next to
# `/context` because they are the two read-side shapes over the same snapshot;
# `/memory/*` below is the legacy derived surface this supersedes.
api_router.include_router(knowledge_router)
api_router.include_router(execution_router)
# The two read edges that close the loop on the wire. Observation and Learning
# have persisted outcomes and attributions since they shipped and neither had a
# reader, so "what happened after Navis decided?" could only be answered by
# fabricating it. Both are read-only and gated on their own capability.
api_router.include_router(observation_router)
api_router.include_router(learning_router)
api_router.include_router(intelligence_router)
api_router.include_router(adaptive_router)
api_router.include_router(observation_webhooks_router)
api_router.include_router(workspace_router)
api_router.include_router(chat_router)
api_router.include_router(memory_router)
api_router.include_router(admin_router)
# The MACHINE-ACCESS boundary — `/api/v1/developer/*`. Mounted here, on the same
# router as every human-facing edge, because there must be exactly one
# application, one OpenAPI schema and one auth chain. Its routes delegate to the
# handlers above rather than reimplementing them, so the Developer contract that
# future MCP/SDK/CLI clients consume can never drift from what the web app sees.
api_router.include_router(developer_router)
# The Agent LIFECYCLE — `/api/v1/developer/agents`. A separate module rather than
# more handlers in `developer_router`, because that file's contract is
# "read-only, by construction" and it is worth keeping literally true. Same
# application, same prefix, same auth chain; an agent is a `service_accounts` row
# and nothing here declares a model or a tenant resolver of its own.
api_router.include_router(developer_agents_router)
# The Agent EXECUTION LEDGER — `/developer/tasks` and `/developer/runs`. Its own
# module for the same reason as the agent lifecycle above: `developer_router`'s
# contract is "nothing here is a write", and these are writes. Reads are gated on
# `view_observation` (this IS the observability surface), writes on
# `record_agent_runs`, so a read permission never authorizes a write.
api_router.include_router(developer_runs_router)
# TASK-AWARE CONTEXT — `POST /developer/context`. Its own module because
# `developer_router`'s contract is "every handler is a GET", which is what makes
# its read-only guarantee checkable at a glance. This is a POST because its
# input is a task in prose, NOT because it writes: it reads State through
# `read_only_state`, which never folds, rebuilds or disposes. It calls the same
# `orchestration.context.build_grounded_context` that `/chat` calls, so an agent
# and a founder are grounded in one organization rather than two.
api_router.include_router(developer_context_router)

