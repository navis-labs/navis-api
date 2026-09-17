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
from navis.signals.microsoft_router import router as signals_microsoft_router
from navis.signals.router import router as signals_router

api_router = APIRouter()
api_router.include_router(identity_router)
api_router.include_router(identity_admin_router)
api_router.include_router(scim_router)
api_router.include_router(organization_router)
api_router.include_router(signals_router)
api_router.include_router(signals_microsoft_router)
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

