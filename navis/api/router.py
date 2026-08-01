"""Aggregate router. Each domain owns its routes; the edge just mounts them."""
from __future__ import annotations

from fastapi import APIRouter

from navis.adaptive.router import router as adaptive_router
from navis.api.admin_router import router as admin_router
from navis.api.brief_router import router as brief_router
from navis.api.chat_router import router as chat_router
from navis.api.compat_router import router as compat_router
from navis.api.context_router import router as context_router
from navis.api.memory_router import router as memory_router
from navis.api.workspace_router import router as workspace_router
from navis.execution.router import router as execution_router
from navis.identity.admin_router import router as identity_admin_router
from navis.identity.router import router as identity_router
from navis.identity.scim_router import router as scim_router
from navis.identity_resolution.router import router as identity_resolution_router
from navis.intelligence.router import router as intelligence_router
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
api_router.include_router(execution_router)
api_router.include_router(intelligence_router)
api_router.include_router(adaptive_router)
api_router.include_router(observation_webhooks_router)
api_router.include_router(workspace_router)
api_router.include_router(chat_router)
api_router.include_router(memory_router)
api_router.include_router(compat_router)
api_router.include_router(admin_router)

