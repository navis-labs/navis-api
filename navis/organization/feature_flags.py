"""Server-side feature flags: release flags, permission flags, kill switches.

Evaluation is server-side only — the client is never trusted. An org-specific row
overrides the global (org_id NULL) default. Kill switches use the naming convention
`kill:<capability>`; when enabled, the capability is forced off.
"""
from __future__ import annotations

import uuid

from navis.organization.repository import OrganizationRepository


def kill_switch_flag(capability: str) -> str:
    return f"kill:{capability}"


class FeatureFlagService:
    def __init__(self, repo: OrganizationRepository) -> None:
        self.repo = repo

    async def evaluate(self, org_id: uuid.UUID, flag: str) -> bool:
        """True/False for this org. Org-specific row wins over the global default."""
        rows = await self.repo.flags_for(org_id, flag)
        if not rows:
            return False
        specific = next((r for r in rows if r.organization_id == org_id), None)
        chosen = specific or rows[0]
        return bool(chosen.enabled)

    async def is_killed(self, org_id: uuid.UUID, capability: str) -> bool:
        return await self.evaluate(org_id, kill_switch_flag(capability))

    async def set(self, org_id: uuid.UUID | None, flag: str, *, kind: str = "release", enabled: bool) -> None:
        await self.repo.set_flag(org_id, flag, kind=kind, enabled=enabled)
