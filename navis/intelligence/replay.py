"""Replay: drop an org's insights and re-project from Learning => identical output.

The append-only table is fully derivable from the immutable Learning log. Deleting
the insights and re-projecting yields byte-identical rows (same deterministic ids).
"""
from __future__ import annotations

import uuid

from navis.intelligence.repository import IntelligenceRepository
from navis.intelligence.service import IntelligenceService


class IntelligenceReplayService:
    def __init__(self, db=None, repo=None) -> None:
        self.repo = repo or IntelligenceRepository(db)
        self.service = IntelligenceService(repo=self.repo)

    async def rebuild(self, org_id: uuid.UUID):
        await self.repo.delete_org_insights(org_id)
        await self.repo.flush()
        return await self.service.project(org_id)
