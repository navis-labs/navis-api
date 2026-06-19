"""Replay: drop the deterministic projection and rebuild it from Intelligence.

The projection = policy_proposals + DRAFT policy_snapshots. Deleting and
re-deriving them yields byte-identical rows (same deterministic ids/payloads).

Governance history (approvals, active/superseded snapshots) is the record of human
decisions and is deliberately OUT of replay scope — you cannot, and must not,
"replay" a founder's approval. Replay proves the *machine half* is deterministic.
"""
from __future__ import annotations

import uuid

from navis.adaptive.repository import AdaptiveRepository
from navis.adaptive.service import AdaptiveService


class AdaptiveReplayService:
    def __init__(self, db=None, repo=None) -> None:
        self.repo = repo or AdaptiveRepository(db)
        self.service = AdaptiveService(self.repo)

    async def rebuild(self, org_id: uuid.UUID):
        await self.repo.delete_projection(org_id)
        await self.repo.flush()
        return await self.service.propose(org_id)
