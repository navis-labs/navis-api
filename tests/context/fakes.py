"""Test doubles for the Context layer — a fake StateReader and an in-memory cache."""
from __future__ import annotations

import uuid

from navis.context.contracts import ContextBundle
from navis.state.v2.models import OrganizationalState


class FakeStateReader:
    """Serves pre-built OrganizationalState per org (no DB, no StateService)."""

    def __init__(self, *states: OrganizationalState) -> None:
        self._by_org: dict[uuid.UUID, OrganizationalState] = {s.org_id: s for s in states}

    async def get_state(self, org_id: uuid.UUID) -> OrganizationalState:
        return self._by_org[org_id]


class MemoryContextCache:
    """In-memory ContextCache double — exercises the cache hit/put path without Postgres."""

    def __init__(self) -> None:
        self.store: dict[str, ContextBundle] = {}
        self.hits = 0
        self.puts = 0

    async def get(self, context_id: str) -> ContextBundle | None:
        bundle = self.store.get(context_id)
        if bundle is not None:
            self.hits += 1
        return bundle

    async def put(self, bundle: ContextBundle) -> None:
        self.puts += 1
        self.store[bundle.context_id] = bundle
