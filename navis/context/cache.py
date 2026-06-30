"""Context bundle cache facade.

The default is a NullContextCache: bundles are content-addressed and cheap to rebuild,
so caching is opt-in. A durable Postgres cache lives in repository.py and is injected when
desired. Both honour the same ContextCache protocol, so the service never changes.
"""
from __future__ import annotations

from navis.context.contracts import ContextBundle


class NullContextCache:
    """No-op cache — every build is fresh. The safe, dependency-free default."""

    async def get(self, context_id: str) -> ContextBundle | None:
        return None

    async def put(self, bundle: ContextBundle) -> None:
        return None
