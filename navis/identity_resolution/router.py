"""Identity Resolution HTTP surface — read-only views over the identity graph.

Consumed by the frontend "who is this person, everywhere" flow and by any
downstream projector that wants a canonical view without duplicating the
BFS. The write path is event-driven (`IdentityResolutionService`), never
exposed as HTTP — clients don't manufacture aliases.
"""
from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from navis.identity.dependencies import CurrentContext, get_current_context
from navis.identity_resolution.ids import alias_id as compute_alias_id
from navis.identity_resolution.models import IdentityAlias
from navis.identity_resolution.resolver import (
    AliasKind,
    AliasProvider,
    canonical_alias_ids,
    normalize_alias_value,
)
from navis.identity_resolution.search import search_persons
from navis.platform.database import get_session

router = APIRouter(prefix="/identity", tags=["identity_resolution"])


def _serialize(row: IdentityAlias) -> dict[str, Any]:
    return {
        "alias_id": str(row.id),
        "provider": row.provider,
        "kind": row.kind,
        "value": row.value,
        "display_name": row.display_name,
        "first_seen_at": row.first_seen_at.isoformat() if row.first_seen_at else None,
        "last_seen_at": row.last_seen_at.isoformat() if row.last_seen_at else None,
    }


@router.get("/persons/lookup")
async def lookup_person(
    kind: str,
    value: str,
    ctx: CurrentContext = Depends(get_current_context),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """Resolve any (kind, value) — email, slack_user_id, github_login, ... — to
    the full equivalence class within the org. Returns every alias reachable via
    the append-only link log (i.e. every provider identity that resolves to the
    same organizational person).

    Provider is intentionally NOT part of the lookup: the whole point of this
    layer is that email `x@y` from Gmail and email `x@y` from HubSpot resolve to
    ONE alias row (both compute the same UUIDv5). Passing `provider` would only
    let a caller ask a narrower question the resolver already answers."""
    try:
        alias_kind = AliasKind(kind)
    except ValueError:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"unknown alias kind: {kind}")

    normalized = normalize_alias_value(alias_kind, value)
    if not normalized:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "empty alias value")

    # An alias is (provider, kind, value); the SAME (kind, value) can exist under
    # multiple providers. Compute all candidate ids for every known provider that
    # supports this kind and look them up in one shot.
    candidate_ids: list[uuid.UUID] = [
        compute_alias_id(ctx.org_id, p.value, alias_kind.value, normalized)
        for p in AliasProvider
    ]
    seeds = (await db.scalars(
        select(IdentityAlias).where(
            IdentityAlias.organization_id == ctx.org_id,
            IdentityAlias.id.in_(candidate_ids),
        )
    )).all()
    if not seeds:
        return {"canonical_id": None, "aliases": []}

    # Fold every seed's equivalence class into one set. Sorted for deterministic response.
    class_ids: set[uuid.UUID] = set()
    for s in seeds:
        class_ids |= await canonical_alias_ids(db, ctx.org_id, s.id)
    class_rows = (await db.scalars(
        select(IdentityAlias)
        .where(
            IdentityAlias.organization_id == ctx.org_id,
            IdentityAlias.id.in_(class_ids),
        )
        .order_by(IdentityAlias.first_seen_at.asc(), IdentityAlias.id.asc())
    )).all()

    canonical = min((str(r.id) for r in class_rows), default=None)
    return {
        "canonical_id": canonical,
        "aliases": [_serialize(r) for r in class_rows],
    }


@router.get("/persons/search")
async def search_person(
    q: str,
    ctx: CurrentContext = Depends(get_current_context),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """Name-substring / email-substring lookup over the identity graph.

    Complements /persons/lookup (exact value only) — used by the chat
    scheduling agent to resolve rough handles like "Rahul" to a person.
    Case-insensitive; results folded by canonical equivalence class so
    two aliases of the same real person collapse to one match. When
    multiple people match, `ambiguous=true` and the caller (agent) is
    expected to ask a clarification instead of picking one arbitrarily.

    Queries shorter than 2 characters return an empty set — the router
    never pretends a 1-char probe is a specific person lookup.
    """
    result = await search_persons(db, ctx.org_id, q)
    return result.to_dict()


@router.get("/persons/{alias_id}")
async def get_person(
    alias_id: uuid.UUID,
    ctx: CurrentContext = Depends(get_current_context),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """Full equivalence class for the given seed alias — the canonical view of
    every provider identity that resolves to this person within the org."""
    seed = await db.get(IdentityAlias, alias_id)
    if seed is None or seed.organization_id != ctx.org_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "alias not found")

    class_ids = await canonical_alias_ids(db, ctx.org_id, alias_id)
    rows = (await db.scalars(
        select(IdentityAlias)
        .where(
            IdentityAlias.organization_id == ctx.org_id,
            IdentityAlias.id.in_(class_ids),
        )
        .order_by(IdentityAlias.first_seen_at.asc(), IdentityAlias.id.asc())
    )).all()
    canonical = min((str(r.id) for r in rows), default=str(alias_id))
    return {"canonical_id": canonical, "aliases": [_serialize(r) for r in rows]}
