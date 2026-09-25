"""Organization-lifecycle seam (P2 extraction prep) + external-account tenancy.

Per the Identity Law, organization profile/lifecycle belongs to a future
Organization V1 domain, NOT to Identity. To keep that extraction a *move* rather
than a *rewrite*, all organization mutations funnel through this one service.
Today it wraps the existing `organizations` table; tomorrow its body is replaced
by a call into the Organization domain and Identity keeps only the seam.

This module ALSO owns `resolve_owning_integration` — the one answer to "which
tenant does this external account belong to?". That lives here because Identity
owns the `Integration` model and its charter is explicit (see
`navis/identity/__init__.py`): *every domain is scoped by organization_id;
Identity enforces that tenancy boundary*. Inbound provider events name an
external account, never an organization, so every webhook path needs exactly
this lookup — and needs it to answer the same way every time.
"""
from __future__ import annotations

import re
import uuid
from typing import Callable, Sequence

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from navis.identity.models import (
    Integration,
    IntegrationProvider,
    Organization,
    OrganizationKind,
)

# 8 hex chars of salt (was 6 -> 16.7M space; 8 -> ~4.3e9) plus the retry loop below.
# Correctness comes from the retry, not from the (low) collision probability.
_SALT_LEN = 8
_MAX_SLUG_ATTEMPTS = 4


class OrganizationError(Exception):
    """Org creation failed (empty name, or no unique slug after retries)."""


def _slugify(name: str, salt: str) -> str:
    base = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "org"
    return f"{base}-{salt[:_SALT_LEN]}"


class OrgLifecycleSeam:
    """The single chokepoint for org creation/lookup that Identity depends on.

    NAMED FOR WHAT IT IS, not for the table it touches. This class was called
    `OrganizationService`, which is also the name of
    `navis.organization.service.OrganizationService` — an entirely different
    class (the capability authority + profile/entitlement/billing front door).
    Two unrelated types sharing one name across two modules made every
    `OrganizationService(db)` call site ambiguous on sight and made an import
    typo resolve to a working-but-wrong object. This is the Identity-side SEAM
    only: create an org row, read an org row. Nothing else.
    """

    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def create(
        self,
        *,
        name: str,
        slug_salt: str | None = None,
        kind: OrganizationKind = OrganizationKind.COMPANY,
    ) -> Organization:
        """Create a tenant. `kind` says what it IS, never what it may do.

        DEFAULTS TO COMPANY, so every existing call site produces exactly the row
        it produced before — which is what `0051_surface_rbac_columns` chose the
        column default for. A caller that wants a personal workspace must ask for
        one; nothing infers it from the shape of the signup.

        `kind` carries PRODUCT meaning only. It is not read by
        `effective_permissions`, it is not an input to any gate, and no query
        branches on it: isolation is the same `organization_id` predicate for
        both kinds. See `OrganizationKind`.
        """
        if not name or not name.strip():
            raise OrganizationError("organization name must not be empty")
        name = name.strip()

        # Production correctness must not depend on slug-collision probability. Insert
        # inside a SAVEPOINT so a UNIQUE(slug) violation rolls back ONLY this insert
        # (not the surrounding signup transaction), then retry with a fresh salt.
        for _ in range(_MAX_SLUG_ATTEMPTS):
            slug = _slugify(name, slug_salt or uuid.uuid4().hex)
            try:
                async with self.db.begin_nested():  # SAVEPOINT
                    org = Organization(name=name, slug=slug, kind=kind.value)
                    self.db.add(org)
                    await self.db.flush()
                return org
            except IntegrityError:
                # A caller-pinned salt is deterministic — retrying yields the same slug,
                # so fail fast instead of looping on a guaranteed collision.
                if slug_salt is not None:
                    raise OrganizationError(f"slug '{slug}' already exists")
                continue
        raise OrganizationError("could not allocate a unique organization slug")

    async def get(self, org_id: uuid.UUID) -> Organization | None:
        return await self.db.get(Organization, org_id)


# ── External account -> tenant ───────────────────────────────────────────────
#
# WHY THIS IS ONE FUNCTION AND NOT THREE INLINE QUERIES.
#
# `uq_org_provider_account` is unique per ORGANIZATION, not globally, and that is
# deliberate: two companies may legitimately connect the same shared mailbox or the
# same Slack workspace. The consequence is that (provider, external_account) is a
# ONE-TO-MANY relation onto organizations, so every inbound provider event —
# which names an external account and never an organization — has to decide what
# to do when it matches more than one tenant.
#
# `db.scalar()` decides by returning whatever row the planner produced first. That
# is not a tie-break, it is a coin flip, and it silently attributes one company's
# mail to another. Gmail's push path already refuses instead; this function is
# that refusal, extracted so the Slack path and the observation webhook path
# cannot answer the question differently.
#
# FAIL CLOSED IS THE ONLY SAFE DEFAULT. There is no ordering — newest, oldest,
# lowest id — that makes an arbitrary choice correct, because which tenant an
# event belongs to is a fact about the world, not a preference.


class AmbiguityReason:
    """Why a tenant could not be resolved. Bounded set — safe as a metric label."""

    NO_ACCOUNT = "no_account"          # the event named no external account
    NO_MATCH = "no_match"              # no active integration has that account
    AMBIGUOUS_TENANT = "ambiguous_tenant"  # >1 organization connected that account


async def integrations_for_external_account(
    db: AsyncSession, *, provider: IntegrationProvider, external_account: str
) -> Sequence[Integration]:
    """Every ACTIVE integration for this (provider, external account), any org.

    Deliberately returns ALL matches rather than one: the caller must SEE the
    ambiguity in order to refuse it. A helper that returned a single row would
    reintroduce exactly the silent tie-break this module exists to remove.
    """
    return (
        await db.scalars(
            select(Integration).where(
                Integration.provider == provider,
                Integration.external_account == external_account,
                Integration.is_active.is_(True),
            )
        )
    ).all()


async def resolve_owning_integration(
    db: AsyncSession,
    *,
    provider: IntegrationProvider,
    external_account: str | None,
    narrow: Callable[[Sequence[Integration]], Sequence[Integration]] | None = None,
) -> tuple[Integration | None, str | None]:
    """The ONE integration that owns this external account, or a refusal reason.

    Returns `(integration, None)` on a unique resolution and `(None, reason)`
    otherwise, where `reason` is an `AmbiguityReason` member.

    `narrow` is an optional provider-specific disambiguator applied ONLY when the
    account matches more than one candidate — it lets a connector that carries a
    stronger install identity in its payload (Slack's per-install bot user id, for
    example) legitimately serve a shared workspace. It may only ever SHRINK the
    candidate set: whatever it returns is re-checked for uniqueness below, so a
    narrower that returns several rows, or rows from several organizations, still
    refuses. A narrower cannot widen a match or invent a tenant.

    Uniqueness is asserted on the ORGANIZATION, not on the row count. One
    organization holding two rows for the same account (a Google-family clone, say)
    is unambiguous about tenancy; two organizations holding one row each is not.
    """
    if not external_account:
        return None, AmbiguityReason.NO_ACCOUNT

    rows = await integrations_for_external_account(
        db, provider=provider, external_account=external_account
    )
    if not rows:
        return None, AmbiguityReason.NO_MATCH

    if len({r.organization_id for r in rows}) > 1 and narrow is not None:
        # Only consulted under genuine cross-tenant ambiguity, and only ever to
        # shrink. The result falls through to the same uniqueness check below.
        narrowed = list(narrow(rows))
        if narrowed:
            rows = narrowed

    if len({r.organization_id for r in rows}) > 1:
        return None, AmbiguityReason.AMBIGUOUS_TENANT
    return rows[0], None
