"""Provider-agnostic alias extraction + canonical resolution.

Two responsibilities:

  1. `extract_aliases(source, payload)` — pure function from a signal payload to a
     deterministic list of `Alias` records and the *co-observation* links between them.
     Same payload → identical output. This is where per-provider knowledge lives.

  2. `canonical_alias_ids(db, org, alias_id)` — BFS over `identity_link_events` returning
     every alias in the same equivalence class as the given alias. Read-only, side-effect
     free. Callers pick the canonical (e.g. smallest UUID, or the one with `primary_email`).

Every alias is normalized before it becomes an id (`normalize_alias_value`) — case-insensitive
for emails/domains, verbatim for opaque provider ids (Slack U…, GitHub logins are case-sensitive).
Conservative on purpose: this module NEVER heuristically merges people (e.g. no `dot-plus`
folding of gmail). Only self-declared, in-record co-observations produce link events.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from email.utils import parseaddr
from enum import Enum

from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from navis.identity_resolution.ids import alias_id as compute_alias_id
from navis.identity_resolution.models import IdentityLinkEvent
from navis.signals.base import SignalSource


class AliasProvider(str, Enum):
    """The system that authored the alias. Extensible — a new provider is
    literally adding a new enum member here plus a per-provider extractor."""
    GMAIL = "gmail"
    CALENDAR = "calendar"
    SLACK = "slack"
    DRIVE = "drive"
    GITHUB = "github"
    NOTION = "notion"
    HUBSPOT = "hubspot"


class AliasKind(str, Enum):
    """The kind of identifier the alias carries. Kinds are provider-agnostic —
    an `email` from Gmail and an `email` from HubSpot resolve to the same alias
    row when the value matches, which is EXACTLY the point of this layer."""
    EMAIL = "email"
    SLACK_USER_ID = "slack_user_id"
    GITHUB_LOGIN = "github_login"
    NOTION_USER_ID = "notion_user_id"
    HUBSPOT_CONTACT_ID = "hubspot_contact_id"
    GOOGLE_USER_ID = "google_user_id"


@dataclass(frozen=True, slots=True)
class Alias:
    """One extracted alias — normalized value + a best-effort display name.

    `id` is the deterministic UUIDv5, precomputed so callers don't have to
    recompute it when writing links."""
    provider: AliasProvider
    kind: AliasKind
    value: str                    # normalized
    display_name: str | None = None

    def compute_id(self, org_id: uuid.UUID) -> uuid.UUID:
        return compute_alias_id(org_id, self.provider.value, self.kind.value, self.value)


def normalize_alias_value(kind: AliasKind, value: str) -> str:
    """Canonicalize a raw alias value.

    Emails are case-INSENSITIVE (RFC 5321 local-part comparison is provider-dependent
    but every mainstream mail system treats mailboxes case-insensitively). Opaque
    provider ids (Slack U…, GitHub logins, Notion user ids) are case-SENSITIVE and
    preserved verbatim — case-folding them can merge two distinct users.

    Trims surrounding whitespace unconditionally. Never gmail-dot-folds or
    plus-strips (those merges are irreversible and provider-behavior-specific)."""
    s = (value or "").strip()
    if kind is AliasKind.EMAIL:
        return s.lower()
    return s


# --- Per-provider extractors -------------------------------------------------
# Each returns (aliases, co_observation_pairs). A "co-observation pair" is two
# aliases the SAME signal payload asserts are the same person (e.g. a Slack
# profile snapshot that carries both a slack_user_id AND an email).
#
# Pure. Deterministic. Same payload → identical output.

def _gmail_aliases(payload: dict) -> tuple[list[Alias], list[tuple[Alias, Alias]]]:
    aliases: list[Alias] = []
    seen: set[tuple[str, str]] = set()
    for role in ("from", "to", "cc"):
        raw = payload.get(role)
        if not raw:
            continue
        for chunk in str(raw).split(","):
            name, email = parseaddr(chunk)
            value = normalize_alias_value(AliasKind.EMAIL, email)
            if not value or (AliasKind.EMAIL.value, value) in seen:
                continue
            seen.add((AliasKind.EMAIL.value, value))
            aliases.append(Alias(
                provider=AliasProvider.GMAIL,
                kind=AliasKind.EMAIL,
                value=value,
                display_name=(name or None),
            ))
    return aliases, []  # Gmail carries no cross-provider co-observations


def _calendar_aliases(payload: dict) -> tuple[list[Alias], list[tuple[Alias, Alias]]]:
    aliases: list[Alias] = []
    seen: set[str] = set()
    for att in payload.get("attendees") or []:
        email = (att.get("email") if isinstance(att, dict) else str(att)) or ""
        value = normalize_alias_value(AliasKind.EMAIL, email)
        if not value or value in seen:
            continue
        seen.add(value)
        name = att.get("displayName") if isinstance(att, dict) else None
        aliases.append(Alias(
            provider=AliasProvider.CALENDAR,
            kind=AliasKind.EMAIL,
            value=value,
            display_name=name,
        ))
    return aliases, []


def _slack_aliases(payload: dict) -> tuple[list[Alias], list[tuple[Alias, Alias]]]:
    aliases: list[Alias] = []
    uid = payload.get("slack_user_id")
    email = payload.get("email")

    slack_alias: Alias | None = None
    if uid:
        value = normalize_alias_value(AliasKind.SLACK_USER_ID, str(uid))
        slack_alias = Alias(
            provider=AliasProvider.SLACK,
            kind=AliasKind.SLACK_USER_ID,
            value=value,
            display_name=payload.get("display_name") or payload.get("name"),
        )
        aliases.append(slack_alias)

    email_alias: Alias | None = None
    if email:
        value = normalize_alias_value(AliasKind.EMAIL, str(email))
        if value:
            email_alias = Alias(
                provider=AliasProvider.SLACK,
                kind=AliasKind.EMAIL,
                value=value,
                display_name=payload.get("display_name") or payload.get("name"),
            )
            aliases.append(email_alias)

    # Structural co-observation: a Slack profile that self-declares an email
    # is authoritative — the SAME record binds the two.
    pairs: list[tuple[Alias, Alias]] = []
    if slack_alias is not None and email_alias is not None:
        pairs.append((slack_alias, email_alias))
    return aliases, pairs


_SOURCE_EXTRACTORS = {
    SignalSource.GMAIL: _gmail_aliases,
    SignalSource.CALENDAR: _calendar_aliases,
    SignalSource.SLACK: _slack_aliases,
}


def extract_aliases(
    source: SignalSource, payload: dict
) -> tuple[list[Alias], list[tuple[Alias, Alias]]]:
    """Pure alias + co-observation extraction from a signal payload.

    Unknown sources return (empty, empty) — the layer is safely no-op until a
    provider-specific extractor is added. Future GitHub / Notion / HubSpot
    connectors add themselves here without touching Memory or any other layer."""
    fn = _SOURCE_EXTRACTORS.get(source)
    if fn is None:
        return [], []
    return fn(payload or {})


# --- Canonical resolution (READ-ONLY BFS over link events) -------------------

async def canonical_alias_ids(
    db: AsyncSession, org_id: uuid.UUID, seed: uuid.UUID, *, max_depth: int = 32
) -> set[uuid.UUID]:
    """Return the equivalence class of `seed` — every alias reachable via any
    number of link edges. Bounded by `max_depth` iterations so a pathological
    fully-connected component can't spin forever; per-org max scans is O(edges).

    READ-ONLY: never writes; the equivalence class is a projection of the
    append-only link log. Compatible with future KG projectors that materialize
    the same equivalence class into a `person_canonical` snapshot table."""
    visited: set[uuid.UUID] = {seed}
    frontier: set[uuid.UUID] = {seed}
    for _ in range(max_depth):
        if not frontier:
            break
        rows = (await db.scalars(
            select(IdentityLinkEvent).where(
                IdentityLinkEvent.organization_id == org_id,
                or_(
                    IdentityLinkEvent.alias_a_id.in_(frontier),
                    IdentityLinkEvent.alias_b_id.in_(frontier),
                ),
            )
        )).all()
        next_frontier: set[uuid.UUID] = set()
        for r in rows:
            for aid in (r.alias_a_id, r.alias_b_id):
                if aid not in visited:
                    visited.add(aid)
                    next_frontier.add(aid)
        frontier = next_frontier
    return visited
