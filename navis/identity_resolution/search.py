"""Person search — name-substring lookup over the identity graph.

Existing `identity_resolution/router.py::/persons/lookup` requires an EXACT
alias value (e.g. the full email). The scheduling agent needs the inverse:
given a rough handle from chat ("Rahul", "Rahul Sharma"), return every
person that plausibly matches so the caller can either use the sole hit
or ask a clarification.

Deliberately narrow:
  * search hits `identity_aliases.display_name` (case-insensitive substring)
    and email `value` (case-insensitive substring on the local-part, i.e.
    before the `@`); email-domain matches are ignored (`@company.com` is
    not a person).
  * results are FOLDED by canonical equivalence class so two aliases of
    the same real person collapse to one PersonMatch.
  * ambiguity is a first-class return value — the router never picks a
    winner. `ambiguous=True` when >1 canonical class matches.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Iterable

from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from navis.identity_resolution.models import IdentityAlias
from navis.identity_resolution.resolver import canonical_alias_ids


# Cap the raw candidate scan. Anything over this is more likely a query
# that hit a common substring ("a" matching every email) than a real
# person lookup — return nothing so the caller asks for a better handle.
_MAX_CANDIDATES = 200
# Terminal cap on the folded result count. Even after equivalence-class
# folding the caller wants a short human-readable pick list.
_MAX_MATCHES = 10
# Reject too-short queries at this layer so ambiguity always means
# "your intent is ambiguous", not "your query was too broad to search".
_MIN_QUERY_LEN = 2


@dataclass(frozen=True, slots=True)
class PersonMatch:
    """One folded person from the identity graph.

    `canonical_id` is the smallest alias-uuid in the equivalence class,
    matching the convention used by `/identity/persons/lookup`. `emails`
    is the sorted set of every email alias that resolves to this person;
    `display_names` is the sorted set of every non-empty display_name
    observed (usually one, occasionally two when a person signs mail
    differently across providers).
    """

    canonical_id: str
    display_names: tuple[str, ...]
    emails: tuple[str, ...]
    alias_ids: tuple[str, ...]

    @property
    def primary_display_name(self) -> str | None:
        return self.display_names[0] if self.display_names else None

    @property
    def primary_email(self) -> str | None:
        return self.emails[0] if self.emails else None

    def to_dict(self) -> dict:
        return {
            "canonical_id": self.canonical_id,
            "display_name": self.primary_display_name,
            "display_names": list(self.display_names),
            "email": self.primary_email,
            "emails": list(self.emails),
            "alias_ids": list(self.alias_ids),
        }


@dataclass(frozen=True, slots=True)
class PersonSearchResult:
    query: str
    matches: tuple[PersonMatch, ...]

    @property
    def ambiguous(self) -> bool:
        return len(self.matches) > 1

    @property
    def single(self) -> PersonMatch | None:
        return self.matches[0] if len(self.matches) == 1 else None

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "ambiguous": self.ambiguous,
            "matches": [m.to_dict() for m in self.matches],
        }


def _local_part(email: str) -> str:
    """Portion of the email before '@' — the meaningful search target for a name."""
    return email.split("@", 1)[0] if "@" in email else email


async def search_persons(
    db: AsyncSession, org_id: uuid.UUID, query: str
) -> PersonSearchResult:
    """Return every person whose display_name or email local-part contains `query`.

    Case-insensitive. Empty / too-short queries return an empty result
    (the caller should treat that as "please ask a clarification") — we
    don't want a 1-char "R" to match every Rachel and Ravi in the org.
    """
    normalized = (query or "").strip().lower()
    if len(normalized) < _MIN_QUERY_LEN:
        return PersonSearchResult(query=normalized, matches=())

    # Escape LIKE metacharacters so a rogue `%` or `_` in the user's
    # query can't become a wildcard. Postgres LIKE uses `\` as the
    # default escape, matching the pattern.
    escaped = normalized.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    pattern = f"%{escaped}%"

    # First pass: raw alias hits (name OR email local-part). Substring on
    # the whole `value` is safe because emails include `@`, so a query
    # like "acme" only hits an address explicitly containing "acme". A
    # query with an `@` in it (e.g. "rahul@") will match on the value
    # column directly, giving domain-scoped lookups almost for free.
    stmt = (
        select(IdentityAlias)
        .where(
            IdentityAlias.organization_id == org_id,
            or_(
                IdentityAlias.display_name.ilike(pattern, escape="\\"),
                IdentityAlias.value.ilike(pattern, escape="\\"),
            ),
        )
        .limit(_MAX_CANDIDATES)
    )
    candidates: list[IdentityAlias] = list((await db.scalars(stmt)).all())
    if not candidates:
        return PersonSearchResult(query=normalized, matches=())

    # Second pass: filter email candidates whose LOCAL-part actually
    # matches (drop pure domain hits — the SQL layer treats `@company.com`
    # as a hit for query `company`, which is not what "search a person"
    # means). We keep display_name-only hits regardless.
    filtered: list[IdentityAlias] = []
    for a in candidates:
        if a.kind != "email":
            filtered.append(a)
            continue
        display_match = bool(a.display_name and normalized in a.display_name.lower())
        local_match = normalized in _local_part(a.value).lower()
        if display_match or local_match:
            filtered.append(a)
    if not filtered:
        return PersonSearchResult(query=normalized, matches=())

    # Fold by canonical equivalence class. Two aliases of the same person
    # collapse to ONE PersonMatch — the caller shouldn't be asked "which
    # Rahul?" when both hits are actually the same Rahul under two
    # providers. Canonical class is derived from the same BFS the
    # existing /persons/lookup uses.
    id_to_alias: dict[uuid.UUID, IdentityAlias] = {a.id: a for a in filtered}
    class_by_seed: dict[uuid.UUID, set[uuid.UUID]] = {}
    seen_seeds: set[uuid.UUID] = set()
    for a in filtered:
        if a.id in seen_seeds:
            continue
        class_ids = await canonical_alias_ids(db, org_id, a.id)
        # Fetch any alias in the class we don't already have in-memory
        # so downstream folding sees the complete membership set (email
        # attributed to the same person via a Slack link, etc.).
        missing = class_ids - id_to_alias.keys()
        if missing:
            extras = (await db.scalars(
                select(IdentityAlias).where(
                    IdentityAlias.organization_id == org_id,
                    IdentityAlias.id.in_(missing),
                )
            )).all()
            for e in extras:
                id_to_alias[e.id] = e
        class_by_seed[min(class_ids)] = class_ids
        seen_seeds.update(class_ids)

    matches: list[PersonMatch] = []
    for canonical_id, class_ids in class_by_seed.items():
        class_aliases: list[IdentityAlias] = [
            id_to_alias[aid] for aid in class_ids if aid in id_to_alias
        ]
        matches.append(_build_match(canonical_id, class_aliases))

    # Deterministic ordering: exact-email hits first, then by display
    # name, then by canonical id. Callers rely on `matches[0]` being the
    # most likely intent.
    matches.sort(
        key=lambda m: (
            0 if m.primary_email and m.primary_email.lower() == normalized else 1,
            (m.primary_display_name or m.primary_email or "").lower(),
            m.canonical_id,
        )
    )
    return PersonSearchResult(query=normalized, matches=tuple(matches[:_MAX_MATCHES]))


def _build_match(
    canonical_id: uuid.UUID, aliases: Iterable[IdentityAlias]
) -> PersonMatch:
    emails: set[str] = set()
    display_names: set[str] = set()
    alias_ids: list[str] = []
    for a in aliases:
        alias_ids.append(str(a.id))
        if a.kind == "email":
            emails.add(a.value)
        if a.display_name and a.display_name.strip():
            display_names.add(a.display_name.strip())
    return PersonMatch(
        canonical_id=str(canonical_id),
        display_names=tuple(sorted(display_names)),
        emails=tuple(sorted(emails)),
        alias_ids=tuple(sorted(alias_ids)),
    )
