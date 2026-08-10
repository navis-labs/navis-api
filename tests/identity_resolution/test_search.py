"""Person-search invariants — case-insensitive name/local-part hits,
ambiguity semantics, canonical folding, minimum-query enforcement.

These tests run the real `search_persons` function against a small
fake async-session that mirrors the surface `search_persons` uses:
`db.scalars(select(...))`, `db.scalar(select(...))`. That's enough to
verify the SQL predicate and the equivalence-class folding without
paying the UUID+JSONB tax of running the pg-typed models on sqlite.
"""
from __future__ import annotations

import uuid
from typing import Any, Iterable, Sequence

import pytest

from navis.identity_resolution.models import IdentityAlias, IdentityLinkEvent
from navis.identity_resolution.search import search_persons

ORG = uuid.UUID("11111111-1111-1111-1111-111111111111")
SIG = uuid.UUID("22222222-2222-2222-2222-222222222222")


# ── Fake async session ─────────────────────────────────────────────────


class _Scalars:
    def __init__(self, rows: Sequence[Any]) -> None:
        self._rows = list(rows)

    def all(self) -> list[Any]:
        return list(self._rows)


class _FakeSession:
    """Minimal read surface `search_persons` calls.

    Instead of parsing SQLAlchemy compiled predicates, we re-implement
    the SAME predicate logic in Python here — but ONLY for the exact
    query shapes `search_persons` and `canonical_alias_ids` emit. If
    those queries change shape (new .where clause, different table),
    this fake needs to grow — the mismatch is the test signal.
    """

    def __init__(self) -> None:
        self.aliases: list[IdentityAlias] = []
        self.links: list[IdentityLinkEvent] = []

    def add_alias(
        self,
        *,
        provider: str,
        email: str,
        display_name: str | None = None,
        org_id: uuid.UUID = ORG,
    ) -> IdentityAlias:
        # Simulate the deterministic id. Storage here is a plain object;
        # the real Postgres path uses UUIDv5 but we only need identity
        # uniqueness within the fake.
        aid = uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"{org_id}|{provider}|email|{email.lower()}",
        )
        row = IdentityAlias(
            id=aid,
            organization_id=org_id,
            provider=provider,
            kind="email",
            value=email.lower(),
            display_name=display_name,
        )
        self.aliases.append(row)
        return row

    def add_link(self, a: uuid.UUID, b: uuid.UUID) -> None:
        row = IdentityLinkEvent(
            id=uuid.uuid5(uuid.NAMESPACE_URL, f"link|{a}|{b}"),
            organization_id=ORG,
            alias_a_id=a,
            alias_b_id=b,
            confidence=1.0,
            evidence={},
        )
        self.links.append(row)

    async def scalars(self, stmt):
        # The library-visible entity for a compiled select is on
        # stmt.column_descriptions[0]["type"]. We branch on that.
        entity = _entity_from_stmt(stmt)
        if entity is IdentityAlias:
            return _Scalars(_apply_alias_filter(stmt, self.aliases))
        if entity is IdentityLinkEvent:
            return _Scalars(_apply_link_filter(stmt, self.links))
        raise AssertionError(f"_FakeSession.scalars: unhandled entity {entity!r}")


def _entity_from_stmt(stmt) -> type:
    # SQLAlchemy 2.x exposes the selected entity as stmt.column_descriptions.
    desc = stmt.column_descriptions
    if desc:
        return desc[0]["type"]
    raise AssertionError("could not determine selected entity")


def _apply_alias_filter(
    stmt, rows: Iterable[IdentityAlias]
) -> list[IdentityAlias]:
    """Emulate the WHERE + ilike predicates search_persons emits."""
    # search_persons uses:
    #   where(org == org_id, or_(display_name ilike pat, value ilike pat))
    #   limit(N)
    # canonical_alias_ids uses:
    #   where(org == org_id, id.in_(candidate_ids))
    #
    # We look at the raw WHERE clause text and decide which mode.
    where_text = str(stmt.whereclause)
    limit_val = stmt._limit_clause.value if stmt._limit_clause is not None else None

    if ".id IN (" in where_text or "identity_aliases.id IN" in where_text:
        params = stmt.compile().params
        wanted = _uuids_from_params(params)
        org_id = _extract_org_id(params)
        matched = [
            r for r in rows if r.organization_id == org_id and r.id in wanted
        ]
        return matched[: (limit_val or len(matched))]

    # Substring predicate — pull the query text out of bind params.
    params = stmt.compile().params
    pattern = None
    for k, v in params.items():
        if isinstance(v, str) and v.startswith("%") and v.endswith("%"):
            pattern = v[1:-1].lower()
            break
    if pattern is None:
        return []
    org_id = _extract_org_id(params)
    matched = [
        r
        for r in rows
        if r.organization_id == org_id
        and (
            (r.display_name and pattern in r.display_name.lower())
            or (r.value and pattern in r.value.lower())
        )
    ]
    return matched[: (limit_val or len(matched))]


def _apply_link_filter(
    stmt, rows: Iterable[IdentityLinkEvent]
) -> list[IdentityLinkEvent]:
    """Emulate canonical_alias_ids' link-lookup: rows whose alias_a_id
    OR alias_b_id is in the frontier set."""
    params = stmt.compile().params
    wanted = _uuids_from_params(params)
    org_id = _extract_org_id(params)
    return [
        r
        for r in rows
        if r.organization_id == org_id
        and (r.alias_a_id in wanted or r.alias_b_id in wanted)
    ]


def _uuids_from_params(params: dict) -> set[uuid.UUID]:
    """Extract every UUID out of a compiled statement's bind params,
    including UUIDs nested inside list-typed `.in_()` params (which
    SQLAlchemy defers as a single key with a list value at compile
    time — see stmt.compile().params for the `.in_()` shape)."""
    out: set[uuid.UUID] = set()
    for _k, v in params.items():
        if isinstance(v, uuid.UUID):
            out.add(v)
        elif isinstance(v, (list, tuple, set)):
            for x in v:
                if isinstance(x, uuid.UUID):
                    out.add(x)
        elif isinstance(v, str):
            try:
                out.add(uuid.UUID(v))
            except ValueError:
                pass
    return out


def _extract_org_id(params: dict) -> uuid.UUID | None:
    """Pull the org_id bind param out of a compiled predicate.

    Convention across the codebase: the org filter is always
    `WHERE organization_id = :organization_id_1` (or a like-named
    param). Match on the KEY containing "organization" so we don't
    accidentally return a same-typed UUID (e.g. an alias_id) when
    the org param comes later in the dict.
    """
    for k, v in params.items():
        if isinstance(v, uuid.UUID) and "organization" in k.lower():
            return v
    # Fallback for older stmts that used `org_id` as the bind key.
    for k, v in params.items():
        if isinstance(v, uuid.UUID) and ("org" in k.lower()):
            return v
    return None


# ── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def db() -> _FakeSession:
    return _FakeSession()


# ── Tests ───────────────────────────────────────────────────────────────


async def test_search_by_display_name_hits(db: _FakeSession):
    db.add_alias(
        provider="gmail", email="rahul@partner.com", display_name="Rahul Sharma"
    )
    result = await search_persons(db, ORG, "rahul")
    assert not result.ambiguous
    assert result.single is not None
    assert result.single.primary_email == "rahul@partner.com"


async def test_search_by_email_local_part_hits(db: _FakeSession):
    db.add_alias(provider="gmail", email="rahul@partner.com")
    result = await search_persons(db, ORG, "rahul")
    assert result.single is not None
    assert result.single.primary_email == "rahul@partner.com"


async def test_search_ignores_domain_only_hits(db: _FakeSession):
    """A query like `partner` must NOT hit an address only because its
    domain contains the substring — search_persons filters those out
    post-SQL in the local-part-check step."""
    db.add_alias(provider="gmail", email="alice@partner.com", display_name="Alice")
    result = await search_persons(db, ORG, "partner")
    assert result.matches == ()


async def test_search_is_case_insensitive(db: _FakeSession):
    db.add_alias(
        provider="gmail", email="rahul@partner.com", display_name="Rahul"
    )
    up = await search_persons(db, ORG, "RaHul")
    assert up.single is not None
    assert up.single.primary_email == "rahul@partner.com"


async def test_short_query_returns_empty(db: _FakeSession):
    db.add_alias(provider="gmail", email="rahul@partner.com")
    result = await search_persons(db, ORG, "r")
    assert result.matches == ()


async def test_ambiguity_when_multiple_persons_match(db: _FakeSession):
    db.add_alias(
        provider="gmail", email="rahul.a@partner.com", display_name="Rahul Agarwal"
    )
    db.add_alias(
        provider="gmail", email="rahul.s@other.com", display_name="Rahul Sharma"
    )
    result = await search_persons(db, ORG, "rahul")
    assert result.ambiguous
    assert len(result.matches) == 2
    emails = {m.primary_email for m in result.matches}
    assert emails == {"rahul.a@partner.com", "rahul.s@other.com"}


async def test_canonical_folding_collapses_linked_aliases(db: _FakeSession):
    """Two aliases (Gmail + Slack) explicitly linked resolve to ONE person."""
    a = db.add_alias(provider="gmail", email="rahul@partner.com", display_name="Rahul")
    b = db.add_alias(provider="slack", email="rahul@partner.com", display_name="Rahul")
    db.add_link(a.id, b.id)
    result = await search_persons(db, ORG, "rahul")
    assert not result.ambiguous, "linked aliases must fold to one match"
    assert result.single is not None
    assert result.single.primary_email == "rahul@partner.com"


async def test_search_scoped_by_org(db: _FakeSession):
    """A cross-org row must not leak into another tenant's result."""
    other_org = uuid.UUID("99999999-9999-9999-9999-999999999999")
    db.add_alias(
        provider="gmail",
        email="rahul@partner.com",
        display_name="Rahul",
        org_id=other_org,
    )
    result = await search_persons(db, ORG, "rahul")
    assert result.matches == ()
