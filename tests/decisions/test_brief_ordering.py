"""`latest_brief` must never return an undated brief ahead of a dated one.

The bug this pins: Postgres orders NULLs FIRST under `DESC`, and a brief written
before an org's first sync has `as_of = NULL` with zero decisions. A plain
`desc(as_of)` therefore returned that empty row ahead of every real brief, for
good. `POST /brief/generate` produced correct decisions, the dashboard rendered
them, and the next SWR revalidation of `GET /brief/today` blanked the page about
a second later. Every generated brief was unreadable through this one method.

This module holds the DB-free guard so the regression is caught even where no
Postgres is configured. The behavioural proof — actual NULL ordering against a
real server — lives in tests/integration/test_brief_ordering_postgres.py, since
that semantic belongs to Postgres and no fake can honestly reproduce it.
"""
from __future__ import annotations

import uuid

from sqlalchemy.dialects import postgresql

from navis.decisions.models import DailyBrief


def _compiled_latest_brief_sql() -> str:
    """Compile the exact statement `DecisionService.latest_brief` builds.

    Reaching into the service would need a live AsyncSession, so this mirrors
    the query. `test_latest_brief_source_uses_nulls_last` below keeps the mirror
    honest by asserting the real source carries the same ordering.
    """
    from sqlalchemy import desc, select

    stmt = (
        select(DailyBrief)
        .where(DailyBrief.organization_id == uuid.uuid4())
        .order_by(
            desc(DailyBrief.as_of).nulls_last(),
            desc(DailyBrief.reasoning_version),
            desc(DailyBrief.id),
        )
        .limit(1)
    )
    return str(stmt.compile(dialect=postgresql.dialect()))


def test_ordering_emits_nulls_last():
    sql = _compiled_latest_brief_sql().upper()
    assert "ORDER BY" in sql
    assert "NULLS LAST" in sql, (
        "latest_brief must order NULLS LAST — without it Postgres puts the "
        "undated onboarding brief first and the dashboard reads an empty brief"
    )


def test_latest_brief_source_uses_nulls_last():
    """The real source, not a mirror of it.

    A plain `desc(DailyBrief.as_of)` reintroduces the bug silently — the code
    still runs, still returns a brief, and only the WRONG one. Asserting on the
    source text is blunt, but it is the thing that actually protects the fix on
    a machine with no Postgres.
    """
    import inspect

    from navis.decisions.service import DecisionService

    src = inspect.getsource(DecisionService.latest_brief)
    assert "nulls_last()" in src, (
        "latest_brief lost its NULLS LAST ordering; an undated brief will "
        "outrank every dated one again"
    )


# ---- the ordering must be TOTAL, not just NULL-safe --------------------------
#
# `as_of` is State's logical time, so reprojecting ONE snapshot under corrected
# reasoning writes a second brief with a byte-identical `as_of`. Live, org
# 071ae55b holds two briefs at 2026-08-17 16:11:12 (reasoning v3 and v4) and
# df7d1c0a holds the same pair at 15:26:36 — so `as_of DESC LIMIT 1` alone chose
# between a corrected brief and the one it corrected by scan order.


def test_ordering_breaks_ties_on_the_reasoning_version():
    sql = _compiled_latest_brief_sql().upper()
    order_by = sql.split("ORDER BY", 1)[1]
    assert "REASONING_VERSION DESC" in order_by, (
        "latest_brief must break an as_of tie on reasoning_version — two "
        "projections of one snapshot share as_of, and the older one can win"
    )
    assert order_by.index("AS_OF DESC") < order_by.index("REASONING_VERSION DESC"), (
        "reasoning_version must be the TIE-BREAK, not the primary key: a brief "
        "describing a newer snapshot stays the current one"
    )
    assert order_by.rindex("ID DESC") > order_by.index("REASONING_VERSION DESC"), (
        "the ordering must be total — `id` is uuid5 over "
        "(org, reasoning_version, state_watermark), so it always settles the rest"
    )


def test_latest_brief_source_orders_by_reasoning_version():
    """The real source, not the mirror. Dropping the tie-break reintroduces the
    bug in its quietest form: the query still returns a brief, just the wrong one,
    and only sometimes."""
    import inspect

    from navis.decisions.service import DecisionService

    src = inspect.getsource(DecisionService.latest_brief)
    assert "desc(DailyBrief.reasoning_version)" in src, (
        "latest_brief lost its reasoning_version tie-break; a superseded brief "
        "can outrank the corrected projection of the same snapshot"
    )
    assert "desc(DailyBrief.id)" in src, "latest_brief's ordering is no longer total"
