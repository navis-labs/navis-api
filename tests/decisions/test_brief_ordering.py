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
        .order_by(desc(DailyBrief.as_of).nulls_last())
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
