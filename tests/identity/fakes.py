"""A minimal in-memory AsyncSession good enough to exercise Identity services.

Identity services talk directly to AsyncSession (no repo seam), and the ORM uses
postgres-only column types (JSONB/UUID) that SQLite can't create. Rather than
require a live Postgres for unit tests, this fake interprets the exact, simple
`select().where(...)` shapes the services use (eq / is_ / is_not / < / in_),
plus order_by + limit. It is deliberately small, not a general SQL engine.
"""
from __future__ import annotations

import itertools
import uuid
from datetime import datetime, timezone

_clock = itertools.count()


def _pk_name(model) -> str:
    return list(model.__table__.primary_key.columns)[0].name


def _eval(obj, clause) -> bool:
    # AND-list of sub-clauses
    if hasattr(clause, "clauses") and not hasattr(clause, "left"):
        return all(_eval(obj, c) for c in clause.clauses)
    key = clause.left.key
    op = clause.operator.__name__
    left = getattr(obj, key, None)
    right = getattr(clause.right, "value", None)
    rtype = type(clause.right).__name__  # True_/False_/Null for `IS <literal>`
    if op == "eq":
        return left == right
    if op in ("is_", "is"):
        if rtype == "True_":
            return left is True
        if rtype == "False_":
            return left is False
        return left is None  # IS NULL
    if op in ("isnot", "is_not"):
        if rtype == "True_":
            return left is not True
        if rtype == "False_":
            return left is not False
        return left is not None  # IS NOT NULL
    if op == "lt":
        return left is not None and left < right
    if op in ("in_op", "in_"):
        vals = right if isinstance(right, (list, tuple)) else [
            getattr(c, "value", c) for c in getattr(clause.right, "clauses", [])
        ]
        return left in vals
    raise NotImplementedError(f"fake session: unsupported operator {op}")


class _RowCount:
    """Mimics the CursorResult returned by a bulk UPDATE/DELETE (.rowcount only)."""

    def __init__(self, rowcount):
        self.rowcount = rowcount


class _Result:
    def __init__(self, rows):
        self._rows = list(rows)

    def all(self):
        return self._rows

    def first(self):
        return self._rows[0] if self._rows else None


def _apply_column_defaults(obj) -> None:
    """Fill scalar column defaults, the way a real flush does.

    WHY THE FAKE HAS TO DO THIS. SQLAlchemy applies `default=` at INSERT time,
    not at construction, so an object added to a session that never talks to a
    database keeps `None` in every defaulted column. A real session does not:
    by the time anything reads the object back, the value is there.

    That difference is invisible until a column becomes NOT NULL and something
    serializes it — `organizations.kind` is the first — at which point every
    test that builds a row without naming it gets a validation error for a
    state production cannot reach, because the database's own NOT NULL
    guarantees the column is populated on every row that is read.

    Only PLAIN SCALARS are applied. A callable or context-sensitive default is
    SQLAlchemy's to evaluate and guessing at it here would make the fake
    disagree with the real thing in a new way, which is the problem this
    function exists to remove rather than to relocate.
    """
    mapper = getattr(type(obj), "__mapper__", None)
    if mapper is None:
        return
    for column in mapper.columns:
        default = column.default
        if default is None or not getattr(default, "is_scalar", False):
            continue
        attr = mapper.get_property_by_column(column).key
        if getattr(obj, attr, None) is None:
            setattr(obj, attr, default.arg)


class FakeSession:
    def __init__(self):
        self.store: dict[type, list] = {}

    # ---- writes ----
    def add(self, obj):
        model = type(obj)
        pk = _pk_name(model)
        if getattr(obj, pk, None) is None:
            setattr(obj, pk, uuid.uuid4())
        # Fill server-default timestamps so ordering works deterministically.
        tick = datetime.fromtimestamp(1_700_000_000 + next(_clock), tz=timezone.utc)
        for col in ("created_at", "updated_at", "occurred_at", "revoked_at_unused"):
            if hasattr(obj, col) and getattr(obj, col, None) is None and col != "revoked_at_unused":
                setattr(obj, col, tick)
        _apply_column_defaults(obj)
        self.store.setdefault(model, [])
        if obj not in self.store[model]:
            self.store[model].append(obj)

    def add_all(self, objs):
        for o in objs:
            self.add(o)

    async def delete(self, obj):
        rows = self.store.get(type(obj), [])
        if obj in rows:
            rows.remove(obj)

    async def flush(self):
        return None

    def begin_nested(self):
        """A SAVEPOINT that never has to roll back.

        `IdentityService.signup`, `google_signin` and `OrgLifecycleSeam.create`
        all wrap their INSERT in `async with db.begin_nested()` so that a
        UNIQUE violation rolls back only that statement. This fake has no
        constraints and therefore no violations to contain, so the honest
        stand-in is a no-op async context manager.

        IT IS NOT A TRANSACTION, and no test may rely on it behaving like one:
        nothing is undone if the body raises. What it buys is the ability to
        exercise the org-CREATION path at all — which was previously untestable
        here (see the note at the top of `test_google_signin_resolution.py`),
        so the one branch that decides what a new workspace IS had no unit
        cover.
        """
        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def _savepoint():
            yield self

        return _savepoint()

    async def execute(self, stmt):
        """Minimal support for bulk UPDATE / DELETE, plus the two-entity JOIN.

        THE JOIN BRANCH EXISTS BECAUSE A REAL ROUTE STARTED USING ONE. P0-8
        moved `GET /identity/members` onto `list_members_with_users` so the
        response could carry each member's email instead of a bare UUID, and
        `find_member_by_email` (SCIM) has always used the same shape. Without
        this, every test that exercises those routes 500s on a statement the
        fake silently did not understand — which reads as a broken route
        rather than as a gap in the fake.

        Handled once here rather than at the call sites, for the same reason
        `_apply_column_defaults` is: the alternative is each test reaching
        past the service it is meant to be testing.

        Deliberately narrow — `select(A, B).join(B, A.fk == B.pk)`, which is
        the only join shape this codebase's identity layer uses. Anything else
        falls through to the UPDATE/DELETE path exactly as before.
        """
        from sqlalchemy import Delete, Select, Update

        if isinstance(stmt, Select) and len(stmt.column_descriptions) == 2:
            return _Result(self._join(stmt))

        tname = getattr(getattr(stmt, "table", None), "name", None)
        model = next(
            (m for m in self.store if getattr(m, "__tablename__", None) == tname), None
        )
        rows = self.store.get(model, []) if model is not None else []
        where = stmt.whereclause
        affected = [r for r in rows if (where is None or _eval(r, where))]
        if isinstance(stmt, Update):
            for col, val in stmt._values.items():
                value = getattr(val, "value", val)
                for r in affected:
                    setattr(r, col.key, value)
        elif isinstance(stmt, Delete) and model is not None:
            self.store[model] = [r for r in rows if r not in affected]
        return _RowCount(len(affected))

    # ---- reads ----
    async def get(self, model, pk):
        pk_name = _pk_name(model)
        for row in self.store.get(model, []):
            if getattr(row, pk_name) == pk:
                return row
        return None

    def _join(self, stmt):
        """`select(A, B).join(B, A.fk == B.pk)` as (A, B) tuples.

        The LEFT side is resolved by the ordinary `_query` path, so the where
        clause, ordering, limit and offset all behave exactly as they do for a
        single-entity select. Each left row is then paired with the right row
        the ON clause points at; a left row whose partner is missing is
        DROPPED, which is what an inner join does.
        """
        left_model = stmt.column_descriptions[0]["entity"]
        right_model = stmt.column_descriptions[1]["entity"]

        # The ON clause names one column on each side. Read them off rather
        # than assuming `user_id`/`id`, so a second join shape does not
        # silently pair the wrong rows.
        onclause = stmt._setup_joins[0][1] if stmt._setup_joins else None
        left_key, right_key = "id", "id"
        if onclause is not None:
            for side in (onclause.left, onclause.right):
                table = getattr(getattr(side, "table", None), "name", None)
                if table == getattr(left_model, "__tablename__", None):
                    left_key = side.key
                elif table == getattr(right_model, "__tablename__", None):
                    right_key = side.key

        right_rows = self.store.get(right_model, [])
        paired = []
        for left in self._query(stmt, model=left_model):
            match = next(
                (r for r in right_rows if getattr(r, right_key) == getattr(left, left_key)),
                None,
            )
            if match is not None:
                paired.append((left, match))
        return paired

    def _query(self, stmt, model=None):
        model = model or stmt.column_descriptions[0]["entity"]
        rows = list(self.store.get(model, []))
        where = stmt.whereclause
        if where is not None:
            rows = [r for r in rows if _eval(r, where)]
        for oc in reversed(getattr(stmt, "_order_by_clauses", ()) or ()):
            desc = getattr(oc, "modifier", None)
            col = getattr(oc, "element", oc)
            key = col.key
            rows.sort(key=lambda r: (getattr(r, key) is not None, getattr(r, key)),
                      reverse=bool(desc and "desc" in str(desc).lower()))
        limit = getattr(stmt, "_limit", None)
        if limit is None:
            limit = getattr(getattr(stmt, "_limit_clause", None), "value", None)
        if limit:
            rows = rows[:limit]
        return rows

    async def scalar(self, stmt):
        rows = self._query(stmt)
        return rows[0] if rows else None

    async def scalars(self, stmt):
        return _Result(self._query(stmt))


ORG_A = uuid.UUID("11111111-1111-1111-1111-111111111111")
ORG_B = uuid.UUID("22222222-2222-2222-2222-222222222222")
ACTOR = uuid.UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
