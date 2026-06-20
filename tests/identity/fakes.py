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

    async def execute(self, stmt):
        """Minimal support for bulk UPDATE / DELETE (used by revoke/purge paths)."""
        from sqlalchemy import Delete, Update

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

    def _query(self, stmt):
        model = stmt.column_descriptions[0]["entity"]
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
