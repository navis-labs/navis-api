"""Locks the /admin/signals/prune endpoint against silent regressions.

The endpoint is destructive (deletes memory + signals + outbox + DLQ + briefs +
state-snapshot for the current org, then rebuilds state from what remains). Every
guarantee below is safety-critical:

  * confirm=false MUST NOT delete anything (double-gate against fat-finger).
  * older_than_days MUST be bounded [7, 3650] at the pydantic layer — a smaller
    number is more likely a typo than intent, a bigger one is unbounded stall.
  * The route MUST sit behind Permission.MANAGE_INTEGRATIONS. View-only roles
    (employee) getting through this gate would be a full-org wipe primitive.
  * DELETE order MUST be memory → signals → outbox → DLQ → briefs → snapshot —
    memory_records has an ondelete=RESTRICT FK back to signal_events, so any
    other order raises a FK violation mid-transaction.
  * Every DELETE MUST scope to ctx.org_id — cross-tenant leakage here would be
    a full-tenant wipe from any authenticated founder.
  * State rebuild MUST run AFTER commit so the fold sees the post-purge memory,
    and a rebuild exception MUST NOT undo the successful purge.
"""
from __future__ import annotations

import inspect
import uuid
from dataclasses import dataclass

import pydantic
import pytest
from fastapi import HTTPException
from sqlalchemy import Delete

from navis.api import admin_router as R
from navis.decisions.models import DailyBrief
from navis.identity.permissions import Permission, default_permissions_for_role
from navis.memory.models import MemoryRecord
from navis.signals.models import SignalDeadLetter, SignalEventRecord, SignalOutbox
from navis.state.v2.snapshots import StateSnapshot


ORG = uuid.UUID("11111111-1111-1111-1111-111111111111")
OTHER_ORG = uuid.UUID("22222222-2222-2222-2222-222222222222")


@dataclass
class _Ctx:
    """Stand-in for CurrentContext — the endpoint only reads .org_id."""
    org_id: uuid.UUID
    # unused fields kept so any future .role / .user access surfaces loudly
    role: str = "founder"
    user: object = None


class _RowCount:
    def __init__(self, n: int) -> None:
        self.rowcount = n


class RecordingDB:
    """Minimal AsyncSession stand-in that records the SEQUENCE and TARGETS of
    every DELETE. Enough to prove ordering, org-scoping, and commit-vs-rebuild
    boundaries — no real SQL execution."""

    def __init__(self, rowcounts: dict[type, int] | None = None) -> None:
        self.deletes: list[type] = []
        self.where_orgs: list[uuid.UUID | None] = []
        self.commits: int = 0
        self._rowcounts = rowcounts or {}

    async def execute(self, stmt):
        assert isinstance(stmt, Delete), f"unexpected stmt type: {type(stmt)}"
        # `stmt.table.name` maps back to a model class we care about; ordering
        # is captured by table name, org-scoping by the WHERE clause literals.
        tname = stmt.table.name
        model = {
            MemoryRecord.__tablename__:      MemoryRecord,
            SignalEventRecord.__tablename__: SignalEventRecord,
            SignalOutbox.__tablename__:      SignalOutbox,
            SignalDeadLetter.__tablename__:  SignalDeadLetter,
            DailyBrief.__tablename__:        DailyBrief,
            StateSnapshot.__tablename__:     StateSnapshot,
        }.get(tname)
        assert model is not None, f"delete against unexpected table: {tname}"
        self.deletes.append(model)
        self.where_orgs.append(_extract_org_literal(stmt.whereclause))
        return _RowCount(self._rowcounts.get(model, 1))

    async def commit(self) -> None:
        self.commits += 1


def _extract_org_literal(clause) -> uuid.UUID | None:
    """Best-effort walk over the WHERE tree to find an organization_id == <uuid>
    literal. Every DELETE in prune_old_signals pins on organization_id — this
    is how the test proves it never spills across tenants."""
    if clause is None:
        return None
    for sub in getattr(clause, "clauses", (clause,)):
        left = getattr(sub, "left", None)
        right = getattr(sub, "right", None)
        if left is not None and getattr(left, "key", None) == "organization_id":
            val = getattr(right, "value", None)
            if isinstance(val, uuid.UUID):
                return val
    return None


# ---- pydantic guard: minimum / maximum bounds on older_than_days ----

def test_prune_body_rejects_short_windows_and_missing_fields():
    """`older_than_days < 7` is more likely a typo than intent — MUST 422 at the
    pydantic boundary, never reach the handler."""
    for bad in (-1, 0, 1, 6):
        with pytest.raises(pydantic.ValidationError):
            R.PruneSignalsBody(older_than_days=bad, confirm=True)
    with pytest.raises(pydantic.ValidationError):
        R.PruneSignalsBody(confirm=True)  # older_than_days is required


def test_prune_body_rejects_absurdly_large_windows():
    """The upper bound (3650 = ~10y) exists so a fat-fingered `9999999`
    doesn't silently succeed — cutoff would be BC-era and the query would
    do a wasted full scan."""
    with pytest.raises(pydantic.ValidationError):
        R.PruneSignalsBody(older_than_days=3651, confirm=True)


def test_prune_body_confirm_defaults_to_false():
    """A body arriving without `confirm` MUST be treated as unconfirmed —
    the field default is the safety net if the client omits it entirely."""
    body = R.PruneSignalsBody(older_than_days=30)
    assert body.confirm is False


# ---- runtime guard: confirm=false is rejected BEFORE any DB write ----

async def test_confirm_false_raises_400_and_touches_nothing():
    db = RecordingDB()
    body = R.PruneSignalsBody(older_than_days=30, confirm=False)
    with pytest.raises(HTTPException) as exc:
        await R.prune_old_signals(body, ctx=_Ctx(ORG), _p=None, db=db)  # type: ignore[arg-type]
    assert exc.value.status_code == 400
    assert db.deletes == []      # no side effects
    assert db.commits == 0


# ---- delete order matches the FK topology ----

async def test_delete_order_respects_fk_topology(monkeypatch):
    """memory_records has ondelete=RESTRICT back to signal_events, so memory
    MUST be deleted first. The remaining tables have no cross-dependency but
    the encoded order is the documentation — keep it stable."""
    db = RecordingDB()
    # Neutralize the post-commit rebuild — this test is about the DELETE trace.
    monkeypatch.setattr(R.StateService, "rebuild_org", _noop_rebuild)

    body = R.PruneSignalsBody(older_than_days=30, confirm=True)
    await R.prune_old_signals(body, ctx=_Ctx(ORG), _p=None, db=db)  # type: ignore[arg-type]

    assert db.deletes == [
        MemoryRecord,        # FK RESTRICT from memory → signals: memory MUST go first
        SignalEventRecord,
        SignalOutbox,
        SignalDeadLetter,
        DailyBrief,
        StateSnapshot,       # via SnapshotRepository(db).delete(...)
    ]


# ---- every DELETE is org-scoped: no cross-tenant leakage ----

async def test_every_delete_scopes_to_ctx_org(monkeypatch):
    db = RecordingDB()
    monkeypatch.setattr(R.StateService, "rebuild_org", _noop_rebuild)

    body = R.PruneSignalsBody(older_than_days=30, confirm=True)
    await R.prune_old_signals(body, ctx=_Ctx(ORG), _p=None, db=db)  # type: ignore[arg-type]

    # Every DELETE we recorded had `organization_id == ORG` in its WHERE —
    # never OTHER_ORG, never NULL.
    assert db.where_orgs and all(o == ORG for o in db.where_orgs), (
        f"org-scoping violated: {db.where_orgs}"
    )


# ---- commit happens once, before the state rebuild ----

async def test_commit_precedes_state_rebuild(monkeypatch):
    """State rebuild MUST run OUTSIDE the purge transaction so the fold reads
    the post-purge memory. If commit hasn't fired first, the rebuild would
    still see the deleted rows via MVCC snapshot isolation."""
    db = RecordingDB()
    order: list[str] = []

    original_commit = db.commit
    async def _traced_commit():
        order.append("commit")
        await original_commit()
    db.commit = _traced_commit  # type: ignore[method-assign]

    async def _traced_rebuild(self, org_id):
        order.append("rebuild")
        return _FakeState()
    monkeypatch.setattr(R.StateService, "rebuild_org", _traced_rebuild)

    body = R.PruneSignalsBody(older_than_days=30, confirm=True)
    await R.prune_old_signals(body, ctx=_Ctx(ORG), _p=None, db=db)  # type: ignore[arg-type]

    assert order == ["commit", "rebuild"], f"ordering violated: {order}"
    assert db.commits == 1


# ---- rebuild failure is swallowed; the purge still succeeds ----

async def test_rebuild_failure_does_not_undo_purge(monkeypatch):
    """`rebuild_org` failure is deliberately non-fatal — the next /context or
    /brief request retries via get_state's rebuild-on-miss path. The response
    reports state_rebuilt=False without raising, so the client sees the purge
    counts and knows to retry the rebuild on the next read."""
    db = RecordingDB()

    async def _boom(self, org_id):
        raise RuntimeError("neon timeout during fold")
    monkeypatch.setattr(R.StateService, "rebuild_org", _boom)

    body = R.PruneSignalsBody(older_than_days=30, confirm=True)
    resp = await R.prune_old_signals(body, ctx=_Ctx(ORG), _p=None, db=db)  # type: ignore[arg-type]

    assert resp.state_rebuilt is False
    assert resp.new_state_counts is None
    # Purge still completed and committed — client can safely retry.
    assert db.commits == 1
    assert len(db.deletes) == 6


# ---- response payload carries the receipt caller needs to trust ----

async def test_response_reports_purge_receipt(monkeypatch):
    """The response is the ONLY receipt the caller gets — the operation is
    async-invisible after commit. Signal-events count in particular gates the
    admin.signals.pruned metric, so a drift here would silently break the
    ops dashboard."""
    counts = {
        MemoryRecord: 12, SignalEventRecord: 7, SignalOutbox: 5,
        SignalDeadLetter: 2, DailyBrief: 3, StateSnapshot: 1,
    }
    db = RecordingDB(rowcounts=counts)
    monkeypatch.setattr(R.StateService, "rebuild_org", _fake_rebuild_returning_counts)

    body = R.PruneSignalsBody(older_than_days=30, confirm=True)
    resp = await R.prune_old_signals(body, ctx=_Ctx(ORG), _p=None, db=db)  # type: ignore[arg-type]

    assert resp.org_id == str(ORG)
    assert resp.deleted["memory_records"]      == 12
    assert resp.deleted["signal_events"]       == 7
    assert resp.deleted["signal_outbox"]       == 5
    assert resp.deleted["signal_dead_letters"] == 2
    assert resp.deleted["daily_briefs"]        == 3
    assert resp.deleted["state_snapshots"]     == 1
    assert resp.state_rebuilt is True
    assert resp.new_state_counts == {"commitments": 1, "relationships": 0,
                                     "projects": 0, "customers": 0, "decisions": 0}


# ---- permission gate: static wiring proof (dependency signature) ----

def test_route_is_gated_on_manage_integrations_via_require_permission():
    """The endpoint MUST require MANAGE_INTEGRATIONS. Prove it by inspecting
    the FastAPI Depends signature of the handler — not by hitting the endpoint,
    which would just re-run FastAPI's plumbing.

    Regression this catches: someone downgrades the gate to a weaker permission
    (or removes it entirely) during a refactor. Without this test, that would
    ship silently — nothing else in the codebase asserts the wiring.
    """
    sig = inspect.signature(R.prune_old_signals)
    perm_param = sig.parameters["_p"]
    dep = perm_param.default  # Depends(require_permission(Permission.MANAGE_INTEGRATIONS))
    # `Depends(...).dependency` is the closure `_dep` returned by
    # `require_permission(perm)`. Peek at its closure cells to recover `perm`.
    closed_perms = [
        c.cell_contents for c in dep.dependency.__closure__ or ()
        if isinstance(c.cell_contents, str)
    ]
    # Permission is a StrEnum, so the constant compares equal to its str form.
    assert Permission.MANAGE_INTEGRATIONS in closed_perms


def test_manage_integrations_excludes_view_only_roles():
    """Complement to the wiring proof: the *permission itself* must not be
    granted to employees. If someone re-grants it (e.g. copies the founder
    default block over employee), a view-only user gets a full-org wipe
    primitive. That is the real safety invariant."""
    assert Permission.MANAGE_INTEGRATIONS in default_permissions_for_role("founder")
    assert Permission.MANAGE_INTEGRATIONS in default_permissions_for_role("admin")
    assert Permission.MANAGE_INTEGRATIONS not in default_permissions_for_role("employee")


# ---- router registration: prefix + tag stable ----

def test_router_prefix_and_route_registered():
    """The /admin/signals/prune path is a stable API contract — a rename would
    silently break any admin CLI / cron / infra script that calls it."""
    assert R.router.prefix == "/admin"
    paths = {r.path for r in R.router.routes}
    assert "/admin/signals/prune" in paths


# ---- helpers ----

class _FakeState:
    def counts(self) -> dict[str, int]:
        return {"commitments": 1, "relationships": 0,
                "projects": 0, "customers": 0, "decisions": 0}


async def _noop_rebuild(self, org_id):
    return _FakeState()


async def _fake_rebuild_returning_counts(self, org_id):
    return _FakeState()
