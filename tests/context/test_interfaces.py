"""Protocol compliance + contract verification for navis/context/interfaces.py.

Verifies that the concrete defaults satisfy their protocols (runtime-checkable), that the
pure protocols are deterministic and never mutate their inputs, and that the Compressor's
fixed-order tuple contract holds.
"""
from __future__ import annotations

from navis.context.cache import NullContextCache
from navis.context.compression import GreedyCompressor
from navis.context.contracts import ActorRef, PermissionScope, TokenBudget
from navis.context.interfaces import (
    Compressor,
    ContextCache,
    PermissionResolver,
    Ranker,
    RetrievalProvider,
    Selector,
    StateReader,
)
from navis.context.models import BudgetReport, ContextScope
from navis.context.permissions import DefaultDenyPermissionFilter
from navis.context.ranking import StructuralRanker
from navis.context.selectors import DeterministicSelector
from navis.context.tasks import TASK_SPECS, ContextTask
from navis.state.v2.builders import build_projection
from tests.context.fakes import FakeStateReader, MemoryContextCache
from tests.state.fakes import ORG_A, corpus

_SPEC = TASK_SPECS[ContextTask.FULL_ORG]


def _state():
    return build_projection(ORG_A, corpus())


# --- Runtime protocol compliance (structural isinstance) --------------------

def test_default_implementations_satisfy_their_protocols():
    assert isinstance(DeterministicSelector(), Selector)
    assert isinstance(DefaultDenyPermissionFilter(), PermissionResolver)
    assert isinstance(StructuralRanker(), Ranker)
    assert isinstance(GreedyCompressor(), Compressor)
    assert isinstance(NullContextCache(), ContextCache)
    assert isinstance(MemoryContextCache(), ContextCache)
    assert isinstance(FakeStateReader(), StateReader)


def test_non_implementation_fails_protocol_check():
    class NotASelector:
        pass

    assert not isinstance(NotASelector(), Selector)


def test_retrieval_provider_protocol_is_satisfiable():
    class AllowAll:
        def candidates(self, state, scope):
            return None

    assert isinstance(AllowAll(), RetrievalProvider)


# --- Determinism contract ---------------------------------------------------

def test_selector_is_deterministic():
    state = _state()
    a = DeterministicSelector().select(state, _SPEC)
    b = DeterministicSelector().select(state, _SPEC)
    assert a.to_dict() == b.to_dict()


def test_ranker_is_deterministic_total_order():
    state = _state()
    scope = DeterministicSelector().select(state, _SPEC)
    a = StructuralRanker().rank(state, scope)
    b = StructuralRanker().rank(state, scope)
    assert [e.entity_id for e in a] == [e.entity_id for e in b]
    assert len({e.entity_id for e in a}) == len(a)  # total order, unique


# --- Mutation rule: pure protocols never mutate their inputs -----------------

def test_selector_does_not_mutate_state():
    state = _state()
    before = state.to_dict()
    DeterministicSelector().select(state, _SPEC)
    assert state.to_dict() == before


def test_ranker_does_not_mutate_inputs():
    state = _state()
    scope = DeterministicSelector().select(state, _SPEC)
    state_before, scope_before = state.to_dict(), scope.to_dict()
    StructuralRanker().rank(state, scope)
    assert state.to_dict() == state_before and scope.to_dict() == scope_before


def test_compressor_does_not_mutate_inputs():
    state = _state()
    scope = DeterministicSelector().select(state, _SPEC)
    ranked = StructuralRanker().rank(state, scope)
    before = state.to_dict()
    GreedyCompressor().compress(ranked, state, TokenBudget.unbounded())
    assert state.to_dict() == before


def test_permission_filter_does_not_mutate_inputs():
    state = _state()
    scope = DeterministicSelector().select(state, _SPEC)
    before = scope.to_dict()
    actor = ActorRef.system()
    DefaultDenyPermissionFilter().resolve(actor, PermissionScope.system(), scope, state)
    assert scope.to_dict() == before


# --- Compressor fixed-order tuple contract -----------------------------------

def test_compressor_returns_fixed_order_tuple():
    state = _state()
    ranked = StructuralRanker().rank(state, DeterministicSelector().select(state, _SPEC))
    result = GreedyCompressor().compress(ranked, state, TokenBudget.unbounded())
    assert isinstance(result, tuple) and len(result) == 3
    admitted, dropped, report = result
    assert isinstance(report, BudgetReport)
    assert report.admitted == len(admitted) and report.dropped == len(dropped)
