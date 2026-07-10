"""Phase 1 parity: the FULL_ORG Context path must produce IDENTICAL reasoning to the
direct State path — so USE_CONTEXT_LAYER on/off is behaviourally indistinguishable.
"""
from __future__ import annotations

from navis.context import ContextRequest, ContextService
from navis.memory.envelope import EntityType
from navis.reasoning.v2 import run_reasoning_v2
from navis.state.v2.builders import build_projection
from tests.context.fakes import FakeStateReader
from tests.state.fakes import ORG_A, corpus, rec


def _stale_state():
    # Heartbeat pushes as_of ~40 days ahead so risk engines actually emit signals.
    recs = corpus(ORG_A) + [
        rec(EntityType.PERSON, "hb@acme.com", {"email": "hb@acme.com", "role": "from"},
            offset=40 * 86400, tag="hb", org_id=ORG_A)
    ]
    return build_projection(ORG_A, recs)


async def test_full_org_context_yields_identical_reasoning_signals():
    state = _stale_state()
    bundle = await ContextService(state_service=FakeStateReader(state)).build(
        ContextRequest.full_org(ORG_A)
    )

    direct = run_reasoning_v2(state)
    via_context = run_reasoning_v2(bundle.state)

    assert direct, "expected reasoning to emit signals on the stale corpus"
    assert [str(s.signal_id) for s in via_context] == [str(s.signal_id) for s in direct]
    assert [s.severity for s in via_context] == [s.severity for s in direct]
    assert bundle.state.as_of == state.as_of and bundle.state.watermark.to_dict() == state.watermark.to_dict()
