"""Cross-domain event wiring.

Domains stay decoupled by reacting to events. This module owns the few
pipeline-level connections that don't belong to any single domain:

    CommitmentDetected -> rebuild OrganizationalState (which emits StateUpdated)

Reasoning/decisions are computed on demand by the brief orchestrator, so we stop
the live chain at State here. Tighten or extend these edges as the product grows.
"""
from __future__ import annotations

from navis.platform.events import DomainEvent, EventType, subscribe
from navis.platform.monitoring import get_logger
from navis.state.service import rebuild_for_org

logger = get_logger("orchestration.wiring")


@subscribe(EventType.COMMITMENT_DETECTED)
async def rebuild_state_on_commitment(event: DomainEvent) -> None:
    if event.org_id is None:
        return
    await rebuild_for_org(event.org_id)
