"""Adaptive runtime injection (no dependency inversion) + the scheduled-work gate.

Reasoning/Decision/Execution must NOT import Adaptive. Instead, the orchestration
layer (which legally sits above everything) resolves an org's ACTIVE adaptive policy
and injects the relevant values into the domain call. Domains stay pure; policy
flows: Adaptive → active snapshot → this resolver → domain params.

Today Adaptive proposes exactly two governing variables that the brief consumes —
brief_min_confidence and brief_max_items — so injecting them here makes the loop
real end-to-end. As Adaptive's vocabulary grows, extend the mapping below; no domain
needs to change.

`scheduled_work_allowed` is the same idea applied to entitlements: orchestration
asks the one authority whether a tenant may run this work, and the scheduled
producers ask before spending anything.
"""
from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from navis.adaptive.policy import merge_defaults
from navis.adaptive.repository import AdaptiveRepository
from navis.organization.contract import Capability
from navis.organization.service import OrganizationService
from navis.platform.monitoring import get_logger
from navis.platform.monitoring.metrics import metrics

logger = get_logger("orchestration.runtime_policy")


async def resolve_policy(db: AsyncSession, org_id: uuid.UUID) -> dict[str, Any]:
    """The org's active policy merged over code defaults, STAMPED with its version.

    `policy_version` is what makes runtime behavior attributable: given a brief,
    it is the only way to answer "which governing snapshot produced this?".
    `ScopedView` has always had a field for it and `scope.py` has always looked
    for it in this dict — but `merge_defaults` drops every key outside
    `POLICY_FIELDS`, and `policy_version` is not a policy field, so the lookup
    could never succeed. `ScopedView.policy_version` was unconditionally None and
    the context audit line logged `policy_version=None` on every request.

    Stamped HERE rather than inside `merge_defaults` for two reasons: that
    function defines the governed VOCABULARY, and a non-field key in it would
    also leak into `AdaptiveView.active_policy` — from which `proposals.py`
    copies `proposed_payload`, which would then carry a stale version number into
    every stored proposal.

    `None` when no snapshot is active, which is exactly the "running on code
    defaults" case and must stay distinguishable from version 0.
    """
    repo = AdaptiveRepository(db)
    snapshot = await repo.get_active_snapshot(org_id)
    policy = merge_defaults(snapshot.payload if snapshot else None)
    policy["policy_version"] = snapshot.policy_version if snapshot else None
    return policy


# ── The scheduled-work entitlement gate (T-14) ──────────────────────────────
#
# WHAT WAS WRONG. Every scheduled producer fanned out over ALL organizations —
# `build_all_briefs` over `SELECT id FROM organizations`, `sync_all_sources` over
# every active integration, `embed_all_orgs` over every org — with no join to
# lifecycle and no capability check anywhere. So the API path and the background
# path enforced different rules for the same work: `POST /brief/generate` gates on
# `Capability.DECISION`, while the hourly `build_brief_for_org` that produces the
# same artefact gated on nothing. A SUSPENDED organization kept syncing mail every
# five minutes, kept building briefs every hour, kept proposing actions and kept
# being embedded — which also means it kept costing provider and LLM spend.
#
# Only Execution closed the loop, because its worker re-authorizes per row
# (`worker._authorization_denial`). This is that same idea for the producers.
#
# THIS IS AN EXECUTION GATE, NOT DELETION. Nothing is removed: no organization, no
# integration, no signal, no memory, no state. Work is SKIPPED while the tenant is
# not entitled and resumes by itself the moment it is — which is the same property
# that makes the execution worker's denial retryable rather than terminal.
#
# NO NEW AUTHORIZATION LOGIC LIVES HERE. It asks `OrganizationService.can_use`,
# the same single authority `require_capability` and the execution worker use, so
# lifecycle (TRIAL/ACTIVE only), the kill switch, explicit entitlement rows and
# plan defaults are all inherited rather than re-implemented.


async def scheduled_work_allowed(
    db: AsyncSession,
    org_id: uuid.UUID | None,
    capability: Capability,
    *,
    task: str,
) -> bool:
    """May this organization run this scheduled work right now?

    FAILS CLOSED on a missing organization and on any error resolving the answer —
    a check that cannot complete is not a pass. That matters more here than on the
    request path: nobody is watching a beat task, so a gate that silently degraded
    to "allow" would look exactly like a working one.

    Skips are OBSERVABLE and the labels are bounded — `task` is a fixed string per
    call site and `capability` is a closed enum, so this cannot blow up metric
    cardinality however many organizations exist.
    """
    if org_id is None:
        metrics.incr(
            "orchestration.scheduled.skipped",
            labels={"task": task, "reason": "no_org"},
        )
        return False
    try:
        allowed = await OrganizationService(db).can_use(org_id, capability)
    except Exception:  # noqa: BLE001 — see docstring: cannot check => deny
        logger.exception(
            "scheduled entitlement check failed task=%s org=%s cap=%s; skipping",
            task, org_id, capability.value,
        )
        metrics.incr(
            "orchestration.scheduled.skipped",
            labels={"task": task, "reason": "check_failed"},
        )
        return False
    if not allowed:
        # INFO, not WARNING: a free-tier org not getting briefs is the system
        # working. An operator wants the counter, not a log line per org per hour.
        logger.info(
            "scheduled work skipped task=%s org=%s cap=%s (not entitled)",
            task, org_id, capability.value,
        )
        metrics.incr(
            "orchestration.scheduled.skipped",
            labels={"task": task, "reason": "not_entitled"},
        )
    return allowed
