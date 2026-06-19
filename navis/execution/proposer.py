"""Decision -> ExecutionRequest bridge.

Decision PROPOSES; Execution executes. This maps decisions (the JSON dicts stored
on a DailyBrief) to candidate actions — deterministically, with NO content
generation and NO LLM. Every proposal stays REQUESTED (human approval required);
nothing here fires a side effect. Lineage (decision_id, reasoning signals,
watermark) rides along so the resulting ExecutionRequest is fully traceable.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

from navis.execution.models import ActionType


@dataclass(frozen=True, slots=True)
class ProposedAction:
    org_id: uuid.UUID
    decision_id: uuid.UUID
    action_type: str
    payload: dict[str, Any]
    reasoning_signal_ids: list[str]
    reasoning_version: int | None
    state_watermark: str | None


def _map(decision: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
    """Deterministic category -> action. Returns (action_type, payload) or (None, {})
    for informational decisions (priority/context) that warrant no action."""
    category = decision.get("category")
    evidence = decision.get("evidence", {}) or {}
    title = decision.get("title", "")
    decision_id = decision.get("decision_id")
    contact = evidence.get("speaker_email") or evidence.get("person_email")

    if category == "risk":
        if contact:
            return ActionType.SEND_EMAIL.value, {
                "to": contact, "subject": title[:120], "body": "", "decision_id": decision_id,
            }
        return ActionType.CREATE_TASK.value, {"title": title[:200], "decision_id": decision_id}
    if category == "opportunity":
        if contact:
            return ActionType.SEND_EMAIL.value, {
                "to": contact, "subject": title[:120], "body": "", "decision_id": decision_id,
            }
        return ActionType.CREATE_TASK.value, {"title": title[:200], "decision_id": decision_id}
    if category == "delegation":
        return ActionType.CREATE_TASK.value, {
            "title": f"Delegate: {title[:180]}", "decision_id": decision_id,
        }
    # priority / context / org drift -> informational, no action proposed.
    return None, {}


def propose_from_decisions(decisions: list[dict], org_id: uuid.UUID) -> list[ProposedAction]:
    out: list[ProposedAction] = []
    for d in decisions:
        action_type, payload = _map(d)
        if action_type is None:
            continue
        out.append(
            ProposedAction(
                org_id=org_id,
                decision_id=uuid.UUID(d["decision_id"]),
                action_type=action_type,
                payload=payload,
                reasoning_signal_ids=d.get("reasoning_signal_ids", []),
                reasoning_version=d.get("reasoning_version"),
                state_watermark=d.get("state_watermark"),
            )
        )
    return out
