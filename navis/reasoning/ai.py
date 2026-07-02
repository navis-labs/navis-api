"""AI-assisted reasoning — an ADDITIVE signal source beside the deterministic engines.

    Reasoning
    ├── Deterministic Engines   navis.reasoning.v2 — authoritative, pure, replayable (frozen)
    ├── AI Reasoning Engine     this module — async, best-effort, gateway-only
    └── Merge                   merge_signals — deterministic-first, same ordering law

Laws:
  * SAME contract. This source emits the SAME `ReasoningSignal` (same `SignalType` enum, same
    provenance fields, same content-addressed uuid5 identity law), so Decision consumes AI and
    deterministic signals identically and cannot tell them apart except by `engine_name`.
  * ONE ML dependency. The only ML import is `navis.ml.gateway` (ReasoningModelGateway) — never
    a provider, vendor SDK, router, prompt registry, telemetry, embeddings, retrieval, grounding,
    or cache. The pipeline-isolation test carves exactly this one seam and nothing more.
  * Organizational reality only. Input is the same normalized `OrganizationalState` the
    deterministic pass consumes (or the Context Bundle's scoped state — identical type). No
    connector names exist here or in the prompt; scaling from 1 founder to 50k employees changes
    the state's contents, never this code.
  * Fail-open. ANY failure — provider outage, timeout, malformed output, validation — returns []
    and logs. The deterministic pass has already produced the authoritative signals; AI can only
    ADD interpretation, never subtract, block, or override.
  * Deterministic identity. `signal_id` = uuid5 over (org, REASONING_VERSION, engine name/version,
    signal_type, entity_id, derived dedup_key). The model NEVER chooses identity: dedup_key is
    derived from (entity_id | signal_type), so re-running over the same state converges on the
    same ids even when the model rephrases titles. Generation itself is non-deterministic; the
    gateway's response cache (Phase 12) is the seam that will make full replay exact.
"""
from __future__ import annotations

import json
from typing import Any

from navis.ml.gateway import Capability, GatewayRequest, gateway
from navis.platform.monitoring import get_logger
from navis.reasoning.v2.base import ReasoningContext, days_between
from navis.reasoning.v2.contract import (
    REASONING_VERSION,
    ReasoningSignal,
    SignalDraft,
    SignalType,
)

# Deliberate one-home reuse of the runner's stamping (draft -> fully provenanced signal): the
# mapping is defined ONCE in v2 and shared, so AI provenance can never drift from deterministic
# provenance. v2 itself is frozen — importing beats duplicating those 20 lines here.
from navis.reasoning.v2.service import _stamp
from navis.state.v2.models import OrganizationalState

logger = get_logger("reasoning.ai")

PROMPT_ID = "reasoning_signals"  # versioned in the ml prompt registry; gateway resolves it

# AI self-reported confidence is capped BELOW the deterministic engines' ceiling (they emit up
# to 0.9/1.0), so at equal severity an AI interpretation never outranks a rule-verified fact in
# Decision's weight = confidence * severity ranking. Deterministic stays authoritative.
AI_CONFIDENCE_CAP = 0.7

# Prompt budget: newest-first per entity type, deterministic tiebreak. Bounds the request for a
# 50k-person org exactly like a 1-founder org — connector growth changes contents, not size.
MAX_ENTITIES_PER_TYPE = 30
MAX_AI_SIGNALS = 10


class AIReasoningEngine:
    """Identity of the AI signal source. Deliberately NOT a v2 `ReasoningEngine` subclass —
    that ABC's `analyze` is sync and pure, and this source is async I/O. It shares the output
    contract, not the ABC. Bump `version` when the prompt version, the state rendering, or the
    parsing/validation semantics change — signals then get new ids (replay-safe), while the
    deterministic engines' ids are untouched because engine identity is part of the uuid5 key.
    """

    name = "ai_reasoning"
    version = "1.0.0"


_ENGINE = AIReasoningEngine()


def _order_key(s: ReasoningSignal) -> tuple[float, str]:
    # The SAME ordering law as run_reasoning_v2: severity desc, signal_id tiebreak.
    return (-s.severity, str(s.signal_id))


def _age(as_of, iso: str | None) -> float | None:
    d = days_between(as_of, iso)
    return round(d, 1) if d is not None else None


def _cap(items: list) -> list:
    # Deterministic top-K: newest last_seen first (ISO strings order chronologically),
    # entity_id tiebreak. Same state -> same rendering, byte for byte.
    return sorted(items, key=lambda e: (e.last_seen or "", e.entity_id), reverse=True)[
        :MAX_ENTITIES_PER_TYPE
    ]


def render_state(state: OrganizationalState) -> str:
    """Deterministic, bounded, connector-free rendering of the normalized state. Sections are
    keyed by the entity_type bucket name, so a model citation (`entity_type`, `entity_id`) can
    be checked verbatim against `state.entities` — the hallucination guard in `_parse_drafts`."""
    as_of = state.as_of
    entities: dict[str, list[dict[str, Any]]] = {
        "person": [
            {"id": p.entity_id, "name": p.name, "email": p.email, "role": p.role}
            for p in _cap(state.people())
        ],
        "company": [
            {"id": c.entity_id, "domain": c.domain, "people": len(c.person_ids)}
            for c in _cap(state.companies())
        ],
        "customer": [
            {"id": c.entity_id, "days_since_signal": _age(as_of, c.last_seen)}
            for c in _cap(state.customers())
        ],
        "project": [
            {
                "id": p.entity_id,
                "documents": len(p.document_ids),
                "days_since_activity": _age(as_of, p.last_seen),
            }
            for p in _cap(state.projects())
        ],
        "meeting": [
            {"id": m.entity_id, "title": m.title, "status": m.status, "attendees": m.participants}
            for m in _cap(state.meetings())
        ],
        "relationship": [
            {
                "id": r.entity_id,
                "person_email": r.person_email,
                "directions": r.directions,
                "days_since_interaction": _age(as_of, r.last_interaction_at),
            }
            for r in _cap(state.relationships())
        ],
        "commitment": [
            {
                "id": c.entity_id,
                "text": c.text,
                "status": c.status,
                "speaker_email": c.speaker_email,
                "age_days": _age(as_of, c.last_seen),
            }
            for c in _cap(state.commitments())
        ],
        "decision": [
            {"id": d.entity_id, "text": d.text, "status": d.status, "age_days": _age(as_of, d.last_seen)}
            for d in _cap(state.decisions())
        ],
        "document": [
            {"id": d.entity_id, "title": d.title, "projects": d.project_ids}
            for d in _cap(state.documents())
        ],
        "thread": [
            {
                "id": t.entity_id,
                "subject": t.subject,
                "messages": len(t.message_ids),
                "participants": t.participants,
            }
            for t in _cap(state.threads())
        ],
    }
    doc = {
        "as_of": as_of.isoformat() if as_of else None,
        "entity_counts": state.counts(),
        "entities": entities,
    }
    return json.dumps(doc, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _json_array(text: str) -> list:
    """Extract the JSON array from a model response, tolerating prose/code fences around it.
    Anything unparseable is [] — fail-open, never an exception."""
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end <= start:
        return []
    try:
        data = json.loads(text[start : end + 1])
    except ValueError:
        return []
    return data if isinstance(data, list) else []


def _parse_drafts(
    text: str, state: OrganizationalState, provenance: dict[str, Any]
) -> list[SignalDraft]:
    """Model output -> validated SignalDrafts. Fail-closed PER ITEM (a bad item is skipped, the
    rest survive): unknown signal_type, an entity citation not present verbatim in the state
    (hallucination guard), or a missing title/why rejects the item. confidence is capped at
    AI_CONFIDENCE_CAP and severity clamped to [0,1] BEFORE SignalDraft's own range validation.
    dedup_key is DERIVED (entity_id | signal_type) — identity is never model-chosen."""
    valid_types = {t.value for t in SignalType}
    drafts: list[SignalDraft] = []
    for item in _json_array(text):
        if len(drafts) >= MAX_AI_SIGNALS:
            break
        if not isinstance(item, dict):
            continue
        stype = item.get("signal_type")
        if stype not in valid_types:
            logger.warning("ai draft dropped: unknown signal_type %r", stype)
            continue
        entity_id = item.get("entity_id") or None
        entity_type = item.get("entity_type") or None
        if entity_id is not None:
            if entity_id not in state.entities.get(entity_type or "", {}):
                logger.warning("ai draft dropped: uncited entity %s/%s", entity_type, entity_id)
                continue
        else:
            entity_type = None  # org-wide signal carries no entity
        title = str(item.get("title") or "").strip()
        why = str(item.get("why") or "").strip()
        if not title or not why:
            continue
        try:
            confidence = float(item.get("confidence", 0.0))
            severity = float(item.get("severity", 0.0))
        except (TypeError, ValueError):
            continue
        evidence = item.get("evidence")
        drafts.append(
            SignalDraft(
                signal_type=SignalType(stype),
                entity_type=entity_type,
                entity_id=entity_id,
                title=title[:200],
                why=why[:500],
                confidence=min(max(confidence, 0.0), AI_CONFIDENCE_CAP),
                severity=min(max(severity, 0.0), 1.0),
                dedup_key=entity_id or f"ai:{stype}",
                evidence={**(evidence if isinstance(evidence, dict) else {}), **provenance},
            )
        )
    return drafts


async def run_ai_reasoning(state: OrganizationalState) -> list[ReasoningSignal]:
    """The AI reasoning pass: OrganizationalState -> ReasoningSignal[] via ONE gateway call.

    Async because the gateway is network I/O — the deterministic runner stays sync and is never
    awaited on this. NEVER raises: any failure returns [] so the caller's merge degrades to
    exactly the deterministic output. Provenance (org, as_of, watermark, reasoning_version,
    logical created_at) is stamped by the SAME v2 mechanism deterministic signals use; model,
    prompt_id and prompt_version land in each signal's evidence for eval/replay attribution."""
    try:
        if sum(state.counts().values()) == 0:
            return []  # nothing to interpret; don't spend a model call
        wm = state.watermark.last_memory_id
        ctx = ReasoningContext(
            org_id=state.org_id,
            as_of=state.as_of,
            watermark=str(wm) if wm else None,
            reasoning_version=REASONING_VERSION,
        )
        resp = await gateway.generate(
            GatewayRequest(
                capability=Capability.REASONING,
                user=render_state(state),
                prompt_id=PROMPT_ID,
                max_tokens=2048,
                temperature=0.0,
                org_id=state.org_id,
            )
        )
        provenance = {
            "model": resp.model,
            "prompt_id": resp.prompt_id,
            "prompt_version": resp.prompt_version,
        }
        unique: dict = {}
        for draft in _parse_drafts(resp.text, state, provenance):
            s = _stamp(draft, _ENGINE, ctx)
            unique.setdefault(s.signal_id, s)  # derived dedup_key -> same-id restatements collapse
        signals = sorted(unique.values(), key=_order_key)
        logger.info(
            "ai reasoning produced %d signals org=%s model=%s", len(signals), state.org_id, resp.model
        )
        return signals
    except Exception:  # noqa: BLE001 — fail-open: deterministic reasoning is the floor
        logger.exception("ai reasoning pass failed — continuing with deterministic signals only")
        return []


def merge_signals(
    deterministic: list[ReasoningSignal], ai: list[ReasoningSignal]
) -> list[ReasoningSignal]:
    """Merge the two sources. Deterministic is AUTHORITATIVE: an AI signal that re-states a
    deterministic finding — same (signal_type, entity_id) — is suppressed, so AI only ADDS
    interpretation and can never duplicate, override, or reorder a deterministic conclusion's
    presence. Output preserves the runner's invariants: unique signal_ids, ordered
    (severity desc, signal_id tiebreak). With ai == [] this returns the deterministic list
    unchanged in content — the fail-open path is byte-equivalent to today's pipeline."""
    covered = {(s.signal_type, s.entity_id) for s in deterministic}
    seen = {s.signal_id for s in deterministic}
    merged = list(deterministic)
    for s in ai:
        if (s.signal_type, s.entity_id) in covered or s.signal_id in seen:
            continue
        merged.append(s)
        seen.add(s.signal_id)
    return sorted(merged, key=_order_key)
