"""AI narration for Intelligence — executive summaries over deterministic insights.

ADDITIVE and NON-AUTHORITATIVE. This seam reads the already-computed, deterministic
IntelligenceInsights (grounded facts) and renders a natural-language executive summary
via the ML gateway. It NEVER writes an insight, never creates a fact, and never feeds
back into the deterministic projection. Flag-gated (USE_INTELLIGENCE_AI, default OFF)
and FAIL-OPEN: a disabled flag or ANY error returns None, and Intelligence's facts stand
untouched — ML explains evidence, it never becomes the evidence.

The ONLY ML dependency is navis.ml.gateway (the one door) — never a provider, router,
prompt registry, telemetry, embeddings, or retrieval. Mirrors reasoning/ai.py exactly;
the ml pipeline-isolation test carves exactly this one seam
(_SANCTIONED_ML_SEAMS = {..., ("intelligence", "ai.py", "navis.ml.gateway")}).
"""
from __future__ import annotations

import os
import uuid
from typing import Any, Iterable

from navis.ml.gateway import Capability, GatewayRequest, gateway
from navis.platform.monitoring import get_logger

logger = get_logger("intelligence.ai")

PROMPT_ID = "intelligence_summary"  # versioned in the ml prompt registry; gateway resolves it
MAX_FACTS = 40  # bound the prompt at any scale — deterministic request size


def enabled() -> bool:
    """USE_INTELLIGENCE_AI gates the ADDITIVE narrative seam. Default OFF: Intelligence
    stays deterministic-only, byte-identical. Even ON, it is fail-open and never
    authoritative — the deterministic insights are always the source of truth."""
    return os.environ.get("USE_INTELLIGENCE_AI", "false").strip().lower() in ("1", "true", "yes")


def render_insights(insights: Iterable[Any]) -> str:
    """Grounded prompt body: the model sees ONLY these cited insight facts (never raw
    Learning/State/Memory). Deterministic + bounded, so it can be checked against the
    insight set that produced it."""
    lines = ["Organizational facts this company has learned about itself (cite each by [n]):", ""]
    for n, i in enumerate(list(insights)[:MAX_FACTS], start=1):
        lines.append(f"[{n}] ({i.insight_type}) {i.title} — {i.summary}")
    return "\n".join(lines)


async def summarize(org_id: uuid.UUID, insights: Iterable[Any]) -> str | None:
    """Deterministic insights -> an executive summary via the gateway. NEVER raises: a
    disabled flag, empty insights, or any gateway error returns None (the facts are
    authoritative and unaffected). model + prompt id/version are accounted in gateway
    telemetry, never written into any fact."""
    facts = list(insights)
    if not enabled() or not facts:
        return None
    try:
        resp = await gateway.generate(
            GatewayRequest(
                capability=Capability.ACCURATE,
                user=render_insights(facts),
                prompt_id=PROMPT_ID,
                temperature=0.0,
                org_id=org_id,
            )
        )
        return resp.text or None
    except Exception:  # noqa: BLE001 — fail-open: deterministic facts are the floor
        logger.exception("intelligence AI summary failed — facts unaffected")
        return None
