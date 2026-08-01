"""Narrative generation — briefs, summaries, drafts, explanations.

The ONLY direction ML is allowed to influence: Decision/Memory facts → narrative.
NEVER narrative → Decision. Every generator:
  * consumes a GroundedContext (LLM never sees raw DB),
  * uses a versioned prompt,
  * records telemetry (tokens/cost/latency per org),
  * returns text + the cited sources (lineage), never structured decisions.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass

from navis.ml.generation.prompts import get_prompt
from navis.ml.providers import GenRequest, get_provider
from navis.ml.rag.grounding import GroundedContext
from navis.ml.routing import Task, provider_for
from navis.ml.telemetry import telemetry

_TASK_PROMPT = {
    Task.BRIEF_NARRATIVE: "brief_narrative",
    Task.SUMMARY: "summary",
    Task.DRAFT: "draft",
    Task.EXPLANATION: "explanation",
    Task.RECOMMENDATION: "recommendation",
}


@dataclass(frozen=True, slots=True)
class Narrative:
    text: str
    sources: list[str]
    prompt_id: str
    prompt_version: int
    model: str


async def generate(org_id: uuid.UUID, task: Task, context: GroundedContext) -> Narrative:
    prompt = get_prompt(_TASK_PROMPT[task])
    provider = provider_for(task) or get_provider()
    req = GenRequest(
        system=prompt.system,
        user=context.render(),  # grounded facts ONLY
        prompt_id=prompt.id,
        prompt_version=prompt.version,
    )
    result = await provider.generate(req)
    telemetry.record(org_id, result, prompt_id=prompt.id, prompt_version=prompt.version)
    return Narrative(
        text=result.text,
        sources=context.sources(),
        prompt_id=prompt.id,
        prompt_version=prompt.version,
        model=result.model,
    )


# Thin, named entry points (briefs.py / summaries.py / drafts.py / explanations.py
# collapse here — one grounded generator, four tasks; no duplication).
async def brief(org_id, context):          return await generate(org_id, Task.BRIEF_NARRATIVE, context)
async def summary(org_id, context):        return await generate(org_id, Task.SUMMARY, context)
async def draft(org_id, context):          return await generate(org_id, Task.DRAFT, context)
async def explanation(org_id, context):    return await generate(org_id, Task.EXPLANATION, context)
async def recommendation(org_id, context): return await generate(org_id, Task.RECOMMENDATION, context)
