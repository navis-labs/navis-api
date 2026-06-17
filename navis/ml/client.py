"""Provider-agnostic LLM client.

Picks the first configured provider (Anthropic preferred, then OpenAI). Degrades
gracefully: if no key is set, `is_available` is False and callers fall back to
heuristics — the whole pipeline must run without an LLM.

Default model is the latest Claude (claude-opus-4-8). Keep provider SDK imports
lazy so the app boots without the SDKs installed.
"""
from __future__ import annotations

import json
from typing import Any

from navis.platform.config import settings
from navis.platform.monitoring import get_logger

logger = get_logger("ml")

DEFAULT_ANTHROPIC_MODEL = "claude-opus-4-8"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"


class LLMClient:
    def __init__(self) -> None:
        if settings.anthropic_api_key:
            self.provider = "anthropic"
        elif settings.openai_api_key:
            self.provider = "openai"
        else:
            self.provider = None

    @property
    def is_available(self) -> bool:
        return self.provider is not None

    async def complete(
        self, system: str, user: str, *, max_tokens: int = 1024, temperature: float = 0.2
    ) -> str:
        if self.provider == "anthropic":
            return await self._anthropic(system, user, max_tokens, temperature)
        if self.provider == "openai":
            return await self._openai(system, user, max_tokens, temperature)
        raise RuntimeError("no LLM provider configured")

    async def extract_json(self, system: str, user: str, *, max_tokens: int = 1024) -> Any:
        """Complete and parse JSON, tolerating fenced code blocks."""
        raw = await self.complete(system, user, max_tokens=max_tokens, temperature=0.0)
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```", 2)[1].lstrip("json").strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("ml: non-JSON response, returning empty")
            return None

    async def _anthropic(self, system: str, user: str, max_tokens: int, temperature: float) -> str:
        from anthropic import AsyncAnthropic

        client = AsyncAnthropic(api_key=settings.anthropic_api_key)
        resp = await client.messages.create(
            model=DEFAULT_ANTHROPIC_MODEL,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(block.text for block in resp.content if block.type == "text")

    async def _openai(self, system: str, user: str, max_tokens: int, temperature: float) -> str:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=settings.openai_api_key)
        resp = await client.chat.completions.create(
            model=DEFAULT_OPENAI_MODEL,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content or ""


llm = LLMClient()
