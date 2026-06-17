"""ML — all LLM / embedding / extraction access, isolated.

ARCHITECTURE RULE: domains never call an LLM SDK directly. Memory asks ml; ml
returns a result. This keeps prompts, providers, retries, and cost control in one
place and the domains pure.

    from navis.ml import llm
    text = await llm.complete("system...", "user...")
"""
from navis.ml.client import LLMClient, llm

__all__ = ["LLMClient", "llm"]
