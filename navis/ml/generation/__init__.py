"""Narrative generation (Decision -> narrative; never narrative -> Decision)."""
from navis.ml.generation.narratives import Narrative, generate, brief, summary, draft, explanation
from navis.ml.generation.prompts import Prompt, get_prompt, REGISTRY

__all__ = ["Narrative", "generate", "brief", "summary", "draft", "explanation", "Prompt", "get_prompt", "REGISTRY"]
