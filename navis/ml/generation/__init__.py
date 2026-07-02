"""Narrative generation (Decision -> narrative; never narrative -> Decision)."""
from navis.ml.generation.narratives import Narrative, brief, summary, draft, explanation
from navis.ml.generation.prompts import Prompt, get_prompt
from navis.ml.generation.prompts import REGISTRY as REGISTRY  # explicit re-export for prompt-enumeration tests; NOT public API

# `generate(task)` removed: it discriminated on a product-Task enum that duplicated the router's;
# the four named generators are the product surface, each naming a CAPABILITY via the gateway.
# REGISTRY is intentionally NOT in __all__ — production code resolves prompts via get_prompt only.
__all__ = ["Narrative", "brief", "summary", "draft", "explanation", "Prompt", "get_prompt"]
