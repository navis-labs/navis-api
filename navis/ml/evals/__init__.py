"""Evaluation-driven development: every prompt change / model swap is testable.

Three deterministic evaluators:
  * retrieval_eval  — recall@k of a retriever against a labeled golden set
  * generation_eval — grounding fidelity: does output cite only provided facts?
  * regression_eval — snapshot a prompt's rendered output; flag drift
"""
from navis.ml.evals.retrieval_eval import RetrievalCase, mean_recall_at_k, recall_at_k
from navis.ml.evals.generation_eval import grounding_score, is_grounded
from navis.ml.evals.regression_eval import prompt_fingerprint

__all__ = [
    "RetrievalCase", "recall_at_k", "mean_recall_at_k",
    "grounding_score", "is_grounded", "prompt_fingerprint",
]
