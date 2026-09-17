"""Admission policy — which Memory facts earn a place in the search index.

Memory is an immutable, append-only log: facts extracted by an older, buggier
extractor stay there forever, and that is correct — the log is the audit trail
and the replay source. But the embedding index is a DERIVED artifact, explicitly
rebuildable from Memory + the embedder, so it does not have to inherit the log's
mistakes.

Concretely: an earlier commitment regex made the apostrophe optional
(`\\bwe'?ll\\b`), so it matched the ordinary word "well". Sentences like
"We hope this email finds you well." were stored as organizational commitments —
75% of the stored commitment/decision rows in the reference org fail today's
rules. Embedding them puts confident-looking garbage in front of the model.

Admission re-applies the CURRENT extractor predicates to the stored text. It
never deletes or mutates a record: a rejected fact simply gets no vector at this
(model, version). Changing the policy later is another re-embed, not a data
migration.

ML reading `navis.memory` is the sanctioned direction (ML may read the truth
layers); the reverse — memory importing ml — is what the architecture forbids.
"""
from __future__ import annotations

from typing import Any, Protocol

from navis.memory.extractors.email import (
    is_commitment_sentence,
    is_decision_sentence,
)


class _Fact(Protocol):
    entity_type: str
    payload: dict[str, Any] | None


# Entity types whose value IS a claim about the organization, and which the
# heuristic extractors can therefore get wrong. Structural facts (person,
# company, thread, relationship, meeting, document, project) are observations —
# an address either appeared on an email or it did not — so they are always
# admitted.
_CLAIM_VALIDATORS = {
    "commitment": is_commitment_sentence,
    "decision": is_decision_sentence,
}


def admits(record: _Fact) -> bool:
    """True if this fact should be embedded at the current policy."""
    validator = _CLAIM_VALIDATORS.get(record.entity_type)
    if validator is None:
        return True
    text = (record.payload or {}).get("text")
    if not isinstance(text, str) or not text:
        return False  # a claim with no claim in it indexes nothing useful
    return validator(text)
