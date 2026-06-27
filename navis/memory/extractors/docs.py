"""Deterministic extractor over Docs SignalEvents (Google Docs, derived from Drive).

A Doc is a first-class Document entity — title + ownership + folder lineage. We record the
observed document fact only; status/health/importance are State/Reasoning, not Memory.
"""
from __future__ import annotations

from navis.memory.envelope import EntityType
from navis.memory.extractors.base import Extractor, MemoryDraft, SignalView
from navis.memory.schemas import current_version
from navis.memory.confidence import STRUCTURAL
from navis.signals.base import SignalSource

_DOCS = {SignalSource.DOCS}


class DocumentExtractor(Extractor):
    name = "docs_document"
    version = "1.0.0"
    sources = _DOCS

    def extract(self, signal: SignalView) -> list[MemoryDraft]:
        p = signal.payload
        owners = [o.get("email") for o in (p.get("owners") or []) if o.get("email")]
        return [MemoryDraft(
            entity_type=EntityType.DOCUMENT,
            entity_natural_key=signal.external_id,        # the Drive/Docs file id
            dedup_key=signal.external_version,            # one record per version (edit)
            payload={
                "document_id": signal.external_id,
                "title": p.get("name"),
                "owners": owners,
                "parents": p.get("parents") or [],
                "drive_id": p.get("drive_id"),
                "source": "docs",
            },
            confidence=STRUCTURAL,
            schema_version=current_version(EntityType.DOCUMENT),
        )]
