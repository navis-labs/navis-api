"""Deterministic extractor over Docs SignalEvents (Google Docs, derived from Drive).

A Doc is a first-class Document entity — title + ownership + folder lineage. We record the
observed document fact only; health/importance are State/Reasoning, not Memory.

`status` is the one lifecycle field, mirroring how MEETING carries a Google-supplied
status for cancelled events. It is derived from the signal's event type rather than
mutating or deleting anything: Memory is append-only, so trashing a Doc APPENDS a
`status="trashed"` fact and State's latest-non-null-wins fold makes it the surviving
value. Restoring appends `status="active"` and the same fold flips it back — one
mechanism, both directions, no second deletion architecture and no hard delete.

The value is ALWAYS written explicitly, never None: the fold skips nulls (a null means
"unknown here", not "cleared"), so an omitted status on a later edit would leave a
stale `trashed` in place forever.
"""
from __future__ import annotations

from navis.memory.envelope import EntityType
from navis.memory.extractors.base import Extractor, MemoryDraft, SignalView
from navis.memory.schemas import current_version
from navis.memory.confidence import STRUCTURAL
from navis.signals.base import SignalEventType, SignalSource

_DOCS = {SignalSource.DOCS}

DOC_ACTIVE = "active"
DOC_TRASHED = "trashed"


class DocumentExtractor(Extractor):
    name = "docs_document"
    # 1.1.0: added the additive `status` lifecycle field (see module docstring).
    version = "1.1.0"
    sources = _DOCS

    def extract(self, signal: SignalView) -> list[MemoryDraft]:
        p = signal.payload
        owners = [o.get("email") for o in (p.get("owners") or []) if o.get("email")]
        status = (
            DOC_TRASHED
            if signal.event_type is SignalEventType.DOC_TRASHED
            else DOC_ACTIVE
        )
        return [MemoryDraft(
            entity_type=EntityType.DOCUMENT,
            entity_natural_key=signal.external_id,        # the Drive/Docs file id
            # One record per version (edit). Trashing bumps Drive's `version`, so a
            # trash fact never collides with the edit that preceded it — and a
            # REPEATED trash event carries the SAME version, hence the same
            # memory_id, hence ON CONFLICT DO NOTHING: idempotent by construction.
            dedup_key=signal.external_version,
            payload={
                "document_id": signal.external_id,
                "title": p.get("name"),
                "owners": owners,
                "parents": p.get("parents") or [],
                "drive_id": p.get("drive_id"),
                "source": "docs",
                "status": status,
            },
            confidence=STRUCTURAL,
            schema_version=current_version(EntityType.DOCUMENT),
        )]
