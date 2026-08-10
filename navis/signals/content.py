"""Reading content that the connectors sealed at capture time.

Gmail/Slack connectors encrypt the body-bearing fields before the signal is
persisted and record which ones in `_content_fields` (see
`signals/gmail/connector.py`). Anything that needs to READ that content has to
undo it first.

This is the canonical counterpart to that sealing, living beside it in the
Signals layer. A field that cannot be decrypted (key rotated out, tampered)
degrades to empty with a metric rather than raising — one unreadable body must
never take down ingestion or a scoring pass.
"""
from __future__ import annotations

from typing import Any

from navis.platform.monitoring import get_logger
from navis.platform.monitoring.metrics import metrics
from navis.platform.security import FieldDecryptError, field_cipher

logger = get_logger("signals.content")


def decrypt_content_fields(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a NEW payload with sealed fields decrypted. Never mutates the input."""
    fields = payload.get("_content_fields")
    if not fields:
        return payload
    out = dict(payload)
    for field in fields:
        value = out.get(field)
        if isinstance(value, str):
            try:
                out[field] = field_cipher.decrypt(value)
            except FieldDecryptError:
                metrics.incr("signals.content_decrypt_failed")
                logger.error("content field %s failed decryption; degrading to empty", field)
                out[field] = ""
    return out
