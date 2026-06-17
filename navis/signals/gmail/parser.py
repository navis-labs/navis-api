"""Parse a raw Gmail message payload into a flat dict of useful fields."""
from __future__ import annotations

import base64
from typing import Any


def _header(headers: list[dict], name: str) -> str | None:
    name_l = name.lower()
    for h in headers:
        if h.get("name", "").lower() == name_l:
            return h.get("value")
    return None


def _decode_body(part: dict) -> str:
    data = part.get("body", {}).get("data")
    if not data:
        return ""
    try:
        return base64.urlsafe_b64decode(data.encode()).decode("utf-8", errors="replace")
    except Exception:
        return ""


def _extract_text(payload: dict) -> str:
    """Walk the MIME tree for the first text/plain body."""
    if payload.get("mimeType") == "text/plain":
        return _decode_body(payload)
    for part in payload.get("parts", []) or []:
        text = _extract_text(part)
        if text:
            return text
    return ""


def parse_message(raw: dict[str, Any]) -> dict[str, Any]:
    payload = raw.get("payload", {})
    headers = payload.get("headers", [])
    return {
        "message_id": raw.get("id"),
        "thread_id": raw.get("threadId"),
        "history_id": raw.get("historyId"),
        "internal_date": raw.get("internalDate"),
        "subject": _header(headers, "Subject"),
        "from": _header(headers, "From"),
        "to": _header(headers, "To"),
        "cc": _header(headers, "Cc"),
        "date": _header(headers, "Date"),
        "snippet": raw.get("snippet"),
        "body_text": _extract_text(payload)[:20000],
        "label_ids": raw.get("labelIds", []),
    }
