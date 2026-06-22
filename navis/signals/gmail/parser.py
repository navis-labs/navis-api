"""Parse a raw Gmail message payload into a flat dict of useful fields.

Reality capture only: structural parsing (header values, address splitting, MIME walk,
base64 decode) — NO interpretation. De-noising (e.g. HTML->text) and graph derivation
(conversation roots) belong to the Memory/State layers, not here. The raw HTML is kept as
captured when no text/plain alternative exists; `body_mime` tells downstream what it got.
"""
from __future__ import annotations

import base64
from email.utils import getaddresses, parseaddr
from typing import Any

# Body capture cap. Truncation is recorded (body_truncated/body_length) so downstream
# summarization knows it is working from a prefix, not the whole message.
_BODY_MAX = 20000


def _header(headers: list[dict], name: str) -> str | None:
    name_l = name.lower()
    for h in headers:
        if h.get("name", "").lower() == name_l:
            return h.get("value")
    return None


def _one_address(raw: str | None) -> tuple[str | None, str | None]:
    """(display_name, email_lower) for a single-address header (From). RFC 2822 aware."""
    if not raw:
        return None, None
    name, email = parseaddr(raw)
    email = email.strip().lower()
    return (name or None), (email or None)


def _address_list(raw: str | None) -> list[str]:
    """Lowercased emails from an address-list header. Uses getaddresses so a display name
    containing a comma ("Doe, John <j@x>") doesn't split into bogus addresses."""
    if not raw:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for _name, email in getaddresses([raw]):
        email = email.strip().lower()
        if email and email not in seen:
            seen.add(email)
            out.append(email)
    return out


def _decode_body(part: dict) -> str:
    data = part.get("body", {}).get("data")
    if not data:
        return ""
    try:
        return base64.urlsafe_b64decode(data.encode()).decode("utf-8", errors="replace")
    except Exception:
        return ""


def _extract_body(payload: dict) -> tuple[str, str | None]:
    """Walk the MIME tree for a body. Prefer text/plain; fall back to text/html so
    HTML-only emails (a large fraction of real mail) are not silently captured empty.
    Returns (text, mime) where mime records which representation was used."""
    if payload.get("mimeType") == "text/plain":
        body = _decode_body(payload)
        if body:
            return body, "text/plain"
    html = ""
    for part in payload.get("parts", []) or []:
        text, mime = _extract_body(part)
        if mime == "text/plain" and text:
            return text, "text/plain"  # plain wins as soon as we find it
        if mime == "text/html" and text and not html:
            html = text
    if payload.get("mimeType") == "text/html":
        html = html or _decode_body(payload)
    if html:
        return html, "text/html"
    return "", None


def _collect_attachments(payload: dict) -> list[dict[str, Any]]:
    """Walk the MIME tree for attachment parts (those carrying a filename). Metadata only —
    filename/mime/size/attachmentId — never the bytes. Lets the future Document ontology
    tell 'Quarterly Report attached' apart from a plain note without fetching content."""
    out: list[dict[str, Any]] = []

    def walk(part: dict) -> None:
        filename = part.get("filename")
        body = part.get("body", {}) or {}
        if filename:
            out.append(
                {
                    "filename": filename,
                    "mime_type": part.get("mimeType"),
                    "size": body.get("size"),
                    "attachment_id": body.get("attachmentId"),
                }
            )
        for child in part.get("parts", []) or []:
            walk(child)

    for child in payload.get("parts", []) or []:
        walk(child)
    return out


def parse_message(raw: dict[str, Any]) -> dict[str, Any]:
    payload = raw.get("payload", {})
    headers = payload.get("headers", [])
    body_full, body_mime = _extract_body(payload)
    sender_name, sender_email = _one_address(_header(headers, "From"))
    return {
        "message_id": raw.get("id"),
        "thread_id": raw.get("threadId"),
        "history_id": raw.get("historyId"),
        "internal_date": raw.get("internalDate"),
        "subject": _header(headers, "Subject"),
        # Raw address headers (reality as sent) ...
        "from": _header(headers, "From"),
        "to": _header(headers, "To"),
        "cc": _header(headers, "Cc"),
        "bcc": _header(headers, "Bcc"),  # present on captured SENT mail; absent on received
        # ... plus structured forms so the people graph isn't re-parsed by every consumer.
        "sender_email": sender_email,
        "sender_name": sender_name,
        "to_emails": _address_list(_header(headers, "To")),
        "cc_emails": _address_list(_header(headers, "Cc")),
        "bcc_emails": _address_list(_header(headers, "Bcc")),
        "date": _header(headers, "Date"),
        # RFC headers that make the reply graph reconstructible (ontology: replied_to /
        # references_message). The Gmail `id` is mailbox-internal; `Message-ID` is the
        # cross-mailbox stable identifier other systems reference.
        "rfc_message_id": _header(headers, "Message-ID"),
        "in_reply_to": _header(headers, "In-Reply-To"),
        "references": _header(headers, "References"),
        # Inbox-intelligence headers (bulk / list / automated / newsletter grouping).
        "list_id": _header(headers, "List-ID") or _header(headers, "List-Id"),
        "auto_submitted": _header(headers, "Auto-Submitted"),
        "precedence": _header(headers, "Precedence"),
        "snippet": raw.get("snippet"),
        "body_text": body_full[:_BODY_MAX],
        "body_mime": body_mime,
        "body_truncated": len(body_full) > _BODY_MAX,
        "body_length": len(body_full),
        "size_estimate": raw.get("sizeEstimate"),
        "attachments": _collect_attachments(payload),
        "label_ids": raw.get("labelIds", []),
    }
