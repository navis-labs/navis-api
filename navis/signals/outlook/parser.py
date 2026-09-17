"""Map a Microsoft Graph `message` resource onto Navis's canonical mail payload.

── Why this file exists at all ────────────────────────────────────────────

The payload keys emitted here are NOT Graph's. They are the same keys
`navis.signals.gmail.parser` emits, deliberately, because everything
downstream of the signal log — the memory extractors, the people graph, the
reply-gap detector, the commitment extractor — reads mail payloads by key and
has no idea which vendor produced them.

If Outlook emitted `conversationId` where Gmail emits `thread_id`, every one of
those consumers would need an `if source == outlook` branch, and the ones nobody
remembered to update would quietly stop seeing half the org's mail. Translating
once, here, at the capture boundary is what keeps the rest of the pipeline
vendor-blind — the backend counterpart of the lane model the frontend uses.

So: when a key here looks oddly un-Graph-like, that is the point. The mapping is
the contract.
"""
from __future__ import annotations

from typing import Any

from navis.platform.security import field_cipher

# Matches the Gmail parser's cap so a long thread is truncated identically
# whichever mailbox it came from.
_BODY_MAX = 100_000

# Sensitive free-text content encrypted at rest in the signal payload, exactly
# as the Gmail connector does it: the append-only log, its backups and any
# analytics replica hold ciphertext, and only the Memory extraction seam
# decrypts on read. Routing/identity fields (from/to/cc, conversation id) stay
# plaintext because the people graph needs them.
_ENCRYPTED_CONTENT_FIELDS = ("body_text", "snippet", "bcc")


def _address(entry: dict[str, Any] | None) -> tuple[str | None, str | None]:
    """Return (name, email) from Graph's `{emailAddress: {name, address}}`."""
    if not isinstance(entry, dict):
        return None, None
    email_address = entry.get("emailAddress")
    if not isinstance(email_address, dict):
        return None, None
    name = email_address.get("name")
    address = email_address.get("address")
    return (name or None), (address.lower() if isinstance(address, str) else None)


def _address_list(entries: Any) -> list[str]:
    """Email addresses from a Graph recipient collection, lowercased."""
    if not isinstance(entries, list):
        return []
    out: list[str] = []
    for entry in entries:
        _, address = _address(entry)
        if address:
            out.append(address)
    return out


def _display_list(entries: Any) -> str | None:
    """Recipients rendered the way an RFC 5322 header would read them.

    Kept alongside the structured list because the Gmail payload carries both,
    and some consumers show the raw header to a founder ("To: Priya Raman
    <priya@...>, ops@..."). Reconstructing it from Graph's structured form is
    lossy in principle and exact enough in practice.
    """
    if not isinstance(entries, list) or not entries:
        return None
    parts: list[str] = []
    for entry in entries:
        name, address = _address(entry)
        if name and address:
            parts.append(f"{name} <{address}>")
        elif address:
            parts.append(address)
    return ", ".join(parts) or None


def _internet_headers(raw: dict[str, Any]) -> dict[str, str]:
    """Index `internetMessageHeaders` by lowercased name.

    Graph only populates this when explicitly selected, and even then only for
    messages where Exchange retained the headers. Absent headers degrade to
    None below rather than failing — the reply graph is richer with them and
    still correct without them.
    """
    headers = raw.get("internetMessageHeaders")
    if not isinstance(headers, list):
        return {}
    out: dict[str, str] = {}
    for header in headers:
        if not isinstance(header, dict):
            continue
        name = header.get("name")
        value = header.get("value")
        if isinstance(name, str) and isinstance(value, str):
            out[name.lower()] = value
    return out


def _body(raw: dict[str, Any]) -> tuple[str, str | None]:
    body = raw.get("body")
    if not isinstance(body, dict):
        return "", None
    content = body.get("content")
    return (content if isinstance(content, str) else ""), body.get("contentType")


def _attachments(raw: dict[str, Any]) -> list[dict[str, Any]]:
    """Attachment metadata only — never content.

    Mirrors the Gmail connector's shape. Filenames are themselves sensitive org
    intelligence, so they are sealed below alongside the body rather than left
    in plaintext.
    """
    entries = raw.get("attachments")
    if not isinstance(entries, list):
        return []
    out: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        out.append({
            "attachment_id": entry.get("id"),
            "filename": entry.get("name"),
            "mime_type": entry.get("contentType"),
            "size": entry.get("size"),
            "is_inline": entry.get("isInline"),
        })
    return out


def _encrypt_content(parsed: dict[str, Any]) -> None:
    """Encrypt sensitive content fields in place and self-describe which were
    sealed, so the reader decrypts generically without importing this module's
    field list. Identical contract to the Gmail parser's `_encrypt_content` —
    `SignalViewFactory.from_record` reads `_content_fields` and does not care
    which connector wrote it."""
    sealed = [f for f in _ENCRYPTED_CONTENT_FIELDS if parsed.get(f) is not None]
    for field_name in sealed:
        parsed[field_name] = field_cipher.encrypt(parsed[field_name])
    parsed["_content_fields"] = list(sealed)
    # Hidden-recipient list and attachment filenames are sealed at rest but NOT
    # listed in _content_fields — each value carries its own tag, and decrypt()
    # is a no-op on plaintext, so a future reader runs one uniform path.
    if parsed.get("bcc_emails"):
        parsed["bcc_emails"] = [
            field_cipher.encrypt(value) for value in parsed["bcc_emails"]
        ]
    for attachment in parsed.get("attachments") or []:
        if attachment.get("filename"):
            attachment["filename"] = field_cipher.encrypt(attachment["filename"])


def parse_message(raw: dict[str, Any]) -> dict[str, Any]:
    """Graph `message` -> canonical mail payload (content fields encrypted)."""
    headers = _internet_headers(raw)
    body_full, body_mime = _body(raw)
    sender_name, sender_email = _address(raw.get("from") or raw.get("sender"))

    parsed: dict[str, Any] = {
        # ── Identity ──
        # `message_id` is Graph's immutable id (the transport asks for
        # ImmutableId, so this survives the message being moved between
        # folders — which matters because it is half of the dedupe key).
        "message_id": raw.get("id"),
        "thread_id": raw.get("conversationId"),
        # No Gmail equivalent; carried because a Graph incremental sync can key
        # off it and because it makes a captured fact traceable back to Graph.
        "change_key": raw.get("changeKey"),
        "internal_date": raw.get("receivedDateTime") or raw.get("sentDateTime"),
        "subject": raw.get("subject"),
        # ── Raw address headers (reality as sent) ──
        "from": _display_list([raw.get("from")]) if raw.get("from") else None,
        "to": _display_list(raw.get("toRecipients")),
        "cc": _display_list(raw.get("ccRecipients")),
        # Present on captured SENT mail; absent on received, same as Gmail.
        "bcc": _display_list(raw.get("bccRecipients")),
        # ── Structured forms, so the people graph isn't re-parsed downstream ──
        "sender_email": sender_email,
        "sender_name": sender_name,
        "to_emails": _address_list(raw.get("toRecipients")),
        "cc_emails": _address_list(raw.get("ccRecipients")),
        "bcc_emails": _address_list(raw.get("bccRecipients")),
        "date": raw.get("sentDateTime") or raw.get("receivedDateTime"),
        # ── Reply graph ──
        # `internetMessageId` is the cross-mailbox stable identifier other
        # systems reference; Graph's `id` is mailbox-internal. In-Reply-To and
        # References only arrive when internetMessageHeaders was selected.
        "rfc_message_id": raw.get("internetMessageId"),
        "in_reply_to": headers.get("in-reply-to"),
        "references": headers.get("references"),
        # ── Inbox intelligence (bulk / list / automated grouping) ──
        "list_id": headers.get("list-id"),
        "auto_submitted": headers.get("auto-submitted"),
        "precedence": headers.get("precedence"),
        # ── Content ──
        "snippet": raw.get("bodyPreview"),
        "body_text": body_full[:_BODY_MAX],
        "body_mime": body_mime,
        "body_truncated": len(body_full) > _BODY_MAX,
        "body_length": len(body_full),
        "attachments": _attachments(raw),
        # ── Mailbox state ──
        # Graph has no label model; the nearest equivalents are user categories
        # plus a few booleans. Named `label_ids` so consumers that already read
        # that key for Gmail keep working.
        "label_ids": raw.get("categories") or [],
        "is_read": raw.get("isRead"),
        "importance": raw.get("importance"),
        "has_attachments": raw.get("hasAttachments"),
        "web_link": raw.get("webLink"),
    }
    _encrypt_content(parsed)
    return parsed
