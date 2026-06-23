"""Slack reality-capture adapter.

A thin protocol layer on top of the existing Navis signal infrastructure: it verifies
inbound Events API webhooks, maps Slack's wire format to the shared SignalEvent contract,
and (for onboarding) backfills history via REST. It owns NO persistence, queue, dedupe, or
event framework of its own — identity, idempotency, outbox, DLQ, and encryption are reused.

    signature.py  inbound webhook HMAC + timestamp verification (replay guard)
    events.py     Slack event JSON -> SignalEvent(s)  (shared by webhook AND backfill)
"""
