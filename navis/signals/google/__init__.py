"""Shared Google Workspace plumbing used by every Google connector.

    transport.py  retry/backoff/429-aware HTTP, token-injecting
    oauth.py      OAuth token-refresh lifecycle (shared across Gmail/Calendar/Drive)
"""
from navis.signals.google.transport import GoogleAPIError, GoogleTransport
from navis.signals.google.oauth import GoogleToken, TokenManager, GOOGLE_SCOPES

__all__ = [
    "GoogleAPIError",
    "GoogleTransport",
    "GoogleToken",
    "TokenManager",
    "GOOGLE_SCOPES",
]
