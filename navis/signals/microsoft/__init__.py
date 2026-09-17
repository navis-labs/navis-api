"""Microsoft 365 (Entra ID + Microsoft Graph) connector foundations.

Mirrors `navis.signals.google` lane for lane: `oauth` owns the token lifecycle,
`transport` owns one resilient HTTP path for every Graph call. The per-resource
connectors live in `navis.signals.outlook`, `.outlook_calendar` and `.onedrive`,
exactly as the Google connectors live outside `navis.signals.google`.

Nothing in this package imports from `navis.signals.google`, and nothing in the
Google packages imports from here. The two suites share the sync engine, the
ingestion service and the dead-letter queue — everything below the connector —
and share nothing above it. That is what lets either be changed without
regression-testing the other.
"""
from navis.signals.microsoft.oauth import (
    MICROSOFT_CONNECTOR_SCOPES,
    MICROSOFT_SCOPE_VERSION,
    MICROSOFT_SIGNIN_SCOPES,
    MicrosoftReconsentRequired,
    MicrosoftToken,
    TokenManager,
    authorization_url,
    calendar_write_granted,
    exchange_code,
    mail_send_granted,
    normalize_scopes,
    refresh_access_token,
)
from navis.signals.microsoft.transport import GraphAPIError, GraphTransport

__all__ = [
    "MICROSOFT_CONNECTOR_SCOPES",
    "MICROSOFT_SCOPE_VERSION",
    "MICROSOFT_SIGNIN_SCOPES",
    "MicrosoftReconsentRequired",
    "MicrosoftToken",
    "TokenManager",
    "authorization_url",
    "calendar_write_granted",
    "exchange_code",
    "mail_send_granted",
    "normalize_scopes",
    "refresh_access_token",
    "GraphAPIError",
    "GraphTransport",
]
