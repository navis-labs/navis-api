"""Outlook mail capture (Microsoft Graph)."""
from navis.signals.outlook.client import MESSAGE_FIELDS, OutlookClient
from navis.signals.outlook.connector import OutlookConnector
from navis.signals.outlook.parser import parse_message

__all__ = ["MESSAGE_FIELDS", "OutlookClient", "OutlookConnector", "parse_message"]
