"""Gmail connector — push-based capture (watch → historyId → history.list)."""
from navis.signals.gmail.client import GmailClient
from navis.signals.gmail.connector import GmailConnector

__all__ = ["GmailClient", "GmailConnector"]
