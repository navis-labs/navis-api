"""Provider outcome webhooks — map real-world results back to outcomes by the
provider_reference Execution already stored (gmail_message_id, calendar_event_id)."""
from navis.observation.webhooks.base import ObservationWebhookEvent

__all__ = ["ObservationWebhookEvent"]
