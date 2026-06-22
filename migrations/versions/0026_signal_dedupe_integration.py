"""Signals dedupe identity — add integration_id

Revision ID: 0026_signal_dedupe_integration
Revises: 0025_dlq_replay_accounting
Create Date: 2026-06-22

Widens uq_signal_dedupe from (organization_id, source, external_id, external_version) to
(organization_id, integration_id, source, external_id, external_version).

Why: a provider's external_id is unique only WITHIN one connected account, not across an
org. Google Drive file ids and Calendar event ids are global, so a file/meeting shared
between two colleagues carries the SAME external_id in both accounts. The org-only key
collapsed those two observations into one row (ON CONFLICT DO NOTHING dropped the second),
silently losing the second person's edge in the org graph. Keying on the observing
integration makes each account's capture its own fact.

Safe on existing data: the new constraint is strictly MORE permissive (an added column can
only split previously-colliding rows apart, never merge), so no existing row can violate
it. The matching event_id (UUIDv5) now folds integration_id in for new captures; dedupe is
enforced by this named constraint (not the PK), so re-capturing a pre-existing fact still
no-ops via the same (now wider) tuple. Reversible.
"""
from typing import Sequence, Union

from alembic import op

revision: str = "0026_signal_dedupe_integration"
down_revision: Union[str, None] = "0025_dlq_replay_accounting"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_COLS = ["organization_id", "integration_id", "source", "external_id", "external_version"]
_OLD_COLS = ["organization_id", "source", "external_id", "external_version"]


def upgrade() -> None:
    op.drop_constraint("uq_signal_dedupe", "signal_events", type_="unique")
    op.create_unique_constraint("uq_signal_dedupe", "signal_events", _COLS)


def downgrade() -> None:
    op.drop_constraint("uq_signal_dedupe", "signal_events", type_="unique")
    op.create_unique_constraint("uq_signal_dedupe", "signal_events", _OLD_COLS)
