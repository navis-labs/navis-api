"""Encrypt existing integration tokens at rest (backward compatible)

Revision ID: 0003_encrypt_tokens
Revises: 0002_signal_dead_letters
Create Date: 2026-06-17

No schema change — `access_token`/`refresh_token` are already TEXT. This is a data
migration that encrypts any legacy *plaintext* tokens in place using the configured
SIGNALS_TOKEN_KEYS. It is:
  * idempotent — rows already tagged `enc:1:` are skipped;
  * backward compatible — if no keys are configured it is a safe no-op (tokens stay
    readable as plaintext; the app reads both forms);
  * forward compatible — once keys exist, the app re-encrypts on the next token write
    anyway, so running this is an optimization, not a hard requirement.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0003_encrypt_tokens"
down_revision: Union[str, None] = "0002_signal_dead_letters"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    from navis.signals.crypto import token_cipher

    if not token_cipher.enabled:
        return  # no keys configured → safe no-op

    conn = op.get_bind()
    rows = conn.execute(
        sa.text("SELECT id, access_token, refresh_token FROM integrations")
    ).fetchall()
    for row in rows:
        updates = {}
        if row.access_token and not token_cipher.is_encrypted(row.access_token):
            updates["access_token"] = token_cipher.encrypt(row.access_token)
        if row.refresh_token and not token_cipher.is_encrypted(row.refresh_token):
            updates["refresh_token"] = token_cipher.encrypt(row.refresh_token)
        if updates:
            set_clause = ", ".join(f"{k} = :{k}" for k in updates)
            conn.execute(
                sa.text(f"UPDATE integrations SET {set_clause} WHERE id = :id"),
                {**updates, "id": row.id},
            )


def downgrade() -> None:
    # Decryption-in-place is intentionally not provided (would write plaintext back).
    pass
