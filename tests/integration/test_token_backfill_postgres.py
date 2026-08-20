"""TASK 6.4.1 — legacy-plaintext token re-encryption, against real PostgreSQL.

Needs a real database (not a fake) because the migration's correctness rests on
SQL the fakes do not model: the `NOT LIKE 'enc:1:%'` selection predicate, the
`SELECT ... FOR UPDATE` re-read, and per-row transaction isolation.

    NAVIS_TEST_DATABASE_URL=postgresql+asyncpg://.../navis_test?ssl=require \
        pytest tests/integration/test_token_backfill_postgres.py -q

Every token in this file is SYNTHETIC. No production credential is copied here.
"""
from __future__ import annotations

import contextlib
import logging
import os
import uuid

import pytest
import pytest_asyncio
from cryptography.fernet import Fernet

pytestmark = [
    pytest.mark.skipif(
        not os.environ.get("NAVIS_TEST_DATABASE_URL"),
        reason="set NAVIS_TEST_DATABASE_URL to run Postgres integration tests",
    ),
    # ONE event loop for the module. asyncpg connections bind to the loop that
    # created them, so a module-scoped engine cannot be shared across per-test
    # loops (the same constraint the Celery worker's --pool=prefork note names).
    pytest.mark.asyncio(loop_scope="module"),
]

# Synthetic, obviously-fake credentials. Long and distinctive so a leak assertion
# (`token not in caplog.text`) cannot pass by coincidence.
LEGACY_REFRESH = "SYNTHETIC-legacy-refresh-1//0eXAMPLEnotarealtoken"
LEGACY_ACCESS = "SYNTHETIC-legacy-access-ya29.EXAMPLEnotarealtoken"
KEY = Fernet.generate_key().decode()


@pytest.fixture()
def cipher():
    from navis.signals.crypto import TokenCipher

    return TokenCipher([KEY])


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def engine():
    """ONE engine and ONE create_all for the whole module.

    Per-test `create_all` + `drop_all` against a hosted Neon database costs a
    reflection round trip per table (~57 of them) and blew past a two-minute
    budget. Schema setup happens once; each test cleans up only the rows it
    created (see `wired`), so tests stay independent without re-reflecting.
    """
    from sqlalchemy.ext.asyncio import create_async_engine

    import navis.registry  # noqa: F401  register all models
    from navis.platform.database import Base

    url = os.environ["NAVIS_TEST_DATABASE_URL"]
    # Guard by DATABASE NAME, never hostname — a hostname check would both block
    # a legitimate disposable Neon database and permit dropping a local one.
    name = url.rsplit("/", 1)[-1].split("?")[0]
    assert name.startswith("navis_test"), f"refusing to run against database {name!r}"

    eng = create_async_engine(url)
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield eng
    await eng.dispose()


@pytest_asyncio.fixture(loop_scope="module")
async def wired(engine, cipher, monkeypatch):
    """Point the migration at the test database and at a known-key cipher.

    `token_backfill` imports both `session_scope` and `token_cipher` BY VALUE at
    module import, so they must be patched on THAT module — patching
    `navis.platform.database.session_scope` would not be seen, and the job would
    quietly run against the URL in `.env`, which is production.

    Teardown deletes only what this module created (organizations cascade to
    users' memberships and to integrations), so tests stay independent without
    paying for a schema drop/recreate per test.
    """
    from sqlalchemy import delete
    from sqlalchemy.ext.asyncio import async_sessionmaker

    from navis.signals import token_backfill

    Session = async_sessionmaker(engine, expire_on_commit=False)

    @contextlib.asynccontextmanager
    async def _scope():
        async with Session() as s:
            try:
                yield s
                await s.commit()
            except Exception:
                await s.rollback()
                raise

    monkeypatch.setattr(token_backfill, "session_scope", _scope)
    monkeypatch.setattr(token_backfill, "token_cipher", cipher)
    yield token_backfill, Session

    from navis.identity.models import Integration, Organization, User

    async with Session() as db:
        await db.execute(delete(Integration))
        await db.execute(delete(Organization))
        await db.execute(delete(User))
        await db.commit()


async def _seed(Session, cipher):
    """Two orgs; six integrations covering every token shape."""
    from navis.identity.models import (
        Integration,
        IntegrationProvider,
        Organization,
        User,
    )

    async with Session() as db:
        org_a = Organization(name="Alpha", slug=f"alpha-{uuid.uuid4().hex[:8]}")
        org_b = Organization(name="Beta", slug=f"beta-{uuid.uuid4().hex[:8]}")
        user_a = User(email=f"a-{uuid.uuid4().hex[:8]}@alpha.test")
        user_b = User(email=f"b-{uuid.uuid4().hex[:8]}@beta.test")
        db.add_all([org_a, org_b, user_a, user_b])
        await db.flush()

        rows = {
            # A. fully legacy plaintext (the production defect)
            "legacy_both": Integration(
                organization_id=org_a.id, user_id=user_a.id,
                provider=IntegrationProvider.GMAIL, external_account="a1@alpha.test",
                access_token=LEGACY_ACCESS, refresh_token=LEGACY_REFRESH,
            ),
            # B. the EXACT production shape: access encrypted, refresh plaintext
            "prod_shape": Integration(
                organization_id=org_a.id, user_id=user_a.id,
                provider=IntegrationProvider.GOOGLE_CALENDAR, external_account="a2@alpha.test",
                access_token=cipher.encrypt(LEGACY_ACCESS), refresh_token=LEGACY_REFRESH,
            ),
            # C. already fully encrypted — must not be rewritten
            "encrypted": Integration(
                organization_id=org_a.id, user_id=user_a.id,
                provider=IntegrationProvider.DRIVE, external_account="a3@alpha.test",
                access_token=cipher.encrypt(LEGACY_ACCESS),
                refresh_token=cipher.encrypt(LEGACY_REFRESH),
            ),
            # D. NULL refresh token (schema permits it)
            "null_refresh": Integration(
                organization_id=org_a.id, user_id=user_a.id,
                provider=IntegrationProvider.SLACK, external_account="T0ALPHA",
                access_token=cipher.encrypt(LEGACY_ACCESS), refresh_token=None,
            ),
            # E. both NULL
            "null_both": Integration(
                organization_id=org_a.id, user_id=user_a.id,
                provider=IntegrationProvider.NOTION, external_account="a5@alpha.test",
                access_token=None, refresh_token=None,
            ),
            # F. a DIFFERENT organization, left plaintext to prove isolation
            "other_org": Integration(
                organization_id=org_b.id, user_id=user_b.id,
                provider=IntegrationProvider.GMAIL, external_account="b1@beta.test",
                access_token=LEGACY_ACCESS, refresh_token=LEGACY_REFRESH,
            ),
        }
        db.add_all(list(rows.values()))
        await db.commit()
        return org_a.id, org_b.id, {k: v.id for k, v in rows.items()}


async def _get(Session, integration_id):
    from navis.identity.models import Integration

    async with Session() as db:
        return await db.get(Integration, integration_id)


# ── 1 / 2 / 3 / 7 — the core migration contract ───────────────────────────────

async def test_plaintext_encrypted_encrypted_unchanged_null_unchanged(wired, cipher):
    backfill, Session = wired
    _org_a, _org_b, ids = await _seed(Session, cipher)

    before_encrypted = (await _get(Session, ids["encrypted"])).refresh_token
    report = await backfill.reencrypt_legacy_tokens()

    # 1. legacy plaintext -> encrypted, and still decryptable to the SAME secret.
    for key in ("legacy_both", "prod_shape", "other_org"):
        row = await _get(Session, ids[key])
        assert cipher.is_encrypted(row.refresh_token), key
        assert cipher.decrypt(row.refresh_token) == LEGACY_REFRESH, key

    # 2. already-encrypted rows are byte-identical (never re-encrypted).
    after = await _get(Session, ids["encrypted"])
    assert after.refresh_token == before_encrypted
    assert cipher.decrypt(after.refresh_token) == LEGACY_REFRESH

    # 3. NULL stays NULL — never coerced to a string, never encrypted.
    assert (await _get(Session, ids["null_refresh"])).refresh_token is None
    null_both = await _get(Session, ids["null_both"])
    assert null_both.access_token is None and null_both.refresh_token is None

    assert report.failed == []
    # 3 rows carried a plaintext refresh token; 2 of those also carried plaintext access.
    assert report.encrypted.get("refresh_token") == 3
    assert report.encrypted.get("access_token") == 2


async def test_unrelated_columns_are_untouched(wired, cipher):
    backfill, Session = wired
    _a, _b, ids = await _seed(Session, cipher)
    before = await _get(Session, ids["prod_shape"])
    snapshot = (
        before.organization_id, before.user_id, before.provider,
        before.external_account, before.is_active, before.scopes,
        before.token_expires_at, dict(before.sync_state or {}),
        before.access_token,  # already encrypted here -> must not change
    )

    await backfill.reencrypt_legacy_tokens()

    after = await _get(Session, ids["prod_shape"])
    assert (
        after.organization_id, after.user_id, after.provider,
        after.external_account, after.is_active, after.scopes,
        after.token_expires_at, dict(after.sync_state or {}),
        after.access_token,
    ) == snapshot
    assert after.refresh_token != before.refresh_token  # the one intended change


# ── 5 — idempotency ───────────────────────────────────────────────────────────

async def test_second_run_changes_zero_rows(wired, cipher):
    backfill, Session = wired
    _a, _b, ids = await _seed(Session, cipher)

    first = await backfill.reencrypt_legacy_tokens()
    assert first.scanned == 3

    frozen = {k: (await _get(Session, v)).refresh_token for k, v in ids.items()}

    second = await backfill.reencrypt_legacy_tokens()
    assert second.scanned == 0
    assert second.encrypted == {}
    assert second.failed == []

    for k, v in ids.items():
        assert (await _get(Session, v)).refresh_token == frozen[k], k


# ── 10 — organization isolation ───────────────────────────────────────────────

async def test_org_scoped_run_never_touches_another_tenant(wired, cipher):
    backfill, Session = wired
    org_a, _org_b, ids = await _seed(Session, cipher)

    report = await backfill.reencrypt_legacy_tokens(org_id=org_a)

    # Org A migrated...
    assert cipher.is_encrypted((await _get(Session, ids["legacy_both"])).refresh_token)
    assert cipher.is_encrypted((await _get(Session, ids["prod_shape"])).refresh_token)
    # ...org B untouched, still exactly the value it started with.
    assert (await _get(Session, ids["other_org"])).refresh_token == LEGACY_REFRESH
    assert report.encrypted.get("refresh_token") == 2


# ── 11 — concurrency (TOCTOU re-check) ────────────────────────────────────────

async def test_row_encrypted_concurrently_is_not_double_encrypted(wired, cipher):
    """A sync pass encrypting the row between the page query and the row write
    must not produce a doubly-encrypted value."""
    backfill, Session = wired
    _a, _b, ids = await _seed(Session, cipher)
    from navis.identity.models import Integration

    target = ids["prod_shape"]
    # Simulate the concurrent writer winning the race.
    async with Session() as db:
        row = await db.get(Integration, target)
        row.refresh_token = cipher.encrypt(LEGACY_REFRESH)
        await db.commit()
    winner = (await _get(Session, target)).refresh_token

    # Now drive the row the migration had already selected.
    report = backfill.BackfillReport()
    async with Session() as db:
        await backfill._encrypt_one(db, target, report)
        await db.commit()

    after = (await _get(Session, target)).refresh_token
    assert after == winner            # untouched by us
    assert report.scanned == 0        # correctly counted as "nothing to do"
    assert cipher.decrypt(after) == LEGACY_REFRESH  # single, not double, encryption


# ── 4 — malformed ciphertext fails safely ─────────────────────────────────────

async def test_malformed_ciphertext_is_left_alone_and_fails_closed(wired, cipher):
    """A value already TAGGED but undecryptable is not a migration target — the
    job must not touch it, and decrypt must still refuse it."""
    backfill, Session = wired
    _a, _b, ids = await _seed(Session, cipher)
    from navis.identity.models import Integration
    from navis.signals.crypto import TokenDecryptError

    async with Session() as db:
        row = await db.get(Integration, ids["encrypted"])
        row.refresh_token = "enc:1:" + "A" * 40  # tagged but corrupt
        await db.commit()

    report = await backfill.reencrypt_legacy_tokens()

    after = (await _get(Session, ids["encrypted"])).refresh_token
    assert after == "enc:1:" + "A" * 40   # untouched — not selected, not clobbered
    assert report.failed == []
    with pytest.raises(TokenDecryptError):
        cipher.decrypt(after)


async def test_disabled_cipher_refuses_to_run(wired, monkeypatch):
    """With no keys, encrypt() is a passthrough — the job must refuse rather than
    report a migration that silently left everything plaintext."""
    backfill, Session = wired
    from navis.signals.crypto import TokenCipher

    await _seed(Session, TokenCipher([KEY]))
    monkeypatch.setattr(backfill, "token_cipher", TokenCipher([]))

    with pytest.raises(backfill.TokenBackfillUnavailable):
        await backfill.reencrypt_legacy_tokens()

    # Nothing was written.
    async with Session() as db:
        from sqlalchemy import func, select

        from navis.identity.models import Integration

        still_plaintext = await db.scalar(
            select(func.count()).select_from(Integration).where(
                Integration.refresh_token.is_not(None),
                Integration.refresh_token.not_like("enc:1:%"),
            )
        )
    assert still_plaintext == 3


# ── 8 / 9 — no token material in logs or exceptions ───────────────────────────

async def test_no_plaintext_token_in_logs(wired, cipher, caplog):
    backfill, Session = wired
    await _seed(Session, cipher)
    with caplog.at_level(logging.DEBUG):
        report = await backfill.reencrypt_legacy_tokens()
    assert LEGACY_REFRESH not in caplog.text
    assert LEGACY_ACCESS not in caplog.text
    # The report itself is safe to serialize into an ops channel.
    assert LEGACY_REFRESH not in repr(report.to_dict())


async def test_no_plaintext_token_in_exception_or_failure_record(wired, cipher, monkeypatch, caplog):
    """Force a write failure and prove the recorded failure carries ids only."""
    backfill, Session = wired
    _a, _b, ids = await _seed(Session, cipher)

    class Boom(Exception):
        pass

    real_encrypt = cipher.encrypt

    def exploding_encrypt(plaintext):
        if plaintext == LEGACY_REFRESH:
            raise Boom("provider unavailable")  # deliberately token-free message
        return real_encrypt(plaintext)

    monkeypatch.setattr(cipher, "encrypt", exploding_encrypt)

    with caplog.at_level(logging.DEBUG):
        report = await backfill.reencrypt_legacy_tokens()

    assert len(report.failed) == 3
    for entry in report.failed:
        assert set(entry) == {"integration_id", "failure"}
        assert LEGACY_REFRESH not in str(entry)
    assert LEGACY_REFRESH not in caplog.text

    # PHASE 7: a failed row is left USABLE — original value intact, not nulled.
    row = await _get(Session, ids["prod_shape"])
    assert row.refresh_token == LEGACY_REFRESH
    assert row.is_active is True


# ── 6 — the sunset switch, before and after ───────────────────────────────────

async def test_sunset_switch_rejects_before_and_accepts_after_migration(wired, cipher, monkeypatch):
    backfill, Session = wired
    _a, _b, ids = await _seed(Session, cipher)
    from navis.platform.config import settings
    from navis.signals.crypto import TokenDecryptError

    # BEFORE: enabling the switch would break the plaintext rows.
    monkeypatch.setattr(settings, "signals_reject_legacy_plaintext", True)
    row = await _get(Session, ids["prod_shape"])
    with pytest.raises(TokenDecryptError):
        cipher.decrypt(row.refresh_token)

    # The migration itself never calls decrypt(), so it runs fine with the
    # switch already on — the two are order-independent.
    await backfill.reencrypt_legacy_tokens()

    # AFTER: every row decrypts cleanly with the switch still enabled.
    async with Session() as db:
        from sqlalchemy import select

        from navis.identity.models import Integration

        for r in (await db.scalars(select(Integration))).all():
            for col in ("access_token", "refresh_token"):
                value = getattr(r, col)
                if value is None:
                    continue
                assert cipher.decrypt(value) is not None
