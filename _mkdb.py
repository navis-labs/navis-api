import asyncio, asyncpg
from urllib.parse import urlsplit, urlunsplit
from navis.platform.config.settings import settings

raw = settings.database_url
u = urlsplit(raw.replace("postgresql+asyncpg://", "postgresql://"))
dev_db = u.path.lstrip("/")
TEST_DB = "navis_brief_ordering_test"
assert TEST_DB != dev_db, "refusing to reuse the dev database"

async def main():
    admin = urlunsplit((u.scheme, u.netloc, "/postgres", "", ""))
    conn = await asyncpg.connect(admin)
    await conn.execute(f'DROP DATABASE IF EXISTS "{TEST_DB}"')
    await conn.execute(f'CREATE DATABASE "{TEST_DB}"')
    await conn.close()
    print(f"dev db   = {dev_db}  (UNTOUCHED)")
    print(f"test db  = {TEST_DB} (created, disposable)")
    print("TEST_URL=" + urlunsplit(("postgresql+asyncpg", u.netloc, "/" + TEST_DB, "", "")))
asyncio.run(main())
