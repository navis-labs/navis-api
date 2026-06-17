"""Create all tables. Dev convenience; use Alembic for real migrations.

    python init_db.py
"""
from __future__ import annotations

import asyncio

import navis.registry  # noqa: F401  (ensures every model is registered on metadata)
from navis.platform.database import Base, engine
from navis.platform.monitoring import configure_logging, get_logger

logger = get_logger("init_db")


async def main() -> None:
    configure_logging()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    tables = ", ".join(sorted(Base.metadata.tables))
    logger.info("created/verified tables: %s", tables)


if __name__ == "__main__":
    asyncio.run(main())
