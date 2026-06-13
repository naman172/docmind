import os

import asyncpg
from dotenv import load_dotenv

load_dotenv()
dsn = os.environ.get("DATABASE_URL")

_pool: asyncpg.Pool | None = None


async def init_pool() -> None:
    global _pool
    _pool = await asyncpg.create_pool(dsn)


async def close_pool() -> None:
    if not _pool:
        raise RuntimeError("_pool not initialized")
    await _pool.close()


def get_pool() -> asyncpg.Pool:
    if not _pool:
        raise RuntimeError("_pool not initialized")
    return _pool
